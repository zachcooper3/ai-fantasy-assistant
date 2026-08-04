"""
AI recommendation service — wraps the Anthropic Claude API to generate
per-pick draft recommendations.

Design:
  - Builds a structured natural-language prompt from the current draft context
  - Calls Claude (Haiku by default for speed) and parses a JSON response
  - Falls back to the top available player by ADP if the API call fails or
    returns unparseable output — draft day is not the time to crash

Usage:
    service = AIService()
    ctx = RecommendationContext(...)
    result = await service.recommend(ctx)

Author: Zach Cooper
"""

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import anthropic

logger = logging.getLogger(__name__)

# Use Haiku on the clock (fast, cheap); override with CLAUDE_MODEL env var for richer analysis
_DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

# Accepted values for RecommendationResult.confidence. Anything else the model
# returns is normalised to "medium".
_CONFIDENCE_LEVELS = {"high", "medium", "low"}

# How many alternatives we keep. The prompt states this explicitly *and*
# _parse_response enforces it — if only the parser knew, the model would spend
# tokens writing entries that get silently discarded.
_MAX_ALTERNATIVES = 3

# Output budget. This has to comfortably fit the whole JSON response: strategy,
# confidence, the recommendation, _MAX_ALTERNATIVES entries that each carry
# both a `reasoning` and a `tradeoff` sentence, and the alerts array. It was
# 1024, which was sized for the older, smaller response shape; adding
# strategy/tradeoff pushed real responses past the ceiling, and a truncated
# response is unparseable JSON — which silently degraded every recommendation
# to the ADP fallback.
_MAX_RESPONSE_TOKENS = 2048

# This is an analytical task with a single best answer, not a creative one.
# The default temperature of 1.0 meant two "Get pick" clicks on an unchanged
# board could return different players with equally confident reasoning,
# which reads as the tool being unreliable rather than the board being close
# (that's what `confidence: low` is for). 0 also measurably reduces malformed
# JSON, which on this path costs a whole recommendation via the ADP fallback.
_TEMPERATURE = 0.0

# How many players from top_available are actually rendered in the tiers
# table / metrics / news sections. ctx.top_available is deliberately deeper
# than this (see _build_context's top_n) so the positional drop-off and
# survival math below can see past the end of the displayed board — you
# cannot reason about "what's left at RB if I wait" from a 25-player global
# ADP slice that might contain three RBs.
_LISTED_PLAYERS = 25

# DST and K are roster taxes, not picks: their week-to-week fantasy output is
# close to unpredictable (this app has no matchup/scheme data for either —
# see _format_metrics_section), so a pick spent on one before the very end of
# the draft is a pick not spent on a skill player who could actually win a
# week. Reserved out of the "rounds remaining" math below rather than left to
# the model's judgement, because the gap logic would otherwise count an open
# DST slot as an urgent starting-lineup hole in round 10 and push exactly the
# pick this rule exists to prevent.
_LATE_ROUND_POSITIONS = ("DST", "K")

# Kickers aren't modeled by DraftConfig at all (it has qb/rb/wr/te/flex/dst
# slots and no k_slots), but every standard Sleeper league starts one, and
# without it here the assistant would never once tell you to draft a kicker —
# the same class of bug as the missing-QB-slot problem documented on
# _STANDARD_STARTING_LINEUP below. Modeled as a constant rather than plumbed
# through DraftConfig/DraftSession/schemas/serializers because the desired
# behavior is a fixed rule ("one kicker, last round"), not a per-league knob;
# if a league ever starts zero or two, that's when it earns a real k_slots
# column and a DB migration.
_K_SLOTS = 1

@dataclass
class PickSuggestion:
    player_id: int
    player_name: str
    position: str
    adp: float
    reasoning: str
    # Why you'd take this player *instead of* the main recommendation — the
    # comparison the model already reasons through internally but previously
    # had nowhere to put. Only meaningful on alternatives; empty on the main
    # recommendation. Optional so a response predating this field still parses.
    tradeoff: str = ""


@dataclass
class RecommendationResult:
    recommendation: PickSuggestion
    alternatives: list[PickSuggestion]
    alerts: list[str]
    model: str
    # One-sentence read on the shape of the roster and what this pick is doing
    # about it. Distinct from the per-player reasoning: it's the plan, not the
    # justification for one name.
    strategy: str = ""
    # "high" | "medium" | "low" — how clear-cut the model considers this call.
    # Defaults to "medium" when absent or unrecognised, so a missing value
    # never reads as false certainty in either direction.
    confidence: str = "medium"


@dataclass
class RecommendationContext:
    """
    All data needed to generate a pick recommendation.
    Built by the API route from the draft state service + player_repo.
    """
    # Draft state
    pick_number: int
    round_number: int
    my_slot: int
    league_size: int
    is_my_turn: bool
    picks_until_my_turn: int
    my_next_pick_number: int | None
    scoring_format: str = "ppr"
    total_rounds: int = 15

    # My roster — list of {player_name, position, nfl_team}
    my_roster: list[dict] = field(default_factory=list)

    # Available players sorted by ADP — list of {id, rank, name, position,
    # team, adp, sleeper_id}. sleeper_id may be None for a player Sleeper
    # doesn't have a crosswalk entry for; retrieval below skips those.
    top_available: list[dict] = field(default_factory=list)

    # How many of each position remain undrafted
    available_counts: dict[str, int] = field(default_factory=dict)

    # Opponent position counts: {slot: {"RB": 2, "WR": 1, ...}}
    opponent_position_counts: dict[int, dict[str, int]] = field(default_factory=dict)

    # Required starting-lineup slot counts, e.g. {"QB": 1, "RB": 2, "WR": 2,
    # "TE": 1, "FLEX": 1, "DST": 1}. Defaults to the standard 1-QB PPR shape
    # (matches DraftConfig's own defaults) so any caller that doesn't set
    # this explicitly — including _build_preview_context below and any
    # older test fixtures — gets identical behavior to before this field
    # existed. The real app always sets this from DraftConfig.starting_lineup
    # (see backend/app/api/recommendations.py::_build_context).
    starting_lineup: dict[str, int] = field(default_factory=lambda: dict(_STANDARD_STARTING_LINEUP))

    # Per-player analytics keyed by Player.id, sourced from PlayerMetrics
    # (see backend/db/models.py) — opportunity/volume, efficiency, team
    # context, consistency/risk, and forward-looking signals computed from
    # nflverse stats. A missing key or a None field both mean "unknown,"
    # not zero. Populated in recommendations.py::_build_context via
    # metrics_repo.get_metrics_bulk(); defaults empty so callers/tests that
    # predate this field still work unchanged (prompt just omits the
    # section entirely — see _format_metrics_section).
    player_metrics: dict[int, dict] = field(default_factory=dict)

    # Draft-day facts (draft_year, draft_round, draft_pick, draft_team,
    # college) keyed by Player.id, sourced from DraftProfile (see
    # backend/db/models.py) — exists specifically for players who will
    # never have a player_metrics entry (rookies), so they aren't a total
    # blank slate. Populated in recommendations.py::_build_context via
    # draft_profile_repo.get_draft_profiles_bulk(); most veterans simply
    # won't have a key here, which is expected, not an error.
    draft_profiles: dict[int, dict] = field(default_factory=dict)

    # The overall pick number of my turn AFTER the one being advised on, or
    # None if this is my last pick of the draft.
    #
    # This is the field the entire opportunity-cost section hangs on, and it
    # is deliberately NOT my_next_pick_number: that property already returns
    # the *current* pick when it's my turn (see DraftStateService), so using
    # it as the horizon would ask "will this player still be here right now,"
    # which is trivially yes for everyone on the board. The only question
    # that changes a pick is "which of these is gone before I pick again."
    #
    # Defaults None so pre-existing callers/tests degrade to the old behavior
    # (the section is simply omitted) rather than computing a wrong horizon.
    my_following_pick_number: int | None = None

    # Draft slots, in pick order, of every team that picks between this pick
    # and my next turn — from DraftStateService.slot_for_pick, so the snake
    # math lives in exactly one place and any future variant (third-round
    # reversal, etc.) is picked up here for free rather than reimplemented.
    # Empty when I pick again immediately or when the caller predates this.
    upcoming_pick_slots: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Starting lineup gaps — gives Claude a concrete target, not just "consider
# roster needs"
# ---------------------------------------------------------------------------

# Standard 1-QB PPR starting lineup — matches DraftConfig's own field
# defaults. Now just a fallback: real requests read the configured lineup
# from DraftConfig.starting_lineup (settable per-session via qb_slots /
# rb_slots / etc. on POST /api/draft/session) and pass it into
# _compute_roster_gaps explicitly. This constant only kicks in for callers
# that don't pass a lineup — RecommendationContext.starting_lineup's own
# default, and any pre-existing test fixture that predates this option.
#
# This exists because a real gap was found live: the AI never once
# recommended a QB across a full draft. Root cause wasn't a bad prompt
# instruction, it was a missing one — nothing anywhere in RecommendationContext
# told Claude how many of each position it actually needs to start, so
# "consider roster needs" had nothing concrete to check against, and
# generic positional-scarcity reasoning (QB is deep, plenty of value left)
# never got overridden even deep into the draft when a starter was still
# needed.
_STANDARD_STARTING_LINEUP = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1}
_FLEX_ELIGIBLE_POSITIONS = {"RB", "WR", "TE"}


def _gaps_from_counts(
    have: dict[str, int],
    lineup: dict[str, int] | None = None,
) -> dict[str, int]:
    """
    Core gap math, expressed over a {position: count} tally rather than a
    roster list — so it can serve both my own roster (via
    _compute_roster_gaps) and every *opponent's* roster, which the draft
    service only ever exposes as position counts (see
    DraftStateService.position_counts_for_slot). Sharing one implementation
    is the point: "which teams ahead of me still need an RB starter" has to
    mean exactly the same thing as "do I still need an RB starter," or the
    run-risk section would be quietly measuring something different from the
    roster-gap section directly above it.
    """
    if lineup is None:
        lineup = _STANDARD_STARTING_LINEUP

    gaps: dict[str, int] = {}
    flex_surplus = 0
    for pos, required in lineup.items():
        if pos == "FLEX":
            continue
        have_count = have.get(pos, 0)
        shortfall = max(0, required - have_count)
        if shortfall:
            gaps[pos] = shortfall
        if pos in _FLEX_ELIGIBLE_POSITIONS:
            flex_surplus += max(0, have_count - required)

    flex_needed = max(0, lineup.get("FLEX", 0) - flex_surplus)
    if flex_needed:
        gaps["FLEX"] = flex_needed

    return gaps


def _compute_roster_gaps(
    my_roster: list[dict],
    lineup: dict[str, int] | None = None,
) -> dict[str, int]:
    """
    Returns {position: still_needed} for every starting slot not yet filled,
    treating extra RB/WR/TE beyond their required minimums as satisfying the
    FLEX slot. Positions already fully covered are omitted entirely (empty
    dict = a complete starting lineup).

    lineup defaults to the standard 1-QB PPR shape when not given (e.g. from
    callers/tests that predate DraftConfig.starting_lineup); the live app
    always passes ctx.starting_lineup explicitly.
    """
    have: dict[str, int] = {}
    for p in my_roster:
        have[p["position"]] = have.get(p["position"], 0) + 1
    return _gaps_from_counts(have, lineup)


def _full_lineup(lineup: dict[str, int]) -> dict[str, int]:
    """
    The configured lineup plus the kicker DraftConfig doesn't model — see
    _K_SLOTS. Everything downstream (gap math, scarcity, the roster-shape
    line) reads the lineup through here so the kicker requirement can't be
    visible in one place and missing in another.
    """
    if not _K_SLOTS:
        return dict(lineup)
    return {**lineup, "K": lineup.get("K", _K_SLOTS)}


# ---------------------------------------------------------------------------
# ADP tiers — groups the ranked player list so the prompt stops implying
# every ADP decimal is a meaningful ordering
# ---------------------------------------------------------------------------

# There's no external expert-consensus tier feed wired into this app (that
# would need its own data source and ingestion script), so this is a
# heuristic proxy, not a real analyst's tier sheet: a new tier starts
# whenever the ADP gap to the previous player exceeds
# max(_MIN_TIER_GAP, previous_adp * _TIER_GAP_RATIO). The ratio term is
# what makes this scale-aware — a 3-pick gap is a real cliff at ADP 5 but
# noise at ADP 150, so the threshold widens proportionally as ADP climbs
# instead of using one fixed number for the whole board.
_MIN_TIER_GAP = 3.0
_TIER_GAP_RATIO = 0.10


def _compute_adp_tiers(players: list[dict]) -> list[list[dict]]:
    """
    Groups ADP-sorted players (ascending) into tiers by clustering on gap
    size — see module comment above for the threshold logic. Recomputed
    fresh from whatever `players` contains on every call, so a refreshed
    ADP dataset (fetch_adp.py) is reflected in tiering on the very next
    recommendation with no separate step or cached/stale tier data.
    """
    if not players:
        return []

    tiers: list[list[dict]] = [[players[0]]]
    for prev, cur in zip(players, players[1:]):
        gap = cur["adp"] - prev["adp"]
        threshold = max(_MIN_TIER_GAP, prev["adp"] * _TIER_GAP_RATIO)
        if gap > threshold:
            tiers.append([])
        tiers[-1].append(cur)
    return tiers


# ---------------------------------------------------------------------------
# Positional scarcity — shared with the standalone /api/recommend/scarcity
# endpoint (backend/app/api/recommendations.py) so the two never drift out
# of sync on what counts as "critical" scarcity for a given league size.
# ---------------------------------------------------------------------------

# Expected starter counts per team for a standard PPR roster (1 QB, 2 RB +
# 1 flex-eligible ~= 3 RB-equivalent demand, similarly for WR, 1 TE, 1 DST,
# 1 K). Approximate guides, not a per-league exact model.
_STARTER_SLOTS = {"QB": 1, "RB": 3, "WR": 3, "TE": 1, "DST": 1, "K": 1}


def compute_position_scarcity(
    available_counts: dict[str, int],
    league_size: int,
    starter_slots: dict[str, int] | None = None,
) -> dict[str, str]:
    """
    Returns {position: "critical"|"low"|"ok"} given how many of each
    position remain undrafted, for a league of this size and starter-slot
    shape. `starter_slots` defaults to the generic 1-QB/3-RB/3-WR/1-TE/
    1-DST/1-K shape (_STARTER_SLOTS, FLEX pre-folded into RB+WR) when not
    given — the standalone /api/recommend/scarcity endpoint instead passes
    the session's real per-league roster shape (qb_slots/rb_slots/etc.,
    with flex_slots added to both RB and WR demand) so its output reflects
    the actual league instead of this generic assumption. Both callers
    share this one function so the critical/low/ok math itself never
    drifts out of sync between them, even when the shapes they pass differ.

    Thresholds (approximate guides, not hard rules):
      critical — fewer players left than half the league's demand
      low      — fewer than 1.5x the demand
      ok       — above that
    """
    starter_slots = starter_slots or _STARTER_SLOTS
    tiers: dict[str, str] = {}
    for pos, slots in starter_slots.items():
        available = available_counts.get(pos, 0)
        teams_needing = league_size * slots
        critical_threshold = teams_needing // 2
        low_threshold = int(teams_needing * 1.5)

        if available <= critical_threshold:
            tiers[pos] = "critical"
        elif available <= low_threshold:
            tiers[pos] = "low"
        else:
            tiers[pos] = "ok"
    return tiers


# ---------------------------------------------------------------------------
# Opportunity cost — which players actually survive to my next turn
#
# This is the single question a snake draft turns on, and nothing in this
# prompt used to answer it. The context carried my_next_pick_number and
# picks_until_my_turn from the very beginning, but they appeared only in a
# one-line "N pick(s) until my turn" header and were never connected to the
# player list, so every recommendation was implicitly reasoning as though
# both candidates would still be available later. They won't be, and which
# one won't be is usually the whole decision: taking the player who survives
# and losing the one who doesn't is strictly worse than the reverse, at
# identical value.
#
# All of this is computed in Python rather than asked of the model on
# purpose. Arithmetic over 25-60 ADP values is exactly what a small fast
# model is worst at and what costs the most reasoning tokens; handing it the
# conclusions leaves it doing the judgement work it's actually good at.
# ---------------------------------------------------------------------------

# ADP is a consensus average, so a player's real draft slot is a distribution
# around it, not a point. These two numbers define that spread: a player goes
# roughly within +/- max(_ADP_NOISE_FLOOR, adp * _ADP_NOISE_RATIO) of their
# ADP. The ratio term matters for the same reason it does in the tier math
# above — 8 picks of slack is enormous at ADP 10 and meaningless at ADP 180,
# where whole positions come off the board in that span.
_ADP_NOISE_FLOOR = 8.0
_ADP_NOISE_RATIO = 0.25

# Labels, not probabilities. A calibrated percentage would imply a precision
# this heuristic doesn't have, and would invite the model to do arithmetic
# with it; three buckets communicate the same actionable distinction.
_SURVIVAL_GONE = "GONE"
_SURVIVAL_TOSSUP = "TOSS-UP"
_SURVIVAL_SAFE = "LIKELY THERE"


def _survival(adp: float, horizon_pick: int | None) -> str | None:
    """
    Whether a player with this ADP is likely to still be on the board at
    `horizon_pick`. Returns None when there's no horizon (my last pick of
    the draft — nothing to wait for, so the question is meaningless and the
    section is omitted rather than filled with a guess).
    """
    if horizon_pick is None:
        return None
    noise = max(_ADP_NOISE_FLOOR, adp * _ADP_NOISE_RATIO)
    if horizon_pick > adp + noise:
        return _SURVIVAL_GONE
    if horizon_pick < adp - noise:
        return _SURVIVAL_SAFE
    return _SURVIVAL_TOSSUP


def _format_survival_section(
    listed: list[dict],
    horizon_pick: int | None,
    picks_between: int,
) -> str:
    """
    Splits the displayed board into "won't be there next time," "might be,"
    and "will almost certainly still be there" — the frame that turns a
    ranked list into a decision.

    The last bucket is as important as the first and is the one a model
    won't infer on its own: a LIKELY THERE player is not a worse player, he's
    a player you can take later, which means spending this pick on him
    forfeits a GONE player for nothing. Stated explicitly in the header
    because "available later" reading as "lower priority" is the entire
    behavior this section exists to produce.
    """
    if horizon_pick is None or picks_between <= 0:
        return ""

    buckets: dict[str, list[str]] = {
        _SURVIVAL_GONE: [], _SURVIVAL_TOSSUP: [], _SURVIVAL_SAFE: [],
    }
    for p in listed:
        label = _survival(p["adp"], horizon_pick)
        if label is not None:
            buckets[label].append(f"{p['name']} ({p['position']}, ADP {p['adp']:g})")

    if not any(buckets.values()):
        return ""

    lines = [
        f"## Opportunity Cost — {picks_between} pick(s) happen before your next turn (#{horizon_pick})",
        "Estimated from each player's ADP versus that pick number. This is the "
        "decisive comparison: taking a LIKELY THERE player now forfeits every GONE "
        "player for nothing, since the LIKELY THERE player can still be had at your "
        "next turn. Only pass on a GONE player for someone clearly better, not for "
        "someone merely similar.",
    ]
    for label, header in (
        (_SURVIVAL_GONE, "Almost certainly gone by then — available now only"),
        (_SURVIVAL_TOSSUP, "Coin flip — could go either way"),
        (_SURVIVAL_SAFE, "Very likely still on the board at your next turn"),
    ):
        names = buckets[label]
        if names:
            lines.append(f"- {header}: " + "; ".join(names))
    lines.append("")
    return "\n".join(lines)


def _format_positional_dropoff(
    board: list[dict],
    horizon_pick: int | None,
    positions: tuple[str, ...] = ("QB", "RB", "WR", "TE"),
) -> str:
    """
    For each position: the best player available now, and the best one still
    expected to be there at my next turn — with the ADP gap between them.

    That gap is the actual cost of waiting at that position, and it's the
    closest thing this app has to a value-over-replacement number. The system
    prompt has always instructed value-based drafting "relative to positional
    replacement level" without ever supplying a replacement level; this
    supplies one, derived per-position from the live board instead of a
    static baseline.

    Reads from `board` (the full ctx.top_available, deeper than the displayed
    _LISTED_PLAYERS slice) rather than the tiers table, because the
    replacement-level player at a position is frequently past the end of a
    25-player global ADP cut — which is exactly when waiting is most
    expensive and the model could least see it.
    """
    if horizon_pick is None:
        return ""

    lines: list[str] = []
    for pos in positions:
        at_pos = sorted(
            (p for p in board if p["position"] == pos), key=lambda p: p["adp"]
        )
        if not at_pos:
            continue
        best = at_pos[0]
        survivor = next(
            (p for p in at_pos if _survival(p["adp"], horizon_pick) != _SURVIVAL_GONE),
            None,
        )
        if survivor is None:
            lines.append(
                f"- {pos}: best available {best['name']} (ADP {best['adp']:g}). "
                f"Every {pos} currently on this board projects to be gone by your "
                f"next turn — waiting means starting from whatever is left."
            )
        elif survivor["id"] == best["id"]:
            lines.append(
                f"- {pos}: best available {best['name']} (ADP {best['adp']:g}) — "
                f"projects to still be there at your next turn. No cost to waiting."
            )
        else:
            gap = survivor["adp"] - best["adp"]
            lines.append(
                f"- {pos}: best available {best['name']} (ADP {best['adp']:g}); best "
                f"likely to survive to your next turn is {survivor['name']} (ADP "
                f"{survivor['adp']:g}). Cost of waiting: {gap:.0f} ADP points at the position."
            )

    if not lines:
        return ""
    return "\n".join([
        "## Cost of Waiting, by Position",
        "The drop from the best player available now to the best one likely to "
        "survive to your next turn. A large drop is the strongest argument for "
        "taking that position now; a small one means the position can wait and "
        "this pick belongs somewhere else.",
        *lines,
        "",
    ])


def _format_run_risk(
    upcoming_pick_slots: list[int],
    opponent_position_counts: dict[int, dict[str, int]],
    lineup: dict[str, int],
) -> str:
    """
    How many of the teams picking before my next turn still need a starter at
    each position — the demand side of run risk, which raw undrafted counts
    (the Positional Availability section) can't express at all.

    Replaces the old full opponent-roster dump: printing all 11 opponents'
    position counts spent real prompt budget on teams that pick *after* me
    and therefore cannot take a player before my next turn. The teams that
    can are the only ones whose needs are actionable, and reducing 11 lines
    to one demand tally per position also trims input tokens on a path where
    latency is a live complaint.
    """
    if not upcoming_pick_slots or not opponent_position_counts:
        return ""

    # Distinct teams, not distinct picks. Across a snake turn most teams
    # appear twice in upcoming_pick_slots (down the order, then back up), and
    # counting each appearance made the tally exceed the number of teams —
    # "WR: 38" for 19 teams, a number that can't mean anything and quietly
    # discredits every other figure in the prompt. A team that needs one
    # starting WR takes roughly one starting WR regardless of how many picks
    # it holds.
    teams = sorted({s for s in upcoming_pick_slots if s in opponent_position_counts})
    if not teams:
        return ""

    demand: dict[str, int] = {}
    flex_demand = 0
    for slot in teams:
        gaps = _gaps_from_counts(opponent_position_counts[slot], lineup)
        for pos, _ in gaps.items():
            if pos == "FLEX":
                # Counted on its own line, NOT added to both RB and WR. Adding
                # it to both is what inflated the totals past 100% and implies
                # one open flex slot generates two positions' worth of demand.
                flex_demand += 1
            elif pos not in _LATE_ROUND_POSITIONS:
                # A DST/K "need" is not run risk — nobody is taking one this
                # early, and listing it invites the model to treat those
                # positions as contested. See _LATE_ROUND_POSITIONS.
                demand[pos] = demand.get(pos, 0) + 1

    if not demand and not flex_demand:
        return ""

    n = len(teams)
    ordered = sorted(demand.items(), key=lambda kv: -kv[1])
    parts = [f"{pos} {cnt}/{n}" for pos, cnt in ordered]
    if flex_demand:
        parts.append(f"FLEX {flex_demand}/{n} (fills with an RB or WR)")

    return "\n".join([
        f"## Run Risk — needs of the {n} team(s) picking before your next turn",
        "Share of those teams with that starting slot still unfilled: " + ", ".join(parts),
        "High demand at a position means a run there is likely before your next "
        "turn, which compounds the cost-of-waiting numbers above. Low demand means "
        "you can probably wait even if raw supply looks thin.",
        "",
    ])


# ---------------------------------------------------------------------------
# Retrieval — grounds the prompt in real news/analysis from ChromaDB
# ---------------------------------------------------------------------------

# Only the most relevant candidates get a retrieval lookup — querying Chroma
# for all 25 listed players would balloon both latency and prompt size for
# players near the bottom of the list that are unlikely to be picked anyway.
_MAX_CONTEXT_PLAYERS = 10
_MAX_CHUNKS_PER_PLAYER = 3

# Chroma content is only ever updated by the offline ingestion scripts
# (chunker.py / fetch_synthesis.py), never by anything a live draft session
# does — so it's safe to cache a player's retrieved chunks for the lifetime
# of this process. Confirmed live: without this, every single "Get pick"
# repeated the same ~10-20 embedding round trips for whichever players
# still happened to be near the top of the board, which barely changes
# pick to pick, and was a large share of a reported 10+ second latency per
# recommendation. Unbounded is fine — a season's player pool is a few
# hundred entries at most, trivial memory for a single draft session.
_retrieval_cache: dict[str, list[str]] = {}


def _query_player_chunks(vector_query, sleeper_id: str, player_name: str) -> list[str]:
    """
    One query per player instead of two — what_happened and what_it_means
    are pulled together via a chunk_type $in filter rather than as separate
    round trips, halving the embedding/query cost per candidate. Cached by
    sleeper_id so repeat lookups across a draft session (very common — the
    top of the board barely changes pick to pick) are free after the first.

    player_name is still passed as the query text (not left blank) — the
    `where` filter already narrows results to one player's own chunks, so
    it mostly only affects tie-breaking when a player has more chunks than
    _MAX_CHUNKS_PER_PLAYER, but there's no reason to introduce an untested
    empty-string-embedding edge case when this is already proven to work.
    """
    if sleeper_id in _retrieval_cache:
        return _retrieval_cache[sleeper_id]

    results = vector_query(
        player_name,
        n_results=_MAX_CHUNKS_PER_PLAYER,
        where={"$and": [
            {"sleeper_id": sleeper_id},
            {"chunk_type": {"$in": ["what_happened", "what_it_means"]}},
        ]},
    )
    _retrieval_cache[sleeper_id] = results
    return results


def _retrieve_player_context(top_available: list[dict]) -> str:
    """
    Pulls "what happened" (Sleeper injury status + RotoWire news) and "what
    it means" (Claude-synthesized analysis) chunks for the top candidate
    players out of ChromaDB, so the recommendation is grounded in more than
    ADP and roster math alone.

    Players that were checked but have nothing in ChromaDB get an explicit
    "no data" line instead of silently vanishing from this section. This
    matters: rookies and other players with no 2025 season data structurally
    can't have anything retrieved for them (see fetch_metrics.py), which
    means without this line, every veteran shows up with a rich, specific
    paragraph and every rookie shows up with nothing at all — confirmed
    live to correlate with the AI never recommending rookies, likely
    because "nothing to say" reads as "nothing good to say" to an LLM
    asked to justify its pick. Making the absence explicit and explained
    (see _SYSTEM_PROMPT) is meant to break that correlation.

    Best-effort and silent on failure: if ChromaDB isn't installed, the
    collection is empty, or a query errors, this returns "" and the prompt
    is built without this section — draft day shouldn't stall (or crash)
    waiting on a vector store, same stance as the Claude API fallback above.
    """
    try:
        from backend.rag.vector_store import query as vector_query
    except Exception as e:
        logger.info(f"Vector store unavailable — building prompt without retrieved context: {e}")
        return ""

    candidates = [
        p for p in top_available[:_MAX_CONTEXT_PLAYERS] if p.get("sleeper_id")
    ]

    # Fan the (network-bound, cache-missing) queries out instead of running
    # them one after another. Each is an embedding round trip of a few
    # hundred ms; ten of them serially was a large, entirely avoidable share
    # of the reported ~10s per recommendation, since they don't depend on
    # each other in any way. The whole call already runs in a worker thread
    # (see AIService.recommend), so this nests a small pool inside it rather
    # than touching the event loop. _retrieval_cache is only ever written
    # with whole-value dict assignment, which is atomic under the GIL — a
    # duplicate concurrent query for the same player is wasted work at worst,
    # never corruption.
    def _lookup(p: dict) -> tuple[dict, list[str] | None]:
        try:
            return p, _query_player_chunks(vector_query, p["sleeper_id"], p["name"])
        except Exception as e:
            logger.warning(f"Vector query failed for {p['name']}: {e}")
            return p, None

    if candidates:
        with ThreadPoolExecutor(max_workers=min(len(candidates), 8)) as pool:
            outcomes = list(pool.map(_lookup, candidates))
    else:
        outcomes = []

    sections: list[str] = []
    hits = 0
    checked = 0
    for p, results in outcomes:
        checked += 1
        if results is None:
            continue

        if results:
            hits += 1
            lines = "\n".join(f"  {r}" for r in results)
            sections.append(f"- {p['name']} ({p['position']}):\n{lines}")
        else:
            sections.append(
                f"- {p['name']} ({p['position']}): No retrieved data (likely a rookie or "
                f"a player with no 2025 season stats) — this is a data-coverage gap, not a "
                f"signal that the player is a worse pick."
            )

    if not sections:
        logger.info(
            "No candidates with a sleeper_id to check (checked 0) — prompt built without a "
            "Player News & Analysis section."
        )
        return ""

    logger.info(
        "Player News & Analysis: %d/%d candidate(s) had retrieved content, %d had none — included in prompt.",
        hits, checked, checked - hits,
    )
    return "\n".join([
        "## Player News & Analysis (retrieved)",
        *sections,
        "",
    ])


# ---------------------------------------------------------------------------
# Opportunity & performance signals — grounds "upside"/"breakout" reasoning
# in actual usage data instead of ADP and name recognition alone
# ---------------------------------------------------------------------------

# Rendered in a fixed order regardless of which fields a given player has.
# (display label, PlayerMetrics dict key, format spec). Fields that are None
# are skipped entirely rather than printed as "N/A" clutter — see
# PlayerMetrics' docstring in backend/db/models.py for why a missing field
# means "unknown," not zero.
# Fields where an exact 0 is a structural non-event (a QB has 0.0
# targets/gm by definition — that's not a signal, it's just the wrong
# stat for that position) rather than meaningful data. Suppressed like
# None for these specific fields only; a true 0% catch rate or 0 games
# missed elsewhere is still worth showing.
_SUPPRESS_ZERO = {"targets_per_game", "carries_per_game", "red_zone_touches_per_game"}

_METRIC_FIELDS: list[tuple[str, str, str]] = [
    ("tgt/gm", "targets_per_game", "{:.1f}"),
    ("car/gm", "carries_per_game", "{:.1f}"),
    ("RZ touches/gm", "red_zone_touches_per_game", "{:.1f}"),
    ("snap %", "snap_pct", "{:.0%}"),
    ("tgt share", "target_share", "{:.0%}"),
    ("carry share", "carry_share", "{:.0%}"),
    ("Y/tgt", "yards_per_target", "{:.1f}"),
    ("Y/carry", "yards_per_carry", "{:.1f}"),
    ("YAC/rec", "yac_per_reception", "{:.1f}"),
    ("RACR", "racr", "{:.2f}"),
    ("catch rate", "catch_rate", "{:.0%}"),
    ("team pass rate", "team_pass_rate", "{:.0%}"),
    ("depth chart rank", "depth_chart_rank", "{}"),
]

# Forward-looking signals are the clearest "breakout" indicators (a rising
# role vs. earlier in the season) so they're called out in their own
# trailing clause rather than blended into the raw volume list above.
#
# NOTE (2026-07-29): these three, plus is_rookie_or_second_year below, are
# unpopulated for every player in the DB right now — fetch_metrics.py logs
# "Could not load snap_counts — snap_pct will be unavailable" during
# ingestion, so the snap/depth-chart data these depend on never lands.
# This code is correct and will start rendering automatically once that
# ingestion gap is fixed; until then, expect this clause to be silent for
# every player. Worth a separate look — see fetch_metrics.py.
_TREND_FIELDS: list[tuple[str, str, str]] = [
    ("target share trend", "target_share_trend", "{:+.0%}"),
    ("snap % trend", "snap_pct_trend", "{:+.0%}"),
    ("depth chart Δ (neg=moving up)", "depth_chart_trend", "{:+d}"),
]

# Below this many games played (out of a 17-game season), a fantasy_points_avg
# is computed over a meaningfully shortened sample — often because of injury
# — and shouldn't be read as equivalent to a full-season average. Flagged
# explicitly in the line itself rather than left for the reader to notice:
# confirmed live (2026-07-29) that a QB averaging more PPR pts/gm than a
# healthier alternative, but over roughly half a season with heavy injury-
# report presence, still got recommended ahead of the healthier player
# despite worse ADP — the raw numbers were all there, nothing told the
# model to weigh them differently.
_SMALL_SAMPLE_GAMES = 10


def _format_metrics_line(m: dict) -> str | None:
    """
    Builds one compact line from whichever metrics are actually populated
    for a player. Returns None if nothing at all is usable — the caller
    renders an explicit "no data" line in that case (see
    _format_metrics_section), same reasoning as _retrieve_player_context's
    "no retrieved data" line: silence reads as "nothing good to say" to an
    LLM, which is not the message a data-coverage gap should send.
    """
    parts = [
        f"{label} {fmt.format(m[key])}"
        for label, key, fmt in _METRIC_FIELDS
        if m.get(key) is not None and not (key in _SUPPRESS_ZERO and m[key] == 0)
    ]

    if m.get("fantasy_points_avg") is not None:
        consistency = f"{m['fantasy_points_avg']:.1f} PPR pts/gm"
        if m.get("fantasy_points_stdev") is not None:
            consistency += f" (±{m['fantasy_points_stdev']:.1f} stdev)"
        if m.get("games_played") is not None:
            consistency += f" over {m['games_played']} gm"
            if m["games_played"] <= _SMALL_SAMPLE_GAMES:
                consistency += " [SMALL SAMPLE — weigh this average with caution]"
        parts.append(consistency)

    risk_bits = []
    if m.get("injury_report_appearances"):
        risk_bits.append(f"{m['injury_report_appearances']} wks on injury report")
    if m.get("games_missed"):
        risk_bits.append(f"{m['games_missed']} games missed")
    if risk_bits:
        parts.append(", ".join(risk_bits))

    trend_parts = [
        f"{label} {fmt.format(m[key])}"
        for label, key, fmt in _TREND_FIELDS
        if m.get(key) is not None
    ]
    if m.get("is_rookie_or_second_year"):
        trend_parts.append("rookie/2nd-year")

    if not parts and not trend_parts:
        return None

    line = ", ".join(parts)
    if trend_parts:
        line += (" | " if line else "") + ", ".join(trend_parts)
    return line


def _format_draft_profile_line(dp: dict) -> str | None:
    """
    Builds one compact line from a DraftProfile — draft capital (round/pick)
    and final-college-season production are the two things this app can say
    about a player with zero NFL snaps. Returns None if the row exists but
    nothing in it actually resolved (see fetch_draft_profiles.py's and
    fetch_college_stats.py's column-resolution disclaimers).

    College production is raw counting stats, not a share-of-team-offense
    metric like a true "Dominator Rating" — see DraftProfile's docstring
    for why that's deliberately out of scope for now.
    """
    bits = []
    if dp.get("draft_round") is not None and dp.get("draft_pick") is not None:
        bits.append(f"{dp['draft_year']} NFL Draft: Round {dp['draft_round']}, Pick {dp['draft_pick']}")
    elif dp.get("draft_year") is not None:
        bits.append(f"{dp['draft_year']} NFL Draft class")
    if dp.get("draft_team"):
        bits.append(f"drafted by {dp['draft_team']}")
    if dp.get("college"):
        bits.append(f"college: {dp['college']}")

    college_bits = []
    if dp.get("passing_yards") is not None:
        passing = f"{dp['passing_yards']} pass yds"
        if dp.get("passing_td") is not None:
            passing += f", {dp['passing_td']} TD"
        if dp.get("interceptions_thrown") is not None:
            passing += f", {dp['interceptions_thrown']} INT"
        college_bits.append(passing)
    if dp.get("rushing_yards") is not None:
        rushing = f"{dp['rushing_yards']} rush yds"
        if dp.get("carries") is not None:
            rushing += f" on {dp['carries']} car"
        if dp.get("rushing_td") is not None:
            rushing += f", {dp['rushing_td']} TD"
        college_bits.append(rushing)
    if dp.get("receiving_yards") is not None:
        receiving = f"{dp['receiving_yards']} rec yds"
        if dp.get("receptions") is not None:
            receiving += f" on {dp['receptions']} rec"
        if dp.get("receiving_td") is not None:
            receiving += f", {dp['receiving_td']} TD"
        college_bits.append(receiving)

    if college_bits:
        season = dp.get("college_season")
        season_str = f"{season} college season" if season is not None else "final college season"
        bits.append(f"{season_str}: " + "; ".join(college_bits))

    if not bits:
        return None
    return ", ".join(bits)


def _format_metrics_section(
    top_available: list[dict],
    player_metrics: dict[int, dict],
    draft_profiles: dict[int, dict] | None = None,
) -> str:
    """
    Renders a prior-season opportunity/efficiency/consistency snapshot for
    the top candidates, so "upside"/"breakout" reasoning is grounded in
    real usage data instead of just ADP and name recognition. Labeled
    "prior season" deliberately — this is trailing-year data, not a 2026
    projection, and the prompt shouldn't read it as a guarantee.

    Same "explicit absence" pattern as _retrieve_player_context: a checked
    player with no PlayerMetrics row gets a stated reason, not silence —
    with one exception: DST/K aren't individual-usage positions at all
    (PlayerMetrics is built around targets/carries/snaps, which have no
    meaning for a team defense or a kicker), so they never have a row and
    that's not a coverage gap to explain away, it's just the wrong table.
    Confirmed live: every DST in the DB has zero PlayerMetrics rows.

    A player with no PlayerMetrics row but a known DraftProfile (see
    backend/db/models.py) is almost certainly a rookie or recent draftee —
    they get draft capital (round/pick/college) instead of the generic
    "no data" line, since that's real, concrete signal rather than nothing.

    Deliberately NOT capped to _MAX_CONTEXT_PLAYERS the way
    _retrieve_player_context is — this function just formats already-fetched
    dicts (no per-player network/embedding call), so there's no cost reason
    to cut it off early. The caller (_build_prompt) passes the same top-25
    slice shown in the ADP tiers table, so every player named there also
    gets an explicit line here. Confirmed live (2026-07-29): with the old
    shared cap, players ranked 11-25 — DST especially, since defenses
    usually have higher ADP than whatever skill players are still on the
    board — got cut from this section entirely with no line at all, worse
    than the explicit "not modeled" case above. A defense recommended with
    strictly less visible information than its alternatives is a sign the
    model reached for outside knowledge instead of the prompt.
    """
    draft_profiles = draft_profiles or {}
    sections: list[str] = []
    for p in top_available:
        m = player_metrics.get(p["id"])
        if m is None:
            if p["position"] in ("DST", "K"):
                sections.append(
                    f"- {p['name']} ({p['position']}): Not modeled by this metrics table "
                    f"(built for individual offensive usage — no matchup, scheme, or "
                    f"opponent data exists anywhere in this app). ADP is the only "
                    f"grounded signal available for this position here — do not "
                    f"substitute outside knowledge about which defense/kicker is "
                    f"'good.'"
                )
                continue

            dp = draft_profiles.get(p["id"])
            dp_line = _format_draft_profile_line(dp) if dp else None
            if dp_line:
                sections.append(
                    f"- {p['name']} ({p['position']}): No NFL performance data yet "
                    f"(rookie/recent draftee) — {dp_line}. Evaluate on draft capital, "
                    f"landing spot, and role expectations rather than prior production."
                )
            else:
                sections.append(
                    f"- {p['name']} ({p['position']}): No prior-season metrics on file "
                    f"(likely a rookie, or minimal snaps last season) — a data-coverage "
                    f"gap, not a signal against the player."
                )
            continue
        line = _format_metrics_line(m)
        if line is None:
            sections.append(
                f"- {p['name']} ({p['position']}): Metrics on file but nothing usable "
                f"was computed for this player."
            )
        else:
            sections.append(f"- {p['name']} ({p['position']}) [{m.get('season')} season]: {line}")

    if not sections:
        return ""

    return "\n".join([
        "## Opportunity & Performance Signals (prior season — top candidates)",
        *sections,
        "",
    ])


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(ctx: RecommendationContext) -> str:
    lines: list[str] = []

    # Draft state
    turn_info = (
        "I am on the clock."
        if ctx.is_my_turn
        else f"{ctx.picks_until_my_turn} pick(s) until my turn (next pick: #{ctx.my_next_pick_number})."
    )
    # Roster spots left, not just rounds left — the same number framed as
    # "how many more players do I get" is what makes a late-round pick feel
    # expensive. Counts the current pick, unlike the old rounds_remaining
    # below (see the off-by-one note there).
    spots_left = max(0, ctx.total_rounds - ctx.round_number + 1)
    lines += [
        "## Draft State",
        f"- Overall pick: #{ctx.pick_number} (Round {ctx.round_number} of {ctx.total_rounds})",
        f"- My draft slot: {ctx.my_slot} of {ctx.league_size}",
        f"- {turn_info}",
        f"- Roster spots left to fill after this one: {max(0, spots_left - 1)}",
        f"- Scoring: {ctx.scoring_format.upper()}",
        "",
    ]

    # My roster
    my_pos = {}
    for p in ctx.my_roster:
        my_pos[p["position"]] = my_pos.get(p["position"], 0) + 1
    if ctx.my_roster:
        roster_str = ", ".join(
            f"{p['player_name']} ({p['position']})" for p in ctx.my_roster
        )
        pos_str = ", ".join(f"{pos}: {n}" for pos, n in sorted(my_pos.items()))
        lines += [
            "## My Current Roster",
            f"Players: {roster_str}",
            f"Composition: {pos_str}",
            "",
        ]
    else:
        lines += ["## My Current Roster", "Empty — no picks yet.", ""]

    # Starting lineup gaps — explicit, computed, unambiguous (see
    # _compute_roster_gaps' docstring for why this exists: without it,
    # nothing told Claude it still needed a starting QB, and it never
    # once got recommended across a full draft as a result)
    lineup = _full_lineup(ctx.starting_lineup)
    gaps = _compute_roster_gaps(ctx.my_roster, lineup)

    # Split the gaps: DST/K are roster taxes you pay at the end (see
    # _LATE_ROUND_POSITIONS), skill slots are what the draft is actually
    # for. Folding them together is what let an open DST slot in round 10
    # register as an urgent starting-lineup hole.
    late_gaps = {p: n for p, n in gaps.items() if p in _LATE_ROUND_POSITIONS}
    skill_gaps = {p: n for p, n in gaps.items() if p not in _LATE_ROUND_POSITIONS}

    # Counts the current pick. The old expression (total_rounds - round_number)
    # excluded it, so on the final round it evaluated to 0 and the `> 0` guard
    # below suppressed the URGENT line entirely — at precisely the pick where
    # an unfilled starting slot is least recoverable.
    rounds_remaining = spots_left
    # Rounds genuinely available for skill players: the last picks are spoken
    # for by the DST/K still owed. Without this subtraction the urgency check
    # is optimistic by exactly the number of roster taxes outstanding, which
    # is how you arrive at the final two rounds needing a QB, a DST and a K
    # with two picks left.
    late_reserved = sum(late_gaps.values())
    skill_rounds_remaining = max(0, rounds_remaining - late_reserved)

    lineup_str = ", ".join(f"{pos} x{n}" for pos, n in sorted(skill_gaps.items()))
    # Built from the configured lineup (not hardcoded) so this line reflects
    # this league's actual roster, e.g. via /api/draft/session.
    _slot_order = ["QB", "RB", "WR", "TE", "FLEX", "DST", "K"]
    roster_str = ", ".join(
        f"{lineup[pos]} {pos}" for pos in _slot_order if lineup.get(pos, 0) > 0
    )
    lines.append(f"## League's Configured Starting Roster Shape ({roster_str})")
    if skill_gaps:
        skill_total = sum(skill_gaps.values())
        # Informational, not imperative — this is roster awareness, not an
        # instruction to fill these before a better value/upside pick. See
        # the Task section and the system prompt below for the actual
        # prioritization: value/upside first, gap-filling only escalates to
        # a hard priority once the URGENT line below appears.
        lines.append(f"Open starting slots (not yet on your roster): {lineup_str}")
        if skill_total >= skill_rounds_remaining:
            lines.append(
                f"URGENT: {skill_total} required starting slot(s) still open with only "
                f"{skill_rounds_remaining} usable round(s) left (of {rounds_remaining} "
                f"total, {late_reserved} reserved for DST/K) — prioritize filling these "
                f"over upside/value picks at positions you've already covered."
            )
    else:
        lines.append("All required starting slots filled — every remaining pick is depth or upside.")

    if late_gaps:
        late_str = ", ".join(f"{pos} x{n}" for pos, n in sorted(late_gaps.items()))
        if rounds_remaining <= late_reserved:
            lines.append(
                f"Still owed (draft now — you are out of rounds to defer them): {late_str}."
            )
        else:
            lines.append(
                f"Still owed, but NOT yet: {late_str}. These are roster taxes with "
                f"near-random weekly output and no supporting data in this app — do NOT "
                f"recommend one until the final {late_reserved} round(s) of the draft "
                f"(round {ctx.total_rounds - late_reserved + 1} onward). Spending an "
                f"earlier pick here wastes a roster spot on a position that will still "
                f"be freely available at the end."
            )
    lines.append("")

    # Opportunity cost — the decisive section (see its own module comment).
    # Placed immediately before the player list so the board gets read
    # through it rather than as a flat ranking.
    #
    # The pick being advised on is my *next* turn, which is the current pick
    # when I'm on the clock and a look-ahead when I'm not — either way it's
    # my_next_pick_number, whose docstring covers both. The horizon is the
    # turn after that, and the picks in between are the ones that can take a
    # player away from me.
    advised_pick = ctx.my_next_pick_number or ctx.pick_number
    horizon = ctx.my_following_pick_number
    picks_between = max(0, horizon - advised_pick - 1) if horizon else 0

    for section in (
        _format_survival_section(
            ctx.top_available[:_LISTED_PLAYERS], horizon, picks_between
        ),
        _format_positional_dropoff(ctx.top_available, horizon),
    ):
        if section:
            lines.append(section)

    # Top available players, grouped into ADP tiers (cap at 25 to keep
    # prompt tight). Tiers, not a flat 1-25 rank — see _compute_adp_tiers'
    # docstring for why: a precise ordinal table invites treating every
    # ADP decimal as meaningful, when a few-point gap is often just noise.
    lines.append("## Top Available Players (grouped into ADP tiers)")
    lines.append(
        "Players within the same tier are roughly interchangeable in ADP terms — "
        "treat a gap within a tier as noise and use the Opportunity & Performance "
        "Signals / Player News sections to choose among them, rather than the exact "
        "ADP decimal. A gap between tiers is more likely to reflect real drop-off."
    )
    tiers = _compute_adp_tiers(ctx.top_available[:_LISTED_PLAYERS])
    for i, tier in enumerate(tiers, start=1):
        adp_lo, adp_hi = tier[0]["adp"], tier[-1]["adp"]
        lines.append(f"\nTier {i} (ADP {adp_lo:g}-{adp_hi:g}):")
        for p in tier:
            lines.append(
                f"  {p['rank']:<5} {p['name']:<22} {p['position']:<5} {p['team']:<6} ADP {p['adp']}"
            )
    lines.append("")

    # Positional availability + scarcity tier (run-risk context). Starter
    # slots derive from this league's actual configured lineup (not a
    # generic assumption) the same way the standalone /api/recommend/
    # scarcity endpoint's own fix does — FLEX demand folded into both RB
    # and WR, since a flex is usually filled by one or the other.
    counts = ctx.available_counts
    scarcity_starter_slots = {
        "QB": lineup.get("QB", 0),
        "RB": lineup.get("RB", 0) + lineup.get("FLEX", 0),
        "WR": lineup.get("WR", 0) + lineup.get("FLEX", 0),
        "TE": lineup.get("TE", 0),
        "DST": lineup.get("DST", 0),
    }
    scarcity_starter_slots = {pos: n for pos, n in scarcity_starter_slots.items() if n > 0}
    scarcity = compute_position_scarcity(counts, ctx.league_size, scarcity_starter_slots)
    scarcity_str = " | ".join(
        f"{pos}: {counts.get(pos, 0)}" + (f" ({scarcity[pos].upper()})" if pos in scarcity else "")
        for pos in ("QB", "RB", "WR", "TE", "DST")
    )
    lines += [
        "## Positional Availability (undrafted players remaining; scarcity tier in parens)",
        scarcity_str,
        "CRITICAL or LOW means that position is thinning across the league — if a "
        "player there fits your team, that's a legitimate reason to secure them now "
        "rather than wait, since a similar-tier option may not be there at your next "
        "turn. OK means no run risk; scarcity alone shouldn't push a pick there.",
        "",
    ]

    # Run risk — replaces the old full opponent-roster dump, which spent ~12
    # lines describing teams that pick after me and therefore can't take
    # anyone before my next turn. See _format_run_risk.
    run_risk = _format_run_risk(
        ctx.upcoming_pick_slots, ctx.opponent_position_counts, lineup
    )
    if run_risk:
        lines.append(run_risk)

    # Opportunity/performance signals (best-effort; omitted if no metrics at
    # all). Scoped to the same slice shown in the tiers table: ctx.top_available
    # now runs deeper than the displayed board (so the positional drop-off math
    # can see replacement level), and rendering a metrics line for a player the
    # model was never shown is pure prompt bloat.
    metrics_section = _format_metrics_section(
        ctx.top_available[:_LISTED_PLAYERS], ctx.player_metrics, ctx.draft_profiles
    )
    if metrics_section:
        lines.append(metrics_section)

    # Retrieved player news/analysis (best-effort; omitted if unavailable)
    retrieved = _retrieve_player_context(ctx.top_available)
    if retrieved:
        lines.append(retrieved)

    # Output schema
    lines += [
        "## Task",
        "Recommend the best pick for my team right now. Work through these steps in "
        "order — each one narrows the field for the next:",
        "",
        "1. SHORTLIST. Take the top tier or two on the board. Same-tier players are "
        "roughly interchangeable in ADP terms, so ignore the exact ADP decimal between "
        "them; a gap *between* tiers is real drop-off and worth respecting.",
        "2. SUBTRACT WHAT WILL KEEP. Remove anyone in the 'Very likely still on the "
        "board at your next turn' bucket unless they are clearly, not marginally, "
        "better than everyone in the GONE bucket. You can have them later; you cannot "
        "have the GONE players later. This step decides most picks.",
        "3. WEIGH THE COST OF WAITING. Use the Cost of Waiting table plus Run Risk. A "
        "large drop-off at a position you still need to start, with several teams ahead "
        "of you needing it too, is the strongest possible case for taking that position "
        "now. A small drop-off means the position can wait and this pick belongs "
        "elsewhere — say so in `strategy`.",
        "4. BREAK REMAINING TIES ON EVIDENCE. Among what survives, use Opportunity & "
        "Performance Signals (real usage, efficiency, consistency, durability) and "
        "Player News. Call out any candidate whose underlying opportunity looks "
        "stronger than their tier implies as a potential breakout. Never break a tie on "
        "name recognition or last season's box score alone.",
        "5. CHECK LEGALITY LAST. Confirm the pick doesn't leave you unable to field a "
        "legal starting lineup — that only overrides steps 2-4 when the roster shape "
        "section shows an URGENT line.",
        "",
        "Respond with ONLY valid JSON — no markdown, no commentary:",
        json.dumps({
            "strategy": "<1 sentence: what this pick does about the roster's shape, and what you are deliberately deferring to your next turn because it will still be there>",
            "confidence": "<high | medium | low — how clear-cut this call is>",
            "recommendation": {
                "player_id": "<int from the tiers above>",
                "player_name": "<string>",
                "position": "<string>",
                "adp": "<float>",
                "reasoning": "<1-2 sentences explaining why this is the best pick>",
            },
            "alternatives": [
                {
                    # Same constraint as the recommendation: an id outside the
                    # listed players is silently dropped by the parser, which
                    # is how a requested 3 quietly became 2 on screen.
                    "player_id": "<int from the tiers above>",
                    "player_name": "<string>",
                    "position": "<string>",
                    "adp": "<float>",
                    "reasoning": "<1 sentence on this player's own case>",
                    "tradeoff": "<1 sentence: what you gain and give up by taking this instead of the main recommendation>",
                }
            ],
            "alerts": ["<scarcity warnings, tier drop-off flags, breakout/value calls, or a note that a listed player will not survive to your next turn>"],
        }, indent=2),
        "",
        "Use `confidence: low` honestly — when the top few players are genuinely "
        "close, saying so is more useful than manufacturing a reason to separate "
        "them. `tradeoff` should be a real comparison against the main "
        "recommendation (floor vs upside, positional need vs value, survives-to-"
        "your-next-turn vs doesn't), not a restatement of `reasoning`.",
        "",
        f"Return exactly {_MAX_ALTERNATIVES} alternatives whenever there are that "
        "many defensible options on the board — fewer only when there genuinely "
        f"aren't (late in the draft, thin position). More than {_MAX_ALTERNATIVES} "
        "are discarded unread and only cost you output budget. Every "
        "`player_id`, in both the recommendation and the alternatives, must be "
        "one of the ids listed above: anything else is dropped, and you'll have "
        "spent the tokens for nothing. Keep every `reasoning`, `tradeoff`, and "
        "`alert` to a single sentence — the whole response must be complete, "
        "valid JSON, and one that runs long gets cut off mid-object and thrown "
        "away entirely.",
    ]

    return "\n".join(lines)


# Scoring-format-specific emphasis. The system prompt used to hardcode "a PPR
# Sleeper league" regardless of ctx.scoring_format, so a half-PPR or standard
# session got confidently PPR-shaped advice — pass-catching backs and slot
# receivers overvalued by exactly the amount the scoring doesn't award.
_SCORING_NOTES = {
    "ppr": (
        "Every reception is a point, so target volume is the most reliable "
        "predictor of fantasy scoring: pass-catching backs, high-target slot "
        "receivers, and target-hog tight ends are worth more than their raw "
        "yardage suggests."
    ),
    "half_ppr": (
        "Half a point per reception, so reception volume matters but less than "
        "in full PPR — weigh targets alongside yardage and touchdown equity "
        "rather than treating catch volume as decisive."
    ),
    "standard": (
        "No points for receptions, so catch volume is worth far less than in "
        "PPR — prioritize yardage and touchdown equity, and specifically do "
        "not inflate pass-catching backs or high-volume, low-yardage slot "
        "receivers the way PPR rankings do."
    ),
}


def _build_system_prompt(scoring_format: str = "ppr") -> str:
    """
    The advisor's standing instructions.

    Structured as an ordered set of rules with headers rather than the one
    long paragraph this used to be. That's not cosmetic: the model on this
    path is Haiku, which follows short, explicitly-scoped, numbered rules
    considerably more reliably than a 600-word block of prose where every
    constraint has equal visual weight and several ("weigh ADP as one input"
    vs "defer to ADP order for DST/K") appear to contradict each other until
    you notice they're scoped to different positions. Every rule below was
    already present in that paragraph; the scoping is what's new.
    """
    scoring_note = _SCORING_NOTES.get(
        scoring_format.lower().replace("-", "_"), _SCORING_NOTES["ppr"]
    )

    return (
        f"You are an expert fantasy football draft advisor for a "
        f"{scoring_format.upper()} Sleeper redraft league. You give concise, "
        f"data-driven recommendations, and you always respond with valid JSON and "
        f"nothing else.\n\n"

        f"SCORING: {scoring_note}\n\n"

        "RULE 1 — OPPORTUNITY COST DECIDES CLOSE CALLS. A draft pick's real cost is "
        "the best player you won't get back. When two players are close, take the one "
        "who will not survive to your next turn and plan to take the other one later. "
        "The prompt tells you explicitly which players fall in each bucket and what "
        "the drop-off at each position is if you wait; use those numbers rather than "
        "estimating. Never spend a pick on a player labeled 'very likely still on the "
        "board at your next turn' unless he is clearly — not marginally — better than "
        "everything that's about to disappear.\n\n"

        "RULE 2 — VALUE OVER NEED, UNTIL LEGALITY IS AT RISK. Draft the best player "
        "relative to positional replacement level, not the next recognizable name and "
        "not whichever slot happens to be empty. A marginal 'need' edge is rarely "
        "worth passing on a better player. The one exception is an URGENT line in the "
        "roster shape section: at that point you are genuinely running out of picks to "
        "field a legal starting lineup, and filling those slots takes priority.\n\n"

        "RULE 3 — ADP IS A PRIOR, NOT A RANKING. The board is grouped into ADP tiers. "
        "Within a tier, ADP differences are noise and must not decide anything; break "
        "those ties on Opportunity & Performance Signals (usage, efficiency, "
        "consistency) and Player News. Between tiers, the gap is likely real drop-off "
        "and deserves weight. A player whose underlying opportunity — targets, "
        "carries, red zone touches, snap share — outstrips his tier is a breakout "
        "candidate; say so explicitly rather than defaulting to the safest name.\n\n"

        "RULE 4 — SAMPLES AND DURABILITY. A per-game average is only as good as the "
        "sample behind it. A line flagged '[SMALL SAMPLE — weigh this average with "
        "caution]' covers a shortened, often injury-affected season and is not "
        "comparable to a healthy player's full-season average even when the number is "
        "higher. Weeks on the injury report and games missed are live risk for the "
        "season ahead, not footnotes. Do not let a modest per-game edge from a small, "
        "injury-affected sample outweigh a healthier, more available player who "
        "already has the better ADP tier.\n\n"

        "RULE 5 — MISSING DATA IS NOT BAD DATA. Players showing 'No retrieved data' or "
        "'No prior-season metrics' have no prior NFL season to compute from. That is a "
        "coverage gap in this app, never evidence against the player, and must not be "
        "the reason you pass on someone. Rookies frequently show draft capital "
        "(round/pick, college) instead — treat that as genuine signal: early draft "
        "capital is among the strongest predictors of a rookie's eventual role, and a "
        "Day 1-2 pick in a favorable offense is a legitimate breakout call with zero "
        "NFL snaps.\n\n"

        "RULE 6 — DST AND K ARE END-OF-DRAFT ROSTER TAXES. This app has no matchup, "
        "scheme, or opponent-strength data for any position, so nothing grounds a "
        "claim that one defense or kicker is better than another; ADP order is the "
        "only signal you have for them. Never invent outside knowledge about which "
        "real defense or kicker is 'good.' Do not recommend either position until the "
        "prompt says you are inside the reserved final rounds — a DST or K taken "
        "earlier costs you a skill player for a position that will still be freely "
        "available at the end.\n\n"

        "RULE 7 — STAY INSIDE THE EVIDENCE. Every factual claim in your reasoning must "
        "trace to something in the prompt. You have no bye-week data, no depth-chart "
        "narrative beyond what is shown, no 2026 projections, and no injury news past "
        "what appears in Player News & Analysis — so do not reason about bye-week "
        "conflicts, stacking, coaching changes, training-camp reports, or contract "
        "situations. If the deciding factor genuinely isn't in the prompt, say the "
        "call is close and set `confidence` to low."
    )


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _clean_text(value) -> str:
    """
    Free-text field from Claude's JSON, or "" if it isn't a string.

    Deliberately not `str(value)`: coercing a dict or list would put a Python
    repr like "{'unexpected': 'object'}" straight into the UI. These fields are
    optional, so dropping a malformed one is strictly better than showing it.
    """
    return value.strip() if isinstance(value, str) else ""


def _restore_prefill(raw: str) -> str:
    """
    Puts back the opening brace consumed by the prefilled assistant turn (see
    AIService.recommend), which the model continues from rather than
    re-emitting — so its text is a JSON object body with no `{` in front.

    Conditional rather than an unconditional `"{" + raw` because the prefill
    is a strong convention, not a guarantee: a model that re-emits the brace
    anyway, or any future call path that drops the prefill, would otherwise
    have a perfectly good response corrupted into `{{...` and thrown away for
    the ADP fallback — turning a harmless model quirk into a lost
    recommendation. Caught by test_non_text_first_block_is_skipped_not_crashed,
    whose stub returns a complete object.
    """
    return raw if raw.lstrip().startswith("{") else "{" + raw


def _parse_response(raw: str, ctx: RecommendationContext) -> RecommendationResult | None:
    """
    Parses Claude's JSON response into a RecommendationResult.
    Returns None if the JSON is malformed or missing required fields.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Claude sometimes wraps JSON in markdown code fences — strip them
        stripped = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as e:
            # Log the tail as well as the head. The most common real failure is
            # truncation, whose evidence is always at the *end* of the string —
            # logging only raw[:200] showed a perfectly healthy-looking opening
            # brace and hid the fact that the object simply stopped.
            logger.warning(
                "Could not parse Claude response as JSON (%s). "
                "Length=%d chars. Head: %s ... Tail: %s",
                e,
                len(raw),
                raw[:300],
                raw[-200:],
            )
            return None

    rec = data.get("recommendation")
    if not rec or not isinstance(rec, dict):
        return None

    # Validate that the recommended player is actually in our available list.
    # Coerce before comparing — Claude occasionally returns player_id as a
    # JSON string ("3" instead of 3), and rejecting that to the ADP fallback
    # would throw away an otherwise-valid recommendation over a type quirk.
    available_ids = {p["id"] for p in ctx.top_available}

    def _as_id(value) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    if _as_id(rec.get("player_id")) not in available_ids:
        logger.warning("Claude recommended unavailable player id=%s", rec.get("player_id"))
        return None

    # Canonical player rows, keyed by id. Every *factual* field on a suggestion
    # is taken from here rather than from Claude's JSON.
    #
    # This matters: validating player_id against the available list while still
    # rendering the model's own player_name/position/adp meant an available id
    # paired with a drafted player's name displayed that drafted player. The id
    # is the only field we verify, so the id is the only field we trust — the
    # model contributes judgement (reasoning, tradeoff), not data. It also
    # closes the prompt-injection path where RotoWire/Sleeper text reaching the
    # prompt could influence a rendered player name.
    by_id = {p["id"]: p for p in ctx.top_available}

    def _pick(d: dict) -> PickSuggestion | None:
        canonical = by_id.get(_as_id(d.get("player_id")))
        if canonical is None:
            # Unknown or already-drafted id — not in top_available.
            return None
        return PickSuggestion(
            player_id=canonical["id"],
            player_name=canonical["name"],
            position=canonical["position"],
            adp=canonical["adp"],
            # Free text is the model's to write; it's the only thing it adds.
            reasoning=str(d.get("reasoning", "")),
            tradeoff=_clean_text(d.get("tradeoff")),
        )

    recommendation = _pick(rec)
    if recommendation is None:
        return None

    # No separate availability filter needed: _pick already returns None for
    # anything not in the available list.
    raw_alternatives = data.get("alternatives", [])[:_MAX_ALTERNATIVES]
    alternatives = [
        s for d in raw_alternatives
        if isinstance(d, dict) and (s := _pick(d)) is not None
    ]

    # Alternatives vanish for two very different reasons — Claude offered fewer
    # than we asked for, or it offered enough but named players that aren't in
    # the available list and _pick discarded them. Both look identical on
    # screen ("only two options"), so say which happened.
    if len(alternatives) < len(raw_alternatives):
        dropped = [d.get("player_name", "?") for d in raw_alternatives
                   if isinstance(d, dict) and _pick(d) is None]
        logger.warning(
            "Dropped %d of %d alternatives — not in the available list: %s",
            len(raw_alternatives) - len(alternatives), len(raw_alternatives), dropped,
        )
    elif len(alternatives) < _MAX_ALTERNATIVES:
        logger.info(
            "Claude returned %d alternatives (asked for %d) — none were invalid.",
            len(alternatives), _MAX_ALTERNATIVES,
        )

    alerts = [str(a) for a in data.get("alerts", []) if a]

    # These are presentational extras — a malformed or missing value degrades
    # to a sane default rather than rejecting an otherwise-good recommendation
    # to the ADP fallback. Same principle as the player_id coercion above.
    strategy = _clean_text(data.get("strategy"))
    confidence = _clean_text(data.get("confidence")).lower()
    if confidence not in _CONFIDENCE_LEVELS:
        confidence = "medium"

    return RecommendationResult(
        recommendation=recommendation,
        alternatives=alternatives,
        alerts=alerts,
        model=_DEFAULT_MODEL,
        strategy=strategy,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Fallback — top available player by ADP (no AI needed)
# ---------------------------------------------------------------------------

def _fallback(ctx: RecommendationContext, model: str) -> RecommendationResult:
    """
    Returns a safe, data-driven recommendation when Claude is unavailable.
    Simply picks the top available player by ADP.
    """
    if not ctx.top_available:
        raise RuntimeError("No available players to recommend.")

    top = ctx.top_available[0]
    return RecommendationResult(
        recommendation=PickSuggestion(
            player_id=top["id"],
            player_name=top["name"],
            position=top["position"],
            adp=top["adp"],
            reasoning="Best available player by consensus ADP (AI service unavailable).",
        ),
        alternatives=[
            PickSuggestion(
                player_id=p["id"],
                player_name=p["name"],
                position=p["position"],
                adp=p["adp"],
                reasoning="Next best available by ADP.",
                # No comparison is possible without the model — say so rather
                # than inventing a trade-off the fallback didn't reason about.
                tradeoff="",
            )
            for p in ctx.top_available[1:4]
        ],
        alerts=["AI service unavailable — showing best available by ADP only."],
        model=f"{model}:fallback",
        strategy="",
        # The fallback is pure ADP ordering with no roster awareness at all —
        # never present it as a confident call.
        confidence="low",
    )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

# .env.example ships this as a fill-in-the-blank value. If someone copies
# the file without editing it, ANTHROPIC_API_KEY looks "set" to os.getenv()
# but isn't a real key — without this check it would silently be sent to
# Anthropic and only fail once the first recommendation is requested.
_PLACEHOLDER_KEYS = {"your_key_here"}


def _resolve_api_key(api_key_env: str = "ANTHROPIC_API_KEY") -> str | None:
    """
    Reads an API key from the given env var, returning None if it's unset or
    still the .env.example placeholder. The placeholder-key guard lives here,
    once, shared by both client builders below.
    """
    api_key = (os.getenv(api_key_env) or "").strip()

    if api_key in _PLACEHOLDER_KEYS:
        logger.warning(
            f"{api_key_env} is still the placeholder value from .env.example — "
            "treating it as unset."
        )
        return None
    if not api_key:
        logger.warning(f"{api_key_env} not set.")
        return None

    return api_key


def build_anthropic_client(api_key_env: str = "ANTHROPIC_API_KEY") -> anthropic.Anthropic | None:
    """
    Returns a configured *synchronous* client, or None if no real key is set.

    Used by backend/ingestion/fetch_synthesis.py (batch, offline — blocking
    is fine there). The live app uses build_async_anthropic_client instead:
    a sync client inside FastAPI's event loop blocks every other coroutine
    (WebSocket pushes, the Sleeper sync poll loop, all other requests) for
    the full duration of each multi-second Claude call.
    """
    api_key = _resolve_api_key(api_key_env)
    return anthropic.Anthropic(api_key=api_key) if api_key else None


def build_async_anthropic_client(api_key_env: str = "ANTHROPIC_API_KEY") -> anthropic.AsyncAnthropic | None:
    """
    Async variant of build_anthropic_client, for use inside the FastAPI
    event loop (AIService.recommend). Same key resolution and placeholder
    guard; the API call is awaited instead of blocking the loop.
    """
    api_key = _resolve_api_key(api_key_env)
    return anthropic.AsyncAnthropic(api_key=api_key) if api_key else None


class AIService:
    """
    Thin wrapper around the Anthropic client.
    Instantiated once in the FastAPI lifespan and stored on app.state.
    """

    def __init__(self) -> None:
        # Async client — recommend() runs inside FastAPI's event loop, and a
        # sync client there would freeze the whole server (WebSocket pushes,
        # the 2s Sleeper sync poll, every other request) for the full
        # duration of each Claude call.
        self._client = build_async_anthropic_client()
        if self._client is None:
            logger.warning("AI recommendations will use fallback mode until a real key is set.")
        self._model = _DEFAULT_MODEL

    @property
    def is_configured(self) -> bool:
        """True if a real (non-placeholder) Anthropic API key is active."""
        return self._client is not None

    @property
    def model_name(self) -> str:
        return self._model

    async def recommend(self, ctx: RecommendationContext) -> RecommendationResult:
        """
        Generates a pick recommendation for the current draft state.
        Falls back to top-ADP logic if the API call fails.
        """
        if self._client is None:
            return _fallback(ctx, self._model)

        # _build_prompt is sync on purpose (it's also used by the CLI
        # preview in main() below), but it contains the ChromaDB retrieval —
        # up to _MAX_CONTEXT_PLAYERS embedding+query round trips on a cache
        # miss. Run it in a worker thread so those blocking calls don't
        # stall the event loop the same way the sync Anthropic client used
        # to. Thread safety: it only touches SQLite-free plain dicts, the
        # GIL-safe _retrieval_cache, and Chroma's own client.
        prompt = await asyncio.to_thread(_build_prompt, ctx)

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=_MAX_RESPONSE_TOKENS,
                temperature=_TEMPERATURE,
                system=_build_system_prompt(ctx.scoring_format),
                messages=[
                    {"role": "user", "content": prompt},
                    # Prefilled assistant turn: the response is forced to begin
                    # mid-JSON, so there is no room for a "Here's my pick:"
                    # preamble or a ```json fence to wrap it. Both were real
                    # parse failures, and a parse failure here doesn't degrade
                    # the recommendation, it discards it for the ADP fallback.
                    # Also saves a few output tokens per call.
                    {"role": "assistant", "content": "{"},
                ],
            )

            # A truncated response is invalid JSON, so it fails in
            # _parse_response looking exactly like the model returned garbage.
            # Calling it out here makes the difference obvious in the log —
            # "raise the budget" and "fix the prompt" are very different fixes.
            if getattr(response, "stop_reason", None) == "max_tokens":
                logger.warning(
                    "Claude response hit the %d-token ceiling and was truncated — "
                    "the JSON will not parse. Raise _MAX_RESPONSE_TOKENS or tighten "
                    "the requested response shape.",
                    _MAX_RESPONSE_TOKENS,
                )

            # Take the first text block rather than indexing content[0]
            # blindly — an empty content list or a non-text first block
            # (both possible API responses) raised IndexError/
            # AttributeError here before, turning into a 500 on draft day
            # instead of the fallback this method promises (audit W7).
            raw = next(
                (block.text for block in response.content if hasattr(block, "text")),
                None,
            )
            if raw is None:
                logger.warning("Falling back to ADP — Claude response had no text content.")
                return _fallback(ctx, self._model)

            result = _parse_response(_restore_prefill(raw), ctx)

            if result is None:
                logger.warning("Falling back to ADP — could not parse Claude response.")
                return _fallback(ctx, self._model)

            return result

        except anthropic.APIError as e:
            logger.error("Anthropic API error: %s", e)
            return _fallback(ctx, self._model)
        except Exception:
            # This method's contract is "never fail on draft day" — any
            # unexpected error (network weirdness the SDK didn't wrap,
            # response-shape surprises) degrades to ADP, never a 500.
            logger.exception("Unexpected error during recommendation — falling back to ADP.")
            return _fallback(ctx, self._model)


# ---------------------------------------------------------------------------
# CLI — preview the exact prompt Claude would receive, without calling Claude
# ---------------------------------------------------------------------------

def _build_preview_context(top_n: int = 60) -> RecommendationContext:
    """
    Builds a RecommendationContext straight from the DB's top-available
    players — no live draft session needed. Only for main() below; the real
    app builds its context from actual draft state
    (backend/app/api/recommendations.py::_build_context).
    """
    from sqlmodel import Session

    from backend.db import player_repo as repo
    from backend.db.database import engine

    with Session(engine) as session:
        top_available = [
            {
                "id": p.id, "rank": p.rank, "name": p.name, "position": p.position,
                "team": p.team, "adp": p.adp, "sleeper_id": p.sleeper_id,
            }
            for p in repo.get_top_available(session, n=top_n)
        ]
        available_counts = repo.count_available_by_position(session)

    # Slot 1 of a 12-team snake: picking at #1 means waiting until #24, with
    # slots 2-12 then 12-2 in between. Hardcoded here (the real app derives
    # it from DraftStateService) purely so the preview actually exercises the
    # opportunity-cost and run-risk sections instead of silently omitting the
    # ones most worth eyeballing.
    return RecommendationContext(
        pick_number=1, round_number=1, my_slot=1, league_size=12,
        is_my_turn=True, picks_until_my_turn=0, my_next_pick_number=1,
        my_following_pick_number=24,
        upcoming_pick_slots=list(range(2, 13)) + list(range(12, 1, -1)),
        top_available=top_available,
        available_counts=available_counts,
    )


def main() -> None:
    """
    Prints the exact prompt a recommendation would send to Claude — including
    the "## Player News & Analysis (retrieved)" section, if ChromaDB has
    anything for the top candidates — WITHOUT calling the API. Zero cost;
    the fastest way to confirm retrieval is actually reaching the prompt
    before spending on a real recommendation.

    Run manually:
        py -m backend.app.services.ai_service
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ctx = _build_preview_context()
    prompt = _build_prompt(ctx)

    print("=" * 70)
    print("SYSTEM PROMPT")
    print("=" * 70)
    print(_build_system_prompt(ctx.scoring_format))
    print("\n" + "=" * 70)
    print("USER PROMPT")
    print("=" * 70)
    print(prompt)
    print("\n" + "=" * 70)
    if "## Player News & Analysis (retrieved)" in prompt:
        print("Confirmed: retrieved News/Analysis section IS present above.")
    else:
        print(
            "No News/Analysis section in this prompt — either none of the top "
            "candidates have a ChromaDB entry, or the vector store is unavailable "
            "(check the INFO/WARNING log lines above for which)."
        )


if __name__ == "__main__":
    main()
