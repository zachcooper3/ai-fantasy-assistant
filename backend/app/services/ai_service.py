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
import re
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
_MAX_ALTERNATIVES = 4

# Output budget. This has to comfortably fit the whole JSON response: strategy,
# confidence, the recommendation, _MAX_ALTERNATIVES entries that each carry
# both a `reasoning` and a `tradeoff` sentence, and the alerts array. It was
# 1024, which was sized for the older, smaller response shape; adding
# strategy/tradeoff pushed real responses past the ceiling, and a truncated
# response is unparseable JSON — which silently degraded every recommendation
# to the ADP fallback.
_MAX_RESPONSE_TOKENS = 3072

# Low but not zero. The default of 1.0 made two clicks on an unchanged board
# return different players with equally confident reasoning, which reads as
# the tool being unreliable rather than the board being close (that's what
# `confidence: low` is for). 0 fixed that but made the advice feel locked:
# re-rolling a pick you disagreed with returned the identical answer.
#
# 0.3 keeps near-identical behaviour on clear-cut boards while allowing
# genuine alternatives to surface when the top few are close — which is
# exactly when a second opinion is worth anything. It has no measurable
# effect on latency (temperature is free) and, with the assistant prefill
# forcing the opening brace and a fully specified schema, negligible effect
# on JSON validity at this level.
_TEMPERATURE = 0.3

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
    # "take_now" | "might_last" | "will_last", or "" when there's no next turn
    # to survive to. Computed here from ADP and the horizon pick, NOT taken
    # from the model's output — this is the one figure on screen that must
    # agree with the prompt exactly, and asking the model to echo it back
    # would reintroduce the possibility of it saying something else.
    survival: str = ""


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
    # One line per MUST EVALUATE player: taken, or passed and why. Exists to
    # make omission visible — every live failure so far was a top-of-board
    # player never mentioned at all rather than rejected on the merits.
    # Empty for the ADP fallback, which evaluates nothing.
    considered: list[str] = field(default_factory=list)


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

    # {team: {"departed": [...], "arrived": [...]}} — who moved between last
    # season and now, with the share of that team's 2025 volume they
    # represent. The only forward-looking evidence in the prompt apart from
    # ADP; see _format_roster_changes. Empty means the section is omitted,
    # which is also what happens on a database predating PlayerMetrics.team.
    roster_changes: dict[str, dict[str, list[dict]]] = field(default_factory=dict)

    # {position: ppg of the last startable player league-wide}, from
    # compute_replacement_levels over the whole player pool — not just the
    # available slice, since replacement level is a property of the position
    # in this league, not of who happens to be left. Empty means the VOR
    # column is omitted rather than guessed.
    replacement_ppg: dict[str, float] = field(default_factory=dict)


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


# ---------------------------------------------------------------------------
# Roster depth — what you can start in a week, versus what you actually hold
#
# Starting-lineup gaps model LEGALITY: can I field a valid lineup at all.
# That is a genuinely different question from roster construction, and
# conflating the two produced a confirmed live failure. In a 15-round draft
# that opened RB-RB, the gap logic marked RB satisfied from round 3 onward
# (2 held >= 2 required) and never mentioned it again; from round 8 the
# section printed "All required starting slots filled" for eight straight
# picks. The draft ended RB4/WR7 — four bench receivers at a position whose
# late-round replacements are nearly free, and no cover at all at the
# position where they aren't.
#
# The fix isn't a hardcoded "draft 5 RBs" heuristic. It's reporting the
# quantity the gap logic silently discards: how many of a position you can
# start in a single week (base slot plus FLEX) against how many you hold.
# Zero cover at RB means one injury or bye forces a replacement whose cost
# is quantified live in the Cost of Waiting section.
# ---------------------------------------------------------------------------

# A note on the FLEX, because two earlier versions of this section got it
# wrong in opposite directions and both produced a flag that fired on nearly
# every roster:
#
#   1. Adding the FLEX to each eligible position's startable count credited
#      one slot to both RB and WR, so the counts summed to 8 against a
#      7-slot lineup. A roster holding exactly 2 RB was reported "short"
#      whether or not it was, and a deliberately RB-heavy 4RB/2WR roster was
#      reported short at WR.
#   2. Allocating the FLEX to whichever position had the largest surplus
#      fixed the arithmetic but made the result depend on an arbitrary
#      tie-break: a balanced 3RB/3WR roster has one spare either way, yet
#      whichever side lost the tie looked exposed and the other looked deep.
#
# Measuring `beyond` against the BASE requirement alone sidesteps both. The
# FLEX is a use for a spare player, not a second requirement, so it never
# enters the exposure math — it's mentioned once in the rendered table so
# the counts aren't misread, and that's all.


def _compute_roster_depth(
    my_roster: list[dict],
    lineup: dict[str, int],
) -> dict[str, dict[str, int]]:
    """
    Returns {position: {"hold", "required", "beyond", "base_met"}} for the
    skill positions, where `required` is that position's mandatory starting
    slots and `beyond` is how many you hold in excess of them.

    `beyond` is measured against the BASE requirement, deliberately not
    against base-plus-FLEX. The FLEX is a *use* for a spare player, not a
    second requirement, and folding it into the denominator produces
    artifacts: a perfectly balanced 3RB/3WR roster has one spare who could
    flex at either position, but crediting the FLEX to whichever side wins
    an arbitrary tie-break makes that side look exposed and the other look
    deep. Measuring against base alone is assignment-independent — 3RB/3WR
    correctly reads as one spare at each and raises nothing.

    beyond == 0 means every player you hold there is locked into a mandatory
    slot: one injury or bye and that slot goes to a waiver pickup. That state
    is entirely invisible to _compute_roster_gaps, which reports nothing once
    the base requirement is met, and it's what let a live draft sit on two
    backs from round 3 to round 11 while the prompt called RB satisfied.
    """
    have: dict[str, int] = {}
    for p in my_roster:
        have[p["position"]] = have.get(p["position"], 0) + 1

    depth: dict[str, dict[str, int]] = {}
    for pos in ("QB", "RB", "WR", "TE"):
        required = lineup.get(pos, 0)
        if required == 0 and not have.get(pos):
            continue
        hold = have.get(pos, 0)
        depth[pos] = {
            "hold": hold,
            "required": required,
            "beyond": hold - required,
            # Positions failing their base requirement are NOT depth problems
            # — they're unfilled starting slots, already reported above.
            # Without this distinction the callout fires on every position
            # for the first several rounds, which is true, useless, and
            # drowns the one position that genuinely is a depth problem.
            "base_met": hold >= required,
        }
    return depth


def _format_roster_depth_section(
    depth: dict[str, dict[str, int]],
    spots_left: int,
    lineup_has_flex: bool = True,
) -> str:
    """
    The roster table plus an explicit read on which positions have no cover.

    Exists to replace a dead end: once every base slot was filled the prompt
    said "All required starting slots filled — every remaining pick is depth
    or upside," which is the emptiest possible instruction at exactly the
    point in a draft where roster construction is the whole decision.
    """
    if not depth:
        return ""

    lines = [
        "## Roster Construction (holdings vs. mandatory starting slots)",
        f"  {'pos':<5}{'hold':>6}{'required':>10}{'beyond':>8}",
    ]
    unfilled: list[str] = []
    exposed: list[str] = []   # multi-slot position with nothing spare
    spare: list[str] = []
    for pos in ("QB", "RB", "WR", "TE"):
        d = depth.get(pos)
        if d is None:
            continue
        note = ""
        if not d["base_met"]:
            note = "  <-- starting slot still unfilled (see above)"
            unfilled.append(pos)
        elif d["beyond"] == 0:
            note = "  <-- every one of them is a mandatory starter"
            # Only positions with two or more mandatory slots count as
            # exposed. Holding exactly one QB and one TE is the normal,
            # correct state of nearly every roster in a 1-QB league —
            # flagging those made this callout fire on 30 of 30 roster
            # shapes, a constant with no information in it. Keyed off
            # `required >= 2` rather than a hardcoded {RB, WR} so a
            # superflex or 2-TE league is handled by the same logic.
            if d["required"] >= 2:
                exposed.append(pos)
        elif d["beyond"] >= 2:
            spare.append(pos)
        lines.append(
            f"  {pos:<5}{d['hold']:>6}{d['required']:>10}{d['beyond']:>+8}{note}"
        )

    if lineup_has_flex:
        lines.append(
            "  (your FLEX is filled by one of the 'beyond' players — an RB, WR or TE)"
        )
    lines.append(f"Roster spots left to fill after this pick: {max(0, spots_left - 1)}.")

    # The callout describes an IMBALANCE, not a state. "No spare anywhere" in
    # round 8 is true of nearly every roster in the league and implies no
    # particular action; "spare at WR while RB has none" names a specific
    # trade you are making and can still unmake. Requiring both sides means
    # this stays silent on balanced rosters instead of adding a fixed line to
    # every prompt.
    #
    # Held back until every base slot is filled: before that, "no spare at
    # RB" competes with "you have no WR2", and the unfilled slot is
    # unambiguously more urgent and already stated above.
    if exposed and spare and not unfilled:
        exposed_str = ", ".join(
            f"{p} (start {depth[p]['required']}, hold {depth[p]['hold']})" for p in exposed
        )
        lines.append(
            f"Imbalance: two or more spare at {', '.join(spare)}, none at "
            f"{exposed_str}. Adding to a position that already has spare, while a "
            f"multi-slot position has none, is a common way to lose a draft you were "
            f"winning — but only when the replacement there is genuinely worse. Use "
            f"the Cost of Waiting figures to decide that, not this line alone."
        )
    lines.append("")
    return "\n".join(lines)


def _full_lineup(lineup: dict[str, int], kickers_available: bool = True) -> dict[str, int]:
    """
    The configured lineup plus the kicker DraftConfig doesn't model — see
    _K_SLOTS. Everything downstream (gap math, scarcity, the roster-shape
    line) reads the lineup through here so the kicker requirement can't be
    visible in one place and missing in another.

    `kickers_available` exists because the requirement is worthless when
    there is nothing to satisfy it with. This app's ADP source returns no
    kickers at all — 226 players, zero K, confirmed live — so _K_SLOTS
    produced an instruction the model physically could not follow: the
    parser only accepts a player_id drawn from the available board, and no
    kicker is ever on it. A full draft ended with the final round spent on a
    receiver while the prompt insisted a kicker was still owed.

    Asserting a requirement the data cannot meet is worse than omitting it,
    because it also distorts the rounds-remaining math that reserves the end
    of the draft for DST and K.
    """
    if not _K_SLOTS or not kickers_available:
        return {k: v for k, v in lineup.items() if k != "K"}
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
# A tier break is a gap unusually large FOR THIS BOARD, judged against the
# other gaps on it — not against an absolute number and not against the ADP
# value itself.
#
# The previous rule, max(3.0, prev_adp * 0.10), was calibrated against the
# wrong quantity and silently produced a single tier at nearly every point in
# a draft. Consecutive available players sit about 1 ADP point apart (roughly
# 230 players spread over 230 ADP points), and the largest gap anywhere in a
# 25-player board is 2.3-4.2 — while the threshold ran from 3.0 early to over
# 10 by ADP 100. It could not be exceeded, so no break was ever emitted.
#
# The damage was not a missing feature. The board carries the instruction
# "players within the same tier are roughly interchangeable, treat a gap
# within a tier as noise" — so a single 25-player tier told the model that
# the ADP-1.6 player and the ADP-24.7 player were equivalent, and that any
# ADP evidence separating them should be discarded. Confirmed live: a back at
# ADP 45.2 went unrecommended in favour of players 25 ADP points later, all
# of them presented as one undifferentiated block.
#
# A percentile self-calibrates to whatever density the board actually has,
# early or late, thin or dense. The size cap then splits any tier that stays
# implausibly large, so the "interchangeable" claim is never made about a
# group big enough for it to be false.
_TIER_GAP_PERCENTILE = 0.80
_MIN_TIER_GAP = 0.5
_MAX_TIER_SIZE = 8


def _split_oversized(tier: list[dict]) -> list[list[dict]]:
    """Recursively splits a tier at its largest internal gap until every
    piece is at most _MAX_TIER_SIZE. A tier of twenty players is not a
    statement anyone would defend about football, whatever the gaps say."""
    if len(tier) <= _MAX_TIER_SIZE:
        return [tier]
    gaps = [(tier[i + 1]["adp"] - tier[i]["adp"], i) for i in range(len(tier) - 1)]
    _, at = max(gaps, key=lambda g: (g[0], -abs(g[1] - len(tier) // 2)))
    return _split_oversized(tier[: at + 1]) + _split_oversized(tier[at + 1:])


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
    if len(players) < 3:
        return [list(players)]

    gaps = sorted(b["adp"] - a["adp"] for a, b in zip(players, players[1:]))
    # Break on gaps in the top fifth of this board's own gap distribution.
    threshold = max(_MIN_TIER_GAP, gaps[min(int(len(gaps) * _TIER_GAP_PERCENTILE), len(gaps) - 1)])

    tiers: list[list[dict]] = [[players[0]]]
    for prev, cur in zip(players, players[1:]):
        if cur["adp"] - prev["adp"] >= threshold:
            tiers.append([])
        tiers[-1].append(cur)

    return [piece for tier in tiers for piece in _split_oversized(tier)]


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
# Value over replacement — the cross-position common currency
#
# Diagnosed from a live draft that took five receivers in seven rounds. Two
# facts about this data explain it, and the prompt expressed neither:
#
#   1. The board is WR-dense at every depth. In the top 24 by ADP there are
#      12 WRs and 11 RBs; by ADP 97-144 it's 20 WRs to 10 RBs. Working down
#      an ADP-sorted board takes more receivers than backs by construction.
#   2. Raw points per game are nearly identical between the two at every
#      rank (RB12 15.2 vs WR12 14.7), so the metrics section gives no reason
#      to prefer either.
#
# What differs is REPLACEMENT level: the worst starter you could still roster
# at that position. RB30 scores 11.1, WR30 scores 11.9, and the gap widens
# deeper (RB36 9.1 vs WR36 11.3). So a 15.2 ppg back is worth +4.1 over the
# back who'd replace him while a 14.7 ppg receiver is worth only +2.8 — at
# equal ADP and near-equal raw points, the back is worth appreciably more.
# Across the top 36 by ADP, mean VOR is +6.02 for RBs against +4.28 for WRs.
#
# This is the only quantity here that compares a back to a receiver on one
# scale, which is exactly what "best player available" requires and what the
# prompt previously had no way to say.
#
# Deliberately NOT used as a ranking on its own. Simulating 12-team drafts
# against this data, a VOR-greedy strategy beat ADP-greedy when last season's
# points were treated as truth (112.9 vs 107.4) but fell BELOW it once
# outcomes were regressed toward the market's view (104.9 vs 114.4) — VOR
# built on trailing data is fragile precisely where trailing data is wrong.
# It's offered to the model as one signal beside ADP, with the divergence
# between them called out, rather than as an ordering to follow.
# ---------------------------------------------------------------------------

def compute_replacement_levels(
    ppg_by_position: dict[str, list[float]],
    league_size: int,
    starter_slots: dict[str, int] | None = None,
) -> dict[str, float]:
    """
    Returns {position: ppg of the last startable player league-wide} — the
    baseline a player at that position has to beat to be worth anything.

    `ppg_by_position` holds every rostered-quality player's prior-season
    points per game, per position, in any order. `starter_slots` is per-team
    demand including a share of the FLEX (defaults to _STARTER_SLOTS, the
    same 1/3/3/1 shape the scarcity math uses, so the two can't disagree
    about how many backs a league starts).

    A position with fewer players on file than the league needs falls back to
    its worst known player rather than raising — an incomplete pool is a data
    gap, and draft day is not the time to crash over one.
    """
    starter_slots = starter_slots or _STARTER_SLOTS
    levels: dict[str, float] = {}
    for pos, slots in starter_slots.items():
        pool = sorted((v for v in ppg_by_position.get(pos, []) if v is not None), reverse=True)
        if not pool:
            continue
        rank = max(1, int(league_size * slots))
        levels[pos] = pool[min(rank, len(pool)) - 1]
    return levels


def _vor(player: dict, player_metrics: dict[int, dict], replacement: dict[str, float]) -> float | None:
    """Points per game above this player's positional replacement level, or
    None when he has no prior-season data. None is not zero: a rookie with no
    NFL snaps has an unknown VOR, and rendering it as 0.0 would read as
    'exactly replacement-level', which is a claim the data doesn't support."""
    m = player_metrics.get(player["id"]) if player_metrics else None
    ppg = m.get("fantasy_points_avg") if m else None
    base = replacement.get(player["position"])
    if ppg is None or base is None:
        return None
    return ppg - base


# ---------------------------------------------------------------------------
# Roster changes — the only genuinely forward-looking signal in the prompt
#
# Everything else the app knows about a player describes a season that has
# already finished. ADP is the one exception, and it arrives as a single
# number with no explanation.
#
# Diagnosed from a live disagreement: FantasyPros' consensus favoured Josh
# Downs over Deebo Samuel 96/4, while every production figure here favoured
# Deebo — 25% target share against 18%, 72% snaps against 59%, 11.8 ppg
# against 8.7. The app had no way to see why. Downs's 18% was earned in 2025
# alongside Michael Pittman Jr., who took 25% of Indianapolis's targets and
# is now on Pittsburgh. His share understates the opportunity in front of
# him, and last season's numbers cannot say so.
#
# This is a VOLUME argument, not a talent one. Vacated targets do not
# automatically flow to whoever remains, and the prompt says so — the point
# is to put the change in front of the model, not to project the outcome.
# ---------------------------------------------------------------------------

# Below this share of a team's 2025 volume, a departure or arrival isn't
# worth a line. A fifth receiver moving on frees up nothing anyone drafts.
_ROSTER_CHANGE_MIN_SHARE = 0.05

# Positions whose comings and goings change the opportunity available to
# other players. A quarterback change matters enormously in reality, but
# not through the share arithmetic this section does.
_ROSTER_CHANGE_POSITIONS = ("RB", "WR", "TE")


# Which volume a position competes for. A receiver cares about vacated
# targets and a back about vacated carries; netting the two together would
# describe nobody's situation.
_POSITION_CURRENCY = {"WR": "targets", "TE": "targets", "RB": "carries"}

# Below this net change, the annotation is noise on an already-long row.
_TEAM_OPPORTUNITY_MIN = 0.08


def _team_opportunity(
    player: dict,
    roster_changes: dict[str, dict[str, list[dict]]],
) -> str:
    """
    A short tag for a player's row: how much of the volume he competes for
    has left his team, or arrived to compete with him.

    This exists because the team-level section alone did not work. Confirmed
    live: Josh Downs sat first on the board carrying "(BELOW replacement)" —
    a backward-looking verdict, stated with certainty, attached directly to
    his name — while the fact that Indianapolis had lost 21% of its targets
    sat fifty lines earlier under a team heading, reachable only by joining
    on the team code. The evidence against him was on his row; the evidence
    for him was not, and he went unrecommended.

    Netted in the player's own currency, and only departures/arrivals at
    positions that actually compete with him: a running back leaving does
    nothing for a receiver's target share.
    """
    team = player.get("team")
    currency = _POSITION_CURRENCY.get(player.get("position", ""))
    if not team or not currency:
        return ""
    change = roster_changes.get(team)
    if not change:
        return ""

    out = sum(p["share"] for p in change.get("departed", [])
              if p.get("currency") == currency)
    inc = sum(p["share"] for p in change.get("arrived", [])
              if p.get("currency") == currency)
    net = out - inc
    # Rendered even at zero. A figure that appears only when it is large
    # reads as an occasional annotation; one that is always present reads as
    # a metric, and this needs to carry the same weight as the ADP and VOR
    # sitting beside it. "OPP +0%" is also real information — it says the
    # situation has NOT changed, which is exactly what distinguishes a
    # player whose numbers still describe his role from one whose don't.
    if abs(net) < _TEAM_OPPORTUNITY_MIN:
        net = 0.0

    short = "tgt" if currency == "targets" else "car"
    return f"OPP {net:+.0%} {short}"


def _format_roster_changes(
    board: list[dict],
    roster_changes: dict[str, dict[str, list[dict]]],
) -> str:
    """
    Per-team departures and arrivals since last season, for the teams
    represented on the visible board.

    Keyed by team rather than repeated per player: several board players
    usually share a team, and the same three lines under each of them would
    be pure duplication. Every board row already shows its team, so the
    model can connect them.

    Only counts players this app knows about — the ADP pool is a few hundred
    fantasy-relevant names, so a retirement or a cut leaves no trace here.
    Departures are therefore a floor, never a complete accounting, and the
    rendered text says so rather than implying the list is exhaustive.
    """
    if not roster_changes:
        return ""

    teams = sorted({p["team"] for p in board if p.get("team")})
    lines: list[str] = []
    for team in teams:
        change = roster_changes.get(team)
        if not change:
            continue
        left = change.get("departed", [])
        joined = change.get("arrived", [])
        if not left and not joined:
            continue

        parts: list[str] = []
        if left:
            parts.append("LOST " + ", ".join(
                f"{p['name']} ({p['share_label']})" for p in left))
        if joined:
            parts.append("GAINED " + ", ".join(
                f"{p['name']} ({p['share_label']} with {p['from_team']})" for p in joined))
        lines.append(f"- {team}: " + " | ".join(parts))

    if not lines:
        return ""

    return "\n".join([
        "## Roster Changes Since Last Season",
        "Shares are of that team's 2025 totals, so they describe how much "
        "opportunity moved, not how good the player is. A team that LOST volume "
        "has work available for whoever stayed — meaning an incumbent's own "
        "2025 share understates what is in front of him now. A team that GAINED "
        "someone has new competition, so an incumbent's share overstates it. "
        "This is the only forward-looking evidence here apart from ADP, and it "
        "is the usual reason ADP disagrees with last season's production: when "
        "the market ranks a player above what his numbers justify, vacated "
        "volume is the first thing to check.",
        "Two limits. Vacated work does not automatically go to whoever remains "
        "— treat it as opportunity, not as a projection. And only players in "
        "this app's pool are visible, so retirements and cuts leave no trace: "
        "these lists are a floor, not a complete accounting.",
        *lines,
        "",
    ])


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
# Label wording is load-bearing, and the first version got it dangerously
# wrong. The bucket meaning "on the board now, but not at your next turn"
# was called GONE and headed "Almost certainly gone by then". Confirmed
# live mid-draft: the model read that as "these players are unavailable"
# and recommended the best receiver it believed was left *after* them,
# writing "the highest-value receiver available after Rice and Brown
# disappear" — while Rice and Brown were both sitting on the board.
#
# That inverts RULE 1 exactly: the group you must act on first got read as
# the group you cannot have. Every label below now leads with present
# availability, and the section header states outright that all three
# groups are available right now.
_SURVIVAL_GONE = "TAKE NOW OR LOSE HIM"
_SURVIVAL_TOSSUP = "MIGHT LAST"
_SURVIVAL_SAFE = "WILL LAST"

# Stable machine-readable codes for the API, so the UI can style each bucket
# without string-matching prose that exists to be tuned. The prompt wording
# above has already been rewritten once (it used to say "GONE", which the
# model read as unavailable) and will be again; the codes must not move with
# it.
_SURVIVAL_CODES = {
    _SURVIVAL_GONE: "take_now",
    _SURVIVAL_TOSSUP: "might_last",
    _SURVIVAL_SAFE: "will_last",
}


def _survival_code(adp: float, horizon_pick: int | None) -> str:
    """The bucket as an API code, or "" when there is no next turn to survive
    to — the last pick of a draft has no opportunity cost to express."""
    label = _survival(adp, horizon_pick)
    return _SURVIVAL_CODES.get(label, "") if label else ""


# Draft value: how far a player has fallen past his own ADP by the time
# you're on the clock. The frontend has always shown this (AIPanel.tsx calls
# adpValue(adp, pickNumber) to render "+4 vs pick 17 - reach"), but the
# PROMPT never did — it handed the model a raw ADP and a pick number as two
# separate facts and left the subtraction to it.
#
# That subtraction is what it got wrong live: at pick 17 it took George
# Pickens (ADP 21.3, a 4.3-pick reach) over Rashee Rice (ADP 11.5, a
# 5.5-pick faller who also led the board on VOR and every usage metric).
# Arithmetic over two dozen ADP values is the thing a small fast model is
# worst at and the thing Python is free at.
_VALUE_NOISE = 2.0   # within this many picks of ADP is neither value nor reach


def _draft_value(adp: float, pick_number: int) -> str:
    """"FALLING +5.5" / "reach -4.3" / "at ADP" for a player at this pick."""
    delta = pick_number - adp
    if delta >= _VALUE_NOISE:
        return f"FALLING +{delta:.0f}"
    if delta <= -_VALUE_NOISE:
        return f"reach {delta:.0f}"
    return "at ADP"


# How many players get flagged as mandatory evaluations. Small on purpose:
# the point is to stop the top of the board being skipped, not to hand over
# a slate so long that the requirement becomes busywork.
_MUST_EVALUATE = 6


def _shortlist(
    board: list[dict],
    pick_number: int,
    player_metrics: dict[int, dict],
    replacement: dict[str, float],
) -> list[dict]:
    """
    The players the model is not allowed to ignore: best ADP, highest VOR,
    biggest faller. Union of three cheap rankings, deduped, capped.

    This exists because every recommendation failure observed live was an
    OMISSION rather than a bad choice — Rashee Rice and A.J. Brown were both
    top-of-board and simply never appeared in the response, not rejected on
    the merits. Rules about how to choose cannot fix a player never being
    considered; only a required consideration set can.

    Deliberately several separate rankings rather than one composite score. A
    composite would be a ranking, and ranking the board for the model is
    exactly the over-constraint to avoid: it should still decide, it just
    has to look first. Separate axes also surface genuinely different players
    — the best ADP and the best VOR are frequently not the same man, and that
    disagreement is the interesting part of the pick.

    NOTE: an earlier version used "biggest faller" as the third axis, scoring
    players by (pick_number - adp). Since pick_number is the same constant
    for everyone on the board, maximising that is just minimising ADP — it
    returned the identical players to the ADP axis and the shortlist was
    silently only two axes wide. Replaced with best-available-per-position,
    which is the axis that was actually missing: without it a board whose top
    is all receivers produces an all-receiver shortlist, and the
    cross-position call never gets forced.
    """
    if not board:
        return []

    def vor_of(p: dict) -> float | None:
        return _vor(p, player_metrics, replacement)

    by_adp = sorted(board, key=lambda p: p["adp"])[:2]
    with_vor = [(v, p) for p in board if (v := vor_of(p)) is not None]
    by_vor = [p for _, p in sorted(with_vor, key=lambda t: -t[0])[:2]]

    # Best available at each position, so the shortlist can never collapse
    # onto one position and hide the choice that matters most.
    per_position: list[dict] = []
    for pos in ("QB", "RB", "WR", "TE"):
        at_pos = [p for p in board if p["position"] == pos]
        if at_pos:
            per_position.append(min(at_pos, key=lambda p: p["adp"]))

    # Priority order is load-bearing, because the cap truncates. Position
    # representatives go first: quarterbacks and tight ends have late ADP by
    # nature, so a cap applied to an ADP-sorted union drops exactly the
    # entries this axis exists to guarantee. Verified: with four distinct
    # receivers winning the ADP and VOR axes, the RB and TE reps were being
    # cut and the shortlist collapsed back to one position.
    #
    # The global best ADP is always some position's representative, so
    # `by_adp` mostly reinforces rather than adds — it ranks last.
    seen: set[int] = set()
    out: list[dict] = []
    for p in [*per_position, *by_vor, *by_adp]:
        if p["id"] not in seen and len(out) < _MUST_EVALUATE:
            seen.add(p["id"])
            out.append(p)
    return sorted(out, key=lambda p: p["adp"])


def _format_shortlist_section(
    shortlist: list[dict],
    pick_number: int,
    player_metrics: dict[int, dict],
    replacement: dict[str, float],
    roster_changes: dict[str, dict[str, list[dict]]] | None = None,
) -> str:
    if not shortlist:
        return ""
    lines = [
        "## MUST EVALUATE — you are required to give a verdict on each of these",
        "These are the best ADP, the highest VOR, and the biggest fallers on the "
        "board. You do NOT have to pick one of them, but you must not ignore them: "
        "return one line for each in the `considered` array saying why you took him "
        "or why you passed. Silently omitting a player here is the single most common "
        "way this tool has gone wrong.",
    ]
    for p in shortlist:
        v = _vor(p, player_metrics, replacement)
        # Same annotation as the board — a player can appear in both, and a
        # warning that shows in one place and not the other invites reading
        # the unflagged copy as the more favourable one.
        if v is None:
            vor_str = "VOR --"
        else:
            vor_str = f"VOR {v:+.1f}" + (" (BELOW replacement)" if v < 0 else "")
        opp = _team_opportunity(p, roster_changes or {})
        lines.append(
            f"  id={p['id']} {p['name']} ({p['position']}, {p['team']}) "
            f"ADP {p['adp']:g}, {vor_str}, {_draft_value(p['adp'], pick_number)}"
            + (f", {opp}" if opp else "")
        )
    lines.append("")
    return "\n".join(lines)


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
        "EVERY PLAYER LISTED BELOW IS AVAILABLE RIGHT NOW AND CAN BE DRAFTED WITH "
        "THIS PICK. The groups describe what is likely to happen by pick "
        f"#{horizon_pick} — they do NOT describe who is on the board today. Never "
        "reason about a player here as though he is already taken.",
        "Given that: a 'WILL LAST' player can still be had at your next turn, so "
        "spending this pick on him forfeits every 'TAKE NOW OR LOSE HIM' player for "
        "nothing. Only pass on a TAKE NOW player for someone clearly better, not for "
        "someone merely similar.",
    ]
    for label, header in (
        (_SURVIVAL_GONE, "TAKE NOW OR LOSE HIM — on the board now, will not be at your next turn"),
        (_SURVIVAL_TOSSUP, "MIGHT LAST — on the board now, could go either way"),
        (_SURVIVAL_SAFE, "WILL LAST — on the board now, and very likely still there at your next turn"),
    ):
        names = buckets[label]
        if names:
            lines.append(f"- {header}: " + "; ".join(names))
    lines.append("")
    return "\n".join(lines)


def _ppg(player_metrics: dict[int, dict], player: dict) -> float | None:
    """Prior-season PPR points per game, or None. Points are the unit every
    roster decision is ultimately denominated in; ADP is only a proxy for
    them, and "46 ADP points of drop-off" is not a quantity anyone can
    weigh against a roster hole."""
    m = player_metrics.get(player["id"]) if player_metrics else None
    return m.get("fantasy_points_avg") if m else None


def _format_positional_dropoff(
    board: list[dict],
    horizon_pick: int | None,
    player_metrics: dict[int, dict] | None = None,
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
            # Points where they're known, ADP only as the fallback — a
            # drop-off stated in prior-season PPR ppg is directly comparable
            # to what a roster hole costs, which "46 ADP points" is not.
            best_ppg = _ppg(player_metrics or {}, best)
            surv_ppg = _ppg(player_metrics or {}, survivor)
            if best_ppg is not None and surv_ppg is not None:
                cost = (
                    f"Cost of waiting: {best_ppg - surv_ppg:+.1f} PPR ppg "
                    f"({best_ppg:.1f} -> {surv_ppg:.1f}), {gap:.0f} ADP points"
                )
            else:
                cost = f"Cost of waiting: {gap:.0f} ADP points at the position"
            lines.append(
                f"- {pos}: best available {best['name']} (ADP {best['adp']:g}); best "
                f"likely to survive to your next turn is {survivor['name']} (ADP "
                f"{survivor['adp']:g}). {cost}."
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
# Retrieval must cover every player the model can actually recommend.
#
# This was 10 while the board showed 25, so fifteen recommendable players
# carried no news and no injury information whatsoever. Confirmed live: at a
# round-13 pick, only the top 10 of the board had any status data, and the
# tight end being recommended sat at position 14 with none. Worse, the gap is
# invisible from inside the prompt — a player with no retrieved chunk and a
# player with nothing wrong with him render identically.
#
# The original cap of 10 was a latency guard from when each candidate cost
# its own similarity search. Retrieval is now a single metadata lookup for
# the whole board (see _retrieve_player_context), so covering 25 players
# costs the same ~2 ms as covering 10 and the cap no longer buys anything.
_MAX_CONTEXT_PLAYERS = _LISTED_PLAYERS
_MAX_CHUNKS_PER_PLAYER = 3

# Chroma content is only ever updated by the offline ingestion scripts
# (chunker.py / fetch_synthesis.py), never by anything a live draft session
# does — so it's safe to cache a player's retrieved chunks for the lifetime
# of this process. It matters much less than it did — the lookup it skips is
# now ~2 ms rather than a per-player embedding — but it still avoids
# re-fetching a board that barely changes pick to pick, and it caches
# NEGATIVE results too, so a player with no chunks isn't re-queried on every
# pick for the rest of the draft. Unbounded is fine: a season's player pool
# is a few hundred entries, trivial memory for one draft session.
_retrieval_cache: dict[str, list[str]] = {}


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
        from backend.rag.vector_store import fetch_by_metadata
    except Exception as e:
        logger.info(f"Vector store unavailable — building prompt without retrieved context: {e}")
        return ""

    candidates = [
        p for p in top_available[:_MAX_CONTEXT_PLAYERS] if p.get("sleeper_id")
    ]
    wanted = [str(p["sleeper_id"]) for p in candidates
              if str(p["sleeper_id"]) not in _retrieval_cache]

    # ONE metadata lookup covering every uncached candidate, instead of one
    # similarity search per player.
    #
    # The old shape ran _MAX_CONTEXT_PLAYERS separate `query` calls, each of
    # which embeds its query text through a local ONNX model. That is real
    # CPU work, so the thread pool wrapped around it bought far less than it
    # appeared to, and raising the candidate count from 10 to 25 scaled the
    # cost linearly — the direct cause of recommendations going from about
    # ten seconds to forty.
    #
    # None of that work was ever needed. The filter already pins results to
    # a single player's chunks by sleeper_id, so the embedded query text was
    # only breaking ties between chunks that are all about that player
    # anyway. `fetch_by_metadata` uses Chroma's `get`, which touches no
    # model: measured at 2.2 ms for all 25 players against the live store.
    if wanted:
        try:
            rows = fetch_by_metadata({"$and": [
                {"sleeper_id": {"$in": wanted}},
                {"chunk_type": {"$in": ["what_happened", "what_it_means"]}},
            ]})
        except Exception as e:
            logger.warning(f"Vector fetch failed — prompt built without retrieved context: {e}")
            rows = []

        grouped: dict[str, list[str]] = {}
        for meta, doc in rows:
            sid = str((meta or {}).get("sleeper_id") or "")
            if sid and doc:
                grouped.setdefault(sid, []).append(doc)
        # Cache every player asked for, including the ones with nothing —
        # a negative result is worth remembering too, and without this an
        # empty player is re-fetched on every single pick.
        for sid in wanted:
            _retrieval_cache[sid] = grouped.get(sid, [])[:_MAX_CHUNKS_PER_PLAYER]

    outcomes = [(p, _retrieval_cache.get(str(p["sleeper_id"]), [])) for p in candidates]

    sections: list[str] = []
    hits = 0
    checked = 0
    for p, results in outcomes:
        checked += 1
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
# These were silent for the whole 2026-07 to 2026-08 stretch: four separate
# column/ID mismatches in fetch_metrics.py meant snap and depth-chart data
# never reached the database, and an absent column reads as 0.0 rather than
# raising, so the fields simply stored as None. All four are fixed and these
# now populate — see fetch_metrics.py's header. If this clause ever goes
# quiet again, run `py -m backend.tools.diagnose_ingestion` before assuming
# the players genuinely have no trend.
_TREND_FIELDS: list[tuple[str, str, str]] = [
    ("target share trend", "target_share_trend", "{:+.0%}"),
    ("snap % trend", "snap_pct_trend", "{:+.0%}"),
    ("depth chart Δ (neg=moving up)", "depth_chart_trend", "{:+d}"),
]

# ---------------------------------------------------------------------------
# Rate-stat trust gates
#
# A rate is only as meaningful as the denominator it was divided by, and the
# DB is full of rates computed over denominators of 1 or 2: "catch rate 100%"
# for a QB targeted once, "Y/carry 40.5" for a WR with two end-arounds,
# "RACR -221.00" for a back with -1.0 season air yards. Every one of these
# reached the prompt as though it were a real efficiency signal.
#
# This is a *display* guard, deliberately duplicated with the ingestion fix in
# fetch_metrics.py rather than replacing it. The DB already on disk holds the
# bad values, re-running ingestion is a separate manual step, and draft day is
# not when you want to discover you skipped it. Ingestion stops writing them;
# this stops showing them regardless of what's stored.
# ---------------------------------------------------------------------------

# Attempts required before a per-attempt rate is worth showing at all.
_MIN_TARGETS_FOR_RECEIVING_RATES = 20
_MIN_CARRIES_FOR_RUSHING_RATES = 20

_RECEIVING_RATE_FIELDS = {
    "yards_per_target", "yac_per_reception", "racr", "catch_rate", "target_share",
}
_RUSHING_RATE_FIELDS = {"yards_per_carry", "carry_share"}

# Backstop for values that clear the volume gate but still can't be right —
# a real stat outside these bounds is a computation error, not a remarkable
# player. D'Andre Swift had 48 targets (well past the gate) and RACR -11.96.
# Ranges are deliberately generous; the goal is catching broken math, not
# second-guessing unusual seasons.
_METRIC_SANITY_BOUNDS: dict[str, tuple[float, float]] = {
    "racr": (0.2, 3.0),
    "catch_rate": (0.0, 1.0),
    "target_share": (0.0, 1.0),
    "carry_share": (0.0, 1.0),
    "snap_pct": (0.0, 1.0),
    "team_pass_rate": (0.0, 1.0),
    "yards_per_target": (0.0, 25.0),
    "yards_per_carry": (0.0, 12.0),
    "yac_per_reception": (0.0, 20.0),
}


def _is_trustworthy(key: str, value, m: dict) -> bool:
    """
    Whether a metric should be shown at all, given how much volume it was
    computed over and whether the result is physically plausible.

    Attempt totals are reconstructed from the per-game rate times games
    played — PlayerMetrics stores rates, not raw season totals, so this is
    the only denominator available. It's approximate, which is fine: the
    distinction being drawn is "2 carries" versus "200," not 19 versus 21.
    """
    bounds = _METRIC_SANITY_BOUNDS.get(key)
    if bounds is not None:
        try:
            if not (bounds[0] <= float(value) <= bounds[1]):
                return False
        except (TypeError, ValueError):
            return False

    games = m.get("games_played") or 0
    if not games:
        return True

    if key in _RECEIVING_RATE_FIELDS:
        approx = (m.get("targets_per_game") or 0) * games
        return approx >= _MIN_TARGETS_FOR_RECEIVING_RATES
    if key in _RUSHING_RATE_FIELDS:
        approx = (m.get("carries_per_game") or 0) * games
        return approx >= _MIN_CARRIES_FOR_RUSHING_RATES
    return True

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
        if m.get(key) is not None
        and not (key in _SUPPRESS_ZERO and m[key] == 0)
        # Drops rates computed over a denominator too small to mean anything,
        # plus anything physically impossible. See _is_trustworthy.
        and _is_trustworthy(key, m[key], m)
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
    # NOTE: the `is_rookie_or_second_year` column used to be read here. It was
    # removed rather than fixed: no ingestion script has ever written it (it's
    # declared on PlayerMetrics, read in three places, set by nothing — 0 of
    # 182 rows populated), so this branch was dead code that silently never
    # fired. Experience is now derived from DraftProfile.draft_year, which IS
    # populated, in _format_metrics_section — where the draft profile is in
    # scope. Leaving both would give the same fact two sources of truth and
    # print it twice the day fetch_metrics.py starts populating the column.

    if not parts and not trend_parts:
        return None

    line = ", ".join(parts)
    if trend_parts:
        line += (" | " if line else "") + ", ".join(trend_parts)
    return line


# Players this many NFL seasons in or fewer keep their draft capital attached
# to their stat line. Beyond that, where a player was drafted stops carrying
# predictive weight relative to what he's actually done on the field, and the
# line is just noise.
_DRAFT_CAPITAL_EXPERIENCE_LIMIT = 2

_ORDINALS = {0: "rookie", 1: "2nd", 2: "3rd"}


def _infer_current_season(
    player_metrics: dict[int, dict],
    draft_profiles: dict[int, dict],
) -> int | None:
    """
    The season being drafted for, inferred from the data itself rather than
    the system clock.

    Two independent signals, and the later one wins: PlayerMetrics holds the
    *prior* completed season (so +1), and the newest DraftProfile draft_year
    is the class that just came in (so as-is). During draft season both agree.
    Deriving it this way means a stale DB reports stale-but-consistent
    experience numbers instead of silently aging every player by a year every
    January, and a data refresh self-corrects with no code change.

    Returns None when neither table has anything to go on, in which case
    experience is simply not rendered — better than a guess that quietly
    mislabels a veteran as a rookie.
    """
    candidates: list[int] = []
    seasons = [m.get("season") for m in player_metrics.values() if m.get("season")]
    if seasons:
        candidates.append(max(seasons) + 1)
    years = [d.get("draft_year") for d in draft_profiles.values() if d.get("draft_year")]
    if years:
        candidates.append(max(years))
    return max(candidates) if candidates else None


def _experience_context(dp: dict | None, current_season: int | None) -> str:
    """
    A short "2nd NFL season, 1st-round pick (#19 overall)" clause for players
    early enough in their careers for it to matter, or "" otherwise.

    This exists because of a confirmed live miss: a second-year WR with
    first-round draft capital was repeatedly passed over for a much older
    veteran whose prior-season per-game average was higher. Every fact needed
    to see the difference was in the database and none of it reached the
    prompt — _format_metrics_section rendered draft capital only as a
    *fallback* for players with no stats at all, so the moment a rookie
    completed a season his pedigree disappeared and he read as an anonymous
    veteran with mediocre numbers.
    """
    if not dp or current_season is None:
        return ""
    draft_year = dp.get("draft_year")
    if draft_year is None:
        return ""

    experience = current_season - draft_year
    if experience < 0 or experience > _DRAFT_CAPITAL_EXPERIENCE_LIMIT:
        return ""

    label = _ORDINALS.get(experience)
    bits = [label if experience == 0 else f"{label} NFL season"]

    rnd, pick = dp.get("draft_round"), dp.get("draft_pick")
    if rnd is not None:
        capital = {1: "1st", 2: "2nd", 3: "3rd"}.get(rnd, f"{rnd}th") + "-round pick"
        if pick is not None:
            capital += f" (#{pick} overall)"
        bits.append(capital)
    elif draft_year is not None:
        bits.append(f"{draft_year} draft class")

    return ", ".join(bits)


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
    current_season = _infer_current_season(player_metrics, draft_profiles)
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
            # Draft capital rides along with the stat line for players still
            # early in their careers, instead of only standing in for missing
            # stats. See _experience_context: without this a second-year
            # first-rounder's numbers are presented as though they came from
            # an established veteran, which is a materially different claim.
            context = _experience_context(draft_profiles.get(p["id"]), current_season)
            bracket = f"{m.get('season')} season"
            if context:
                bracket += f" | {context}"
            sections.append(f"- {p['name']} ({p['position']}) [{bracket}]: {line}")

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

def _board_for_prompt(ctx: RecommendationContext, gaps: dict[str, int]) -> list[dict]:
    """
    The players actually shown: the top _LISTED_PLAYERS by ADP, plus the best
    available player at every position still missing from the starting
    lineup.

    A global ADP cut does not guarantee positional coverage, and late in a
    draft it stops providing it entirely. Confirmed live at pick 128: the
    user's only unfilled starting slot was TE, five picks remained, and the
    board held FIFTEEN receivers and ZERO tight ends — the best available TE
    sat at overall position 37. The model was being asked to fill a slot from
    a list containing nothing that could fill it, so it could only recommend
    another receiver for a roster already two deep beyond requirement.

    Nothing in the prompt could express that failure either: a position
    absent from the board is indistinguishable from a position with no good
    options left. The fix belongs here rather than in a larger
    _LISTED_PLAYERS, which would pay for coverage on every position at every
    pick when what is needed is one or two players at one position,
    occasionally.

    Pulled in on TWO axes, cheapest-by-ADP and best-by-VOR, because one is
    not enough when they disagree — and at a needed position they often do.
    Confirmed live: filling the TE slot by ADP alone surfaced Pat Freiermuth
    (ADP 152, VOR -2.94, 11% target share, and his team had just added 21%
    of a team's targets) while leaving Brenton Strange invisible eleven
    picks later (VOR -0.68, 16% target share, 75% snaps, no new
    competition). The better player was off the board because he was
    slightly more expensive.
    """
    board = list(ctx.top_available[:_LISTED_PLAYERS])
    present = {p["id"] for p in board}

    def _add(player: dict | None) -> None:
        if player is not None and player["id"] not in present:
            board.append(player)
            present.add(player["id"])

    for pos in gaps:
        # FLEX is filled by an RB/WR/TE, all of which are covered by their
        # own entries — there is no "best available FLEX" to look up.
        if pos == "FLEX" or pos in _LATE_ROUND_POSITIONS:
            continue
        at_pos = [p for p in ctx.top_available if p["position"] == pos]
        if not at_pos:
            continue

        # Cheapest — what the market says you can get away with waiting for.
        _add(next((p for p in at_pos if p["id"] not in present), None))

        # Most productive — what last season says is actually best. Skipped
        # when nobody at the position has a VOR, rather than falling back to
        # ADP and silently adding the same player twice.
        scored = [
            (v, p) for p in at_pos
            if (v := _vor(p, ctx.player_metrics, ctx.replacement_ppg)) is not None
        ]
        if scored:
            _add(max(scored, key=lambda t: t[0])[1])

    return sorted(board, key=lambda p: p["adp"])


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
    # Kickers only count as a requirement if any exist to draft — see
    # _full_lineup. available_counts is the live board, so this also
    # correctly drops the requirement once the last kicker is gone.
    kickers_available = bool(ctx.available_counts.get("K", 0))
    lineup = _full_lineup(ctx.starting_lineup, kickers_available)
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
        # Note the absence of a "you're done" line here. The depth table
        # immediately below carries the real content from this point on; the
        # old "every remaining pick is depth or upside" wording ended the
        # section on a shrug for the entire back half of the draft.
        lines.append("All base starting slots filled — see roster construction below.")

    if _K_SLOTS and not kickers_available:
        # Stated rather than silently omitted: the league does start a kicker,
        # so a draft that ends without one leaves a hole the user has to fill
        # from waivers. Dropping the requirement quietly would mean nobody
        # finds out until week 1.
        lines.append(
            "NOTE: this league starts a kicker, but this app's ADP data contains no "
            "kickers at all, so one cannot be drafted here and K is excluded from the "
            "requirements above. Do NOT spend a pick trying — add a kicker from "
            "waivers after the draft instead."
        )

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

    # Everything below shows THIS list, not a raw ADP slice — see
    # _board_for_prompt for why the two differ late in a draft.
    board = _board_for_prompt(ctx, skill_gaps)
    shortlist = _shortlist(board, advised_pick, ctx.player_metrics, ctx.replacement_ppg)

    for section in (
        _format_roster_changes(board, ctx.roster_changes),
        _format_roster_depth_section(
            _compute_roster_depth(ctx.my_roster, lineup), spots_left,
            lineup_has_flex=bool(lineup.get("FLEX")),
        ),
        _format_shortlist_section(
            shortlist, advised_pick, ctx.player_metrics, ctx.replacement_ppg,
            ctx.roster_changes,
        ),
        _format_survival_section(board, horizon, picks_between),
        _format_positional_dropoff(ctx.top_available, horizon, ctx.player_metrics),
    ):
        if section:
            lines.append(section)

    # Top available players, grouped into ADP tiers (cap at 25 to keep
    # prompt tight). Tiers, not a flat 1-25 rank — see _compute_adp_tiers'
    # docstring for why: a precise ordinal table invites treating every
    # ADP decimal as meaningful, when a few-point gap is often just noise.
    lines.append("## Top Available Players (grouped into ADP tiers)")
    if len(board) > _LISTED_PLAYERS:
        lines.append(
            "This list is the top players by ADP PLUS the best available player at "
            "every position you still need to start — those may sit well past the "
            "ADP cut and would otherwise be invisible here, leaving you unable to "
            "fill a required slot from this board at all."
        )
    lines.append(
        "Tier boundaries fall at the largest ADP gaps on this particular board, so "
        "they adapt to how tightly packed it is. Within a tier, a fraction of an ADP "
        "point is noise and should not decide anything — use the Opportunity & "
        "Performance Signals / Player News sections to choose among them. Across "
        "tiers the gap is more likely to be real drop-off, and a lower tier is NOT "
        "interchangeable with a higher one: preferring a player two tiers down needs "
        "a reason you can point at, not just a better-looking stat line."
    )
    if ctx.replacement_ppg:
        repl_str = ", ".join(
            f"{pos} {v:.1f}" for pos, v in sorted(ctx.replacement_ppg.items())
        )
        lines.append(
            f"VOR = prior-season PPR ppg minus the last startable player at that "
            f"position league-wide ({repl_str}). It is the ONLY figure here that "
            f"compares players at different positions on one scale: equal ADP and "
            f"equal points per game do not mean equal value, because what replaces "
            f"them differs. VOR is backward-looking where ADP is the market's "
            f"forward view — when the two disagree sharply, say so rather than "
            f"silently trusting one. A player marked '(BELOW replacement)' scored "
            f"LESS last season than the player you can have at that position for "
            f"nothing later — taking one costs you a roster spot and gains you "
            f"nothing over waiting, so it needs a reason beyond his ADP. '--' means "
            f"no prior-season data (usually a rookie), which is unknown value, NOT "
            f"replacement-level value: an unproven player and a proven-below-"
            f"replacement player are different cases, and the second is the one the "
            f"numbers actually argue against.\n"
            f"OPP is the third figure and carries the same weight as the other "
            f"two. It is the net share of his team's volume — targets for a "
            f"receiver or tight end, carries for a back — that has CHANGED HANDS "
            f"since these numbers were recorded. 'OPP +21% tgt' means players "
            f"taking 21% of that team's targets have left, so his own share "
            f"understates what is in front of him and a below-replacement VOR may "
            f"describe a role he no longer has. 'OPP -43% tgt' means that much new "
            f"competition arrived, so his share overstates it. 'OPP +0%' means the "
            f"situation is unchanged and last season's numbers still describe it — "
            f"which is real information, not an absent signal. '--' means the "
            f"position has no single volume to compete for (QB, DST).\n"
            f"The three are not ranked. ADP is what the market expects, VOR is what "
            f"he produced, OPP is what has changed around him — and each is blind "
            f"to what the others see. Do not resolve a conflict by defaulting to "
            f"VOR because it is the most precise-looking number; a precise "
            f"measurement of a role that no longer exists is not evidence about "
            f"this season. When they disagree, say which one you are trusting and "
            f"why, in `reasoning`."
        )
    tiers = _compute_adp_tiers(board)
    for i, tier in enumerate(tiers, start=1):
        adp_lo, adp_hi = tier[0]["adp"], tier[-1]["adp"]
        lines.append(f"\nTier {i} (ADP {adp_lo:g}-{adp_hi:g}):")
        for p in tier:
            row = (
                f"  {p['rank']:<5} {p['name']:<22} {p['position']:<5} "
                f"{p['team']:<6} ADP {p['adp']:<6}"
            )
            if ctx.replacement_ppg:
                v = _vor(p, ctx.player_metrics, ctx.replacement_ppg)
                if v is None:
                    row += " VOR   --"
                else:
                    row += f" VOR {v:+.1f}"
                    # A negative VOR is easy to read as "roughly neutral"
                    # when it actually means the player produced less than
                    # the man you can have at this position for nothing
                    # later. Confirmed live: a back at VOR -0.17, with
                    # falling target and snap share, was recommended over a
                    # first-round rookie — the sign was there and did no
                    # work. Stating it costs four words and invents nothing.
                    if v < 0:
                        row += " (BELOW replacement)"
            # Value against the pick you are actually making, computed here
            # rather than left as a subtraction for the model. See _draft_value.
            row += f"  {_draft_value(p['adp'], advised_pick)}"
            # Forward-looking counterweight, on the same row as the
            # backward-looking VOR verdict. See _team_opportunity: with this
            # fifty lines away under a team heading, it lost every time.
            # Third column, always present — see _team_opportunity.
            row += "  " + (_team_opportunity(p, ctx.roster_changes) or "OPP    --")
            lines.append(row)
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
        board, ctx.player_metrics, ctx.draft_profiles
    )
    if metrics_section:
        lines.append(metrics_section)

    # Retrieved player news/analysis (best-effort; omitted if unavailable)
    retrieved = _retrieve_player_context(board)
    if retrieved:
        lines.append(retrieved)

    # Output schema
    lines += [
        "## Task",
        "Recommend the best pick for my team right now. Work through these steps in "
        "order — each one narrows the field for the next:",
        "",
        "1. SHORTLIST. Take the top tier or two on the board. Ignore ADP differences "
        "WITHIN a tier — those are noise. Do not treat a lower tier as equivalent to a "
        "higher one: tier boundaries sit at the real gaps on this board, so dropping a "
        "tier costs something and needs a reason beyond a nicer-looking stat line.",
        "2. DEPRIORITISE WHAT WILL KEEP. Every player in the Opportunity Cost section "
        "is available to draft with this pick — the groups say what happens LATER, not "
        "what is on the board now, and you must never treat a listed player as already "
        "taken. Given that, deprioritise the 'WILL LAST' group unless one of them is "
        "clearly, not marginally, better than everyone in 'TAKE NOW OR LOSE HIM': you "
        "can still get a WILL LAST player at your next turn and you cannot get the "
        "others. This step decides most picks.",
        "3. WEIGH THE COST OF WAITING AGAINST YOUR ROSTER. Use the Cost of Waiting "
        "table, Run Risk, and the Roster Construction table together. A large drop-off "
        "at a position you still need to start — or at one sitting at zero cover — "
        "with several teams ahead of you needing it too, is the strongest possible "
        "case for taking that position now. A small drop-off, or a position you are "
        "already deep at, means this pick belongs elsewhere. Say which in `strategy`.",
        "4. BREAK REMAINING TIES ON EVIDENCE. Among what survives, use Opportunity & "
        "Performance Signals (real usage, efficiency, consistency, durability) and "
        "Player News. Call out any candidate whose underlying opportunity looks "
        "stronger than their tier implies as a potential breakout. Never break a tie on "
        "name recognition or last season's box score alone. Before you commit, check "
        "your pick against the board: if another available player beats the one you "
        "chose on ADP, VOR and OPP together, you need a specific, stated reason for "
        "passing on him — and 'durability' does not qualify on its own, because ADP "
        "already accounts for it (RULE 6). If you cannot name that reason, take the "
        "better player. Where they split, say which signal decided it.",
        "5. CHECK LEGALITY LAST. Confirm the pick doesn't leave you unable to field a "
        "legal starting lineup — that only overrides steps 2-4 when the roster shape "
        "section shows an URGENT line. Filling every base slot is not a finished "
        "roster; do not call a position secure just because its minimum is met.",
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
            "considered": ["<one line per MUST EVALUATE player: 'Name — taken' or 'Name — passed because <specific reason>'. One entry for every id in that section, no exceptions.>"],
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
        "are discarded unread and only cost you output budget. The alternatives must "
        "cover at least TWO different positions whenever the board allows: a list of "
        "five players at one position is not a set of options, it is one option "
        "restated, and it hides the cross-position calls that decide most picks. "
        "Every "
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

        "RULE 0 — NEVER RECOMMEND A PLAYER WHO CANNOT PLAY. This overrides every "
        "other rule here. If the Player News & Analysis section says a player is on "
        "IR or injured reserve, out for the season, on PUP, suspended, or has already "
        "been ruled out, he is not a pick — he is a wasted roster spot, whatever his "
        "ADP, VOR or production says. Those numbers all describe a season he will not "
        "play. Do not soften this into 'a durability concern' or 'a risk worth taking': "
        "an unavailable player scores zero. The only exception is the very end of the "
        "draft when the alternative is a player you would cut anyway, and even then "
        "say plainly in your reasoning that he is a stash who will not play this "
        "season. A note about a knee issue, a missed practice or a questionable tag is "
        "different — that IS ordinary durability risk, and RULE 6 governs it.\n\n"

        "RULE 1 — OPPORTUNITY COST DECIDES CLOSE CALLS. A draft pick's real cost is "
        "the best player you won't get back. When two players are close, take the one "
        "who will not survive to your next turn and plan to take the other one later. "
        "The prompt tells you explicitly which players fall in each bucket and what "
        "the drop-off at each position is if you wait; use those numbers rather than "
        "estimating. Never spend a pick on a player labeled 'WILL LAST' unless he is "
        "clearly — not marginally — better than everything about to disappear.\n\n"

        "RULE 1a — EVERY LISTED PLAYER IS AVAILABLE. Every name anywhere in this "
        "prompt is on the board and can be drafted with this pick; already-drafted "
        "players are removed before you see it. The Opportunity Cost groups forecast "
        "what will be true at your NEXT turn — they never mean a player is unavailable "
        "now. Reasoning such as 'the best option once X and Y are gone' is a serious "
        "error when X and Y are listed: if they are the better players, one of THEM is "
        "the pick. Recommend around a player only if he is genuinely absent from the "
        "board above.\n\n"

        "RULE 2 — VALUE OVER NEED, UNTIL LEGALITY IS AT RISK. Draft the best player "
        "relative to positional replacement level, not the next recognizable name and "
        "not whichever slot happens to be empty. A marginal 'need' edge is rarely "
        "worth passing on a better player, ESPECIALLY in the early rounds. Two things "
        "outrank that: an URGENT line in the roster shape section, meaning you are "
        "running out of picks to field a legal lineup at all, and the depth reasoning "
        "in RULE 3 once the base slots are filled.\n\n"

        "RULE 3 — A LEGAL LINEUP IS NOT A COMPLETE ROSTER. Filling every starting "
        "slot is the floor, not the goal. The Roster Construction table shows how many "
        "of each position you can start in a single week against how many you hold; a "
        "position at zero cover has no answer to an injury, a bye, or a bust, and you "
        "will need one over a full season. Once the base slots are filled, judge each "
        "remaining pick by how many points it protects or adds, using the Cost of "
        "Waiting figures — a position whose replacement level falls off steeply "
        "deserves a bench spot well before a position where the 6th-best available is "
        "nearly as good as the 3rd. Adding a fourth bench player at a position you are "
        "already deep at, while another position sits at zero cover, is the most common "
        "way to lose a draft you were winning. Never describe a position as secure "
        "merely because its base requirement is met.\n\n"

        "RULE 4 — THREE SIGNALS, NONE OF THEM SENIOR. Every board row carries "
        "ADP, VOR and OPP, and they answer different questions: what the market "
        "expects, what he produced, and what has changed around him. Each is blind "
        "to what the others see, so none of them outranks the rest.\n"
        "VOR exists because equal points per game at different positions are not "
        "equal value — what replaces each player differs. In this data a 15.2 ppg "
        "back is +4.1 over the back who would replace him while a 14.7 ppg receiver "
        "is only +2.8 over his, so use VOR whenever you compare ACROSS positions. "
        "Note also that an ADP-sorted board surfaces more receivers than backs, "
        "which is a property of the list and not a judgement about them.\n"
        "The failure mode to avoid is treating VOR as the tiebreaker because it is "
        "the most precise-looking figure. It is a measurement of LAST season. When "
        "OPP shows the volume around a player has moved, VOR is measuring a role "
        "that may no longer exist — and a precise measurement of the wrong thing is "
        "not better evidence than an imprecise measurement of the right one. A "
        "below-replacement VOR beside a large positive OPP is the single most "
        "common shape of an underpriced player, and a healthy VOR beside a large "
        "negative OPP is the most common shape of a trap.\n"
        "Two further limits on VOR: it only measures a player's value AS A STARTER, "
        "so once a position's starting slots are full the next player there is "
        "insurance and worth a fraction of it — a high VOR never justifies a fourth "
        "player at a position that starts one. And near the replacement line it "
        "exaggerates: -0.1 against -3.2 looks decisive but is three points per "
        "game, so do not treat small VOR gaps between marginal players as settling "
        "anything on their own.\n"
        "When the three disagree, name in `reasoning` which you are trusting and "
        "why. Do not silently average them.\n\n"

        "RULE 5 — ADP IS A PRIOR, NOT A RANKING. The board is grouped into ADP tiers. "
        "Within a tier, ADP differences are noise and must not decide anything; break "
        "those ties on Opportunity & Performance Signals (usage, efficiency, "
        "consistency) and Player News. Between tiers, the gap is likely real drop-off "
        "and deserves weight. A player whose underlying opportunity — targets, "
        "carries, red zone touches, snap share — outstrips his tier is a breakout "
        "candidate; say so explicitly rather than defaulting to the safest name.\n\n"

        "RULE 6 — SAMPLES AND DURABILITY, WITHOUT DOUBLE-COUNTING. A per-game average "
        "is only as good as the sample behind it. A line flagged '[SMALL SAMPLE — "
        "weigh this average with caution]' covers a shortened, often injury-affected "
        "season, and its SCORING AVERAGE is not directly comparable to a healthy "
        "player's full season. The caveat applies to that average, not to the whole "
        "line: usage rates on the same line — targets per game, carries per game, "
        "target share, RACR, catch rate — describe the role he held while playing and "
        "stabilise far faster than points do.\n"
        "Weeks on the injury report and games missed are real risk. But ADP ALREADY "
        "PRICES KNOWN INJURY HISTORY: a player the market ranks highly despite a lost "
        "season is ranked there BECAUSE the market weighed it. Applying your own "
        "discount on top of that charges him twice for the same fact. So: if a flagged "
        "player has BOTH the better ADP and the higher VOR, durability is NOT "
        "sufficient reason to pass on him — take him and state the risk in your "
        "reasoning. Only let durability decide when the players are otherwise close, "
        "and never claim a healthier player's efficiency 'outpaces' a flagged player "
        "whose efficiency figures are plainly better.\n\n"

        "RULE 7 — MISSING DATA IS NOT BAD DATA. Players showing 'No retrieved data' or "
        "'No prior-season metrics' have no prior NFL season to compute from. That is a "
        "coverage gap in this app, never evidence against the player, and must not be "
        "the reason you pass on someone. Rookies frequently show draft capital "
        "(round/pick, college) instead — treat that as genuine signal: early draft "
        "capital is among the strongest predictors of a rookie's eventual role, and a "
        "Day 1-2 pick in a favorable offense is a legitimate breakout call with zero "
        "NFL snaps.\n\n"

        "RULE 8 — YOUNG PLAYERS ARE ASCENDING; TREAT THEIR NUMBERS AS A FLOOR. Some "
        "stat lines are tagged with the player's NFL season and draft capital, e.g. "
        "'2nd NFL season, 1st-round pick (#19 overall)'. That tag changes what the "
        "numbers mean. A first- or second-year player's per-game average was produced "
        "while he was splitting snaps, learning the offense, and behind incumbents; it "
        "is a floor he is likely to beat, not a projection. An established veteran's "
        "average is his true talent and, past his prime, an optimistic one. So do NOT "
        "compare the two as like-for-like evidence: a modest per-game edge for an older "
        "player over a high-draft-capital ascending one is weak grounds for preferring "
        "the veteran, particularly when the younger player also carries the better ADP "
        "— the market is pricing in growth you can see the reason for right in the "
        "tag.\n\n"

        "RULE 9 — DST AND K ARE END-OF-DRAFT ROSTER TAXES. This app has no matchup, "
        "scheme, or opponent-strength data for any position, so nothing grounds a "
        "claim that one defense or kicker is better than another; ADP order is the "
        "only signal you have for them. Never invent outside knowledge about which "
        "real defense or kicker is 'good.' Do not recommend either position until the "
        "prompt says you are inside the reserved final rounds — a DST or K taken "
        "earlier costs you a skill player for a position that will still be freely "
        "available at the end.\n\n"

        "RULE 10 — STAY INSIDE THE EVIDENCE. Every factual claim in your reasoning must "
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

# ---------------------------------------------------------------------------
# Dominated-pick guard
#
# "Dominated" in the decision-theory sense: player X dominates Y if X is at
# least as good on every axis and better on at least one. Here, restricted to
# ADP and VOR, and to players at the SAME position.
#
# Same-position only, deliberately. Across positions, taking a "worse" player
# is routinely correct — you need a tight end and the dominating player is a
# receiver you're already deep at — so a cross-position check fires constantly
# and degrades into the same noise the first roster-imbalance flag did. Within
# a position, roster need cannot explain the gap: if a receiver has both the
# better market price and the higher value over replacement, and you took the
# other receiver, that wants a reason.
#
# This is what happened live at pick 17: Rashee Rice (ADP 11.5, VOR +6.9) sat
# on the board while the model took George Pickens (ADP 21.3, VOR +5.3) and
# claimed Pickens' efficiency was superior, which was false on every metric
# shown.
#
# ADVISORY ONLY. It never blocks, never rewrites the pick, and never sends
# anything back to the model — it appends an alert the user reads. Legitimate
# reasons to take a dominated player exist (injury news in the retrieved
# section, a bye-week conflict this app cannot see, plain disagreement with
# last season's numbers), and the person drafting is better placed to judge
# them than a two-variable rule.
# ---------------------------------------------------------------------------

# Margins, so trivial differences stay quiet. A player 1 pick earlier with
# 0.2 more ppg of value is not meaningfully better, and an alert that fires
# on noise gets ignored — including on the occasions it is right.
_DOMINANCE_ADP_MARGIN = 2.0
_DOMINANCE_VOR_MARGIN = 0.5


def _mentions_player(text: str, name: str) -> bool:
    """
    Whether `text` names this player, matched on whole-word boundaries.

    A plain substring test looks correct and is a latent false-negative: with
    a suffixed pair like "Michael Pittman" and "Michael Pittman Jr.", a
    verdict about the junior counts as engagement with the senior, and the
    dominance alert is silently suppressed for a player nobody considered.
    The current ADP feed happens to contain no such pair — checked, zero
    collisions — but that is a property of today's data, not of the code, and
    one refresh could introduce one.

    Anchoring on word boundaries makes "Pittman Jr." stop matching "Pittman"
    unless the exact name appears. Punctuation in names ("A.J. Brown",
    "De'Von Achane") is escaped rather than special.
    """
    if not text or not name:
        return False
    # Word boundaries alone are insufficient: "Michael Pittman Jr." genuinely
    # contains "Michael Pittman" followed by a space, so the boundary test
    # passes and the two players collapse. The trailing lookahead rejects a
    # match immediately followed by a generational suffix, which is the only
    # way this collision arises in practice — and the way Sleeper's own
    # player data differs from the ADP feed (see sync_sleeper_ids.py, where
    # suffix handling was already a source of 16 unmatched players).
    return re.search(
        rf"(?<!\w){re.escape(name)}(?!\w)(?!\s+(?:Jr|Sr|II|III|IV|V)\b\.?)",
        text,
        re.IGNORECASE,
    ) is not None


def _find_dominating_player(
    pick: PickSuggestion,
    ctx: RecommendationContext,
) -> dict | None:
    """
    An available player at the same position with both a materially better
    ADP and a materially higher VOR than the recommended one, or None.

    Scoped to the players actually shown to the model (_LISTED_PLAYERS). A
    complaint about someone it was never given would be unfair and, worse,
    unactionable — that would be a symptom of the board being too short,
    which is a different problem with a different fix.

    Returns None whenever either side lacks a VOR: a rookie with no prior
    season has unknown value, not low value, and this guard must never imply
    otherwise.
    """
    if not ctx.replacement_ppg:
        return None
    mine = _vor({"id": pick.player_id, "position": pick.position},
                ctx.player_metrics, ctx.replacement_ppg)
    if mine is None:
        return None

    best: dict | None = None
    best_vor = mine + _DOMINANCE_VOR_MARGIN
    for p in ctx.top_available[:_LISTED_PLAYERS]:
        if p["id"] == pick.player_id or p["position"] != pick.position:
            continue
        if p["adp"] > pick.adp - _DOMINANCE_ADP_MARGIN:
            continue
        v = _vor(p, ctx.player_metrics, ctx.replacement_ppg)
        if v is not None and v >= best_vor:
            best, best_vor = p, v
    return best


def _dominance_alert(
    pick: PickSuggestion,
    ctx: RecommendationContext,
    engaged_text: str = "",
) -> str | None:
    """
    The user-facing wording for _find_dominating_player, or None.

    Suppressed when `engaged_text` — the model's own verdicts, reasoning and
    alternatives — already names the dominating player. That distinction is
    the whole point of the guard: taking a dominated player after weighing
    him is a judgement call this rule has no business overriding, while
    taking one without ever mentioning him is the omission failure that
    produced Rice and A.J. Brown.

    It also fixes the rule's specificity problem. Against a real board, 20 of
    the top 25 players are dominated by someone at their position, so a
    check on the pick alone fires on almost any non-leader; tightening the
    margins far enough to quieten it also loses the case it was built for.
    Filtering on "was he considered" targets the actual defect instead of
    guessing at thresholds.
    """
    other = _find_dominating_player(pick, ctx)
    if other is None:
        return None
    if _mentions_player(engaged_text, other["name"]):
        return None
    mine = _vor({"id": pick.player_id, "position": pick.position},
                ctx.player_metrics, ctx.replacement_ppg)
    theirs = _vor(other, ctx.player_metrics, ctx.replacement_ppg)
    return (
        f"Check this: {other['name']} ({other['position']}, ADP {other['adp']:g}, "
        f"VOR {theirs:+.1f}) has both a better ADP and a higher VOR than the "
        f"recommended {pick.player_name} (ADP {pick.adp:g}, VOR {mine:+.1f}), and "
        f"plays the same position — so roster need does not explain the gap. There "
        f"may still be a good reason (injury news, a bye conflict this app cannot "
        f"see); this is a prompt to check, not a verdict."
    )


def _clean_text(value) -> str:
    """
    Free-text field from Claude's JSON, or "" if it isn't a string.

    Deliberately not `str(value)`: coercing a dict or list would put a Python
    repr like "{'unexpected': 'object'}" straight into the UI. These fields are
    optional, so dropping a malformed one is strictly better than showing it.
    """
    return value.strip() if isinstance(value, str) else ""


def _extract_complete_object(text: str, key: str) -> dict | None:
    """
    Parses `"<key>": { ... }` out of a partially-received JSON document, or
    returns None while it's still arriving.

    Streaming exists to show the pick before the rest of the response
    finishes generating, and the pick is a nested object roughly a fifth of
    the way into a document that takes twenty seconds to produce. So the
    stream has to answer "is this one object closed yet" on text that is
    invalid JSON everywhere else — `json.loads` can't help until the very
    end, which is the moment streaming is trying to beat.

    Brace-matching rather than a JSON parser, and string-aware: a brace
    inside a reasoning string ("he's a 'bell-cow' {sic}") would otherwise
    close the object early and yield a truncated pick. Escapes are honoured
    so a literal quote in the prose can't drop us out of string context.

    Returns None on anything malformed rather than raising — a
    half-delivered object is the normal case here, not an error.
    """
    marker = f'"{key}"'
    at = text.find(marker)
    if at == -1:
        return None
    start = text.find("{", at + len(marker))
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


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


def _as_player_id(value) -> int | None:
    """Claude occasionally returns player_id as a JSON string ("3" not 3);
    rejecting that would throw away a good recommendation over a type quirk."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pick_from(d: dict, ctx: RecommendationContext) -> PickSuggestion | None:
    """
    Builds a PickSuggestion from one of Claude's objects, taking every
    factual field from the canonical player row and only the free text from
    the model.

    Module-level so the streaming path validates identically to the batch
    one. A partially-received response is exactly where a laxer check would
    be tempting, and exactly where it would be worst: the pick rendered
    first from a stream is the one the user acts on.

    Returns None for an id that isn't in top_available — unknown or already
    drafted. Validating player_id while still rendering the model's own
    player_name once displayed a drafted player under an available id, so
    the id is the only field trusted, and it's the only one verified.
    """
    canonical = {p["id"]: p for p in ctx.top_available}.get(_as_player_id(d.get("player_id")))
    if canonical is None:
        return None
    return PickSuggestion(
        player_id=canonical["id"],
        player_name=canonical["name"],
        position=canonical["position"],
        adp=canonical["adp"],
        reasoning=str(d.get("reasoning", "")),
        tradeoff=_clean_text(d.get("tradeoff")),
        survival=_survival_code(canonical["adp"], ctx.my_following_pick_number),
    )


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

    # Every *factual* field on a suggestion comes from the canonical player
    # row, never from Claude's JSON — see _pick_from, which both this and the
    # streaming path go through.
    def _pick(d: dict) -> PickSuggestion | None:
        return _pick_from(d, ctx)

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
    considered = [str(x) for x in data.get("considered", []) if x]

    # Appended last so it reads as the final word on the pick. Computed here
    # rather than asked of the model, because the model is the thing being
    # checked — see _find_dominating_player.
    #
    # Everything the model wrote counts as engagement: a player named in the
    # verdicts, in the reasoning, or offered as an alternative was considered,
    # and the guard stays quiet about him.
    engaged = " ".join([
        *considered,
        recommendation.reasoning,
        *(a.player_name for a in alternatives),
        *(a.reasoning for a in alternatives),
        *(a.tradeoff for a in alternatives),
    ])
    dominance = _dominance_alert(recommendation, ctx, engaged)
    if dominance:
        logger.info(
            "Dominated pick: %s was recommended over a same-position player with "
            "better ADP and higher VOR.", recommendation.player_name,
        )
        alerts.append(dominance)

    # Logged, not enforced. A missing verdict is worth knowing about — it's
    # the exact failure this section exists to catch — but rejecting an
    # otherwise-valid recommendation over a missing explanation would trade a
    # good pick for a bookkeeping complaint on draft day.
    if not considered:
        logger.info("Claude returned no `considered` verdicts for the must-evaluate shortlist.")

    positions = {a.position for a in alternatives}
    if len(alternatives) >= 3 and len(positions) < 2:
        logger.info(
            "All %d alternatives are %s — the prompt asks for at least two positions.",
            len(alternatives), positions.pop() if positions else "?",
        )

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
        considered=considered,
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
            survival=_survival_code(top["adp"], ctx.my_following_pick_number),
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
                survival=_survival_code(p["adp"], ctx.my_following_pick_number),
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

    async def recommend_stream(self, ctx: RecommendationContext):
        """
        Same recommendation as recommend(), yielded in two stages:

            ("pick",     PickSuggestion)      as soon as it has been written
            ("complete", RecommendationResult) when the whole response lands

        Total time is unchanged — this does not make the model faster. What
        changes is when the answer becomes visible. Measured on a live
        board: 1,664 output tokens at ~75 tok/sec is ~22 seconds, and
        generation is sequential, so nothing appears until it finishes. The
        recommendation object is about a fifth of the way in, so the pick
        itself exists after roughly four seconds and then sits there while
        alternatives, verdicts and alerts are still being written.

        Consumers should render the pick on the first event and fill in the
        rest on the second. Both are the same underlying response; the first
        is not a guess or a cheaper model.

        Falls back exactly like recommend(): any failure yields a single
        ("complete", <ADP fallback>) rather than raising, because a draft
        clock does not care why the API is unhappy.
        """
        if self._client is None:
            yield "complete", _fallback(ctx, self._model)
            return

        prompt = await asyncio.to_thread(_build_prompt, ctx)
        buffer = ""
        sent_pick = False

        try:
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=_MAX_RESPONSE_TOKENS,
                temperature=_TEMPERATURE,
                system=_build_system_prompt(ctx.scoring_format),
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "{"},
                ],
            ) as stream:
                async for chunk in stream.text_stream:
                    buffer += chunk
                    if sent_pick:
                        continue
                    # Cheap guard: don't attempt a scan until the object it
                    # would look for has plausibly closed.
                    if "alternatives" not in buffer and buffer.count("}") < 1:
                        continue
                    rec = _extract_complete_object(_restore_prefill(buffer), "recommendation")
                    if rec:
                        pick = _pick_from(rec, ctx)
                        if pick is not None:
                            sent_pick = True
                            yield "pick", pick

            result = _parse_response(_restore_prefill(buffer), ctx)
            if result is None:
                logger.warning("Falling back to ADP — could not parse streamed response.")
                result = _fallback(ctx, self._model)
            yield "complete", result

        except anthropic.APIError as e:
            logger.error("Anthropic API error during stream: %s", e)
            yield "complete", _fallback(ctx, self._model)
        except Exception:
            logger.exception("Unexpected error during streamed recommendation.")
            yield "complete", _fallback(ctx, self._model)

    async def recommend(self, ctx: RecommendationContext) -> RecommendationResult:
        """
        Generates a pick recommendation for the current draft state.
        Falls back to top-ADP logic if the API call fails.
        """
        if self._client is None:
            return _fallback(ctx, self._model)

        # _build_prompt is sync on purpose (it's also used by the CLI
        # preview in main() below) and it performs the ChromaDB lookup, so
        # it runs in a worker thread rather than on the event loop. That
        # lookup is now a single metadata `get` costing ~2 ms rather than one
        # embedding round trip per player, but it is still blocking I/O and
        # the loop is also serving WebSocket pushes and the Sleeper poll.
        # Thread safety: it touches only SQLite-free plain dicts, the
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

    from backend.db import draft_profile_repo, metrics_repo
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

        # Metrics and draft profiles, exactly as recommendations.py::
        # _build_context loads them. These were missing entirely, which made
        # this CLI actively misleading: with player_metrics empty, EVERY
        # player rendered as "No prior-season metrics on file (likely a
        # rookie...)" — so the one tool whose whole purpose is showing the
        # real prompt was the only place that claimed the app had no data on
        # anyone. Flattened inline rather than importing _metrics_dict from
        # recommendations.py, which imports this module (circular).
        ids = [p["id"] for p in top_available]
        player_metrics = {
            pid: {c: getattr(m, c) for c in type(m).model_fields}
            for pid, m in metrics_repo.get_metrics_bulk(session, ids).items()
        }
        draft_profiles = {
            pid: {c: getattr(d, c) for c in type(d).model_fields}
            for pid, d in draft_profile_repo.get_draft_profiles_bulk(session, ids).items()
        }

        # Replacement levels over the whole pool, same as the live path — the
        # preview is only useful if the VOR column it shows is the real one.
        from sqlmodel import select as _select

        from backend.db.models import Player as _Player
        from backend.db.models import PlayerMetrics as _PM

        pool: dict[str, list[float]] = {}
        for pos, ppg in session.exec(
            _select(_Player.position, _PM.fantasy_points_avg)
            .join(_PM, _PM.player_id == _Player.id)
            .where(_PM.fantasy_points_avg.is_not(None))
        ):
            pool.setdefault(pos, []).append(ppg)
        replacement_ppg = compute_replacement_levels(
            pool, 12, {"QB": 1, "RB": 2.5, "WR": 2.5, "TE": 1}
        )

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
        player_metrics=player_metrics,
        draft_profiles=draft_profiles,
        replacement_ppg=replacement_ppg,
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
