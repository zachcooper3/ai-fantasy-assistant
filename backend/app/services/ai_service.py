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

import json
import logging
import os
from dataclasses import dataclass, field

import anthropic

logger = logging.getLogger(__name__)

# Use Haiku on the clock (fast, cheap); override with CLAUDE_MODEL env var for richer analysis
_DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PickSuggestion:
    player_id: int
    player_name: str
    position: str
    adp: float
    reasoning: str


@dataclass
class RecommendationResult:
    recommendation: PickSuggestion
    alternatives: list[PickSuggestion]
    alerts: list[str]
    model: str


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
    if lineup is None:
        lineup = _STANDARD_STARTING_LINEUP

    have: dict[str, int] = {}
    for p in my_roster:
        have[p["position"]] = have.get(p["position"], 0) + 1

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

    sections: list[str] = []
    hits = 0
    checked = 0
    for p in top_available[:_MAX_CONTEXT_PLAYERS]:
        sleeper_id = p.get("sleeper_id")
        if not sleeper_id:
            continue
        checked += 1

        try:
            results = _query_player_chunks(vector_query, sleeper_id, p["name"])
        except Exception as e:
            logger.warning(f"Vector query failed for {p['name']}: {e}")
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


def _format_metrics_section(top_available: list[dict], player_metrics: dict[int, dict]) -> str:
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
    """
    sections: list[str] = []
    for p in top_available[:_MAX_CONTEXT_PLAYERS]:
        m = player_metrics.get(p["id"])
        if m is None:
            if p["position"] in ("DST", "K"):
                sections.append(
                    f"- {p['name']} ({p['position']}): Not modeled by this metrics table "
                    f"(built for individual offensive usage) — evaluate by matchup, "
                    f"scheme, and ADP instead."
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
    lines += [
        "## Draft State",
        f"- Overall pick: #{ctx.pick_number} (Round {ctx.round_number} of {ctx.total_rounds})",
        f"- My draft slot: {ctx.my_slot} of {ctx.league_size}",
        f"- {turn_info}",
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
    gaps = _compute_roster_gaps(ctx.my_roster, ctx.starting_lineup)
    rounds_remaining = max(0, ctx.total_rounds - ctx.round_number)
    lineup_str = ", ".join(f"{pos} x{n}" for pos, n in sorted(gaps.items()))
    # Built from ctx.starting_lineup (not hardcoded) so this line reflects
    # this league's actual configured roster, e.g. via /api/draft/session.
    _slot_order = ["QB", "RB", "WR", "TE", "FLEX", "DST"]
    roster_str = ", ".join(
        f"{ctx.starting_lineup[pos]} {pos}"
        for pos in _slot_order
        if ctx.starting_lineup.get(pos, 0) > 0
    )
    if gaps:
        gap_total = sum(gaps.values())
        lines.append(f"## League's Configured Starting Roster Shape ({roster_str})")
        # Informational, not imperative — this is roster awareness, not an
        # instruction to fill these before a better value/upside pick. See
        # the Task section and _SYSTEM_PROMPT below for the actual
        # prioritization: value/upside first, gap-filling only escalates to
        # a hard priority once the URGENT line below appears.
        lines.append(f"Open slots (not yet on your roster): {lineup_str}")
        if gap_total >= rounds_remaining and rounds_remaining > 0:
            lines.append(
                f"URGENT: {gap_total} required starting slot(s) still open with only "
                f"{rounds_remaining} round(s) left — prioritize filling these over upside/"
                f"value picks at positions you've already covered."
            )
        lines.append("")
    else:
        lines += [
            f"## League's Configured Starting Roster Shape ({roster_str})",
            "All required starting slots filled.",
            "",
        ]

    # Top available players (cap at 25 to keep prompt tight)
    lines.append("## Top Available Players (by ADP)")
    lines.append(f"{'Rank':<5} {'Player':<22} {'Pos':<5} {'Team':<6} {'ADP'}")
    for p in ctx.top_available[:25]:
        lines.append(
            f"{p['rank']:<5} {p['name']:<22} {p['position']:<5} {p['team']:<6} {p['adp']}"
        )
    lines.append("")

    # Positional availability (scarcity context)
    counts = ctx.available_counts
    lines += [
        "## Positional Availability (undrafted players remaining)",
        (
            f"QB: {counts.get('QB', 0)} | "
            f"RB: {counts.get('RB', 0)} | "
            f"WR: {counts.get('WR', 0)} | "
            f"TE: {counts.get('TE', 0)} | "
            f"DST: {counts.get('DST', 0)}"
        ),
        "",
    ]

    # Opponent rosters (position counts only — keeps prompt compact)
    if ctx.opponent_position_counts:
        lines.append("## Opponent Rosters (position counts per team)")
        for slot, pos_counts in sorted(ctx.opponent_position_counts.items()):
            if not pos_counts:
                lines.append(f"  Slot {slot:>2}: [empty]")
            else:
                pos_str = ", ".join(f"{pos}: {n}" for pos, n in sorted(pos_counts.items()))
                lines.append(f"  Slot {slot:>2}: {pos_str}")
        lines.append("")

    # Opportunity/performance signals (best-effort; omitted if no metrics at all)
    metrics_section = _format_metrics_section(ctx.top_available, ctx.player_metrics)
    if metrics_section:
        lines.append(metrics_section)

    # Retrieved player news/analysis (best-effort; omitted if unavailable)
    retrieved = _retrieve_player_context(ctx.top_available)
    if retrieved:
        lines.append(retrieved)

    # Output schema
    lines += [
        "## Task",
        "Recommend the best pick for my team right now.",
        "Weigh ADP as one input among several, not the deciding factor. Prioritize "
        "long-term value and upside: use the Opportunity & Performance Signals section "
        "to judge real usage and efficiency, and call out any candidate whose underlying "
        "opportunity looks stronger than their ADP suggests as a potential breakout. Use "
        "the roster shape section above for awareness of which starting slots are still "
        "open, but don't let that alone override a clearly better value/upside pick — the "
        "exception is when that section includes an URGENT line, at which point rounds "
        "are genuinely running out to complete a legal starting lineup, and filling the "
        "open slot(s) takes priority.",
        "",
        "Respond with ONLY valid JSON — no markdown, no commentary:",
        json.dumps({
            "recommendation": {
                "player_id": "<int from table above>",
                "player_name": "<string>",
                "position": "<string>",
                "adp": "<float>",
                "reasoning": "<1-2 sentences explaining why this is the best pick>",
            },
            "alternatives": [
                {
                    "player_id": "<int>",
                    "player_name": "<string>",
                    "position": "<string>",
                    "adp": "<float>",
                    "reasoning": "<brief>",
                }
            ],
            "alerts": ["<any scarcity warnings, handcuff notes, tier drop-off flags, or breakout/value calls>"],
        }, indent=2),
    ]

    return "\n".join(lines)


_SYSTEM_PROMPT = (
    "You are an expert fantasy football draft advisor for a PPR Sleeper league. You give "
    "concise, data-driven recommendations and draw on standard fantasy football draft "
    "strategy: value-based drafting (the best player relative to positional replacement "
    "level, not just the next recognizable name), taking on more boom/bust upside as the "
    "draft moves past the first few rounds, and not overpaying for reputation or last "
    "season's box score over a player's actual current opportunity. "
    "ADP is one input, not the deciding one — weigh it alongside the Opportunity & "
    "Performance Signals section (real usage, efficiency, and week-to-week consistency "
    "from last season) and the Player News & Analysis section (current role and injury "
    "context). A player whose underlying opportunity — targets, carries, red zone usage, "
    "snap share — looks stronger than their ADP suggests is a potential breakout; call "
    "that out explicitly in your reasoning or alerts rather than defaulting to the "
    "safest name-brand pick. "
    "Favor long-term value and upside over reflexively filling a roster gap — a marginal "
    "'need' edge is rarely worth passing on the better player, especially early in a "
    "draft. Use the roster shape section to stay aware of which starting slots remain "
    "open, but only let that override a better value/upside pick once that section "
    "includes an URGENT line — at that point rounds are genuinely running out to "
    "complete a legal starting lineup, and filling it takes priority. "
    "Some players — especially rookies — will show 'No retrieved data' or 'No "
    "prior-season metrics' in the sections above. That reflects a gap in available "
    "statistics (no prior NFL season to compute from), not a judgment about the player. "
    "Do not treat a missing data section as a reason to avoid a player; judge them on "
    "ADP/consensus value and whatever other context you do have. "
    "You always respond with valid JSON and nothing else."
)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

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
        except json.JSONDecodeError:
            logger.warning("Could not parse Claude response as JSON: %s", raw[:200])
            return None

    rec = data.get("recommendation")
    if not rec or not isinstance(rec, dict):
        return None

    # Validate that the recommended player is actually in our available list
    available_ids = {p["id"] for p in ctx.top_available}
    if rec.get("player_id") not in available_ids:
        logger.warning("Claude recommended unavailable player id=%s", rec.get("player_id"))
        return None

    def _pick(d: dict) -> PickSuggestion | None:
        try:
            return PickSuggestion(
                player_id=int(d["player_id"]),
                player_name=str(d["player_name"]),
                position=str(d["position"]),
                adp=float(d["adp"]),
                reasoning=str(d.get("reasoning", "")),
            )
        except (KeyError, ValueError, TypeError):
            return None

    recommendation = _pick(rec)
    if recommendation is None:
        return None

    alternatives = [
        s for d in data.get("alternatives", [])[:3]
        if (s := _pick(d)) is not None
        and s.player_id in available_ids
    ]

    alerts = [str(a) for a in data.get("alerts", []) if a]

    return RecommendationResult(
        recommendation=recommendation,
        alternatives=alternatives,
        alerts=alerts,
        model=_DEFAULT_MODEL,
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
            )
            for p in ctx.top_available[1:4]
        ],
        alerts=["AI service unavailable — showing best available by ADP only."],
        model=f"{model}:fallback",
    )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

# .env.example ships this as a fill-in-the-blank value. If someone copies
# the file without editing it, ANTHROPIC_API_KEY looks "set" to os.getenv()
# but isn't a real key — without this check it would silently be sent to
# Anthropic and only fail once the first recommendation is requested.
_PLACEHOLDER_KEYS = {"your_key_here"}


def build_anthropic_client(api_key_env: str = "ANTHROPIC_API_KEY") -> anthropic.Anthropic | None:
    """
    Reads an API key from the given env var and returns a configured client,
    or None if the key is unset or still the .env.example placeholder.

    Shared by AIService (live per-pick recommendations) and
    backend/ingestion/fetch_synthesis.py (batch "what it means" note
    generation) so the placeholder-key guard — the part actually worth
    not duplicating — only has to live in one place.
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

    return anthropic.Anthropic(api_key=api_key)


class AIService:
    """
    Thin wrapper around the Anthropic client.
    Instantiated once in the FastAPI lifespan and stored on app.state.
    """

    def __init__(self) -> None:
        self._client = build_anthropic_client()
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

        prompt = _build_prompt(ctx)

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            result = _parse_response(raw, ctx)

            if result is None:
                logger.warning("Falling back to ADP — could not parse Claude response.")
                return _fallback(ctx, self._model)

            return result

        except anthropic.APIError as e:
            logger.error("Anthropic API error: %s", e)
            return _fallback(ctx, self._model)


# ---------------------------------------------------------------------------
# CLI — preview the exact prompt Claude would receive, without calling Claude
# ---------------------------------------------------------------------------

def _build_preview_context(top_n: int = 10) -> RecommendationContext:
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

    return RecommendationContext(
        pick_number=1, round_number=1, my_slot=1, league_size=12,
        is_my_turn=True, picks_until_my_turn=0, my_next_pick_number=1,
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
