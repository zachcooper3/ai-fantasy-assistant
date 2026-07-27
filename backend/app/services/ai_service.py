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


# ---------------------------------------------------------------------------
# Retrieval — grounds the prompt in real news/analysis from ChromaDB
# ---------------------------------------------------------------------------

# Only the most relevant candidates get a retrieval lookup — querying Chroma
# for all 25 listed players would balloon both latency and prompt size for
# players near the bottom of the list that are unlikely to be picked anyway.
_MAX_CONTEXT_PLAYERS = 10
_MAX_CHUNKS_PER_PLAYER_PER_TYPE = 2


def _retrieve_player_context(top_available: list[dict]) -> str:
    """
    Pulls "what happened" (Sleeper injury status + RotoWire news) and "what
    it means" (Claude-synthesized analysis) chunks for the top candidate
    players out of ChromaDB, so the recommendation is grounded in more than
    ADP and roster math alone.

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
    for p in top_available[:_MAX_CONTEXT_PLAYERS]:
        sleeper_id = p.get("sleeper_id")
        if not sleeper_id:
            continue

        player_lines: list[str] = []
        for chunk_type, label in (("what_happened", "News"), ("what_it_means", "Analysis")):
            try:
                results = vector_query(
                    p["name"],
                    n_results=_MAX_CHUNKS_PER_PLAYER_PER_TYPE,
                    where={"$and": [{"sleeper_id": sleeper_id}, {"chunk_type": chunk_type}]},
                )
            except Exception as e:
                logger.warning(f"Vector query failed for {p['name']} ({chunk_type}): {e}")
                continue
            player_lines += [f"  [{label}] {r}" for r in results]

        if player_lines:
            sections.append(f"- {p['name']} ({p['position']}):\n" + "\n".join(player_lines))

    if not sections:
        return ""

    return "\n".join([
        "## Player News & Analysis (retrieved)",
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
        f"- Overall pick: #{ctx.pick_number} (Round {ctx.round_number} of 15)",
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

    # Retrieved player news/analysis (best-effort; omitted if unavailable)
    retrieved = _retrieve_player_context(ctx.top_available)
    if retrieved:
        lines.append(retrieved)

    # Output schema
    lines += [
        "## Task",
        "Recommend the best pick for my team right now.",
        "Consider: roster needs, positional scarcity, ADP value, and how long until my next turn.",
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
            "alerts": ["<any scarcity warnings, handcuff notes, or tier drop-off flags>"],
        }, indent=2),
    ]

    return "\n".join(lines)


_SYSTEM_PROMPT = (
    "You are an expert fantasy football draft advisor for a PPR Sleeper league. "
    "You give concise, data-driven recommendations grounded in ADP, positional scarcity, "
    "and roster construction strategy. "
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
