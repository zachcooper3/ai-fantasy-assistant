"""
Builds the "what it means" ChromaDB layer — short, Claude-generated
scouting notes synthesized from computed PlayerMetrics (opportunity/volume,
efficiency, team context, consistency & risk, forward-looking/prospect
signals). This is original analysis, not scraped text — see chunker.py's
"what happened" layer for the factual-reporting counterpart (Sleeper injury
status + RotoWire RSS).

Design:
  - One Claude call per player, grounded only in that player's own
    PlayerMetrics row (no invented stats)
  - Players with no metrics row yet (not computed, or nflverse has no data
    for them) are silently skipped — nothing to synthesize from
  - One failed synthesis call shouldn't block the rest of the batch, same
    graceful-degradation pattern as fetch_metrics.py and chunker.py
  - Uses the same Anthropic client/placeholder-key guard as ai_service.py
    (via build_anthropic_client()) rather than duplicating that logic

Run manually:
    py -m backend.ingestion.fetch_synthesis                    # all players with metrics
    py -m backend.ingestion.fetch_synthesis --limit 5           # smoke test, first 5
    py -m backend.ingestion.fetch_synthesis --sleeper-id 4866   # single player
    py -m backend.ingestion.fetch_synthesis --dry-run           # print notes, don't write to Chroma

Author: Zach Cooper
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from sqlmodel import Session, select

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.services.ai_service import build_anthropic_client
from backend.db.database import engine
from backend.db.models import Player, PlayerMetrics

logger = logging.getLogger(__name__)

# Batch synthesis is offline, so it can afford a stronger model than the
# on-the-clock recommendation call if the user wants richer notes — override
# independently of CLAUDE_MODEL. Defaults to the same Haiku model for cost.
SYNTHESIS_MODEL = os.getenv("SYNTHESIS_MODEL", os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001"))

_SYSTEM_PROMPT = (
    "You are an expert NFL fantasy football analyst. You will be given one "
    "player's computed statistical profile (opportunity/volume, efficiency, "
    "team context, consistency/risk, and forward-looking signals) — and "
    "NOTHING else. You do not have outside knowledge about this player's "
    "draft history, experience level (rookie, veteran, etc.), coaching "
    "staff, or team situation beyond what's in this data block, and must "
    "not state or imply any such fact unless it's explicitly present in the "
    "data given to you — for example, never call a player a 'rookie' or "
    "'veteran' unless a field literally says so. "
    "Write a concise 2-4 sentence scouting note synthesizing what this data "
    "means for a fantasy manager evaluating this player. "
    "Reference specific numbers you're given where useful. Do not invent "
    "stats, injuries, or narratives that aren't in the data provided, and "
    "do not attach a qualitative label (e.g. 'run-heavy', 'pass-heavy', "
    "'high-powered') to a number unless that label is clearly and "
    "unambiguously the correct read of it — when in doubt, state the number "
    "plainly instead of characterizing it. "
    "If a category has no data, simply don't mention it — never guess or pad. "
    "Respond with plain text only: no markdown, no headers, no bullet points."
)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _pct(x: float | None) -> str | None:
    return f"{x * 100:.0f}%" if x is not None else None


def _fmt(label: str, value) -> str | None:
    return f"- {label}: {value}" if value is not None else None


def format_metrics_prompt(player: Player, m: PlayerMetrics) -> str:
    """Builds the per-player data block Claude synthesizes from. Only
    non-None fields are included — an absent metric should never be
    silently rendered as 0 or 'N/A' and treated as a real signal."""
    lines = [
        f"Player: {player.name} ({player.position}, {player.team})",
        f"Data window: {m.season} season through week {m.through_week} "
        f"({m.games_played} games played)",
        "",
    ]

    opportunity = list(filter(None, [
        _fmt("Targets/game", m.targets_per_game),
        _fmt("Carries/game", m.carries_per_game),
        _fmt("Red zone touches/game", m.red_zone_touches_per_game),
        _fmt("Snap %", _pct(m.snap_pct)),
        _fmt("Target share", _pct(m.target_share)),
        _fmt("Carry share", _pct(m.carry_share)),
    ]))
    if opportunity:
        lines += ["Opportunity / Volume:"] + opportunity + [""]

    efficiency = list(filter(None, [
        _fmt("Yards/target", m.yards_per_target),
        _fmt("Yards/carry", m.yards_per_carry),
        _fmt("YAC/reception", m.yac_per_reception),
        _fmt("RACR", m.racr),
        _fmt("Catch rate", _pct(m.catch_rate)),
    ]))
    if efficiency:
        lines += ["Efficiency:"] + efficiency + [""]

    team_context = list(filter(None, [
        _fmt("Team pass rate", _pct(m.team_pass_rate)),
        _fmt("Depth chart rank at position", m.depth_chart_rank),
    ]))
    if team_context:
        lines += ["Team Context:"] + team_context + [""]

    consistency = list(filter(None, [
        _fmt("Fantasy points/game (PPR)", m.fantasy_points_avg),
        _fmt("Week-to-week std dev (PPR)", m.fantasy_points_stdev),
        _fmt("Weeks on injury report", m.injury_report_appearances or None),
        _fmt("Games missed to injury", m.games_missed or None),
    ]))
    if consistency:
        lines += ["Consistency & Risk:"] + consistency + [""]

    forward = list(filter(None, [
        _fmt("Target share trend (last 3 wks vs. season)", (
            f"{m.target_share_trend:+.1%}" if m.target_share_trend is not None else None
        )),
        _fmt("Snap % trend (last 3 wks vs. season)", (
            f"{m.snap_pct_trend:+.1%}" if m.snap_pct_trend is not None else None
        )),
        _fmt("Depth chart trend (negative = moving up)", m.depth_chart_trend),
        # A "Rookie or second-year" line used to sit here, reading a column
        # no ingestion script wrote — so it was False for all 185 rows and
        # never rendered once. Rookies get their own grounding data via
        # fetch_rookie_synthesis.py (draft capital + college production),
        # which is a far better signal than a bare boolean anyway.
    ]))
    if forward:
        lines += ["Forward-Looking / Prospect Signals:"] + forward + [""]

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synthesize_note(
    client: anthropic.Anthropic,
    player: Player,
    metrics: PlayerMetrics,
    model: str = SYNTHESIS_MODEL,
) -> str | None:
    """Calls Claude to synthesize one player's note. Returns None (rather
    than raising) on any API error — one bad call shouldn't stop the batch."""
    prompt = format_metrics_prompt(player, metrics)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=300,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        return text or None
    except anthropic.APIError as e:
        logger.warning(f"Synthesis failed for {player.name}: {e}")
        return None


def chunk_synthesis_note(player: Player, note: str) -> tuple[str, dict]:
    """Wraps a synthesized note as a (chunk_text, metadata) pair, tagged
    distinctly from "what happened" chunks so retrieval can tell factual
    reporting apart from AI-generated analysis (see vector_store.query's
    `where={"chunk_type": ...}` scoping)."""
    meta = {
        "chunk_type": "what_it_means",
        "source": "claude_synthesis",
        "player_name": player.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if player.sleeper_id:
        meta["sleeper_id"] = player.sleeper_id
        # One current note per player — regenerating should replace the
        # old note, not pile up a new stale entry each run (Claude's
        # wording differs slightly every call, so a text-hash ID would
        # never collide with the previous run's chunk on its own).
        # See vector_store._chunk_id.
        meta["dedupe_key"] = f"synthesis:{player.sleeper_id}"
    return note, meta


def build_what_it_means_chunks(
    limit: int | None = None,
    sleeper_id: str | None = None,
    sleep_between: float = 0.0,
) -> tuple[list[str], list[dict]]:
    """
    Orchestrator: pulls every (Player, PlayerMetrics) pair with computed
    metrics, synthesizes a note for each, and returns (chunks, metadatas)
    ready for vector_store.add_chunks(). Returns ([], []) — not an
    exception — if no Anthropic key is configured, matching the project's
    established "draft day shouldn't crash over a missing feature" stance.

    Joins Player to PlayerMetrics via sleeper_id, not the player_id FK.
    player_id can go stale the moment a Player reingest reassigns
    autoincrement IDs (see metrics_repo.py's module docstring for the full
    story — this is exactly the join that silently mismatched two players'
    stats in practice). sleeper_id is re-resolved by name on every reingest,
    so it's the one identity guaranteed to still point at the right person.
    """
    client = build_anthropic_client()
    if client is None:
        logger.warning(
            "No Anthropic API key configured — skipping 'what it means' synthesis."
        )
        return [], []

    with Session(engine) as session:
        query = select(Player, PlayerMetrics).where(PlayerMetrics.sleeper_id == Player.sleeper_id)
        if sleeper_id:
            query = query.where(Player.sleeper_id == sleeper_id)
        rows = session.exec(query).all()

    if limit is not None:
        rows = rows[:limit]

    chunks: list[str] = []
    metadatas: list[dict] = []

    for player, metrics in rows:
        note = synthesize_note(client, player, metrics)
        if note is None:
            continue
        text, meta = chunk_synthesis_note(player, note)
        chunks.append(text)
        metadatas.append(meta)

        if sleep_between:
            time.sleep(sleep_between)

    logger.info(f"Synthesized {len(chunks)}/{len(rows)} 'what it means' notes")
    return chunks, metadatas


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate Claude 'what it means' player notes.")
    parser.add_argument("--limit", type=int, default=None, help="Only synthesize the first N players (smoke testing).")
    parser.add_argument("--sleeper-id", type=str, default=None, help="Only synthesize this one player.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between Claude calls (rate limiting).")
    parser.add_argument("--dry-run", action="store_true", help="Print notes instead of writing them to ChromaDB.")
    args = parser.parse_args()

    chunks, metadatas = build_what_it_means_chunks(
        limit=args.limit, sleeper_id=args.sleeper_id, sleep_between=args.sleep,
    )

    if args.dry_run:
        for chunk, meta in zip(chunks, metadatas):
            print(f"\n--- {meta.get('player_name', '?')} ---\n{chunk}")
        print(f"\n({len(chunks)} notes generated, not written — --dry-run)")
        return

    from backend.rag.vector_store import add_chunks
    add_chunks(chunks, metadatas)
    print(f"Added {len(chunks)} 'what it means' chunks to the collection.")


if __name__ == "__main__":
    main()
