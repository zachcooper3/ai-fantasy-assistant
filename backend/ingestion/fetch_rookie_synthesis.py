"""
Builds "what it means" ChromaDB notes for rookies and recent draftees who
have a DraftProfile but no PlayerMetrics row — i.e., exactly the players
fetch_synthesis.py's join structurally excludes (see that file's
docstring: "Players with no metrics row yet ... are silently skipped").

Same chunk_type ("what_it_means") and dedupe_key scheme
(f"synthesis:{sleeper_id}") as fetch_synthesis.py, deliberately:
  - ai_service.py's retrieval reads both without any changes — it just
    queries chunk_type in ("what_happened", "what_it_means") and doesn't
    care which script wrote a given chunk.
  - Once a rookie gets their first real PlayerMetrics row (their first NFL
    season lands), fetch_synthesis.py's own note for them shares this same
    dedupe_key and will overwrite this one on its next run — no explicit
    "graduation" logic needed anywhere; real NFL data should always win
    over pre-draft projection once it exists, and it does automatically by
    virtue of build_rookie_chunks() below excluding anyone who already has
    a PlayerMetrics row (so once they get one, this stops generating notes
    for them, and fetch_synthesis.py picks them up on its own next run).

Design (mirrors fetch_synthesis.py's shape; different grounding data and
system prompt):
  - One Claude call per player, grounded only in that player's DraftProfile
    row (draft capital + final college season production) — no invented
    scouting knowledge, same anti-hallucination discipline as
    fetch_synthesis.py.
  - Players with neither draft capital nor college production resolved
    (fetch_draft_profiles.py / fetch_college_stats.py haven't run, or
    found nothing for them) are silently skipped — nothing to synthesize
    from.
  - One failed synthesis call shouldn't block the rest of the batch.
  - Uses the same Anthropic client/placeholder-key guard as ai_service.py
    (via build_anthropic_client()).

Run manually (after fetch_draft_profiles.py, and fetch_college_stats.py if
you have a CFBD key — this works with just draft capital if you don't):
    py -m backend.ingestion.fetch_rookie_synthesis                    # all eligible rookies
    py -m backend.ingestion.fetch_rookie_synthesis --limit 5           # smoke test, first 5
    py -m backend.ingestion.fetch_rookie_synthesis --sleeper-id 4866   # single player
    py -m backend.ingestion.fetch_rookie_synthesis --dry-run           # print notes, don't write to Chroma

Author: Zach Cooper
"""

from __future__ import annotations

import argparse
import logging
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
from backend.db.models import DraftProfile, Player, PlayerMetrics
from backend.ingestion.fetch_draft_profiles import _current_draft_year
from backend.ingestion.fetch_synthesis import SYNTHESIS_MODEL, _fmt

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert NFL fantasy football analyst. You will be given one "
    "player's known pre-NFL facts — draft capital (round/pick/team) and/or "
    "final college season production — and NOTHING else. You do not have "
    "outside knowledge about this player's landing-spot depth chart, "
    "coaching staff, offensive scheme, or current NFL role beyond what's in "
    "this data block, and must not state or imply any such fact unless it's "
    "explicitly present in the data given to you. "
    "The data block opens with a Status line. READ IT FIRST and let it "
    "decide your framing. Only a player whose Status line says 'incoming "
    "rookie' may be described as a rookie or as having never played; for "
    "anyone else the Status line will tell you they were drafted in an "
    "earlier year and simply have no NFL production on file, which is NOT "
    "the same thing as being new to the league. Never call a player a "
    "rookie, a prospect, or 'unproven at the NFL level' unless the Status "
    "line supports it. "
    "Write a concise 2-4 sentence scouting note synthesizing what this data "
    "suggests for a fantasy manager. Reference specific numbers you're "
    "given where useful: draft capital (round/pick) is one of the strongest "
    "predictors of a player's eventual role, and college production is the "
    "closest thing to a track record present in this data. "
    "If only draft capital is given, evaluate on that alone rather than "
    "guessing at production; if only college production is given (no "
    "draft capital resolved), evaluate on that alone instead. "
    "Do not invent NFL stats, depth chart standing, or scheme fit that "
    "isn't in the data provided, and do not attach a qualitative label "
    "(e.g. 'high-upside', 'boom-or-bust') unless it's clearly and "
    "unambiguously the correct read of the numbers given — when in doubt, "
    "state the number plainly instead of characterizing it. "
    "Respond with plain text only: no markdown, no headers, no bullet points."
)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _status_line(dp: DraftProfile, current_class: int | None = None) -> str:
    """One sentence telling Claude whether this player is actually new to
    the league, so it can't infer that from the mere presence of college
    stats.

    This module selects players who have a DraftProfile but no
    PlayerMetrics row. That was a clean proxy for "rookie" while
    fetch_draft_profiles only pulled two classes. It stopped being one when
    that widened to four (2026-08-14), which deliberately added players
    whose problem is no RECENT season rather than no season at all —
    Jonathon Brooks and MarShawn Lloyd, who have missed nearly their whole
    careers injured, and Tank Dell, a third-year receiver with a productive
    2023 behind him. Fourteen players got notes calling them rookie
    prospects; for Dell that's simply false.

    The selection is still right — these players genuinely have no usage
    data to reason from, which is why they're here. Only the framing was
    wrong, so state the fact rather than narrow the query.
    """
    current_class = current_class or _current_draft_year()
    if dp.draft_year >= current_class:
        return (
            f"Status: incoming rookie, drafted {dp.draft_year} — has not yet "
            f"played an NFL season."
        )
    seasons = current_class - dp.draft_year
    return (
        f"Status: drafted {dp.draft_year}, {seasons} season(s) ago — NOT a rookie. "
        f"No NFL production is on file for him in the most recent season "
        f"(injury, inactive, or a role too small to register), so only his "
        f"pre-NFL facts are available below. Do not describe him as new to "
        f"the league or as never having played."
    )


def format_draft_profile_prompt(player: Player, dp: DraftProfile) -> str:
    """Builds the per-player data block Claude synthesizes from. Only
    non-None fields are included — an absent fact should never be silently
    rendered as 0/"unknown" and treated as a real signal."""
    lines = [
        f"Player: {player.name} ({player.position})",
        _status_line(dp),
        "",
    ]

    draft_bits = list(filter(None, [
        _fmt("Draft year", dp.draft_year),
        _fmt("Round", dp.draft_round),
        _fmt("Pick (overall)", dp.draft_pick),
        _fmt("Drafted by", dp.draft_team),
        _fmt("College", dp.college),
    ]))
    if draft_bits:
        lines += ["Draft Capital:"] + draft_bits + [""]

    college_bits = list(filter(None, [
        _fmt("Season", dp.college_season),
        _fmt("Passing yards", dp.passing_yards),
        _fmt("Passing touchdowns", dp.passing_td),
        _fmt("Interceptions thrown", dp.interceptions_thrown),
        _fmt("Rushing yards", dp.rushing_yards),
        _fmt("Rushing touchdowns", dp.rushing_td),
        _fmt("Carries", dp.carries),
        _fmt("Receiving yards", dp.receiving_yards),
        _fmt("Receiving touchdowns", dp.receiving_td),
        _fmt("Receptions", dp.receptions),
    ]))
    if college_bits:
        lines += ["Final College Season Production:"] + college_bits + [""]

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synthesize_rookie_note(
    client: anthropic.Anthropic,
    player: Player,
    dp: DraftProfile,
    model: str = SYNTHESIS_MODEL,
) -> str | None:
    """Calls Claude to synthesize one rookie's note. Returns None (rather
    than raising) on any API error — one bad call shouldn't stop the batch."""
    prompt = format_draft_profile_prompt(player, dp)
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
        logger.warning(f"Rookie synthesis failed for {player.name}: {e}")
        return None


def chunk_rookie_note(player: Player, note: str) -> tuple[str, dict]:
    """Wraps a synthesized note as a (chunk_text, metadata) pair. Same
    chunk_type and dedupe_key scheme as fetch_synthesis.py's veteran notes
    — see this module's docstring for why that's deliberate."""
    meta = {
        "chunk_type": "what_it_means",
        "source": "claude_synthesis_rookie",
        "player_name": player.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if player.sleeper_id:
        meta["sleeper_id"] = player.sleeper_id
        meta["dedupe_key"] = f"synthesis:{player.sleeper_id}"
    return note, meta


def build_rookie_chunks(
    limit: int | None = None,
    sleeper_id: str | None = None,
    sleep_between: float = 0.0,
) -> tuple[list[str], list[dict]]:
    """
    Orchestrator: pulls every (Player, DraftProfile) pair for a player who
    does NOT already have a PlayerMetrics row, synthesizes a note for each,
    and returns (chunks, metadatas) ready for vector_store.add_chunks().
    Returns ([], []) — not an exception — if no Anthropic key is
    configured, matching fetch_synthesis.py's stance.

    Joins on sleeper_id, not the player_id FK, for the same reason as
    fetch_synthesis.py and everywhere else that reads DraftProfile/
    PlayerMetrics — player_id churns on every ADP reingest (see
    metrics_repo.py's module docstring), sleeper_id doesn't.
    """
    client = build_anthropic_client()
    if client is None:
        logger.warning(
            "No Anthropic API key configured — skipping rookie 'what it means' synthesis."
        )
        return [], []

    with Session(engine) as session:
        query = select(Player, DraftProfile).where(DraftProfile.sleeper_id == Player.sleeper_id)
        if sleeper_id:
            query = query.where(Player.sleeper_id == sleeper_id)
        rows = session.exec(query).all()

        has_metrics = {
            m.sleeper_id for m in session.exec(select(PlayerMetrics)).all() if m.sleeper_id
        }

    # Exclude anyone who already has real NFL data — fetch_synthesis.py is
    # the authoritative source for them (real production beats a pre-draft
    # projection), and generating a redundant rookie note here would just
    # be lower-quality noise competing for the same dedupe_key.
    rows = [(p, dp) for p, dp in rows if p.sleeper_id not in has_metrics]

    if limit is not None:
        rows = rows[:limit]

    chunks: list[str] = []
    metadatas: list[dict] = []

    for player, dp in rows:
        note = synthesize_rookie_note(client, player, dp)
        if note is None:
            continue
        text, meta = chunk_rookie_note(player, note)
        chunks.append(text)
        metadatas.append(meta)

        if sleep_between:
            time.sleep(sleep_between)

    logger.info(f"Synthesized {len(chunks)}/{len(rows)} rookie 'what it means' notes")
    return chunks, metadatas


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate Claude 'what it means' notes for rookies (draft capital + college production)."
    )
    parser.add_argument("--limit", type=int, default=None, help="Only synthesize the first N players (smoke testing).")
    parser.add_argument("--sleeper-id", type=str, default=None, help="Only synthesize this one player.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between Claude calls (rate limiting).")
    parser.add_argument("--dry-run", action="store_true", help="Print notes instead of writing them to ChromaDB.")
    args = parser.parse_args()

    chunks, metadatas = build_rookie_chunks(
        limit=args.limit, sleeper_id=args.sleeper_id, sleep_between=args.sleep,
    )

    if args.dry_run:
        for chunk, meta in zip(chunks, metadatas):
            print(f"\n--- {meta.get('player_name', '?')} ---\n{chunk}")
        print(f"\n({len(chunks)} notes generated, not written — --dry-run)")
        return

    from backend.rag.vector_store import add_chunks
    add_chunks(chunks, metadatas)
    print(f"Added {len(chunks)} rookie 'what it means' chunks to the collection.")


if __name__ == "__main__":
    main()
