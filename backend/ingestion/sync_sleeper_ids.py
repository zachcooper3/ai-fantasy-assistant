"""
Populates the sleeper_id column on Player rows by matching against
Sleeper's full NFL player database.

Run once after ingest_players.py, and again whenever the Sleeper
player database diverges noticeably from our CSV (e.g. mid-season trades).

    python -m backend.ingestion.sync_sleeper_ids

Matching strategy (in order):
  1. Exact full_name + position match
  2. Normalised name match (lowercase, strip punctuation) + position
  3. DST: match by NFL team abbreviation

Author: Zach Cooper
"""

import asyncio
import re
import sys
import logging

from sqlmodel import Session, select

from backend.db.database import create_db_and_tables, engine
from backend.db.models import Player
from backend.app.services.sleeper_client import get_nfl_players

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    """Lowercase, remove punctuation and extra whitespace."""
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)   # strip apostrophes, dots, hyphens
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ---------------------------------------------------------------------------
# Build lookup indexes from Sleeper player data
# ---------------------------------------------------------------------------

def _build_indexes(
    sleeper_players: dict[str, dict],
) -> tuple[dict[tuple, str], dict[tuple, str], dict[str, str]]:
    """
    Returns three lookup dicts:
      exact_index    — {(full_name, position): sleeper_id}
      norm_index     — {(normalised_name, position): sleeper_id}
      dst_index      — {team_abbreviation: sleeper_id}   (position == "DEF")
    """
    exact_index: dict[tuple, str] = {}
    norm_index: dict[tuple, str] = {}
    dst_index: dict[str, str] = {}

    for sid, p in sleeper_players.items():
        pos = (p.get("position") or "").upper()
        full_name = p.get("full_name") or ""
        team = (p.get("team") or "").upper()

        if pos == "DEF" and team:
            dst_index[team] = sid

        if full_name and pos:
            exact_index[(full_name, pos)] = sid
            norm_index[(_normalise(full_name), pos)] = sid

    return exact_index, norm_index, dst_index


# ---------------------------------------------------------------------------
# Main sync
# ---------------------------------------------------------------------------

async def _sync() -> None:
    logger.info("Fetching Sleeper NFL player database…")
    sleeper_players = await get_nfl_players()
    logger.info(f"  {len(sleeper_players):,} players received from Sleeper")

    exact_index, norm_index, dst_index = _build_indexes(sleeper_players)

    create_db_and_tables()

    matched = unmatched = 0
    unmatched_names: list[str] = []

    with Session(engine) as session:
        players = session.exec(select(Player)).all()

        for player in players:
            pos = player.position.upper()
            # Sleeper uses "DEF" for what we call "DST"
            sleeper_pos = "DEF" if pos == "DST" else pos

            sid: str | None = None

            if pos == "DST":
                # Match by team abbreviation
                sid = dst_index.get(player.team.upper())
            else:
                # 1. Exact match
                sid = exact_index.get((player.name, sleeper_pos))
                # 2. Normalised match
                if sid is None:
                    sid = norm_index.get((_normalise(player.name), sleeper_pos))

            if sid:
                player.sleeper_id = sid
                session.add(player)
                matched += 1
            else:
                unmatched += 1
                unmatched_names.append(f"{player.name} ({player.position})")

        session.commit()

    logger.info(f"\nResults: {matched} matched, {unmatched} unmatched")
    if unmatched_names:
        logger.info("Unmatched players:")
        for name in unmatched_names[:20]:
            logger.info(f"  - {name}")
        if len(unmatched_names) > 20:
            logger.info(f"  … and {len(unmatched_names) - 20} more")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(_sync())
