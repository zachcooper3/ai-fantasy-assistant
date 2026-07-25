"""
Populates the sleeper_id column on Player rows by matching against
Sleeper's full NFL player database.

Run once after ingest_players.py, and again whenever the Sleeper
player database diverges noticeably from our CSV (e.g. mid-season trades).

    python -m backend.ingestion.sync_sleeper_ids

Matching strategy (in order):
  1. Exact full_name + position match
  2. Normalised name match (lowercase, strip punctuation) + position
  3. Suffix-stripped match (drop a trailing Jr/Sr/II/III/IV/V) + position —
     Sleeper's player database omits generational suffixes entirely (verified
     2026-07-25: Sleeper has "James Cook", "Kenneth Walker", "Travis
     Etienne", "Marvin Harrison" — no "III"/"Jr." anywhere), while our ADP
     data keeps them. This was the cause of 16/207 unmatched players after
     the 2026-07-24 refresh — every single one had a suffix.
  4. DST: match by NFL team abbreviation

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


_GENERATIONAL_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _strip_suffix(normalised_name: str) -> str | None:
    """
    Drops a trailing generational suffix from an already-normalised name,
    e.g. "kenneth walker iii" -> "kenneth walker". Returns None if there was
    no suffix to strip, so callers can tell "stripped" apart from "unchanged"
    without a second normalise() call.
    """
    parts = normalised_name.split()
    if len(parts) > 1 and parts[-1] in _GENERATIONAL_SUFFIXES:
        return " ".join(parts[:-1])
    return None


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

async def sync_sleeper_ids() -> tuple[int, int]:
    """
    Matches every Player row against Sleeper's player database and populates
    sleeper_id. Returns (matched, unmatched) so callers can log a summary or
    decide whether to alert on a bad match rate.

    Public (no leading underscore) because backend/ingestion/fetch_adp.py
    calls this directly after every re-ingest — ingest_players.py does a
    full delete-and-reinsert of the Player table, which wipes sleeper_id
    every time, so this has to be re-run after every ADP refresh or live
    Sleeper draft sync silently degrades to name-matching.
    """
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
                # 3. Suffix-stripped match — Sleeper's names never carry a
                #    generational suffix (see module docstring), ours do.
                if sid is None:
                    stripped = _strip_suffix(_normalise(player.name))
                    if stripped is not None:
                        sid = norm_index.get((stripped, sleeper_pos))

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

    return matched, unmatched


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(sync_sleeper_ids())
