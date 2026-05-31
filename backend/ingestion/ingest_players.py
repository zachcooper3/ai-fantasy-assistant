"""
Ingests the FantasyPros ADP CSV into the SQLite Player table.

Run from the repo root:
    python -m backend.ingestion.ingest_players
    python -m backend.ingestion.ingest_players --csv data/raw/fantasypros_adp.csv

The script is idempotent: running it again truncates and reloads the table,
so you can safely re-run whenever you have a fresher CSV.
Author: Zach Cooper
"""

import argparse
import csv
import re
import sys
from pathlib import Path

from sqlmodel import Session, delete

from backend.db.database import create_db_and_tables, engine
from backend.db.models import Player


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_position(pos_rank: str) -> str:
    """
    Extracts the position letters from a pos_rank string.
    Examples: "RB1" -> "RB", "WR12" -> "WR", "QB3" -> "QB", "DEF" -> "DEF"
    """
    match = re.match(r"([A-Za-z]+)", pos_rank)
    return match.group(1).upper() if match else pos_rank.upper()


def _parse_bye(bye_str: str) -> int | None:
    """Converts the bye week string to an int, or None if blank/invalid."""
    bye_str = bye_str.strip()
    if not bye_str:
        return None
    try:
        return int(bye_str)
    except ValueError:
        return None


def _parse_adp(adp_str: str) -> float:
    """
    Converts the ADP string to a float.
    Falls back to a high number (999.0) if the value is missing or malformed,
    so undraftable players sort to the bottom.
    """
    adp_str = adp_str.strip()
    if not adp_str:
        return 999.0
    try:
        return float(adp_str)
    except ValueError:
        return 999.0


def _row_to_player(row: dict[str, str]) -> Player:
    """Converts a single CSV row dict into a Player model instance."""
    pos_rank = row.get("POS", "").strip()
    return Player(
        rank=int(row.get("Rank", "0").strip()),
        name=row.get("Player", "").strip(),
        team=row.get("Team", "").strip(),
        bye=_parse_bye(row.get("Bye", "")),
        pos_rank=pos_rank,
        position=_parse_position(pos_rank),
        adp=_parse_adp(row.get("AVG", "")),
        is_available=True,
    )


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def ingest_csv(csv_path: str) -> int:
    """
    Loads all rows from csv_path into the Player table.
    Clears existing rows first (full reload).
    Returns the number of players inserted.
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"[ERROR] CSV not found: {path.resolve()}", file=sys.stderr)
        sys.exit(1)

    players: list[Player] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                players.append(_row_to_player(row))
            except Exception as e:
                print(f"[WARN] Skipping row {row}: {e}", file=sys.stderr)

    create_db_and_tables()

    with Session(engine) as session:
        # Full reload — clear existing rows so re-runs don't leave stale data
        session.exec(delete(Player))
        session.commit()

        session.add_all(players)
        session.commit()

    return len(players)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest FantasyPros ADP CSV into SQLite")
    parser.add_argument(
        "--csv",
        default="data/raw/fantasypros_adp.csv",
        help="Path to the FantasyPros ADP CSV file (default: data/raw/fantasypros_adp.csv)",
    )
    args = parser.parse_args()

    print(f"Ingesting {args.csv} ...")
    count = ingest_csv(args.csv)
    print(f"Done. {count} players loaded into SQLite.")
