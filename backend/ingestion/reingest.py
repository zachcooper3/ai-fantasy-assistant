"""
Re-runs ingest -> Sleeper ID resync -> PlayerMetrics/DraftProfile relink
against whatever CSV is already sitting at data/raw/fantasypros_adp.csv (or
--csv), without fetching anything new first.

Why this exists separately from fetch_adp.py: that script's post-fetch
sequence (ingest_csv -> sync_sleeper_ids -> relink_player_ids x2) only runs
after ITS OWN network fetch from FantasyFootballCalculator. There was no
entry point for "I already dropped a fresh CSV in data/raw/ by hand (e.g.
via convert_fantasypros_export.py) — just load it," so replacing the ADP
source manually meant either faking a fetch or copy-pasting these four
steps by hand. This is that entry point, same four steps, same order.

ingest_csv() below does a full delete-and-reinsert of the Player table,
which reassigns every Player.id and wipes sleeper_id — the two relink/resync
steps repair that immediately rather than leaving PlayerMetrics/DraftProfile
silently pointing at the wrong player until someone remembers to run them
(see metrics_repo.relink_player_ids' docstring for the live incident this
guards against).

Run:
    py -m backend.ingestion.reingest
    py -m backend.ingestion.reingest --csv data/raw/fantasypros_adp.csv
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_DEFAULT_CSV = str(Path(__file__).resolve().parents[2] / "data" / "raw" / "fantasypros_adp.csv")


async def reingest(csv_path: str = _DEFAULT_CSV) -> None:
    from backend.ingestion.ingest_players import ingest_csv
    count = ingest_csv(csv_path)
    print(f"Ingested {count} players from {csv_path}.")

    print("Re-syncing Sleeper player IDs (ingest wipes sleeper_id) ...")
    from backend.ingestion.sync_sleeper_ids import sync_sleeper_ids
    matched, unmatched = await sync_sleeper_ids()
    print(f"Done. {matched} matched, {unmatched} unmatched.")

    # A sync that returns but matches almost nobody is the dangerous case:
    # it doesn't raise, so the relinks below would run against a crosswalk
    # that's mostly empty and read the gap as "these players left the pool."
    # The repos' own MAX_ORPHAN_FRACTION guard is the real backstop (it
    # refuses to delete and raises RelinkAborted), but say so up front too —
    # the number is right here and worth seeing before the relink output.
    if matched == 0:
        print(
            "\n!! Sleeper matched 0 players. Not relinking — every metrics and "
            "draft-profile row would look orphaned. Check Sleeper's API, then "
            "re-run this command.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    from backend.db import draft_profile_repo, metrics_repo
    from backend.db.database import engine
    from backend.db.metrics_repo import RelinkAborted
    from sqlmodel import Session

    print("Relinking PlayerMetrics to the reassigned Player IDs ...")
    try:
        with Session(engine) as session:
            relinked, orphaned = metrics_repo.relink_player_ids(session)
        print(f"Done. {relinked} relinked, {orphaned} orphaned.")
    except RelinkAborted as e:
        print(f"\n!! PlayerMetrics relink aborted, nothing deleted: {e}", file=sys.stderr)
        raise SystemExit(1)

    print("Relinking DraftProfile to the reassigned Player IDs ...")
    try:
        with Session(engine) as session:
            relinked, orphaned = draft_profile_repo.relink_player_ids(session)
        print(f"Done. {relinked} relinked, {orphaned} orphaned.")
    except RelinkAborted as e:
        print(f"\n!! DraftProfile relink aborted, nothing deleted: {e}", file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-ingest data/raw/fantasypros_adp.csv (or --csv) and repair the IDs that depend on it."
    )
    parser.add_argument("--csv", default=_DEFAULT_CSV, help="CSV to ingest (default: data/raw/fantasypros_adp.csv).")
    args = parser.parse_args()
    asyncio.run(reingest(args.csv))


if __name__ == "__main__":
    main()
