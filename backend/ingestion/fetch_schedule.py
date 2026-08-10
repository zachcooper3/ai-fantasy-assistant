"""
Pulls the published NFL schedule for a season from nflverse data via
nflreadpy, and replaces backend/db/models.py::Game's rows for that season.

nflreadpy (not the deprecated nfl_data_py) is already a dependency — see
fetch_metrics.py. load_schedules() pulls from the same nflverse-data GitHub
release infrastructure fetch_metrics.py's other loaders already depend on,
so this adds no new library and no API key.

Run manually:
    py -m backend.ingestion.fetch_schedule                # current draft season (see _default_season)
    py -m backend.ingestion.fetch_schedule --season 2026   # explicit override

Not part of main.py's startup auto-refresh, same reasoning as
fetch_metrics.py: the schedule for a season is published once (typically
mid-May) and essentially never changes after that except rare game-time
moves, so a manual, occasional pull is enough — there's no draft-day reason
to redo this on every backend restart the way fetch_adp.py's ADP pull is.

Column names below are nflverse's well-established public schema for the
schedules/games dataset — stable for years and unrelated to the columns
fetch_metrics.py had to guess at (those are per-play/per-week stat feeds,
which nflverse has reworked before; the schedule table is a much simpler,
more stable shape). Still looked up through _first()-style multi-candidate
fallback rather than hardcoded, for the same reason fetch_metrics.py does
it: an upstream rename should degrade a row to missing/skipped, not crash
the whole pull. Run `py -m backend.tools.diagnose_ingestion` to confirm the
real column names against a live pull before trusting this blindly — that
tool now checks load_schedules too.

Author: Zach Cooper
"""

import argparse
import logging
import sys
from datetime import datetime
from typing import Any

from sqlmodel import Session

from backend.db import game_repo
from backend.db.database import create_db_and_tables, engine

logger = logging.getLogger(__name__)


def _default_season(today: datetime | None = None) -> int:
    """
    The season currently being drafted for or played, named for the
    calendar year it kicks off in — same naming convention fetch_metrics.py
    documents in its own _default_season, used in the OPPOSITE direction
    there on purpose: that function wants the last season with real STATS,
    which lags a year for most of the calendar (2026 has no stats until
    2026's games are played). This one wants the season whose SCHEDULE
    already exists, and unlike stats, the schedule is published months
    before kickoff (typically mid-May) — so today.year is correct for the
    entire Jan-Dec span once that year's schedule is out, with no lag.

    Known gap: for a few months after a season ends and before the next
    one's schedule is released (roughly February-April), today.year points
    at a season nflverse doesn't have schedule data for yet, and the pull
    below will fail the same way fetch_metrics.py's does when run too early
    for its season. Pass --season explicitly if you hit that window.
    """
    today = today or datetime.now()
    return today.year


def _first(row: dict, *candidates: str) -> Any:
    """Returns the first non-None value found under any of the candidate
    column names, or None if none of them are present in this row. Mirrors
    fetch_metrics.py's helper of the same name/behavior."""
    for c in candidates:
        if c in row and row[c] is not None:
            return row[c]
    return None


def _parse_game_date(value: Any) -> datetime | None:
    """nflreadpy's `gameday` column is typically an ISO date string
    ("2026-09-10") or a date/datetime object depending on the polars dtype
    it resolves to for a given pull. Accepts either; returns None (not a
    crash) for anything else, since a missing kickoff date shouldn't block
    storing the week/opponent, which is the data this table exists for."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _shape_games(rows: list[dict], season: int) -> list[dict]:
    """
    Converts raw load_schedules() rows into game_repo.replace_season's
    input shape, skipping any row missing a field this table treats as
    required (week, home_team, away_team) rather than storing a
    half-populated game that would just be a landmine for get_opponent.
    """
    games = []
    skipped = 0
    for row in rows:
        week = _first(row, "week")
        home = _first(row, "home_team")
        away = _first(row, "away_team")
        if week is None or not home or not away:
            skipped += 1
            continue
        games.append({
            "week": int(week),
            "game_type": _first(row, "game_type", "season_type") or "REG",
            "home_team": home,
            "away_team": away,
            "game_date": _parse_game_date(_first(row, "gameday", "game_date")),
        })
    if skipped:
        logger.warning(
            "Skipped %d row(s) missing week/home_team/away_team while "
            "shaping the %d schedule — check load_schedules' real column "
            "names with diagnose_ingestion.py if this is more than a "
            "handful.",
            skipped, season,
        )
    return games


def refresh_schedule(season: int) -> int:
    """
    Pulls the schedule for `season` and replaces Game's rows for it.
    Returns the number of games stored (0 signals a failed/empty pull —
    check the logged warning for why).
    """
    import nflreadpy as nfl

    logger.info("Pulling %d schedule from nflverse via nflreadpy...", season)
    try:
        df = nfl.load_schedules(seasons=season)
    except Exception:
        logger.exception(
            "load_schedules failed for season %d — nothing stored. If this "
            "season's schedule hasn't been published yet, pass an earlier "
            "--season.",
            season,
        )
        return 0

    rows = df.to_dicts()
    if not rows:
        logger.warning("load_schedules returned 0 rows for season %d.", season)
        return 0

    games = _shape_games(rows, season)
    with Session(engine) as session:
        stored = game_repo.replace_season(session, season, games)
    return stored


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull the NFL schedule from nflverse data (via nflreadpy) and load into SQLite."
    )
    parser.add_argument(
        "--season", type=int, default=None,
        help="NFL season year (default: the current draft season — see _default_season)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Debug-level logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    season = args.season or _default_season()
    create_db_and_tables()
    stored = refresh_schedule(season)
    if stored == 0:
        print(
            "No games stored — check the warnings above (season may not "
            "be published yet, or a column may have been renamed upstream; "
            "run py -m backend.tools.diagnose_ingestion to check).",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Done. {stored} games stored for season {season}.")


if __name__ == "__main__":
    main()
