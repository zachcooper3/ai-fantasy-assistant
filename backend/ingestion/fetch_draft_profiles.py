"""
Computes DraftProfile rows (draft year/round/pick/team/college) from
nflverse data via nflreadpy, and upserts them into SQLite.

This exists so rookies — who structurally can never have a PlayerMetrics
row (see that model's docstring) — have *something* concrete for the AI
prompt to reason about instead of a blank "no data" line. Draft capital
(round/pick) is one of the most predictive signals for a rookie's fantasy
outlook.

Run manually:
    py -m backend.ingestion.fetch_draft_profiles                  # last 2 draft classes
    py -m backend.ingestion.fetch_draft_profiles --years 1         # this year's class only
    py -m backend.ingestion.fetch_draft_profiles --dry-run         # print matches, don't write
    py -m backend.ingestion.fetch_draft_profiles --sleeper-id 1234 # debug one player

This is NOT part of main.py's startup auto-refresh, same reasoning as
fetch_metrics.py — draft classes don't change intra-day, so there's no
value in pulling this on every server boot. Run it once per draft class
(effectively: once a year, right after the NFL draft) or whenever you want
to confirm this season's rookies are covered.

A note on verification: like fetch_metrics.py, this was written without
access to nflreadpy's live data from the dev sandbox (network-restricted
there — GitHub release-asset downloads are blocked). It reuses
fetch_metrics.py's own column-resolution helpers (_load_dicts, _first) for
exactly the same reason that file already documents: nflreadpy's exact
column names are resolved defensively, tried against a few plausible
candidates, rather than assumed correct. Run with logging at INFO level
(the default) the first time and check which columns actually resolved —
if draft_round/draft_pick/college come back None for everyone, that's the
thing to fix, not a sign the whole approach is broken.

Author: Zach Cooper
"""

import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import Session, select

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db import draft_profile_repo
from backend.db.database import create_db_and_tables, engine
from backend.db.models import Player
from backend.ingestion.fetch_metrics import _first, _load_dicts

logger = logging.getLogger(__name__)

# Draft picks only matter for the fantasy-relevant offensive positions this
# app actually tracks as individual players — DST rows in our Player table
# represent whole team defenses, not a draftable individual, and kickers
# aren't modeled anywhere else in this app either (see ai_service.py's
# _format_metrics_section DST/K handling), so there's nothing useful to
# attach a draft profile to for either.
_FANTASY_POSITIONS = {"QB", "RB", "WR", "TE"}


def _current_draft_year(today: datetime | None = None) -> int:
    """
    The NFL draft happens every year in late April. Unlike NFL *seasons*
    (named for kickoff year, spanning into the next calendar year — see
    fetch_metrics.py's _default_season), a draft *class* is simply the
    calendar year it happened in. From May onward, that year's draft is
    already in the books; before that, the most recent draft was last
    calendar year's.
    """
    today = today or datetime.now()
    return today.year if today.month >= 5 else today.year - 1


# ---------------------------------------------------------------------------
# Name matching — same normalize/suffix-strip approach as sync_sleeper_ids.py
# (duplicated rather than imported, matching chunker.py's precedent: a few
# lines of pure string logic isn't worth an inter-module coupling for)
# ---------------------------------------------------------------------------

_GENERATIONAL_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _normalise(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _strip_suffix(normalised_name: str) -> str | None:
    parts = normalised_name.split()
    if len(parts) > 1 and parts[-1] in _GENERATIONAL_SUFFIXES:
        return " ".join(parts[:-1])
    return None


def _build_player_index(players: list[Player]) -> dict[tuple[str, str], Player]:
    """
    {(normalised_name, position): Player}. Indexes each player under BOTH
    their full normalised name AND a suffix-stripped variant (when they
    have one) — nflverse's draft_picks names, like Sleeper's (see
    sync_sleeper_ids.py's docstring), omit generational suffixes entirely,
    while our own Player.name (from FantasyPros) keeps them. The suffix
    lives on OUR side of this particular match, not the query side, so
    unlike sync_sleeper_ids.py (which strips the query), this has to widen
    the index itself to catch both forms.
    """
    index: dict[tuple[str, str], Player] = {}
    for p in players:
        pos = p.position.upper()
        normalised = _normalise(p.name)
        index[(normalised, pos)] = p
        stripped = _strip_suffix(normalised)
        if stripped is not None:
            index.setdefault((stripped, pos), p)
    return index


def _match_player(index: dict[tuple[str, str], Player], name: str, position: str) -> Player | None:
    """Best-effort name+position match against local Player rows. A rookie
    drafted this year should already be in our Player table (FantasyPros
    ADP lists draft-eligible rookies before the season starts) with
    sleeper_id already resolved by sync_sleeper_ids.py — this just needs to
    find which row is them."""
    return index.get((_normalise(name), position.upper()))


# ---------------------------------------------------------------------------
# Draft pick data
# ---------------------------------------------------------------------------

def _load_draft_picks(draft_years: list[int]) -> list[dict]:
    """
    Loads nflreadpy's draft_picks table for the given draft classes and
    filters to fantasy-relevant offensive positions. Column names are
    resolved defensively via _first (see module docstring).
    """
    rows = _load_dicts("load_draft_picks", seasons=draft_years)
    logger.info(f"load_draft_picks: {len(rows):,} raw rows for seasons {draft_years}")

    picks: list[dict] = []
    for row in rows:
        position = (_first(row, "position", "pos") or "").upper()
        if position not in _FANTASY_POSITIONS:
            continue

        name = _first(row, "pfr_player_name", "player_name", "name", "cfb_player_name")
        if not name:
            continue

        picks.append({
            "name": name,
            "position": position,
            "draft_year": _first(row, "season", "draft_year", "year"),
            "draft_round": _first(row, "round"),
            "draft_pick": _first(row, "pick", "overall", "pick_number"),
            "draft_team": _first(row, "team", "franchise", "club_code"),
            "college": _first(row, "college", "school"),
        })

    logger.info(f"  {len(picks):,} rows at a fantasy-relevant position ({sorted(_FANTASY_POSITIONS)})")
    return picks


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def refresh_draft_profiles(years: int = 2, sleeper_id: str | None = None) -> tuple[int, int]:
    """
    Pulls the last `years` draft classes, matches each pick to a local
    Player row, and upserts a DraftProfile for each match. Returns
    (matched, unmatched).

    One bad row (missing name, no matching Player) is skipped and logged,
    not fatal to the batch — same graceful-degradation stance as
    fetch_metrics.py and chunker.py.
    """
    create_db_and_tables()

    current = _current_draft_year()
    draft_years = list(range(current - years + 1, current + 1))

    try:
        picks = _load_draft_picks(draft_years)
    except Exception as e:
        logger.warning(f"Could not load draft_picks — aborting: {e}")
        return 0, 0

    with Session(engine) as session:
        players = session.exec(select(Player)).all()
        index = _build_player_index(players)

        matched = unmatched = 0
        unmatched_names: list[str] = []

        for pick in picks:
            player = _match_player(index, pick["name"], pick["position"])
            if player is None:
                unmatched += 1
                unmatched_names.append(f"{pick['name']} ({pick['position']}, {pick['draft_year']})")
                continue

            if sleeper_id and player.sleeper_id != sleeper_id:
                continue

            if pick["draft_year"] is None:
                logger.warning(f"Skipping {pick['name']} — no draft_year resolved from any candidate column")
                unmatched += 1
                continue

            draft_profile_repo.upsert_draft_profile(
                session,
                player_id=player.id,
                sleeper_id=player.sleeper_id,
                draft_year=int(pick["draft_year"]),
                draft_round=int(pick["draft_round"]) if pick["draft_round"] is not None else None,
                draft_pick=int(pick["draft_pick"]) if pick["draft_pick"] is not None else None,
                draft_team=pick["draft_team"],
                college=pick["college"],
            )
            matched += 1

    logger.info(f"Draft profiles: {matched} matched, {unmatched} unmatched")
    if unmatched_names:
        logger.info("Unmatched picks (likely not in our ADP pool — undrafted-in-fantasy rookies, e.g. deep-bench picks):")
        for name in unmatched_names[:20]:
            logger.info(f"  - {name}")
        if len(unmatched_names) > 20:
            logger.info(f"  … and {len(unmatched_names) - 20} more")

    return matched, unmatched


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Fetch NFL draft-day facts (round/pick/college) for recent draft classes.")
    parser.add_argument("--years", type=int, default=2, help="How many recent draft classes to pull (default 2: rookies + second-years).")
    parser.add_argument("--sleeper-id", type=str, default=None, help="Only process this one player (debugging).")
    parser.add_argument("--dry-run", action="store_true", help="Log matches without writing to the database.")
    args = parser.parse_args()

    if args.dry_run:
        current = _current_draft_year()
        draft_years = list(range(current - args.years + 1, current + 1))
        picks = _load_draft_picks(draft_years)
        with Session(engine) as session:
            index = _build_player_index(session.exec(select(Player)).all())
        for pick in picks:
            player = _match_player(index, pick["name"], pick["position"])
            status = f"-> matched Player #{player.id} ({player.sleeper_id})" if player else "-> NO MATCH"
            print(f"{pick['name']} ({pick['position']}, {pick['draft_year']}, "
                  f"Rd{pick['draft_round']} Pk{pick['draft_pick']}, {pick['college']}) {status}")
        return

    matched, unmatched = refresh_draft_profiles(years=args.years, sleeper_id=args.sleeper_id)
    print(f"Done. {matched} matched, {unmatched} unmatched.")


if __name__ == "__main__":
    main()
