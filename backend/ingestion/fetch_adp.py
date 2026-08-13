"""
Fetches fresh PPR ADP data from FantasyFootballCalculator.com, writes it to
data/raw/fantasypros_adp.csv in the format expected by ingest_players.py,
then re-ingests the database automatically.

Source: https://fantasyfootballcalculator.com/api/v1/adp/ppr

Refresh cadence (auto, on startup):
  - August 1 – September 10 (draft season): every 2 days
  - Rest of year:                            every 7 days

Run manually:
    python -m backend.ingestion.fetch_adp
    python -m backend.ingestion.fetch_adp --year 2026 --teams 12
    python -m backend.ingestion.fetch_adp --no-ingest   # write CSV only

Author: Zach Cooper
"""

import argparse
import asyncio
import csv
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import httpx

# Allow running this file directly (`py backend/ingestion/fetch_adp.py`) as
# well as the documented `py -m backend.ingestion.fetch_adp`. Direct
# execution puts this file's own directory on sys.path, not the repo root,
# so the deferred `from backend.X import Y` imports inside auto_refresh()
# and main() below would otherwise fail with "No module named 'backend'".
# Only kicks in when this file is the actual entry point — running via -m
# already sets __package__ correctly, so this is a no-op in that case.
if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FFC_URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr"
# Anchored to the repo root, not the CWD — same reasoning as DB_PATH in
# backend/db/database.py (audit W10): a CWD-relative path here meant the
# staleness check, the CSV write, and the startup banner could all be
# looking at a different file depending on where the process was launched.
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "fantasypros_adp.csv"
CURRENT_YEAR = datetime.now().year

# Positions to include — matches what ingest_players.py and the app expect
VALID_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF"}

# A full PPR ADP pull is normally ~150-200 draftable players across these
# positions (161 as of 2026-07-24). If FFC ever returns a degraded response
# that's non-empty but suspiciously small (a partial API response, a
# mid-deploy blip on their end, etc.), we'd otherwise silently overwrite a
# good CSV with a bad one. This is a floor well below any real response,
# just high enough to catch "something's wrong" before it hits SQLite.
MIN_EXPECTED_PLAYERS = 100


# ---------------------------------------------------------------------------
# Refresh cadence
# ---------------------------------------------------------------------------

def adp_max_age_days() -> int:
    """
    Returns the staleness threshold in days based on time of year.
    Draft season (Aug 1 – Sep 10): every 2 days — ADP shifts fast.
    Off-season: every 7 days — data barely moves.
    """
    now = datetime.now()
    in_draft_season = now.month == 8 or (now.month == 9 and now.day <= 10)
    return 2 if in_draft_season else 7


# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------

def should_refresh(path: Path = OUT_PATH, max_age_days: int | None = None) -> bool:
    """Returns True if the CSV is missing or older than the threshold."""
    if max_age_days is None:
        max_age_days = adp_max_age_days()
    if not path.exists():
        return True
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age >= timedelta(days=max_age_days)


def adp_age_str(path: Path = OUT_PATH) -> str:
    """Human-readable age of the CSV file, e.g. '3d ago'."""
    if not path.exists():
        return "not found"
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    if age.days == 0:
        hours = age.seconds // 3600
        return f"{hours}h ago" if hours else "just now"
    return f"{age.days}d ago"


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_adp(year: int, teams: int) -> list[dict]:
    """Synchronous fetch — safe to call from CLI or asyncio.to_thread."""
    url = f"{FFC_URL}?teams={teams}&year={year}&count=400"
    print(f"Fetching PPR ADP from {url} ...")
    resp = httpx.get(url, timeout=15.0, follow_redirects=True)
    resp.raise_for_status()
    data = resp.json()
    players = data.get("players", [])
    if not players:
        raise ValueError(
            f"API returned 0 players (year={year}, teams={teams}). "
            "ADP data for this year may not be available yet — "
            "FantasyFootballCalculator data typically appears in July/August."
        )
    return players


# ---------------------------------------------------------------------------
# Normalise to ingest_players CSV format
# ---------------------------------------------------------------------------

def build_csv_rows(players: list[dict]) -> list[dict]:
    """
    Converts FFC player dicts to CSV rows matching the ingest_players format:
        Rank, Player, Team, Bye, POS, AVG
    """
    filtered = [
        p for p in players
        if p.get("position", "").upper() in VALID_POSITIONS
        and float(p.get("adp", 0)) > 0
    ]
    filtered.sort(key=lambda p: float(p.get("adp", 999)))

    pos_counters: dict[str, int] = defaultdict(int)
    rows = []

    for overall_rank, player in enumerate(filtered, start=1):
        pos = player.get("position", "").upper()
        pos_counters[pos] += 1

        bye_raw = player.get("bye")
        try:
            bye_val = int(bye_raw) if bye_raw else ""
        except (ValueError, TypeError):
            bye_val = ""

        rows.append({
            "Rank": overall_rank,
            "Player": player.get("name", "").strip(),
            "Team":   player.get("team", "").strip().upper(),
            "Bye":    bye_val,
            "POS":    f"{pos}{pos_counters[pos]}",
            "AVG":    round(float(player.get("adp", 0)), 1),
        })

    return rows


# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Rank", "Player", "Team", "Bye", "POS", "AVG"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} players → {path}")


# ---------------------------------------------------------------------------
# Validation — catch a bad fetch before it overwrites good data
# ---------------------------------------------------------------------------

def _validate_rows(rows: list[dict]) -> None:
    """
    Raises ValueError if the fetched/filtered data looks implausible. Row
    count is a cheap, effective check: a real PPR ADP pull is never anywhere
    close to MIN_EXPECTED_PLAYERS, so falling below it means something went
    wrong upstream (a partial API response, an FFC-side bug, etc.) even
    though the request itself returned 200 with a non-empty body.
    """
    if len(rows) < MIN_EXPECTED_PLAYERS:
        raise ValueError(
            f"Only {len(rows)} players survived filtering — expected at "
            f"least {MIN_EXPECTED_PLAYERS}. Refusing to overwrite "
            f"{OUT_PATH} with what looks like a bad or partial response."
        )


def _existing_row_count(path: Path) -> int | None:
    """Row count of the CSV currently on disk, or None if it doesn't exist yet."""
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def _write_and_report(rows: list[dict], path: Path = OUT_PATH) -> None:
    """
    Validates the new data, writes it, and logs an old-count -> new-count
    diff — the at-a-glance signal for "did this refresh look sane?" without
    having to open the CSV or query the DB by hand.
    """
    _validate_rows(rows)
    previous = _existing_row_count(path)
    write_csv(rows, path)
    if previous is None:
        print(f"ADP diff: no previous CSV on disk — {len(rows)} players written.")
    else:
        delta = len(rows) - previous
        sign = "+" if delta >= 0 else ""
        print(f"ADP diff: {previous} -> {len(rows)} players ({sign}{delta})")


# ---------------------------------------------------------------------------
# Auto-refresh (called from app lifespan)
# ---------------------------------------------------------------------------

async def auto_refresh(year: int = CURRENT_YEAR, teams: int = 12) -> None:
    """
    Checks whether the ADP CSV is stale and, if so, fetches fresh data and
    re-ingests the database. The staleness threshold is 2 days during draft
    season (Aug–early Sep) and 7 days the rest of the year.

    Failures are logged as warnings — a stale ADP file is better than a
    crashed app.
    """
    max_age = adp_max_age_days()
    season = "draft season" if max_age == 2 else "off-season"

    if not should_refresh(OUT_PATH, max_age):
        print(
            f"ADP data is current ({adp_age_str()}, refreshing every {max_age}d "
            f"during {season}) — skipping auto-refresh."
        )
        return

    print(
        f"ADP data is stale ({adp_age_str()}, threshold {max_age}d during "
        f"{season}) — fetching {year} PPR ADP ..."
    )
    try:
        players = await asyncio.to_thread(fetch_adp, year, teams)
        rows = build_csv_rows(players)
        await asyncio.to_thread(_write_and_report, rows, OUT_PATH)

        from backend.ingestion.ingest_players import ingest_csv
        count = await asyncio.to_thread(ingest_csv, str(OUT_PATH))
        print(f"ADP auto-refresh complete: {count} players loaded.")

        # ingest_csv() above did a full delete-and-reinsert of the Player
        # table, which wipes sleeper_id on every row. Re-sync immediately so
        # live Sleeper draft sync doesn't silently degrade to name-matching
        # until someone remembers to run this by hand.
        synced = False
        try:
            from backend.ingestion.sync_sleeper_ids import sync_sleeper_ids
            matched, unmatched = await sync_sleeper_ids()
            synced = True
            print(f"Sleeper ID sync complete: {matched} matched, {unmatched} unmatched.")
        except Exception as e:
            print(
                f"[WARN] Sleeper ID sync failed after ADP refresh: {e}. "
                "Player data is still current — live Sleeper draft sync will "
                "fall back to name-matching until this is re-run.",
                file=sys.stderr,
            )

        # Both relinks below read Player.sleeper_id to decide which rows
        # still have a valid owner and which to delete as orphans. The sync
        # that populates that column just failed, so every row would look
        # orphaned and both tables would be wiped. The repos raise
        # RelinkAborted on exactly this, but don't even ask — a skipped
        # relink leaves stale-but-recoverable FKs, which is strictly better
        # than an empty table.
        if not synced:
            print(
                "[WARN] Skipping the PlayerMetrics/DraftProfile relinks — they key "
                "off the sleeper_id crosswalk the failed sync above was meant to "
                "rebuild. Run `py -m backend.ingestion.reingest` once Sleeper is "
                "reachable; until then metrics may be attributed to the wrong player.",
                file=sys.stderr,
            )
            return

        # The same reingest above also reassigns every Player.id (delete +
        # reinsert), which silently invalidates PlayerMetrics.player_id —
        # confirmed live to misattribute one player's computed stats to a
        # different one. Repair it the same way sleeper_id is repaired
        # above: immediately, every time, rather than trusting someone to
        # remember. See metrics_repo.relink_player_ids' docstring.
        try:
            from backend.db import metrics_repo
            from backend.db.database import engine
            from sqlmodel import Session
            with Session(engine) as session:
                relinked, orphaned = metrics_repo.relink_player_ids(session)
            print(f"PlayerMetrics relink complete: {relinked} relinked, {orphaned} orphaned.")
        except Exception as e:
            print(
                f"[WARN] PlayerMetrics relink failed after ADP refresh: {e}. "
                "Existing metrics rows may now point at the wrong player until "
                "this is re-run — treat AI-generated analysis as unverified "
                "until it succeeds.",
                file=sys.stderr,
            )

        # Same player_id-instability problem, same fix, for DraftProfile
        # (see draft_profile_repo.relink_player_ids' docstring).
        try:
            from backend.db import draft_profile_repo
            from backend.db.database import engine
            from sqlmodel import Session
            with Session(engine) as session:
                relinked, orphaned = draft_profile_repo.relink_player_ids(session)
            print(f"DraftProfile relink complete: {relinked} relinked, {orphaned} orphaned.")
        except Exception as e:
            print(
                f"[WARN] DraftProfile relink failed after ADP refresh: {e}. "
                "Existing draft-profile rows may now point at the wrong player "
                "until this is re-run.",
                file=sys.stderr,
            )

    except httpx.HTTPStatusError as e:
        print(
            f"[WARN] ADP auto-refresh skipped: HTTP {e.response.status_code} "
            f"from FantasyFootballCalculator. Using existing data.",
            file=sys.stderr,
        )
    except httpx.RequestError as e:
        print(
            f"[WARN] ADP auto-refresh skipped: network error ({e}). "
            "Using existing data.",
            file=sys.stderr,
        )
    except ValueError as e:
        print(f"[WARN] ADP auto-refresh skipped: {e}", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] ADP auto-refresh failed unexpectedly: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch PPR ADP from FantasyFootballCalculator and update the player database."
    )
    parser.add_argument(
        "--year", type=int, default=CURRENT_YEAR,
        help=f"NFL season year (default: {CURRENT_YEAR})"
    )
    parser.add_argument(
        "--teams", type=int, default=12,
        help="League size used to filter ADP (default: 12)"
    )
    parser.add_argument(
        "--no-ingest", dest="no_ingest", action="store_true",
        help="Write the CSV but skip re-running ingest_players"
    )
    args = parser.parse_args()

    max_age = adp_max_age_days()
    print(f"Current ADP data: {adp_age_str()} (auto-refresh threshold: {max_age}d)")

    try:
        players = fetch_adp(year=args.year, teams=args.teams)
        print(f"Received {len(players)} players from FantasyFootballCalculator")

        rows = build_csv_rows(players)
        _write_and_report(rows, OUT_PATH)

        if args.no_ingest:
            print("Skipping database ingest (--no-ingest).")
            return

        print("Re-loading player database ...")
        from backend.ingestion.ingest_players import ingest_csv
        count = ingest_csv(str(OUT_PATH))
        print(f"Done. {count} players loaded into SQLite.")

        print("Re-syncing Sleeper player IDs (ingest wipes sleeper_id) ...")
        from backend.ingestion.sync_sleeper_ids import sync_sleeper_ids
        matched, unmatched = asyncio.run(sync_sleeper_ids())
        print(f"Done. {matched} matched, {unmatched} unmatched.")

        # Both relinks read Player.sleeper_id to decide which rows still
        # have an owner. A sync that matched nobody makes every row look
        # orphaned; the repos refuse to act on that (RelinkAborted), but
        # stopping here gives a clearer message than letting them.
        if matched == 0:
            print(
                "\nSleeper matched 0 players — not relinking, since every "
                "PlayerMetrics and DraftProfile row would look orphaned. "
                "Check Sleeper's API, then re-run "
                "`py -m backend.ingestion.reingest`.",
                file=sys.stderr,
            )
            sys.exit(1)

        from backend.db import draft_profile_repo, metrics_repo
        from backend.db.database import engine
        from backend.db.metrics_repo import RelinkAborted
        from sqlmodel import Session

        try:
            print("Relinking PlayerMetrics to the reassigned Player IDs ...")
            with Session(engine) as session:
                relinked, orphaned = metrics_repo.relink_player_ids(session)
            print(f"Done. {relinked} relinked, {orphaned} orphaned.")

            print("Relinking DraftProfile to the reassigned Player IDs ...")
            with Session(engine) as session:
                relinked, orphaned = draft_profile_repo.relink_player_ids(session)
            print(f"Done. {relinked} relinked, {orphaned} orphaned.")
        except RelinkAborted as e:
            print(f"\nRelink aborted, nothing deleted: {e}", file=sys.stderr)
            sys.exit(1)

    except httpx.HTTPStatusError as e:
        print(f"\nHTTP error: {e.response.status_code} {e.request.url}", file=sys.stderr)
        sys.exit(1)
    except httpx.RequestError as e:
        print(f"\nNetwork error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
