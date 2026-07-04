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

FFC_URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr"
OUT_PATH = Path("data/raw/fantasypros_adp.csv")
CURRENT_YEAR = datetime.now().year

# Positions to include — matches what ingest_players.py and the app expect
VALID_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF"}


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
        await asyncio.to_thread(write_csv, rows, OUT_PATH)

        from backend.ingestion.ingest_players import ingest_csv
        count = await asyncio.to_thread(ingest_csv, str(OUT_PATH))
        print(f"ADP auto-refresh complete: {count} players loaded.")

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
        write_csv(rows, OUT_PATH)

        if args.no_ingest:
            print("Skipping database ingest (--no-ingest).")
            return

        print("Re-loading player database ...")
        from backend.ingestion.ingest_players import ingest_csv
        count = ingest_csv(str(OUT_PATH))
        print(f"Done. {count} players loaded into SQLite.")

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
