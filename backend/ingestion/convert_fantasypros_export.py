"""
Converts a FantasyPros "Overall ADP Rankings" CSV export (the file you
download from fantasypros.com/nfl/adp/overall.php) into the shape
ingest_players.py expects: Rank, Player, Team, Bye, POS, AVG.

Why this exists: the raw export packs name, team, and bye into one column
("Jahmyr Gibbs   DET (6)") and formats defenses differently again
("Houston Texans DST   (8)" — full team name, not a code). A straight copy
onto data/raw/fantasypros_adp.csv silently ingests every player with a
blank name/team, no crash, garbage data. This does the actual parsing:

  - Regular players: "Name   TEAM (Bye)" -> Player/Team/Bye split on the
    trailing " CODE (N)" pattern.
  - Defenses: "City Mascot DST   (Bye)" -> team NAME mapped to the same
    abbreviation this app uses everywhere else (see _TEAM_NAME_TO_CODE;
    cross-checked against the live Player.team column, not guessed).
  - Rows with neither (a handful of inactive/free-agent players near the
    bottom of the sheet, e.g. no current team on file) -> Team="", Bye=None,
    kept rather than dropped, since ADP/rank/POS are still real.

POS and AVG need no conversion — this export already uses "DST" (not FFC's
"DEF") and already has an AVG column with the cross-site consensus.

Run:
    py -m backend.ingestion.convert_fantasypros_export "FantasyPros_2026_Overall_ADP_Rankings.csv"
    py -m backend.ingestion.convert_fantasypros_export IN.csv --out data/raw/fantasypros_adp.csv

Author: Zach Cooper
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Reused rather than duplicated: recommendations.py already maps historical/
# alternative team codes (JAC, WSH, OAK, ...) onto the convention Player.team
# actually uses. Confirmed needed here — FantasyPros' export labels
# Jacksonville "JAC" everywhere, while every other source and Player.team
# itself use "JAX"; without this, all 14 Jaguars rows would silently fail
# every downstream team-keyed join (Game, roster changes, OPP tags).
from backend.app.api.recommendations import _TEAM_ALIASES

# Cross-checked against the live Player.team column (`select distinct team
# from player`), not guessed — a mismatch here means a defense silently
# fails to join against PlayerMetrics/Game/roster-change data downstream,
# the exact class of bug _TEAM_ALIASES in recommendations.py exists to
# catch (see that file's "left_only" abbreviation-mismatch warning).
_TEAM_NAME_TO_CODE = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

# "Name   TEAM (Bye)" — two-plus spaces before the code (the export pads
# with a fixed-width gap), an all-caps 2-3 letter code, then "(N)".
_PLAYER_RE = re.compile(r"^(?P<name>.+?)\s{2,}(?P<team>[A-Z]{2,3})\s+\((?P<bye>\d+)\)\s*$")
# "City Mascot DST   (Bye)" — same trailing "(N)", but the token before it
# is the literal word DST, and everything before THAT is the full team name.
_DST_RE = re.compile(r"^(?P<team_name>.+?)\s+DST\s{2,}\(\s*(?P<bye>\d+)\s*\)\s*$")


def parse_player_bye(raw: str) -> tuple[str, str, int | None]:
    """
    Splits the export's combined "Player (Bye)" field into (name, team, bye).

    Falls back to (raw, "", None) for the rows that carry neither a team
    nor a bye at all — a handful of inactive/free-agent veterans near the
    bottom of a 600-row sheet (confirmed on the 2026 export: Kareem Hunt,
    Joe Mixon, Justin Tucker, Philip Rivers). Their rank/POS/AVG are still
    real ADP data, so they're kept rather than dropped; Team="" satisfies
    the NOT NULL column without asserting a team that isn't there.
    """
    raw = raw.strip()

    m = _DST_RE.match(raw)
    if m:
        team_name = m.group("team_name").strip()
        code = _TEAM_NAME_TO_CODE.get(team_name)
        if code is None:
            raise ValueError(
                f"Unmapped defense team name {team_name!r} — add it to "
                "_TEAM_NAME_TO_CODE before trusting this row."
            )
        return f"{team_name} DST", code, int(m.group("bye"))

    m = _PLAYER_RE.match(raw)
    if m:
        code = m.group("team")
        return m.group("name").strip(), _TEAM_ALIASES.get(code, code), int(m.group("bye"))

    return raw, "", None


def convert_rows(reader: csv.DictReader) -> list[dict]:
    out = []
    skipped = 0
    for row in reader:
        raw_player = row.get("Player (Bye)") or row.get("Player") or ""
        pos = (row.get("POS") or "").strip()
        avg = (row.get("AVG") or "").strip()
        rank = (row.get("Rank") or "").strip()
        if not raw_player or not pos or not avg or not rank:
            skipped += 1
            continue
        name, team, bye = parse_player_bye(raw_player)
        out.append({
            "Rank": rank,
            "Player": name,
            "Team": team,
            "Bye": bye if bye is not None else "",
            "POS": pos,
            "AVG": avg,
        })
    if skipped:
        print(f"[WARN] Skipped {skipped} row(s) missing Rank/Player/POS/AVG.", file=sys.stderr)
    return out


def convert(in_path: Path, out_path: Path) -> int:
    with open(in_path, newline="", encoding="utf-8-sig") as f:
        rows = convert_rows(csv.DictReader(f))

    no_team = [r for r in rows if not r["Team"]]
    if no_team:
        print(
            f"[INFO] {len(no_team)} row(s) had no parseable team/bye "
            f"(kept anyway, Team=\"\"): "
            + ", ".join(r["Player"] for r in no_team[:10])
            + (", ..." if len(no_team) > 10 else "")
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Rank", "Player", "Team", "Bye", "POS", "AVG"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} players -> {out_path}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a raw FantasyPros ADP export into ingest_players.py's expected CSV shape."
    )
    parser.add_argument("input", help="Path to the raw FantasyPros export CSV.")
    parser.add_argument(
        "--out", default="data/raw/fantasypros_adp.csv",
        help="Where to write the converted CSV (default: data/raw/fantasypros_adp.csv, "
             "the same path fetch_adp.py/ingest_players.py already use).",
    )
    args = parser.parse_args()
    convert(Path(args.input), Path(args.out))


if __name__ == "__main__":
    main()
