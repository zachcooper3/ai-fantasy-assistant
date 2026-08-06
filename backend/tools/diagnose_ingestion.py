"""
Reports exactly which nflverse sources load, which fail and why, and — for
the ones that load — whether the columns fetch_metrics.py actually reads are
present in the data.

This exists because ingestion failures here are silent by design. Every
source in refresh_metrics() is wrapped in its own try/except so one flaky
feed can't take down a draft-day refresh, which is the right call — but the
consequence is a warning line in a log nobody reads and a PlayerMetrics
column that stays NULL forever. Confirmed live: snap_pct, target_share,
carry_share, depth_chart_rank and all three trend fields are NULL for all
182 players, and the only evidence was one line during ingestion.

A missing column is even quieter than a failed download. `_num()` returns
0.0 for an absent field, so a share computed from a column that doesn't
exist becomes 0/0 and is stored as None — indistinguishable from "this
player genuinely has no data" at every layer above.

Read-only: loads data, prints, writes nothing.

Run:
    py -m backend.tools.diagnose_ingestion
    py -m backend.tools.diagnose_ingestion --season 2024
"""

import argparse
import logging
import re
import sys

logger = logging.getLogger(__name__)

# (loader, kwargs, what breaks without it) — mirrors refresh_metrics().
_SOURCES: list[tuple[str, dict, str]] = [
    ("load_ff_playerids", {},
     "EVERYTHING — this is the gsis_id -> sleeper_id crosswalk"),
    ("load_player_stats", {"summary_level": "week"},
     "opportunity, efficiency, consistency (ppg, targets, carries)"),
    ("load_snap_counts", {}, "snap_pct, snap_pct_trend"),
    ("load_injuries", {}, "injury_report_appearances, games_missed"),
    ("load_depth_charts", {}, "depth_chart_rank, depth_chart_trend"),
    ("load_team_stats", {"summary_level": "week"}, "team_pass_rate"),
    ("load_pbp", {}, "red_zone_touches_per_game"),
]

# Columns fetch_metrics.py reads, by source. Each entry is the candidate list
# passed to _first()/_num() — present if ANY candidate resolves.
_EXPECTED: dict[str, dict[str, tuple[str, ...]]] = {
    "load_player_stats": {
        "player id":      ("player_id", "gsis_id"),
        "targets":        ("targets",),
        "carries":        ("carries", "rushing_attempts"),
        "receiving yds":  ("receiving_yards",),
        "rushing yds":    ("rushing_yards",),
        "receptions":     ("receptions",),
        "air yards":      ("receiving_air_yards", "air_yards"),
        "YAC":            ("receiving_yards_after_catch", "yac"),
        "TEAM targets":   ("team_targets",),
        "TEAM carries":   ("team_carries", "team_rushing_attempts"),
        "PPR points":     ("fantasy_points_ppr", "ppr_points"),
        "season_type":    ("season_type",),
    },
    "load_snap_counts": {
        "player id":  ("gsis_id", "player_id", "pfr_player_id"),
        "snap pct":   ("offense_pct", "offense_snaps_pct", "snap_pct"),
        "season_type": ("season_type",),
    },
    "load_depth_charts": {
        "player id":   ("gsis_id", "player_id"),
        "depth rank":  ("depth_team", "depth_position", "rank"),
    },
    "load_injuries": {
        "player id":   ("gsis_id", "player_id"),
        "status":      ("report_status", "game_status", "status"),
    },
    "load_team_stats": {
        "team":        ("team", "recent_team", "team_abbr"),
        "pass atts":   ("attempts", "passing_attempts"),
        "rush atts":   ("carries", "rushing_attempts"),
    },
}


def _tidy(exc: Exception, limit: int = 240) -> str:
    """
    Readable one-liner from an nflreadpy download error.

    Those exceptions embed a GitHub signed release URL — several hundred
    characters of JWT and SAS query string — which buries the one fact that
    matters (404 vs proxy vs timeout) under noise. The path is kept because
    it names the dataset and season; everything after '?' is dropped.
    """
    msg = " ".join(str(exc).split())
    msg = re.sub(r"(https?://\S+?)\?\S+", r"\1?…", msg)
    return msg if len(msg) <= limit else msg[:limit] + " …"


def _check(season: int) -> int:
    try:
        import nflreadpy  # noqa: F401
    except ImportError:
        print("nflreadpy is not installed — run: pip install nflreadpy")
        return 1

    from backend.ingestion.fetch_metrics import _load_dicts

    print("=" * 74)
    print(f"nflverse ingestion diagnostic — season {season}")
    print("=" * 74)

    loaded: dict[str, list[dict]] = {}
    failures = 0

    for name, kwargs, breaks in _SOURCES:
        call = dict(kwargs)
        if name != "load_ff_playerids":
            call["seasons"] = season
        try:
            rows = _load_dicts(name, **call)
            loaded[name] = rows
            print(f"\n  OK    {name}  ({len(rows):,} rows)")
        except Exception as e:
            failures += 1
            print(f"\n  FAIL  {name}")
            print(f"        {type(e).__name__}: {_tidy(e)}")
            print(f"        -> silently disables: {breaks}")

    print("\n" + "=" * 74)
    print("COLUMN CHECK — a missing column is as damaging as a failed download,")
    print("and quieter: _num() returns 0.0 for an absent field, so the metric")
    print("computes to 0/0 and stores as None.")
    print("=" * 74)

    for source, expected in _EXPECTED.items():
        rows = loaded.get(source)
        if not rows:
            print(f"\n  {source}: not loaded, skipping")
            continue
        actual = set(rows[0].keys())
        print(f"\n  {source}  ({len(actual)} columns)")
        missing = False
        for label, candidates in expected.items():
            hit = next((c for c in candidates if c in actual), None)
            if hit:
                print(f"      OK      {label:<16} -> {hit}")
            else:
                missing = True
                print(f"      MISSING {label:<16} -> tried {', '.join(candidates)}")
                near = [c for c in sorted(actual)
                        if any(k in c for k in candidates[0].split("_"))][:6]
                if near:
                    print(f"                                 similar: {', '.join(near)}")
        # A "similar:" hint is only useful when the real name shares a word
        # with the one we guessed. When a dataset's schema has been reworked
        # upstream it usually doesn't, so dump the lot — for a source with a
        # handful of columns that is far more useful than a near-miss list,
        # and it is the difference between one round trip and three.
        if missing and len(actual) <= 40:
            print(f"      ALL COLUMNS: {', '.join(sorted(actual))}")
            sample = {k: rows[0][k] for k in sorted(actual)[:8]}
            print(f"      SAMPLE ROW:  {sample}")

    print("\n" + "=" * 74)
    print(f"{failures} source(s) failed to load.")
    print("Any MISSING column above explains a permanently-NULL PlayerMetrics")
    print("field — fix by adding the real name to the candidate tuple in")
    print("fetch_metrics.py, which already accepts several aliases per field.")
    print("=" * 74)
    return 1 if failures else 0


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    from backend.ingestion.fetch_metrics import CURRENT_YEAR

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=CURRENT_YEAR,
                    help=f"NFL season to test (default: {CURRENT_YEAR})")
    args = ap.parse_args()
    sys.exit(_check(args.season))


if __name__ == "__main__":
    main()
