"""
Computes PlayerMetrics (opportunity/volume, efficiency, team context,
consistency & risk, forward-looking signals) from nflverse data via
nflreadpy, and upserts them into SQLite.

nflreadpy (not the deprecated nfl_data_py) is the data source — see
backend/db/models.py::PlayerMetrics for the field-by-field rationale.

Run manually:
    py -m backend.ingestion.fetch_metrics                # most recent season with real data (see _default_season)
    py -m backend.ingestion.fetch_metrics --season 2024   # explicit override, any past season
    py -m backend.ingestion.fetch_metrics --no-redzone    # skip the play-by-play pull (slow)

This is NOT part of main.py's startup auto-refresh the way fetch_adp.py is —
it pulls several full-season datasets (including play-by-play for red zone
touches) via nflreadpy, which is meaningfully heavier than the single small
ADP API call. Run it manually on the same cadence you'd want the analytics
refreshed (weekly during the season is plenty).

A note on verification: this was written without access to nflreadpy's live
data from the dev sandbox (network-restricted there, same as every other
external source this session) — it's built from nflreadpy's documented
functions and well-established nflverse column conventions, but the exact
column names are resolved defensively (a few plausible candidates tried per
field, see _first) rather than assumed to be exactly right. Run with
--verbose the first time and check the "resolved columns" log block — if
anything shows as unresolved, that's the thing to fix, not a sign the whole
approach is broken.

Author: Zach Cooper
"""

import argparse
import logging
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

from sqlmodel import Session, select

from backend.db.database import create_db_and_tables, engine
from backend.db import metrics_repo
from backend.db.models import Player

logger = logging.getLogger(__name__)


def _default_season(today: datetime | None = None) -> int:
    """
    Picks the most recent NFL season with real data to compute metrics from.

    NFL seasons are named for the year they kick off in (September) and run
    into February of the following calendar year. A season's nflverse data
    doesn't exist until Week 1 actually happens, so `datetime.now().year`
    is the wrong default for most of the calendar: draft prep in July 2026
    is drafting for the *2026* season, but the only real stats available are
    from the *2025* season (Sep 2025-Feb 2026), since 2026's games haven't
    been played yet. Confirmed live: running this in July defaulted to
    season=2026 and every nflreadpy call 404'd or hit the package's own
    "season must be <= 2025" validation — there's simply nothing there yet.

    Rule: January-August -> last calendar year (most recent completed/no
    current season). September-December -> this calendar year (the season
    that just kicked off, with data accumulating week by week).
    """
    today = today or datetime.now()
    return today.year if today.month >= 9 else today.year - 1


CURRENT_YEAR = _default_season()

# Trend window for "forward-looking" signals — last N weeks vs. season avg.
TREND_WINDOW_WEEKS = 3

# Standard PPR scoring, used only if nflverse's own fantasy_points_ppr column
# isn't present in a given row (it usually is — see _fantasy_points_ppr).
_PPR_WEIGHTS = {
    "receptions": 1.0,
    "receiving_yards": 0.1,
    "receiving_tds": 6.0,
    "rushing_yards": 0.1,
    "rushing_tds": 6.0,
    "passing_yards": 0.04,
    "passing_tds": 4.0,
    "interceptions": -2.0,
    "sack_fumbles_lost": -2.0,
    "rushing_fumbles_lost": -2.0,
    "receiving_fumbles_lost": -2.0,
}


# ---------------------------------------------------------------------------
# Column resolution — tolerant of nflverse's exact naming per field
# ---------------------------------------------------------------------------

def _first(row: dict, *candidates: str) -> Any:
    """Returns the first non-None value found under any of the candidate
    column names, or None if none of them are present in this row."""
    for c in candidates:
        if c in row and row[c] is not None:
            return row[c]
    return None


def _num(row: dict, *candidates: str, default: float = 0.0) -> float:
    val = _first(row, *candidates)
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _filter_regular_season(rows: list[dict]) -> list[dict]:
    """
    Keeps only regular-season rows, dropping preseason/postseason.

    nflreadpy's week-level loaders (load_player_stats, load_snap_counts,
    load_team_stats, load_pbp) return every season_type mixed together in
    one table when called at week granularity — nothing filters this for
    you. Every "season stats" figure this module computes (yards per
    target, target share, fantasy points per game, etc.) is a sum or
    average across whatever rows it's handed, so postseason games quietly
    inflate/deflate those numbers relative to the regular-season-only
    figures sites like ESPN report by default.

    Confirmed live: a user found Jahmyr Gibbs's computed yards_per_target
    (7.96) didn't match ESPN's regular-season figure (6.55). Root cause was
    exactly this — his team made the playoffs, and those extra games got
    summed into the same season totals with no way to separate them back
    out after the fact.

    "REG" is nflverse's standard season_type value; matched
    case-insensitively since it's not worth breaking over capitalization.
    Rows missing season_type entirely are kept rather than dropped — an
    unrecognized/absent column is a "can't tell" situation, and silently
    discarding otherwise-valid data on a defensive assumption would just
    trade one kind of wrong number for another.

    Deliberately NOT applied to injuries or depth charts (see
    refresh_metrics) — a playoff injury is real, relevant risk signal for
    next season, and depth-chart movement isn't a summed ratio that
    "regular season only" convention applies to in the same way.
    """
    filtered = []
    for row in rows:
        season_type = _first(row, "season_type")
        if season_type is None or str(season_type).strip().upper() == "REG":
            filtered.append(row)
    return filtered


# ---------------------------------------------------------------------------
# Data loading — isolated so a failure in one source doesn't kill the rest
# ---------------------------------------------------------------------------

def _load_dicts(loader_name: str, *args, **kwargs) -> list[dict]:
    """
    Calls an nflreadpy loader by name and converts the result to a list of
    plain dicts. Isolated behind a name lookup (rather than importing the
    function directly at module level) so one missing/renamed function in a
    future nflreadpy version doesn't prevent this module from importing at
    all — see refresh_metrics()'s per-source try/except for how failures
    here are handled.
    """
    import nflreadpy as nfl

    fn = getattr(nfl, loader_name)
    df = fn(*args, **kwargs)
    return df.to_dicts()


def _load_id_crosswalk() -> dict[str, str]:
    """Returns {gsis_id: sleeper_id} from nflreadpy's fantasy-platform ID
    crosswalk (load_ff_playerids) — this is what lets us match nflverse's
    stats (keyed by gsis_id) back to our Player rows (keyed by sleeper_id,
    already populated by sync_sleeper_ids.py)."""
    rows = _load_dicts("load_ff_playerids")
    crosswalk: dict[str, str] = {}
    for row in rows:
        gsis_id = _first(row, "gsis_id")
        sleeper_id = _first(row, "sleeper_id")
        if gsis_id and sleeper_id:
            crosswalk[str(gsis_id)] = str(sleeper_id)
    logger.info(f"ID crosswalk: {len(crosswalk):,} gsis_id -> sleeper_id mappings")
    return crosswalk


# ---------------------------------------------------------------------------
# Per-category computation
# ---------------------------------------------------------------------------

def _group_by_player(rows: list[dict], id_field_candidates: tuple[str, ...]) -> dict[str, list[dict]]:
    """Groups arbitrary nflverse rows by gsis-style player ID."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pid = _first(row, *id_field_candidates)
        if pid:
            grouped[str(pid)].append(row)
    return grouped


def _fantasy_points_ppr(row: dict) -> float:
    """Prefers nflverse's own precomputed PPR column; falls back to a
    manual calculation from base stats if that column isn't present."""
    precomputed = _first(row, "fantasy_points_ppr", "ppr_points")
    if precomputed is not None:
        try:
            return float(precomputed)
        except (TypeError, ValueError):
            pass
    return sum(_num(row, stat) * weight for stat, weight in _PPR_WEIGHTS.items())


# Minimum season air-yard total before RACR is computed at all. Below this
# the denominator is either negative (RB screen/checkdown usage) or so small
# that the ratio is dominated by rounding — see the comment at the "racr" key
# below. 50 yards is roughly a handful of downfield targets; anyone with a
# genuine receiving role clears it easily.
_MIN_AIR_YARDS_FOR_RACR = 50.0


def _compute_opportunity_efficiency(weeks: list[dict]) -> dict:
    games = len(weeks)
    if games == 0:
        return {}

    targets = [_num(w, "targets") for w in weeks]
    carries = [_num(w, "carries", "rushing_attempts") for w in weeks]
    rec_yards = [_num(w, "receiving_yards") for w in weeks]
    rush_yards = [_num(w, "rushing_yards") for w in weeks]
    receptions = [_num(w, "receptions") for w in weeks]
    air_yards = [_num(w, "receiving_air_yards", "air_yards") for w in weeks]
    yac = [_num(w, "receiving_yards_after_catch", "yac") for w in weeks]
    team_targets = [_num(w, "team_targets") for w in weeks]
    team_carries = [_num(w, "team_carries", "team_rushing_attempts") for w in weeks]

    sum_targets, sum_carries = sum(targets), sum(carries)
    sum_air_yards = sum(air_yards)
    sum_rec_yards = sum(rec_yards)

    return {
        "targets_per_game": sum_targets / games,
        "carries_per_game": sum_carries / games,
        "yards_per_target": (sum_rec_yards / sum_targets) if sum_targets else None,
        "yards_per_carry": (sum(rush_yards) / sum_carries) if sum_carries else None,
        "yac_per_reception": (sum(yac) / sum(receptions)) if sum(receptions) else None,
        # RACR (receiving yards / air yards) is a WR/TE metric and is
        # undefined for players whose targets come at or behind the line of
        # scrimmage. Running backs catch screens, checkdowns and dumpoffs,
        # whose depth of target is negative, so their season air-yard total
        # is often negative or a handful of yards — which turned this
        # division into nonsense for 50 of 57 RBs in the DB (TreVeyon
        # Henderson: 221 receiving yards on -1.0 air yards = RACR -221).
        #
        # The old guard only rejected exactly 0. Requiring a meaningful
        # positive denominator rejects both the negative case and the
        # tiny-denominator case, where a 1-yard air total inflates RACR by
        # two orders of magnitude. None means "not a meaningful stat for
        # this player," which the prompt already renders as absence rather
        # than as zero.
        "racr": (
            (sum_rec_yards / sum_air_yards)
            if sum_air_yards >= _MIN_AIR_YARDS_FOR_RACR
            else None
        ),
        "catch_rate": (sum(receptions) / sum_targets) if sum_targets else None,
        "target_share": (sum_targets / sum(team_targets)) if sum(team_targets) else None,
        "carry_share": (sum_carries / sum(team_carries)) if sum(team_carries) else None,
    }


def _compute_consistency_risk(weeks: list[dict], injury_weeks: list[dict]) -> dict:
    games = len(weeks)
    points = [_fantasy_points_ppr(w) for w in weeks]

    result = {
        "fantasy_points_avg": (sum(points) / games) if games else None,
        "fantasy_points_stdev": statistics.pstdev(points) if len(points) > 1 else None,
        "injury_report_appearances": len(injury_weeks),
    }

    # games_missed: weeks where the player was on the injury report with an
    # "Out" (or equivalent) designation. Column name and exact status
    # strings are the least certain part of this module — resolve
    # defensively and treat anything unrecognized as "not a miss" rather
    # than over-counting.
    out_statuses = {"out", "ir", "injured reserve", "pup", "suspended"}
    missed = 0
    for w in injury_weeks:
        status = _first(w, "report_status", "game_status", "status")
        if status and str(status).strip().lower() in out_statuses:
            missed += 1
    result["games_missed"] = missed

    return result


def _compute_forward_looking(
    weeks: list[dict],
    snap_weeks: list[dict],
    depth_chart_weeks: list[dict],
) -> dict:
    """
    Trend = last TREND_WINDOW_WEEKS minus season average, for target share
    and snap share. Positive means an increasing role recently — exactly
    the kind of signal ADP (which lags) won't capture yet.

    Note: snap_weeks must be the player's rows from load_snap_counts, not
    player_stats — snap % lives in a separate nflverse dataset, it's not a
    column on the weekly stats rows passed in as `weeks`.
    """
    result: dict[str, Optional[float]] = {
        "target_share_trend": None,
        "snap_pct_trend": None,
        "depth_chart_trend": None,
    }

    weeks_sorted = sorted(weeks, key=lambda w: _num(w, "week"))
    if len(weeks_sorted) >= 2:
        recent = weeks_sorted[-TREND_WINDOW_WEEKS:]

        season_targets = sum(_num(w, "targets") for w in weeks_sorted)
        season_team_targets = sum(_num(w, "team_targets") for w in weeks_sorted)
        recent_targets = sum(_num(w, "targets") for w in recent)
        recent_team_targets = sum(_num(w, "team_targets") for w in recent)

        season_share = (season_targets / season_team_targets) if season_team_targets else None
        recent_share = (recent_targets / recent_team_targets) if recent_team_targets else None
        if season_share is not None and recent_share is not None:
            result["target_share_trend"] = recent_share - season_share

    snap_weeks_sorted = sorted(snap_weeks, key=lambda w: _num(w, "week"))
    if len(snap_weeks_sorted) >= 2:
        recent_snap_weeks = snap_weeks_sorted[-TREND_WINDOW_WEEKS:]
        season_snaps = [_num(w, "offense_pct", "snap_pct") for w in snap_weeks_sorted if _first(w, "offense_pct", "snap_pct") is not None]
        recent_snaps = [_num(w, "offense_pct", "snap_pct") for w in recent_snap_weeks if _first(w, "offense_pct", "snap_pct") is not None]
        if season_snaps and recent_snaps:
            result["snap_pct_trend"] = (sum(recent_snaps) / len(recent_snaps)) - (sum(season_snaps) / len(season_snaps))

    dc_sorted = sorted(depth_chart_weeks, key=lambda w: _num(w, "week"))
    if len(dc_sorted) >= 2:
        first_rank = _first(dc_sorted[0], "depth_team", "depth_position", "rank")
        last_rank = _first(dc_sorted[-1], "depth_team", "depth_position", "rank")
        try:
            if first_rank is not None and last_rank is not None:
                result["depth_chart_trend"] = int(last_rank) - int(first_rank)
        except (TypeError, ValueError):
            pass

    return result


def _compute_red_zone_touches(pbp_rows: list[dict]) -> dict[str, float]:
    """
    Red zone touches per game, keyed by gsis_id. Requires play-by-play data
    (the one heavy pull in this module — see --no-redzone to skip it).
    Standard definition: yardline_100 <= 20 (i.e. inside the opponent's
    20-yard line).
    """
    touches_by_player: dict[str, list[int]] = defaultdict(list)
    games_by_player: dict[str, set] = defaultdict(set)

    for play in pbp_rows:
        yardline = _first(play, "yardline_100")
        if yardline is None or float(yardline) > 20:
            continue

        game_id = _first(play, "game_id")
        rusher = _first(play, "rusher_player_id", "rusher_id")
        receiver = _first(play, "receiver_player_id", "receiver_id")

        if rusher:
            touches_by_player[str(rusher)].append(1)
            games_by_player[str(rusher)].add(game_id)
        if receiver and _first(play, "pass_attempt"):
            touches_by_player[str(receiver)].append(1)
            games_by_player[str(receiver)].add(game_id)

    return {
        pid: len(touches) / max(len(games_by_player[pid]), 1)
        for pid, touches in touches_by_player.items()
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def refresh_metrics(season: int = CURRENT_YEAR, include_redzone: bool = True) -> int:
    """
    Pulls nflverse data via nflreadpy, computes PlayerMetrics for every
    player we can match to a local Player row (via sleeper_id), and upserts
    the results. Returns the number of players updated.

    Each data source is wrapped independently — a failure pulling, say,
    depth charts shouldn't prevent opportunity/efficiency metrics (which
    only need player_stats) from still being computed and saved. Draft day
    is not the time for one flaky source to take down the whole refresh.
    """
    crosswalk = _load_id_crosswalk()
    if not crosswalk:
        logger.error("Empty ID crosswalk — cannot match any nflverse rows to local players. Aborting.")
        return 0

    stats_by_player: dict[str, list[dict]] = {}
    snaps_by_player: dict[str, list[dict]] = {}
    injuries_by_player: dict[str, list[dict]] = {}
    depth_by_player: dict[str, list[dict]] = {}
    redzone_by_player: dict[str, float] = {}
    team_pass_rate: dict[str, float] = {}

    try:
        stats_rows = _load_dicts("load_player_stats", seasons=season, summary_level="week")
        stats_rows = _filter_regular_season(stats_rows)
        stats_by_player = _group_by_player(stats_rows, ("player_id", "gsis_id"))
        logger.info(f"player_stats: {len(stats_rows):,} regular-season rows, {len(stats_by_player):,} players")
    except Exception as e:
        logger.warning(f"Could not load player_stats — opportunity/efficiency/consistency metrics will be skipped: {e}")

    try:
        snap_rows = _load_dicts("load_snap_counts", seasons=season)
        snap_rows = _filter_regular_season(snap_rows)
        snaps_by_player = _group_by_player(snap_rows, ("gsis_id", "player_id", "pfr_player_id"))
    except Exception as e:
        logger.warning(f"Could not load snap_counts — snap_pct will be unavailable: {e}")

    try:
        injury_rows = _load_dicts("load_injuries", seasons=season)
        injuries_by_player = _group_by_player(injury_rows, ("gsis_id", "player_id"))
    except Exception as e:
        logger.warning(f"Could not load injuries — injury-based risk metrics will be skipped: {e}")

    try:
        depth_rows = _load_dicts("load_depth_charts", seasons=season)
        depth_by_player = _group_by_player(depth_rows, ("gsis_id", "player_id"))
    except Exception as e:
        logger.warning(f"Could not load depth_charts — depth-chart signals will be skipped: {e}")

    try:
        team_rows = _load_dicts("load_team_stats", seasons=season, summary_level="week")
        team_rows = _filter_regular_season(team_rows)
        pass_by_team: dict[str, list[float]] = defaultdict(list)
        for row in team_rows:
            team = _first(row, "team")
            pass_att = _num(row, "attempts", "pass_attempts")
            rush_att = _num(row, "carries", "rushing_attempts")
            total = pass_att + rush_att
            if team and total:
                pass_by_team[team].append(pass_att / total)
        team_pass_rate = {t: sum(v) / len(v) for t, v in pass_by_team.items()}
    except Exception as e:
        logger.warning(f"Could not load team_stats — team_pass_rate will be unavailable: {e}")

    if include_redzone:
        try:
            pbp_rows = _load_dicts("load_pbp", seasons=season)
            pbp_rows = _filter_regular_season(pbp_rows)
            redzone_by_player = _compute_red_zone_touches(pbp_rows)
            logger.info(f"Red zone touches computed for {len(redzone_by_player):,} players")
        except Exception as e:
            logger.warning(f"Could not load play-by-play — red_zone_touches_per_game will be unavailable: {e}")
    else:
        logger.info("Skipping play-by-play pull (--no-redzone) — red_zone_touches_per_game will be unavailable")

    through_week = max(
        (int(_num(w, "week")) for weeks in stats_by_player.values() for w in weeks),
        default=0,
    )

    updated = 0
    with Session(engine) as session:
        for gsis_id, weeks in stats_by_player.items():
            sleeper_id = crosswalk.get(gsis_id)
            if not sleeper_id:
                continue

            player = session.exec(select(Player).where(Player.sleeper_id == sleeper_id)).first()
            if player is None:
                continue

            snap_weeks = snaps_by_player.get(gsis_id, [])
            depth_weeks = depth_by_player.get(gsis_id, [])

            fields: dict = {
                "season": season,
                "through_week": through_week,
                "games_played": len(weeks),
            }
            fields.update(_compute_opportunity_efficiency(weeks))
            fields.update(_compute_consistency_risk(weeks, injuries_by_player.get(gsis_id, [])))
            fields.update(_compute_forward_looking(weeks, snap_weeks, depth_weeks))

            if snap_weeks:
                pcts = [_num(w, "offense_pct", "snap_pct") for w in snap_weeks]
                fields["snap_pct"] = sum(pcts) / len(pcts) if pcts else None

            if gsis_id in redzone_by_player:
                fields["red_zone_touches_per_game"] = redzone_by_player[gsis_id]

            if player.team in team_pass_rate:
                fields["team_pass_rate"] = team_pass_rate[player.team]

            if depth_weeks:
                # Sort by week rather than trusting input order — nflverse
                # doesn't guarantee row order within a player's group.
                latest = max(depth_weeks, key=lambda w: _num(w, "week"))
                latest_rank = _first(latest, "depth_team", "depth_position", "rank")
                try:
                    fields["depth_chart_rank"] = int(latest_rank) if latest_rank is not None else None
                except (TypeError, ValueError):
                    pass

            metrics_repo.upsert_metrics(session, player_id=player.id, sleeper_id=sleeper_id, **fields)
            updated += 1

    logger.info(f"PlayerMetrics refresh complete: {updated} players updated.")
    return updated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute PlayerMetrics from nflverse data (via nflreadpy) and load into SQLite."
    )
    parser.add_argument("--season", type=int, default=CURRENT_YEAR, help=f"NFL season year (default: {CURRENT_YEAR})")
    parser.add_argument(
        "--no-redzone", dest="no_redzone", action="store_true",
        help="Skip the play-by-play pull (slowest step) — red_zone_touches_per_game will be unavailable",
    )
    parser.add_argument("--verbose", action="store_true", help="Log at DEBUG level")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    create_db_and_tables()
    count = refresh_metrics(season=args.season, include_redzone=not args.no_redzone)
    if count == 0:
        print("No players updated — check the warnings above for which data source(s) failed.", file=sys.stderr)
        sys.exit(1)
    print(f"Done. {count} players' metrics updated for season {args.season}.")


if __name__ == "__main__":
    main()
