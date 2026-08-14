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

A note on column names: these were originally guessed from nflverse
conventions without access to live data, and several were wrong in ways that
produced permanently-NULL fields rather than errors — player_stats has no
team_targets/team_carries at all, snap_counts is keyed by pfr_player_id with
no gsis_id and uses game_type rather than season_type, and depth_charts was
reworked upstream to pos_rank on a dt-keyed snapshot feed. All four are now
resolved against the live 2025 release.

Every field is still looked up through _first() with several candidates, so
an upstream rename degrades to a NULL rather than a crash. That tolerance is
also what hid the four bugs above, so after any nflverse update run:

    py -m backend.tools.diagnose_ingestion

which checks every column this module reads against what the data actually
contains and dumps the real column list when one is missing.

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
        # snap_counts has no `season_type` column — it uses `game_type`
        # (confirmed live: 16 columns, game_type present, season_type not).
        # Rows lacking the field entirely are kept, so before this alias every
        # postseason snap was silently summed into snap_pct.
        season_type = _first(row, "season_type", "game_type")
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


def _load_pfr_crosswalk() -> dict[str, str]:
    """
    Returns {pfr_id: gsis_id} from load_ff_playerids.

    Needed because load_snap_counts is keyed by `pfr_player_id` and carries
    no gsis_id at all (confirmed live: 16 columns, pfr_player_id present,
    gsis_id absent). Everything else in this module is keyed by gsis_id, so
    without a translation step the snap rows group under PFR ids and the
    per-player lookup — which asks for a gsis_id — matches nobody. That is
    why snap_pct and snap_pct_trend were NULL for all 182 players even
    though the download succeeded: the data arrived and was then silently
    filed under keys nothing ever asks for.

    Column name is resolved defensively; ff_playerids is a many-platform
    crosswalk and the PFR field has been spelled a few ways.
    """
    rows = _load_dicts("load_ff_playerids")
    crosswalk: dict[str, str] = {}
    for row in rows:
        pfr_id = _first(row, "pfr_id", "pfr_player_id", "pfrid")
        gsis_id = _first(row, "gsis_id")
        if pfr_id and gsis_id:
            crosswalk[str(pfr_id)] = str(gsis_id)
    if not crosswalk:
        logger.warning(
            "No pfr_id -> gsis_id mappings found in load_ff_playerids — snap "
            "counts cannot be matched to players and snap_pct will stay NULL. "
            "Check the PFR column name in that dataset."
        )
    else:
        logger.info(f"PFR crosswalk: {len(crosswalk):,} pfr_id -> gsis_id mappings")
    return crosswalk


def _order_key(row: dict) -> tuple:
    """
    Sort key for a player's weekly rows, tolerant of datasets that have no
    week number.

    load_depth_charts is a snapshot feed keyed by `dt` (an ISO-8601
    timestamp), not a weekly table — so sorting it by "week" put every row
    at 0.0 and made "first vs last" an arbitrary pick, which is what
    depth_chart_trend was measuring. ISO-8601 sorts correctly as a string,
    so no parsing is needed.
    """
    week = _first(row, "week")
    if week is not None:
        try:
            return (0, float(week))
        except (TypeError, ValueError):
            pass
    dt = _first(row, "dt", "updated", "timestamp")
    return (1, str(dt)) if dt is not None else (2, "")


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

# RACR is a receiver metric and is only computed for receivers.
#
# A floor on the denominator turned out not to be enough. A running back can
# accumulate 50+ season air yards while still catching essentially everything
# at or behind the line of scrimmage, because air yards on screens are
# negative and partially cancel the downfield ones — leaving a tiny positive
# total under a large receiving-yard numerator. Confirmed live on the 2025
# data AFTER the floor was already in place: all 7 RBs who cleared it had
# implausible values, and they were the highest-ADP backs on the board —
# Gibbs 11.41, Robinson 8.12, Cook 5.20, McCaffrey 2.76, against a real-world
# range of roughly 0.5-1.5.
#
# The floor was treating this as a sample-size problem. It isn't: RACR is
# undefined for a usage pattern built on targets at or behind the line, no
# matter how many of them there are. So gate on position, which is what the
# stat actually depends on.
_RACR_POSITIONS = {"WR", "TE"}

# Even within WR/TE, a ratio this high means the air-yards denominator is
# broken rather than that the player is extraordinary — a receiver cannot
# generate three times his air yards over a full season. Catches the 8
# remaining implausible WR/TE values (Greg Dortch 3.55 and friends), all of
# them low-volume players whose denominators are small enough to be noise.
_MAX_PLAUSIBLE_RACR = 3.0


def _racr(sum_rec_yards: float, sum_air_yards: float, position: str | None) -> float | None:
    """Receiving yards / air yards, or None when that ratio isn't meaningful.

    None means "not a meaningful stat for this player," which the prompt
    already renders as absence rather than as zero — so a suppressed RACR
    costs nothing, while a wrong one is a number Claude will happily reason
    from out loud.
    """
    if position is not None and position.upper() not in _RACR_POSITIONS:
        return None
    if sum_air_yards < _MIN_AIR_YARDS_FOR_RACR:
        return None
    value = sum_rec_yards / sum_air_yards
    return value if value <= _MAX_PLAUSIBLE_RACR else None


def _team_week_totals(stats_rows: list[dict]) -> dict[tuple[str, int], dict[str, float]]:
    """
    {(team, week): {"targets": n, "carries": n}} summed over every player.

    nflverse's player_stats has NO team-level columns — confirmed against the
    live 2025 release, 145 columns and neither `team_targets` nor
    `team_carries` among them. The share calculations below read exactly
    those names, and `_num()` returns 0.0 for an absent field, so
    `sum(team_targets)` was 0 for every player and `target_share` /
    `carry_share` / `target_share_trend` stored as None for all 182 players
    in the database. Nothing surfaced: a share that cannot be computed and a
    player who genuinely has no role look identical downstream.

    Derived by summing the player rows rather than pulling load_team_stats,
    for two reasons. It is exact by construction — a team's targets ARE the
    sum of its players' targets, so shares total to 1.0 — whereas team_stats
    reports pass ATTEMPTS, which differ from targets by throwaways, spikes
    and sacks. And it needs no second source, so a share never silently
    depends on a feed that might fail independently of the one it divides.
    """
    totals: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: {"targets": 0.0, "carries": 0.0}
    )
    for row in stats_rows:
        team = _first(row, "team", "recent_team", "team_abbr")
        week = _first(row, "week")
        if team is None or week is None:
            continue
        try:
            key = (str(team), int(week))
        except (TypeError, ValueError):
            continue
        totals[key]["targets"] += _num(row, "targets")
        totals[key]["carries"] += _num(row, "carries", "rushing_attempts")
    return dict(totals)


def _team_totals_for(
    weeks: list[dict],
    team_totals: dict[tuple[str, int], dict[str, float]],
    field: str,
) -> float:
    """This player's team's total for `field`, over exactly the weeks he
    played. Summing the whole season instead would divide his partial-season
    targets by a full-season denominator and understate every injured
    player's share."""
    out = 0.0
    for w in weeks:
        team = _first(w, "team", "recent_team", "team_abbr")
        week = _first(w, "week")
        if team is None or week is None:
            continue
        try:
            out += team_totals.get((str(team), int(week)), {}).get(field, 0.0)
        except (TypeError, ValueError):
            continue
    return out


def _compute_opportunity_efficiency(
    weeks: list[dict],
    team_totals: dict[tuple[str, int], dict[str, float]] | None = None,
    position: str | None = None,
) -> dict:
    games = len(weeks)
    if games == 0:
        return {}
    team_totals = team_totals or {}

    targets = [_num(w, "targets") for w in weeks]
    carries = [_num(w, "carries", "rushing_attempts") for w in weeks]
    rec_yards = [_num(w, "receiving_yards") for w in weeks]
    rush_yards = [_num(w, "rushing_yards") for w in weeks]
    receptions = [_num(w, "receptions") for w in weeks]
    air_yards = [_num(w, "receiving_air_yards", "air_yards") for w in weeks]
    yac = [_num(w, "receiving_yards_after_catch", "yac") for w in weeks]
    sum_team_targets = _team_totals_for(weeks, team_totals, "targets")
    sum_team_carries = _team_totals_for(weeks, team_totals, "carries")

    sum_targets, sum_carries = sum(targets), sum(carries)
    sum_air_yards = sum(air_yards)
    sum_rec_yards = sum(rec_yards)

    return {
        "targets_per_game": sum_targets / games,
        "carries_per_game": sum_carries / games,
        "yards_per_target": (sum_rec_yards / sum_targets) if sum_targets else None,
        "yards_per_carry": (sum(rush_yards) / sum_carries) if sum_carries else None,
        "yac_per_reception": (sum(yac) / sum(receptions)) if sum(receptions) else None,
        # RACR (receiving yards / air yards) — see _racr for why this is
        # gated on position rather than on the size of the denominator, and
        # for the two rounds of live data that got it there.
        "racr": _racr(sum_rec_yards, sum_air_yards, position),
        "catch_rate": (sum(receptions) / sum_targets) if sum_targets else None,
        "target_share": (sum_targets / sum_team_targets) if sum_team_targets else None,
        "carry_share": (sum_carries / sum_team_carries) if sum_team_carries else None,
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
    team_totals: dict[tuple[str, int], dict[str, float]] | None = None,
) -> dict:
    """
    Trend = last TREND_WINDOW_WEEKS minus season average, for target share
    and snap share. Positive means an increasing role recently — exactly
    the kind of signal ADP (which lags) won't capture yet.

    Note: snap_weeks must be the player's rows from load_snap_counts, not
    player_stats — snap % lives in a separate nflverse dataset, it's not a
    column on the weekly stats rows passed in as `weeks`.
    """
    team_totals = team_totals or {}
    result: dict[str, Optional[float]] = {
        "target_share_trend": None,
        "snap_pct_trend": None,
        "depth_chart_trend": None,
    }

    weeks_sorted = sorted(weeks, key=lambda w: _num(w, "week"))
    if len(weeks_sorted) >= 2:
        recent = weeks_sorted[-TREND_WINDOW_WEEKS:]

        # Same absent-column problem as target_share above: player_stats has
        # no team_targets, so this read 0 and the trend was always None.
        season_targets = sum(_num(w, "targets") for w in weeks_sorted)
        season_team_targets = _team_totals_for(weeks_sorted, team_totals, "targets")
        recent_targets = sum(_num(w, "targets") for w in recent)
        recent_team_targets = _team_totals_for(recent, team_totals, "targets")

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

    dc_sorted = sorted(depth_chart_weeks, key=_order_key)
    if len(dc_sorted) >= 2:
        first_rank = _first(dc_sorted[0], "pos_rank", "depth_team", "depth_position", "rank")
        last_rank = _first(dc_sorted[-1], "pos_rank", "depth_team", "depth_position", "rank")
        try:
            if first_rank is not None and last_rank is not None:
                result["depth_chart_trend"] = int(last_rank) - int(first_rank)
        except (TypeError, ValueError):
            pass

    return result


def _compute_red_zone_touches(pbp_rows: list[dict]) -> dict[str, int]:
    """
    Total red zone touches for the season, keyed by gsis_id. Requires
    play-by-play data (the one heavy pull in this module — see --no-redzone
    to skip it). Standard definition: yardline_100 <= 20 (i.e. inside the
    opponent's 20-yard line).

    Returns raw COUNTS, not a per-game rate. The caller divides by the
    player's games played from player_stats (see refresh_metrics) because
    that is the only denominator available here that means what the field
    name claims.

    This used to divide by len({game_id for games in which this player had a
    red zone touch}), which is a different and much smaller number: a back
    with four red zone touches in one game and none in his other sixteen
    scored 4.0 "per game," identical to a genuine every-week goal-line back.
    Confirmed live on the 2025 data — Jaydon Blue (5 games, 4.1 PPR ppg),
    RJ Harvey and Kimani Vidal all landed on exactly 4.00, tied with Josh
    Jacobs and Jahmyr Gibbs, and Tank Bigsby showed 2.71 red zone touches a
    game on 3.6 PPR points, which is arithmetically impossible. The bug
    inflated precisely the low-usage players, and red zone volume is the
    signal the recommendation prompt leans on hardest for upside reasoning,
    so it was manufacturing fake breakout candidates.
    """
    touches_by_player: dict[str, int] = defaultdict(int)

    for play in pbp_rows:
        yardline = _first(play, "yardline_100")
        if yardline is None:
            continue
        try:
            if float(yardline) > 20:
                continue
        except (TypeError, ValueError):
            continue

        rusher = _first(play, "rusher_player_id", "rusher_id")
        receiver = _first(play, "receiver_player_id", "receiver_id")

        if rusher:
            touches_by_player[str(rusher)] += 1
        if receiver and _first(play, "pass_attempt"):
            touches_by_player[str(receiver)] += 1

    return dict(touches_by_player)


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
    redzone_by_player: dict[str, int] = {}
    # Distinguishes "play-by-play loaded and this player had zero red zone
    # touches" (a real 0.0) from "we never got play-by-play at all" (None,
    # meaning unknown). Without this flag a skipped/failed pbp pull would
    # write 0.0 for every player, which reads as a confident "no red zone
    # role" in the prompt rather than as a coverage gap.
    redzone_available = False
    team_pass_rate: dict[str, float] = {}
    team_totals: dict[tuple[str, int], dict[str, float]] = {}

    try:
        stats_rows = _load_dicts("load_player_stats", seasons=season, summary_level="week")
        stats_rows = _filter_regular_season(stats_rows)
        stats_by_player = _group_by_player(stats_rows, ("player_id", "gsis_id"))
        # Team-level denominators for target_share / carry_share, summed from
        # the same rows — player_stats ships no team columns. See
        # _team_week_totals for why this is derived rather than pulled.
        team_totals = _team_week_totals(stats_rows)
        logger.info(
            f"player_stats: {len(stats_rows):,} regular-season rows, "
            f"{len(stats_by_player):,} players, {len(team_totals):,} team-weeks"
        )
    except Exception as e:
        logger.warning(f"Could not load player_stats — opportunity/efficiency/consistency metrics will be skipped: {e}")

    try:
        snap_rows = _load_dicts("load_snap_counts", seasons=season)
        snap_rows = _filter_regular_season(snap_rows)
        # snap_counts carries only pfr_player_id. Stamp a gsis_id onto each
        # row first so it groups under the same key everything else uses —
        # see _load_pfr_crosswalk for why this was silently dropping all of it.
        if snap_rows and _first(snap_rows[0], "gsis_id") is None:
            pfr_map = _load_pfr_crosswalk()
            matched = 0
            for row in snap_rows:
                gsis = pfr_map.get(str(_first(row, "pfr_player_id") or ""))
                if gsis:
                    row["gsis_id"] = gsis
                    matched += 1
            logger.info(
                f"snap_counts: mapped {matched:,}/{len(snap_rows):,} rows "
                f"from pfr_player_id to gsis_id"
            )
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
            redzone_available = True
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
                # The team he earned these numbers with, taken from his LAST
                # week of the season so a mid-season trade records where he
                # finished. Compared against Player.team to detect who has
                # moved — see _format_roster_changes in ai_service.py.
                "team": _first(
                    max(weeks, key=_order_key), "team", "recent_team", "team_abbr"
                ),
            }
            fields.update(
                _compute_opportunity_efficiency(weeks, team_totals, player.position)
            )
            fields.update(_compute_consistency_risk(weeks, injuries_by_player.get(gsis_id, [])))
            fields.update(_compute_forward_looking(weeks, snap_weeks, depth_weeks, team_totals))

            if snap_weeks:
                pcts = [_num(w, "offense_pct", "snap_pct") for w in snap_weeks]
                fields["snap_pct"] = sum(pcts) / len(pcts) if pcts else None

            # Divide the season's raw touch count by games played — the same
            # denominator every other per-game field in this row uses. Only
            # written when play-by-play actually loaded; see
            # `redzone_available` above for why a 0.0 has to be earned.
            if redzone_available and weeks:
                fields["red_zone_touches_per_game"] = (
                    redzone_by_player.get(gsis_id, 0) / len(weeks)
                )

            if player.team in team_pass_rate:
                fields["team_pass_rate"] = team_pass_rate[player.team]

            if depth_weeks:
                # Sort by timestamp rather than trusting input order —
                # nflverse doesn't guarantee row order within a player's
                # group. Candidate list must match the one in
                # _compute_forward_looking: `pos_rank` first, since that is
                # what current nflverse depth charts actually use. Missing it
                # here while the trend had it is why depth_chart_rank stayed
                # NULL for all 182 players on the first corrected run.
                latest = max(depth_weeks, key=_order_key)
                latest_rank = _first(latest, "pos_rank", "depth_team", "depth_position", "rank")
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
