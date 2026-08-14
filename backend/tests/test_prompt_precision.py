"""
Tests for synthesis prompt formatting precision and last_updated freshness.

Both pin things that were wrong in a way no error could reveal: a prompt
that reads fine but asserts twelve significant figures from a 17-game
sample, and a timestamp field whose entire job is to report freshness and
which reported the wrong answer for 170 of 433 rows.

Author: Zach Cooper
"""

import re
from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import Session, select

from backend.db import draft_profile_repo, metrics_repo
from backend.db.models import DraftProfile, Player, PlayerMetrics
from backend.ingestion.fetch_synthesis import _round, format_metrics_prompt


# ---------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------

def test_round_handles_none():
    assert _round(None) is None
    assert _round(None, 2) is None


def test_round_defaults_to_one_decimal():
    assert _round(5.529411764705882) == 5.5
    assert _round(21.582352941176474) == 21.6


def test_round_takes_an_explicit_precision():
    assert _round(0.8993630573248408, 2) == 0.9


def _metrics(**over) -> PlayerMetrics:
    base = dict(
        player_id=1, sleeper_id="s1", season=2025, through_week=18,
        games_played=17,
        targets_per_game=5.529411764705882,
        carries_per_game=14.294117647058824,
        red_zone_touches_per_game=1.2666666666666666,
        snap_pct=0.67, target_share=0.171, carry_share=0.55,
        yards_per_target=6.553191489361702,
        yards_per_carry=5.032921810699588,
        yac_per_reception=7.987012987012987,
        racr=0.8993630573248408,
        catch_rate=0.819,
        team_pass_rate=0.567, depth_chart_rank=1,
        fantasy_points_avg=21.582352941176474,
        fantasy_points_stdev=13.123317160199903,
        injury_report_appearances=0, games_missed=0,
        target_share_trend=0.009, snap_pct_trend=0.083, depth_chart_trend=0,
    )
    base.update(over)
    return PlayerMetrics(**base)


def _player() -> Player:
    return Player(id=1, rank=1, name="Jahmyr Gibbs", team="DET", bye=6,
                  pos_rank="RB1", position="RB", adp=1.0, sleeper_id="s1")


def test_no_long_floats_survive_into_the_prompt():
    """The actual regression: any number with more than two decimals is a
    precision claim the underlying counting stats can't support."""
    out = format_metrics_prompt(_player(), _metrics())
    offenders = [
        tok for tok in re.findall(r"\d+\.\d{3,}", out)
    ]
    assert offenders == [], f"unrounded values leaked into the prompt: {offenders}"


def test_the_numbers_are_still_right_after_rounding():
    out = format_metrics_prompt(_player(), _metrics())
    assert "Targets/game: 5.5" in out
    assert "Carries/game: 14.3" in out
    assert "Red zone touches/game: 1.3" in out
    assert "Fantasy points/game (PPR): 21.6" in out
    assert "Week-to-week std dev (PPR): 13.1" in out


def test_racr_keeps_two_decimals():
    """One decimal would collapse meaningfully different receivers onto the
    same value, since RACR lives in roughly 0.5-1.5."""
    assert "RACR: 0.9" in format_metrics_prompt(_player(), _metrics())


def test_percentages_are_unaffected():
    out = format_metrics_prompt(_player(), _metrics())
    assert "Snap %: 67%" in out
    assert "Target share: 17%" in out


def test_rounding_shortens_the_prompt():
    """Cheap sanity check that this actually saves tokens rather than just
    looking tidier."""
    out = format_metrics_prompt(_player(), _metrics())
    assert len(out) < 700


def test_a_metric_that_is_none_stays_absent():
    """Rounding must not turn a missing metric into a rendered 0."""
    out = format_metrics_prompt(_player(), _metrics(racr=None, yards_per_carry=None))
    assert "RACR" not in out
    assert "Yards/carry" not in out


# ---------------------------------------------------------------------------
# last_updated freshness
# ---------------------------------------------------------------------------

def _is_recent(ts: datetime) -> bool:
    """SQLite round-trips these naive; compare on a tolerant window."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return abs(now - ts.replace(tzinfo=None)) < timedelta(minutes=5)


def test_metrics_update_refreshes_last_updated(db):
    metrics_repo.upsert_metrics(db, player_id=1, sleeper_id="s1",
                                season=2025, through_week=18, games_played=17)
    row = db.exec(select(PlayerMetrics)).one()
    row.last_updated = datetime(2026, 7, 29, 1, 40, 43)   # the live stale value
    db.add(row)
    db.commit()

    metrics_repo.upsert_metrics(db, player_id=1, sleeper_id="s1",
                                season=2025, through_week=18, games_played=17,
                                red_zone_touches_per_game=1.27)

    refreshed = db.exec(select(PlayerMetrics)).one()
    assert refreshed.red_zone_touches_per_game == 1.27
    assert _is_recent(refreshed.last_updated), (
        "last_updated still reports the insert date after an update"
    )


def test_draft_profile_update_refreshes_last_updated(db):
    """Matters more here — two scripts upsert into the same row, so the
    timestamp should track whichever ran last."""
    draft_profile_repo.upsert_draft_profile(db, player_id=1, sleeper_id="s1",
                                            draft_year=2025, draft_round=1)
    row = db.exec(select(DraftProfile)).one()
    row.last_updated = datetime(2026, 7, 29, 1, 40, 43)
    db.add(row)
    db.commit()

    # Second source enriching the same row (fetch_college_stats' path).
    draft_profile_repo.upsert_draft_profile(db, player_id=1, sleeper_id="s1",
                                            draft_year=2025, college_season=2024,
                                            rushing_yards=1660)

    refreshed = db.exec(select(DraftProfile)).one()
    assert refreshed.rushing_yards == 1660
    assert refreshed.draft_round == 1, "the other source's field was clobbered"
    assert _is_recent(refreshed.last_updated)


def test_insert_still_stamps_last_updated(db):
    metrics_repo.upsert_metrics(db, player_id=2, sleeper_id="s2",
                                season=2025, through_week=18, games_played=17)
    assert _is_recent(db.exec(select(PlayerMetrics)).one().last_updated)
