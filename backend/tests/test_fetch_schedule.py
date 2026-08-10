"""
Tests for fetch_schedule.py's pure row-shaping logic — _shape_games,
_first, _parse_game_date, _default_season. Deliberately does NOT touch
nflreadpy or the network: refresh_metrics()-style live pulls are exercised
manually via diagnose_ingestion.py, not in the test suite (see this app's
existing fetch_metrics.py, which has no live-pull tests either).
"""

from datetime import datetime

from backend.ingestion.fetch_schedule import (
    _default_season,
    _first,
    _parse_game_date,
    _shape_games,
)


def test_first_returns_first_present_candidate():
    row = {"season_type": "REG"}
    assert _first(row, "game_type", "season_type") == "REG"


def test_first_returns_none_when_no_candidate_present():
    assert _first({"foo": "bar"}, "game_type", "season_type") is None


def test_first_skips_none_values_not_just_missing_keys():
    row = {"game_type": None, "season_type": "REG"}
    assert _first(row, "game_type", "season_type") == "REG"


def test_parse_game_date_accepts_iso_string():
    assert _parse_game_date("2026-09-10") == datetime(2026, 9, 10)


def test_parse_game_date_passes_through_a_real_datetime():
    dt = datetime(2026, 9, 10, 13, 0)
    assert _parse_game_date(dt) is dt


def test_parse_game_date_returns_none_for_garbage_or_missing():
    assert _parse_game_date(None) is None
    assert _parse_game_date("not a date") is None


def test_shape_games_maps_expected_fields():
    rows = [
        {"week": 1, "home_team": "DET", "away_team": "GB",
         "game_type": "REG", "gameday": "2026-09-10"},
    ]
    out = _shape_games(rows, season=2026)
    assert out == [{
        "week": 1, "game_type": "REG", "home_team": "DET", "away_team": "GB",
        "game_date": datetime(2026, 9, 10),
    }]


def test_shape_games_defaults_game_type_to_reg_when_absent():
    rows = [{"week": 1, "home_team": "DET", "away_team": "GB"}]
    out = _shape_games(rows, season=2026)
    assert out[0]["game_type"] == "REG"


def test_shape_games_skips_rows_missing_required_fields():
    rows = [
        {"week": 1, "home_team": "DET", "away_team": "GB"},  # complete
        {"week": None, "home_team": "DET", "away_team": "GB"},  # no week
        {"week": 2, "home_team": None, "away_team": "GB"},  # no home_team
        {"week": 3, "home_team": "DET", "away_team": None},  # no away_team
    ]
    out = _shape_games(rows, season=2026)
    assert len(out) == 1
    assert out[0]["week"] == 1


def test_default_season_is_the_calendar_year_regardless_of_month():
    # Unlike fetch_metrics.py's _default_season, this one does NOT lag a
    # year for Jan-Aug — the schedule for the season kicking off this
    # September is already published well before that month arrives.
    assert _default_season(datetime(2026, 1, 15)) == 2026
    assert _default_season(datetime(2026, 8, 10)) == 2026
    assert _default_season(datetime(2026, 11, 1)) == 2026
