"""
Tests for db/game_repo.py — the NFL schedule table backing the Opportunity
section's future matchup context.
"""

from backend.db import game_repo
from backend.db.models import Game


def _games(season: int) -> list[dict]:
    return [
        {"week": 1, "game_type": "REG", "home_team": "DET", "away_team": "GB", "game_date": None},
        {"week": 1, "game_type": "REG", "home_team": "KC", "away_team": "BAL", "game_date": None},
        {"week": 2, "game_type": "REG", "home_team": "GB", "away_team": "DET", "game_date": None},
        {"week": 3, "game_type": "REG", "home_team": "DET", "away_team": "CHI", "game_date": None},
        # Postseason row — should be excluded from get_remaining_schedule.
        {"week": 1, "game_type": "POST", "home_team": "DET", "away_team": "TB", "game_date": None},
    ]


def test_replace_season_stores_every_row(db):
    from sqlmodel import select

    stored = game_repo.replace_season(db, 2026, _games(2026))
    assert stored == 5
    assert len(db.exec(select(Game).where(Game.season == 2026)).all()) == 5


def test_replace_season_is_idempotent_and_replaces_not_appends(db):
    game_repo.replace_season(db, 2026, _games(2026))
    # Re-run with a trimmed set — the old rows should be gone, not just
    # added to, since the whole point is "here's the current full picture."
    trimmed = _games(2026)[:2]
    stored = game_repo.replace_season(db, 2026, trimmed)
    assert stored == 2
    assert not game_repo.get_opponent(db, "DET", 2026, 3)  # week 3 game is gone


def test_replace_season_scopes_to_the_given_season_only(db):
    game_repo.replace_season(db, 2025, _games(2025))
    game_repo.replace_season(db, 2026, _games(2026))
    # Re-replacing 2026 must not touch 2025's rows.
    game_repo.replace_season(db, 2026, _games(2026)[:1])
    assert game_repo.get_opponent(db, "DET", 2025, 2) == "GB"


def test_get_opponent_works_from_either_home_or_away_side(db):
    game_repo.replace_season(db, 2026, _games(2026))
    assert game_repo.get_opponent(db, "DET", 2026, 1) == "GB"   # DET is home
    assert game_repo.get_opponent(db, "GB", 2026, 1) == "DET"   # same game, from GB's side


def test_get_opponent_returns_none_for_a_bye_or_unknown_week(db):
    game_repo.replace_season(db, 2026, _games(2026))
    assert game_repo.get_opponent(db, "DET", 2026, 9) is None
    assert game_repo.get_opponent(db, "NOTATEAM", 2026, 1) is None


def test_get_remaining_schedule_excludes_postseason(db):
    game_repo.replace_season(db, 2026, _games(2026))
    out = game_repo.get_remaining_schedule(db, "DET", 2026, from_week=1)
    weeks = [g["week"] for g in out]
    assert weeks == [1, 2, 3]  # not the week-1 POST row


def test_get_remaining_schedule_respects_from_week_and_reports_home_away(db):
    game_repo.replace_season(db, 2026, _games(2026))
    out = game_repo.get_remaining_schedule(db, "DET", 2026, from_week=2)
    assert [g["week"] for g in out] == [2, 3]
    assert out[0] == {"week": 2, "opponent": "GB", "is_home": False}
    assert out[1] == {"week": 3, "opponent": "CHI", "is_home": True}


def test_has_season_distinguishes_never_ingested_from_a_real_bye(db):
    assert game_repo.has_season(db, 2026) is False
    game_repo.replace_season(db, 2026, _games(2026))
    assert game_repo.has_season(db, 2026) is True
    # A team with no rows in an ingested season is still distinguishable
    # from the season never having been ingested at all.
    assert game_repo.get_opponent(db, "NOTATEAM", 2026, 1) is None
    assert game_repo.has_season(db, 2026) is True
