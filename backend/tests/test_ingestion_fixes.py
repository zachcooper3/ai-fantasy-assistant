"""
Regression tests for the 2026-08-13 ingestion audit.

Each test here pins a bug that was found by reading live data rather than
by reading code, and that produced wrong-but-plausible output rather than
an error — the class of bug this pipeline is most exposed to, since almost
every field degrades to None/0 instead of raising.

Author: Zach Cooper
"""

import pytest
from sqlmodel import Session, select

from backend.db import draft_profile_repo, metrics_repo
from backend.db.metrics_repo import RelinkAborted
from backend.db.models import DraftProfile, PlayerMetrics
from backend.ingestion.fetch_metrics import _compute_red_zone_touches
from backend.tests.conftest import make_player


# ---------------------------------------------------------------------------
# Red zone touches — denominator must be games played, not games with a touch
# ---------------------------------------------------------------------------

def _rz_play(game_id: str, rusher: str | None = None, receiver: str | None = None):
    play = {"yardline_100": 5, "game_id": game_id}
    if rusher:
        play["rusher_player_id"] = rusher
    if receiver:
        play["receiver_player_id"] = receiver
        play["pass_attempt"] = 1
    return play


def test_red_zone_returns_raw_counts_not_a_rate():
    """The heart of the bug: four touches in one game is four touches, and
    the per-game division belongs to the caller, which knows how many games
    the player actually played."""
    plays = [_rz_play("g1", rusher="p1") for _ in range(4)]
    assert _compute_red_zone_touches(plays) == {"p1": 4}


def test_red_zone_counts_span_games():
    plays = [
        _rz_play("g1", rusher="p1"),
        _rz_play("g2", rusher="p1"),
        _rz_play("g2", receiver="p2"),
    ]
    assert _compute_red_zone_touches(plays) == {"p1": 2, "p2": 1}


def test_red_zone_ignores_plays_outside_the_twenty():
    plays = [
        {"yardline_100": 45, "game_id": "g1", "rusher_player_id": "p1"},
        _rz_play("g1", rusher="p1"),
    ]
    assert _compute_red_zone_touches(plays) == {"p1": 1}


def test_red_zone_ignores_targets_that_are_not_pass_attempts():
    """A receiver_player_id on a non-pass play (e.g. a lateral or a
    penalty-nullified row) isn't a red zone target."""
    plays = [{"yardline_100": 10, "game_id": "g1", "receiver_player_id": "p1"}]
    assert _compute_red_zone_touches(plays) == {}


def test_red_zone_tolerates_unparseable_yardline():
    plays = [
        {"yardline_100": "not a number", "game_id": "g1", "rusher_player_id": "p1"},
        _rz_play("g1", rusher="p1"),
    ]
    assert _compute_red_zone_touches(plays) == {"p1": 1}


def test_red_zone_rate_no_longer_inflates_the_one_game_wonder():
    """The live signature of the bug: a back with four touches in a single
    game out of seventeen used to tie Jahmyr Gibbs at 4.00/game. Same raw
    data, correct denominator, three orders of relevance apart."""
    plays = [_rz_play("g1", rusher="spike") for _ in range(4)]
    plays += [_rz_play(f"g{i}", rusher="workhorse") for i in range(1, 18)]

    counts = _compute_red_zone_touches(plays)
    games_played = 17

    assert counts["spike"] / games_played == pytest.approx(0.235, abs=0.001)
    assert counts["workhorse"] / games_played == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Relink guards — a broken sleeper_id crosswalk must not empty the tables
# ---------------------------------------------------------------------------

def _seed_metrics(db: Session, player_id: int, sleeper_id: str):
    db.add(PlayerMetrics(
        player_id=player_id, sleeper_id=sleeper_id,
        season=2025, through_week=18, games_played=17,
    ))
    db.commit()


def _seed_profile(db: Session, player_id: int, sleeper_id: str):
    db.add(DraftProfile(player_id=player_id, sleeper_id=sleeper_id, draft_year=2026))
    db.commit()


def test_relink_aborts_when_no_player_has_a_sleeper_id(db):
    """The Sleeper-outage path: ingest wiped sleeper_id, sync failed to
    restore it, so every metrics row looks orphaned. Deleting them all is
    the one outcome that isn't recoverable."""
    for i in (1, 2, 3):
        db.add(make_player(i, f"Player {i}", sleeper_id=None))
    db.commit()
    for i in (1, 2, 3):
        _seed_metrics(db, i, f"s{i}")

    with pytest.raises(RelinkAborted, match="sync_sleeper_ids"):
        metrics_repo.relink_player_ids(db)

    assert len(db.exec(select(PlayerMetrics)).all()) == 3


def test_relink_aborts_on_a_partial_sleeper_payload(db):
    """Not an outage — a degraded response. Four of twenty players resolve,
    so 80% of rows would orphan. Above the ceiling, so nothing is deleted."""
    for i in range(1, 21):
        db.add(make_player(i, f"Player {i}", sleeper_id=f"s{i}" if i <= 4 else None))
    db.commit()
    for i in range(1, 21):
        _seed_metrics(db, i, f"s{i}")

    with pytest.raises(RelinkAborted, match="orphaned"):
        metrics_repo.relink_player_ids(db)

    assert len(db.exec(select(PlayerMetrics)).all()) == 20


def test_orphan_fraction_guard_is_skipped_on_a_tiny_table(db):
    """A fraction over three rows carries no signal — one legitimate
    dropout would read as 33% and block a correct relink. Below
    MIN_ROWS_FOR_ORPHAN_GUARD only the empty-crosswalk check applies."""
    for i in (1, 2):
        db.add(make_player(i, f"Player {i}", sleeper_id=f"s{i}"))
    db.commit()
    for i in (1, 2):
        _seed_metrics(db, i, f"s{i}")
    _seed_metrics(db, 99, "gone")

    relinked, orphaned = metrics_repo.relink_player_ids(db)
    assert orphaned == 1


def test_relink_still_orphans_a_normal_amount_of_churn(db):
    """The guard must not block the thing it's guarding. One player of four
    genuinely dropping out of the ADP pool is routine and still deleted."""
    for i in (1, 2, 3):
        db.add(make_player(i, f"Player {i}", sleeper_id=f"s{i}"))
    db.commit()
    for i in (1, 2, 3):
        _seed_metrics(db, i, f"s{i}")
    _seed_metrics(db, 99, "gone-from-the-pool")

    relinked, orphaned = metrics_repo.relink_player_ids(db)

    assert orphaned == 1
    assert relinked == 0
    remaining = {m.sleeper_id for m in db.exec(select(PlayerMetrics)).all()}
    assert remaining == {"s1", "s2", "s3"}


def test_relink_still_swaps_two_players_who_traded_ids(db):
    """The original Gibbs/Robinson case — the guard must leave it working."""
    db.add(make_player(1, "Gibbs", sleeper_id="gibbs"))
    db.add(make_player(2, "Robinson", sleeper_id="robinson"))
    db.commit()
    _seed_metrics(db, 2, "gibbs")
    _seed_metrics(db, 1, "robinson")

    relinked, orphaned = metrics_repo.relink_player_ids(db)

    assert (relinked, orphaned) == (2, 0)
    rows = {m.sleeper_id: m.player_id for m in db.exec(select(PlayerMetrics)).all()}
    assert rows == {"gibbs": 1, "robinson": 2}


def test_draft_profile_relink_aborts_on_the_same_signal(db):
    for i in (1, 2, 3):
        db.add(make_player(i, f"Player {i}", sleeper_id=None))
    db.commit()
    for i in (1, 2, 3):
        _seed_profile(db, i, f"s{i}")

    with pytest.raises(RelinkAborted):
        draft_profile_repo.relink_player_ids(db)

    assert len(db.exec(select(DraftProfile)).all()) == 3


def test_relink_no_ops_on_an_empty_table(db):
    """No rows to link is not a degraded crosswalk — it's a fresh install."""
    db.add(make_player(1, "Player", sleeper_id=None))
    db.commit()
    assert metrics_repo.relink_player_ids(db) == (0, 0)
