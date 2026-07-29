"""
relink_player_ids — the repair pass for Player.id churn after a full ADP
reingest. Pins the exact Gibbs/Robinson incident: two players close in ADP
flip row order between pulls, so their autoincrement IDs swap, and every
metrics/profile row silently points at the wrong real player until relink
trades the IDs back (staged through negative placeholders because
player_id is UNIQUE).
"""

from sqlmodel import delete

from backend.db import draft_profile_repo, metrics_repo
from backend.db.models import DraftProfile, Player, PlayerMetrics
from backend.tests.conftest import make_player


def seed_two_players(db):
    db.add(make_player(1, "Jahmyr Gibbs", "RB", "DET", adp=1.7, rank=1, sleeper_id="g1"))
    db.add(make_player(2, "Bijan Robinson", "RB", "ATL", adp=1.9, rank=2, sleeper_id="b2"))
    db.commit()


def swap_player_ids(db):
    """Simulates a reingest where the two players' CSV order flipped:
    delete-and-reinsert with the OTHER player getting each ID."""
    db.exec(delete(Player))
    db.commit()
    db.add(make_player(1, "Bijan Robinson", "RB", "ATL", adp=1.9, rank=1, sleeper_id="b2"))
    db.add(make_player(2, "Jahmyr Gibbs", "RB", "DET", adp=1.7, rank=2, sleeper_id="g1"))
    db.commit()


def test_metrics_relink_swaps_ids_back(db):
    seed_two_players(db)
    metrics_repo.upsert_metrics(db, player_id=1, sleeper_id="g1", season=2025, through_week=18, games_played=17)
    metrics_repo.upsert_metrics(db, player_id=2, sleeper_id="b2", season=2025, through_week=18, games_played=16)

    swap_player_ids(db)
    relinked, orphaned = metrics_repo.relink_player_ids(db)

    assert (relinked, orphaned) == (2, 0)
    # Gibbs's metrics must follow Gibbs (now id=2), not stay on id=1
    assert metrics_repo.get_metrics_by_sleeper_id(db, "g1").player_id == 2
    assert metrics_repo.get_metrics_by_sleeper_id(db, "b2").player_id == 1


def test_metrics_relink_deletes_orphans(db):
    seed_two_players(db)
    metrics_repo.upsert_metrics(db, player_id=1, sleeper_id="g1", season=2025, through_week=18, games_played=17)
    # Player g1 drops out of the ADP pool entirely
    db.exec(delete(Player))
    db.commit()
    db.add(make_player(1, "Bijan Robinson", "RB", "ATL", adp=1.9, rank=1, sleeper_id="b2"))
    db.commit()

    relinked, orphaned = metrics_repo.relink_player_ids(db)
    assert (relinked, orphaned) == (0, 1)
    assert metrics_repo.get_metrics_by_sleeper_id(db, "g1") is None


def test_metrics_rows_without_sleeper_id_left_untouched(db):
    seed_two_players(db)
    metrics_repo.upsert_metrics(db, player_id=1, sleeper_id=None, season=2025, through_week=18, games_played=17)
    relinked, orphaned = metrics_repo.relink_player_ids(db)
    assert (relinked, orphaned) == (0, 0)
    assert metrics_repo.get_metrics(db, 1) is not None


def test_draft_profile_relink_swaps_ids_back(db):
    seed_two_players(db)
    draft_profile_repo.upsert_draft_profile(db, player_id=1, sleeper_id="g1", draft_year=2023, draft_round=1)
    draft_profile_repo.upsert_draft_profile(db, player_id=2, sleeper_id="b2", draft_year=2023, draft_round=1)

    swap_player_ids(db)
    relinked, orphaned = draft_profile_repo.relink_player_ids(db)

    assert (relinked, orphaned) == (2, 0)
    assert draft_profile_repo.get_draft_profile_by_sleeper_id(db, "g1").player_id == 2
    assert draft_profile_repo.get_draft_profile_by_sleeper_id(db, "b2").player_id == 1
