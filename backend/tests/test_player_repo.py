"""
player_repo — the "available player" queries that feed the recommendation
board, scarcity counts, and handcuff lookups.

Live failure (2026-08-09): Ricky Pearsall (IR, out for the season) was
recommended mid-draft. Player.injury_status was added on 2026-08-07
specifically after an earlier IR-recommendation incident, with a comment
promising it would be "a hard exclusion, not a risk to weigh" — but nothing
in app/ ever actually read the column. It was populated by ingestion and
then queried nowhere, so the same class of bug reproduced days later. These
tests pin the fix: injury_status must be enforced at the query layer itself
so no caller (board, scarcity, handcuff) can forget to check it.
"""

from backend.db import player_repo as repo
from backend.tests.conftest import make_player


def _seed(db, *players):
    for p in players:
        db.add(p)
    db.commit()


def test_ir_player_excluded_from_top_available(db):
    _seed(
        db,
        make_player(1, "Healthy WR", "WR", adp=10.0),
        make_player(2, "Ricky Pearsall", "WR", adp=12.0, injury_status="IR"),
    )
    names = [p.name for p in repo.get_top_available(db, n=10)]
    assert "Ricky Pearsall" not in names
    assert "Healthy WR" in names


def test_pup_and_suspended_and_out_are_also_excluded(db):
    _seed(
        db,
        make_player(1, "On PUP", "WR", adp=10.0, injury_status="PUP"),
        make_player(2, "Suspended Guy", "WR", adp=11.0, injury_status="Suspended"),
        make_player(3, "Ruled Out", "WR", adp=12.0, injury_status="Out"),
        make_player(4, "Clean Board", "WR", adp=13.0),
    )
    names = [p.name for p in repo.get_top_available(db, n=10)]
    assert names == ["Clean Board"]


def test_questionable_and_doubtful_are_not_excluded(db):
    # These are game-day risk tags, not "cannot play" — the model should
    # still see and weigh them, not have them silently vanish from the board.
    _seed(
        db,
        make_player(1, "Iffy Guy", "WR", adp=10.0, injury_status="Questionable"),
        make_player(2, "Longshot", "WR", adp=11.0, injury_status="Doubtful"),
    )
    names = [p.name for p in repo.get_top_available(db, n=10)]
    assert "Iffy Guy" in names and "Longshot" in names


def test_null_injury_status_is_not_excluded(db):
    # Regression guard for the NULL-unsafe version of this filter: a plain
    # `.notin_(UNDRAFTABLE_STATUSES)` drops every NULL row too, since SQL's
    # NOT IN evaluates to NULL (not true) against a NULL column — which
    # would have emptied the board of every healthy player.
    _seed(db, make_player(1, "Healthy", "WR", adp=10.0, injury_status=None))
    names = [p.name for p in repo.get_top_available(db, n=10)]
    assert names == ["Healthy"]


def test_ir_player_excluded_from_available_from_adp(db):
    _seed(
        db,
        make_player(1, "IR Guy", "RB", adp=50.0, injury_status="IR"),
        make_player(2, "Healthy Guy", "RB", adp=55.0),
    )
    names = [p.name for p in repo.get_available_from_adp(db, min_adp=40.0, n=5)]
    assert names == ["Healthy Guy"]


def test_ir_player_excluded_from_best_available_by_ppg(db):
    from backend.db.models import PlayerMetrics

    ir = make_player(1, "IR Stud", "RB", adp=20.0, injury_status="IR")
    healthy = make_player(2, "Healthy Value", "RB", adp=60.0)
    _seed(db, ir, healthy)
    db.add(PlayerMetrics(player_id=1, season=2025, through_week=17, games_played=17,
                          fantasy_points_avg=20.0))
    db.add(PlayerMetrics(player_id=2, season=2025, through_week=17, games_played=17,
                          fantasy_points_avg=10.0))
    db.commit()

    names = [p.name for p in repo.get_best_available_by_ppg(db, n=5)]
    assert names == ["Healthy Value"]


def test_ir_only_position_reads_as_zero_available(db):
    # count_available_by_position feeds the DST/K "drop the requirement"
    # check in ai_service.py — an IR-only kicker must read as 0 available,
    # not 1, or the requirement never gets dropped for an undraftable slot.
    _seed(db, make_player(1, "Hurt Kicker", "K", adp=200.0, injury_status="IR"))
    counts = repo.count_available_by_position(db)
    assert counts.get("K", 0) == 0


def test_handcuff_excludes_an_injured_backup(db):
    _seed(
        db,
        make_player(1, "Starter RB", "RB", team="SF", adp=10.0),
        make_player(2, "Hurt Backup", "RB", team="SF", adp=80.0, injury_status="IR"),
        make_player(3, "Healthy Backup", "RB", team="SF", adp=90.0),
    )
    handcuff = repo.get_handcuff(db, player_id=1)
    assert handcuff is not None
    assert handcuff.name == "Healthy Backup"
