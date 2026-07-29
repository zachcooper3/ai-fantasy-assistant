"""
draft_session_repo — the C2 crash-recovery journal. Round-trips config +
picks through SQLite exactly the way main.py's lifespan rehydration does.
"""

from backend.app.services.draft_state import DraftConfig, DraftStateService
from backend.db import draft_session_repo as jrepo


CFG = DraftConfig(
    league_size=10, my_draft_position=4, total_rounds=16,
    qb_slots=1, rb_slots=2, wr_slots=3, te_slots=1, flex_slots=2, dst_slots=0,
)


def journal_picks(db, svc, specs):
    for pid, name, pos, team in specs:
        pick = svc.record_pick(player_id=pid, player_name=name, position=pos, nfl_team=team)
        jrepo.append_pick(db, pick)


def test_full_round_trip_restores_config_and_picks(db):
    jrepo.save_config(db, CFG)
    svc = DraftStateService()
    svc.start_session(CFG)
    journal_picks(db, svc, [
        (11, "A", "RB", "DET"),
        (-1, "?Unknown", "?", "?"),   # sync placeholder must survive the journal
        (13, "C", "WR", "KC"),
    ])

    restored = jrepo.load_state(db)
    assert restored is not None
    rcfg, rpicks = restored
    assert rcfg == CFG

    svc2 = DraftStateService()
    svc2.restore_session(rcfg, rpicks)
    assert svc2.is_active
    assert svc2.current_pick_number == 4
    assert [p.pick_number for p in svc2.picks] == [1, 2, 3]
    assert svc2.picks[1].player_id == -1
    assert svc2.config.wr_slots == 3 and svc2.config.dst_slots == 0


def test_remove_pick_by_number(db):
    jrepo.save_config(db, CFG)
    svc = DraftStateService()
    svc.start_session(CFG)
    journal_picks(db, svc, [(1, "A", "RB", "X"), (2, "B", "WR", "Y"), (3, "C", "TE", "Z")])

    jrepo.remove_pick(db, 3)
    _, picks = jrepo.load_state(db)
    assert [p.pick_number for p in picks] == [1, 2]

    # Removing a nonexistent pick is a no-op, not an error
    jrepo.remove_pick(db, 99)
    _, picks = jrepo.load_state(db)
    assert len(picks) == 2


def test_save_config_replaces_previous_session_and_journal(db):
    jrepo.save_config(db, CFG)
    svc = DraftStateService()
    svc.start_session(CFG)
    journal_picks(db, svc, [(1, "A", "RB", "X")])

    cfg2 = DraftConfig(league_size=12, my_draft_position=1, total_rounds=15)
    jrepo.save_config(db, cfg2)

    restored_cfg, picks = jrepo.load_state(db)
    assert restored_cfg.league_size == 12
    assert picks == []


def test_clear_removes_everything(db):
    jrepo.save_config(db, CFG)
    jrepo.clear(db)
    assert jrepo.load_state(db) is None


def test_load_state_none_when_no_session(db):
    assert jrepo.load_state(db) is None
