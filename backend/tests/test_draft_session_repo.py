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
    rcfg, rpicks, rstarted_at = restored
    assert rcfg == CFG
    assert rstarted_at is not None

    svc2 = DraftStateService()
    svc2.restore_session(rcfg, rpicks, started_at=rstarted_at)
    assert svc2.is_active
    # A rehydrated session must announce itself — the UI shows a resume banner
    # off this flag, and without it a backend restart silently resumes an old
    # draft with no indication anything happened.
    assert svc2.was_restored is True
    assert svc2.started_at == rstarted_at
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
    _, picks, _ = jrepo.load_state(db)
    assert [p.pick_number for p in picks] == [1, 2]

    # Removing a nonexistent pick is a no-op, not an error
    jrepo.remove_pick(db, 99)
    _, picks, _ = jrepo.load_state(db)
    assert len(picks) == 2


def test_save_config_replaces_previous_session_and_journal(db):
    jrepo.save_config(db, CFG)
    svc = DraftStateService()
    svc.start_session(CFG)
    journal_picks(db, svc, [(1, "A", "RB", "X")])

    cfg2 = DraftConfig(league_size=12, my_draft_position=1, total_rounds=15)
    jrepo.save_config(db, cfg2)

    restored_cfg, picks, _ = jrepo.load_state(db)
    assert restored_cfg.league_size == 12
    assert picks == []


def test_clear_removes_everything(db):
    jrepo.save_config(db, CFG)
    jrepo.clear(db)
    assert jrepo.load_state(db) is None


def test_load_state_none_when_no_session(db):
    assert jrepo.load_state(db) is None


# ---------------------------------------------------------------------------
# ai_model — the AI panel's Haiku/Sonnet toggle
# ---------------------------------------------------------------------------

def test_ai_model_defaults_to_unset(db):
    jrepo.save_config(db, CFG)
    assert jrepo.get_ai_model(db) is None


def test_save_config_carries_an_explicit_ai_model(db):
    jrepo.save_config(db, CFG, ai_model="sonnet")
    assert jrepo.get_ai_model(db) == "sonnet"


def test_set_ai_model_updates_in_place_without_touching_picks(db):
    jrepo.save_config(db, CFG)
    svc = DraftStateService()
    svc.start_session(CFG)
    journal_picks(db, svc, [(1, "A", "RB", "X")])

    jrepo.set_ai_model(db, "sonnet")

    assert jrepo.get_ai_model(db) == "sonnet"
    _, picks, _ = jrepo.load_state(db)
    assert len(picks) == 1  # untouched — set_ai_model must not be save_config


def test_set_ai_model_is_a_noop_with_no_active_session(db):
    # Toggling the model before a draft has started shouldn't raise or
    # create a partial session row — it just has nothing to write to yet.
    jrepo.set_ai_model(db, "sonnet")
    assert jrepo.load_state(db) is None


def test_get_ai_model_none_with_no_active_session(db):
    assert jrepo.get_ai_model(db) is None


# ---------------------------------------------------------------------------
# sleeper_draft_id — resuming live Sleeper sync across a backend restart
# ---------------------------------------------------------------------------

def test_sleeper_draft_id_defaults_to_unset(db):
    jrepo.save_config(db, CFG)
    assert jrepo.get_sleeper_draft_id(db) is None


def test_set_sleeper_draft_id_updates_in_place_without_touching_picks(db):
    jrepo.save_config(db, CFG)
    svc = DraftStateService()
    svc.start_session(CFG)
    journal_picks(db, svc, [(1, "A", "RB", "X")])

    jrepo.set_sleeper_draft_id(db, "123456789")

    assert jrepo.get_sleeper_draft_id(db) == "123456789"
    _, picks, _ = jrepo.load_state(db)
    assert len(picks) == 1  # untouched — set_sleeper_draft_id must not be save_config


def test_set_sleeper_draft_id_clears_a_previously_set_value(db):
    # Mirrors DELETE /api/sync/stop: an explicit stop must be honored on the
    # next restart, not silently reconnect to the old draft.
    jrepo.save_config(db, CFG)
    jrepo.set_sleeper_draft_id(db, "123456789")
    assert jrepo.get_sleeper_draft_id(db) == "123456789"

    jrepo.set_sleeper_draft_id(db, None)
    assert jrepo.get_sleeper_draft_id(db) is None


def test_set_sleeper_draft_id_is_a_noop_with_no_active_session(db):
    jrepo.set_sleeper_draft_id(db, "123456789")
    assert jrepo.load_state(db) is None


def test_get_sleeper_draft_id_none_with_no_active_session(db):
    assert jrepo.get_sleeper_draft_id(db) is None


def test_save_config_does_not_carry_over_a_previous_sleeper_draft_id(db):
    # A new session (save_config replaces the row wholesale) must start with
    # no sync to resume — otherwise starting a fresh, unsynced draft after a
    # previously-synced one would try to reconnect to the OLD draft on the
    # next restart.
    jrepo.save_config(db, CFG)
    jrepo.set_sleeper_draft_id(db, "123456789")

    jrepo.save_config(db, CFG)
    assert jrepo.get_sleeper_draft_id(db) is None
