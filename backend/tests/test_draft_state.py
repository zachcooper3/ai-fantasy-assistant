"""
DraftStateService — snake-draft math, pick recording, undo, restore.

The highest-value tests in the suite: every downstream feature (roster
views, AI context, sync attribution) silently corrupts if this math is
wrong, and it's all pure functions.
"""

import pytest

from backend.app.services.draft_state import DraftConfig, DraftStateService, PickRecord


def make_service(league_size=12, my_pos=3, rounds=15) -> DraftStateService:
    svc = DraftStateService()
    svc.start_session(DraftConfig(
        league_size=league_size, my_draft_position=my_pos, total_rounds=rounds,
    ))
    return svc


# ---------------------------------------------------------------------------
# Snake math
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pick,slot", [
    (1, 1), (12, 12),   # round 1 ascending
    (13, 12), (24, 1),  # round 2 descending (snake reverses)
    (25, 1), (36, 12),  # round 3 ascending again
])
def test_slot_for_pick_docstring_examples(pick, slot):
    assert make_service().slot_for_pick(pick) == slot


def test_every_round_contains_every_slot_exactly_once():
    svc = make_service(league_size=10, rounds=4)
    for rnd in range(4):
        slots = sorted(
            svc.slot_for_pick(p) for p in range(rnd * 10 + 1, rnd * 10 + 11)
        )
        assert slots == list(range(1, 11))


def test_round_for_pick_boundaries():
    svc = make_service(league_size=12)
    assert svc.round_for_pick(1) == 1
    assert svc.round_for_pick(12) == 1
    assert svc.round_for_pick(13) == 2


# ---------------------------------------------------------------------------
# Turn tracking
# ---------------------------------------------------------------------------

def test_my_next_pick_and_picks_until_turn():
    svc = make_service(league_size=12, my_pos=3)
    assert svc.my_next_pick_number == 3
    assert svc.picks_until_my_turn == 2
    assert not svc.is_my_turn

    for i in range(2):
        svc.record_pick(player_id=i + 1, player_name=f"P{i}", position="RB", nfl_team="X")
    assert svc.is_my_turn
    assert svc.picks_until_my_turn == 0

    svc.record_pick(player_id=99, player_name="Mine", position="WR", nfl_team="Y")
    # Snake: slot 3's next turn is pick 22 (round 2 descending)
    assert svc.my_next_pick_number == 22
    assert svc.picks_until_my_turn == 22 - 4


def test_picks_until_my_turn_is_negative_one_when_no_turns_remain():
    svc = make_service(league_size=8, my_pos=1, rounds=10)
    # Fill everything after slot 1's final turn
    total = 8 * 10
    for p in range(1, total + 1):
        svc.record_pick(player_id=p, player_name=f"P{p}", position="RB", nfl_team="X")
    assert svc.draft_complete
    assert svc.my_next_pick_number is None
    assert svc.picks_until_my_turn == -1


# ---------------------------------------------------------------------------
# Recording / undo / completion
# ---------------------------------------------------------------------------

def test_record_pick_raises_when_complete():
    svc = make_service(league_size=8, rounds=10)
    for p in range(80):
        svc.record_pick(player_id=p, player_name=f"P{p}", position="RB", nfl_team="X")
    with pytest.raises(RuntimeError):
        svc.record_pick(player_id=999, player_name="Z", position="QB", nfl_team="Y")


def test_undo_on_empty_returns_none():
    assert make_service().undo_last_pick() is None


def test_undo_returns_last_pick_and_rewinds():
    svc = make_service()
    svc.record_pick(player_id=1, player_name="A", position="RB", nfl_team="X")
    p2 = svc.record_pick(player_id=2, player_name="B", position="WR", nfl_team="Y")
    popped = svc.undo_last_pick()
    assert popped == p2
    assert svc.current_pick_number == 2


def test_my_roster_filters_to_my_slot():
    svc = make_service(league_size=12, my_pos=1)
    svc.record_pick(player_id=1, player_name="Mine", position="RB", nfl_team="X")
    svc.record_pick(player_id=2, player_name="Theirs", position="WR", nfl_team="Y")
    assert [p.player_name for p in svc.my_roster] == ["Mine"]


# ---------------------------------------------------------------------------
# record_synced_pick (audit W2/W3)
# ---------------------------------------------------------------------------

def test_synced_pick_uses_explicit_sleeper_attribution():
    svc = make_service(league_size=12)
    # Traded pick: slot 7 owns pick 1 — local snake math would say slot 1
    p = svc.record_synced_pick(
        player_id=1, player_name="A", position="RB", nfl_team="DET",
        pick_number=1, round_number=1, team_slot=7,
    )
    assert (p.pick_number, p.round_number, p.team_slot) == (1, 1, 7)


def test_synced_pick_falls_back_to_local_inference():
    svc = make_service(league_size=12)
    p = svc.record_synced_pick(player_id=1, player_name="A", position="RB", nfl_team="DET")
    assert (p.pick_number, p.round_number, p.team_slot) == (1, 1, 1)


def test_synced_pick_raises_when_complete():
    svc = make_service(league_size=8, rounds=10)
    for p in range(80):
        svc.record_synced_pick(player_id=p, player_name=f"P{p}", position="RB", nfl_team="X")
    with pytest.raises(RuntimeError):
        svc.record_synced_pick(player_id=999, player_name="Z", position="QB", nfl_team="Y")


# ---------------------------------------------------------------------------
# restore_session (audit C2)
# ---------------------------------------------------------------------------

def test_restore_session_rehydrates_state():
    cfg = DraftConfig(league_size=10, my_draft_position=4, total_rounds=16)
    picks = [
        PickRecord(pick_number=1, round_number=1, team_slot=1,
                   player_id=11, player_name="A", position="RB", nfl_team="DET"),
        PickRecord(pick_number=2, round_number=1, team_slot=2,
                   player_id=-1, player_name="?Unknown", position="?", nfl_team="?"),
    ]
    svc = DraftStateService()
    svc.restore_session(cfg, picks)
    assert svc.is_active
    assert svc.config == cfg
    assert svc.current_pick_number == 3
    assert svc.picks[1].player_id == -1
