"""
DraftSyncService._process_pick — Sleeper attribution (audit W2), the
placeholder path for unresolvable players, the already-recorded-manually
skip, and the config-mismatch guard. No HTTP: picks are fed in as the
dicts Sleeper's API would return.
"""

import asyncio

from sqlmodel import Session

from backend.app.services.connection_manager import ConnectionManager
from backend.app.services.draft_state import DraftConfig, DraftStateService
from backend.app.services.draft_sync import DraftSyncService
from backend.db import draft_session_repo as jrepo
from backend.db.models import Player


def make_sync(league_size=12, my_pos=1, rounds=15):
    draft_svc = DraftStateService()
    draft_svc.start_session(DraftConfig(
        league_size=league_size, my_draft_position=my_pos, total_rounds=rounds,
    ))
    return DraftSyncService(draft_svc, ConnectionManager()), draft_svc


def sleeper_pick(pick_no, sleeper_id="s1", slot=None, rnd=None, first="Alpha", last="Back"):
    return {
        "pick_no": pick_no,
        "player_id": sleeper_id,
        "draft_slot": slot,
        "round": rnd,
        "metadata": {"first_name": first, "last_name": last, "position": "RB", "team": "DET"},
    }


def process(sync, pick, db):
    asyncio.run(sync._process_pick(pick, db))


def test_known_player_recorded_with_sleeper_attribution(db, seeded_players, engine):
    sync, svc = make_sync()
    # Traded pick: Sleeper says slot 7 owns pick 1
    process(sync, sleeper_pick(1, "s1", slot=7, rnd=1), db)

    pick = svc.picks[0]
    assert (pick.pick_number, pick.team_slot, pick.player_id) == (1, 7, 1)
    assert db.get(Player, 1).is_available is False
    assert sync._synced_pick_count == 1
    # Journaled for crash recovery (config not journaled here — routes own that)
    with Session(engine) as s:
        from backend.db.models import DraftPick
        from sqlmodel import select
        rows = s.exec(select(DraftPick)).all()
        assert len(rows) == 1 and rows[0].team_slot == 7


def test_unknown_player_records_placeholder(db, seeded_players):
    sync, svc = make_sync()
    process(sync, sleeper_pick(1, "unknown-id", slot=2, rnd=1, first="Nobody", last="Home"), db)

    pick = svc.picks[0]
    assert pick.player_id == -1
    assert pick.player_name == "Nobody Home"
    assert pick.team_slot == 2
    assert sync._synced_pick_count == 1


def test_name_fallback_when_sleeper_id_unmatched(db, seeded_players):
    sync, svc = make_sync()
    # sleeper_id unknown, but metadata name matches a local player exactly
    process(sync, sleeper_pick(1, "not-in-db", slot=1, rnd=1, first="Delta", last="Quarter"), db)
    assert svc.picks[0].player_id == 4  # matched by name, not placeholder


def test_manually_recorded_pick_is_skipped_not_duplicated(db, seeded_players):
    sync, svc = make_sync()
    # User entered the pick manually before sync connected
    svc.record_pick(player_id=1, player_name="Alpha Back", position="RB", nfl_team="DET")
    player = db.get(Player, 1)
    player.is_available = False
    db.add(player)
    db.commit()

    process(sync, sleeper_pick(1, "s1", slot=1, rnd=1), db)

    assert len(svc.picks) == 1               # no duplicate
    assert sync._synced_pick_count == 1      # cursor still advanced


def test_pick_beyond_local_draft_capacity_ignored(db, seeded_players):
    sync, svc = make_sync(league_size=8, rounds=10)
    for p in range(80):
        svc.record_synced_pick(player_id=1000 + p, player_name=f"P{p}", position="RB", nfl_team="X")
    assert svc.draft_complete

    # Sleeper reports an 81st pick (config mismatch) — must not raise,
    # must not record, must advance the cursor so the poll loop moves on.
    process(sync, sleeper_pick(81, "s2", slot=1, rnd=11), db)
    assert len(svc.picks) == 80
    assert sync._synced_pick_count == 81


def test_stop_resets_to_clean_idle_state(db):
    sync, _ = make_sync()
    sync.status = "error"
    sync.error = "boom"
    sync._draft_id = "123"
    sync._synced_pick_count = 7

    asyncio.run(sync.stop())

    assert sync.status == "idle"
    assert sync.error is None
    assert sync._draft_id is None
    assert sync._synced_pick_count == 0
