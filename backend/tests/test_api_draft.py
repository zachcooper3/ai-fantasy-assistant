"""
API integration tests — session lifecycle, pick/undo flow, board, and the
crash-recovery journal's write-through, via TestClient against the real
routers with an in-memory DB.
"""

from sqlmodel import Session

from backend.db import draft_session_repo as jrepo
from backend.db.models import Player


def start_session(client, **overrides):
    body = {"league_size": 12, "my_draft_position": 1, "total_rounds": 15, **overrides}
    return client.post("/api/draft/session", json=body)


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

def test_get_session_404_when_inactive(client):
    assert client.get("/api/draft/session").status_code == 404


def test_start_session_returns_state(client, seeded_players):
    r = start_session(client)
    assert r.status_code == 200
    state = r.json()
    assert state["is_active"] and state["current_pick_number"] == 1
    assert state["is_my_turn"] is True  # slot 1, pick 1


def test_invalid_draft_position_is_422(client):
    r = start_session(client, my_draft_position=14)  # 12-team league
    assert r.status_code == 422


def test_end_session_resets_everything(client, seeded_players, engine):
    start_session(client)
    client.post("/api/draft/pick", json={"player_id": 1})
    assert client.delete("/api/draft/session").status_code == 204
    assert client.get("/api/draft/session").status_code == 404
    with Session(engine) as db:
        assert jrepo.load_state(db) is None            # journal cleared
        assert db.get(Player, 1).is_available is True  # availability reset


# ---------------------------------------------------------------------------
# Picks
# ---------------------------------------------------------------------------

def test_pick_flow_and_journal_write_through(client, seeded_players, engine):
    start_session(client)

    r = client.post("/api/draft/pick", json={"player_id": 1})
    assert r.status_code == 200
    pick = r.json()
    assert pick["player_name"] == "Alpha Back" and pick["is_mine"] is True

    with Session(engine) as db:
        assert db.get(Player, 1).is_available is False
        _, journal = jrepo.load_state(db)
        assert [p.player_id for p in journal] == [1]


def test_pick_requires_active_session(client, seeded_players):
    assert client.post("/api/draft/pick", json={"player_id": 1}).status_code == 400


def test_unknown_player_404(client, seeded_players):
    start_session(client)
    assert client.post("/api/draft/pick", json={"player_id": 999}).status_code == 404


def test_double_draft_conflict_409(client, seeded_players):
    start_session(client)
    client.post("/api/draft/pick", json={"player_id": 1})
    assert client.post("/api/draft/pick", json={"player_id": 1}).status_code == 409


def test_undo_restores_availability_and_journal(client, seeded_players, engine):
    start_session(client)
    client.post("/api/draft/pick", json={"player_id": 1})
    client.post("/api/draft/pick", json={"player_id": 2})

    r = client.delete("/api/draft/pick")
    assert r.status_code == 200
    assert r.json()["player_id"] == 2

    with Session(engine) as db:
        assert db.get(Player, 2).is_available is True
        _, journal = jrepo.load_state(db)
        assert [p.player_id for p in journal] == [1]


def test_undo_with_no_picks_400(client, seeded_players):
    start_session(client)
    assert client.delete("/api/draft/pick").status_code == 400


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------

def test_board_excludes_drafted_and_counts_positions(client, seeded_players):
    start_session(client)
    client.post("/api/draft/pick", json={"player_id": 1})

    board = client.get("/api/draft/board").json()
    ids = [p["id"] for p in board["players"]]
    assert 1 not in ids and 2 in ids
    assert board["scarcity"]["RB"] == 1   # one of two RBs drafted
    assert board["scarcity"]["WR"] == 2


def test_board_requires_active_session(client):
    assert client.get("/api/draft/board").status_code == 400
