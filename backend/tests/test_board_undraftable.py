"""
Tests for the one asymmetry in how undraftable players are treated:
the big board lists them, everything else still excludes them.

This split is load-bearing and easy to undo by accident. The exclusion was
added because a live recommendation went to an IR player with the
designation sitting unread in the prompt; the inclusion was added because
hiding a player from the human's board makes "on IR" and "already drafted"
look identical, and stashing a returning stud in the last round is a real
move. A future refactor that unifies the two — in either direction —
breaks one of those, so both halves are pinned here.

Author: Zach Cooper
"""

import pytest
from sqlmodel import Session

from backend.db import player_repo as repo
from backend.db.models import Player
from backend.tests.conftest import make_player


@pytest.fixture
def mixed_health_db(engine):
    """One healthy RB, two who cannot play, one merely dinged."""
    with Session(engine) as db:
        db.add(make_player(1, "Healthy Back", "RB", adp=1.0, rank=1))
        db.add(make_player(2, "IR Back", "RB", adp=2.0, rank=2, injury_status="IR"))
        db.add(make_player(3, "PUP Back", "RB", adp=3.0, rank=3, injury_status="PUP"))
        db.add(make_player(4, "Dinged Back", "RB", adp=4.0, rank=4, injury_status="Questionable"))
        db.commit()
    return engine


def _start_draft(client):
    return client.post(
        "/api/draft/session",
        json={"league_size": 12, "my_draft_position": 1, "total_rounds": 15},
    )


def test_board_lists_players_who_cannot_play(client, mixed_health_db):
    _start_draft(client)
    body = client.get("/api/draft/board?limit=50").json()

    names = [p["name"] for p in body["players"]]
    assert names == ["Healthy Back", "IR Back", "PUP Back", "Dinged Back"]

    # And the designation rides along, or the frontend can't mark them.
    by_name = {p["name"]: p["injury_status"] for p in body["players"]}
    assert by_name["IR Back"] == "IR"
    assert by_name["Healthy Back"] is None


def test_board_scarcity_still_excludes_them(client, mixed_health_db):
    """
    Scarcity means *startable supply*. An IR running back is not one of the
    RBs left to fill your flex, and these counts feed the same
    compute_position_scarcity the recommendation prompt uses — so counting
    him here would also mislead the model.
    """
    _start_draft(client)
    body = client.get("/api/draft/board?limit=50").json()

    # Healthy + Questionable only.
    assert body["scarcity"]["RB"] == 2


def test_ai_candidate_pool_still_excludes_them(db, mixed_health_db):
    """
    The default must stay exclusive: only the board route opts in. This is
    the regression that matters — an IR player reaching the model's slate is
    a bug that has already shipped once.
    """
    names = [p.name for p in repo.get_top_available(db, n=50)]
    assert names == ["Healthy Back", "Dinged Back"]


def test_opting_in_is_what_changes_it(db, mixed_health_db):
    names = [p.name for p in repo.get_top_available(db, n=50, include_undraftable=True)]
    assert names == ["Healthy Back", "IR Back", "PUP Back", "Dinged Back"]


def test_questionable_is_not_treated_as_undraftable(db, mixed_health_db):
    """
    The line is "cannot play", not "carries risk". A game-time decision is a
    judgement for the drafter and for the model, not something to filter out
    on their behalf. Mirrored client-side in lib/injury.ts — see its test.
    """
    assert "Dinged Back" in [p.name for p in repo.get_top_available(db, n=50)]
    assert repo.count_available_by_position(db)["RB"] == 2


def test_undraftable_set_is_exactly_what_the_client_mirrors(db):
    """
    frontend/src/lib/injury.ts hardcodes this same set to decide whether a
    drafted player should decrement a scarcity count. There's no shared
    schema between front and back end, so this pins the value that the
    client copy has to match.
    """
    assert repo.UNDRAFTABLE_STATUSES == {"IR", "PUP", "Suspended", "Out"}


def test_drafting_an_undraftable_player_is_allowed(client, mixed_health_db):
    """
    Listing them would be pointless if you couldn't take one — stashing a
    returning stud is the entire reason they're on the board.
    """
    _start_draft(client)
    r = client.post("/api/draft/pick", json={"player_id": 2})
    assert r.status_code == 200
    assert r.json()["player_name"] == "IR Back"

    # He leaves the board, and the scarcity count — which never included
    # him — is unmoved. The client mirrors this skip in useDraft's
    # applyPickLocally.
    body = client.get("/api/draft/board?limit=50").json()
    assert "IR Back" not in [p["name"] for p in body["players"]]
    assert body["scarcity"]["RB"] == 2
