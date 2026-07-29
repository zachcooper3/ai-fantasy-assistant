"""
Regression tests for the audit's W4/W7/W11/W12 fixes — position-aware name
matching, recommend() resilience, config-driven scarcity, and undo-during-
sync blocking.
"""

import asyncio

import pytest

from backend.app.services.ai_service import AIService, RecommendationContext
from backend.db import player_repo
from backend.tests.conftest import make_player


# ---------------------------------------------------------------------------
# W4 — get_player_by_name
# ---------------------------------------------------------------------------

def test_name_match_is_exact_not_substring(db):
    db.add(make_player(1, "Ken Walker", "RB"))
    db.commit()
    # The old ilike('%name%') would have matched "Ken Walker" for "Walker"
    assert player_repo.get_player_by_name(db, "Walker") is None
    assert player_repo.get_player_by_name(db, "Ken Walker").id == 1


def test_name_match_position_filter(db):
    db.add(make_player(1, "Josh Allen", "QB", "BUF"))
    db.add(make_player(2, "Josh Allen", "LB", "ARI"))
    db.commit()
    assert player_repo.get_player_by_name(db, "Josh Allen", position="QB").id == 1
    # Same name at two positions with no filter = ambiguous = None
    assert player_repo.get_player_by_name(db, "Josh Allen") is None


def test_name_match_strips_suffixes_both_ways(db):
    # Our ADP data keeps suffixes; Sleeper metadata omits them
    db.add(make_player(1, "Kenneth Walker III", "RB", "SEA"))
    db.commit()
    assert player_repo.get_player_by_name(db, "Kenneth Walker", position="RB").id == 1
    assert player_repo.get_player_by_name(db, "Kenneth Walker III", position="RB").id == 1


def test_name_match_ignores_punctuation_and_case(db):
    db.add(make_player(1, "Ja'Marr Chase", "WR", "CIN"))
    db.commit()
    assert player_repo.get_player_by_name(db, "jamarr chase", position="WR").id == 1


def test_sql_wildcards_are_not_special(db):
    db.add(make_player(1, "Alpha Back", "RB"))
    db.commit()
    assert player_repo.get_player_by_name(db, "%") is None
    assert player_repo.get_player_by_name(db, "_____ ____") is None


# ---------------------------------------------------------------------------
# W7 — recommend() never raises
# ---------------------------------------------------------------------------

def simple_ctx() -> RecommendationContext:
    return RecommendationContext(
        pick_number=1, round_number=1, my_slot=1, league_size=12,
        is_my_turn=True, picks_until_my_turn=0, my_next_pick_number=1,
        top_available=[
            {"id": 1, "rank": 1, "name": "Alpha Back", "position": "RB",
             "team": "DET", "adp": 1.5, "sleeper_id": None},
            {"id": 2, "rank": 2, "name": "Bravo Wide", "position": "WR",
             "team": "KC", "adp": 2.1, "sleeper_id": None},
        ],
        available_counts={"RB": 1, "WR": 1},
    )


class _FakeMessages:
    def __init__(self, behavior):
        self._behavior = behavior

    async def create(self, **kwargs):
        return self._behavior()


class _FakeClient:
    def __init__(self, behavior):
        self.messages = _FakeMessages(behavior)


def make_service_with(behavior) -> AIService:
    svc = AIService.__new__(AIService)  # skip __init__ (env-dependent)
    svc._client = _FakeClient(behavior)
    svc._model = "test-model"
    return svc


def test_unexpected_exception_falls_back_not_raises():
    def boom():
        raise ConnectionResetError("socket died mid-call")

    result = asyncio.run(make_service_with(boom).recommend(simple_ctx()))
    assert result.model.endswith(":fallback")
    assert result.recommendation.player_id == 1  # top ADP


def test_empty_content_falls_back():
    class Response:
        content = []

    result = asyncio.run(make_service_with(Response).recommend(simple_ctx()))
    assert result.model.endswith(":fallback")


def test_non_text_first_block_is_skipped_not_crashed():
    class ToolBlock:
        pass  # no .text attribute

    class TextBlock:
        text = (
            '{"recommendation": {"player_id": 2, "player_name": "Bravo Wide", '
            '"position": "WR", "adp": 2.1, "reasoning": "ok"}, '
            '"alternatives": [], "alerts": []}'
        )

    class Response:
        content = [ToolBlock(), TextBlock()]

    result = asyncio.run(make_service_with(Response).recommend(simple_ctx()))
    assert not result.model.endswith(":fallback")
    assert result.recommendation.player_id == 2


def test_no_client_uses_fallback_with_alternatives():
    svc = AIService.__new__(AIService)
    svc._client = None
    svc._model = "test-model"
    result = asyncio.run(svc.recommend(simple_ctx()))
    assert result.model.endswith(":fallback")
    assert [a.player_id for a in result.alternatives] == [2]


# ---------------------------------------------------------------------------
# W7 (route) + recommend endpoint behavior
# ---------------------------------------------------------------------------

def start(client, **kw):
    return client.post("/api/draft/session", json={
        "league_size": 12, "my_draft_position": 1, "total_rounds": 15, **kw,
    })


def test_recommend_route_returns_fallback(client, seeded_players):
    start(client)
    r = client.get("/api/recommend/pick")
    assert r.status_code == 200
    body = r.json()
    assert body["model"].endswith(":fallback")
    assert body["recommendation"]["player_id"] == 1  # best ADP on the board


def test_recommend_route_404_when_board_empty(client):
    # Session active but zero players ingested — used to be a 500
    start(client)
    assert client.get("/api/recommend/pick").status_code == 404


def test_recommend_route_400_without_session(client, seeded_players):
    assert client.get("/api/recommend/pick").status_code == 400


# ---------------------------------------------------------------------------
# W11 — scarcity uses live session config
# ---------------------------------------------------------------------------

def test_scarcity_uses_session_league_size_and_lineup(client, seeded_players):
    # 8-team league that starts no DSTs at all
    start(client, league_size=8, dst_slots=0)
    body = client.get("/api/recommend/scarcity").json()
    positions = {a["position"] for a in body["alerts"]}
    assert "DST" not in positions  # league doesn't start one

    # 2 QBs... wait, seeded data has 1 QB; demand = 8 teams x 1 slot = 8,
    # critical threshold = 4 -> 1 available is critical.
    qb = next(a for a in body["alerts"] if a["position"] == "QB")
    assert qb["tier"] == "critical"


def test_scarcity_falls_back_to_standard_without_session(client, seeded_players):
    body = client.get("/api/recommend/scarcity").json()
    positions = {a["position"] for a in body["alerts"]}
    assert positions == {"QB", "RB", "WR", "TE", "DST", "K"}


# ---------------------------------------------------------------------------
# W12 — undo blocked while sync is live
# ---------------------------------------------------------------------------

def test_undo_blocked_while_syncing(client, seeded_players):
    start(client)
    client.post("/api/draft/pick", json={"player_id": 1})

    client.app.state.sync_service.status = "syncing"
    r = client.delete("/api/draft/pick")
    assert r.status_code == 409
    assert "sync" in r.json()["detail"].lower()

    client.app.state.sync_service.status = "idle"
    assert client.delete("/api/draft/pick").status_code == 200


# ---------------------------------------------------------------------------
# Minor — board limit bounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_limit", [-1, 0, 401])
def test_board_limit_bounds(client, seeded_players, bad_limit):
    start(client)
    assert client.get(f"/api/draft/board?limit={bad_limit}").status_code == 422
