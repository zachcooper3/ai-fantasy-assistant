"""
The AI panel's Haiku/Sonnet toggle — AIService.set_model()/model_alias,
GET/POST /api/recommend/model, and persistence through DraftSession.ai_model
(draft_session_repo.set_ai_model/get_ai_model, and start_session carrying
the current choice into a fresh session).
"""

from backend.app.services.ai_service import AI_MODEL_CHOICES, AIService
from backend.db import draft_session_repo as jrepo


# ---------------------------------------------------------------------------
# AIService — the actual model switch
# ---------------------------------------------------------------------------

def test_set_model_switches_the_raw_id_used_by_recommend():
    svc = AIService.__new__(AIService)
    svc._client = None
    svc._model = "claude-haiku-4-5-20251001"

    svc.set_model("sonnet")

    assert svc.model_name == AI_MODEL_CHOICES["sonnet"]
    assert svc.model_alias == "sonnet"


def test_set_model_rejects_unknown_alias():
    svc = AIService.__new__(AIService)
    svc._client = None
    svc._model = "claude-haiku-4-5-20251001"

    try:
        svc.set_model("gpt-5")
        assert False, "expected ValueError"
    except ValueError:
        pass
    # Rejected switch must not have partially applied.
    assert svc.model_name == "claude-haiku-4-5-20251001"


def test_model_alias_reflects_whatever_model_is_actually_set():
    # Covers AIService.__new__ test doubles (conftest's `client` fixture)
    # that set _model directly without ever calling __init__ or set_model —
    # model_alias must not depend on an __init__-only side effect.
    svc = AIService.__new__(AIService)
    svc._client = None
    svc._model = AI_MODEL_CHOICES["haiku"]
    assert svc.model_alias == "haiku"


def test_model_alias_is_custom_for_an_unrecognized_override():
    # e.g. a CLAUDE_MODEL env override that isn't either toggle option —
    # must not crash or falsely claim "haiku"/"sonnet".
    svc = AIService.__new__(AIService)
    svc._client = None
    svc._model = "claude-opus-5"
    assert svc.model_alias == "custom"


# ---------------------------------------------------------------------------
# API — GET/POST /api/recommend/model
# ---------------------------------------------------------------------------

def test_get_model_reports_current_alias(client):
    r = client.get("/api/recommend/model")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "custom"  # conftest's fixture uses "test-model"
    assert set(body["choices"]) == {"haiku", "sonnet"}


def test_post_model_switches_it(client):
    r = client.post("/api/recommend/model", json={"model": "sonnet"})
    assert r.status_code == 200
    assert r.json()["model"] == "sonnet"

    # And it stuck — a second GET sees the same value.
    assert client.get("/api/recommend/model").json()["model"] == "sonnet"


def test_post_model_rejects_unknown_choice(client):
    r = client.post("/api/recommend/model", json={"model": "gpt-5"})
    assert r.status_code == 422


def test_post_model_persists_to_the_active_session(client, engine):
    client.post("/api/draft/session", json={
        "league_size": 12, "my_draft_position": 1, "total_rounds": 15,
    })
    client.post("/api/recommend/model", json={"model": "sonnet"})

    from sqlmodel import Session
    with Session(engine) as db:
        assert jrepo.get_ai_model(db) == "sonnet"


def test_post_model_is_harmless_with_no_active_session(client):
    # No /api/draft/session call first — set_ai_model's no-op path.
    r = client.post("/api/recommend/model", json={"model": "sonnet"})
    assert r.status_code == 200
    assert r.json()["model"] == "sonnet"


def test_start_session_carries_the_current_model_choice_through(client, engine):
    client.post("/api/recommend/model", json={"model": "sonnet"})
    client.post("/api/draft/session", json={
        "league_size": 12, "my_draft_position": 1, "total_rounds": 15,
    })

    from sqlmodel import Session
    with Session(engine) as db:
        # A fresh draft must not silently reset an already-chosen model
        # back to unset.
        assert jrepo.get_ai_model(db) == "sonnet"
