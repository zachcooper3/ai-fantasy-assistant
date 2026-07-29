"""
Shared-token auth (audit C3) — disabled by default, enforced when
APP_AUTH_TOKEN is set, applied at include_router time like main.py does.
"""

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from backend.app.auth import require_auth


@pytest.fixture
def auth_client():
    """Minimal app with one protected route, mirroring main.py's
    include_router(..., dependencies=[Depends(require_auth)]) pattern."""
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/api/ping")
    def ping():
        return {"ok": True}

    app = FastAPI()
    app.include_router(router, dependencies=[Depends(require_auth)])

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return TestClient(app)


def test_auth_disabled_when_env_unset(auth_client, monkeypatch):
    monkeypatch.delenv("APP_AUTH_TOKEN", raising=False)
    assert auth_client.get("/api/ping").status_code == 200


def test_placeholder_token_means_disabled(auth_client, monkeypatch):
    monkeypatch.setenv("APP_AUTH_TOKEN", "your_token_here")
    assert auth_client.get("/api/ping").status_code == 200


def test_missing_token_rejected_when_enabled(auth_client, monkeypatch):
    monkeypatch.setenv("APP_AUTH_TOKEN", "s3cret")
    r = auth_client.get("/api/ping")
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


def test_wrong_token_rejected(auth_client, monkeypatch):
    monkeypatch.setenv("APP_AUTH_TOKEN", "s3cret")
    r = auth_client.get("/api/ping", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_correct_token_accepted(auth_client, monkeypatch):
    monkeypatch.setenv("APP_AUTH_TOKEN", "s3cret")
    r = auth_client.get("/api/ping", headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200


def test_non_bearer_scheme_rejected(auth_client, monkeypatch):
    monkeypatch.setenv("APP_AUTH_TOKEN", "s3cret")
    r = auth_client.get("/api/ping", headers={"Authorization": "Basic s3cret"})
    assert r.status_code == 401


def test_health_stays_open(auth_client, monkeypatch):
    monkeypatch.setenv("APP_AUTH_TOKEN", "s3cret")
    assert auth_client.get("/health").status_code == 200
