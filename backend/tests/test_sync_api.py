"""
/api/sync/start and /api/sync/stop — persisting/clearing the Sleeper draft
ID so a backend restart can resume live sync instead of silently dropping
to idle (see DraftSession.sleeper_draft_id's docstring and main.py's
lifespan).

DraftSyncService.start() spawns a background task that immediately polls
the real Sleeper API — not something a unit test should trigger (no
network in this suite, see the codebase's existing convention of poking
sync_service.status directly in test_warning_fixes.py rather than going
through the real route). Start/stop are monkeypatched here to no-ops that
only touch the fields the route reads back, isolating "did the route
persist the right thing" from "does live polling work" (already covered,
without network, by test_draft_sync.py).
"""

import pytest

from backend.app.services.draft_sync import DraftSyncService
from backend.db import draft_session_repo as jrepo
from backend.db.database import get_session


@pytest.fixture(autouse=True)
def no_network_sync(monkeypatch):
    async def fake_start(self, draft_id):
        self._draft_id = draft_id
        self.status = "syncing"
        self.error = None

    async def fake_stop(self):
        self._draft_id = None
        self.status = "idle"
        self.error = None

    monkeypatch.setattr(DraftSyncService, "start", fake_start)
    monkeypatch.setattr(DraftSyncService, "stop", fake_stop)


def start_session(client, **kw):
    return client.post("/api/draft/session", json={
        "league_size": 12, "my_draft_position": 1, "total_rounds": 15, **kw,
    })


def test_start_sync_persists_the_draft_id(client, engine):
    start_session(client)
    r = client.post("/api/sync/start", json={"draft_id": "123456789"})
    assert r.status_code == 200
    assert r.json()["status"] == "syncing"

    from sqlmodel import Session
    with Session(engine) as db:
        assert jrepo.get_sleeper_draft_id(db) == "123456789"


def test_stop_sync_clears_the_persisted_draft_id(client, engine):
    start_session(client)
    client.post("/api/sync/start", json={"draft_id": "123456789"})

    r = client.delete("/api/sync/stop")
    assert r.status_code == 204

    from sqlmodel import Session
    with Session(engine) as db:
        assert jrepo.get_sleeper_draft_id(db) is None


def test_start_sync_requires_an_active_session(client):
    r = client.post("/api/sync/start", json={"draft_id": "123456789"})
    assert r.status_code == 400


def test_re_syncing_a_different_draft_overwrites_the_persisted_id(client, engine):
    start_session(client)
    client.post("/api/sync/start", json={"draft_id": "111111111"})
    client.post("/api/sync/start", json={"draft_id": "222222222"})

    from sqlmodel import Session
    with Session(engine) as db:
        assert jrepo.get_sleeper_draft_id(db) == "222222222"


def test_starting_a_new_session_clears_any_previously_synced_draft_id(client, engine):
    # A fresh POST /api/draft/session replaces the whole DraftSession row
    # (draft_session_repo.save_config) — must not let the OLD draft's synced
    # ID carry into a session that hasn't been synced yet.
    start_session(client)
    client.post("/api/sync/start", json={"draft_id": "123456789"})

    start_session(client)  # starts a brand new session

    from sqlmodel import Session
    with Session(engine) as db:
        assert jrepo.get_sleeper_draft_id(db) is None
