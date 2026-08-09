"""
Shared pytest fixtures.

Design notes:
  - Every test runs against a fresh in-memory SQLite engine (StaticPool so
    all sessions in one test share the same in-memory DB). The real
    data/fantasy.db is never touched.
  - API tests use a purpose-built FastAPI app (`client` fixture) that
    includes the real routers and real services but skips main.py's
    lifespan entirely — no ADP auto-refresh, no network, no rehydration.
  - Async service methods are exercised with asyncio.run() inside sync
    tests, so there's no pytest-asyncio dependency to keep in sync.

Author: Zach Cooper
"""

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import backend.db.models  # noqa: F401 — registers all tables on SQLModel.metadata
from backend.db.models import Player


@pytest.fixture
def engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def db(engine):
    with Session(engine) as session:
        yield session


def make_player(
    id: int,
    name: str,
    position: str = "RB",
    team: str = "DET",
    adp: float = 10.0,
    rank: int = 1,
    sleeper_id: str | None = None,
    is_available: bool = True,
    injury_status: str | None = None,
) -> Player:
    return Player(
        id=id,
        rank=rank,
        name=name,
        team=team,
        bye=None,
        pos_rank=f"{position}{rank}",
        position=position,
        adp=adp,
        sleeper_id=sleeper_id,
        is_available=is_available,
        injury_status=injury_status,
    )


@pytest.fixture
def seeded_players(db):
    """Six players across positions, ordered by ADP."""
    players = [
        make_player(1, "Alpha Back", "RB", "DET", adp=1.5, rank=1, sleeper_id="s1"),
        make_player(2, "Bravo Wide", "WR", "KC", adp=2.1, rank=2, sleeper_id="s2"),
        make_player(3, "Charlie Back", "RB", "DET", adp=3.4, rank=3, sleeper_id="s3"),
        make_player(4, "Delta Quarter", "QB", "BUF", adp=4.0, rank=4, sleeper_id="s4"),
        make_player(5, "Echo End", "TE", "SF", adp=5.2, rank=5, sleeper_id="s5"),
        make_player(6, "Foxtrot Wide", "WR", "MIA", adp=6.6, rank=6, sleeper_id=None),
    ]
    for p in players:
        db.add(p)
    db.commit()
    return players


@pytest.fixture
def client(engine):
    """TestClient against the real routers/services, no lifespan."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.app.api import draft, players, recommendations, sync as sync_api
    from backend.app.services.ai_service import AIService
    from backend.app.services.connection_manager import ConnectionManager
    from backend.app.services.draft_state import DraftStateService
    from backend.app.services.draft_sync import DraftSyncService
    from backend.db.database import get_session

    app = FastAPI()
    app.include_router(players.router)
    app.include_router(draft.router)
    app.include_router(recommendations.router)
    app.include_router(sync_api.router)

    draft_service = DraftStateService()
    conn_mgr = ConnectionManager()
    app.state.draft_service = draft_service
    app.state.connection_manager = conn_mgr
    app.state.sync_service = DraftSyncService(draft_service, conn_mgr)
    # Build the AIService WITHOUT __init__ — the constructor reads
    # ANTHROPIC_API_KEY from the environment, and a developer shell (or CI)
    # with a real key set would construct a live client and make these
    # tests env-dependent. Fallback mode, deterministically, always.
    ai = AIService.__new__(AIService)
    ai._client = None
    ai._model = "test-model"
    app.state.ai_service = ai

    def _get_session_override():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = _get_session_override
    with TestClient(app) as tc:
        yield tc
