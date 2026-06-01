"""
FastAPI application entry point.

Run with:
    uvicorn backend.app.main:app --reload

Or (from repo root):
    python -m uvicorn backend.app.main:app --reload

Author: Zach Cooper
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.db.database import create_db_and_tables
from backend.app.services.draft_state import DraftStateService
from backend.app.services.connection_manager import ConnectionManager
from backend.app.services.ai_service import AIService
from backend.app.services.draft_sync import DraftSyncService
from backend.app.api import players, draft, websocket, recommendations, sync


# ---------------------------------------------------------------------------
# Lifespan — runs on startup and shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create DB tables and attach services to app state
    create_db_and_tables()
    draft_service = DraftStateService()
    connection_manager = ConnectionManager()
    app.state.draft_service = draft_service
    app.state.connection_manager = connection_manager
    app.state.ai_service = AIService()
    app.state.sync_service = DraftSyncService(draft_service, connection_manager)
    print("Fantasy Draft Assistant API is ready.")
    yield
    # Shutdown: stop any active Sleeper sync task cleanly
    await app.state.sync_service.stop()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fantasy Football Draft Assistant",
    description="AI-powered draft day co-pilot for Sleeper PPR leagues.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the Next.js frontend (localhost:3000 in dev, Vercel URL in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(players.router)
app.include_router(draft.router)
app.include_router(recommendations.router)
app.include_router(sync.router)
app.include_router(websocket.router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "version": app.version}
