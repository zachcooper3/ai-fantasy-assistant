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
from backend.app.api import players, draft, websocket, recommendations


# ---------------------------------------------------------------------------
# Lifespan — runs on startup and shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create DB tables and attach services to app state
    create_db_and_tables()
    app.state.draft_service = DraftStateService()
    app.state.connection_manager = ConnectionManager()
    app.state.ai_service = AIService()
    print("Fantasy Draft Assistant API is ready.")
    yield
    # Shutdown: nothing to clean up for now


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
app.include_router(websocket.router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "version": app.version}
