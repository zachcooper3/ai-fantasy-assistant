"""
FastAPI application entry point.

Run with:
    uvicorn backend.app.main:app --reload

Or (from repo root):
    python -m uvicorn backend.app.main:app --reload

Author: Zach Cooper
"""

import logging
import os

# Configured once, here, before any other backend module is imported. This
# is the single source of truth for logging format across the whole running
# app — previously, backend/ingestion/sync_sleeper_ids.py called
# logging.basicConfig() itself at import time, and since it's now imported
# transitively (fetch_adp.auto_refresh() -> sync_sleeper_ids) during every
# startup, whichever module happened to import first was silently deciding
# the log format for every logger in the process. See sync_sleeper_ids.py
# for the other half of this fix (its basicConfig call moved into its own
# `if __name__ == "__main__"` guard so it only applies when run standalone).
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.db.database import create_db_and_tables
from backend.app.services.draft_state import DraftStateService
from backend.app.services.connection_manager import ConnectionManager
from backend.app.services.ai_service import AIService
from backend.app.services.draft_sync import DraftSyncService
from backend.app.services import sleeper_client
from backend.app.api import players, draft, websocket, recommendations, sync, sleeper
from backend.ingestion.fetch_adp import auto_refresh as _refresh_adp, adp_age_str, OUT_PATH


# ---------------------------------------------------------------------------
# Lifespan — runs on startup and shutdown
# ---------------------------------------------------------------------------

def _print_startup_banner(ai_service: AIService) -> None:
    """
    One-glance config summary printed on every boot, so a misconfiguration
    (missing API key, stale ADP data) is obvious immediately instead of
    being discovered later via a buried log line or unexpectedly different
    app behavior.
    """
    claude_status = (
        f"configured ({ai_service.model_name})"
        if ai_service.is_configured
        else "NOT SET — recommendations will use ADP fallback"
    )
    db_path = os.getenv("DB_PATH", "data/fantasy.db")
    print("=" * 64)
    print("  Fantasy Draft Assistant — startup configuration")
    print(f"    Claude API : {claude_status}")
    print(f"    Database   : {db_path}")
    print(f"    ADP data   : {adp_age_str(OUT_PATH)}")
    print("=" * 64)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create DB tables and attach services to app state
    create_db_and_tables()

    # Auto-refresh ADP data if the CSV is older than 7 days
    await _refresh_adp()

    draft_service = DraftStateService()
    connection_manager = ConnectionManager()
    app.state.draft_service = draft_service
    app.state.connection_manager = connection_manager
    app.state.ai_service = AIService()
    app.state.sync_service = DraftSyncService(draft_service, connection_manager)

    _print_startup_banner(app.state.ai_service)
    print("Fantasy Draft Assistant API is ready.")
    yield
    # Shutdown: stop sync task and close the persistent Sleeper HTTP client
    await app.state.sync_service.stop()
    await sleeper_client.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fantasy Football Draft Assistant",
    description="AI-powered draft day co-pilot for Sleeper PPR leagues.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the Next.js frontend. Defaults cover local dev; override with
# a comma-separated CORS_ORIGINS env var once deployed (e.g. the Vercel URL)
# so going live doesn't require a code change.
_DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://localhost:5173"
CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", _DEFAULT_CORS_ORIGINS).split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(players.router)
app.include_router(draft.router)
app.include_router(recommendations.router)
app.include_router(sync.router)
app.include_router(sleeper.router)
app.include_router(websocket.router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "version": app.version}
