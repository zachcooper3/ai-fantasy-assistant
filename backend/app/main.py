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
# logging.basicConfig() itself at import time, and since it was imported
# transitively (fetch_adp.auto_refresh() -> sync_sleeper_ids, back when
# startup called that automatically — see the 2026-08-13 removal below) at
# whatever point the chain happened to fire, whichever module imported first
# was silently deciding the log format for every logger in the process.
# refresh.py's subprocess-per-step design routes around this same class of
# problem a different way, but sync_sleeper_ids can still be imported
# directly (reingest.py, manual runs), so this guard stays. See
# sync_sleeper_ids.py for the other half of this fix (its basicConfig call
# moved into its own `if __name__ == "__main__"` guard so it only applies
# when run standalone).
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.auth import auth_enabled, require_auth

from sqlmodel import Session

from backend.db.database import create_db_and_tables, engine
from backend.db import draft_session_repo
from backend.db import player_repo
from backend.app.services.draft_state import DraftStateService
from backend.app.services.connection_manager import ConnectionManager
from backend.app.services.ai_service import AIService
from backend.app.services.draft_sync import DraftSyncService
from backend.app.services import sleeper_client
from backend.app.api import players, draft, websocket, recommendations, sync, sleeper
from backend.ingestion.fetch_adp import adp_age_str, OUT_PATH


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
    auth_status = (
        "enabled (APP_AUTH_TOKEN set)"
        if auth_enabled()
        else "DISABLED — fine locally; set APP_AUTH_TOKEN before deploying"
    )
    db_path = os.getenv("DB_PATH", "data/fantasy.db")
    print("=" * 64)
    print("  Fantasy Draft Assistant — startup configuration")
    print(f"    Claude API : {claude_status}")
    print(f"    API auth   : {auth_status}")
    print(f"    Database   : {db_path}")
    print(f"    ADP data   : {adp_age_str(OUT_PATH)}")
    print("=" * 64)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create DB tables and attach services to app state
    create_db_and_tables()

    # ADP no longer auto-refreshes on startup (removed 2026-08-13). It used
    # to silently re-fetch from FantasyFootballCalculator whenever the CSV
    # looked stale — which meant a manually-curated ADP source (e.g. a
    # FantasyPros export dropped in by hand, see convert_fantasypros_
    # export.py) would get quietly overwritten on the next non-mid-draft
    # restart with no visible sign it had happened. data/raw/fantasypros_
    # adp.csv now only changes when you explicitly run `py -m backend.
    # ingestion.fetch_adp` yourself, or `--only adp` via refresh.py, which
    # excludes it from the default "refresh everything" plan for the same
    # reason. The banner below still reports how old the on-disk data is,
    # so a stale file is visible, just never silently fixed for you.
    with Session(engine) as db:
        persisted = draft_session_repo.load_state(db)
        persisted_model = draft_session_repo.get_ai_model(db)

    draft_service = DraftStateService()
    connection_manager = ConnectionManager()
    app.state.draft_service = draft_service
    app.state.connection_manager = connection_manager
    app.state.ai_service = AIService()
    app.state.sync_service = DraftSyncService(draft_service, connection_manager)

    if persisted_model is not None:
        # Resume the Haiku/Sonnet choice from before the restart, rather
        # than silently reverting to the CLAUDE_MODEL env default — see
        # DraftSession.ai_model's docstring.
        try:
            app.state.ai_service.set_model(persisted_model)
        except ValueError:
            # The persisted alias no longer exists (e.g. AI_MODEL_CHOICES
            # changed since it was saved). Not fatal — keep the built-in
            # default rather than failing startup over a stale toggle.
            print(f"Ignoring unrecognized persisted AI model {persisted_model!r}.")

    if persisted is not None:
        config, picks, started_at = persisted
        draft_service.restore_session(config, picks, started_at=started_at)
        # Re-assert availability flags from the journal. Normally already
        # correct (both are written per-pick), but this heals the one bad
        # case: a crash landing between the two writes, or an availability
        # reset that happened without the journal being cleared.
        with Session(engine) as db:
            for p in picks:
                if p.player_id != -1:
                    player_repo.mark_as_drafted(db, p.player_id)
        print(
            f"Recovered active draft session: {len(picks)} pick(s), "
            f"round {draft_service.current_round}, "
            f"pick #{draft_service.current_pick_number} on the clock. "
            "Note: Sleeper live sync does not auto-resume — POST /api/sync/start again if needed."
        )

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

# Routers — every /api route requires the shared bearer token when
# APP_AUTH_TOKEN is set (no-op otherwise; see backend/app/auth.py).
# Applied here at include time rather than per-route so a future router
# can't be added unprotected by accident. The WebSocket router handles
# auth itself (browsers can't set headers on a WS handshake — it checks
# a ?token= query param inside the endpoint), and /health stays open on
# purpose (Render's health checks need it, and it exposes nothing).
_protected = {"dependencies": [Depends(require_auth)]}
app.include_router(players.router, **_protected)
app.include_router(draft.router, **_protected)
app.include_router(recommendations.router, **_protected)
app.include_router(sync.router, **_protected)
app.include_router(sleeper.router, **_protected)
app.include_router(websocket.router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "version": app.version}
