"""
Shared-token API auth.

This app has exactly one user, so auth is a single shared bearer token:
set APP_AUTH_TOKEN on the backend, give the same value to the frontend
(NEXT_PUBLIC_API_TOKEN), and every /api route plus the WebSocket requires
it. Unset (the local-dev default), everything behaves exactly as before —
no header needed, nothing to configure.

Why this exists (audit C3): without it, a deployed backend (Render) lets
anyone on the internet reset the draft mid-round (total state loss),
spam /api/recommend/pick (each call is a paid Claude call, unmetered), or
attach the session to an arbitrary Sleeper draft. CORS does not prevent
any of that — it's a browser courtesy, ignored by curl and scripts.

Honest limitation, documented rather than hidden: the frontend token is
baked into the public JS bundle (NEXT_PUBLIC_*), so anyone who loads the
deployed frontend page can extract it. This protects the backend from
scanners/bots and anyone who doesn't have the frontend URL — treat that
URL itself as semi-secret. Real per-user auth is deliberately out of
scope for a single-user tool.

The token is read from the environment on every check (not cached at
import) so it works regardless of when/how .env gets loaded, and uses
secrets.compare_digest to avoid timing side-channels.

Author: Zach Cooper
"""

import logging
import os
import secrets

from fastapi import HTTPException, Request, WebSocket

logger = logging.getLogger(__name__)

# Same fill-in-the-blank convention as ANTHROPIC_API_KEY/CFBD_API_KEY —
# an unedited placeholder means "not configured," never a literal token.
_PLACEHOLDER_TOKENS = {"your_token_here"}


def configured_token() -> str | None:
    """Returns the active token, or None if auth is disabled (unset or
    still the .env.example placeholder)."""
    token = (os.getenv("APP_AUTH_TOKEN") or "").strip()
    if not token or token in _PLACEHOLDER_TOKENS:
        return None
    return token


def auth_enabled() -> bool:
    return configured_token() is not None


def _matches(candidate: str, expected: str) -> bool:
    return bool(candidate) and secrets.compare_digest(candidate, expected)


async def require_auth(request: Request) -> None:
    """
    FastAPI dependency for HTTP routes — attach at include_router time so
    every /api endpoint is covered without per-route boilerplate (see
    main.py). No-op when APP_AUTH_TOKEN is unset.

    Expects: Authorization: Bearer <token>
    """
    expected = configured_token()
    if expected is None:
        return

    header = request.headers.get("authorization", "")
    candidate = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not _matches(candidate, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def ws_token_valid(websocket: WebSocket) -> bool:
    """
    WebSocket variant — browsers can't set an Authorization header on a
    WebSocket handshake, so the token rides a ?token= query param instead
    (see useDraft.ts). Returns True when auth is disabled. The caller
    closes the socket on False (see backend/app/api/websocket.py) —
    returning bool instead of raising keeps the close-code choice at the
    endpoint, where the connection object lives.
    """
    expected = configured_token()
    if expected is None:
        return True
    return _matches(websocket.query_params.get("token", ""), expected)
