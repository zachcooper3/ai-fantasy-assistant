"""
Async HTTP client for the Sleeper API.

Sleeper API is public — no auth required for read-only operations.
Base URL: https://api.sleeper.app/v1

A single persistent httpx.AsyncClient is used for all requests so that
TCP connections and TLS sessions are reused across polls, avoiding the
expensive per-request handshake overhead.

Call `await close()` during app shutdown to release the connection pool.

Author: Zach Cooper
"""

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE = "https://api.sleeper.app/v1"
TIMEOUT = 10.0  # seconds

# ---------------------------------------------------------------------------
# Persistent client — one connection pool for all Sleeper API calls
# ---------------------------------------------------------------------------

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Returns the shared client, creating it on first use."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=TIMEOUT,
            headers={"User-Agent": "ai-fantasy-assistant/1.0"},
        )
        logger.debug("Created persistent Sleeper API client")
    return _client


async def close() -> None:
    """Close the persistent client. Should be called from app lifespan cleanup."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        logger.debug("Closed Sleeper API client")
    _client = None


async def _get(path: str, bust_cache: bool = False) -> Any:
    """Makes a GET request to the Sleeper API and returns parsed JSON.

    Pass bust_cache=True on polling endpoints to append a millisecond
    timestamp query param, bypassing any CDN cache on Sleeper's side.
    """
    url = f"{BASE}{path}"
    if bust_cache:
        sep = "&" if "?" in path else "?"
        url = f"{url}{sep}_={int(time.time() * 1000)}"
    client = _get_client()
    resp = await client.get(url)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------

async def get_user(username: str) -> dict:
    return await _get(f"/user/{username}")


# ---------------------------------------------------------------------------
# League
# ---------------------------------------------------------------------------

async def get_league(league_id: str) -> dict:
    return await _get(f"/league/{league_id}")


async def get_league_drafts(league_id: str) -> list[dict]:
    return await _get(f"/league/{league_id}/drafts")


# ---------------------------------------------------------------------------
# Draft
# ---------------------------------------------------------------------------

async def get_draft(draft_id: str) -> dict:
    """
    Returns draft metadata including:
    - draft_order: {user_id: slot} mapping
    - settings: rounds, pick_timer, scoring_type
    - status: "pre_draft" | "drafting" | "complete"
    - slot_to_roster_id: {slot: roster_id}
    """
    return await _get(f"/draft/{draft_id}")


async def get_draft_picks(draft_id: str) -> list[dict]:
    """
    Returns all picks made so far in a draft, ordered by pick_no (ascending).
    Cache is busted on every call so Sleeper's CDN returns a fresh response.

    Each pick has:
      player_id   — Sleeper's internal player ID (string)
      picked_by   — user_id of the drafter
      roster_id   — which roster slot (1-indexed)
      draft_slot  — drafter's slot in the draft order (1-indexed)
      round       — round number (1-indexed)
      pick_no     — overall pick number (1-indexed)
      metadata    — {first_name, last_name, position, team, ...}
    """
    return await _get(f"/draft/{draft_id}/picks", bust_cache=True)


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------

async def get_nfl_players() -> dict[str, dict]:
    """
    Returns the full Sleeper NFL player database as {player_id: player_data}.
    This is a large payload (~7 MB). Cache it — it changes infrequently.
    """
    return await _get("/players/nfl")
