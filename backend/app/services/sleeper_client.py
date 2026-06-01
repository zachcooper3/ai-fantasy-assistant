"""
Async HTTP client for the Sleeper API.

Sleeper API is public — no auth required for read-only operations.
Base URL: https://api.sleeper.app/v1

Key endpoints used:
  GET /user/{username}                → resolve username to user_id
  GET /league/{league_id}/drafts      → list drafts for a league
  GET /draft/{draft_id}               → draft metadata (order, settings)
  GET /draft/{draft_id}/picks         → all picks made so far
  GET /players/nfl                    → full player database (~7 MB, cache daily)

Author: Zach Cooper
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE = "https://api.sleeper.app/v1"
TIMEOUT = 10.0  # seconds


async def _get(path: str) -> Any:
    """Makes a GET request to the Sleeper API and returns parsed JSON."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(f"{BASE}{path}")
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------

async def get_user(username: str) -> dict:
    """
    Returns Sleeper user info for a given username.
    Useful for resolving a username to a user_id.
    """
    return await _get(f"/user/{username}")


# ---------------------------------------------------------------------------
# League
# ---------------------------------------------------------------------------

async def get_league(league_id: str) -> dict:
    """Returns league metadata — name, roster positions, scoring settings."""
    return await _get(f"/league/{league_id}")


async def get_league_drafts(league_id: str) -> list[dict]:
    """
    Returns all drafts for a league.
    The most recent one is typically the current season's draft.
    """
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

    Each pick has:
      player_id   — Sleeper's internal player ID (string)
      picked_by   — user_id of the drafter
      roster_id   — which roster slot (1-indexed)
      draft_slot  — drafter's slot in the draft order (1-indexed)
      round       — round number (1-indexed)
      pick_no     — overall pick number (1-indexed)
      metadata    — {first_name, last_name, position, team, ...}
    """
    return await _get(f"/draft/{draft_id}/picks")


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------

async def get_nfl_players() -> dict[str, dict]:
    """
    Returns the full Sleeper NFL player database as {player_id: player_data}.
    This is a large payload (~7 MB). Cache it — it changes infrequently.

    Relevant fields per player:
      player_id   — Sleeper's string ID
      full_name   — "Patrick Mahomes"
      first_name  — "Patrick"
      last_name   — "Mahomes"
      position    — "QB" | "RB" | "WR" | "TE" | "K" | "DEF"
      team        — "KC" (NFL team abbreviation, or None if FA)
    """
    return await _get("/players/nfl")
