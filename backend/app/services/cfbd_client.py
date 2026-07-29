"""
Thin HTTP client for the CollegeFootballData.com API.

Used only by fetch_college_stats.py (an offline ingestion script), so this
is synchronous — unlike sleeper_client.py, which is async because it's also
called from inside the live FastAPI app (draft_sync.py's polling loop).

Deliberately NOT using the official `cfbd` PyPI package — see the note in
requirements.txt. Its latest release (5.20.1, confirmed 2026-07) hard-pins
pydantic<2, which would downgrade this project's pydantic and break
FastAPI/SQLModel/pydantic-settings, all of which require pydantic 2.x.
CollegeFootballData.com is a plain REST API with Bearer-token auth, so a
direct httpx call sidesteps the conflict entirely.

Free API key, no credit card, email signup: https://collegefootballdata.com/key

Author: Zach Cooper
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

BASE = "https://api.collegefootballdata.com"
TIMEOUT = 20.0

# .env.example ships this as a fill-in-the-blank value, same convention as
# ANTHROPIC_API_KEY (see ai_service.py's build_anthropic_client /
# _PLACEHOLDER_KEYS) — treat an unedited placeholder as "not configured,"
# not a literal key to send.
_PLACEHOLDER_KEYS = {"your_key_here"}


def _get_api_key() -> str | None:
    key = (os.getenv("CFBD_API_KEY") or "").strip()
    if not key or key in _PLACEHOLDER_KEYS:
        return None
    return key


def is_configured() -> bool:
    """True if a real (non-placeholder) CFBD_API_KEY is set."""
    return _get_api_key() is not None


def get_player_season_stats(year: int, category: str | None = None) -> list[dict]:
    """
    Returns raw PlayerSeasonStat rows for a season, optionally filtered to
    one stat category (e.g. "passing", "rushing", "receiving").

    Long/narrow format — one row per (player, category, stat_type), e.g.
    {"season": 2025, "player": "Some Player", "team": "Boise State",
     "category": "rushing", "statType": "YDS", "stat": 1516}
    — see fetch_college_stats.py for how these get pivoted into per-player
    totals. Exact key casing (statType vs stat_type) hasn't been confirmed
    against a live response yet (no API key available in the dev sandbox
    that wrote this) — fetch_college_stats.py resolves it defensively,
    same stance as fetch_metrics.py takes with nflreadpy's column names.

    Raises RuntimeError if CFBD_API_KEY isn't configured — callers should
    check is_configured() first if they want to no-op gracefully instead
    (see fetch_college_stats.py's main()).
    """
    key = _get_api_key()
    if key is None:
        raise RuntimeError("CFBD_API_KEY is not set (or is still the .env.example placeholder).")

    params: dict[str, str | int] = {"year": year}
    if category:
        params["category"] = category

    resp = httpx.get(
        f"{BASE}/stats/player/season",
        params=params,
        headers={"Authorization": f"Bearer {key}"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()
