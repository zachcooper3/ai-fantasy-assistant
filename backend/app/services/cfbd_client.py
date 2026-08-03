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
import time

import httpx

logger = logging.getLogger(__name__)

BASE = "https://api.collegefootballdata.com"

# /stats/player/season with no `team` filter returns every FBS player for
# the whole category/season — thousands of rows. Confirmed live (2026-07-29):
# 20s wasn't enough and every category/season combination read-timed out.
# Bumped generously rather than finely tuned, since the actual payload size
# varies by category and there's no way to measure it from this sandbox
# (no CFBD_API_KEY available here — see module docstring).
TIMEOUT = 60.0

# A single slow response shouldn't be fatal — retry with backoff before
# giving up, same "one bad call doesn't kill the batch" stance as the rest
# of this app's ingestion scripts. Only retries timeouts/connection issues;
# an actual 4xx (bad key, bad params) fails immediately via raise_for_status.
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF = (3.0, 8.0)  # seconds before attempt 2, then attempt 3

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

    Retries on timeout/connection errors (see _MAX_ATTEMPTS/_RETRY_BACKOFF)
    before raising — an unfiltered full-season pull is a large enough
    response that a single slow attempt shouldn't be treated as a hard
    failure. Does NOT retry a non-2xx response (bad key, bad params) —
    raise_for_status fails immediately for those since a retry wouldn't help.
    """
    key = _get_api_key()
    if key is None:
        raise RuntimeError("CFBD_API_KEY is not set (or is still the .env.example placeholder).")

    params: dict[str, str | int] = {"year": year}
    if category:
        params["category"] = category

    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            resp = httpx.get(
                f"{BASE}/stats/player/season",
                params=params,
                headers={"Authorization": f"Bearer {key}"},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_error = e
            if attempt < _MAX_ATTEMPTS:
                backoff = _RETRY_BACKOFF[attempt - 1]
                logger.warning(
                    f"CFBD request timed out (attempt {attempt}/{_MAX_ATTEMPTS}, "
                    f"year={year}, category={category}) — retrying in {backoff}s: {e}"
                )
                time.sleep(backoff)

    raise last_error
