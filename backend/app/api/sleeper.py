"""
Sleeper league/draft settings lookup.

GET /api/sleeper/prefill — given a Sleeper draft ID (and optionally a
username), returns whatever draft/league settings can be confidently
mapped onto DraftConfigRequest, for the setup form to offer as suggestions.

Author: Zach Cooper
"""

from fastapi import APIRouter, Query

from backend.app.schemas import SleeperPrefillResponse
from backend.app.services.sleeper_prefill import build_prefill

router = APIRouter(prefix="/api/sleeper", tags=["sleeper"])


@router.get("/prefill", response_model=SleeperPrefillResponse)
async def prefill(
    draft_id: str = Query(..., description="Sleeper draft ID"),
    username: str | None = Query(
        None, description="Your Sleeper username, to auto-detect your draft slot"
    ),
):
    """
    Best-effort settings lookup — never errors on a bad ID or Sleeper API
    failure. Fields that couldn't be detected come back as null; see
    SleeperPrefillResponse's docstring and the `warnings` list for why.
    """
    return await build_prefill(draft_id, username)
