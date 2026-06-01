"""
Sleeper draft sync endpoints.

POST /api/sync/start   — begin polling a Sleeper draft
DELETE /api/sync/stop  — stop polling
GET  /api/sync/status  — current sync status

Author: Zach Cooper
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from backend.app.services.draft_sync import DraftSyncService
from backend.app.services.draft_state import DraftStateService

router = APIRouter(prefix="/api/sync", tags=["sync"])


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def get_sync_service(request: Request) -> DraftSyncService:
    return request.app.state.sync_service

def get_draft_service(request: Request) -> DraftStateService:
    return request.app.state.draft_service


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SyncStartRequest(BaseModel):
    draft_id: str


class SyncStatusResponse(BaseModel):
    status: str           # "idle" | "syncing" | "complete" | "error"
    draft_id: str | None
    synced_pick_count: int
    error: str | None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/start", response_model=SyncStatusResponse)
async def start_sync(
    body: SyncStartRequest,
    sync: DraftSyncService = Depends(get_sync_service),
    draft: DraftStateService = Depends(get_draft_service),
):
    """
    Begin polling a Sleeper draft for live picks.
    The draft session must already be active (POST /api/draft/session first).
    """
    if not draft.is_active:
        raise HTTPException(status_code=400, detail="Start a draft session first.")

    await sync.start(body.draft_id)

    return SyncStatusResponse(
        status=sync.status,
        draft_id=sync._draft_id,
        synced_pick_count=sync._synced_pick_count,
        error=sync.error,
    )


@router.delete("/stop", status_code=204)
async def stop_sync(sync: DraftSyncService = Depends(get_sync_service)):
    """Stop the Sleeper polling task."""
    await sync.stop()


@router.get("/status", response_model=SyncStatusResponse)
def get_status(sync: DraftSyncService = Depends(get_sync_service)):
    """Returns the current sync status."""
    return SyncStatusResponse(
        status=sync.status,
        draft_id=sync._draft_id,
        synced_pick_count=sync._synced_pick_count,
        error=sync.error,
    )
