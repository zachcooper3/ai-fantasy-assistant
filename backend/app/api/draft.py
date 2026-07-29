"""
Draft session endpoints.

POST   /api/draft/session        — start or reset a draft session
GET    /api/draft/session        — get full draft state
DELETE /api/draft/session        — end/reset the session

POST   /api/draft/pick           — record a pick (marks player unavailable in DB)
DELETE /api/draft/pick           — undo the last pick (restores player availability)

GET    /api/draft/board          — big board: top available players + scarcity

Author: Zach Cooper
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session

from backend.db.database import get_session
from backend.db import player_repo as repo
from backend.app.schemas import (
    BoardResponse,
    DraftConfigRequest,
    DraftStateResponse,
    PickRequest,
    PickResponse,
    PlayerResponse,
    ScarcityResponse,
)
from backend.app.serializers import build_pick_response, build_state_response
from backend.app.services.draft_state import DraftConfig, DraftStateService
from backend.app.services.connection_manager import ConnectionManager
from backend.app.services.draft_sync import DraftSyncService

router = APIRouter(prefix="/api/draft", tags=["draft"])


# ---------------------------------------------------------------------------
# Dependencies — pull services from app.state (set in main.py lifespan)
# ---------------------------------------------------------------------------

def get_draft_service(request: Request) -> DraftStateService:
    return request.app.state.draft_service


def get_connection_manager(request: Request) -> ConnectionManager:
    return request.app.state.connection_manager


def get_sync_service(request: Request) -> DraftSyncService:
    return request.app.state.sync_service


# NOTE: the _pick_response/_state_response helpers that used to live here
# moved to backend/app/serializers.py (as build_pick_response /
# build_state_response) — they're shared with draft_sync.py and
# websocket.py, and a service importing them from a route module was a
# layering inversion that turned into a circular import once this module
# needed DraftSyncService for the session-lifecycle sync.stop() calls.


# ---------------------------------------------------------------------------
# Session routes
# ---------------------------------------------------------------------------

@router.post("/session", response_model=DraftStateResponse)
async def start_session(
    body: DraftConfigRequest,
    svc: DraftStateService = Depends(get_draft_service),
    mgr: ConnectionManager = Depends(get_connection_manager),
    sync: DraftSyncService = Depends(get_sync_service),
    db: Session = Depends(get_session),
):
    """
    Creates or resets a draft session.
    Also stops any running Sleeper sync (a poller left over from a previous
    session would pump the OLD draft's picks into this fresh one) and resets
    all player availability in SQLite so you start with a clean board.
    """
    await sync.stop()
    config = DraftConfig(
        league_size=body.league_size,
        my_draft_position=body.my_draft_position,
        total_rounds=body.total_rounds,
        scoring_format=body.scoring_format,
        qb_slots=body.qb_slots,
        rb_slots=body.rb_slots,
        wr_slots=body.wr_slots,
        te_slots=body.te_slots,
        flex_slots=body.flex_slots,
        dst_slots=body.dst_slots,
    )
    svc.start_session(config)
    repo.reset_draft_availability(db)
    await mgr.broadcast({"type": "reset"})
    return build_state_response(svc)


@router.get("/session", response_model=DraftStateResponse)
def get_session_state(svc: DraftStateService = Depends(get_draft_service)):
    """Returns the current draft state."""
    if not svc.is_active:
        raise HTTPException(status_code=404, detail="No active draft session.")
    return build_state_response(svc)


@router.delete("/session", status_code=204)
async def end_session(
    svc: DraftStateService = Depends(get_draft_service),
    mgr: ConnectionManager = Depends(get_connection_manager),
    sync: DraftSyncService = Depends(get_sync_service),
    db: Session = Depends(get_session),
):
    """Ends the session, stops any running Sleeper sync, and resets all
    player availability."""
    await sync.stop()
    svc.reset()
    repo.reset_draft_availability(db)
    await mgr.broadcast({"type": "reset"})


# ---------------------------------------------------------------------------
# Pick routes
# ---------------------------------------------------------------------------

@router.post("/pick", response_model=PickResponse)
async def record_pick(
    body: PickRequest,
    svc: DraftStateService = Depends(get_draft_service),
    mgr: ConnectionManager = Depends(get_connection_manager),
    db: Session = Depends(get_session),
):
    """
    Records the next pick in draft order.
    - Marks the player unavailable in SQLite.
    - Broadcasts a "pick" event over the WebSocket.
    """
    if not svc.is_active:
        raise HTTPException(status_code=400, detail="No active draft session.")
    if svc.draft_complete:
        raise HTTPException(status_code=400, detail="Draft is already complete.")

    # Validate player exists and is available
    player = repo.get_player_by_id(db, body.player_id)
    if player is None:
        raise HTTPException(status_code=404, detail=f"Player {body.player_id} not found.")
    if not player.is_available:
        raise HTTPException(
            status_code=409,
            detail=f"{player.name} has already been drafted.",
        )

    # Record in memory and persist availability to DB
    pick = svc.record_pick(
        player_id=player.id,
        player_name=player.name,
        position=player.position,
        nfl_team=player.team,
    )
    repo.mark_as_drafted(db, player.id)

    my_slot = svc.config.my_draft_position
    pick_resp = build_pick_response(pick, my_slot)

    await mgr.broadcast({
        "type": "pick",
        "pick": pick_resp.model_dump(),
        "state": build_state_response(svc).model_dump(),
    })

    return pick_resp


@router.delete("/pick", response_model=PickResponse)
async def undo_pick(
    svc: DraftStateService = Depends(get_draft_service),
    mgr: ConnectionManager = Depends(get_connection_manager),
    db: Session = Depends(get_session),
):
    """
    Undoes the most recent pick.
    - Restores player availability in SQLite.
    - Broadcasts an "undo" event over the WebSocket.
    """
    if not svc.is_active:
        raise HTTPException(status_code=400, detail="No active draft session.")

    pick = svc.undo_last_pick()
    if pick is None:
        raise HTTPException(status_code=400, detail="No picks to undo.")

    repo.mark_available(db, pick.player_id)

    my_slot = svc.config.my_draft_position
    pick_resp = build_pick_response(pick, my_slot)

    await mgr.broadcast({
        "type": "undo",
        "pick": pick_resp.model_dump(),
        "state": build_state_response(svc).model_dump(),
    })

    return pick_resp


# ---------------------------------------------------------------------------
# Big board
# ---------------------------------------------------------------------------

@router.get("/board", response_model=BoardResponse)
def get_board(
    limit: int = 30,
    svc: DraftStateService = Depends(get_draft_service),
    db: Session = Depends(get_session),
):
    """
    Returns the current big board: top available players by ADP plus
    positional scarcity counts. This is the primary data source for
    the frontend's main draft view.
    """
    if not svc.is_active:
        raise HTTPException(status_code=400, detail="No active draft session.")

    players = repo.get_top_available(db, n=limit)
    counts = repo.count_available_by_position(db)

    return BoardResponse(
        players=[PlayerResponse.model_validate(p) for p in players],
        scarcity=ScarcityResponse(
            QB=counts.get("QB", 0),
            RB=counts.get("RB", 0),
            WR=counts.get("WR", 0),
            TE=counts.get("TE", 0),
            DST=counts.get("DST", 0),
            K=counts.get("K", 0),
        ),
        picks_until_my_turn=svc.picks_until_my_turn,
        is_my_turn=svc.is_my_turn,
    )
