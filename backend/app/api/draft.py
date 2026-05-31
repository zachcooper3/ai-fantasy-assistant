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
from backend.app.services.draft_state import DraftConfig, DraftStateService
from backend.app.services.connection_manager import ConnectionManager

router = APIRouter(prefix="/api/draft", tags=["draft"])


# ---------------------------------------------------------------------------
# Dependencies — pull services from app.state (set in main.py lifespan)
# ---------------------------------------------------------------------------

def get_draft_service(request: Request) -> DraftStateService:
    return request.app.state.draft_service


def get_connection_manager(request: Request) -> ConnectionManager:
    return request.app.state.connection_manager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_response(pick, my_slot: int) -> PickResponse:
    return PickResponse(
        pick_number=pick.pick_number,
        round_number=pick.round_number,
        team_slot=pick.team_slot,
        player_id=pick.player_id,
        player_name=pick.player_name,
        position=pick.position,
        nfl_team=pick.nfl_team,
        is_mine=(pick.team_slot == my_slot),
    )


def _state_response(svc: DraftStateService) -> DraftStateResponse:
    my_slot = svc.config.my_draft_position
    return DraftStateResponse(
        is_active=svc.is_active,
        league_size=svc.config.league_size,
        my_draft_position=my_slot,
        total_rounds=svc.config.total_rounds,
        scoring_format=svc.config.scoring_format,
        current_pick_number=svc.current_pick_number,
        current_round=svc.current_round,
        current_team_slot=svc.current_team_slot,
        is_my_turn=svc.is_my_turn,
        picks_until_my_turn=svc.picks_until_my_turn,
        my_next_pick_number=svc.my_next_pick_number,
        draft_complete=svc.draft_complete,
        picks=[_pick_response(p, my_slot) for p in svc.picks],
        my_roster=[_pick_response(p, my_slot) for p in svc.my_roster],
    )


# ---------------------------------------------------------------------------
# Session routes
# ---------------------------------------------------------------------------

@router.post("/session", response_model=DraftStateResponse)
async def start_session(
    body: DraftConfigRequest,
    svc: DraftStateService = Depends(get_draft_service),
    mgr: ConnectionManager = Depends(get_connection_manager),
    db: Session = Depends(get_session),
):
    """
    Creates or resets a draft session.
    Also resets all player availability in SQLite so you start with a clean board.
    """
    config = DraftConfig(
        league_size=body.league_size,
        my_draft_position=body.my_draft_position,
        total_rounds=body.total_rounds,
        scoring_format=body.scoring_format,
    )
    svc.start_session(config)
    repo.reset_draft_availability(db)
    await mgr.broadcast({"type": "reset"})
    return _state_response(svc)


@router.get("/session", response_model=DraftStateResponse)
def get_session_state(svc: DraftStateService = Depends(get_draft_service)):
    """Returns the current draft state."""
    if not svc.is_active:
        raise HTTPException(status_code=404, detail="No active draft session.")
    return _state_response(svc)


@router.delete("/session", status_code=204)
async def end_session(
    svc: DraftStateService = Depends(get_draft_service),
    mgr: ConnectionManager = Depends(get_connection_manager),
    db: Session = Depends(get_session),
):
    """Ends the session and resets all player availability."""
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
    pick_resp = _pick_response(pick, my_slot)

    await mgr.broadcast({
        "type": "pick",
        "pick": pick_resp.model_dump(),
        "state": _state_response(svc).model_dump(),
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
    pick_resp = _pick_response(pick, my_slot)

    await mgr.broadcast({
        "type": "undo",
        "pick": pick_resp.model_dump(),
        "state": _state_response(svc).model_dump(),
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
