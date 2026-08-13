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

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlmodel import Session

from backend.db.database import get_session
from backend.db import player_repo as repo
from backend.db import draft_session_repo
from backend.app.schemas import (
    BoardResponse,
    DraftConfigRequest,
    DraftStateResponse,
    PickRequest,
    PickResponse,
    PlayerResponse,
    ScarcityResponse,
)
from backend.app.serializers import build_pick_response, build_state_response, state_payload
from backend.app.services.draft_state import DraftConfig, DraftStateService
from backend.app.services.connection_manager import ConnectionManager
from backend.app.services.draft_sync import DraftSyncService
from backend.app.services.ai_service import AIService

router = APIRouter(prefix="/api/draft", tags=["draft"])


# ---------------------------------------------------------------------------
# Dependencies — pull services from app.state (set in main.py lifespan)
# ---------------------------------------------------------------------------

def get_draft_service(request: Request) -> DraftStateService:
    return request.app.state.draft_service


def get_connection_manager(request: Request) -> ConnectionManager:
    return request.app.state.connection_manager


def get_ai_service(request: Request) -> AIService:
    return request.app.state.ai_service


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
    ai: AIService = Depends(get_ai_service),
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
    # Carry the currently-active Haiku/Sonnet choice into the new session
    # row — otherwise starting a fresh draft after switching models would
    # silently reset the persisted value back to unset (None) on the very
    # next restart, even though the toggle itself keeps showing your choice
    # in memory until then.
    draft_session_repo.save_config(db, config, ai_model=ai.model_alias)
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
    draft_session_repo.clear(db)
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
    draft_session_repo.append_pick(db, pick)  # journal for crash recovery

    my_slot = svc.config.my_draft_position
    pick_resp = build_pick_response(pick, my_slot)

    await mgr.broadcast({
        "type": "pick",
        "pick": pick_resp.model_dump(mode="json"),
        "state": state_payload(svc),
    })

    return pick_resp


@router.delete("/pick", response_model=PickResponse)
async def undo_pick(
    svc: DraftStateService = Depends(get_draft_service),
    mgr: ConnectionManager = Depends(get_connection_manager),
    sync: DraftSyncService = Depends(get_sync_service),
    db: Session = Depends(get_session),
):
    """
    Undoes the most recent pick.
    - Restores player availability in SQLite.
    - Broadcasts an "undo" event over the WebSocket.

    Blocked while Sleeper live sync is active (audit W12): the pick still
    exists in the real Sleeper draft, and the sync cursor has already moved
    past it, so undoing locally creates a permanent divergence — the player
    shows available here while actually drafted, and sync will never
    re-record them. Stop sync first if a synced pick truly needs fixing.
    """
    if not svc.is_active:
        raise HTTPException(status_code=400, detail="No active draft session.")
    if sync.status == "syncing":
        raise HTTPException(
            status_code=409,
            detail=(
                "Undo is disabled while Sleeper live sync is active — the pick "
                "exists in the real draft and local state would permanently "
                "diverge. Stop sync first (DELETE /api/sync/stop)."
            ),
        )

    pick = svc.undo_last_pick()
    if pick is None:
        raise HTTPException(status_code=400, detail="No picks to undo.")

    repo.mark_available(db, pick.player_id)
    draft_session_repo.remove_pick(db, pick.pick_number)

    my_slot = svc.config.my_draft_position
    pick_resp = build_pick_response(pick, my_slot)

    await mgr.broadcast({
        "type": "undo",
        "pick": pick_resp.model_dump(mode="json"),
        "state": state_payload(svc),
    })

    return pick_resp


# ---------------------------------------------------------------------------
# Big board
# ---------------------------------------------------------------------------

@router.get("/board", response_model=BoardResponse)
def get_board(
    limit: int = Query(default=30, ge=1, le=400),
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

    # The board is the human's view, so it lists players who cannot currently
    # play (IR/PUP/Suspended/Out) rather than silently omitting them — see
    # get_top_available's include_undraftable docstring. The frontend marks
    # them; stashing one late is a legitimate move.
    #
    # `counts` deliberately does NOT opt in. Scarcity means *startable
    # supply* — an IR running back is not one of the RBs left to fill your
    # flex — and these counts feed the same compute_position_scarcity the
    # recommendation prompt uses, so they must keep meaning what the AI
    # thinks they mean.
    players = repo.get_top_available(db, n=limit, include_undraftable=True)
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
