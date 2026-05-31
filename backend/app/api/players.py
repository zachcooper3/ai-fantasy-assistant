"""
Player endpoints.

GET /api/players          — list players with optional filters
GET /api/players/{id}     — single player detail

Author: Zach Cooper
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session

from backend.db.database import get_session
from backend.db import player_repo as repo
from backend.app.schemas import PlayerResponse, ScarcityResponse

router = APIRouter(prefix="/api/players", tags=["players"])


@router.get("", response_model=list[PlayerResponse])
def list_players(
    position: str | None = Query(default=None, description="Filter by position (RB, WR, QB, TE, DST, K)"),
    available_only: bool = Query(default=False, description="Only return undrafted players"),
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_session),
):
    """
    Returns players ordered by ADP (lowest = highest priority).
    Use ?available_only=true during a draft to see who's still on the board.
    """
    if available_only:
        players = repo.get_available_players(session, position=position)
    elif position:
        players = repo.get_players_by_position(session, position=position)
    else:
        players = repo.get_all_players(session)

    return [PlayerResponse.model_validate(p) for p in players[:limit]]


@router.get("/scarcity", response_model=ScarcityResponse)
def get_scarcity(session: Session = Depends(get_session)):
    """
    Returns the count of available players at each position.
    Used by the AI layer and the frontend to drive scarcity alerts.
    """
    counts = repo.count_available_by_position(session)
    return ScarcityResponse(
        QB=counts.get("QB", 0),
        RB=counts.get("RB", 0),
        WR=counts.get("WR", 0),
        TE=counts.get("TE", 0),
        DST=counts.get("DST", 0),
        K=counts.get("K", 0),
    )


@router.get("/{player_id}", response_model=PlayerResponse)
def get_player(player_id: int, session: Session = Depends(get_session)):
    """Returns a single player by ID."""
    player = repo.get_player_by_id(session, player_id)
    if player is None:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
    return PlayerResponse.model_validate(player)
