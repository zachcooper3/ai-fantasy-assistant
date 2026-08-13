"""
Player endpoints.

GET /api/players             — list players with optional filters
GET /api/players/scarcity    — available counts per position
GET /api/players/{id}        — a single player's ADP row
GET /api/players/{id}/detail — that row plus metrics, draft capital, schedule

Author: Zach Cooper
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlmodel import Session, select

from backend.db.database import get_session
from backend.db import player_repo as repo
from backend.db import draft_profile_repo, game_repo, metrics_repo
from backend.db.models import DraftProfile, PlayerMetrics
from backend.app.schemas import (
    DraftProfileResponse,
    PlayerDetailResponse,
    PlayerMetricsResponse,
    PlayerResponse,
    ScarcityResponse,
    ScheduleGameResponse,
)

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


def _infer_current_season(session: Session) -> int | None:
    """
    The season being drafted for, inferred from the data rather than the clock.

    Same rule as ai_service._infer_current_season, which does this over
    already-loaded dicts during prompt assembly: PlayerMetrics holds the
    *prior* completed season, so +1; the newest DraftProfile.draft_year is
    the class that just came in, so as-is; the later of the two wins. Kept
    as two small aggregate queries here rather than importing that function,
    which wants the caller's in-memory context and would drag the whole AI
    service into a plain player lookup.

    Returns None when neither table has anything to go on. Callers must treat
    that as "unknown" — the schedule section is omitted rather than guessed,
    for the reason in Game's docstring: the schedule for the season being
    drafted is published after Claude's cutoff, and a wrong opponent is worse
    than no opponent.
    """
    latest_metrics_season = session.exec(select(func.max(PlayerMetrics.season))).one()
    latest_draft_year = session.exec(select(func.max(DraftProfile.draft_year))).one()

    candidates = [
        s for s in (
            latest_metrics_season + 1 if latest_metrics_season else None,
            latest_draft_year,
        ) if s
    ]
    return max(candidates) if candidates else None


@router.get("/{player_id}/detail", response_model=PlayerDetailResponse)
def get_player_detail(player_id: int, session: Session = Depends(get_session)):
    """
    Everything the app knows about one player — ADP row, NFL metrics, draft
    capital and college production, and the team's upcoming schedule.

    Backs the detail drawer. All of this data already existed and already fed
    the recommendation prompt; none of it was reachable from the UI, which
    made the AI's picks impossible to audit against their own inputs.

    Registered above GET /{player_id}: FastAPI matches routes in declaration
    order, and while "/{player_id}/detail" can't be swallowed by the
    single-segment "/{player_id}" pattern, keeping the more specific path
    first is the habit that stops the next such addition from being a
    silent 422.

    A missing metrics or draft-profile row is a 200 with a null field, not a
    404 — a rookie has no NFL metrics by construction and an undrafted player
    has no draft capital, and neither is an error worth interrupting a live
    draft for. Only an unknown player_id is a 404.
    """
    player = repo.get_player_by_id(session, player_id)
    if player is None:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")

    metrics = metrics_repo.get_metrics(session, player_id)
    profile = draft_profile_repo.get_draft_profile(session, player_id)

    season = _infer_current_season(session)
    schedule: list[ScheduleGameResponse] = []
    if season is not None and player.team:
        schedule = [
            ScheduleGameResponse(**g)
            for g in game_repo.get_remaining_schedule(
                session, team=player.team, season=season, from_week=1
            )
        ]

    return PlayerDetailResponse(
        player=PlayerResponse.model_validate(player),
        metrics=PlayerMetricsResponse.model_validate(metrics) if metrics else None,
        draft_profile=(
            DraftProfileResponse.model_validate(profile) if profile else None
        ),
        schedule=schedule,
        season=season,
    )


@router.get("/{player_id}", response_model=PlayerResponse)
def get_player(player_id: int, session: Session = Depends(get_session)):
    """Returns a single player by ID."""
    player = repo.get_player_by_id(session, player_id)
    if player is None:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
    return PlayerResponse.model_validate(player)
