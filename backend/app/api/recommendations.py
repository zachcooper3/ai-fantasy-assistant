"""
Recommendation endpoints.

GET /api/recommend/pick                    — AI pick for the current draft state
GET /api/recommend/handcuff?player_id=X   — handcuff target for a drafted RB
GET /api/recommend/scarcity               — positional scarcity analysis

Author: Zach Cooper
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlmodel import Session

from backend.db.database import get_session
from backend.db import player_repo as repo
from backend.app.schemas import PlayerResponse
from backend.app.services.ai_service import AIService, RecommendationContext
from backend.app.services.draft_state import DraftStateService

router = APIRouter(prefix="/api/recommend", tags=["recommendations"])


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def get_draft_service(request: Request) -> DraftStateService:
    return request.app.state.draft_service


def get_ai_service(request: Request) -> AIService:
    return request.app.state.ai_service


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class PickSuggestionResponse(BaseModel):
    player_id: int
    player_name: str
    position: str
    adp: float
    reasoning: str


class RecommendationResponse(BaseModel):
    recommendation: PickSuggestionResponse
    alternatives: list[PickSuggestionResponse]
    alerts: list[str]
    model: str
    # Draft context echoed back for the frontend
    pick_number: int
    is_my_turn: bool
    picks_until_my_turn: int


class ScarcityAlert(BaseModel):
    position: str
    available: int
    tier: str           # "critical", "low", "ok"
    message: str


class ScarcityAnalysisResponse(BaseModel):
    alerts: list[ScarcityAlert]
    available_counts: dict[str, int]


# ---------------------------------------------------------------------------
# Context builder — assembles data from draft state + DB
# ---------------------------------------------------------------------------

def _build_context(
    svc: DraftStateService,
    db: Session,
    top_n: int = 25,
) -> RecommendationContext:
    """Builds the full RecommendationContext from live draft state and DB."""

    # Top available players as plain dicts
    top_available = [
        {
            "id": p.id,
            "rank": p.rank,
            "name": p.name,
            "position": p.position,
            "team": p.team,
            "adp": p.adp,
        }
        for p in repo.get_top_available(db, n=top_n)
    ]

    # My roster as plain dicts
    my_roster = [
        {"player_name": pick.player_name, "position": pick.position, "nfl_team": pick.nfl_team}
        for pick in svc.my_roster
    ]

    # Opponent position counts (exclude my slot)
    all_rosters = svc.all_rosters()
    opponent_position_counts = {
        slot: svc.position_counts_for_slot(slot)
        for slot in all_rosters
        if slot != svc.config.my_draft_position
    }

    return RecommendationContext(
        pick_number=svc.current_pick_number,
        round_number=svc.current_round,
        my_slot=svc.config.my_draft_position,
        league_size=svc.config.league_size,
        is_my_turn=svc.is_my_turn,
        picks_until_my_turn=svc.picks_until_my_turn,
        my_next_pick_number=svc.my_next_pick_number,
        scoring_format=svc.config.scoring_format,
        my_roster=my_roster,
        top_available=top_available,
        available_counts=repo.count_available_by_position(db),
        opponent_position_counts=opponent_position_counts,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/pick", response_model=RecommendationResponse)
async def recommend_pick(
    svc: DraftStateService = Depends(get_draft_service),
    ai: AIService = Depends(get_ai_service),
    db: Session = Depends(get_session),
):
    """
    Returns an AI-generated pick recommendation for the current draft state.

    Uses Claude Haiku for speed. Falls back to top-ADP logic if the API key
    is not set or if the API call fails.
    """
    if not svc.is_active:
        raise HTTPException(status_code=400, detail="No active draft session.")
    if svc.draft_complete:
        raise HTTPException(status_code=400, detail="Draft is complete.")

    ctx = _build_context(svc, db)
    result = await ai.recommend(ctx)

    return RecommendationResponse(
        recommendation=PickSuggestionResponse(**result.recommendation.__dict__),
        alternatives=[PickSuggestionResponse(**a.__dict__) for a in result.alternatives],
        alerts=result.alerts,
        model=result.model,
        pick_number=ctx.pick_number,
        is_my_turn=ctx.is_my_turn,
        picks_until_my_turn=ctx.picks_until_my_turn,
    )


@router.get("/handcuff", response_model=PlayerResponse)
def recommend_handcuff(
    player_id: int = Query(description="ID of the RB you've already drafted"),
    db: Session = Depends(get_session),
):
    """
    Returns the best available handcuff target for a drafted RB.
    A handcuff is the next-best available RB on the same NFL team.

    Only meaningful for RBs — returns 404 for other positions or if
    no handcuff is available on the roster.
    """
    player = repo.get_player_by_id(db, player_id)
    if player is None:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found.")
    if player.position != "RB":
        raise HTTPException(
            status_code=400,
            detail=f"{player.name} is a {player.position}, not an RB. Handcuffs are only for RBs.",
        )

    handcuff = repo.get_handcuff(db, player_id)
    if handcuff is None:
        raise HTTPException(
            status_code=404,
            detail=f"No available handcuff found for {player.name} ({player.team}).",
        )

    return PlayerResponse.model_validate(handcuff)


@router.get("/scarcity", response_model=ScarcityAnalysisResponse)
def analyze_scarcity(db: Session = Depends(get_session)):
    """
    Returns a positional scarcity analysis with tiered alerts.

    Thresholds (based on a 12-team, 15-round PPR draft):
      critical  — fewer players than half the league has at that position
      low       — fewer than 1.5x the league size
      ok        — above that

    These thresholds are approximate guides, not hard rules.
    """
    counts = repo.count_available_by_position(db)

    # Expected starter counts per team for a standard PPR roster
    # (1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX ~= 1 RB or WR, 1 K, 1 DST)
    starter_slots = {"QB": 1, "RB": 3, "WR": 3, "TE": 1, "DST": 1, "K": 1}
    league_size = 12  # default; in future read from draft config

    alerts: list[ScarcityAlert] = []
    for pos, slots in starter_slots.items():
        available = counts.get(pos, 0)
        # Players needed to fill starter slots for all remaining teams
        teams_needing = league_size * slots
        critical_threshold = teams_needing // 2
        low_threshold = int(teams_needing * 1.5)

        if available <= critical_threshold:
            tier = "critical"
            msg = f"Only {available} {pos}s left — critical scarcity. Consider drafting one now."
        elif available <= low_threshold:
            tier = "low"
            msg = f"{available} {pos}s remaining — supply is thinning."
        else:
            tier = "ok"
            msg = f"{pos} supply is healthy ({available} available)."

        alerts.append(ScarcityAlert(position=pos, available=available, tier=tier, message=msg))

    # Sort: critical first, then low, then ok
    tier_order = {"critical": 0, "low": 1, "ok": 2}
    alerts.sort(key=lambda a: tier_order[a.tier])

    return ScarcityAnalysisResponse(alerts=alerts, available_counts=counts)
