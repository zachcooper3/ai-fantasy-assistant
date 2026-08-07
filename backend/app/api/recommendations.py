"""
Recommendation endpoints.

GET /api/recommend/pick                    — AI pick for the current draft state
GET /api/recommend/pick/stream             — same, as SSE; the pick arrives well
                                             before the alternatives finish
GET /api/recommend/handcuff?player_id=X   — handcuff target for a drafted RB
GET /api/recommend/scarcity               — positional scarcity analysis

Author: Zach Cooper
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from backend.db.database import get_session
from backend.db import player_repo as repo
from backend.db import metrics_repo
from backend.db import draft_profile_repo
from backend.app.schemas import PlayerResponse
from backend.db.models import Player, PlayerMetrics
from backend.app.services.ai_service import (
    AIService,
    RecommendationContext,
    compute_position_scarcity,
    compute_replacement_levels,
    _ROSTER_CHANGE_MIN_SHARE,
    _ROSTER_CHANGE_POSITIONS,
)
from backend.app.services.draft_state import DraftStateService

logger = logging.getLogger(__name__)

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
    # Populated on alternatives only: what taking this player instead of the
    # main recommendation gains and costs. Empty string when absent (older
    # model output, or the no-AI fallback path).
    tradeoff: str = ""
    # "take_now" | "might_last" | "will_last", or "" on the last pick of the
    # draft. Computed server-side from ADP vs the horizon pick — see
    # PickSuggestion.survival — so the badge always matches the prompt.
    survival: str = ""


class RecommendationResponse(BaseModel):
    recommendation: PickSuggestionResponse
    alternatives: list[PickSuggestionResponse]
    alerts: list[str]
    model: str
    # One sentence on the roster's shape and what this pick does about it.
    strategy: str = ""
    # "high" | "medium" | "low". The ADP fallback always reports "low".
    confidence: str = "medium"
    # One line per must-evaluate player: taken, or passed and why. Surfaces
    # omission, which is how every live mis-recommendation has manifested.
    considered: list[str] = []
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

def _metrics_dict(m) -> dict:
    """
    Flattens a PlayerMetrics row into a plain dict — decouples ai_service.py
    from the SQLModel/DB layer, matching how top_available/my_roster below
    are already built as plain dicts rather than passed as ORM rows.
    Every field can legitimately be None (see PlayerMetrics' docstring in
    backend/db/models.py); ai_service.py's formatting is expected to treat
    a missing/None field as "unknown," never as zero.
    """
    return {
        "season": m.season,
        "through_week": m.through_week,
        "games_played": m.games_played,
        "targets_per_game": m.targets_per_game,
        "carries_per_game": m.carries_per_game,
        "red_zone_touches_per_game": m.red_zone_touches_per_game,
        "snap_pct": m.snap_pct,
        "target_share": m.target_share,
        "carry_share": m.carry_share,
        "yards_per_target": m.yards_per_target,
        "yards_per_carry": m.yards_per_carry,
        "yac_per_reception": m.yac_per_reception,
        "racr": m.racr,
        "catch_rate": m.catch_rate,
        "team_pass_rate": m.team_pass_rate,
        "depth_chart_rank": m.depth_chart_rank,
        "fantasy_points_avg": m.fantasy_points_avg,
        "fantasy_points_stdev": m.fantasy_points_stdev,
        "injury_report_appearances": m.injury_report_appearances,
        "games_missed": m.games_missed,
        "target_share_trend": m.target_share_trend,
        "snap_pct_trend": m.snap_pct_trend,
        "depth_chart_trend": m.depth_chart_trend,
        "is_rookie_or_second_year": m.is_rookie_or_second_year,
    }


# Team abbreviations differ between the two sources this compares:
# PlayerMetrics.team comes from nflverse, Player.team from the ADP feed.
# They agree on 31 of 32 clubs and disagree on the Rams — nflverse says LA,
# FantasyFootballCalculator says LAR — which made every Rams player look
# like he had moved from "LA" to "LAR". Confirmed live: Kyren Williams was
# reported as having left LA with 56% of its carries, alongside three other
# Rams, in a list otherwise full of genuine moves.
#
# Mapped toward the ADP feed's convention, because that is what Player.team
# holds and what the board displays — a change keyed "LA" would never match
# a board row reading "LAR". The extra aliases are historical or
# alternative codes that nflverse and other feeds have used; none appear in
# today's data, and they cost nothing to tolerate.
_TEAM_ALIASES = {
    "LA": "LAR", "STL": "LAR",          # Rams
    "SD": "LAC",                         # Chargers
    "OAK": "LV", "LVR": "LV",            # Raiders
    "WSH": "WAS",                        # Commanders
    "JAC": "JAX", "ARZ": "ARI",
    "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
}


def _normalise_team(code: str | None) -> str | None:
    """Team code in the ADP feed's convention, or None."""
    if not code:
        return None
    code = code.strip().upper()
    return _TEAM_ALIASES.get(code, code)


def _compute_roster_changes(db: Session) -> dict[str, dict[str, list[dict]]]:
    """
    Who moved between last season and now, per team, with the share of that
    team's 2025 volume they carried.

    Built by comparing PlayerMetrics.team (where the numbers were earned)
    against Player.team (where he is now). Both are already in the database;
    nothing here needs a new source.

    Runs over the WHOLE player pool, not the visible board — the departure
    that matters to a player is a teammate's, and that teammate may be
    hundreds of picks away or already drafted. Restricting to the board
    would miss exactly the case this exists for.

    Returns {} when PlayerMetrics.team is empty, which is what a database
    that hasn't re-run fetch_metrics since the column was added will look
    like. The prompt section is then omitted rather than rendering a
    confident "nobody moved".
    """
    rows = db.exec(
        select(
            Player.name, Player.position, Player.team,
            PlayerMetrics.team, PlayerMetrics.target_share, PlayerMetrics.carry_share,
        ).join(PlayerMetrics, PlayerMetrics.player_id == Player.id)
    ).all()

    changes: dict[str, dict[str, list[dict]]] = {}
    for name, position, now_team_raw, then_team_raw, tgt_share, carry_share in rows:
        if position not in _ROSTER_CHANGE_POSITIONS:
            continue
        now_team = _normalise_team(now_team_raw)
        then_team = _normalise_team(then_team_raw)
        if not now_team or not then_team or now_team == then_team:
            continue

        # Describe the move in whichever currency the player actually dealt
        # in. A back who took a third of the carries and a receiver who took
        # a quarter of the targets both moved something worth naming, but
        # they are not the same quantity and shouldn't be summed.
        share, currency = max(
            ((tgt_share or 0.0), "targets"),
            ((carry_share or 0.0), "carries"),
        )
        if share < _ROSTER_CHANGE_MIN_SHARE:
            continue
        # `share` and `currency` are kept alongside the rendered label so
        # callers can do arithmetic: sorting on the formatted string puts
        # "8% of targets" above "25%", and the per-player annotation needs
        # to net departures against arrivals in the same currency.
        entry = {"name": name, "share": share, "currency": currency,
                 "share_label": f"{share:.0%} of {currency}"}

        changes.setdefault(then_team, {"departed": [], "arrived": []})
        changes[then_team]["departed"].append(entry)
        changes.setdefault(now_team, {"departed": [], "arrived": []})
        changes[now_team]["arrived"].append({**entry, "from_team": then_team})

    # A club that appears on only one side of the diff is the signature of an
    # abbreviation mismatch, not of thirty players changing teams at once —
    # it means every player at that club looks like he moved. Logged rather
    # than corrected, because the fix is a new entry in _TEAM_ALIASES and
    # guessing it automatically would be how the next silent wrong answer
    # gets in.
    left_only = {t for t, v in changes.items() if v["departed"] and not v["arrived"]}
    suspicious = {t for t in left_only if len(changes[t]["departed"]) >= 4}
    if suspicious:
        logger.warning(
            "Roster changes: %s lost 4+ players and gained none. Usually an "
            "unmapped team abbreviation between nflverse and the ADP feed — "
            "check _TEAM_ALIASES before believing it.",
            ", ".join(sorted(suspicious)),
        )

    # Biggest mover first — the 25% departure is the one that changes a
    # recommendation, and it shouldn't be third in a list of five.
    for team in changes.values():
        for direction in ("departed", "arrived"):
            team[direction].sort(key=lambda p: p["share"], reverse=True)
    return changes


def _build_context(
    svc: DraftStateService,
    db: Session,
    top_n: int = 60,
) -> RecommendationContext:
    """
    Builds the full RecommendationContext from live draft state and DB.

    top_n is deliberately deeper than the board the prompt actually displays
    (ai_service._LISTED_PLAYERS, 25). The extra players are never rendered;
    they exist so the "cost of waiting" math can find replacement level at
    each position. On a 25-player global ADP slice, the next available TE or
    QB is frequently past the cut — which is exactly the situation where
    waiting is most expensive and the model could least see it. The cost is
    two bulk keyed lookups over 60 ids instead of 25.
    """

    # Top available players as plain dicts
    top_available = [
        {
            "id": p.id,
            "rank": p.rank,
            "name": p.name,
            "position": p.position,
            "team": p.team,
            "adp": p.adp,
            "sleeper_id": p.sleeper_id,
        }
        for p in repo.get_top_available(db, n=top_n)
    ]

    # Analytics for the same players, keyed by Player.id — see
    # ai_service.py's RecommendationContext.player_metrics docstring and
    # PlayerMetrics in backend/db/models.py. Missing from this dict simply
    # means "never computed for this player" (e.g. a rookie with no prior
    # NFL season), not zero.
    metrics_rows = metrics_repo.get_metrics_bulk(db, [p["id"] for p in top_available])
    player_metrics = {pid: _metrics_dict(m) for pid, m in metrics_rows.items()}

    # Draft-day facts for the same players — exists for rookies/recent
    # draftees specifically, who structurally can never appear in
    # player_metrics above. See ai_service.py's RecommendationContext
    # .draft_profiles docstring and DraftProfile in backend/db/models.py.
    draft_profile_rows = draft_profile_repo.get_draft_profiles_bulk(db, [p["id"] for p in top_available])
    draft_profiles = {
        pid: {
            "draft_year": dp.draft_year,
            "draft_round": dp.draft_round,
            "draft_pick": dp.draft_pick,
            "draft_team": dp.draft_team,
            "college": dp.college,
            "college_season": dp.college_season,
            "passing_yards": dp.passing_yards,
            "passing_td": dp.passing_td,
            "interceptions_thrown": dp.interceptions_thrown,
            "rushing_yards": dp.rushing_yards,
            "rushing_td": dp.rushing_td,
            "carries": dp.carries,
            "receiving_yards": dp.receiving_yards,
            "receiving_td": dp.receiving_td,
            "receptions": dp.receptions,
        }
        for pid, dp in draft_profile_rows.items()
    }

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

    # Replacement level per position, over the WHOLE player pool rather than
    # the available slice — it's a property of the position in this league
    # ("what does the worst startable back score"), not of who happens to be
    # left on the board. Drafted players still define that baseline.
    #
    # This is what lets the prompt compare a back to a receiver at all. See
    # compute_replacement_levels for the live diagnosis it came from.
    ppg_by_position: dict[str, list[float]] = {}
    for pos, ppg in db.exec(
        select(Player.position, PlayerMetrics.fantasy_points_avg)
        .join(PlayerMetrics, PlayerMetrics.player_id == Player.id)
        .where(PlayerMetrics.fantasy_points_avg.is_not(None))  # type: ignore[union-attr]
    ):
        ppg_by_position.setdefault(pos, []).append(ppg)

    cfg = svc.config
    replacement_ppg = compute_replacement_levels(
        ppg_by_position,
        cfg.league_size,
        # Per-team demand with the FLEX split evenly between RB and WR — the
        # same assumption the scarcity block makes, so the two can't disagree
        # about how many backs a league starts.
        {
            "QB": cfg.qb_slots,
            "RB": cfg.rb_slots + cfg.flex_slots / 2,
            "WR": cfg.wr_slots + cfg.flex_slots / 2,
            "TE": cfg.te_slots,
        },
    )

    # Look-ahead: the turn AFTER the one being advised on, plus every team
    # that picks in between. This is the opportunity-cost horizon the
    # recommendation prompt reasons against — see
    # RecommendationContext.my_following_pick_number for why it can't just
    # be my_next_pick_number (that property returns the *current* pick when
    # it's already my turn).
    #
    # Computed here rather than in ai_service so the snake math stays in
    # DraftStateService alone; any future variant (third-round reversal and
    # friends) changes slot_for_pick and this follows automatically.
    total_picks = svc.config.league_size * svc.config.total_rounds
    advised_pick = svc.my_next_pick_number
    my_following_pick_number = None
    upcoming_pick_slots: list[int] = []
    if advised_pick is not None:
        for p in range(advised_pick + 1, total_picks + 1):
            if svc.slot_for_pick(p) == svc.config.my_draft_position:
                my_following_pick_number = p
                break
            upcoming_pick_slots.append(svc.slot_for_pick(p))
        if my_following_pick_number is None:
            # Advising on my final pick of the draft — nothing to defer to,
            # so drop the partial list rather than implying a next turn.
            upcoming_pick_slots = []

    return RecommendationContext(
        pick_number=svc.current_pick_number,
        round_number=svc.current_round,
        my_slot=svc.config.my_draft_position,
        league_size=svc.config.league_size,
        is_my_turn=svc.is_my_turn,
        picks_until_my_turn=svc.picks_until_my_turn,
        my_next_pick_number=svc.my_next_pick_number,
        scoring_format=svc.config.scoring_format,
        total_rounds=svc.config.total_rounds,
        my_roster=my_roster,
        top_available=top_available,
        available_counts=repo.count_available_by_position(db),
        opponent_position_counts=opponent_position_counts,
        starting_lineup=svc.config.starting_lineup,
        player_metrics=player_metrics,
        draft_profiles=draft_profiles,
        my_following_pick_number=my_following_pick_number,
        upcoming_pick_slots=upcoming_pick_slots,
        replacement_ppg=replacement_ppg,
        roster_changes=_compute_roster_changes(db),
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
    if not ctx.top_available:
        # Without this, an empty board reached _fallback's RuntimeError
        # and surfaced as a 500 (audit W7) — a clean 404 tells the client
        # what actually happened.
        raise HTTPException(status_code=404, detail="No available players left to recommend.")
    result = await ai.recommend(ctx)

    return RecommendationResponse(
        recommendation=PickSuggestionResponse(**result.recommendation.__dict__),
        alternatives=[PickSuggestionResponse(**a.__dict__) for a in result.alternatives],
        alerts=result.alerts,
        model=result.model,
        strategy=result.strategy,
        confidence=result.confidence,
        considered=result.considered,
        pick_number=ctx.pick_number,
        is_my_turn=ctx.is_my_turn,
        picks_until_my_turn=ctx.picks_until_my_turn,
    )


@router.get("/pick/stream")
async def recommend_pick_stream(
    svc: DraftStateService = Depends(get_draft_service),
    ai: AIService = Depends(get_ai_service),
    db: Session = Depends(get_session),
):
    """
    Same recommendation as GET /pick, delivered as Server-Sent Events so the
    pick can be shown before the rest of the response finishes generating.

    Two event types:
        event: pick      — the recommendation alone, ~4s in
        event: complete  — the full payload, identical to GET /pick

    Total time is the same. Generation is sequential and output-bound (about
    1,660 tokens at ~75 tok/sec), so the batch endpoint shows nothing for
    twenty seconds while the answer has in fact existed since second four.
    This changes only when it becomes visible.

    GET /pick is deliberately kept: it is simpler to call, and a client that
    does not need progressive rendering should not have to parse an event
    stream to get a pick.
    """
    if not svc.is_active:
        raise HTTPException(status_code=400, detail="No active draft session.")
    if svc.draft_complete:
        raise HTTPException(status_code=400, detail="Draft is complete.")

    ctx = _build_context(svc, db)
    if not ctx.top_available:
        raise HTTPException(status_code=404, detail="No available players left to recommend.")

    def _event(name: str, payload: dict) -> str:
        return f"event: {name}\ndata: {json.dumps(payload)}\n\n"

    async def _generate():
        try:
            async for kind, value in ai.recommend_stream(ctx):
                if kind == "pick":
                    yield _event("pick", {
                        "recommendation": PickSuggestionResponse(**value.__dict__).model_dump(),
                        "pick_number": ctx.pick_number,
                    })
                else:
                    yield _event("complete", RecommendationResponse(
                        recommendation=PickSuggestionResponse(**value.recommendation.__dict__),
                        alternatives=[PickSuggestionResponse(**a.__dict__) for a in value.alternatives],
                        alerts=value.alerts,
                        model=value.model,
                        strategy=value.strategy,
                        confidence=value.confidence,
                        considered=value.considered,
                        pick_number=ctx.pick_number,
                        is_my_turn=ctx.is_my_turn,
                        picks_until_my_turn=ctx.picks_until_my_turn,
                    ).model_dump())
        except Exception:
            # The stream has already begun, so an HTTP error status is no
            # longer available — the client would see a truncated stream and
            # no reason. Send the failure as a normal event instead.
            logger.exception("Streamed recommendation failed after the response started.")
            yield _event("error", {"detail": "Recommendation failed — try again."})

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Without this, nginx-style proxies buffer the whole stream and
            # deliver it at once, which silently undoes the entire feature.
            "X-Accel-Buffering": "no",
        },
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
def analyze_scarcity(
    svc: DraftStateService = Depends(get_draft_service),
    db: Session = Depends(get_session),
):
    """
    Returns a positional scarcity analysis with tiered alerts.

    Thresholds derive from the ACTIVE session's league size and configured
    starting lineup when one exists (audit W11 — this used to hardcode a
    12-team standard roster, producing live-wrong output for 8/10/14-team
    or custom-lineup leagues); the 12-team standard shape remains only as
    the no-session fallback. FLEX demand is approximated by adding
    flex_slots to both RB and WR — the same "a flex is usually an RB or
    WR" assumption the original hardcoded 3/3 encoded.

    The critical/low/ok tier math itself lives in compute_position_scarcity
    (ai_service.py) — shared with the main recommendation prompt's
    Positional Availability section so the two never drift out of sync,
    even though each passes its own starter_slots shape.

      critical  — fewer players left than half the league's demand
      low       — fewer than 1.5x the demand
      ok        — above that

    These thresholds are approximate guides, not hard rules.
    """
    counts = repo.count_available_by_position(db)

    if svc.is_active:
        cfg = svc.config
        league_size = cfg.league_size
        starter_slots = {
            "QB": cfg.qb_slots,
            "RB": cfg.rb_slots + cfg.flex_slots,
            "WR": cfg.wr_slots + cfg.flex_slots,
            "TE": cfg.te_slots,
            "DST": cfg.dst_slots,
            "K": 1,
        }
        # A position this league doesn't start at all can't be scarce —
        # dropping it also avoids nonsense "critical: 0 needed" alerts.
        starter_slots = {pos: n for pos, n in starter_slots.items() if n > 0}
    else:
        # No session — fall back to the standard 12-team PPR shape
        # (1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX ~= 1 RB or WR, 1 K, 1 DST).
        starter_slots = {"QB": 1, "RB": 3, "WR": 3, "TE": 1, "DST": 1, "K": 1}
        league_size = 12

    tiers = compute_position_scarcity(counts, league_size, starter_slots)

    messages = {
        "critical": lambda pos, n: f"Only {n} {pos}s left — critical scarcity. Consider drafting one now.",
        "low": lambda pos, n: f"{n} {pos}s remaining — supply is thinning.",
        "ok": lambda pos, n: f"{pos} supply is healthy ({n} available).",
    }

    alerts: list[ScarcityAlert] = []
    for pos, tier in tiers.items():
        available = counts.get(pos, 0)
        alerts.append(ScarcityAlert(
            position=pos, available=available, tier=tier,
            message=messages[tier](pos, available),
        ))

    # Sort: critical first, then low, then ok
    tier_order = {"critical": 0, "low": 1, "ok": 2}
    alerts.sort(key=lambda a: tier_order[a.tier])

    return ScarcityAnalysisResponse(alerts=alerts, available_counts=counts)
