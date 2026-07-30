"""
Pydantic schemas for API request bodies and response shapes.
These are separate from the SQLModel DB models in backend/db/models.py.
Author: Zach Cooper
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------

class PlayerResponse(BaseModel):
    id: int
    rank: int
    name: str
    team: str
    bye: int | None
    position: str
    pos_rank: str
    adp: float
    is_available: bool

    model_config = {"from_attributes": True}


class ScarcityResponse(BaseModel):
    """Available player counts per position — used to drive scarcity alerts."""
    QB: int = 0
    RB: int = 0
    WR: int = 0
    TE: int = 0
    DST: int = 0
    K: int = 0


# ---------------------------------------------------------------------------
# Draft session
# ---------------------------------------------------------------------------

class DraftConfigRequest(BaseModel):
    """Request body for POST /api/draft/session.

    The roster-slot fields (qb_slots through dst_slots) default to a
    standard 1-QB PPR lineup — the same assumption the app made implicitly
    (hardcoded, unconfigurable) before this field existed. Leaving them
    unset preserves that exact prior behavior; only someone who deliberately
    overrides one changes anything. No support for a position satisfying
    more than one slot type (e.g. superflex QB-as-FLEX) — not needed today,
    and would meaningfully complicate ai_service.py's gap/FLEX-surplus math
    if added later.
    """
    league_size: int = Field(default=12, ge=8, le=16)
    my_draft_position: int = Field(default=1, ge=1, le=16)
    total_rounds: int = Field(default=15, ge=10, le=20)
    scoring_format: Literal["ppr", "half_ppr", "standard"] = "ppr"
    qb_slots: int = Field(default=1, ge=0, le=4)
    rb_slots: int = Field(default=2, ge=0, le=6)
    wr_slots: int = Field(default=2, ge=0, le=6)
    te_slots: int = Field(default=1, ge=0, le=4)
    flex_slots: int = Field(default=1, ge=0, le=4)
    dst_slots: int = Field(default=1, ge=0, le=2)

    @model_validator(mode="after")
    def _draft_position_within_league(self) -> "DraftConfigRequest":
        """Slot 14 in a 12-team league passed the per-field bounds but broke
        everything downstream: is_my_turn could never be true and the UI
        showed '-1 picks away'. Cross-field rules need a model validator."""
        if self.my_draft_position > self.league_size:
            raise ValueError(
                f"my_draft_position ({self.my_draft_position}) can't exceed "
                f"league_size ({self.league_size})."
            )
        return self


class SleeperPrefillResponse(BaseModel):
    """
    Best-effort settings detected from a Sleeper draft/league, for the setup
    form to use as suggestions — never a silent override. Any field that
    couldn't be confidently detected (API failure, league setting we don't
    model) is left as None; the frontend leaves the user's existing/manual
    value untouched in that case rather than clobbering it with something
    made up.

    detected_scoring_format is informational only. This app's ADP data,
    player metrics, and AI system prompt are all PPR-specific regardless of
    what a given league actually scores (see ingestion/fetch_adp.py,
    ai_service.py's _SYSTEM_PROMPT) — there's no code path today that
    changes behavior based on scoring_format. So this field is surfaced to
    warn the user if their league isn't PPR, but is deliberately never
    written into DraftConfigRequest.scoring_format; doing so would just
    relabel the UI without making anything underneath actually accurate for
    a non-PPR league.
    """
    league_size: int | None = None
    total_rounds: int | None = None
    my_draft_position: int | None = None
    qb_slots: int | None = None
    rb_slots: int | None = None
    wr_slots: int | None = None
    te_slots: int | None = None
    flex_slots: int | None = None
    dst_slots: int | None = None
    detected_scoring_format: str | None = None
    warnings: list[str] = Field(default_factory=list)


class PickRequest(BaseModel):
    """Request body for POST /api/draft/pick."""
    player_id: int


class PickResponse(BaseModel):
    pick_number: int
    round_number: int
    team_slot: int
    player_id: int
    player_name: str
    position: str
    nfl_team: str
    is_mine: bool          # True if this pick belongs to my team slot


class DraftStateResponse(BaseModel):
    is_active: bool
    league_size: int
    my_draft_position: int
    total_rounds: int
    scoring_format: str
    # Echoed back so the frontend's "reset with current settings" flow (see
    # page.tsx's onReset) can re-POST the league's actual roster config
    # instead of silently falling back to DraftConfigRequest's defaults.
    qb_slots: int
    rb_slots: int
    wr_slots: int
    te_slots: int
    flex_slots: int
    dst_slots: int
    current_pick_number: int
    current_round: int
    current_team_slot: int
    is_my_turn: bool
    picks_until_my_turn: int
    my_next_pick_number: int | None
    draft_complete: bool
    # True when this session was rehydrated from disk at boot rather than
    # started by a user. Sessions persist across restarts by design, so without
    # this the app silently resumes an old draft with no indication it did.
    was_restored: bool = False
    # When the session originally began — preserved through a restore, so the
    # client can name the draft it resumed rather than just announcing one.
    started_at: datetime | None = None
    picks: list[PickResponse]
    my_roster: list[PickResponse]


class BoardResponse(BaseModel):
    """The big board: top available players plus scarcity context."""
    players: list[PlayerResponse]
    scarcity: ScarcityResponse
    picks_until_my_turn: int
    is_my_turn: bool


# ---------------------------------------------------------------------------
# WebSocket events
# These dicts are broadcast over the WebSocket on every state change.
# ---------------------------------------------------------------------------
#
# Event types:
#   "pick"   — a pick was recorded
#   "undo"   — the last pick was reversed
#   "reset"  — draft session was reset
#
# Shape (all events share a "type" discriminator):
#   { "type": "pick",  "pick": <PickResponse dict>, "state": <DraftStateResponse dict> }
#   { "type": "undo",  "pick": <PickResponse dict>, "state": <DraftStateResponse dict> }
#   { "type": "reset" }
