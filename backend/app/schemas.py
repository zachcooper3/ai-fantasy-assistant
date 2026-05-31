"""
Pydantic schemas for API request bodies and response shapes.
These are separate from the SQLModel DB models in backend/db/models.py.
Author: Zach Cooper
"""

from pydantic import BaseModel, Field


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
    """Request body for POST /api/draft/session."""
    league_size: int = Field(default=12, ge=8, le=16)
    my_draft_position: int = Field(default=1, ge=1, le=16)
    total_rounds: int = Field(default=15, ge=10, le=20)
    scoring_format: str = Field(default="ppr")


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
    current_pick_number: int
    current_round: int
    current_team_slot: int
    is_my_turn: bool
    picks_until_my_turn: int
    my_next_pick_number: int | None
    draft_complete: bool
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
