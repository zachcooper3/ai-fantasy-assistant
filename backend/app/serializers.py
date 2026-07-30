"""
Shared response serializers — build API response models from draft-state
service objects.

These used to live in backend/app/api/draft.py, but they're needed by both
the API layer (draft routes, websocket hydration) and the service layer
(DraftSyncService broadcasts). A service importing from a route module is a
layering inversion, and it became a hard circular import the moment
draft.py needed DraftSyncService for the session-lifecycle sync.stop()
calls — so they live here now, below both layers.

Author: Zach Cooper
"""

from backend.app.schemas import DraftStateResponse, PickResponse
from backend.app.services.draft_state import DraftStateService, PickRecord


def build_pick_response(pick: PickRecord, my_slot: int) -> PickResponse:
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


def build_state_response(svc: DraftStateService) -> DraftStateResponse:
    my_slot = svc.config.my_draft_position
    return DraftStateResponse(
        is_active=svc.is_active,
        league_size=svc.config.league_size,
        my_draft_position=my_slot,
        total_rounds=svc.config.total_rounds,
        scoring_format=svc.config.scoring_format,
        qb_slots=svc.config.qb_slots,
        rb_slots=svc.config.rb_slots,
        wr_slots=svc.config.wr_slots,
        te_slots=svc.config.te_slots,
        flex_slots=svc.config.flex_slots,
        dst_slots=svc.config.dst_slots,
        current_pick_number=svc.current_pick_number,
        current_round=svc.current_round,
        current_team_slot=svc.current_team_slot,
        is_my_turn=svc.is_my_turn,
        picks_until_my_turn=svc.picks_until_my_turn,
        my_next_pick_number=svc.my_next_pick_number,
        draft_complete=svc.draft_complete,
        was_restored=svc.was_restored,
        started_at=svc.started_at,
        picks=[build_pick_response(p, my_slot) for p in svc.picks],
        my_roster=[build_pick_response(p, my_slot) for p in svc.my_roster],
    )


# ---------------------------------------------------------------------------
# WebSocket payloads
#
# `WebSocket.send_json` serialises with the *stdlib* json module, which knows
# nothing about datetime, Decimal, UUID and friends — unlike the HTTP path,
# where FastAPI runs every response through its own encoder first. So a field
# that serialises fine over HTTP can still blow up the socket with
# "Object of type datetime is not JSON serializable", taking down the
# connection that carries every live pick.
#
# That's exactly what adding `started_at` did. These helpers exist so no call
# site has to remember `mode="json"`: use them for anything sent over the wire.
# ---------------------------------------------------------------------------

def state_payload(svc: DraftStateService) -> dict:
    """JSON-safe draft state for WebSocket broadcast."""
    return build_state_response(svc).model_dump(mode="json")


def pick_payload(pick: PickRecord, my_slot: int) -> dict:
    """JSON-safe pick for WebSocket broadcast."""
    return build_pick_response(pick, my_slot).model_dump(mode="json")
