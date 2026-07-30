"""
WebSocket payload serialisation.

`WebSocket.send_json` uses the stdlib json module, not FastAPI's encoder, so a
field that serialises fine over HTTP can still kill the socket. Adding
`started_at` (a datetime) to DraftStateResponse did exactly that: every HTTP
test passed while the WebSocket carrying live picks raised
"Object of type datetime is not JSON serializable" on connect.

These tests serialise the payloads the same way Starlette does.
"""

import json

from backend.app.serializers import pick_payload, state_payload
from backend.app.services.draft_state import DraftConfig, DraftStateService

CFG = DraftConfig(league_size=12, my_draft_position=3, total_rounds=15)


def active_service() -> DraftStateService:
    svc = DraftStateService()
    svc.start_session(CFG)
    svc.record_pick(player_id=1, player_name="A", position="RB", nfl_team="DET")
    svc.record_pick(player_id=2, player_name="B", position="WR", nfl_team="CIN")
    return svc


def test_state_payload_is_json_serializable():
    # json.dumps with no default= is exactly what send_json does.
    json.dumps(state_payload(active_service()))


def test_pick_payload_is_json_serializable():
    svc = active_service()
    json.dumps(pick_payload(svc.picks[-1], CFG.my_draft_position))


def test_restored_session_payload_is_json_serializable():
    # The regression: started_at is only non-None after a restore, so a plain
    # start_session() session wouldn't have caught this.
    svc = DraftStateService()
    svc.restore_session(CFG, [])
    from datetime import datetime, timezone
    svc.restore_session(CFG, [], started_at=datetime.now(timezone.utc))

    payload = state_payload(svc)
    json.dumps(payload)
    # Serialised as an ISO string, matching what the HTTP path emits — the
    # frontend types started_at as `string | null` for both transports.
    assert isinstance(payload["started_at"], str)
    assert payload["was_restored"] is True


def test_started_at_is_null_not_missing_when_unset():
    svc = DraftStateService()
    svc.restore_session(CFG, [], started_at=None)
    payload = state_payload(svc)
    json.dumps(payload)
    assert payload["started_at"] is None
