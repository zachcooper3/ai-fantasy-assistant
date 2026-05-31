"""
WebSocket endpoint — real-time push to all connected clients.

Clients connect to WS /ws/draft and receive JSON events on every
pick, undo, or reset. The initial connection sends the current draft
state so the client doesn't need a separate HTTP call to hydrate.

Event shapes are documented in backend/app/schemas.py.

Author: Zach Cooper
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request

from backend.app.services.connection_manager import ConnectionManager
from backend.app.services.draft_state import DraftStateService

router = APIRouter(tags=["websocket"])


def get_draft_service(request: Request) -> DraftStateService:
    return request.app.state.draft_service


def get_connection_manager(request: Request) -> ConnectionManager:
    return request.app.state.connection_manager


@router.websocket("/ws/draft")
async def draft_websocket(websocket: WebSocket):
    """
    Persistent WebSocket connection for real-time draft updates.

    On connect: sends current draft state (or a "no session" message).
    On each pick/undo/reset: the draft API broadcasts to all connections.
    The client doesn't need to send anything — this is receive-only for now.
    """
    svc: DraftStateService = websocket.app.state.draft_service
    mgr: ConnectionManager = websocket.app.state.connection_manager

    await mgr.connect(websocket)

    try:
        # Send current state immediately on connect so the client hydrates
        if svc.is_active:
            from backend.app.api.draft import _state_response
            await websocket.send_json({
                "type": "connected",
                "state": _state_response(svc).model_dump(),
            })
        else:
            await websocket.send_json({"type": "connected", "state": None})

        # Keep the connection alive — all writes come from the HTTP routes
        while True:
            await websocket.receive_text()  # blocks; client can send pings

    except WebSocketDisconnect:
        mgr.disconnect(websocket)
