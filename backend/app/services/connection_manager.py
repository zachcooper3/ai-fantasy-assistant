"""
WebSocket connection manager.

Maintains the list of active WebSocket connections and provides a
broadcast helper so pick events are pushed to all connected clients
simultaneously (e.g. the laptop and the phone viewing the same draft).

Author: Zach Cooper
"""

import json
from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, event: dict) -> None:
        """
        Sends a JSON event to every connected client.
        Silently removes clients that have already disconnected.
        """
        dead: list[WebSocket] = []
        for ws in self.active_connections:
            try:
                await ws.send_text(json.dumps(event))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)
