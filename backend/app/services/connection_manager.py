"""
WebSocket connection manager.

Maintains the list of active WebSocket connections and provides a
broadcast helper so pick events are pushed to all connected clients
simultaneously (e.g. the laptop and the phone viewing the same draft).

Author: Zach Cooper
"""

import asyncio
import json
from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []
        # Serializes broadcasts. Two can overlap in real use — a manual
        # POST /pick and the Sleeper sync poller both broadcast, and each
        # awaits per-connection sends — and Starlette raises if two tasks
        # write to the same WebSocket concurrently. One lock around the
        # whole send loop is the simplest correct fix at this scale
        # (a handful of clients, small payloads).
        self._broadcast_lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, event: dict) -> None:
        """
        Sends a JSON event to every connected client, one broadcast at a
        time (see _broadcast_lock). Silently removes clients that have
        already disconnected.
        """
        async with self._broadcast_lock:
            dead: list[WebSocket] = []
            # Iterate over a copy — disconnect() mutates the list, and a
            # client disconnecting mid-broadcast shouldn't skip its neighbor.
            for ws in list(self.active_connections):
                try:
                    await ws.send_text(json.dumps(event))
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.disconnect(ws)
