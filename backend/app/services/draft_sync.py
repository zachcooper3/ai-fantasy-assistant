"""
Background Sleeper draft sync service.

Polls GET /draft/{draft_id}/picks every POLL_INTERVAL seconds.
For each new pick, looks up the player in SQLite by sleeper_id and
calls record_pick — which updates draft state and broadcasts via WebSocket.

Author: Zach Cooper
"""

import asyncio
import logging

from sqlmodel import Session

from backend.db.database import engine
from backend.db import player_repo as repo
from backend.app.services import sleeper_client
from backend.app.services.draft_state import DraftStateService
from backend.app.services.connection_manager import ConnectionManager
from backend.app.api.draft import _pick_response, _state_response

logger = logging.getLogger(__name__)

POLL_INTERVAL = 5  # seconds between Sleeper API polls


class DraftSyncService:
    """
    Manages a background task that polls Sleeper for new picks
    and automatically records them in the local draft state.

    Usage (in FastAPI lifespan):
        sync_svc = DraftSyncService(draft_svc, conn_mgr)
        app.state.sync_service = sync_svc

    Then to start syncing:
        await sync_svc.start("your_draft_id_here")
    """

    def __init__(
        self,
        draft_service: DraftStateService,
        connection_manager: ConnectionManager,
    ) -> None:
        self._draft_svc = draft_service
        self._conn_mgr = connection_manager
        self._draft_id: str | None = None
        self._task: asyncio.Task | None = None
        self._synced_pick_count: int = 0  # picks we've already processed
        self.status: str = "idle"          # "idle" | "syncing" | "complete" | "error"
        self.error: str | None = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def start(self, draft_id: str) -> None:
        """Start polling a Sleeper draft. Stops any existing poll first."""
        await self.stop()
        self._draft_id = draft_id
        self._synced_pick_count = len(self._draft_svc.picks) if self._draft_svc.is_active else 0
        self.status = "syncing"
        self.error = None
        self._task = asyncio.create_task(self._poll_loop(), name="sleeper-sync")
        logger.info(f"Sleeper sync started for draft {draft_id}")

    async def stop(self) -> None:
        """Cancel the polling task."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self._draft_id = None
        if self.status == "syncing":
            self.status = "idle"
        logger.info("Sleeper sync stopped")

    # -----------------------------------------------------------------------
    # Internal polling loop
    # -----------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Runs until cancelled or the draft is complete."""
        consecutive_errors = 0

        while True:
            try:
                await self._poll_once()
                consecutive_errors = 0
            except asyncio.CancelledError:
                raise
            except Exception as e:
                consecutive_errors += 1
                logger.warning(f"Sleeper poll error ({consecutive_errors}): {e}")
                if consecutive_errors >= 5:
                    self.status = "error"
                    self.error = str(e)
                    logger.error("Too many consecutive Sleeper errors — sync stopped")
                    return

            if self.status != "syncing":
                return

            await asyncio.sleep(POLL_INTERVAL)

    async def _poll_once(self) -> None:
        """Fetches the latest picks from Sleeper and processes any new ones."""
        if not self._draft_id or not self._draft_svc.is_active:
            return

        picks = await sleeper_client.get_draft_picks(self._draft_id)

        # Only process picks we haven't seen yet
        new_picks = [p for p in picks if p.get("pick_no", 0) > self._synced_pick_count]

        if not new_picks:
            return

        # Sort ascending so we process in order
        new_picks.sort(key=lambda p: p.get("pick_no", 0))

        with Session(engine) as db:
            for sleeper_pick in new_picks:
                await self._process_pick(sleeper_pick, db)

        # Check if draft is now complete
        draft_info = await sleeper_client.get_draft(self._draft_id)
        if draft_info.get("status") == "complete":
            self.status = "complete"
            logger.info("Sleeper draft complete — sync finished")

    async def _process_pick(self, sleeper_pick: dict, db: Session) -> None:
        """Resolves a single Sleeper pick to a local player and records it."""
        pick_no = sleeper_pick.get("pick_no", 0)
        sleeper_player_id = str(sleeper_pick.get("player_id", ""))
        metadata = sleeper_pick.get("metadata", {})

        # Try to find the player by sleeper_id
        player = repo.get_player_by_sleeper_id(db, sleeper_player_id)

        if player is None:
            # Fallback: match by name from pick metadata
            full_name = (
                f"{metadata.get('first_name', '')} {metadata.get('last_name', '')}".strip()
            )
            if full_name:
                player = repo.get_player_by_name(db, full_name)

        if player is None:
            logger.warning(
                f"Pick #{pick_no}: could not find player "
                f"sleeper_id={sleeper_player_id!r} "
                f"name={metadata.get('first_name')} {metadata.get('last_name')}"
            )
            # Still advance the draft state with a placeholder
            # so pick numbers stay in sync
            pick = self._draft_svc.record_pick(
                player_id=-1,
                player_name=f"{metadata.get('first_name', '?')} {metadata.get('last_name', '?')}",
                position=metadata.get("position", "?"),
                nfl_team=metadata.get("team", "?"),
            )
        else:
            if not player.is_available:
                # Already recorded (e.g. entered manually before sync started)
                self._synced_pick_count = pick_no
                return

            repo.mark_as_drafted(db, player.id)
            pick = self._draft_svc.record_pick(
                player_id=player.id,
                player_name=player.name,
                position=player.position,
                nfl_team=player.team,
            )

        self._synced_pick_count = pick_no

        # Broadcast to all WebSocket clients
        my_slot = self._draft_svc.config.my_draft_position
        await self._conn_mgr.broadcast({
            "type": "pick",
            "pick": _pick_response(pick, my_slot).model_dump(),
            "state": _state_response(self._draft_svc).model_dump(),
        })

        logger.info(
            f"Synced pick #{pick_no}: "
            f"{pick.player_name} ({pick.position}) → slot {pick.team_slot}"
        )
