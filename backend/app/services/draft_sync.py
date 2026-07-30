"""
Background Sleeper draft sync service.

Polls GET /draft/{draft_id}/picks every POLL_INTERVAL seconds.
For each new pick, looks up the player in SQLite by sleeper_id and calls
record_synced_pick with Sleeper's own pick_no/round/draft_slot attribution
— which updates draft state and broadcasts via WebSocket.

Author: Zach Cooper
"""

import asyncio
import logging

from sqlmodel import Session

from backend.db.database import engine
from backend.db import player_repo as repo
from backend.db import draft_session_repo
from backend.app.services import sleeper_client
from backend.app.services.draft_state import DraftStateService
from backend.app.services.connection_manager import ConnectionManager
from backend.app.serializers import pick_payload, state_payload

logger = logging.getLogger(__name__)

POLL_INTERVAL = 2          # seconds between Sleeper API polls
COMPLETION_CHECK_EVERY = 5  # only check draft completion every N polls


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
        self._poll_count: int = 0          # total polls run (for completion check cadence)
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
        self._poll_count = 0
        self.status = "syncing"
        self.error = None
        self._task = asyncio.create_task(self._poll_loop(), name="sleeper-sync")
        logger.info(f"Sleeper sync started for draft {draft_id}")

    async def stop(self) -> None:
        """Cancel the polling task and return to a clean idle state.

        Always resets status/error (not just from "syncing") — stop() is
        now part of the session lifecycle (called on every session create/
        reset, see backend/app/api/draft.py), and a new session should
        never inherit a stale "error"/"complete" marker from a previous
        draft's sync.
        """
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self._draft_id = None
        self._synced_pick_count = 0
        self.status = "idle"
        self.error = None
        logger.info("Sleeper sync stopped")

    # -----------------------------------------------------------------------
    # Internal polling loop
    # -----------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """
        Runs until cancelled or the draft is complete.

        Sleep is at the TOP of the loop so that:
        - The very first poll fires immediately (before the loop body runs the
          first time the task is created, start() has already set up state).
        - Every subsequent cycle is a clean POLL_INTERVAL regardless of how
          long _poll_once takes, keeping end-to-end pick latency close to
          POLL_INTERVAL rather than POLL_INTERVAL + HTTP round-trip time.
        """
        consecutive_errors = 0

        # First poll fires immediately on start — no initial sleep.
        first = True
        while True:
            if not first:
                await asyncio.sleep(POLL_INTERVAL)
            first = False

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

    async def _poll_once(self) -> None:
        """Fetches the latest picks from Sleeper and processes any new ones."""
        if not self._draft_id or not self._draft_svc.is_active:
            return

        self._poll_count += 1
        picks = await sleeper_client.get_draft_picks(self._draft_id)

        # Only process picks we haven't seen yet
        new_picks = [p for p in picks if p.get("pick_no", 0) > self._synced_pick_count]

        if new_picks:
            # Sort ascending so we process in order
            new_picks.sort(key=lambda p: p.get("pick_no", 0))

            with Session(engine) as db:
                for sleeper_pick in new_picks:
                    await self._process_pick(sleeper_pick, db)

        # Check draft completion only every COMPLETION_CHECK_EVERY polls to
        # avoid an extra HTTP round trip on every cycle.
        if self._poll_count % COMPLETION_CHECK_EVERY == 0:
            draft_info = await sleeper_client.get_draft(self._draft_id)
            if draft_info.get("status") == "complete":
                self.status = "complete"
                logger.info("Sleeper draft complete — sync finished")

    async def _process_pick(self, sleeper_pick: dict, db: Session) -> None:
        """Resolves a single Sleeper pick to a local player and records it.

        Attribution (pick number / round / team slot) comes from Sleeper's
        own pick_no / round / draft_slot fields rather than local snake
        inference — see DraftStateService.record_synced_pick for why
        (traded picks, third-round reversal, and manual/live-sync mixing
        all break the inferred math and misattribute every later pick).
        """
        pick_no = sleeper_pick.get("pick_no", 0)
        sleeper_player_id = str(sleeper_pick.get("player_id", ""))
        metadata = sleeper_pick.get("metadata", {})
        # Explicit attribution from Sleeper; None if absent so
        # record_synced_pick falls back to local inference per-field.
        draft_slot = sleeper_pick.get("draft_slot")
        round_number = sleeper_pick.get("round")

        if self._draft_svc.draft_complete:
            # More picks than league_size x total_rounds can hold — the
            # local session config doesn't match the real Sleeper draft.
            # Don't let record_* raise on every poll (5 consecutive errors
            # would kill the sync); log once per pick and move on.
            logger.warning(
                f"Pick #{pick_no} from Sleeper ignored — local draft is already "
                f"complete. Check league_size/total_rounds in the session config."
            )
            self._synced_pick_count = pick_no
            return

        # Try to find the player by sleeper_id
        player = repo.get_player_by_sleeper_id(db, sleeper_player_id)

        if player is None:
            # Fallback: match by name from pick metadata, constrained to the
            # position Sleeper reports so a shared/similar name at another
            # position can't be marked drafted by mistake (audit W4).
            # Sleeper says "DEF" where this app says "DST".
            full_name = (
                f"{metadata.get('first_name', '')} {metadata.get('last_name', '')}".strip()
            )
            meta_pos = (metadata.get("position") or "").upper()
            if meta_pos == "DEF":
                meta_pos = "DST"
            if full_name:
                player = repo.get_player_by_name(db, full_name, position=meta_pos or None)

        if player is None:
            logger.warning(
                f"Pick #{pick_no}: could not find player "
                f"sleeper_id={sleeper_player_id!r} "
                f"name={metadata.get('first_name')} {metadata.get('last_name')}"
            )
            # Still advance the draft state with a placeholder
            # so pick numbers stay in sync
            pick = self._draft_svc.record_synced_pick(
                player_id=-1,
                player_name=f"{metadata.get('first_name', '?')} {metadata.get('last_name', '?')}",
                position=metadata.get("position", "?"),
                nfl_team=metadata.get("team", "?"),
                pick_number=pick_no or None,
                round_number=round_number,
                team_slot=draft_slot,
            )
        else:
            if not player.is_available:
                # Already recorded (e.g. entered manually before sync
                # started). Cross-check the manual entry's attribution
                # against Sleeper's — a mismatch means the local board has
                # this pick under the wrong slot/number, which corrupts
                # roster views and AI context downstream.
                local = next(
                    (p for p in self._draft_svc.picks if p.player_id == player.id),
                    None,
                )
                if local is None:
                    logger.warning(
                        f"Pick #{pick_no}: {player.name} is marked unavailable in "
                        f"the DB but has no local pick record — availability may "
                        f"be stale from a previous session."
                    )
                elif (draft_slot is not None and local.team_slot != draft_slot) or (
                    pick_no and local.pick_number != pick_no
                ):
                    logger.warning(
                        f"Pick #{pick_no}: {player.name} was entered manually as "
                        f"pick #{local.pick_number} (slot {local.team_slot}), but "
                        f"Sleeper reports pick #{pick_no} (slot {draft_slot}) — "
                        f"local attribution has diverged from the real draft."
                    )
                self._synced_pick_count = pick_no
                return

            repo.mark_as_drafted(db, player.id)
            pick = self._draft_svc.record_synced_pick(
                player_id=player.id,
                player_name=player.name,
                position=player.position,
                nfl_team=player.team,
                pick_number=pick_no or None,
                round_number=round_number,
                team_slot=draft_slot,
            )

        self._synced_pick_count = pick_no
        draft_session_repo.append_pick(db, pick)  # journal for crash recovery

        # Broadcast to all WebSocket clients
        my_slot = self._draft_svc.config.my_draft_position
        await self._conn_mgr.broadcast({
            "type": "pick",
            "pick": pick_payload(pick, my_slot),
            "state": state_payload(self._draft_svc),
        })

        logger.info(
            f"Synced pick #{pick_no}: "
            f"{pick.player_name} ({pick.position}) → slot {pick.team_slot}"
        )
