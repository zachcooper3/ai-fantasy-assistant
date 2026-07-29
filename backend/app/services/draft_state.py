"""
In-memory draft state manager.

Tracks the full state of an active draft session: picks made, roster
composition per team, whose turn it is, and how far away your next pick is.
All snake-draft position math lives here.

This service is intentionally stateless with respect to the database — it
operates on plain Python objects and delegates DB writes to the caller
(the API layer calls player_repo.mark_as_drafted separately).

Author: Zach Cooper
"""

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DraftConfig:
    """Immutable settings for a draft session.

    The roster-slot fields default to a standard 1-QB PPR lineup — the same
    assumption ai_service.py made implicitly (hardcoded, unconfigurable)
    before these fields existed. See RecommendationContext.starting_lineup
    for where this feeds into pick recommendations.
    """
    league_size: int = 12
    my_draft_position: int = 1    # 1-indexed slot (1 = first overall pick)
    total_rounds: int = 15
    scoring_format: str = "ppr"
    qb_slots: int = 1
    rb_slots: int = 2
    wr_slots: int = 2
    te_slots: int = 1
    flex_slots: int = 1
    dst_slots: int = 1

    @property
    def starting_lineup(self) -> dict[str, int]:
        """Roster-slot config as a {position: required_count} dict, in the
        shape ai_service.py's gap-computation logic expects."""
        return {
            "QB": self.qb_slots,
            "RB": self.rb_slots,
            "WR": self.wr_slots,
            "TE": self.te_slots,
            "FLEX": self.flex_slots,
            "DST": self.dst_slots,
        }


@dataclass
class PickRecord:
    """One completed pick in the draft."""
    pick_number: int         # Overall pick (1 = first pick of the draft)
    round_number: int        # Round (1-indexed)
    team_slot: int           # Drafter's slot (1-indexed)
    player_id: int           # FK to Player.id in SQLite
    player_name: str         # Denormalised for fast display
    position: str            # e.g. "RB", "WR"
    nfl_team: str            # e.g. "DET"


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class DraftStateService:
    """
    Manages the state of one active draft session.

    Usage:
        service = DraftStateService()
        service.start_session(DraftConfig(league_size=12, my_draft_position=3))
        pick = service.record_pick(player_id=1, player_name="Jahmyr Gibbs",
                                   position="RB", nfl_team="DET")
    """

    def __init__(self) -> None:
        self._config: DraftConfig | None = None
        self._picks: list[PickRecord] = []

    # -----------------------------------------------------------------------
    # Session lifecycle
    # -----------------------------------------------------------------------

    def start_session(self, config: DraftConfig) -> None:
        """
        Initialises (or resets) a draft session with the given config.
        Any existing pick history is wiped.
        """
        self._config = config
        self._picks = []

    def reset(self) -> None:
        """Clears the active session entirely."""
        self._config = None
        self._picks = []

    @property
    def is_active(self) -> bool:
        return self._config is not None

    @property
    def config(self) -> DraftConfig:
        if self._config is None:
            raise RuntimeError("No active draft session. Call start_session() first.")
        return self._config

    # -----------------------------------------------------------------------
    # Snake-draft math
    # -----------------------------------------------------------------------

    def slot_for_pick(self, pick_number: int) -> int:
        """
        Returns the 1-indexed team slot assigned to a given overall pick number
        in a standard snake draft.

        Examples (12-team league):
            Pick 1  → slot 1   (round 1, position 1)
            Pick 12 → slot 12  (round 1, position 12)
            Pick 13 → slot 12  (round 2, position 1 — snake reverses)
            Pick 24 → slot 1   (round 2, position 12)
            Pick 25 → slot 1   (round 3, position 1)
        """
        n = self.config.league_size
        round_num = math.ceil(pick_number / n)
        pos_in_round = pick_number - (round_num - 1) * n  # 1-indexed within round
        if round_num % 2 == 1:   # odd round: ascending order
            return pos_in_round
        else:                     # even round: descending order
            return n + 1 - pos_in_round

    def round_for_pick(self, pick_number: int) -> int:
        return math.ceil(pick_number / self.config.league_size)

    # -----------------------------------------------------------------------
    # Current state
    # -----------------------------------------------------------------------

    @property
    def current_pick_number(self) -> int:
        """The overall pick number of the *next* pick to be made (1-indexed)."""
        return len(self._picks) + 1

    @property
    def current_round(self) -> int:
        return self.round_for_pick(self.current_pick_number)

    @property
    def current_team_slot(self) -> int:
        """The draft slot of the team currently on the clock."""
        return self.slot_for_pick(self.current_pick_number)

    @property
    def is_my_turn(self) -> bool:
        return self.current_team_slot == self.config.my_draft_position

    @property
    def draft_complete(self) -> bool:
        total = self.config.league_size * self.config.total_rounds
        return len(self._picks) >= total

    # -----------------------------------------------------------------------
    # Look-ahead
    # -----------------------------------------------------------------------

    @property
    def my_next_pick_number(self) -> int | None:
        """
        Returns the overall pick number of my next turn, or None if the draft
        is complete. Includes the current pick if it is already my turn.
        """
        total = self.config.league_size * self.config.total_rounds
        for p in range(self.current_pick_number, total + 1):
            if self.slot_for_pick(p) == self.config.my_draft_position:
                return p
        return None

    @property
    def picks_until_my_turn(self) -> int:
        """
        How many picks stand between now and my next turn.
        0 means I'm currently on the clock.
        """
        next_pick = self.my_next_pick_number
        if next_pick is None:
            return -1
        return next_pick - self.current_pick_number

    # -----------------------------------------------------------------------
    # Pick recording
    # -----------------------------------------------------------------------

    def record_pick(
        self,
        player_id: int,
        player_name: str,
        position: str,
        nfl_team: str,
    ) -> PickRecord:
        """
        Records a pick for whoever is currently on the clock.
        Returns the completed PickRecord.

        Raises RuntimeError if the draft is already complete.
        """
        if self.draft_complete:
            raise RuntimeError("Draft is already complete.")

        pick = PickRecord(
            pick_number=self.current_pick_number,
            round_number=self.current_round,
            team_slot=self.current_team_slot,
            player_id=player_id,
            player_name=player_name,
            position=position,
            nfl_team=nfl_team,
        )
        self._picks.append(pick)
        return pick

    def record_synced_pick(
        self,
        *,
        player_id: int,
        player_name: str,
        position: str,
        nfl_team: str,
        pick_number: int | None = None,
        round_number: int | None = None,
        team_slot: int | None = None,
    ) -> PickRecord:
        """
        Records a pick using explicit attribution from the platform
        (Sleeper's pick_no / round / draft_slot), falling back to local
        snake-math inference for any field the caller couldn't provide.

        Exists because record_pick's inference is only correct for a pure
        snake draft with no manual/live-sync mixing: traded picks and
        third-round-reversal orders don't follow slot_for_pick's math, and
        a single divergence there silently misattributes every subsequent
        pick — corrupting my_roster, roster-gap math, and therefore AI
        recommendations. When the platform tells us exactly who picked at
        which slot, trust it over our own model.

        Logs a warning when the platform's pick number disagrees with the
        local count — the signal that manual entries and live sync have
        drifted apart and the board should be double-checked.

        Raises RuntimeError if the draft is already complete, same
        contract as record_pick.
        """
        if self.draft_complete:
            raise RuntimeError("Draft is already complete.")

        local_pick_number = self.current_pick_number
        if pick_number is not None and pick_number != local_pick_number:
            logger.warning(
                f"Sleeper pick #{pick_number} arrived while local state expected "
                f"#{local_pick_number} — manual entries and live sync may have "
                f"diverged; recording with Sleeper's numbering."
            )

        pick_number = pick_number if pick_number is not None else local_pick_number
        pick = PickRecord(
            pick_number=pick_number,
            round_number=round_number if round_number is not None else self.round_for_pick(pick_number),
            team_slot=team_slot if team_slot is not None else self.slot_for_pick(pick_number),
            player_id=player_id,
            player_name=player_name,
            position=position,
            nfl_team=nfl_team,
        )
        self._picks.append(pick)
        return pick

    def undo_last_pick(self) -> PickRecord | None:
        """
        Removes the most recent pick and returns it, or None if there are no picks.
        The caller is responsible for restoring player availability in the DB.
        """
        if not self._picks:
            return None
        return self._picks.pop()

    # -----------------------------------------------------------------------
    # Roster views
    # -----------------------------------------------------------------------

    @property
    def picks(self) -> list[PickRecord]:
        return list(self._picks)

    @property
    def my_roster(self) -> list[PickRecord]:
        """All picks made by my team slot, in draft order."""
        return [p for p in self._picks if p.team_slot == self.config.my_draft_position]

    @property
    def drafted_player_ids(self) -> set[int]:
        return {p.player_id for p in self._picks}

    def roster_for_slot(self, team_slot: int) -> list[PickRecord]:
        """Returns the roster for a specific team slot."""
        return [p for p in self._picks if p.team_slot == team_slot]

    def all_rosters(self) -> dict[int, list[PickRecord]]:
        """
        Returns a dict mapping each team slot to their picks.
        Includes slots with no picks yet (empty list).
        """
        rosters: dict[int, list[PickRecord]] = {
            slot: [] for slot in range(1, self.config.league_size + 1)
        }
        for pick in self._picks:
            rosters[pick.team_slot].append(pick)
        return rosters

    def position_counts_for_slot(self, team_slot: int) -> dict[str, int]:
        """Returns {position: count} for a team's current roster. Useful for opponent tracking."""
        counts: dict[str, int] = {}
        for pick in self.roster_for_slot(team_slot):
            counts[pick.position] = counts.get(pick.position, 0) + 1
        return counts
