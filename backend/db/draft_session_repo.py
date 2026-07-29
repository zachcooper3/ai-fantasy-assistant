"""
Persistence for the active draft session — config + pick journal.

All functions take an explicit Session argument, same as the other repos.

This is the crash-recovery layer for DraftStateService (see DraftSession's
docstring in models.py): the service stays purely in-memory and DB-free,
and the API/sync layers write through to this journal in the same places
they already write Player.is_available. main.py's lifespan calls
load_state() on boot and rehydrates the service if a session was active.

Layering note: this module imports the DraftConfig/PickRecord dataclasses
from the service layer so callers get back exactly the shapes
DraftStateService.restore_session() consumes. That's an unusual direction
for a db/ module, but draft_state.py imports nothing from backend.db, so
it's acyclic — and the alternative (every caller hand-converting rows to
dataclasses) just spreads the same coupling across more files.

Author: Zach Cooper
"""

from sqlmodel import Session, delete, select

from backend.app.services.draft_state import DraftConfig, PickRecord
from backend.db.models import DraftPick, DraftSession

# Fixed primary key — this app manages exactly one draft at a time.
_SESSION_ROW_ID = 1


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

def save_config(session: Session, config: DraftConfig) -> None:
    """
    Persists the session config, replacing any previous session and its
    pick journal — mirrors DraftStateService.start_session(), which also
    wipes pick history.
    """
    session.exec(delete(DraftPick))
    session.exec(delete(DraftSession))
    session.add(DraftSession(
        id=_SESSION_ROW_ID,
        league_size=config.league_size,
        my_draft_position=config.my_draft_position,
        total_rounds=config.total_rounds,
        scoring_format=config.scoring_format,
        qb_slots=config.qb_slots,
        rb_slots=config.rb_slots,
        wr_slots=config.wr_slots,
        te_slots=config.te_slots,
        flex_slots=config.flex_slots,
        dst_slots=config.dst_slots,
    ))
    session.commit()


def clear(session: Session) -> None:
    """Removes the persisted session and its pick journal — mirrors
    DraftStateService.reset()."""
    session.exec(delete(DraftPick))
    session.exec(delete(DraftSession))
    session.commit()


# ---------------------------------------------------------------------------
# Pick journal
# ---------------------------------------------------------------------------

def append_pick(session: Session, pick: PickRecord) -> None:
    """Journals one recorded pick — call right after
    DraftStateService.record_pick / record_synced_pick succeeds."""
    session.add(DraftPick(
        pick_number=pick.pick_number,
        round_number=pick.round_number,
        team_slot=pick.team_slot,
        player_id=pick.player_id,
        player_name=pick.player_name,
        position=pick.position,
        nfl_team=pick.nfl_team,
    ))
    session.commit()


def remove_pick(session: Session, pick_number: int) -> None:
    """
    Removes a journaled pick by pick_number — call after
    DraftStateService.undo_last_pick returns the popped record. Keyed by
    pick_number (not "highest id") so it stays correct even if journal
    insert order ever diverges from pick order.
    """
    row = session.exec(
        select(DraftPick).where(DraftPick.pick_number == pick_number)
    ).first()
    if row is not None:
        session.delete(row)
        session.commit()


# ---------------------------------------------------------------------------
# Rehydration
# ---------------------------------------------------------------------------

def load_state(session: Session) -> tuple[DraftConfig, list[PickRecord]] | None:
    """
    Returns (config, picks-in-pick-number-order) for a persisted session,
    or None if no session was active. Feed straight into
    DraftStateService.restore_session().
    """
    row = session.get(DraftSession, _SESSION_ROW_ID)
    if row is None:
        return None

    config = DraftConfig(
        league_size=row.league_size,
        my_draft_position=row.my_draft_position,
        total_rounds=row.total_rounds,
        scoring_format=row.scoring_format,
        qb_slots=row.qb_slots,
        rb_slots=row.rb_slots,
        wr_slots=row.wr_slots,
        te_slots=row.te_slots,
        flex_slots=row.flex_slots,
        dst_slots=row.dst_slots,
    )

    pick_rows = session.exec(
        select(DraftPick).order_by(DraftPick.pick_number)
    ).all()
    picks = [
        PickRecord(
            pick_number=p.pick_number,
            round_number=p.round_number,
            team_slot=p.team_slot,
            player_id=p.player_id,
            player_name=p.player_name,
            position=p.position,
            nfl_team=p.nfl_team,
        )
        for p in pick_rows
    ]

    return config, picks
