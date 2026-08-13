"""
CRUD operations for the DraftProfile table.
All functions take an explicit Session argument — no globals, easy to test.

Mirrors metrics_repo.py's shape and reasoning exactly, including the
player_id-instability problem: ingest_players.ingest_csv() does a full
delete-and-reinsert of Player on every ADP refresh, so player_id is not a
stable identity across refreshes. sleeper_id is the durable one (re-resolved
by name-matching on every refresh via sync_sleeper_ids.py), so it's stored
here too and relink_player_ids() repairs the player_id FK against it after
every Player reingest — see fetch_adp.py, right after sync_sleeper_ids and
metrics_repo.relink_player_ids run.

Run manually (rarely needed — fetch_adp.py already calls this after every
Player reingest):
    py -m backend.db.draft_profile_repo

Author: Zach Cooper
"""

from sqlmodel import Session, select

from backend.db.metrics_repo import (
    MAX_ORPHAN_FRACTION,
    MIN_ROWS_FOR_ORPHAN_GUARD,
    RelinkAborted,
)
from backend.db.models import DraftProfile, Player


def upsert_draft_profile(session: Session, player_id: int, sleeper_id: str | None = None, **fields) -> DraftProfile:
    """
    Inserts or updates the single DraftProfile row for a player. `fields`
    should match DraftProfile column names (draft_year, draft_round, etc.)
    — unrecognized keys will raise, same as constructing DraftProfile(**fields)
    directly would.

    sleeper_id should be passed whenever the caller has it, so
    relink_player_ids() has something durable to repair player_id against
    later. Optional only so ad-hoc/test callers without a real sleeper_id
    don't break.
    """
    existing = session.exec(
        select(DraftProfile).where(DraftProfile.player_id == player_id)
    ).first()

    if existing:
        for key, value in fields.items():
            setattr(existing, key, value)
        if sleeper_id is not None:
            existing.sleeper_id = sleeper_id
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return existing

    profile = DraftProfile(player_id=player_id, sleeper_id=sleeper_id, **fields)
    session.add(profile)
    session.commit()
    session.refresh(profile)
    return profile


def get_draft_profile(session: Session, player_id: int) -> DraftProfile | None:
    """Returns the DraftProfile row for a player, or None if never computed.

    Looks up by player_id — fine for same-session use where the FK is known
    fresh. For anything that persists across a Player reingest, prefer
    get_draft_profile_by_sleeper_id instead (see module docstring)."""
    return session.exec(
        select(DraftProfile).where(DraftProfile.player_id == player_id)
    ).first()


def get_draft_profile_by_sleeper_id(session: Session, sleeper_id: str) -> DraftProfile | None:
    """Returns the DraftProfile row for a player by their durable sleeper_id
    — safe to use even if player_id has gone stale since this row was written."""
    return session.exec(
        select(DraftProfile).where(DraftProfile.sleeper_id == sleeper_id)
    ).first()


def get_draft_profiles_bulk(session: Session, player_ids: list[int]) -> dict[int, DraftProfile]:
    """
    Returns {player_id: DraftProfile} for every ID that has a row. Missing
    from this dict just means "not a recent draftee we have a profile for"
    (most veterans) — callers should treat that as "unknown," not as a
    negative signal.
    """
    if not player_ids:
        return {}
    rows = session.exec(
        select(DraftProfile).where(DraftProfile.player_id.in_(player_ids))
    ).all()
    return {row.player_id: row for row in rows}


def relink_player_ids(session: Session) -> tuple[int, int]:
    """
    Repairs DraftProfile.player_id to match each row's durable sleeper_id,
    after a Player reingest may have reassigned every Player.id. Call this
    right after ingest_players + sync_sleeper_ids, before player_id is
    trusted for anything (see fetch_adp.py). Identical logic to
    metrics_repo.relink_player_ids — see that docstring for the full
    Gibbs/Robinson incident this pattern exists because of.

    Returns (relinked, orphaned):
      relinked — rows whose player_id was updated to the current match
      orphaned — rows deleted because their sleeper_id no longer matches
                 any current Player (dropped out of the ADP pool entirely)

    Raises metrics_repo.RelinkAborted, deleting nothing, when the Player
    table's sleeper_id coverage looks broken rather than merely changed —
    same guard, same reasoning, same threshold as metrics_repo (see that
    docstring for the Sleeper-outage path that makes it reachable).
    """
    all_players = session.exec(select(Player)).all()
    sleeper_to_player_id = {p.sleeper_id: p.id for p in all_players if p.sleeper_id}

    rows = session.exec(select(DraftProfile)).all()
    linkable = [dp for dp in rows if dp.sleeper_id]

    if linkable and not sleeper_to_player_id:
        raise RelinkAborted(
            f"No Player row carries a sleeper_id ({len(all_players)} players "
            f"checked), but {len(linkable)} DraftProfile row(s) do. That means "
            "sync_sleeper_ids has not run (or failed) since the last reingest — "
            "relinking now would orphan and delete every draft profile. Re-run "
            "`py -m backend.ingestion.sync_sleeper_ids`, then this."
        )

    to_delete: list[DraftProfile] = []
    to_update: list[tuple[DraftProfile, int]] = []
    for profile in linkable:
        current_player_id = sleeper_to_player_id.get(profile.sleeper_id)
        if current_player_id is None:
            to_delete.append(profile)
        elif profile.player_id != current_player_id:
            to_update.append((profile, current_player_id))

    if (
        len(linkable) >= MIN_ROWS_FOR_ORPHAN_GUARD
        and len(to_delete) / len(linkable) > MAX_ORPHAN_FRACTION
    ):
        raise RelinkAborted(
            f"{len(to_delete)}/{len(linkable)} DraftProfile rows would be "
            f"orphaned, above the {MAX_ORPHAN_FRACTION:.0%} ceiling. This looks "
            "like a partial Sleeper player database rather than a real roster "
            "change. Nothing was deleted; re-run sync_sleeper_ids and check its "
            "matched/unmatched counts before retrying."
        )

    for profile in to_delete:
        session.delete(profile)
    session.commit()

    if to_update:
        # player_id is UNIQUE and a relink can require two rows to trade IDs
        # with each other — stage through a temporary negative placeholder
        # first (never collides with a real Player.id) so this is
        # conflict-free regardless of how many rows are involved in a swap.
        # Same reasoning as metrics_repo.relink_player_ids.
        for profile, _ in to_update:
            profile.player_id = -profile.id
            session.add(profile)
        session.commit()

        for profile, new_player_id in to_update:
            profile.player_id = new_player_id
            session.add(profile)
        session.commit()

    return len(to_update), len(to_delete)


def main() -> None:
    """CLI wrapper around relink_player_ids — see module docstring for when
    you need this (anything that reingests Player outside of fetch_adp.py,
    which already calls this automatically)."""
    from backend.db.database import engine

    with Session(engine) as session:
        relinked, orphaned = relink_player_ids(session)
    print(f"DraftProfile relink complete: {relinked} relinked, {orphaned} orphaned.")


if __name__ == "__main__":
    main()
