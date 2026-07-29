"""
CRUD operations for the PlayerMetrics table.
All functions take an explicit Session argument — no globals, easy to test.

Uses upsert-in-place (find by player_id, update or insert) rather than the
full-delete-and-reinsert pattern player_repo/ingest_players uses for Player.

IMPORTANT — player_id is NOT a stable cross-refresh identity, despite being
a FK. This module originally assumed it was ("nothing references Player
rows by a stable ID across a reload except sleeper_id... PlayerMetrics is
looked up by the player.id foreign key directly, so a full wipe-and-reinsert
would just be extra churn for no benefit") — that assumption was wrong and
caused a real bug: ingest_players.ingest_csv() deletes and reinserts every
Player row on each ADP refresh, so autoincrement IDs get reassigned in
whatever order the CSV lists players *that time*. Two players close in ADP
can flip relative order between pulls, which flips which one gets the lower
ID — silently reattaching an existing PlayerMetrics row (and everything
built from it, like a synthesized note) to a DIFFERENT real player after
the next reingest. Confirmed live: Jahmyr Gibbs's and Bijan Robinson's
metrics swapped identities this way.

sleeper_id doesn't have this problem — sync_sleeper_ids.py re-resolves it
by name-matching on every refresh, so it always points at the same real
person regardless of row order. PlayerMetrics now stores sleeper_id
directly (captured at upsert time) as the durable identity, and
relink_player_ids() repairs the player_id FK to match it after every Player
reingest (wired into fetch_adp.py, right after sync_sleeper_ids runs).
Callers that need cross-session reliability (e.g. fetch_synthesis.py's
Player/PlayerMetrics join) should prefer joining on sleeper_id over
player_id for exactly this reason.

Run manually (needed after the manual `ingest_players` + `sync_sleeper_ids`
path documented in the README — fetch_adp.py already does this step for you):
    py -m backend.db.metrics_repo

Author: Zach Cooper
"""

from sqlmodel import Session, select

from backend.db.models import Player, PlayerMetrics


def upsert_metrics(session: Session, player_id: int, sleeper_id: str | None = None, **fields) -> PlayerMetrics:
    """
    Inserts or updates the single PlayerMetrics row for a player.
    `fields` should match PlayerMetrics column names (season, through_week,
    targets_per_game, etc.) — unrecognized keys will raise, same as
    constructing PlayerMetrics(**fields) directly would.

    sleeper_id should be passed whenever the caller has it (fetch_metrics.py
    always does — it's the same crosswalk-resolved value already used to
    find the Player row) so relink_player_ids() has something durable to
    repair player_id against later. Optional only so ad-hoc/test callers
    without a real sleeper_id don't break.
    """
    existing = session.exec(
        select(PlayerMetrics).where(PlayerMetrics.player_id == player_id)
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

    metrics = PlayerMetrics(player_id=player_id, sleeper_id=sleeper_id, **fields)
    session.add(metrics)
    session.commit()
    session.refresh(metrics)
    return metrics


def get_metrics(session: Session, player_id: int) -> PlayerMetrics | None:
    """Returns the PlayerMetrics row for a player, or None if never computed.

    Looks up by player_id — fine for same-session use (e.g. right after
    upsert_metrics in fetch_metrics.py's own loop) where the FK is known
    fresh. For anything that persists across a Player reingest, prefer
    get_metrics_by_sleeper_id instead (see module docstring)."""
    return session.exec(
        select(PlayerMetrics).where(PlayerMetrics.player_id == player_id)
    ).first()


def get_metrics_by_sleeper_id(session: Session, sleeper_id: str) -> PlayerMetrics | None:
    """Returns the PlayerMetrics row for a player by their durable sleeper_id
    — safe to use even if player_id has gone stale since this row was written."""
    return session.exec(
        select(PlayerMetrics).where(PlayerMetrics.sleeper_id == sleeper_id)
    ).first()


def get_metrics_bulk(session: Session, player_ids: list[int]) -> dict[int, PlayerMetrics]:
    """
    Returns {player_id: PlayerMetrics} for every ID that has a row.
    Player IDs with no computed metrics yet are simply absent from the
    result — callers should treat a missing key the same as "unknown,"
    same as a None field within a row that does exist.
    """
    if not player_ids:
        return {}
    rows = session.exec(
        select(PlayerMetrics).where(PlayerMetrics.player_id.in_(player_ids))
    ).all()
    return {row.player_id: row for row in rows}


def relink_player_ids(session: Session) -> tuple[int, int]:
    """
    Repairs PlayerMetrics.player_id to match each row's durable sleeper_id,
    after a Player reingest may have reassigned every Player.id. Call this
    right after ingest_players + sync_sleeper_ids, before player_id is
    trusted for anything (see fetch_adp.py).

    Returns (relinked, orphaned):
      relinked — rows whose player_id was updated to the current match
      orphaned — rows deleted because their sleeper_id no longer matches
                 any current Player (that player dropped out of the ADP
                 pool entirely; keeping a metrics row with no valid owner
                 would just be a landmine for the next join)

    Rows with no sleeper_id at all (shouldn't happen via fetch_metrics.py,
    but possible from an old row predating this column, or an ad-hoc
    upsert_metrics call that didn't pass one) are left untouched — there's
    nothing durable to repair them against, so player_id is all they have.
    """
    sleeper_to_player_id = {
        p.sleeper_id: p.id
        for p in session.exec(select(Player)).all()
        if p.sleeper_id
    }

    to_delete: list[PlayerMetrics] = []
    to_update: list[tuple[PlayerMetrics, int]] = []
    for metrics in session.exec(select(PlayerMetrics)).all():
        if not metrics.sleeper_id:
            continue

        current_player_id = sleeper_to_player_id.get(metrics.sleeper_id)
        if current_player_id is None:
            to_delete.append(metrics)
        elif metrics.player_id != current_player_id:
            to_update.append((metrics, current_player_id))

    for metrics in to_delete:
        session.delete(metrics)
    session.commit()

    if to_update:
        # player_id is UNIQUE, and a relink can require two rows to trade
        # IDs with each other — exactly what happens when two players close
        # in ADP flip relative order in a reingest (confirmed live: Gibbs
        # and Robinson). Writing straight to each row's new value can
        # collide with another row's CURRENT value mid-flush. Stage through
        # a temporary negative placeholder first (never collides with a
        # real Player.id, which SQLite autoincrements from 1) so this is
        # conflict-free regardless of how many rows are involved in a
        # swap or longer cycle.
        for metrics, _ in to_update:
            metrics.player_id = -metrics.id
            session.add(metrics)
        session.commit()

        for metrics, new_player_id in to_update:
            metrics.player_id = new_player_id
            session.add(metrics)
        session.commit()

    return len(to_update), len(to_delete)


def main() -> None:
    """CLI wrapper around relink_player_ids — see module docstring for when
    you need this (anything that reingests Player outside of fetch_adp.py,
    which already calls this automatically)."""
    from backend.db.database import engine

    with Session(engine) as session:
        relinked, orphaned = relink_player_ids(session)
    print(f"PlayerMetrics relink complete: {relinked} relinked, {orphaned} orphaned.")


if __name__ == "__main__":
    main()
