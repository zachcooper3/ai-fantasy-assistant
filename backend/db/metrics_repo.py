"""
CRUD operations for the PlayerMetrics table.
All functions take an explicit Session argument — no globals, easy to test.

Uses upsert-in-place (find by player_id, update or insert) rather than the
full-delete-and-reinsert pattern player_repo/ingest_players uses for Player.
That reload strategy is fine for Player because nothing references Player
rows by a stable ID across a reload except sleeper_id (which fetch_adp.py
now explicitly re-syncs afterward). PlayerMetrics is looked up by the
player.id foreign key directly, so a full wipe-and-reinsert would just be
extra churn for no benefit — updating existing rows in place is simpler
and cheaper.

Author: Zach Cooper
"""

from sqlmodel import Session, select

from backend.db.models import PlayerMetrics


def upsert_metrics(session: Session, player_id: int, **fields) -> PlayerMetrics:
    """
    Inserts or updates the single PlayerMetrics row for a player.
    `fields` should match PlayerMetrics column names (season, through_week,
    targets_per_game, etc.) — unrecognized keys will raise, same as
    constructing PlayerMetrics(**fields) directly would.
    """
    existing = session.exec(
        select(PlayerMetrics).where(PlayerMetrics.player_id == player_id)
    ).first()

    if existing:
        for key, value in fields.items():
            setattr(existing, key, value)
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return existing

    metrics = PlayerMetrics(player_id=player_id, **fields)
    session.add(metrics)
    session.commit()
    session.refresh(metrics)
    return metrics


def get_metrics(session: Session, player_id: int) -> PlayerMetrics | None:
    """Returns the PlayerMetrics row for a player, or None if never computed."""
    return session.exec(
        select(PlayerMetrics).where(PlayerMetrics.player_id == player_id)
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
