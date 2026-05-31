"""
CRUD operations for the Player table.
All functions take an explicit Session argument — no globals, easy to test.
Author: Zach Cooper
"""

from sqlmodel import Session, select
from backend.db.models import Player


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def get_all_players(session: Session) -> list[Player]:
    """Returns every player ordered by ADP (lowest = highest priority)."""
    return session.exec(select(Player).order_by(Player.adp)).all()


def get_available_players(
    session: Session,
    position: str | None = None,
) -> list[Player]:
    """
    Returns undrafted players ordered by ADP.
    Optionally filter by position (e.g. "RB", "WR").
    """
    query = select(Player).where(Player.is_available == True)
    if position:
        query = query.where(Player.position == position.upper())
    return session.exec(query.order_by(Player.adp)).all()


def get_top_available(
    session: Session,
    n: int = 10,
    position: str | None = None,
) -> list[Player]:
    """Returns the top N available players by ADP, optionally filtered by position."""
    query = (
        select(Player)
        .where(Player.is_available == True)
    )
    if position:
        query = query.where(Player.position == position.upper())
    return session.exec(query.order_by(Player.adp).limit(n)).all()


def get_player_by_id(session: Session, player_id: int) -> Player | None:
    """Returns a player by primary key, or None if not found."""
    return session.get(Player, player_id)


def get_player_by_name(session: Session, name: str) -> Player | None:
    """
    Returns the first player whose name matches (case-insensitive).
    Useful for resolving manual pick inputs like "Ja'Marr Chase".
    """
    return session.exec(
        select(Player).where(Player.name.ilike(f"%{name}%"))
    ).first()


def get_players_by_position(session: Session, position: str) -> list[Player]:
    """Returns all players at a position (available or not), ordered by ADP."""
    return session.exec(
        select(Player)
        .where(Player.position == position.upper())
        .order_by(Player.adp)
    ).all()


def count_available_by_position(session: Session) -> dict[str, int]:
    """
    Returns a dict of {position: count_of_available_players}.
    Used for positional scarcity calculations.

    Example: {"QB": 18, "RB": 42, "WR": 58, "TE": 16, "K": 12, "DEF": 10}
    """
    players = session.exec(
        select(Player).where(Player.is_available == True)
    ).all()

    counts: dict[str, int] = {}
    for player in players:
        counts[player.position] = counts.get(player.position, 0) + 1
    return counts


def get_player_by_sleeper_id(session: Session, sleeper_id: str) -> Player | None:
    """Returns a player by their Sleeper platform ID. Used for live draft sync."""
    return session.exec(
        select(Player).where(Player.sleeper_id == sleeper_id)
    ).first()


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def mark_as_drafted(session: Session, player_id: int) -> Player | None:
    """
    Marks a player as unavailable (drafted).
    Returns the updated player, or None if not found.
    """
    player = session.get(Player, player_id)
    if player is None:
        return None
    player.is_available = False
    session.add(player)
    session.commit()
    session.refresh(player)
    return player


def mark_available(session: Session, player_id: int) -> Player | None:
    """
    Re-marks a player as available. Useful for undoing a mis-entered pick.
    Returns the updated player, or None if not found.
    """
    player = session.get(Player, player_id)
    if player is None:
        return None
    player.is_available = True
    session.add(player)
    session.commit()
    session.refresh(player)
    return player


def reset_draft_availability(session: Session) -> int:
    """
    Resets all players to is_available=True. Used at the start of a new draft session.
    Returns the number of players reset.
    """
    players = session.exec(select(Player).where(Player.is_available == False)).all()
    for player in players:
        player.is_available = True
        session.add(player)
    session.commit()
    return len(players)


def get_handcuff(session: Session, player_id: int) -> Player | None:
    """
    Returns the best available handcuff target for a given RB.
    A handcuff is the next-highest ADP available RB on the same NFL team.

    Returns None if:
    - The player isn't found
    - The player isn't an RB
    - No other available RBs exist on the same team
    """
    player = session.get(Player, player_id)
    if player is None or player.position != "RB":
        return None

    return session.exec(
        select(Player)
        .where(Player.team == player.team)
        .where(Player.position == "RB")
        .where(Player.is_available == True)
        .where(Player.id != player_id)
        .order_by(Player.adp)
    ).first()


def get_handcuff(session: Session, player_id: int) -> "Player | None":
    """
    Returns the best available handcuff target for a given RB.
    A handcuff is the next-highest ADP available RB on the same NFL team.

    Returns None if:
    - The player isn't found
    - The player isn't an RB
    - No other available RBs exist on the same team
    """
    player = session.get(Player, player_id)
    if player is None or player.position != "RB":
        return None

    return session.exec(
        select(Player)
        .where(Player.team == player.team)
        .where(Player.position == "RB")
        .where(Player.is_available == True)
        .where(Player.id != player_id)
        .order_by(Player.adp)
    ).first()
