"""
CRUD operations for the Player table.
All functions take an explicit Session argument — no globals, easy to test.
Author: Zach Cooper
"""

import re

from sqlmodel import Session, select
from backend.db.models import Player


# --- Name normalization — same idea as sync_sleeper_ids.py's normalizer,
# duplicated here rather than imported (a db-layer module importing from
# ingestion would be an odd dependency direction for ~10 lines; chunker.py
# made the same call for the same reason). ---

_GENERATIONAL_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _normalise_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def _strip_suffix(normalised: str) -> str:
    parts = normalised.split()
    if len(parts) > 1 and parts[-1] in _GENERATIONAL_SUFFIXES:
        return " ".join(parts[:-1])
    return normalised


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


def get_player_by_name(
    session: Session,
    name: str,
    position: str | None = None,
) -> Player | None:
    """
    Resolves a player by name — exact match after normalization (lowercase,
    punctuation stripped, generational suffix dropped), optionally
    constrained by position. Returns None on no match OR an ambiguous one:
    this is used by live sync's fallback path to mark players as drafted,
    where tagging the wrong player is strictly worse than a placeholder.

    This replaced an unanchored `ilike('%name%')` substring match (audit
    W4), which could hit the wrong player entirely — a substring of a
    longer name, a duplicate name at another position, or anything when
    the input contained SQL wildcard characters (%/_).

    Suffixes are stripped from BOTH sides before comparing because the two
    data sources disagree: Sleeper's names omit "Jr./III" entirely while
    the ADP data keeps them (see sync_sleeper_ids.py's module docstring).

    position, when given, should be this app's convention ("DST", not
    Sleeper's "DEF") — callers translate first (see draft_sync.py).
    """
    target = _strip_suffix(_normalise_name(name or ""))
    if not target:
        return None

    query = select(Player)
    if position:
        query = query.where(Player.position == position.upper())
    candidates = session.exec(query).all()

    matches = [
        p for p in candidates
        if _strip_suffix(_normalise_name(p.name)) == target
    ]
    return matches[0] if len(matches) == 1 else None


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

    Example: {"QB": 18, "RB": 42, "WR": 58, "TE": 16, "K": 12, "DST": 10}
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
