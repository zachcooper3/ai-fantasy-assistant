"""
SQLModel schemas for the fantasy football draft assistant.
Author: Zach Cooper
"""

from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel


class Player(SQLModel, table=True):
    """
    Represents a draftable NFL player, sourced from FantasyPros ADP data
    and enriched with Sleeper IDs when available.

    Columns sourced from FantasyPros CSV:
        rank, name, team, bye, pos_rank, adp

    Derived columns:
        position  — letters only, e.g. "RB" from "RB1"

    Runtime columns (mutated during draft):
        is_available  — set to False when a player is drafted
    """

    id: Optional[int] = Field(default=None, primary_key=True)

    # --- FantasyPros ADP data ---
    rank: int = Field(index=True)                        # Overall consensus rank (1 = best)
    name: str = Field(index=True)                        # Full player name
    team: str                                            # NFL team abbreviation, e.g. "DET"
    bye: Optional[int] = Field(default=None)             # Bye week (None if not yet set)
    pos_rank: str                                        # Position + rank, e.g. "RB1", "WR12"
    position: str = Field(index=True)                   # Position only, e.g. "RB", "WR"
    adp: float = Field(index=True)                      # Average draft position (lower = earlier)

    # --- Sleeper integration (populated later) ---
    sleeper_id: Optional[str] = Field(default=None, index=True)

    # --- Draft state ---
    is_available: bool = Field(default=True, index=True) # False once the player is drafted

    # --- Metadata ---
    updated_at: datetime = Field(default_factory=datetime.utcnow)
