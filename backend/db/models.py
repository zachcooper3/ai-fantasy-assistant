"""
SQLModel schemas for the fantasy football draft assistant.
Author: Zach Cooper
"""

from datetime import datetime, timezone
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


class PlayerMetrics(SQLModel, table=True):
    """
    Analytics for a player, computed from nflverse stats (via nflreadpy) —
    separate from Player's ADP/rank fields, which come from
    FantasyFootballCalculator instead. See backend/ingestion/fetch_metrics.py.

    This is a rolling snapshot, not a per-week time series: one row per
    player, recomputed in place on each refresh. `season` and `through_week`
    record how much data went into it so callers (and Claude's prompt) can
    judge small-sample-size rows appropriately.

    Every metric is Optional — nflverse data can be sparse for a given
    player/week (e.g. a rookie with one game played, or a position where a
    stat doesn't apply), and a missing metric should read as "unknown," not
    silently become a misleading 0.

    Five categories, per the project's original scope:
      Opportunity/Volume, Efficiency, Team Context, Consistency & Risk,
      Forward-looking/Prospect Signals.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    player_id: int = Field(foreign_key="player.id", index=True, unique=True)

    season: int = Field(index=True)
    through_week: int          # last week of data included in this snapshot
    games_played: int          # games in the window — a small-sample guard

    # --- Opportunity / Volume ---
    targets_per_game: Optional[float] = None
    carries_per_game: Optional[float] = None
    red_zone_touches_per_game: Optional[float] = None
    snap_pct: Optional[float] = None              # 0-1
    target_share: Optional[float] = None          # 0-1, share of team targets
    carry_share: Optional[float] = None           # 0-1, share of team carries

    # --- Efficiency ---
    yards_per_target: Optional[float] = None
    yards_per_carry: Optional[float] = None
    yac_per_reception: Optional[float] = None
    racr: Optional[float] = None                  # receiving yards / air yards
    catch_rate: Optional[float] = None            # 0-1, receptions / targets

    # --- Team Context ---
    team_pass_rate: Optional[float] = None        # 0-1, team's pass-play rate
    depth_chart_rank: Optional[int] = None         # 1 = starter at the position

    # --- Consistency & Risk ---
    fantasy_points_avg: Optional[float] = None     # PPR, per game
    fantasy_points_stdev: Optional[float] = None   # week-to-week, PPR
    injury_report_appearances: int = 0             # weeks appearing on any injury report
    games_missed: int = 0                          # games missed to injury this season

    # --- Forward-looking / Prospect Signals ---
    target_share_trend: Optional[float] = None    # last 3 wks minus season avg; + = rising role
    snap_pct_trend: Optional[float] = None        # same idea, snap share
    depth_chart_trend: Optional[int] = None       # rank change over last 3 wks; negative = moving up
    is_rookie_or_second_year: bool = False

    # --- Metadata ---
    source: str = Field(default="nflreadpy")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
