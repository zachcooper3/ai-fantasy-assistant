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

    player_id is NOT a stable cross-refresh identity, despite being a FK —
    ingest_players.ingest_csv() does a full delete-and-reinsert of Player on
    every ADP refresh, so autoincrement IDs get reassigned in whatever order
    that CSV happens to list players *this time*. Two players close in ADP
    (confirmed live: Jahmyr Gibbs and Bijan Robinson, ADP 1.7 vs 1.9) can
    have their relative CSV order flip between two pulls, which flips which
    one gets the lower ID — silently reattaching this row's stats to a
    DIFFERENT real player after the next reingest, no error, no crash.
    sleeper_id doesn't have this problem (sync_sleeper_ids.py re-resolves it
    by name-matching on every refresh, so it always points at the same real
    person) — that's why it's stored here too and why fetch_synthesis.py
    joins on it instead of player_id. player_id is kept in sync by
    metrics_repo.relink_player_ids(), called right after every Player
    reingest, but treat sleeper_id as the source of truth if the two ever
    disagree.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    player_id: int = Field(foreign_key="player.id", index=True, unique=True)
    sleeper_id: Optional[str] = Field(default=None, index=True)

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


class DraftProfile(SQLModel, table=True):
    """
    NFL draft-day facts for a player — round, pick, team, college — sourced
    from nflreadpy's load_draft_picks(). Unlike PlayerMetrics, this is a
    static historical fact rather than a recomputed snapshot: it doesn't
    change once a draft class is in the books.

    This exists specifically to give rookies (and recent draftees generally)
    *something* concrete to reason about before they have any NFL season to
    generate a PlayerMetrics row from — draft capital (round/pick) is one of
    the most predictive signals for a rookie's fantasy outlook, and ADP
    alone doesn't carry it. See ai_service.py's Opportunity & Performance
    Signals section, which falls back to this for exactly that case.

    Keyed by sleeper_id, not just player_id, for the same reason as
    PlayerMetrics (see its docstring above) — ingest_players.ingest_csv()
    does a full delete-and-reinsert of Player on every ADP refresh, so
    player_id is not a stable identity across refreshes. draft_profile_repo
    .relink_player_ids() repairs this the same way metrics_repo's does.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    player_id: int = Field(foreign_key="player.id", index=True, unique=True)
    sleeper_id: Optional[str] = Field(default=None, index=True)

    draft_year: int = Field(index=True)
    draft_round: Optional[int] = None
    draft_pick: Optional[int] = None       # overall pick number
    draft_team: Optional[str] = None       # NFL team abbreviation at draft time
    college: Optional[str] = None

    # --- Metadata ---
    source: str = Field(default="nflreadpy")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
