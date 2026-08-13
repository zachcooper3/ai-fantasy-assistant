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
    # Sleeper's designation: "IR", "Out", "PUP", "Suspended", "Questionable",
    # "Doubtful", or None when healthy. Populated by sync_sleeper_ids.py.
    # Structured rather than left as prose in a retrieved chunk, because a
    # player who cannot play is a hard exclusion, not a risk to weigh — and
    # confirmed live, an IR player was recommended with that note sitting
    # unread in the prompt.
    injury_status: Optional[str] = Field(default=None, index=True)

    # --- Metadata ---
    # Timezone-aware, matching every other model here — the old naive
    # datetime.utcnow is deprecated and mixed aware/naive timestamps
    # across tables.
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DraftSession(SQLModel, table=True):
    """
    Single-row persistence of the active draft session's configuration.

    Exists so a backend crash/restart mid-draft is recoverable: before this
    table, all draft state lived in DraftStateService's process memory, and
    a restart on draft day lost every pick with no recovery path (the
    frontend dropped to the setup modal, and starting over wiped
    Player.is_available too). Config lives here; the pick journal is the
    DraftPick table below. Rehydration happens in main.py's lifespan via
    draft_session_repo.load_state().

    At most one row ever exists (fixed id=1 — this app manages exactly one
    draft at a time, same assumption as DraftStateService itself). Created
    on POST /api/draft/session, deleted on DELETE /api/draft/session.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    league_size: int
    my_draft_position: int
    total_rounds: int
    scoring_format: str = "ppr"
    qb_slots: int = 1
    rb_slots: int = 2
    wr_slots: int = 2
    te_slots: int = 1
    flex_slots: int = 1
    dst_slots: int = 1
    # Which Claude model the AI panel's toggle is currently set to ("haiku" /
    # "sonnet" — see AI_MODEL_CHOICES in ai_service.py), so a mid-draft
    # backend restart resumes on the model you'd switched to rather than
    # silently falling back to the CLAUDE_MODEL env default. None means "no
    # explicit choice yet" — AIService keeps using its own env-derived
    # default in that case.
    ai_model: Optional[str] = Field(default=None)
    # The Sleeper draft ID live sync is (or was) polling, so a backend
    # restart mid-draft can resume sync instead of silently dropping it.
    # Before this field, main.py's lifespan restored picks/config but never
    # sync itself — DraftSyncService always starts at status="idle" on a
    # fresh process, and a null sync status is indistinguishable from "no
    # sync" to the frontend, so the manual pick/undo controls silently
    # reappeared over a draft the user still believed was syncing live.
    # Set in POST /api/sync/start, cleared in DELETE /api/sync/stop (an
    # explicit stop means don't resume this on the next restart) and
    # implicitly on any save_config (a new session has no sync to inherit).
    sleeper_draft_id: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DraftPick(SQLModel, table=True):
    """
    Append-only journal of picks in the active draft session — the
    persistence counterpart of DraftStateService._picks (see DraftSession
    above for why this exists). One row per recorded pick; rows are removed
    on undo and cleared wholesale on session create/reset.

    player_id is deliberately NOT a foreign key: sync records unresolvable
    players with the placeholder id -1 (see draft_sync._process_pick), and
    Player.id itself isn't reingest-stable (see PlayerMetrics' docstring).
    The denormalized name/position/team fields make a journaled pick
    self-describing even if the Player table shifts underneath it — and
    startup rehydration deliberately skips the ADP auto-refresh while a
    session exists, precisely so that shift can't happen mid-draft.

    pick_number is indexed but not unique: manual entry and live sync can
    disagree on numbering (logged as a divergence warning in
    record_synced_pick), and refusing to journal the pick would be worse
    than journaling the disagreement.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    pick_number: int = Field(index=True)
    round_number: int
    team_slot: int
    player_id: int
    player_name: str
    position: str
    nfl_team: str


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
    # The team these numbers were earned with — NOT necessarily the player's
    # current team. Comparing the two is what lets the app see that a
    # teammate who took 25% of the targets has left, so the incumbent's own
    # share understates the opportunity in front of him. See
    # _format_roster_changes in ai_service.py.
    team: Optional[str] = Field(default=None, index=True)
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

    # NOTE: `is_rookie_or_second_year` was declared here and read in three
    # places, but no ingestion script ever wrote it — 0 of 185 rows were ever
    # populated, so every reader silently saw False for everyone, including a
    # "Rookie or second-year: Yes" line in fetch_synthesis.py's prompt that
    # could never fire. Experience now comes from DraftProfile.draft_year,
    # which IS populated and has exactly one source of truth (see
    # ai_service._format_metrics_section).
    #
    # Dropped from the model rather than backfilled, but NOT dropped from
    # existing databases: migrations.py is additive-only on purpose (see its
    # docstring), and an unmapped leftover column is inert — SQLAlchemy names
    # the columns it selects, so it's simply never read. Fresh databases
    # won't have it at all.

    # --- Metadata ---
    source: str = Field(default="nflreadpy")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DraftProfile(SQLModel, table=True):
    """
    Pre-NFL prospect facts for a player — draft capital (round/pick/team,
    college) plus final-college-season production — for players who
    structurally can never have a PlayerMetrics row (see that model's
    docstring) because they haven't played an NFL down yet. Unlike
    PlayerMetrics, these are static historical facts rather than a
    recomputed snapshot: they don't change once a draft class and college
    career are in the books.

    Two independent sources populate this one row over time, upserting
    different field subsets (see upsert_draft_profile — it only touches
    fields explicitly passed, so one source never clobbers the other's
    columns):
      - fetch_draft_profiles.py   — draft_year/round/pick/team/college,
        via nflreadpy's load_draft_picks().
      - fetch_college_stats.py    — the college production fields below,
        via CollegeFootballData.com's get_player_season_stats(). Only
        enriches players who already have a row from the source above
        (i.e., were actually drafted) — an undrafted rookie with real
        college production is a real but rare gap this doesn't cover yet.

    This exists specifically to give rookies *something* concrete to
    reason about before they have any NFL season to generate a
    PlayerMetrics row from — draft capital (round/pick) is one of the most
    predictive signals for a rookie's fantasy outlook, and college
    production is the closest thing to a performance track record they
    have. ADP alone carries neither. See ai_service.py's Opportunity &
    Performance Signals section, which falls back to this for exactly
    that case.

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

    # --- College production (final college season — typically draft_year
    # minus 1) — see fetch_college_stats.py. A true "Dominator Rating"
    # (share of team's total yards/TDs) would need team-level college
    # totals too, which this doesn't pull yet; these are raw counting
    # stats only, deliberately simpler for a first pass.
    college_season: Optional[int] = None
    passing_yards: Optional[int] = None
    passing_td: Optional[int] = None
    interceptions_thrown: Optional[int] = None
    rushing_yards: Optional[int] = None
    rushing_td: Optional[int] = None
    carries: Optional[int] = None
    receiving_yards: Optional[int] = None
    receiving_td: Optional[int] = None
    receptions: Optional[int] = None

    # --- Metadata ---
    source: str = Field(default="nflreadpy")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Game(SQLModel, table=True):
    """
    The published NFL schedule — one row per game — sourced from
    nflreadpy's load_schedules() (same nflverse data source fetch_metrics.py
    already depends on; see backend/ingestion/fetch_schedule.py).

    Exists so the AI service can reason about a player's ACTUAL upcoming
    opponent instead of asking Claude to recall or guess the schedule from
    its own training data. That matters specifically because the schedule
    for the season being drafted is typically published (mid-May) AFTER
    Claude's reliable knowledge cutoff for that season, so an unaided model
    has no real way to know it and risks stating a confident, wrong
    opponent/week rather than admitting the gap.

    Keyed by team abbreviation, not by any Player foreign key — deliberately.
    PlayerMetrics and DraftProfile both need player_id relinking machinery
    (see their docstrings) because Player rows get reassigned new
    autoincrement IDs on every ADP reingest. NFL team abbreviations don't
    have that problem: there are 32 of them, they don't get "reingested,"
    and "DET" means the same team from one refresh to the next. So this
    table needs none of that — see game_repo.replace_season, which does a
    plain delete-and-reinsert per season with no relink step at all.

    One row per game (not one row per team per week) — home_team/away_team
    together are enough to answer "who does team X play in week N" from
    either side, without doubling the row count.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    season: int = Field(index=True)
    week: int = Field(index=True)
    # "REG" | "POST" | "PRE" — nflverse's own values (see fetch_metrics.py's
    # _filter_regular_season, which confirmed "REG" live for this same
    # nflverse data source). Stored rather than pre-filtered out at
    # ingestion time, in case a caller ever wants playoff-week context.
    game_type: str = Field(default="REG", index=True)
    home_team: str = Field(index=True)
    away_team: str = Field(index=True)
    # Calendar date of the game, if nflverse provides one for this row yet
    # (early-offseason schedule releases sometimes have the week set but not
    # every kickoff time finalized) — optional so a missing value degrades
    # to "unknown," not a bad default.
    game_date: Optional[datetime] = None

    # --- Metadata ---
    source: str = Field(default="nflreadpy")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
