"""
CRUD operations for the Game table (the published NFL schedule).
All functions take an explicit Session argument — no globals, easy to test.

Unlike metrics_repo/draft_profile_repo, there is no player_id relink step
here — Game rows are keyed by team abbreviation, which is stable across
refreshes (see Game's docstring). replace_season is a plain
delete-and-reinsert, the same shape as ingest_players.ingest_csv() uses for
Player, and it's safe for the same reason: nothing else holds a foreign key
into this table, so there's no downstream row that could be silently
reattached to the wrong game the way PlayerMetrics/DraftProfile can be
reattached to the wrong player.

Author: Zach Cooper
"""

from sqlmodel import Session, select

from backend.db.models import Game


def replace_season(session: Session, season: int, games: list[dict]) -> int:
    """
    Deletes every existing Game row for `season`, then inserts `games`
    fresh. The published schedule doesn't change incrementally — a re-run
    of fetch_schedule.py is always "here is the current full picture for
    this season," not a delta — so wipe-and-reinsert is both simpler and
    more correct than trying to diff and update in place (e.g. a game that
    got flexed to a different week is naturally handled: the old row is
    gone and the new one just has the new week, rather than needing
    explicit move-detection logic).

    `games` entries should have keys matching Game's non-metadata columns
    (season, week, game_type, home_team, away_team, game_date) — season is
    set here from the `season` argument regardless of what's in the dict,
    so callers don't need to repeat it per row.

    Returns the number of rows inserted.
    """
    existing = session.exec(select(Game).where(Game.season == season)).all()
    for row in existing:
        session.delete(row)
    session.commit()

    inserted = 0
    for g in games:
        session.add(Game(
            season=season,
            week=g["week"],
            game_type=g.get("game_type") or "REG",
            home_team=g["home_team"],
            away_team=g["away_team"],
            game_date=g.get("game_date"),
        ))
        inserted += 1
    session.commit()
    return inserted


def get_opponent(session: Session, team: str, season: int, week: int) -> str | None:
    """
    Returns the opponent team abbreviation for `team` in a given
    season/week, or None if there's no game on record (bye week, team not
    found, or schedule not yet ingested for that season — all three look
    the same from here, which is correct: callers should treat "no
    opponent found" as "unknown," not as "confirmed bye").
    """
    game = session.exec(
        select(Game).where(
            Game.season == season,
            Game.week == week,
            (Game.home_team == team) | (Game.away_team == team),
        )
    ).first()
    if game is None:
        return None
    return game.away_team if game.home_team == team else game.home_team


def get_remaining_schedule(
    session: Session, team: str, season: int, from_week: int
) -> list[dict]:
    """
    Returns [{week, opponent, is_home}, ...] for `team`, from `from_week`
    onward (inclusive), ordered by week. Regular season only (game_type
    "REG") — a bench pick's playoff-schedule strength isn't a real signal
    fantasy leagues care about, since most leagues finish before Week 18.
    """
    games = session.exec(
        select(Game).where(
            Game.season == season,
            Game.week >= from_week,
            Game.game_type == "REG",
            (Game.home_team == team) | (Game.away_team == team),
        ).order_by(Game.week)
    ).all()
    return [
        {
            "week": g.week,
            "opponent": g.away_team if g.home_team == team else g.home_team,
            "is_home": g.home_team == team,
        }
        for g in games
    ]


def get_schedules_bulk(
    session: Session, teams: list[str], season: int, from_week: int, through_week: int,
) -> dict[str, list[dict]]:
    """
    Returns {team: [{week, opponent, is_home}, ...]} for every team in
    `teams`, one query instead of one per team — this is what
    RecommendationContext.team_schedules is populated from (see
    recommendations.py::_build_context), and the board can have a couple
    dozen distinct teams on it, so a per-team round trip would be the same
    N+1 shape player_id relinking was built to avoid elsewhere in this app.

    Regular season only, `from_week` through `through_week` inclusive.
    Teams with no rows in range (bye week inside the window, team not
    found, or schedule never ingested for this season) are simply absent
    from the result — same "missing means unknown" convention
    get_opponent's docstring describes, just bulk.
    """
    if not teams:
        return {}
    games = session.exec(
        select(Game).where(
            Game.season == season,
            Game.week >= from_week,
            Game.week <= through_week,
            Game.game_type == "REG",
            (Game.home_team.in_(teams)) | (Game.away_team.in_(teams)),
        ).order_by(Game.week)
    ).all()

    out: dict[str, list[dict]] = {}
    team_set = set(teams)
    for g in games:
        for team, opponent, is_home in (
            (g.home_team, g.away_team, True),
            (g.away_team, g.home_team, False),
        ):
            if team in team_set:
                out.setdefault(team, []).append(
                    {"week": g.week, "opponent": opponent, "is_home": is_home}
                )
    return out


def has_season(session: Session, season: int) -> bool:
    """Whether any schedule data has been ingested for `season` at all —
    lets callers distinguish "no game this week" (bye) from "schedule was
    never pulled for this season" without guessing from an empty list."""
    return session.exec(
        select(Game.id).where(Game.season == season).limit(1)
    ).first() is not None
