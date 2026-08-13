"""
Tests for GET /api/players/{id}/detail — the endpoint behind the player
detail drawer.

The behaviour worth pinning down here is almost all about ABSENCE. Every
metric on PlayerMetrics is Optional, PlayerMetrics itself is missing for
anyone without an NFL season, DraftProfile is missing for undrafted
players, and the schedule table may not have been ingested at all. The
endpoint's contract is that each of those is a 200 with a null/empty
field — never a 404, and never a zero standing in for a real measurement,
because a 0 target share and an unknown target share look identical once
they reach the UI and mean opposite things.

Author: Zach Cooper
"""

import pytest
from sqlmodel import Session

from backend.db.models import DraftProfile, Game, PlayerMetrics
from tests.conftest import make_player


@pytest.fixture
def detail_db(engine):
    """
    A veteran with metrics, a rookie with only a draft profile, and a
    player with neither — the three shapes the drawer has to render.
    """
    with Session(engine) as db:
        db.add(make_player(1, "Vet Back", "RB", "DET", adp=1.5, rank=1, sleeper_id="s1"))
        db.add(make_player(2, "Rook Back", "RB", "ARI", adp=25.0, rank=20, sleeper_id="s2"))
        db.add(make_player(3, "Nobody Wide", "WR", "MIA", adp=180.0, rank=180))

        db.add(
            PlayerMetrics(
                player_id=1,
                sleeper_id="s1",
                season=2025,
                team="DET",
                through_week=18,
                games_played=17,
                target_share=0.17,
                snap_pct=0.67,
                fantasy_points_avg=21.5,
                depth_chart_trend=-1,
                # Everything else deliberately left None — this is the
                # sparse-coverage case the response has to preserve.
            )
        )
        db.add(
            DraftProfile(
                player_id=2,
                sleeper_id="s2",
                draft_year=2026,
                draft_round=1,
                draft_pick=3,
                draft_team="ARI",
                college="Notre Dame",
                college_season=2025,
                rushing_yards=1372,
                rushing_td=18,
            )
        )

        # Two weeks of schedule for DET, none for ARI or MIA.
        db.add(Game(season=2026, week=1, game_type="REG", home_team="DET", away_team="NO"))
        db.add(Game(season=2026, week=2, game_type="REG", home_team="BUF", away_team="DET"))
        # A postseason row that must not leak into the response.
        db.add(Game(season=2026, week=19, game_type="POST", home_team="DET", away_team="GB"))
        db.commit()
    return engine


def test_veteran_returns_metrics_and_no_draft_profile(client, detail_db):
    r = client.get("/api/players/1/detail")
    assert r.status_code == 200
    body = r.json()

    assert body["player"]["name"] == "Vet Back"
    assert body["draft_profile"] is None

    m = body["metrics"]
    assert m["season"] == 2025
    assert m["games_played"] == 17
    assert m["target_share"] == pytest.approx(0.17)

    # The sparse fields must survive as null, not as 0. This is the whole
    # reason the model's fields are Optional; a 0.0 here would render in the
    # drawer as a measured zero.
    assert m["carries_per_game"] is None
    assert m["yards_per_carry"] is None
    assert m["racr"] is None


def test_rookie_returns_draft_profile_and_no_metrics(client, detail_db):
    r = client.get("/api/players/2/detail")
    assert r.status_code == 200
    body = r.json()

    assert body["metrics"] is None

    dp = body["draft_profile"]
    assert dp["draft_round"] == 1
    assert dp["draft_pick"] == 3
    assert dp["college"] == "Notre Dame"
    assert dp["rushing_yards"] == 1372
    # A running back has no passing stats; those columns stay null rather
    # than becoming a 0-yard passing line.
    assert dp["passing_yards"] is None


def test_player_with_neither_is_still_a_200(client, detail_db):
    """A thin ADP row is an ordinary state, not an error."""
    r = client.get("/api/players/3/detail")
    assert r.status_code == 200
    body = r.json()
    assert body["metrics"] is None
    assert body["draft_profile"] is None
    assert body["player"]["name"] == "Nobody Wide"


def test_season_is_inferred_from_the_data_not_the_clock(client, detail_db):
    """
    max(PlayerMetrics.season) + 1 vs. max(DraftProfile.draft_year), later
    wins — the same rule as ai_service._infer_current_season. Both signals
    say 2026 here, which is what a real draft-season database looks like.
    """
    assert client.get("/api/players/1/detail").json()["season"] == 2026


def test_schedule_is_regular_season_only_and_ordered(client, detail_db):
    schedule = client.get("/api/players/1/detail").json()["schedule"]

    assert [g["week"] for g in schedule] == [1, 2]
    assert schedule[0] == {"week": 1, "opponent": "NO", "is_home": True}
    # Read from the away side too.
    assert schedule[1] == {"week": 2, "opponent": "BUF", "is_home": False}


def test_missing_schedule_is_empty_not_an_error(client, detail_db):
    """
    No rows for this team is "unknown", the same convention game_repo uses —
    it must not be reported as a bye, and it must not fail the request.
    """
    body = client.get("/api/players/2/detail").json()
    assert body["schedule"] == []
    assert body["season"] == 2026


def test_unknown_player_is_404(client, detail_db):
    assert client.get("/api/players/999999/detail").status_code == 404


def test_detail_route_does_not_shadow_the_plain_player_route(client, detail_db):
    """
    Both live under /api/players/{player_id}. Declaration order keeps the
    more specific path first; this asserts the plain one still resolves.
    """
    r = client.get("/api/players/1")
    assert r.status_code == 200
    assert r.json()["name"] == "Vet Back"
    assert "metrics" not in r.json()


def test_injury_status_reaches_the_client(client, engine):
    """
    The field the frontend renders as a badge. It has been on PlayerResponse
    for a while but nothing consumed it, so nothing caught it going missing.
    """
    with Session(engine) as db:
        db.add(make_player(10, "Hurt Back", "RB", "DET", injury_status="IR"))
        db.commit()

    assert client.get("/api/players/10/detail").json()["player"]["injury_status"] == "IR"
