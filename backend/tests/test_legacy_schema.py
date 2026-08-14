"""
Tests that run against a database built the way a REAL one was built, not
the way the current model would build it.

Why this file exists: on 2026-08-13 a field was removed from PlayerMetrics
after confirming nothing read it. Every test passed. fetch_metrics then
crashed on the live database with "NOT NULL constraint failed", because the
real table still had the column — `BOOLEAN NOT NULL` with the default in
Python rather than in SQL — so any INSERT that didn't name it was invalid.
UPDATEs of existing rows were fine, which is what made it look safe.

The whole suite missed it for one structural reason: conftest's `engine`
fixture calls SQLModel.metadata.create_all(), so every test database is
built from the model as it exists RIGHT NOW. A column the model has since
forgotten about cannot appear, so the failure mode cannot be reproduced.

Anything that changes PlayerMetrics' or Player's shape should get a case
here, built with explicit DDL that mirrors what's actually on disk.

Author: Zach Cooper
"""

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select, text

from backend.db import metrics_repo
from backend.db.models import PlayerMetrics


# Copied verbatim from the live database on 2026-08-13:
#   sqlite> select sql from sqlite_master where name='playermetrics';
#
# Reproduced as raw DDL rather than built with ALTER TABLE, because the
# distinction that caused the bug cannot survive an ALTER: SQLite REFUSES to
# add a NOT NULL column without a DEFAULT, so `ALTER TABLE ... ADD COLUMN x
# BOOLEAN NOT NULL DEFAULT 0` produces a column that INSERTs can legally
# omit — which is precisely the case that does NOT fail, and therefore
# precisely the wrong thing to test against. The real column has
# `notnull=1, dflt_value=None`, only reachable via CREATE TABLE.
_LEGACY_PLAYERMETRICS_DDL = """
CREATE TABLE playermetrics (
    id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    sleeper_id VARCHAR,
    season INTEGER NOT NULL,
    through_week INTEGER NOT NULL,
    games_played INTEGER NOT NULL,
    targets_per_game FLOAT,
    carries_per_game FLOAT,
    red_zone_touches_per_game FLOAT,
    snap_pct FLOAT,
    target_share FLOAT,
    carry_share FLOAT,
    yards_per_target FLOAT,
    yards_per_carry FLOAT,
    yac_per_reception FLOAT,
    racr FLOAT,
    catch_rate FLOAT,
    team_pass_rate FLOAT,
    depth_chart_rank INTEGER,
    fantasy_points_avg FLOAT,
    fantasy_points_stdev FLOAT,
    injury_report_appearances INTEGER NOT NULL,
    games_missed INTEGER NOT NULL,
    target_share_trend FLOAT,
    snap_pct_trend FLOAT,
    depth_chart_trend INTEGER,
    is_rookie_or_second_year BOOLEAN NOT NULL,
    source VARCHAR NOT NULL,
    last_updated DATETIME NOT NULL,
    team VARCHAR,
    PRIMARY KEY (id),
    FOREIGN KEY(player_id) REFERENCES player (id)
)
"""


@pytest.fixture
def legacy_engine():
    """A database carrying the pre-2026-08-13 PlayerMetrics shape.

    Every other table comes from create_all(); playermetrics is dropped and
    recreated from the captured DDL so the retired column keeps its real
    constraints. That's the state every database created before the field
    was retired is actually in.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        s.exec(text("DROP TABLE playermetrics"))
        s.exec(text(_LEGACY_PLAYERMETRICS_DDL))
        s.commit()

    assert _legacy_column_is_notnull_without_default(engine), (
        "fixture failed to reproduce the legacy constraint — these tests "
        "would pass vacuously"
    )
    return engine


def _legacy_column_is_notnull_without_default(engine) -> bool:
    with Session(engine) as s:
        for r in s.exec(text("PRAGMA table_info(playermetrics)")):
            if r[1] == "is_rookie_or_second_year":
                notnull, default = r[3], r[4]
                return bool(notnull) and default is None
    return False


def test_the_retired_column_still_exists_in_the_model():
    """If this fails, someone removed the field again. Read the comment on
    it in models.py first — the removal is what broke the live database."""
    assert "is_rookie_or_second_year" in PlayerMetrics.model_fields


def test_insert_succeeds_against_a_legacy_table(legacy_engine):
    """The exact operation that crashed: a brand-new metrics row. This is
    an INSERT, so every NOT NULL column has to be named."""
    with Session(legacy_engine) as s:
        metrics_repo.upsert_metrics(
            s, player_id=387, sleeper_id="17",
            season=2025, through_week=18, games_played=15,
            red_zone_touches_per_game=0.0,
        )
        rows = s.exec(select(PlayerMetrics)).all()

    assert len(rows) == 1
    assert rows[0].sleeper_id == "17"


def test_update_of_an_existing_row_still_works(legacy_engine):
    """UPDATE was never broken — pinned so a future fix doesn't trade one
    failure mode for the other."""
    with Session(legacy_engine) as s:
        metrics_repo.upsert_metrics(
            s, player_id=1, sleeper_id="a",
            season=2025, through_week=18, games_played=17,
            red_zone_touches_per_game=2.0,
        )
        metrics_repo.upsert_metrics(
            s, player_id=1, sleeper_id="a",
            season=2025, through_week=18, games_played=17,
            red_zone_touches_per_game=0.5,
        )
        rows = s.exec(select(PlayerMetrics)).all()

    assert len(rows) == 1
    assert rows[0].red_zone_touches_per_game == 0.5


def test_many_inserts_in_one_run(legacy_engine):
    """fetch_metrics writes ~185 rows in a loop, committing per player. The
    live crash happened on the first insert; make sure a batch is fine too."""
    with Session(legacy_engine) as s:
        for i in range(1, 51):
            metrics_repo.upsert_metrics(
                s, player_id=i, sleeper_id=str(i),
                season=2025, through_week=18, games_played=17,
            )
        assert len(s.exec(select(PlayerMetrics)).all()) == 50


def test_relink_works_against_a_legacy_table(legacy_engine):
    """relink stages player_id through a negative placeholder and re-commits,
    which is another write path that has to satisfy the same constraint."""
    from backend.tests.conftest import make_player

    with Session(legacy_engine) as s:
        s.add(make_player(1, "Gibbs", sleeper_id="gibbs"))
        s.add(make_player(2, "Robinson", sleeper_id="robinson"))
        s.commit()
        metrics_repo.upsert_metrics(s, player_id=2, sleeper_id="gibbs",
                                    season=2025, through_week=18, games_played=17)
        metrics_repo.upsert_metrics(s, player_id=1, sleeper_id="robinson",
                                    season=2025, through_week=18, games_played=17)

        relinked, orphaned = metrics_repo.relink_player_ids(s)
        assert (relinked, orphaned) == (2, 0)

        rows = {m.sleeper_id: m.player_id for m in s.exec(select(PlayerMetrics)).all()}
    assert rows == {"gibbs": 1, "robinson": 2}
