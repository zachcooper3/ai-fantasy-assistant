"""
Tests for ingest_players' position filtering and the refresh plan's step
list — the two "minor" findings from the 2026-08-13 ingestion audit that
change observable behaviour.

Author: Zach Cooper
"""

import csv

import pytest
from sqlmodel import Session, select

from backend.db.models import Player
from backend.ingestion import ingest_players, refresh
from backend.ingestion.ingest_players import VALID_POSITIONS


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Rank", "Player", "Team", "Bye", "POS", "AVG"])
        w.writeheader()
        w.writerows(rows)
    return str(path)


@pytest.fixture
def ingest_to(engine, monkeypatch):
    """Points ingest_csv at the in-memory test engine instead of the real
    data/fantasy.db — see the project's standing rule about never writing to
    the live database from a test run."""
    monkeypatch.setattr(ingest_players, "engine", engine)
    monkeypatch.setattr(ingest_players, "create_db_and_tables", lambda: None)
    return engine


# ---------------------------------------------------------------------------
# Position filtering
# ---------------------------------------------------------------------------

def test_defensive_player_is_dropped(tmp_path, ingest_to):
    """The live case: FantasyPros' 600-row export carries the odd IDP pick
    (Ben VanSumeren, LB, rank 574), which has no home in a standard league."""
    path = _write_csv(tmp_path / "adp.csv", [
        {"Rank": 1, "Player": "Jahmyr Gibbs", "Team": "DET", "Bye": 6, "POS": "RB1", "AVG": 1.0},
        {"Rank": 574, "Player": "Ben VanSumeren", "Team": "BUF", "Bye": 7, "POS": "LB", "AVG": 471.0},
    ])
    count = ingest_players.ingest_csv(path)

    assert count == 1
    with Session(ingest_to) as db:
        names = [p.name for p in db.exec(select(Player)).all()]
    assert names == ["Jahmyr Gibbs"]


def test_every_modelled_position_survives(tmp_path, ingest_to):
    """The filter must not be stricter than the app itself — DST and K are
    real draftable entries here even though they carry no PlayerMetrics."""
    rows = [
        {"Rank": 1, "Player": "A Quarterback", "Team": "BUF", "Bye": 7, "POS": "QB1", "AVG": 30.0},
        {"Rank": 2, "Player": "A Back", "Team": "DET", "Bye": 6, "POS": "RB1", "AVG": 1.0},
        {"Rank": 3, "Player": "A Receiver", "Team": "KC", "Bye": 10, "POS": "WR1", "AVG": 5.0},
        {"Rank": 4, "Player": "An End", "Team": "SF", "Bye": 9, "POS": "TE1", "AVG": 20.0},
        {"Rank": 5, "Player": "A Kicker", "Team": "LV", "Bye": 13, "POS": "K1", "AVG": 150.0},
        {"Rank": 6, "Player": "Houston Texans DST", "Team": "HOU", "Bye": 8, "POS": "DST1", "AVG": 140.0},
    ]
    count = ingest_players.ingest_csv(_write_csv(tmp_path / "adp.csv", rows))

    assert count == 6
    with Session(ingest_to) as db:
        positions = {p.position for p in db.exec(select(Player)).all()}
    assert positions == VALID_POSITIONS


def test_def_is_normalised_to_dst_before_the_filter_runs(tmp_path, ingest_to):
    """FantasyFootballCalculator says DEF, this app says DST. The filter has
    to sit downstream of that normalisation or the whole FFC path drops all
    32 defenses."""
    path = _write_csv(tmp_path / "adp.csv", [
        {"Rank": 1, "Player": "Houston Texans", "Team": "HOU", "Bye": 8, "POS": "DEF", "AVG": 140.0},
    ])
    assert ingest_players.ingest_csv(path) == 1

    with Session(ingest_to) as db:
        assert db.exec(select(Player)).first().position == "DST"


def test_a_csv_of_only_bad_rows_ingests_nothing_without_crashing(tmp_path, ingest_to):
    path = _write_csv(tmp_path / "adp.csv", [
        {"Rank": 1, "Player": "A Linebacker", "Team": "BUF", "Bye": 7, "POS": "LB", "AVG": 400.0},
    ])
    assert ingest_players.ingest_csv(path) == 0


# ---------------------------------------------------------------------------
# Refresh plan
# ---------------------------------------------------------------------------

def test_schedule_is_in_the_default_plan():
    """It was absent until 2026-08-13 despite the module promising to
    refresh every source the app draws on."""
    names = [s.name for s in refresh._plan(with_ai=False, only=None)]
    assert "schedule" in names


def test_schedule_costs_no_claude_calls():
    schedule = next(s for s in refresh._STEPS if s.name == "schedule")
    assert not schedule.uses_claude
    assert not schedule.critical


def test_adp_is_still_excluded_from_the_default_plan():
    """The hand-curated CSV must never be clobbered by a habitual run."""
    assert "adp" not in [s.name for s in refresh._plan(with_ai=False, only=None)]
    assert "adp" not in [s.name for s in refresh._plan(with_ai=True, only=None)]


def test_adp_still_runs_when_named_explicitly():
    assert [s.name for s in refresh._plan(False, ["adp"])] == ["adp"]


def test_schedule_can_be_run_alone():
    assert [s.name for s in refresh._plan(False, ["schedule"])] == ["schedule"]


def test_ids_runs_before_everything_that_depends_on_it():
    names = [s.name for s in refresh._plan(with_ai=True, only=None)]
    for dependent in ("metrics", "draft", "college", "news", "synthesis", "rookies"):
        assert names.index("ids") < names.index(dependent)
