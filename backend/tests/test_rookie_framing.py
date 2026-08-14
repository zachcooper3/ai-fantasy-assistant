"""
Tests that the rookie synthesis prompt only calls a player a rookie when he
actually is one.

Background: this module selects players who have a DraftProfile but no
PlayerMetrics row. That was a sound proxy for "rookie" while
fetch_draft_profiles pulled two draft classes. Widening it to four on
2026-08-14 — done deliberately, to reach players whose problem is no RECENT
season rather than no season at all — silently broke the proxy. Fourteen
players were then told to be evaluated "as a rookie prospect," including
Tank Dell, a third-year receiver with a productive 2023 behind him.

The selection was never wrong: these players genuinely have no usage data,
which is exactly why they need a draft-capital note. Only the framing was.

Author: Zach Cooper
"""

import pytest

from backend.db.models import DraftProfile, Player
from backend.ingestion.fetch_rookie_synthesis import (
    _status_line,
    format_draft_profile_prompt,
)

CURRENT_CLASS = 2026


def _profile(year: int, **over) -> DraftProfile:
    base = dict(player_id=1, sleeper_id="s1", draft_year=year,
                draft_round=2, draft_pick=46, draft_team="CAR", college="Texas")
    base.update(over)
    return DraftProfile(**base)


def _player(name="Jonathon Brooks", position="RB") -> Player:
    return Player(id=1, rank=100, name=name, team="CAR", bye=7,
                  pos_rank=f"{position}30", position=position, adp=107.6,
                  sleeper_id="s1")


# ---------------------------------------------------------------------------
# The status line
# ---------------------------------------------------------------------------

def test_current_class_is_labelled_an_incoming_rookie():
    line = _status_line(_profile(2026), CURRENT_CLASS)
    assert "incoming rookie" in line
    assert "has not yet played" in line


def test_a_future_class_is_still_treated_as_incoming():
    """Guards the boundary if this ever runs between the draft and the
    season rolling over."""
    assert "incoming rookie" in _status_line(_profile(2027), CURRENT_CLASS)


@pytest.mark.parametrize("year,seasons", [(2025, 1), (2024, 2), (2023, 3)])
def test_earlier_classes_are_explicitly_not_rookies(year, seasons):
    line = _status_line(_profile(year), CURRENT_CLASS)
    assert "NOT a rookie" in line
    assert f"{seasons} season(s) ago" in line
    assert "Do not describe him as new to the league" in line


def test_the_tank_dell_case():
    """The clearest live miss — a 2023 third-rounder with real NFL
    production, described as a prospect who had never played."""
    line = _status_line(_profile(2023, draft_round=3, draft_pick=69,
                                 draft_team="HOU", college="Houston"), CURRENT_CLASS)
    assert "NOT a rookie" in line
    assert "rookie," not in line.split("NOT a rookie")[0]


# ---------------------------------------------------------------------------
# Placement in the prompt
# ---------------------------------------------------------------------------

def test_status_is_the_second_line_so_it_cannot_be_missed():
    out = format_draft_profile_prompt(_player(), _profile(2024))
    lines = out.splitlines()
    assert lines[0].startswith("Player:")
    assert lines[1].startswith("Status:")


def test_status_appears_for_incoming_rookies_too():
    out = format_draft_profile_prompt(_player(), _profile(2026))
    assert "Status: incoming rookie" in out


def test_the_rest_of_the_block_is_unchanged():
    out = format_draft_profile_prompt(_player(), _profile(2024, college_season=2023,
                                                          rushing_yards=1139, rushing_td=10))
    assert "Draft Capital:" in out
    assert "- Round: 2" in out
    assert "- Pick (overall): 46" in out
    assert "Final College Season Production:" in out
    assert "- Rushing yards: 1139" in out


def test_a_profile_with_no_college_stats_still_gets_a_status():
    """Dell's shape — draft capital only, CFBD never matched him."""
    out = format_draft_profile_prompt(
        _player("Tank Dell", "WR"),
        _profile(2023, draft_round=3, draft_pick=69, draft_team="HOU", college="Houston"),
    )
    assert "NOT a rookie" in out
    assert "Final College Season Production" not in out
