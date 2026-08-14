"""
Tests for RACR suppression (fetch_metrics._racr).

RACR = receiving yards / air yards. Real values sit around 0.5-1.5. This
metric has now been wrong twice in production, each time in a way that
produced a confident number rather than an error:

  1. No guard at all -> negative denominators. TreVeyon Henderson: 221
     receiving yards on -1.0 air yards = RACR -221.
  2. A floor on the denominator (50 air yards) -> still wrong for every RB
     that cleared it, and those were the highest-ADP backs on the board.
     Gibbs 11.41, Robinson 8.12, Cook 5.20, McCaffrey 2.76.

The lesson encoded here: it was never a sample-size problem. RACR is
undefined for a usage pattern built on targets at or behind the line of
scrimmage, however many of them there are — so the gate is position, not
volume.

Author: Zach Cooper
"""

import pytest

from backend.ingestion.fetch_metrics import (
    _MAX_PLAUSIBLE_RACR,
    _MIN_AIR_YARDS_FOR_RACR,
    _compute_opportunity_efficiency,
    _racr,
)


# ---------------------------------------------------------------------------
# Position gating
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("position", ["RB", "QB", "K", "DST"])
def test_non_receivers_never_get_a_racr(position):
    """Even with a healthy-looking denominator, RACR is meaningless off
    the WR/TE route tree."""
    assert _racr(sum_rec_yards=550.0, sum_air_yards=48.0, position=position) is None


@pytest.mark.parametrize("position", ["WR", "TE", "wr", "te"])
def test_receivers_get_a_racr(position):
    assert _racr(1000.0, 1200.0, position) == pytest.approx(0.8333, abs=1e-4)


def test_the_live_gibbs_case_is_suppressed():
    """The exact numbers that produced 11.41 in the database."""
    assert _racr(sum_rec_yards=548.0, sum_air_yards=48.0, position="RB") is None


def test_a_pass_catching_back_is_still_suppressed():
    """McCaffrey-shaped usage — huge receiving volume, still not a WR."""
    assert _racr(sum_rec_yards=900.0, sum_air_yards=326.0, position="RB") is None


# ---------------------------------------------------------------------------
# Denominator floor — kept from the previous fix
# ---------------------------------------------------------------------------

def test_air_yards_below_the_floor_are_rejected():
    assert _racr(200.0, _MIN_AIR_YARDS_FOR_RACR - 0.1, "WR") is None


def test_air_yards_exactly_at_the_floor_are_accepted():
    assert _racr(50.0, _MIN_AIR_YARDS_FOR_RACR, "WR") == pytest.approx(1.0)


def test_negative_air_yards_are_rejected():
    """The original bug — a negative denominator flips the sign of the stat."""
    assert _racr(221.0, -1.0, "WR") is None


def test_zero_air_yards_are_rejected():
    assert _racr(100.0, 0.0, "WR") is None


# ---------------------------------------------------------------------------
# Plausibility ceiling
# ---------------------------------------------------------------------------

def test_implausibly_high_racr_is_rejected_even_for_a_receiver():
    """Greg Dortch shape: 3.55 on a small denominator. A receiver cannot
    generate three times his air yards across a season — that's a broken
    denominator, not a talent signal."""
    assert _racr(355.0, 100.0, "WR") is None


def test_a_value_exactly_at_the_ceiling_is_kept():
    assert _racr(_MAX_PLAUSIBLE_RACR * 100, 100.0, "WR") == pytest.approx(_MAX_PLAUSIBLE_RACR)


def test_a_high_but_believable_racr_survives():
    """Deep-threat receivers legitimately run above 1.0 — the ceiling must
    not clip real football."""
    assert _racr(169.0, 100.0, "WR") == pytest.approx(1.69)


# ---------------------------------------------------------------------------
# Wiring — the value that actually reaches the database
# ---------------------------------------------------------------------------

def _week(rec_yards: float, air_yards: float) -> dict:
    return {
        "week": 1, "team": "DET", "targets": 5, "carries": 0,
        "receiving_yards": rec_yards, "receiving_air_yards": air_yards,
        "receptions": 4, "rushing_yards": 0, "receiving_yards_after_catch": 10,
    }


def test_position_reaches_the_computed_field():
    weeks = [_week(300.0, 30.0), _week(248.0, 18.0)]
    assert _compute_opportunity_efficiency(weeks, {}, "RB")["racr"] is None
    assert _compute_opportunity_efficiency(weeks, {}, "WR")["racr"] is None  # ceiling


def test_omitting_position_still_computes_for_backwards_compatibility():
    """position defaults to None — callers that don't pass it (ad-hoc
    scripts, older tests) keep the volume-only behaviour rather than
    silently losing the metric entirely."""
    weeks = [_week(60.0, 100.0)]
    assert _compute_opportunity_efficiency(weeks, {})["racr"] == pytest.approx(0.6)
