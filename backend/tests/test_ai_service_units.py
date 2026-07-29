"""
ai_service pure units — _compute_roster_gaps (FLEX-surplus logic) and
_parse_response (Claude output validation).

_compute_roster_gaps has a live regression pinned here: the AI never once
recommended a QB across a full draft because nothing computed the open-QB
gap (see the function's docstring in ai_service.py).
"""

import json

from backend.app.services.ai_service import (
    RecommendationContext,
    _compute_roster_gaps,
    _parse_response,
)


def roster(*positions: str) -> list[dict]:
    return [{"player_name": f"P{i}", "position": pos, "nfl_team": "X"}
            for i, pos in enumerate(positions)]


# ---------------------------------------------------------------------------
# _compute_roster_gaps
# ---------------------------------------------------------------------------

def test_empty_roster_has_every_gap_including_flex():
    gaps = _compute_roster_gaps([])
    assert gaps == {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1}


def test_full_lineup_no_gaps():
    gaps = _compute_roster_gaps(roster("QB", "RB", "RB", "WR", "WR", "TE", "RB", "DST"))
    assert gaps == {}  # third RB covers FLEX


def test_surplus_rb_satisfies_flex():
    gaps = _compute_roster_gaps(roster("RB", "RB", "RB"))
    assert "FLEX" not in gaps
    assert gaps["QB"] == 1 and gaps["WR"] == 2


def test_qb_surplus_never_satisfies_flex():
    gaps = _compute_roster_gaps(roster("QB", "QB"))
    assert gaps["FLEX"] == 1


def test_qb_gap_regression_never_disappears_until_filled():
    # The live bug: everything else filled, QB still open — the gap dict
    # must say so explicitly.
    gaps = _compute_roster_gaps(roster("RB", "RB", "WR", "WR", "TE", "WR", "DST"))
    assert gaps == {"QB": 1}


def test_custom_lineup_zero_slot_positions_are_omitted():
    lineup = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 2, "DST": 0}
    gaps = _compute_roster_gaps([], lineup)
    assert "DST" not in gaps
    assert gaps["WR"] == 3 and gaps["FLEX"] == 2


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

def ctx_with_available(*ids: int) -> RecommendationContext:
    return RecommendationContext(
        pick_number=1, round_number=1, my_slot=1, league_size=12,
        is_my_turn=True, picks_until_my_turn=0, my_next_pick_number=1,
        top_available=[
            {"id": i, "rank": i, "name": f"P{i}", "position": "RB",
             "team": "X", "adp": float(i), "sleeper_id": None}
            for i in ids
        ],
    )


def valid_payload(rec_id=1, alt_ids=(2, 3)) -> dict:
    return {
        "recommendation": {
            "player_id": rec_id, "player_name": "P", "position": "RB",
            "adp": 1.0, "reasoning": "best available",
        },
        "alternatives": [
            {"player_id": i, "player_name": f"A{i}", "position": "RB",
             "adp": float(i), "reasoning": ""}
            for i in alt_ids
        ],
        "alerts": ["tier break at RB"],
    }


def test_valid_json_parses():
    result = _parse_response(json.dumps(valid_payload()), ctx_with_available(1, 2, 3))
    assert result is not None
    assert result.recommendation.player_id == 1
    assert [a.player_id for a in result.alternatives] == [2, 3]
    assert result.alerts == ["tier break at RB"]


def test_markdown_fenced_json_parses():
    raw = "```json\n" + json.dumps(valid_payload()) + "\n```"
    assert _parse_response(raw, ctx_with_available(1, 2, 3)) is not None


def test_garbage_returns_none():
    assert _parse_response("I think you should draft Gibbs!", ctx_with_available(1)) is None


def test_missing_recommendation_returns_none():
    assert _parse_response(json.dumps({"alternatives": []}), ctx_with_available(1)) is None


def test_unavailable_player_returns_none():
    # Claude recommending someone who's already drafted must fall back
    result = _parse_response(json.dumps(valid_payload(rec_id=99)), ctx_with_available(1, 2, 3))
    assert result is None


def test_string_player_id_is_coerced_not_rejected():
    payload = valid_payload()
    payload["recommendation"]["player_id"] = "1"  # Claude sometimes stringifies
    result = _parse_response(json.dumps(payload), ctx_with_available(1, 2, 3))
    assert result is not None
    assert result.recommendation.player_id == 1


def test_unavailable_alternatives_are_dropped_and_capped_at_three():
    payload = valid_payload(alt_ids=(2, 99, 3, 4))  # 99 unavailable, 4 alts total
    result = _parse_response(json.dumps(payload), ctx_with_available(1, 2, 3, 4))
    # Capped to first 3 entries, then filtered: [2, 99, 3] -> [2, 3]
    assert [a.player_id for a in result.alternatives] == [2, 3]


def test_malformed_alternative_entries_are_skipped():
    payload = valid_payload(alt_ids=(2,))
    payload["alternatives"].append("not a dict")
    payload["alternatives"].append({"player_id": "not-an-int"})
    result = _parse_response(json.dumps(payload), ctx_with_available(1, 2))
    assert [a.player_id for a in result.alternatives] == [2]
