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
    _build_prompt,
    _MAX_ALTERNATIVES,
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


# ---------------------------------------------------------------------------
# _parse_response — strategy / confidence / tradeoff
#
# These three fields are presentational extras. The rule they all share: a
# missing or malformed value must degrade to a default, never reject an
# otherwise-valid recommendation to the ADP fallback.
# ---------------------------------------------------------------------------

def test_strategy_and_confidence_are_parsed():
    payload = valid_payload()
    payload["strategy"] = "You're thin at RB with two picks before the tier breaks."
    payload["confidence"] = "high"
    result = _parse_response(json.dumps(payload), ctx_with_available(1, 2, 3))
    assert result.strategy.startswith("You're thin at RB")
    assert result.confidence == "high"


def test_alternative_tradeoff_is_parsed():
    payload = valid_payload(alt_ids=(2,))
    payload["alternatives"][0]["tradeoff"] = "Safer floor, less weekly ceiling."
    result = _parse_response(json.dumps(payload), ctx_with_available(1, 2))
    assert result.alternatives[0].tradeoff == "Safer floor, less weekly ceiling."


def test_missing_extras_fall_back_to_defaults():
    # valid_payload() has no strategy/confidence/tradeoff at all — the shape
    # Claude returned before these fields existed.
    result = _parse_response(json.dumps(valid_payload()), ctx_with_available(1, 2, 3))
    assert result is not None
    assert result.strategy == ""
    assert result.confidence == "medium"
    assert all(a.tradeoff == "" for a in result.alternatives)


def test_unrecognised_confidence_normalises_to_medium():
    for bogus in ["very high", "", None, 7, "CERTAIN"]:
        payload = valid_payload()
        payload["confidence"] = bogus
        result = _parse_response(json.dumps(payload), ctx_with_available(1, 2, 3))
        assert result is not None, f"{bogus!r} should not reject the response"
        assert result.confidence == "medium"


def test_confidence_is_case_and_whitespace_insensitive():
    payload = valid_payload()
    payload["confidence"] = "  LOW "
    result = _parse_response(json.dumps(payload), ctx_with_available(1, 2, 3))
    assert result.confidence == "low"


def test_non_string_strategy_is_dropped_not_stringified():
    # str() on a dict would put "{'unexpected': 'object'}" in front of the user.
    payload = valid_payload()
    payload["strategy"] = {"unexpected": "object"}
    result = _parse_response(json.dumps(payload), ctx_with_available(1, 2, 3))
    assert result is not None
    assert result.strategy == ""


def test_non_string_tradeoff_is_dropped_not_stringified():
    payload = valid_payload(alt_ids=(2,))
    payload["alternatives"][0]["tradeoff"] = ["a", "list"]
    result = _parse_response(json.dumps(payload), ctx_with_available(1, 2))
    assert result.alternatives[0].tradeoff == ""


# ---------------------------------------------------------------------------
# Response-size guards
#
# The recommendation JSON is capped by max_tokens. A response that runs over is
# truncated, which makes it invalid JSON, which silently degrades every pick to
# the ADP fallback — the failure this suite exists to catch early.
# ---------------------------------------------------------------------------

def test_alternatives_cap_matches_the_documented_constant():
    # The prompt tells Claude "at most _MAX_ALTERNATIVES"; the parser enforces
    # the same number. If these drift, the model spends output budget on
    # entries that are discarded unread.
    payload = valid_payload(alt_ids=tuple(range(2, 12)))
    result = _parse_response(json.dumps(payload), ctx_with_available(*range(1, 12)))
    assert len(result.alternatives) <= _MAX_ALTERNATIVES


def test_prompt_states_the_alternatives_cap():
    ctx = ctx_with_available(1, 2, 3)
    prompt = _build_prompt(ctx)
    assert str(_MAX_ALTERNATIVES) in prompt
    assert "alternatives" in prompt


def test_truncated_json_returns_none_rather_than_raising():
    # What a max_tokens cut-off actually looks like: valid opening, no close.
    truncated = json.dumps(valid_payload())[:120]
    assert _parse_response(truncated, ctx_with_available(1, 2, 3)) is None
