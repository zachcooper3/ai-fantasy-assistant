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
    _build_system_prompt,
    _format_positional_dropoff,
    _format_run_risk,
    _restore_prefill,
    _survival,
    _SURVIVAL_GONE,
    _SURVIVAL_SAFE,
    _SURVIVAL_TOSSUP,
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


# ---------------------------------------------------------------------------
# _parse_response — the id is the only trusted field
#
# Live bug 2026-07-30: the parser checked player_id against the available list
# but rendered Claude's own player_name/position/adp, so an available id paired
# with a drafted player's name displayed that drafted player. Every factual
# field now comes from ctx.top_available; the model supplies only free text.
# ---------------------------------------------------------------------------

def test_player_name_comes_from_our_data_not_claudes():
    payload = valid_payload()
    payload["recommendation"]["player_name"] = "Some Drafted Guy"
    result = _parse_response(json.dumps(payload), ctx_with_available(1, 2, 3))
    # ctx_with_available names players P{id}
    assert result.recommendation.player_name == "P1"


def test_position_and_adp_come_from_our_data_not_claudes():
    payload = valid_payload()
    payload["recommendation"]["position"] = "QB"   # actually RB in our data
    payload["recommendation"]["adp"] = 999.0       # actually 1.0
    result = _parse_response(json.dumps(payload), ctx_with_available(1, 2, 3))
    assert result.recommendation.position == "RB"
    assert result.recommendation.adp == 1.0


def test_alternative_fields_are_also_taken_from_our_data():
    payload = valid_payload(alt_ids=(2,))
    payload["alternatives"][0]["player_name"] = "Wrong Name"
    payload["alternatives"][0]["adp"] = 0.1
    result = _parse_response(json.dumps(payload), ctx_with_available(1, 2))
    assert result.alternatives[0].player_name == "P2"
    assert result.alternatives[0].adp == 2.0


def test_reasoning_and_tradeoff_still_come_from_claude():
    # The model's judgement is the one thing it does supply.
    payload = valid_payload(alt_ids=(2,))
    payload["recommendation"]["reasoning"] = "Elite volume in a good offence."
    payload["alternatives"][0]["tradeoff"] = "Higher floor, lower ceiling."
    result = _parse_response(json.dumps(payload), ctx_with_available(1, 2))
    assert result.recommendation.reasoning == "Elite volume in a good offence."
    assert result.alternatives[0].tradeoff == "Higher floor, lower ceiling."


def test_suggestion_missing_name_and_adp_entirely_still_parses():
    # Those fields are ours now, so Claude omitting them is harmless.
    payload = valid_payload(alt_ids=(2,))
    for key in ("player_name", "position", "adp"):
        payload["recommendation"].pop(key)
        payload["alternatives"][0].pop(key)
    result = _parse_response(json.dumps(payload), ctx_with_available(1, 2))
    assert result is not None
    assert result.recommendation.player_name == "P1"
    assert result.alternatives[0].player_name == "P2"


def test_alternatives_outside_the_available_list_are_dropped_not_kept():
    # The silent-shrink case: Claude names a real player who isn't in
    # top_available, so _pick rejects them and 3 alternatives become 2.
    payload = valid_payload(alt_ids=(2, 999, 3))
    result = _parse_response(json.dumps(payload), ctx_with_available(1, 2, 3))
    assert [a.player_id for a in result.alternatives] == [2, 3]


def test_prompt_asks_for_ids_from_the_listed_players_in_alternatives_too():
    # Only the recommendation carried this constraint originally, which let
    # Claude spend tokens on alternatives that were then discarded unread.
    prompt = _build_prompt(ctx_with_available(1, 2, 3))
    # The JSON template block should constrain both id fields identically.
    assert prompt.count('"<int from the tiers above>"') >= 2


# ---------------------------------------------------------------------------
# Opportunity cost — survival, cost of waiting, run risk
#
# These pin the behaviors the sections were added to produce, not their
# wording. Each one corresponds to a way the recommendation was wrong before:
# spending a pick on a player who'd still be there, ignoring a positional
# cliff, taking a DST in the middle rounds, and reporting demand percentages
# above 100%.
# ---------------------------------------------------------------------------

def _p(pid: int, name: str, pos: str, adp: float) -> dict:
    return {"id": pid, "rank": pid, "name": name, "position": pos,
            "team": "X", "adp": adp, "sleeper_id": None}


def test_survival_buckets_split_on_the_horizon_pick():
    # ADP 20 with 25 picks of slack is gone; ADP 90 is not close.
    assert _survival(20.0, 60) == _SURVIVAL_GONE
    assert _survival(90.0, 60) == _SURVIVAL_SAFE
    # Inside the noise band either way — deliberately not called.
    assert _survival(60.0, 60) == _SURVIVAL_TOSSUP


def test_survival_noise_band_widens_with_adp():
    # 10 picks of slack is decisive at ADP 12 (band = the 8.0 floor) but
    # noise at ADP 200 (band = 50), which is the whole point of the ratio.
    assert _survival(12.0, 22) == _SURVIVAL_GONE
    assert _survival(200.0, 210) == _SURVIVAL_TOSSUP


def test_survival_without_a_horizon_is_unanswerable_not_guessed():
    # My last pick of the draft: there is no "next turn" to wait for.
    assert _survival(50.0, None) is None


def test_cost_of_waiting_finds_replacement_level_past_the_displayed_board():
    # The regression this exists for: the next TE is deep down the ADP list,
    # so a 25-player global slice would hide the cliff entirely.
    board = [_p(1, "Cliff TE", "TE", 40.0)] + [
        _p(i, f"WR{i}", "WR", 41.0 + i) for i in range(2, 30)
    ] + [_p(99, "Deep TE", "TE", 130.0)]
    out = _format_positional_dropoff(board, horizon_pick=70)
    assert "Deep TE" in out
    assert "Cost of waiting: 90 ADP points" in out


def test_cost_of_waiting_reports_no_cost_when_the_best_player_survives():
    board = [_p(1, "Safe QB", "QB", 120.0)]
    out = _format_positional_dropoff(board, horizon_pick=70)
    assert "No cost to waiting" in out


def test_run_risk_never_exceeds_the_number_of_teams():
    # The live bug: teams appear twice across a snake turn and FLEX was added
    # to both RB and WR, producing "WR: 38" out of 19 teams.
    lineup = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1}
    slots = [4, 5, 6, 6, 5, 4]            # each team picks twice
    counts = {s: {"RB": 1} for s in (4, 5, 6)}
    out = _format_run_risk(slots, counts, lineup)
    assert "3 team(s)" in out
    for token in out.split():
        if "/" in token and token.split("/")[0].isdigit():
            assert int(token.split("/")[0]) <= 3


def test_run_risk_excludes_dst_and_k_demand():
    lineup = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1}
    counts = {4: {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}}
    out = _format_run_risk([4], counts, lineup)
    assert "DST" not in out and "K " not in out


def test_dst_and_k_are_deferred_out_of_the_urgency_math():
    # Round 13 of 15 needing QB + DST + K: only ONE round is genuinely
    # available for the QB, because the last two are owed to DST and K.
    # The old math saw 2 rounds free and stayed quiet.
    ctx = ctx_with_available(1, 2, 3)
    ctx.round_number = 13
    ctx.total_rounds = 15
    ctx.my_roster = roster("RB", "RB", "WR", "WR", "TE", "RB")
    prompt = _build_prompt(ctx)
    assert "URGENT" in prompt
    assert "1 usable round(s) left" in prompt
    assert "do NOT recommend one until the final" in prompt


def test_dst_and_k_are_demanded_once_the_reserved_rounds_arrive():
    ctx = ctx_with_available(1, 2, 3)
    ctx.round_number = 14
    ctx.total_rounds = 15
    ctx.my_roster = roster("QB", "RB", "RB", "WR", "WR", "TE", "RB")
    prompt = _build_prompt(ctx)
    assert "out of rounds to defer them" in prompt


def test_urgent_still_fires_on_the_final_round():
    # Off-by-one regression: rounds_remaining excluded the current round, so
    # on round 15 of 15 it was 0 and the `> 0` guard suppressed URGENT at the
    # least recoverable moment in the draft.
    ctx = ctx_with_available(1, 2, 3)
    ctx.round_number = 15
    ctx.total_rounds = 15
    ctx.my_roster = roster("RB", "RB", "WR", "WR", "TE", "RB", "DST", "K")
    assert "URGENT" in _build_prompt(ctx)


def test_prefill_brace_is_restored_only_when_missing():
    assert _restore_prefill('"a": 1}') == '{"a": 1}'
    # A model that emits its own brace must not get it doubled — that turned
    # a good response into unparseable JSON and lost it to the ADP fallback.
    assert _restore_prefill('{"a": 1}') == '{"a": 1}'
    assert _restore_prefill('  {"a": 1}') == '  {"a": 1}'


def test_system_prompt_tracks_the_session_scoring_format():
    # It used to hardcode PPR regardless, so a standard-scoring league got
    # confidently PPR-shaped advice.
    assert "No points for receptions" in _build_system_prompt("standard")
    assert "Every reception is a point" in _build_system_prompt("ppr")
    assert "Half a point" in _build_system_prompt("half_ppr")
    # Unknown formats fall back rather than rendering an empty scoring note.
    assert "SCORING:" in _build_system_prompt("weird_custom")
