"""
ai_service pure units — _compute_roster_gaps (FLEX-surplus logic) and
_parse_response (Claude output validation).

_compute_roster_gaps has a live regression pinned here: the AI never once
recommended a QB across a full draft because nothing computed the open-QB
gap (see the function's docstring in ai_service.py).
"""

import json

import pytest

from backend.app.services.ai_service import (
    RecommendationContext,
    _compute_roster_gaps,
    _parse_response,
    _build_prompt,
    _build_system_prompt,
    _experience_context,
    _format_metrics_section,
    _infer_current_season,
    _compute_roster_depth,
    _compute_adp_tiers,
    compute_replacement_levels,
    _vor,
    _format_roster_depth_section,
    _is_trustworthy,
    _format_metrics_line,
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
    # available_counts must contain kickers: the K requirement is dropped
    # entirely when none exist to draft (this app's real ADP feed has none),
    # and this test is about the deferral math, not that data gap.
    # Round 13 of 15 needing QB + DST + K: only ONE round is genuinely
    # available for the QB, because the last two are owed to DST and K.
    # The old math saw 2 rounds free and stayed quiet.
    ctx = ctx_with_available(1, 2, 3)
    ctx.available_counts = {"K": 12, "DST": 12}
    ctx.round_number = 13
    ctx.total_rounds = 15
    ctx.my_roster = roster("RB", "RB", "WR", "WR", "TE", "RB")
    prompt = _build_prompt(ctx)
    assert "URGENT" in prompt
    assert "1 usable round(s) left" in prompt
    assert "do NOT recommend one until the final" in prompt


def test_dst_and_k_are_demanded_once_the_reserved_rounds_arrive():
    ctx = ctx_with_available(1, 2, 3)
    ctx.available_counts = {"K": 12, "DST": 12}
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


# ---------------------------------------------------------------------------
# Draft capital + experience
#
# Live miss (2026-08-04): a 2nd-year WR with 1st-round capital and the better
# ADP was repeatedly passed over for a much older veteran whose prior-season
# per-game average was higher. Every fact needed was in the DB; none of it
# reached the prompt. Two independent causes, both pinned below.
# ---------------------------------------------------------------------------

def _metrics(season=2025, ppg=11.5, games=17):
    return {"season": season, "games_played": games, "fantasy_points_avg": ppg,
            "targets_per_game": 7.5}


def _profile(year=2025, rnd=1, pick=19):
    return {"draft_year": year, "draft_round": rnd, "draft_pick": pick}


def test_current_season_inferred_from_metrics_not_the_clock():
    # Prior completed season + 1. Deriving from the system clock would age
    # every player by a year each January without a data refresh.
    assert _infer_current_season({1: _metrics(season=2025)}, {}) == 2026


def test_current_season_inferred_from_newest_draft_class():
    assert _infer_current_season({}, {1: _profile(year=2026)}) == 2026


def test_current_season_takes_the_later_of_the_two_signals():
    assert _infer_current_season({1: _metrics(season=2024)}, {1: _profile(year=2026)}) == 2026


def test_current_season_unknown_rather_than_guessed():
    # No basis to infer — experience must not be rendered at all.
    assert _infer_current_season({}, {}) is None
    assert _experience_context(_profile(), None) == ""


def test_experience_context_labels_rookie_and_second_year():
    assert _experience_context(_profile(year=2026), 2026) == \
        "rookie, 1st-round pick (#19 overall)"
    assert _experience_context(_profile(year=2025), 2026) == \
        "2nd NFL season, 1st-round pick (#19 overall)"
    assert _experience_context(_profile(year=2024), 2026).startswith("3rd NFL season")


def test_experience_context_suppressed_for_established_veterans():
    # Where a 6-year vet was drafted stops being predictive next to what he
    # has actually done. Real data has no such profiles yet, so this is the
    # only place the limit is exercised.
    assert _experience_context(_profile(year=2020), 2026) == ""


def test_experience_context_handles_undrafted_and_partial_rows():
    assert _experience_context(None, 2026) == ""
    assert _experience_context({"draft_year": None}, 2026) == ""
    # Round known, exact pick missing — still worth saying.
    assert "2nd-round pick" in _experience_context(
        {"draft_year": 2025, "draft_round": 2, "draft_pick": None}, 2026)
    assert "overall" not in _experience_context(
        {"draft_year": 2025, "draft_round": 2, "draft_pick": None}, 2026)


def test_draft_capital_now_rides_along_with_the_stat_line():
    # THE regression: draft capital used to render only as a fallback for
    # players with NO metrics, so the moment a rookie completed a season his
    # pedigree vanished and he read as an anonymous veteran.
    player = [{"id": 1, "name": "Young WR", "position": "WR", "team": "TB",
               "adp": 35.6, "rank": 37, "sleeper_id": "x"}]
    out = _format_metrics_section(player, {1: _metrics()}, {1: _profile()})
    assert "2nd NFL season" in out
    assert "1st-round pick (#19 overall)" in out
    assert "11.5 PPR pts/gm" in out  # stats still there, not replaced


def test_veteran_with_metrics_gets_no_experience_tag():
    player = [{"id": 1, "name": "Old WR", "position": "WR", "team": "LAR",
               "adp": 41.2, "rank": 42, "sleeper_id": "y"}]
    out = _format_metrics_section(player, {1: _metrics(ppg=15.9, games=14)},
                                  {1: _profile(year=2014, rnd=2, pick=53)})
    assert "NFL season" not in out
    assert "15.9 PPR pts/gm" in out


def test_player_without_metrics_still_gets_the_full_draft_profile_fallback():
    # The pre-existing rookie path must keep working — college production and
    # all — now that the tag exists alongside it.
    player = [{"id": 1, "name": "True Rookie", "position": "RB", "team": "DEN",
               "adp": 90.0, "rank": 88, "sleeper_id": "z"}]
    out = _format_metrics_section(player, {}, {1: {**_profile(year=2026),
                                                  "college": "Ohio St.",
                                                  "receiving_yards": 1011}})
    assert "No NFL performance data yet" in out
    assert "Ohio St." in out


def test_dead_rookie_flag_column_is_no_longer_read():
    # is_rookie_or_second_year is written by no ingestion script (0/182 rows).
    # Even if it were set, experience must come from draft_year alone so the
    # fact has exactly one source of truth.
    player = [{"id": 1, "name": "Someone", "position": "WR", "team": "TB",
               "adp": 50.0, "rank": 50, "sleeper_id": "q"}]
    m = {**_metrics(), "is_rookie_or_second_year": True}
    out = _format_metrics_section(player, {1: m}, {})
    assert "rookie/2nd-year" not in out


def test_system_prompt_tells_the_model_what_the_tag_means():
    sp = _build_system_prompt("ppr")
    assert "ASCENDING" in sp
    assert "floor" in sp


# ---------------------------------------------------------------------------
# Rate-stat trust gates
#
# Live finding (2026-08-04): 50 of 57 RBs carried a nonsense RACR because
# their season air-yard total is negative (screens/checkdowns are targeted
# behind the line of scrimmage), plus a long tail of rates computed over
# denominators of 1-2 attempts. All of it reached the prompt as real
# efficiency signal. Guarded in ingestion AND at render — the DB on disk
# still holds the bad values.
# ---------------------------------------------------------------------------

def _rb(**over):
    m = {"season": 2025, "games_played": 16, "targets_per_game": 3.0,
         "carries_per_game": 13.9, "fantasy_points_avg": 14.3}
    m.update(over)
    return m


def test_negative_racr_is_suppressed():
    # D'Andre Swift: 299 receiving yards on -25 air yards.
    assert not _is_trustworthy("racr", -11.96, _rb())
    assert "RACR" not in (_format_metrics_line(_rb(racr=-11.96)) or "")


def test_wildly_inflated_racr_is_suppressed():
    # Tiny positive denominator: 86 receiving yards on 1.0 air yards.
    assert not _is_trustworthy("racr", 86.0, _rb())


def test_legitimate_receiver_racr_survives():
    wr = {"season": 2025, "games_played": 14, "targets_per_game": 5.6}
    assert _is_trustworthy("racr", 0.56, wr)
    assert "RACR 0.56" in _format_metrics_line({**wr, "racr": 0.56})


def test_rate_over_a_trivial_denominator_is_suppressed():
    # Mahomes: "catch rate 100%" on roughly one target all season.
    qb = {"season": 2025, "games_played": 14, "targets_per_game": 0.1,
          "carries_per_game": 4.6}
    assert not _is_trustworthy("catch_rate", 1.0, qb)
    assert "catch rate" not in (_format_metrics_line({**qb, "catch_rate": 1.0}) or "")


def test_rushing_rate_suppressed_for_a_receiver_with_two_carries():
    # Jordan Addison: "Y/carry 40.5" on ~2 end-arounds.
    wr = {"season": 2025, "games_played": 14, "targets_per_game": 5.6,
          "carries_per_game": 0.1}
    assert not _is_trustworthy("yards_per_carry", 40.5, wr)


def test_rushing_rate_survives_for_an_actual_running_back():
    assert _is_trustworthy("yards_per_carry", 4.9, _rb())
    assert "Y/carry 4.9" in _format_metrics_line(_rb(yards_per_carry=4.9))


def test_volume_stats_are_never_gated_by_the_rate_thresholds():
    # The gate applies to per-attempt RATES. The attempt counts themselves
    # are the evidence you most need when volume is low.
    low = {"season": 2025, "games_played": 16, "targets_per_game": 0.1,
           "carries_per_game": 0.1}
    assert _is_trustworthy("targets_per_game", 0.1, low)
    assert _is_trustworthy("carries_per_game", 0.1, low)
    assert _is_trustworthy("fantasy_points_avg", 5.0, low)


def test_impossible_percentages_are_rejected_regardless_of_volume():
    heavy = {"season": 2025, "games_played": 17, "targets_per_game": 9.0}
    assert not _is_trustworthy("catch_rate", 1.4, heavy)
    assert not _is_trustworthy("snap_pct", -0.2, heavy)


def test_missing_games_played_does_not_suppress_everything():
    # Without games_played there's no denominator to reconstruct; fall back
    # to sanity bounds only rather than hiding the whole line.
    m = {"season": 2025, "targets_per_game": 5.0}
    assert _is_trustworthy("catch_rate", 0.62, m)
    assert not _is_trustworthy("racr", -3.0, m)


# ---------------------------------------------------------------------------
# Roster construction / depth
#
# Live failure (2026-08-04 draft review): a 15-round draft opened RB-RB, and
# from round 3 the gap logic reported RB satisfied (2 held >= 2 required) and
# never mentioned it again; from round 8 the section read "All required
# starting slots filled" for eight straight picks. Final roster: RB4/WR7.
# Legality was met the whole time — that was never the question.
# ---------------------------------------------------------------------------

_PPR_LINEUP = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1}


def test_holding_only_your_mandatory_starters_reads_as_exposed():
    # THE regression: 2 RB meets the base requirement, so the gap logic went
    # silent on RB from round 3 to round 11 of a real draft. `beyond == 0`
    # is the state it couldn't express — every back you own is a starter.
    depth = _compute_roster_depth(roster("RB", "RB"), _PPR_LINEUP)
    assert depth["RB"]["base_met"] is True
    assert depth["RB"]["required"] == 2
    assert depth["RB"]["beyond"] == 0


def test_beyond_is_measured_against_base_not_base_plus_flex():
    # The FLEX is a use for a spare player, not a second requirement.
    # Counting it as one credited the same slot to both RB and WR, making
    # the startable counts sum to 8 against a 7-slot lineup.
    depth = _compute_roster_depth(roster("RB", "RB", "WR", "WR", "TE", "QB"),
                                  _PPR_LINEUP)
    assert depth["RB"]["required"] == 2 and depth["WR"]["required"] == 2
    assert depth["TE"]["required"] == 1 and depth["QB"]["required"] == 1
    assert sum(d["required"] for d in depth.values()) == 6  # + FLEX = 7 slots


def test_balanced_roster_raises_nothing():
    # 3RB/3WR has one spare at each; whichever fills the FLEX, neither side
    # is exposed. An earlier version allocated the FLEX by surplus and made
    # whichever position lost an arbitrary tie-break look short.
    out = _format_roster_depth_section(
        _compute_roster_depth(roster("QB", "TE", *(["RB"] * 3), *(["WR"] * 3)),
                              _PPR_LINEUP), spots_left=8)
    assert "Imbalance" not in out


def test_roster_with_no_spare_anywhere_raises_nothing():
    # Legal, nothing spare, nothing to rebalance — there is no action this
    # could recommend, so it must stay quiet rather than add a fixed line.
    out = _format_roster_depth_section(
        _compute_roster_depth(roster("QB", "RB", "RB", "WR", "WR", "TE"),
                              _PPR_LINEUP), spots_left=9)
    assert "Imbalance" not in out


def test_single_slot_positions_never_count_as_exposed():
    # Holding one QB and one TE is the normal state of nearly every roster
    # in a 1-QB league. Flagging them made the callout fire on every roster
    # shape tested — a constant with no information in it.
    out = _format_roster_depth_section(
        _compute_roster_depth(roster("QB", "TE", "RB", "RB", *(["WR"] * 5)),
                              _PPR_LINEUP), spots_left=6)
    assert "Imbalance" in out
    assert "none at RB" in out
    assert "QB" not in out.split("Imbalance")[1]
    assert "TE" not in out.split("Imbalance")[1]


def test_exposure_generalises_to_a_superflex_league():
    # Keyed off required >= 2, not a hardcoded {RB, WR}.
    superflex = {"QB": 2, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1}
    out = _format_roster_depth_section(
        _compute_roster_depth(
            roster("QB", "QB", "TE", *(["RB"] * 4), "WR", "WR"), superflex),
        spots_left=6)
    assert "none at QB" in out


def test_imbalance_fires_on_the_real_draft_shape():
    # Round 9 of the reviewed draft: two backs, four receivers.
    out = _format_roster_depth_section(
        _compute_roster_depth(roster("QB", "TE", "RB", "RB", *(["WR"] * 4)),
                              _PPR_LINEUP), spots_left=7)
    assert "Imbalance" in out
    assert "spare at WR" in out and "none at RB" in out


def test_imbalance_is_symmetric_not_rb_biased():
    # The mirror image must read identically — this encodes no opinion that
    # RB is special, only that a multi-slot position with no spare while
    # another has two is worth naming.
    out = _format_roster_depth_section(
        _compute_roster_depth(roster("QB", "TE", *(["RB"] * 5), "WR", "WR"),
                              _PPR_LINEUP), spots_left=6)
    assert "spare at RB" in out and "none at WR" in out


def test_unfilled_starting_slots_suppress_the_depth_callout():
    # An unfilled WR2 is unambiguously more urgent than RB depth and is
    # already reported by the gap section; running both at once is how the
    # first version flagged all four positions in round 3.
    out = _format_roster_depth_section(
        _compute_roster_depth(roster("RB", "RB", "RB", "RB", "WR"), _PPR_LINEUP),
        spots_left=10)
    assert "Imbalance" not in out
    assert "starting slot still unfilled" in out


def test_depth_section_appears_in_the_built_prompt():
    ctx = ctx_with_available(1, 2, 3)
    ctx.my_roster = roster("RB", "RB", "WR", "WR", "TE", "WR", "WR", "QB")
    prompt = _build_prompt(ctx)
    assert "Roster Construction" in prompt
    assert "Imbalance" in prompt and "none at RB" in prompt
    # The old dead-end wording must be gone — it ended the section on a shrug
    # for the entire back half of the draft.
    assert "every remaining pick is depth or upside" not in prompt


def test_cost_of_waiting_prefers_points_over_adp_when_known():
    board = [{"id": 1, "rank": 1, "name": "Now RB", "position": "RB",
              "team": "X", "adp": 30.0, "sleeper_id": None},
             {"id": 2, "rank": 2, "name": "Later RB", "position": "RB",
              "team": "X", "adp": 95.0, "sleeper_id": None}]
    metrics = {1: {"fantasy_points_avg": 15.2}, 2: {"fantasy_points_avg": 11.1}}
    out = _format_positional_dropoff(board, horizon_pick=70, player_metrics=metrics)
    assert "PPR ppg" in out and "15.2 -> 11.1" in out


def test_cost_of_waiting_falls_back_to_adp_without_metrics():
    board = [{"id": 1, "rank": 1, "name": "Now RB", "position": "RB",
              "team": "X", "adp": 30.0, "sleeper_id": None},
             {"id": 2, "rank": 2, "name": "Later RB", "position": "RB",
              "team": "X", "adp": 95.0, "sleeper_id": None}]
    out = _format_positional_dropoff(board, horizon_pick=70, player_metrics={})
    assert "ADP points" in out and "PPR ppg" not in out


def test_system_prompt_forbids_calling_a_base_filled_position_secure():
    sp = _build_system_prompt("ppr")
    assert "A LEGAL LINEUP IS NOT A COMPLETE ROSTER" in sp
    assert "zero cover" in sp


# ---------------------------------------------------------------------------
# Value over replacement
#
# Live failure: five receivers in seven rounds. Root cause was that nothing
# in the prompt compared positions on one scale — an ADP-sorted board is
# WR-dense at every depth (12 WR vs 11 RB in the top 24; 20 vs 10 by ADP
# 97-144) and raw ppg is near-identical between them (RB12 15.2, WR12 14.7).
# What differs is replacement level, which the prompt never stated.
# ---------------------------------------------------------------------------

_POOL = {
    # 40 backs descending from 24.0, 40 receivers descending from 23.0.
    "RB": [24.0 - 0.4 * i for i in range(40)],
    "WR": [23.0 - 0.3 * i for i in range(40)],
    "TE": [18.0 - 0.6 * i for i in range(20)],
    "QB": [23.0 - 0.5 * i for i in range(20)],
}


def test_replacement_level_is_the_last_startable_player_league_wide():
    levels = compute_replacement_levels(_POOL, league_size=12,
                                        starter_slots={"RB": 2.5, "WR": 2.5})
    # 12 teams x 2.5 startable = rank 30.
    assert levels["RB"] == pytest.approx(24.0 - 0.4 * 29)
    assert levels["WR"] == pytest.approx(23.0 - 0.3 * 29)


def test_deeper_position_has_the_higher_replacement_level():
    # This is the whole point: receivers are deeper, so each one is worth
    # less over the man who'd replace him even at identical raw points.
    levels = compute_replacement_levels(_POOL, 12, {"RB": 2.5, "WR": 2.5})
    assert levels["WR"] > levels["RB"]


def test_equal_points_at_different_positions_are_not_equal_value():
    levels = compute_replacement_levels(_POOL, 12, {"RB": 2.5, "WR": 2.5})
    metrics = {1: {"fantasy_points_avg": 15.0}, 2: {"fantasy_points_avg": 15.0}}
    rb = {"id": 1, "position": "RB"}
    wr = {"id": 2, "position": "WR"}
    assert _vor(rb, metrics, levels) > _vor(wr, metrics, levels)


def test_shallow_position_pool_falls_back_instead_of_raising():
    # A league that needs more starters than the DB has players on file is a
    # data gap; draft day is not the time to crash over one.
    levels = compute_replacement_levels({"TE": [12.0, 10.0]}, 12, {"TE": 1})
    assert levels["TE"] == 10.0


def test_positions_absent_from_the_pool_are_omitted_not_zeroed():
    levels = compute_replacement_levels({"RB": [10.0]}, 12, {"RB": 2, "WR": 2})
    assert "WR" not in levels


def test_vor_is_none_for_a_player_with_no_prior_season():
    # A rookie's VOR is unknown. Rendering it as 0.0 would assert he is
    # exactly replacement-level, which the data does not support.
    assert _vor({"id": 99, "position": "RB"}, {}, {"RB": 11.1}) is None


def test_vor_is_none_when_the_position_has_no_replacement_level():
    metrics = {1: {"fantasy_points_avg": 15.0}}
    assert _vor({"id": 1, "position": "DST"}, metrics, {"RB": 11.1}) is None


def test_board_renders_vor_and_marks_unknowns():
    ctx = ctx_with_available(1, 2)
    ctx.top_available[1]["position"] = "WR"
    ctx.replacement_ppg = {"RB": 11.1, "WR": 11.9}
    ctx.player_metrics = {1: {"fantasy_points_avg": 15.2}}   # id 2 has none
    prompt = _build_prompt(ctx)
    assert "VOR +4.1" in prompt
    assert "VOR   --" in prompt
    assert "unknown value, NOT replacement-level value" in prompt


def test_board_omits_vor_entirely_without_replacement_levels():
    # Older callers and tests that never set replacement_ppg must render the
    # board exactly as before rather than showing a column of dashes.
    ctx = ctx_with_available(1, 2)
    prompt = _build_prompt(ctx)
    assert "VOR" not in prompt


def test_system_prompt_guards_against_stacking_on_high_vor():
    # The failure mode a draft simulation surfaced: VOR-greedy drafting takes
    # six tight ends, because VOR measures starter value and says nothing
    # about a position you can only start one of.
    sp = _build_system_prompt("ppr")
    assert "COMPARE POSITIONS BY VOR" in sp
    assert "stacking a fourth player at a position that starts one" in sp
    # And the honest caveat about where the number comes from.
    assert "computed from last season" in sp


# ---------------------------------------------------------------------------
# ADP tiers
#
# Live failure: the board rendered as ONE tier of 25 players spanning 25 ADP
# points, and the board's own instruction says same-tier players are
# interchangeable and within-tier gaps are noise. So the prompt was actively
# telling the model to discard the ADP evidence separating a back at 45.2
# from a receiver at 71.7. Cause: the threshold was max(3.0, adp * 0.10)
# while consecutive available players sit ~1 ADP point apart, so it could
# never be exceeded.
# ---------------------------------------------------------------------------

def _board(*adps: float) -> list[dict]:
    return [{"id": i, "rank": i, "name": f"P{i}", "position": "RB",
             "team": "X", "adp": a, "sleeper_id": None}
            for i, a in enumerate(adps, start=1)]


def test_dense_board_still_produces_multiple_tiers():
    # 25 players ~1 ADP apart with a few 2+ gaps: the exact shape that
    # collapsed into a single tier under the old absolute threshold.
    adps = []
    x = 1.0
    for i in range(25):
        adps.append(x)
        x += 2.4 if i % 6 == 5 else 0.9
    tiers = _compute_adp_tiers(_board(*adps))
    assert len(tiers) > 1, "a board with real gaps must not render as one blob"
    assert all(len(t) <= 8 for t in tiers)


def test_threshold_scales_with_the_board_not_with_adp_value():
    # Same relative shape, shifted deep into the draft. The old ratio term
    # made the threshold grow with ADP (>10 by ADP 100) so late boards always
    # collapsed; tiering must depend on gap distribution, not position on it.
    shape = [0, 1, 2, 5, 6, 7, 10, 11, 12]
    early = _compute_adp_tiers(_board(*[1.0 + s for s in shape]))
    late = _compute_adp_tiers(_board(*[120.0 + s for s in shape]))
    assert len(early) == len(late)
    assert [len(t) for t in early] == [len(t) for t in late]


def test_no_tier_exceeds_the_size_cap():
    # A perfectly uniform board has no meaningful break, but a 25-player
    # "these are all interchangeable" claim is indefensible regardless.
    tiers = _compute_adp_tiers(_board(*[10.0 + i for i in range(25)]))
    assert all(len(t) <= 8 for t in tiers)
    assert sum(len(t) for t in tiers) == 25


def test_tiers_preserve_every_player_and_their_order():
    adps = [1.0, 1.2, 4.0, 4.1, 4.2, 9.0, 20.0, 20.1]
    tiers = _compute_adp_tiers(_board(*adps))
    flat = [p["adp"] for t in tiers for p in t]
    assert flat == adps


def test_a_clear_cliff_starts_a_new_tier():
    tiers = _compute_adp_tiers(_board(1.0, 1.1, 1.2, 40.0, 40.1, 40.2))
    assert len(tiers) >= 2
    assert tiers[0][-1]["adp"] == 1.2
    assert tiers[1][0]["adp"] == 40.0


def test_tiny_boards_are_left_alone():
    assert _compute_adp_tiers([]) == []
    assert len(_compute_adp_tiers(_board(1.0))) == 1
    assert len(_compute_adp_tiers(_board(1.0, 50.0))) == 1


def test_board_no_longer_claims_distant_players_are_interchangeable():
    ctx = ctx_with_available(*range(1, 26))
    for i, p in enumerate(ctx.top_available):
        p["adp"] = 40.0 + i * 1.1
    prompt = _build_prompt(ctx)
    assert "roughly interchangeable" not in prompt
    assert "NOT interchangeable with a higher one" in prompt


# ---------------------------------------------------------------------------
# Kicker requirement vs. kicker availability
#
# Live failure: a full 15-round draft ended with no kicker. _K_SLOTS asserted
# the league starts one, but this app's ADP source returns zero kickers (226
# players, 0 K — FantasyFootballCalculator simply doesn't include them), and
# _parse_response only accepts a player_id drawn from the available board. So
# the prompt demanded a pick that could not physically be made, and the
# reserved-rounds math held a round open for it.
# ---------------------------------------------------------------------------

def _late_ctx(k_available: int, round_number: int = 15):
    ctx = ctx_with_available(1)
    ctx.available_counts = {"DST": 12, "K": k_available}
    ctx.round_number = round_number
    ctx.total_rounds = 15
    ctx.my_roster = roster("QB", "RB", "RB", "WR", "WR", "TE", "RB", "DST")
    return ctx


def test_kicker_requirement_dropped_when_none_can_be_drafted():
    prompt = _build_prompt(_late_ctx(k_available=0))
    assert "Still owed" not in prompt
    assert "K x1" not in prompt


def test_missing_kickers_are_announced_not_silently_dropped():
    # Silently omitting it would mean the user finds out in week 1.
    prompt = _build_prompt(_late_ctx(k_available=0))
    assert "no kickers at all" in prompt
    assert "waivers" in prompt


def test_kicker_requirement_stands_when_kickers_exist():
    prompt = _build_prompt(_late_ctx(k_available=12))
    assert "Still owed" in prompt and "K x1" in prompt
    assert "no kickers at all" not in prompt


def test_absent_kickers_do_not_consume_a_reserved_round():
    # The deferral math holds the tail of the draft for DST+K. An
    # undraftable kicker must not reserve a round: doing so makes the
    # urgency check pessimistic by a full round for the whole back half.
    # (This roster already has its DST, so K is the only late slot at
    # issue — with no kicker there is nothing left to reserve for.)
    no_k = _build_prompt(_late_ctx(k_available=0, round_number=13))
    with_k = _build_prompt(_late_ctx(k_available=12, round_number=13))
    assert "Still owed" not in no_k
    assert "final 1 round(s)" in with_k
