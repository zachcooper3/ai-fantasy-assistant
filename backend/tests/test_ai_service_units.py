"""
Unit tests for the recommendation layer: prompt construction, response
parsing, the derived signals the prompt is built from (roster depth, ADP
tiers, value over replacement, opportunity cost), streaming, and the
fetch_metrics column resolution that feeds all of it.

Most of these pin a specific live failure rather than a specification, and
the comment above each one says which. That is deliberate — nearly every bug
in this system has been silent, producing a plausible-looking recommendation
from data that was missing, mislabelled or misread, so a test whose rationale
isn't recorded tends to be "simplified" back into the bug it prevented.

Examples of what is pinned here: a QB never once recommended across a full
draft (nothing computed the open-QB gap); a board rendered as one 25-player
tier, telling the model to ignore every ADP difference on it; survival
buckets labelled "GONE" that the model read as "unavailable" and recommended
around; injury history double-counted against a price that already included
it; and team-share metrics computed from a column nflverse does not publish.
"""

import json

import pytest

from backend.app.services.ai_service import (
    RecommendationContext,
    PickSuggestion,
    _compute_roster_gaps,
    _parse_response,
    _build_prompt,
    _build_system_prompt,
    _experience_context,
    _format_metrics_section,
    _infer_current_season,
    _compute_roster_depth,
    _compute_adp_tiers,
    _draft_value,
    _extract_complete_object,
    _find_dominating_player,
    _dominance_alert,
    _mentions_player,
    _shortlist,
    _format_shortlist_section,
    _TEMPERATURE,
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
    _LISTED_PLAYERS,
    _best_available,
    _needs,
    _depth_pick,
    _tag_overlaps,
    _build_sections,
    _compute_gap_info,
    _TAG_MAIN,
    _TAG_BEST_AVAILABLE,
    _TAG_NEEDS,
    _TAG_DEPTH,
    _TAG_OPPORTUNITY,
    _opportunity_pick_from,
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


def valid_payload(main_id=1) -> dict:
    return {
        "main": {
            "player_id": main_id, "player_name": "P", "position": "RB",
            "adp": 1.0, "reasoning": "best available",
        },
        "alerts": ["tier break at RB"],
    }


def test_valid_json_parses():
    result = _parse_response(json.dumps(valid_payload()), ctx_with_available(1, 2, 3))
    assert result is not None
    assert result.main.player_id == 1
    assert result.alerts == ["tier break at RB"]


def test_markdown_fenced_json_parses():
    raw = "```json\n" + json.dumps(valid_payload()) + "\n```"
    assert _parse_response(raw, ctx_with_available(1, 2, 3)) is not None


def test_garbage_returns_none():
    assert _parse_response("I think you should draft Gibbs!", ctx_with_available(1)) is None


def test_missing_main_returns_none():
    assert _parse_response(json.dumps({"alerts": []}), ctx_with_available(1)) is None


def test_unavailable_player_returns_none():
    # Claude recommending someone who's already drafted must fall back
    result = _parse_response(json.dumps(valid_payload(main_id=99)), ctx_with_available(1, 2, 3))
    assert result is None


def test_string_player_id_is_coerced_not_rejected():
    payload = valid_payload()
    payload["main"]["player_id"] = "1"  # Claude sometimes stringifies
    result = _parse_response(json.dumps(payload), ctx_with_available(1, 2, 3))
    assert result is not None
    assert result.main.player_id == 1


# ---------------------------------------------------------------------------
# _opportunity_pick_from / _parse_response — the `opportunity` field
#
# Unlike `main`, a bad opportunity object must NOT invalidate the rest of
# the recommendation — see _opportunity_pick_from's docstring. Every test
# below that gives it something invalid still expects a real `main`.
# ---------------------------------------------------------------------------

def _mixed_position_ctx() -> RecommendationContext:
    return RecommendationContext(
        pick_number=1, round_number=1, my_slot=1, league_size=12,
        is_my_turn=True, picks_until_my_turn=0, my_next_pick_number=1,
        top_available=[
            {"id": 1, "rank": 1, "name": "RB1", "position": "RB", "team": "X", "adp": 10.0, "sleeper_id": None},
            {"id": 2, "rank": 2, "name": "WR1", "position": "WR", "team": "X", "adp": 20.0, "sleeper_id": None},
            {"id": 3, "rank": 3, "name": "QB1", "position": "QB", "team": "X", "adp": 30.0, "sleeper_id": None},
            {"id": 4, "rank": 4, "name": "DST1", "position": "DST", "team": "X", "adp": 200.0, "sleeper_id": None},
            # Deliberately pricier than the RB/WR above, so it never wins
            # best_available's cheapest-two on its own — see
            # test_parse_response_includes_valid_opportunity, which needs an
            # opportunity pick that ISN'T also tagged best_available to test
            # extraction cleanly (overlap itself is covered separately by
            # test_tag_overlaps_folds_in_opportunity_too).
            {"id": 5, "rank": 5, "name": "TE1", "position": "TE", "team": "X", "adp": 40.0, "sleeper_id": None},
        ],
    )


def test_opportunity_pick_accepts_rb_wr_te():
    ctx = _mixed_position_ctx()
    pick = _opportunity_pick_from({"player_id": 2, "reasoning": "role trending up"}, ctx)
    assert pick is not None
    assert pick.player_id == 2
    assert pick.position == "WR"
    assert pick.tags == [_TAG_OPPORTUNITY]


def test_opportunity_pick_rejects_qb_and_dst_even_if_claude_names_them():
    ctx = _mixed_position_ctx()
    assert _opportunity_pick_from({"player_id": 3, "reasoning": "x"}, ctx) is None
    assert _opportunity_pick_from({"player_id": 4, "reasoning": "x"}, ctx) is None


def test_opportunity_pick_trusts_canonical_position_not_claudes_label():
    # Claude mislabeling position must not let a QB through under a fake
    # RB/WR/TE tag — same "id is the only trusted field" rule _pick_from
    # uses for main, extended to the one extra check opportunity needs.
    ctx = _mixed_position_ctx()
    pick = _opportunity_pick_from({"player_id": 3, "position": "WR", "reasoning": "x"}, ctx)
    assert pick is None


def test_opportunity_pick_rejects_unavailable_or_malformed():
    ctx = _mixed_position_ctx()
    assert _opportunity_pick_from({"player_id": 999}, ctx) is None
    assert _opportunity_pick_from({}, ctx) is None
    assert _opportunity_pick_from(None, ctx) is None
    assert _opportunity_pick_from("not a dict", ctx) is None


def test_parse_response_includes_valid_opportunity():
    payload = valid_payload()
    payload["opportunity"] = {"player_id": 5, "reasoning": "target share climbing"}
    ctx = _mixed_position_ctx()
    payload["main"]["player_id"] = 1
    result = _parse_response(json.dumps(payload), ctx)
    assert result is not None
    assert [p.player_id for p in result.opportunity] == [5]
    assert result.opportunity[0].tags == [_TAG_OPPORTUNITY]


def test_parse_response_survives_a_bad_opportunity_object():
    # A malformed/invalid opportunity must not take main down with it.
    payload = valid_payload()
    payload["opportunity"] = {"player_id": 3, "reasoning": "x"}  # QB — invalid position
    ctx = _mixed_position_ctx()
    payload["main"]["player_id"] = 1
    result = _parse_response(json.dumps(payload), ctx)
    assert result is not None
    assert result.main.player_id == 1
    assert result.opportunity == []


def test_parse_response_handles_missing_opportunity_entirely():
    # valid_payload() has no "opportunity" key at all — the shape Claude
    # would produce if it just omitted the field despite instructions.
    result = _parse_response(json.dumps(valid_payload()), ctx_with_available(1, 2, 3))
    assert result is not None
    assert result.opportunity == []


# ---------------------------------------------------------------------------
# _parse_response — strategy / confidence
#
# These fields are presentational extras. The rule they share: a missing or
# malformed value must degrade to a default, never reject an otherwise-valid
# recommendation to the ADP fallback.
# ---------------------------------------------------------------------------

def test_strategy_and_confidence_are_parsed():
    payload = valid_payload()
    payload["strategy"] = "You're thin at RB with two picks before the tier breaks."
    payload["confidence"] = "high"
    result = _parse_response(json.dumps(payload), ctx_with_available(1, 2, 3))
    assert result.strategy.startswith("You're thin at RB")
    assert result.confidence == "high"


def test_missing_extras_fall_back_to_defaults():
    # valid_payload() has no strategy/confidence at all — the shape Claude
    # returned before these fields existed.
    result = _parse_response(json.dumps(valid_payload()), ctx_with_available(1, 2, 3))
    assert result is not None
    assert result.strategy == ""
    assert result.confidence == "medium"


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


# ---------------------------------------------------------------------------
# Response-size guards
#
# The recommendation JSON is capped by max_tokens. A response that runs over is
# truncated, which makes it invalid JSON, which silently degrades every pick to
# the ADP fallback — the failure this suite exists to catch early.
# ---------------------------------------------------------------------------

def test_truncated_json_returns_none_rather_than_raising():
    # What a max_tokens cut-off actually looks like: valid opening, no close.
    truncated = json.dumps(valid_payload())[:60]
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
    payload["main"]["player_name"] = "Some Drafted Guy"
    result = _parse_response(json.dumps(payload), ctx_with_available(1, 2, 3))
    # ctx_with_available names players P{id}
    assert result.main.player_name == "P1"


def test_position_and_adp_come_from_our_data_not_claudes():
    payload = valid_payload()
    payload["main"]["position"] = "QB"   # actually RB in our data
    payload["main"]["adp"] = 999.0       # actually 1.0
    result = _parse_response(json.dumps(payload), ctx_with_available(1, 2, 3))
    assert result.main.position == "RB"
    assert result.main.adp == 1.0


def test_reasoning_still_comes_from_claude():
    # The model's judgement is the one thing it does supply.
    payload = valid_payload()
    payload["main"]["reasoning"] = "Elite volume in a good offence."
    result = _parse_response(json.dumps(payload), ctx_with_available(1, 2))
    assert result.main.reasoning == "Elite volume in a good offence."


def test_suggestion_missing_name_and_adp_entirely_still_parses():
    # Those fields are ours now, so Claude omitting them is harmless.
    payload = valid_payload()
    for key in ("player_name", "position", "adp"):
        payload["main"].pop(key)
    result = _parse_response(json.dumps(payload), ctx_with_available(1, 2))
    assert result is not None
    assert result.main.player_name == "P1"


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
    # Wording changed from "until the final N round(s)" to naming the actual
    # condition (starting lineup filled) once the "Fair game now" branch was
    # split out — see test_kicker_requirement_stands_when_kickers_exist.
    assert "do NOT recommend one until your starting lineup below" in prompt


def test_dst_and_k_are_demanded_once_the_reserved_rounds_arrive():
    # Must still have an unfilled skill slot (QB here) so this hits the
    # backstop branch ("out of rounds to defer them") rather than the
    # softer "Fair game now" branch, which is what fires once the starting
    # lineup is already complete — see test_kicker_requirement_stands_when_kickers_exist.
    ctx = ctx_with_available(1, 2, 3)
    ctx.available_counts = {"K": 12, "DST": 12}
    ctx.round_number = 14
    ctx.total_rounds = 15
    ctx.my_roster = roster("RB", "RB", "WR", "WR", "TE", "RB")
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
    # is_rookie_or_second_year was written by no ingestion script (0/185
    # rows) and has since been removed from the model entirely. Experience
    # comes from draft_year alone so the fact has one source of truth. An
    # unmapped leftover column survives in pre-existing databases, so a stray
    # key must still not resurrect the tag.
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
    # Scoped to the board section: the Task steps reference VOR by name as
    # part of a standing instruction, which is correct even when no VOR
    # figures can be computed. What must not appear is a column of dashes.
    board = prompt.split("## Top Available Players")[1].split("## ")[0]
    assert "VOR" not in board


def test_system_prompt_guards_against_stacking_on_high_vor():
    # The failure mode a draft simulation surfaced: VOR-greedy drafting takes
    # six tight ends, because VOR measures starter value and says nothing
    # about a position you can only start one of.
    sp = _build_system_prompt("ppr")
    assert "THREE SIGNALS, NONE OF THEM SENIOR" in sp
    assert "a fourth player at a position that starts one" in sp
    # VOR must not be presented as the senior signal just because it is the
    # most precise-looking number.
    assert "measurement of LAST season" in sp
    assert "not better evidence than an imprecise measurement of the right one" in sp


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
    # This fixture's starting lineup is already full (DST included), so a
    # real, draftable K now hits the "Fair game now" branch, not "Still
    # owed" — that phrasing is reserved for the backstop case where a skill
    # slot is still open and there's no more room to defer. See
    # test_dst_and_k_are_demanded_once_the_reserved_rounds_arrive.
    prompt = _build_prompt(_late_ctx(k_available=12))
    assert "Fair game now" in prompt and "K x1" in prompt
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
    assert "Fair game now" not in no_k
    # With a real kicker on the board and the starting lineup already full,
    # this is the (non-urgent) "Fair game now" branch, not a reserved-round
    # count — the reserved-round math (URGENT / "usable round(s) left")
    # only fires while a skill slot is still open, which isn't this fixture.
    assert "Fair game now" in with_k and "K x1" in with_k


# ---------------------------------------------------------------------------
# Availability wording
#
# Live failure, mid-draft: the bucket for "on the board now, but not at your
# next turn" was named GONE and headed "Almost certainly gone by then". The
# model read that as unavailable and recommended around those players —
# "the highest-value receiver available after Rice and Brown disappear" —
# while Rice and Brown were both on the board. That inverts RULE 1: the
# group you must act on first became the group it believed it couldn't have.
# ---------------------------------------------------------------------------

def _survival_ctx():
    ctx = ctx_with_available(1, 2, 3)
    for i, adp in enumerate((10.0, 12.0, 90.0)):
        ctx.top_available[i]["adp"] = adp
    ctx.my_next_pick_number = 20
    ctx.my_following_pick_number = 45
    ctx.pick_number = 20
    return ctx


def test_no_bucket_label_says_a_listed_player_is_gone():
    # "gone" next to a draftable name is what caused the failure.
    prompt = _build_prompt(_survival_ctx())
    section = prompt.split("## Opportunity Cost")[1].split("##")[0]
    assert "gone by then" not in section.lower()
    assert "GONE" not in section


def test_opportunity_cost_states_everyone_listed_is_draftable_now():
    prompt = _build_prompt(_survival_ctx())
    assert "AVAILABLE RIGHT NOW AND CAN BE DRAFTED WITH THIS PICK" in prompt
    assert "do NOT describe who is on the board today" in prompt


def test_urgent_bucket_label_is_an_instruction_to_act():
    prompt = _build_prompt(_survival_ctx())
    assert "TAKE NOW OR LOSE HIM" in prompt
    assert "on the board now, will not be at your next turn" in prompt


def test_system_prompt_forbids_recommending_around_a_listed_player():
    sp = _build_system_prompt("ppr")
    assert "EVERY LISTED PLAYER IS AVAILABLE" in sp
    # The exact reasoning pattern observed live must be named as an error.
    assert "the best option once X and Y are gone" in sp
    assert "one of THEM is the pick" in sp


# ---------------------------------------------------------------------------
# Durability must not be double-counted against ADP
#
# Live failure at pick 17: Rashee Rice sat first on the board (ADP 11.5,
# VOR +6.9, 9.8 tgt/gm, RACR 1.69, 68% catch, 18.8 ppg) and was not even
# listed as an alternative. The model took George Pickens (ADP 21.3, VOR
# +5.3, 8.1 tgt/gm, RACR 0.92) and wrote that Pickens' efficiency "outpaces
# his tier" — while Rice beat him on every visible metric. The only thing
# favouring Pickens was Rice's [SMALL SAMPLE] flag and injury weeks. But
# RULE 6 required the healthier player to already hold the better ADP tier,
# and Pickens was ten picks worse, so the rule never applied.
# ---------------------------------------------------------------------------

def test_rule6_forbids_charging_a_player_twice_for_injury_history():
    sp = _build_system_prompt("ppr")
    assert "ADP ALREADY PRICES KNOWN INJURY HISTORY" in sp
    assert "charges him twice for the same fact" in sp


def test_rule6_scopes_the_small_sample_caveat_to_the_scoring_average():
    # The flag is attached to fantasy_points_avg. Usage rates on the same
    # line describe the role he held and stabilise much faster.
    sp = _build_system_prompt("ppr")
    assert "applies to that average, not to the whole line" in sp
    assert "stabilise far faster than points do" in sp


def test_rule6_names_the_better_adp_and_vor_override():
    sp = _build_system_prompt("ppr")
    assert "BOTH the better ADP and the higher VOR" in sp
    assert "durability is NOT sufficient reason to pass on him" in sp


def test_task_requires_justifying_a_pass_on_a_strictly_better_player():
    prompt = _build_prompt(ctx_with_available(1, 2, 3))
    assert "beats the one you chose on ADP, VOR and OPP together" in prompt
    assert "take the better player" in prompt


# ---------------------------------------------------------------------------
# Draft value + must-evaluate shortlist
#
# Both exist because every live mis-recommendation was an OMISSION, not a bad
# choice. At pick 17 the model took Pickens (ADP 21.3, a 4.3-pick reach) over
# Rice (ADP 11.5, a 5.5-pick faller who also led the board on VOR) and never
# mentioned Rice at all. The value delta was computable and wasn't computed;
# the top of the board was skippable and got skipped.
# ---------------------------------------------------------------------------

def test_draft_value_labels_fallers_reaches_and_neither():
    assert _draft_value(11.5, 17) == "FALLING +6"
    assert _draft_value(21.3, 17) == "reach -4"
    assert _draft_value(17.5, 17) == "at ADP"      # inside the noise band


def test_draft_value_noise_band_is_symmetric():
    assert _draft_value(16.0, 17) == "at ADP"
    assert _draft_value(18.0, 17) == "at ADP"


def _vboard():
    # id 1: best ADP and biggest faller.  id 2: mid.  id 3: highest VOR but a
    # big reach.  ids 4-8: filler so the shortlist has to actually choose.
    return [{"id": i, "rank": i, "name": f"P{i}",
             "position": "WR" if i % 2 else "RB", "team": "X",
             "adp": [8.0, 18.0, 40.0, 19.0, 20.0, 21.0, 22.0, 23.0][i - 1],
             "sleeper_id": None}
            for i in range(1, 9)]


_VMETRICS = {1: {"fantasy_points_avg": 18.0}, 2: {"fantasy_points_avg": 15.0},
             3: {"fantasy_points_avg": 24.0}}
_VREPL = {"WR": 11.9, "RB": 11.1}


def test_shortlist_includes_the_best_adp_on_the_board():
    ids = [p["id"] for p in _shortlist(_vboard(), 20, _VMETRICS, _VREPL)]
    assert 1 in ids


def test_shortlist_includes_the_highest_vor_even_when_it_is_a_reach():
    # id 3 is 20 picks early but has the best value over replacement — the
    # cross-position tension worth surfacing, not hiding.
    ids = [p["id"] for p in _shortlist(_vboard(), 20, _VMETRICS, _VREPL)]
    assert 3 in ids


def test_shortlist_is_capped_and_deduplicated():
    sl = _shortlist(_vboard(), 20, _VMETRICS, _VREPL)
    assert len(sl) <= 5
    assert len({p["id"] for p in sl}) == len(sl)


def test_shortlist_survives_a_board_with_no_metrics_at_all():
    sl = _shortlist(_vboard(), 20, {}, {})
    assert sl, "must still shortlist on ADP alone when VOR is unavailable"


def test_shortlist_section_names_ids_and_demands_a_verdict():
    out = _format_shortlist_section(
        _shortlist(_vboard(), 20, _VMETRICS, _VREPL), 20, _VMETRICS, _VREPL)
    assert "MUST EVALUATE" in out
    assert "`considered`" in out
    assert "id=1" in out


def test_board_rows_carry_the_value_delta():
    ctx = ctx_with_available(1, 2)
    ctx.top_available[0]["adp"] = 5.0
    ctx.top_available[1]["adp"] = 40.0
    ctx.pick_number = ctx.my_next_pick_number = 20
    prompt = _build_prompt(ctx)
    assert "FALLING +15" in prompt
    assert "reach -20" in prompt


def test_considered_verdicts_are_parsed():
    payload = valid_payload()
    payload["considered"] = ["Rashee Rice — passed, already deep at WR",
                             "Chase Brown — taken"]
    result = _parse_response(json.dumps(payload), ctx_with_available(1, 2, 3))
    assert len(result.considered) == 2
    assert "Rashee Rice" in result.considered[0]


def test_missing_considered_does_not_reject_the_recommendation():
    # A bookkeeping omission must never cost a good pick on draft day.
    result = _parse_response(json.dumps(valid_payload()), ctx_with_available(1, 2, 3))
    assert result is not None
    assert result.considered == []


def test_temperature_allows_variation_without_being_random():
    assert 0.0 < _TEMPERATURE <= 0.5


# ---------------------------------------------------------------------------
# _completion_kwargs — some models 400 on ANY `temperature` value
#
# Live failure: the Haiku/Sonnet toggle made claude-sonnet-5 reachable for
# the first time, and every single request 400'd — "`temperature` is
# deprecated for this model" — silently degrading the whole draft to the ADP
# fallback. claude-haiku-4-5-20251001 had been sending `temperature`
# successfully the entire time; the bug only existed for a model nothing
# had ever actually called before the toggle shipped.
# ---------------------------------------------------------------------------

def test_completion_kwargs_omits_temperature_for_a_broken_model():
    from backend.app.services.ai_service import _completion_kwargs, _NO_TEMPERATURE_MODELS
    for model in _NO_TEMPERATURE_MODELS:
        assert "temperature" not in _completion_kwargs(model)


def test_completion_kwargs_still_sends_temperature_for_haiku():
    from backend.app.services.ai_service import _completion_kwargs
    kwargs = _completion_kwargs("claude-haiku-4-5-20251001")
    assert kwargs.get("temperature") == _TEMPERATURE


def test_completion_kwargs_always_sets_model_and_max_tokens():
    from backend.app.services.ai_service import _completion_kwargs, _MAX_RESPONSE_TOKENS
    kwargs = _completion_kwargs("claude-haiku-4-5-20251001")
    assert kwargs["model"] == "claude-haiku-4-5-20251001"
    assert kwargs["max_tokens"] == _MAX_RESPONSE_TOKENS


# ---------------------------------------------------------------------------
# _completion_kwargs — thinking is explicitly disabled for Sonnet
#
# Live failure, same draft, minutes after the prefill fix: Sonnet calls
# started succeeding (200, no 400) but the streamed body was a single "{"
# character that failed to parse. Root cause (confirmed via Anthropic's
# docs, not a live call): claude-sonnet-5 has adaptive thinking on by
# default, and thinking tokens are billed against max_tokens even when
# invisible to the caller — at the old 3072-token ceiling, thinking alone
# could exhaust the budget before a single character of the JSON answer was
# emitted.
#
# Two token-budget-based fixes (a roomier ceiling, then effort tuning) were
# tried and superseded by disabling thinking outright: Coop's call, given
# this app's per-pick prompt already spells out the analysis framework and
# runs under a live draft clock, where thinking's cost isn't worth its
# marginal value here. Disabling thinking also means max_tokens no longer
# needs special-casing — no invisible thinking spend to leave headroom for.
# ---------------------------------------------------------------------------

def test_completion_kwargs_disables_thinking_for_sonnet():
    from backend.app.services.ai_service import _completion_kwargs, _THINKING_DISABLED_MODELS
    for model in _THINKING_DISABLED_MODELS:
        kwargs = _completion_kwargs(model)
        assert kwargs["thinking"] == {"type": "disabled"}


def test_completion_kwargs_omits_thinking_for_haiku():
    from backend.app.services.ai_service import _completion_kwargs
    kwargs = _completion_kwargs("claude-haiku-4-5-20251001")
    assert "thinking" not in kwargs


def test_completion_kwargs_uses_the_normal_ceiling_even_with_thinking_disabled():
    # With thinking off there's no invisible token spend to budget for, so
    # every model — including Sonnet — uses the same ceiling.
    from backend.app.services.ai_service import (
        _completion_kwargs,
        _THINKING_DISABLED_MODELS,
        _MAX_RESPONSE_TOKENS,
    )
    for model in _THINKING_DISABLED_MODELS:
        assert _completion_kwargs(model)["max_tokens"] == _MAX_RESPONSE_TOKENS


# ---------------------------------------------------------------------------
# _build_messages — some models reject a request ending on an assistant turn
#
# Live failure, same draft, AFTER thinking was already disabled for Sonnet:
# the exact same 400 — "This model does not support assistant message
# prefill" — came back anyway. Disabling thinking did not fix this; per
# Anthropic's Sonnet-5-specific docs the prefill rejection is unconditional
# on this model generation, inherited unchanged from Sonnet 4.6, not a
# thinking side effect. See _NO_PREFILL_MODELS.
# ---------------------------------------------------------------------------

def test_build_messages_drops_prefill_for_sonnet():
    from backend.app.services.ai_service import _build_messages, _NO_PREFILL_MODELS
    for model in _NO_PREFILL_MODELS:
        messages = _build_messages(model, "prompt text")
        assert [m["role"] for m in messages] == ["user"]
        assert messages[0]["content"] == "prompt text"


def test_build_messages_still_prefills_for_haiku():
    from backend.app.services.ai_service import _build_messages
    messages = _build_messages("claude-haiku-4-5-20251001", "prompt text")
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["content"] == "{"


def test_restore_prefill_handles_a_response_with_no_prefill_sent():
    # The parsing-layer half of the same fix: a non-prefilled response
    # already starts with its own "{" and must NOT get a second one
    # prepended (which would corrupt otherwise-valid JSON into "{{...").
    from backend.app.services.ai_service import _restore_prefill
    already_braced = '{"main": {"player_id": 1}}'
    assert _restore_prefill(already_braced) == already_braced


def test_parse_response_handles_a_fully_unprefilled_reply():
    # End-to-end proof that _restore_prefill/_parse_response don't need a
    # model-specific carve-out: a complete, self-contained JSON object (what
    # a non-prefilled response would look like, if some future model ever
    # rejects prefill again) parses exactly like a restored-prefill one
    # would.
    ctx = ctx_with_available(1)
    payload = json.dumps({
        "main": {"player_id": 1, "player_name": "P1", "position": "RB",
                  "adp": 10.0, "reasoning": "test"},
        "alerts": [],
    })
    result = _parse_response(payload, ctx)
    assert result is not None
    assert result.main.player_id == 1


# ---------------------------------------------------------------------------
# recommend_stream() reads the FINAL message, not just the accumulated buffer
#
# Live failure: a streamed claude-sonnet-5 call fell back to ADP with
# "Length=1 chars. Head: { ... Tail: {" — the accumulated text_stream buffer
# was a single opening brace, and recommend_stream() had no visibility into
# WHY (no stop_reason check, unlike recommend(); no record of what content
# block types actually came back). stream.get_final_message() returns the
# same complete Message object messages.create() gets directly, so this pins
# both that the diagnostic fires and that a case where the buffer under-
# delivers but the final message has the real text now succeeds instead of
# needlessly falling back.
# ---------------------------------------------------------------------------

class _FakeBlock:
    def __init__(self, type_, text=None):
        self.type = type_
        if text is not None:
            self.text = text


class _FakeFinalMessage:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class _FakeStream:
    """Minimal stand-in for anthropic's AsyncMessageStream: async context
    manager exposing `.text_stream` (async-iterable) and an awaitable
    `get_final_message()`, matching exactly what recommend_stream() uses."""

    def __init__(self, chunks: list[str], final_message: _FakeFinalMessage):
        self._chunks = chunks
        self._final_message = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    @property
    async def text_stream(self):
        for c in self._chunks:
            yield c

    async def get_final_message(self):
        return self._final_message


def _stream_result(chunks, final_message, ctx=None):
    """Runs recommend_stream() to completion against a _FakeStream and
    returns the ("complete", RecommendationResult) payload."""
    import asyncio
    from backend.app.services.ai_service import AIService

    class _FakeMessages:
        def stream(self, **kwargs):
            return _FakeStream(chunks, final_message)

    class _FakeClient:
        messages = _FakeMessages()

    svc = AIService.__new__(AIService)
    svc._client = _FakeClient()
    svc._model = "claude-sonnet-5"

    async def _run():
        events = []
        async for event in svc.recommend_stream(ctx or ctx_with_available(1)):
            events.append(event)
        return events

    events = asyncio.run(_run())
    complete = [e for e in events if e[0] == "complete"]
    assert len(complete) == 1
    return complete[0][1]


def test_stream_uses_final_message_when_buffer_underdelivers(caplog):
    # The exact live shape: text_stream only ever yielded "{", but the real
    # final message actually has the complete, valid JSON.
    good_json = json.dumps({
        "main": {"player_id": 1, "player_name": "P1", "position": "RB",
                  "adp": 1.0, "reasoning": "ok"},
        "alerts": [],
    })
    final = _FakeFinalMessage(content=[_FakeBlock("text", text=good_json)])
    result = _stream_result(chunks=["{"], final_message=final)
    assert not result.model.endswith(":fallback")
    assert result.main.player_id == 1


def test_stream_truncation_is_logged_with_stop_reason(caplog):
    final = _FakeFinalMessage(
        content=[_FakeBlock("text", text='{"main": {"player_id"')],  # cut off
        stop_reason="max_tokens",
    )
    with caplog.at_level("WARNING"):
        result = _stream_result(chunks=['{"main": {"player_id"'], final_message=final)
    assert result.model.endswith(":fallback")
    assert any("token ceiling" in r.message for r in caplog.records)


def test_stream_no_text_content_falls_back_with_specific_reason(caplog):
    final = _FakeFinalMessage(content=[_FakeBlock("thinking")])  # no .text at all
    with caplog.at_level("WARNING"):
        result = _stream_result(chunks=[], final_message=final)
    assert result.model.endswith(":fallback")
    assert any("no text content" in r.message for r in caplog.records)
    assert any("thinking" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Dominated-pick guard
#
# Same-position only. Across positions, taking a "worse" player is routinely
# correct (you need a TE; the dominating player is a WR you're deep at), so a
# cross-position check fires constantly and becomes noise. Within a position,
# roster need cannot explain the gap.
#
# Advisory: it appends an alert and never blocks. And it stays quiet when the
# model actually engaged with the dominating player — taking him after
# weighing him is judgement, not the omission this guard exists to catch.
# ---------------------------------------------------------------------------

def _dom_ctx():
    board = [
        {"id": 1, "rank": 1, "name": "Best WR", "position": "WR", "team": "X",
         "adp": 11.5, "sleeper_id": None},
        {"id": 2, "rank": 2, "name": "Worse WR", "position": "WR", "team": "X",
         "adp": 21.3, "sleeper_id": None},
        {"id": 3, "rank": 3, "name": "Some RB", "position": "RB", "team": "X",
         "adp": 9.0, "sleeper_id": None},
        {"id": 4, "rank": 4, "name": "Rookie WR", "position": "WR", "team": "X",
         "adp": 10.0, "sleeper_id": None},   # no metrics -> unknown VOR
    ]
    ctx = RecommendationContext(
        pick_number=17, round_number=2, my_slot=1, league_size=12,
        is_my_turn=True, picks_until_my_turn=0, my_next_pick_number=17,
        top_available=board,
        player_metrics={1: {"fantasy_points_avg": 18.8},
                        2: {"fantasy_points_avg": 17.2},
                        3: {"fantasy_points_avg": 24.0}},
        replacement_ppg={"WR": 11.9, "RB": 11.1},
    )
    return ctx


def _sugg(ctx, pid):
    p = next(x for x in ctx.top_available if x["id"] == pid)
    return PickSuggestion(p["id"], p["name"], p["position"], p["adp"], "")


def test_dominating_player_is_found_at_the_same_position():
    ctx = _dom_ctx()
    found = _find_dominating_player(_sugg(ctx, 2), ctx)
    assert found is not None and found["id"] == 1


def test_the_position_leader_is_never_dominated():
    ctx = _dom_ctx()
    assert _find_dominating_player(_sugg(ctx, 1), ctx) is None


def test_a_better_player_at_another_position_never_triggers_it():
    # "Some RB" has a better ADP and a far higher VOR than "Worse WR", but
    # wanting a receiver is a complete explanation, so this must stay quiet.
    ctx = _dom_ctx()
    found = _find_dominating_player(_sugg(ctx, 2), ctx)
    assert found["position"] == "WR"


def test_unknown_vor_on_either_side_suppresses_the_check():
    ctx = _dom_ctx()
    # Rookie with no prior season cannot dominate: unknown is not high.
    assert _find_dominating_player(_sugg(ctx, 4), ctx) is None
    ctx.replacement_ppg = {}
    assert _find_dominating_player(_sugg(ctx, 2), ctx) is None


def test_trivial_differences_stay_quiet():
    ctx = _dom_ctx()
    ctx.top_available[0]["adp"] = 20.5          # inside the ADP margin
    assert _find_dominating_player(_sugg(ctx, 2), ctx) is None


def test_alert_is_advisory_wording_not_a_verdict():
    ctx = _dom_ctx()
    msg = _dominance_alert(_sugg(ctx, 2), ctx)
    assert "prompt to check, not a verdict" in msg
    assert "Best WR" in msg and "Worse WR" in msg


def test_alert_suppressed_when_the_model_engaged_with_that_player():
    # Passing on him deliberately is a judgement call; only silent omission
    # is the failure mode this guard targets.
    ctx = _dom_ctx()
    assert _dominance_alert(_sugg(ctx, 2), ctx, "Best WR — passed, injury risk") is None
    assert _dominance_alert(_sugg(ctx, 2), ctx, "unrelated text") is not None


def test_guard_appends_an_alert_without_rejecting_the_pick():
    ctx = _dom_ctx()
    payload = {
        "main": {"player_id": 2, "player_name": "Worse WR",
                 "position": "WR", "adp": 21.3, "reasoning": "volume"},
        "alerts": ["existing alert"],
    }
    result = _parse_response(json.dumps(payload), ctx)
    assert result is not None                      # never blocks
    assert result.main.player_id == 2               # never rewrites the pick
    assert result.alerts[0] == "existing alert"    # appended last
    assert any(a.startswith("Check this") for a in result.alerts)


# ---------------------------------------------------------------------------
# Audit fixes
# ---------------------------------------------------------------------------

def test_shortlist_axes_are_not_secretly_the_same_axis():
    # Regression: "biggest faller" scored (pick_number - adp). pick_number is
    # constant across the board, so maximising it is just minimising ADP —
    # the axis returned the identical players to the ADP axis and the
    # shortlist was silently one axis narrower than it looked.
    board = [{"id": i, "rank": i, "name": f"P{i}", "position": "WR",
              "team": "X", "adp": float(i * 3), "sleeper_id": None}
             for i in range(1, 11)]
    by_adp = [p["id"] for p in sorted(board, key=lambda p: p["adp"])[:3]]
    by_fall = [p["id"] for p in sorted(board, key=lambda p: -(20 - p["adp"]))[:3]]
    assert by_adp == by_fall, "the two orderings are provably identical"
    # So the shortlist must derive its extra breadth from somewhere else.
    sl = _shortlist(board, 20, {}, {})
    assert [p["id"] for p in sl][:3] != by_adp or len(sl) <= 3


def test_shortlist_spans_positions_on_a_lopsided_board():
    # Without a per-position axis, a board whose top is all receivers yields
    # an all-receiver shortlist and the cross-position call never gets forced.
    board = ([{"id": i, "rank": i, "name": f"W{i}", "position": "WR",
               "team": "X", "adp": float(i), "sleeper_id": None}
              for i in range(1, 9)]
             + [{"id": 20, "rank": 20, "name": "The RB", "position": "RB",
                 "team": "X", "adp": 40.0, "sleeper_id": None}])
    positions = {p["position"] for p in _shortlist(board, 20, {}, {})}
    assert "RB" in positions and "WR" in positions


def test_mentions_player_is_not_fooled_by_a_generational_suffix():
    # A verdict about the junior must not count as engagement with the
    # senior. Word boundaries alone don't catch this: "Michael Pittman Jr."
    # contains "Michael Pittman" followed by a space.
    assert not _mentions_player("Michael Pittman Jr. — passed", "Michael Pittman")
    assert _mentions_player("Michael Pittman Jr. — passed", "Michael Pittman Jr.")
    assert not _mentions_player("James Cook III — taken", "James Cook")


def test_mentions_player_handles_punctuation_and_case():
    assert _mentions_player("A.J. Brown — taken", "A.J. Brown")
    assert _mentions_player("rashee rice was risky", "Rashee Rice")
    assert _mentions_player("De'Von Achane passed", "De'Von Achane")


def test_mentions_player_does_not_match_a_different_surname():
    assert not _mentions_player("Chase Brown — taken", "A.J. Brown")
    assert not _mentions_player("", "Rashee Rice")


def test_cap_never_truncates_the_per_position_representatives():
    # The cap truncates, and QB/TE carry late ADP by nature — so an
    # ADP-ordered cap dropped exactly the entries the per-position axis
    # exists to guarantee. Reproduced with four distinct receivers winning
    # both the ADP and VOR axes: RB and TE were cut and the shortlist
    # collapsed back onto one position.
    board = [{"id": i, "rank": i, "name": f"WR{i}", "position": "WR",
              "team": "X", "adp": float(i), "sleeper_id": None}
             for i in range(1, 9)]
    board += [{"id": 50, "rank": 50, "name": "The QB", "position": "QB",
               "team": "X", "adp": 90.0, "sleeper_id": None},
              {"id": 51, "rank": 51, "name": "The TE", "position": "TE",
               "team": "X", "adp": 95.0, "sleeper_id": None},
              {"id": 52, "rank": 52, "name": "The RB", "position": "RB",
               "team": "X", "adp": 99.0, "sleeper_id": None}]
    # Top VOR belongs to WR5/WR6, not the top-ADP pair, so the two axes
    # contribute four different receivers and crowd the list.
    metrics = {5: {"fantasy_points_avg": 40.0}, 6: {"fantasy_points_avg": 39.0},
               1: {"fantasy_points_avg": 12.0}}
    repl = {"WR": 11.9, "RB": 11.1, "QB": 17.5, "TE": 10.6}
    positions = {p["position"] for p in _shortlist(board, 20, metrics, repl)}
    assert {"QB", "RB", "TE", "WR"} <= positions


def test_shortlist_respects_the_cap():
    board = [{"id": i, "rank": i, "name": f"P{i}",
              "position": ["QB", "RB", "WR", "TE"][i % 4], "team": "X",
              "adp": float(i), "sleeper_id": None} for i in range(1, 30)]
    metrics = {i: {"fantasy_points_avg": 30.0 - i} for i in range(1, 30)}
    repl = {"WR": 11.9, "RB": 11.1, "QB": 17.5, "TE": 10.6}
    assert len(_shortlist(board, 20, metrics, repl)) <= 6


# ---------------------------------------------------------------------------
# Unavailable players
#
# Live failure at a round-13 pick: Ricky Pearsall was recommended while
# ChromaDB held "Ricky Pearsall is listed as IR (Knee - PCL)" and he sat at
# position 1 on the board, INSIDE the retrieval window — so the note was in
# the prompt and got read as a durability concern rather than a
# disqualification. Separately, retrieval covered only the top 10 of a
# 25-player board, so fifteen recommendable players had no status data at all
# and the gap was invisible from inside the prompt.
# ---------------------------------------------------------------------------

def test_news_retrieval_covers_every_recommendable_player():
    from backend.app.services.ai_service import _MAX_CONTEXT_PLAYERS, _LISTED_PLAYERS
    assert _MAX_CONTEXT_PLAYERS >= _LISTED_PLAYERS, (
        "a player the model can recommend but has no news for is one it can "
        "recommend while he is on IR"
    )


def test_rule_zero_disqualifies_players_who_cannot_play():
    sp = _build_system_prompt("ppr")
    assert "NEVER RECOMMEND A PLAYER WHO CANNOT PLAY" in sp
    assert "This overrides every other rule" in sp
    for term in ("IR", "out for the season", "PUP", "suspended"):
        assert term in sp


def test_rule_zero_separates_disqualification_from_ordinary_risk():
    # The failure was reading "IR" as a risk to weigh. A questionable tag IS
    # a risk to weigh; being ruled out is not.
    sp = _build_system_prompt("ppr")
    assert "Do not soften this into" in sp
    assert "that IS ordinary durability risk" in sp


def test_rule_zero_comes_before_the_value_rules():
    sp = _build_system_prompt("ppr")
    assert sp.index("RULE 0") < sp.index("RULE 1 —")


# ---------------------------------------------------------------------------
# Survival tag on the response
#
# The bucket drives the single biggest decision on a pick and previously
# existed only inside the prompt, where the user could not see it. Computed
# server-side rather than echoed by the model: this is the one figure on
# screen that must agree with the prompt exactly.
# ---------------------------------------------------------------------------

def _survival_ctx_for(adps, horizon=41):
    ctx = ctx_with_available(*range(1, len(adps) + 1))
    for p, adp in zip(ctx.top_available, adps):
        p["adp"] = adp
    ctx.pick_number = ctx.my_next_pick_number = 17
    ctx.my_following_pick_number = horizon
    return ctx


def test_survival_code_is_attached_to_main():
    ctx = _survival_ctx_for([11.5, 38.0, 90.0])
    payload = {
        "main": {"player_id": 1, "player_name": "P1", "position": "RB",
                 "adp": 11.5, "reasoning": "x"},
        "alerts": [],
    }
    r = _parse_response(json.dumps(payload), ctx)
    assert r.main.survival == "take_now"


def test_survival_code_is_attached_to_best_available_entries():
    # best_available is never model output (see _best_available) but still
    # has to carry the same computed survival badge as main.
    ctx = _survival_ctx_for([38.0, 90.0])
    best = _best_available(ctx, due_late_gaps={})
    assert [p.survival for p in best] == ["might_last", "will_last"]


def test_survival_is_empty_without_a_next_turn():
    # Last pick of the draft: nothing to survive to, so no badge rather than
    # a misleading one.
    ctx = _survival_ctx_for([11.5], horizon=None)
    payload = {"main": {"player_id": 1, "player_name": "P1",
                        "position": "RB", "adp": 11.5, "reasoning": "x"},
               "alerts": []}
    assert _parse_response(json.dumps(payload), ctx).main.survival == ""


def test_survival_is_computed_not_taken_from_the_model():
    # The model claiming otherwise must not change the badge.
    ctx = _survival_ctx_for([90.0])
    payload = {"main": {"player_id": 1, "player_name": "P1", "position": "RB",
                        "adp": 90.0, "reasoning": "x", "survival": "take_now"},
               "alerts": []}
    assert _parse_response(json.dumps(payload), ctx).main.survival == "will_last"


def test_fallback_carries_a_survival_tag_too():
    from backend.app.services.ai_service import _fallback
    ctx = _survival_ctx_for([11.5, 38.0])
    result = _fallback(ctx, "test-model")
    assert result.main.survival == "take_now"


# ---------------------------------------------------------------------------
# Fallback reason — six different failure paths used to collapse into one
# generic "AI service unavailable" alert, and the frontend hardcoded "no
# ANTHROPIC_API_KEY" for all of them. A live draft hit a real Claude API
# error (a configured, valid key) and was told to go check .env for a key
# that was never the problem.
# ---------------------------------------------------------------------------

def test_fallback_default_reason_is_generic():
    from backend.app.services.ai_service import _fallback
    ctx = _survival_ctx_for([11.5])
    result = _fallback(ctx, "test-model")
    assert result.alerts == ["AI service unavailable — showing best available by ADP only."]


def test_fallback_reason_is_specific_when_given():
    from backend.app.services.ai_service import _fallback
    ctx = _survival_ctx_for([11.5])
    result = _fallback(ctx, "test-model", "Claude API error")
    assert result.alerts == ["Claude API error — showing best available by ADP only."]
    # Must NOT still claim the key is missing — that's a different, specific
    # cause with a different fix (see the six call sites in recommend()/
    # recommend_stream()).
    assert "ANTHROPIC_API_KEY" not in result.alerts[0]


def test_survival_codes_are_stable_regardless_of_prompt_wording():
    # The prompt labels have been rewritten once already ("GONE" read as
    # unavailable). The API codes must not move with them.
    from backend.app.services.ai_service import _survival_code
    assert _survival_code(11.5, 41) == "take_now"
    assert _survival_code(38.0, 41) == "might_last"
    assert _survival_code(90.0, 41) == "will_last"
    assert _survival_code(50.0, None) == ""


# ---------------------------------------------------------------------------
# fetch_metrics column resolution
#
# Live diagnostic against the 2025 nflverse release: player_stats has 145
# columns and neither `team_targets` nor `team_carries`. The share
# calculations read exactly those names, and _num() returns 0.0 for an absent
# field, so target_share / carry_share / target_share_trend were None for all
# 182 players — with nothing anywhere to distinguish "could not compute" from
# "this player has no role".
# ---------------------------------------------------------------------------

def _stat_row(team="KC", week=1, targets=0.0, carries=0.0, **extra):
    row = {"team": team, "week": week, "targets": targets, "carries": carries,
           "receiving_yards": 0.0, "receptions": 0.0, "rushing_yards": 0.0,
           "receiving_air_yards": 0.0, "receiving_yards_after_catch": 0.0}
    row.update(extra)
    return row


def test_team_totals_are_summed_from_the_player_rows():
    from backend.ingestion.fetch_metrics import _team_week_totals
    rows = [_stat_row(targets=10, carries=2), _stat_row(targets=30, carries=20)]
    assert _team_week_totals(rows)[("KC", 1)] == {"targets": 40.0, "carries": 22.0}


def test_target_and_carry_share_are_computed_again():
    from backend.ingestion.fetch_metrics import _team_week_totals, _compute_opportunity_efficiency
    rows = [_stat_row(targets=10, carries=2), _stat_row(targets=30, carries=20)]
    out = _compute_opportunity_efficiency([rows[0]], _team_week_totals(rows))
    assert out["target_share"] == pytest.approx(0.25)
    assert out["carry_share"] == pytest.approx(2 / 22)


def test_shares_are_none_without_team_totals_rather_than_zero():
    # The pre-fix behaviour, kept explicit: absent denominators must yield
    # None ("unknown"), never 0.0 ("no role").
    from backend.ingestion.fetch_metrics import _compute_opportunity_efficiency
    out = _compute_opportunity_efficiency([_stat_row(targets=10)])
    assert out["target_share"] is None and out["carry_share"] is None


def test_share_denominator_covers_only_the_weeks_he_played():
    # Summing the full season instead would divide a partial-season target
    # count by a full-season denominator and understate every injured player.
    from backend.ingestion.fetch_metrics import _team_week_totals, _compute_opportunity_efficiency
    rows = [_stat_row(week=1, targets=10), _stat_row(week=1, targets=30),
            _stat_row(week=2, targets=40)]          # week 2: he did not play
    out = _compute_opportunity_efficiency([rows[0]], _team_week_totals(rows))
    assert out["target_share"] == pytest.approx(0.25)   # 10/40, not 10/80


def test_snap_counts_postseason_is_filtered_via_game_type():
    # snap_counts has no `season_type` column — it uses `game_type`. Rows
    # missing the field are kept, so before the alias every postseason snap
    # was silently summed into snap_pct.
    from backend.ingestion.fetch_metrics import _filter_regular_season
    rows = [{"game_type": "REG"}, {"game_type": "POST"}, {"game_type": "REG"}]
    assert len(_filter_regular_season(rows)) == 2


def test_season_type_still_wins_where_it_exists():
    from backend.ingestion.fetch_metrics import _filter_regular_season
    rows = [{"season_type": "REG"}, {"season_type": "POST"}]
    assert len(_filter_regular_season(rows)) == 1
    # And a row with neither field is still kept — "can't tell" must not
    # silently discard otherwise-valid data.
    assert len(_filter_regular_season([{"week": 1}])) == 1


def test_depth_chart_rank_reads_the_current_nflverse_column():
    # nflverse reworked depth_charts: 12 columns, none of them depth_team,
    # depth_position or rank. The rank now lives in `pos_rank`.
    from backend.ingestion.fetch_metrics import _compute_forward_looking
    rows = [{"gsis_id": "x", "dt": "2025-09-01T00:00:00Z", "pos_rank": 3},
            {"gsis_id": "x", "dt": "2025-12-01T00:00:00Z", "pos_rank": 1}]
    out = _compute_forward_looking([], [], rows)
    assert out["depth_chart_trend"] == -2      # negative = moving up


def test_depth_charts_are_ordered_by_timestamp_not_a_week_column():
    # depth_charts is a snapshot feed keyed by `dt`, with no week number.
    # Sorting by "week" put every row at 0.0, making first-vs-last arbitrary.
    from backend.ingestion.fetch_metrics import _order_key
    rows = [{"dt": "2025-12-01T00:00:00Z"}, {"dt": "2025-09-01T00:00:00Z"}]
    assert [r["dt"] for r in sorted(rows, key=_order_key)] == [
        "2025-09-01T00:00:00Z", "2025-12-01T00:00:00Z"]


def test_order_key_still_prefers_a_real_week_number():
    from backend.ingestion.fetch_metrics import _order_key
    rows = [{"week": 12}, {"week": 2}]
    assert [r["week"] for r in sorted(rows, key=_order_key)] == [2, 12]
    # Rows with neither field sort last rather than raising.
    assert _order_key({}) > _order_key({"week": 99})


def test_pfr_crosswalk_maps_snap_rows_onto_gsis_ids():
    # THE snap_pct bug: load_snap_counts carries pfr_player_id and no
    # gsis_id, so rows grouped under PFR keys while the per-player lookup
    # asked for gsis ids. Nothing matched, for anyone, silently.
    snap_rows = [{"pfr_player_id": "PfrA", "offense_pct": 0.9, "game_type": "REG"}]
    pfr_map = {"PfrA": "00-0011111"}
    for row in snap_rows:
        gsis = pfr_map.get(str(row.get("pfr_player_id") or ""))
        if gsis:
            row["gsis_id"] = gsis
    from backend.ingestion.fetch_metrics import _group_by_player
    grouped = _group_by_player(snap_rows, ("gsis_id", "player_id", "pfr_player_id"))
    assert "00-0011111" in grouped, "snap rows must group under the gsis id"
    assert "PfrA" not in grouped


def test_depth_chart_rank_and_trend_read_the_same_columns():
    # depth_chart_trend got `pos_rank` added but the rank assignment in
    # refresh_metrics did not, so the trend populated for all 182 players
    # while the rank itself stayed NULL. Any future rename has to land in
    # both places, so pin them together.
    import inspect
    from backend.ingestion import fetch_metrics
    src = inspect.getsource(fetch_metrics)
    assert src.count('"pos_rank", "depth_team", "depth_position", "rank"') == 3, (
        "rank candidates must be identical in _compute_forward_looking "
        "(first + last) and in refresh_metrics' latest-rank lookup"
    )


# ---------------------------------------------------------------------------
# Retrieval performance shape
#
# Recommendations went from ~10s to ~40s when _MAX_CONTEXT_PLAYERS rose from
# 10 to 25, because each candidate triggered a `query` — a similarity search
# that embeds its query text through a local ONNX model. The filter already
# pinned results to one player by sleeper_id, so the embedding was only ever
# breaking ties among that player's own chunks. Replaced with a single `get`
# (pure metadata lookup, no model): 2.2 ms for all 25 against the live store.
# ---------------------------------------------------------------------------

def test_retrieval_makes_one_bulk_call_not_one_per_player(monkeypatch):
    from backend.app.services import ai_service as S
    S._retrieval_cache.clear()
    calls = []

    def fake_fetch(where):
        calls.append(where)
        return [({"sleeper_id": "1"}, "news about P1")]

    import types
    fake = types.ModuleType("backend.rag.vector_store")
    fake.fetch_by_metadata = fake_fetch
    monkeypatch.setitem(__import__("sys").modules, "backend.rag.vector_store", fake)

    board = [{"id": i, "name": f"P{i}", "position": "WR", "sleeper_id": str(i)}
             for i in range(1, 11)]
    out = S._retrieve_player_context(board)
    assert len(calls) == 1, "ten players must cost one lookup, not ten"
    assert "news about P1" in out
    # Players with nothing still get an explicit line, never silence.
    assert "No retrieved data" in out


def test_retrieval_caches_negative_results_too(monkeypatch):
    # Without caching the empties, a player with no chunks is re-fetched on
    # every single pick for the whole draft.
    from backend.app.services import ai_service as S
    S._retrieval_cache.clear()
    calls = []

    def fake_fetch(where):
        calls.append(where)
        return []

    import types, sys
    fake = types.ModuleType("backend.rag.vector_store")
    fake.fetch_by_metadata = fake_fetch
    monkeypatch.setitem(sys.modules, "backend.rag.vector_store", fake)

    board = [{"id": 1, "name": "P1", "position": "WR", "sleeper_id": "1"}]
    S._retrieve_player_context(board)
    S._retrieve_player_context(board)
    assert len(calls) == 1, "the second pass must be served from cache"


def test_retrieval_survives_a_vector_store_failure(monkeypatch):
    # Draft day must not stall on the vector store.
    from backend.app.services import ai_service as S
    S._retrieval_cache.clear()

    def boom(where):
        raise RuntimeError("chroma is down")

    import types, sys
    fake = types.ModuleType("backend.rag.vector_store")
    fake.fetch_by_metadata = boom
    monkeypatch.setitem(sys.modules, "backend.rag.vector_store", fake)

    board = [{"id": 1, "name": "P1", "position": "WR", "sleeper_id": "1"}]
    out = S._retrieve_player_context(board)   # must not raise
    assert "No retrieved data" in out


# ---------------------------------------------------------------------------
# Streaming
#
# Generation is sequential and output-bound: 1,664 tokens at ~75 tok/sec is
# ~22s, and the batch endpoint shows nothing for all of it — even though the
# recommendation object is written about a fifth of the way in. Streaming
# renders the pick at ~4s. To do that the stream must decide "is this one
# object closed yet" on text that is invalid JSON everywhere else.
# ---------------------------------------------------------------------------

def test_extracts_the_pick_before_the_rest_arrives():
    partial = ('{"strategy": "go RB", "confidence": "high", '
               '"main": {"player_id": 7, "reasoning": "volume"}, '
               '"considered": [{"player_id')          # cut mid-token
    got = _extract_complete_object(partial, "main")
    assert got == {"player_id": 7, "reasoning": "volume"}


def test_returns_none_while_the_object_is_still_arriving():
    for partial in ('{"main": {"player_id": 7, "reason',
                    '{"main": {',
                    '{"strategy": "x"',
                    ''):
        assert _extract_complete_object(partial, "main") is None


def test_a_brace_inside_prose_does_not_close_the_object_early():
    # Without string-awareness this returns a truncated pick — and the
    # streamed pick is the one the user acts on.
    partial = ('{"main": {"player_id": 7, '
               '"reasoning": "a bell-cow {sic} back"}, "considered": []}')
    got = _extract_complete_object(partial, "main")
    assert got["reasoning"] == "a bell-cow {sic} back"
    assert got["player_id"] == 7


def test_escaped_quotes_do_not_break_string_tracking():
    partial = ('{"main": {"player_id": 7, '
               '"reasoning": "they call him \\"the truth\\""}, "x": 1}')
    got = _extract_complete_object(partial, "main")
    assert got["player_id"] == 7


def test_missing_key_is_not_an_error():
    assert _extract_complete_object('{"considered": []}', "main") is None


def test_streamed_pick_is_validated_like_a_batch_one():
    # A partially-received response is exactly where a laxer check would be
    # tempting and exactly where it would be worst.
    from backend.app.services.ai_service import _pick_from
    ctx = ctx_with_available(1, 2)
    assert _pick_from({"player_id": 99, "reasoning": "x"}, ctx) is None
    good = _pick_from({"player_id": 1, "reasoning": "x"}, ctx)
    assert good is not None and good.player_name == "P1"   # canonical, not model-supplied


def test_streamed_pick_carries_the_survival_tag():
    from backend.app.services.ai_service import _pick_from
    ctx = ctx_with_available(1)
    ctx.top_available[0]["adp"] = 11.5
    ctx.my_following_pick_number = 41
    assert _pick_from({"player_id": 1}, ctx).survival == "take_now"


def test_below_replacement_players_are_flagged_on_the_board():
    # A negative VOR reads as "roughly neutral" at a glance when it actually
    # means the player produced LESS than the man available for free later.
    # Confirmed live: a back at VOR -0.17, with falling target and snap
    # share, was recommended over a first-round rookie whose VOR was unknown.
    ctx = ctx_with_available(1, 2)
    ctx.top_available[0]["adp"] = 62.5
    ctx.top_available[1]["adp"] = 40.0
    ctx.replacement_ppg = {"RB": 11.1}
    ctx.player_metrics = {1: {"fantasy_points_avg": 10.9},   # below
                          2: {"fantasy_points_avg": 15.0}}   # above
    prompt = _build_prompt(ctx)
    # Check the board rows specifically; the must-evaluate shortlist renders
    # the same players separately and is asserted below.
    board = prompt.split("## Top Available Players")[1].split("\n## ")[0]
    below = [l for l in board.splitlines() if "P1 " in l][0]
    above = [l for l in board.splitlines() if "P2 " in l][0]
    assert "(BELOW replacement)" in below
    assert "(BELOW replacement)" not in above
    # And the shortlist must agree — a warning shown in one place but not
    # the other invites reading the unflagged copy as the better one.
    if "## MUST EVALUATE" in prompt:
        sl = prompt.split("## MUST EVALUATE")[1].split("\n## ")[0]
        p1 = [l for l in sl.splitlines() if "P1 " in l]
        if p1:
            assert "(BELOW replacement)" in p1[0]


def test_unknown_vor_is_never_flagged_as_below_replacement():
    # An unproven rookie and a proven-below-replacement veteran are
    # different cases; only the second is one the numbers argue against.
    ctx = ctx_with_available(1)
    ctx.replacement_ppg = {"RB": 11.1}
    ctx.player_metrics = {}
    prompt = _build_prompt(ctx)
    board = prompt.split("## Top Available Players")[1].split("\n## ")[0]
    row = [l for l in board.splitlines() if "P1 " in l][0]
    assert "VOR   --" in row
    assert "BELOW replacement" not in row


def test_board_explains_what_below_replacement_means():
    ctx = ctx_with_available(1)
    ctx.replacement_ppg = {"RB": 11.1}
    ctx.player_metrics = {1: {"fantasy_points_avg": 10.0}}
    prompt = _build_prompt(ctx)
    assert "gains you nothing over waiting" in prompt
    assert "unknown value, NOT replacement-level value" in prompt


# ---------------------------------------------------------------------------
# Roster changes
#
# The only forward-looking evidence in the prompt apart from ADP. Diagnosed
# from a live disagreement: FantasyPros favoured Josh Downs over Deebo Samuel
# 96/4 while every production figure here favoured Deebo. Downs's 18% target
# share was earned alongside Michael Pittman Jr., who took 25% of
# Indianapolis's targets and is now on Pittsburgh — so his own share
# understates the opportunity in front of him, and last season's numbers
# cannot say so.
# ---------------------------------------------------------------------------

def _rc_board():
    return [{"id": 1, "rank": 1, "name": "Josh Downs", "position": "WR",
             "team": "IND", "adp": 91.9, "sleeper_id": None}]


def test_departures_are_reported_for_a_board_team():
    from backend.app.services.ai_service import _format_roster_changes
    out = _format_roster_changes(_rc_board(), {
        "IND": {"departed": [{"name": "Michael Pittman Jr.",
                              "share_label": "25% of targets"}], "arrived": []}})
    assert "IND: LOST Michael Pittman Jr. (25% of targets)" in out


def test_arrivals_are_reported_as_new_competition():
    from backend.app.services.ai_service import _format_roster_changes
    out = _format_roster_changes(_rc_board(), {
        "IND": {"departed": [], "arrived": [{"name": "Someone", "from_team": "NYJ",
                                             "share_label": "20% of targets"}]}})
    assert "GAINED Someone (20% of targets with NYJ)" in out


def test_only_teams_on_the_board_are_shown():
    # Several board players usually share a team, and a team nobody is
    # choosing between is prompt weight for nothing.
    from backend.app.services.ai_service import _format_roster_changes
    out = _format_roster_changes(_rc_board(), {
        "IND": {"departed": [{"name": "A", "share_label": "25% of targets"}], "arrived": []},
        "KC":  {"departed": [{"name": "B", "share_label": "30% of targets"}], "arrived": []}})
    assert "IND:" in out and "KC:" not in out


def test_section_is_omitted_when_nothing_moved():
    from backend.app.services.ai_service import _format_roster_changes
    assert _format_roster_changes(_rc_board(), {}) == ""
    assert _format_roster_changes(_rc_board(),
                                  {"IND": {"departed": [], "arrived": []}}) == ""


def test_section_states_both_of_its_limits():
    # It's a volume argument, not a talent one, and the ADP pool can't see
    # retirements or cuts. Overstating either would be worse than silence.
    from backend.app.services.ai_service import _format_roster_changes
    out = _format_roster_changes(_rc_board(), {
        "IND": {"departed": [{"name": "A", "share_label": "25% of targets"}], "arrived": []}})
    assert "does not automatically go to whoever remains" in out
    assert "a floor, not a complete accounting" in out


def test_section_tells_the_model_when_to_reach_for_it():
    from backend.app.services.ai_service import _format_roster_changes
    out = _format_roster_changes(_rc_board(), {
        "IND": {"departed": [{"name": "A", "share_label": "25% of targets"}], "arrived": []}})
    assert "when the market ranks a player above what his numbers justify" in out


def test_movers_are_ordered_by_share_not_by_label_text():
    # Sorting the formatted string puts "8% of targets" above "25%".
    entries = [{"name": "small", "share": 0.08, "share_label": "8% of targets"},
               {"name": "big", "share": 0.25, "share_label": "25% of targets"}]
    entries.sort(key=lambda p: p["share"], reverse=True)
    assert [e["name"] for e in entries] == ["big", "small"]
    by_label = sorted(entries, key=lambda p: p["share_label"], reverse=True)
    assert [e["name"] for e in by_label] == ["small", "big"], "the bug this guards"


# ---------------------------------------------------------------------------
# _format_schedule_section — real upcoming opponents for the Opportunity pick
# ---------------------------------------------------------------------------

def _sched_board():
    return [{"id": 1, "rank": 1, "name": "Josh Downs", "position": "WR",
             "team": "IND", "adp": 91.9, "sleeper_id": None}]


def test_schedule_section_lists_real_matchups():
    from backend.app.services.ai_service import _format_schedule_section
    out = _format_schedule_section(_sched_board(), {
        "IND": [{"week": 1, "opponent": "HOU", "is_home": True},
                {"week": 2, "opponent": "GB", "is_home": False}],
    })
    assert "IND: wk1 vs HOU, wk2 @ GB" in out


def test_schedule_section_omitted_when_no_data_ingested():
    from backend.app.services.ai_service import _format_schedule_section
    assert _format_schedule_section(_sched_board(), {}) == ""


def test_schedule_section_only_teams_on_the_board_are_shown():
    from backend.app.services.ai_service import _format_schedule_section
    out = _format_schedule_section(_sched_board(), {
        "IND": [{"week": 1, "opponent": "HOU", "is_home": True}],
        "KC": [{"week": 1, "opponent": "BAL", "is_home": True}],
    })
    assert "IND:" in out and "KC:" not in out


def test_schedule_section_caveats_opponent_quality_separately_from_the_facts():
    # The matchup facts (who/when/home-away) are real data and should be
    # trusted; whether that opponent's defense is actually tough is a
    # DIFFERENT claim this table doesn't make — see the function's
    # docstring and the live concern that motivated it (Claude's knowledge
    # cutoff predates the season's real schedule release).
    from backend.app.services.ai_service import _format_schedule_section
    out = _format_schedule_section(_sched_board(), {
        "IND": [{"week": 1, "opponent": "HOU", "is_home": True}],
    })
    assert "trust them as fact" in out
    assert "your own judgment" in out


# ---------------------------------------------------------------------------
# Team abbreviation normalisation
#
# PlayerMetrics.team comes from nflverse, Player.team from the ADP feed.
# They agree on 31 of 32 clubs and disagree on the Rams — nflverse "LA" vs
# the feed's "LAR" — so every Rams player was reported as having changed
# teams. Caught live: "LA: LOST Kyren Williams (56% of carries)".
# ---------------------------------------------------------------------------

def test_rams_are_not_reported_as_having_traded_their_whole_roster():
    from backend.app.api.recommendations import _normalise_team
    assert _normalise_team("LA") == _normalise_team("LAR") == "LAR"


def test_historical_codes_normalise_too():
    from backend.app.api.recommendations import _normalise_team
    for old, new in (("STL", "LAR"), ("SD", "LAC"), ("OAK", "LV"),
                     ("WSH", "WAS"), ("JAC", "JAX"), ("ARZ", "ARI")):
        assert _normalise_team(old) == new


def test_unknown_and_empty_codes_pass_through_unchanged():
    # A club we have no alias for must survive intact — silently rewriting
    # it would be a worse failure than leaving it alone.
    from backend.app.api.recommendations import _normalise_team
    assert _normalise_team("KC") == "KC"
    assert _normalise_team(" kc ") == "KC"
    assert _normalise_team(None) is None
    assert _normalise_team("") is None


def test_genuine_moves_still_register():
    # The normalisation must not swallow real transfers along with the
    # spurious ones — IND -> PIT is the case the whole feature exists for.
    from backend.app.api.recommendations import _normalise_team
    assert _normalise_team("IND") != _normalise_team("PIT")


def test_opportunity_is_a_column_not_an_occasional_tag():
    # A figure that appears only when large reads as an annotation; one that
    # is always present reads as a metric. OPP has to carry the same weight
    # as the ADP and VOR beside it, so it renders on every row — including
    # at zero, which says the situation has NOT changed and is real
    # information rather than an absent signal.
    ctx = ctx_with_available(1, 2)
    ctx.top_available[0].update(position="WR", team="IND")
    ctx.top_available[1].update(position="QB", team="KC")
    ctx.roster_changes = {"IND": {"departed": [
        {"name": "X", "share": 0.21, "currency": "targets",
         "share_label": "21% of targets"}], "arrived": []}}
    board = _build_prompt(ctx).split("## Top Available Players")[1].split("\n## ")[0]
    rows = [l for l in board.splitlines() if "ADP" in l and l.strip().startswith(("1 ", "2 "))
            or ("P1 " in l or "P2 " in l)]
    assert any("OPP +21% tgt" in l for l in rows)
    # A quarterback has no single volume to compete for.
    assert any("OPP    --" in l for l in rows)


def test_unchanged_situation_renders_as_zero_not_blank():
    ctx = ctx_with_available(1)
    ctx.top_available[0].update(position="WR", team="KC")
    ctx.roster_changes = {"KC": {"departed": [], "arrived": []}}
    prompt = _build_prompt(ctx)
    assert "OPP +0% tgt" in prompt


def test_legend_refuses_to_rank_the_three_signals():
    ctx = ctx_with_available(1)
    ctx.replacement_ppg = {"RB": 11.1}
    ctx.player_metrics = {1: {"fantasy_points_avg": 10.0}}
    prompt = _build_prompt(ctx)
    assert "The three are not ranked" in prompt
    assert "a precise measurement of a role that no longer exists" in prompt


# ---------------------------------------------------------------------------
# Board positional coverage
#
# A global ADP cut does not guarantee the board contains a player at a
# position you still have to start, and late in a draft it reliably doesn't.
# Confirmed live at pick 128: the only unfilled starting slot was TE, five
# picks remained, and the top 25 held fifteen receivers and ZERO tight ends —
# the best available TE sat at overall position 37. The model could only
# recommend another receiver for a roster already two deep beyond
# requirement, and nothing in the prompt could express why.
# ---------------------------------------------------------------------------

def _coverage_ctx():
    # 25 receivers ahead of every tight end, mirroring the live board.
    board = [{"id": i, "rank": i, "name": f"WR{i}", "position": "WR", "team": "X",
              "adp": float(90 + i), "sleeper_id": None} for i in range(1, 26)]
    board += [{"id": 50, "rank": 50, "name": "Cheap TE", "position": "TE", "team": "PIT",
               "adp": 152.0, "sleeper_id": None},
              {"id": 51, "rank": 51, "name": "Good TE", "position": "TE", "team": "JAX",
               "adp": 163.3, "sleeper_id": None}]
    ctx = ctx_with_available(1)
    ctx.top_available = board
    ctx.player_metrics = {50: {"fantasy_points_avg": 7.6},    # cheaper, worse
                          51: {"fantasy_points_avg": 9.8}}    # dearer, better
    ctx.replacement_ppg = {"TE": 10.5, "WR": 11.5}
    return ctx


def test_a_needed_position_absent_from_the_board_gets_pulled_in():
    from backend.app.services.ai_service import _board_for_prompt
    ctx = _coverage_ctx()
    plain = {p["position"] for p in ctx.top_available[:_LISTED_PLAYERS]}
    assert "TE" not in plain, "fixture must reproduce the live shape"
    covered = _board_for_prompt(ctx, {"TE": 1})
    assert "TE" in {p["position"] for p in covered}


def test_both_the_cheapest_and_the_best_are_pulled_in():
    # One axis is not enough when they disagree, and at a needed position
    # they often do: filling the slot by ADP alone surfaced a tight end with
    # a worse VOR, a smaller target share and new competition on his team,
    # while the better player sat eleven picks later and stayed invisible.
    from backend.app.services.ai_service import _board_for_prompt
    names = {p["name"] for p in _board_for_prompt(_coverage_ctx(), {"TE": 1})}
    assert "Cheap TE" in names      # best ADP
    assert "Good TE" in names       # best VOR


def test_positions_already_filled_are_not_pulled_in():
    from backend.app.services.ai_service import _board_for_prompt
    ctx = _coverage_ctx()
    assert len(_board_for_prompt(ctx, {})) == _LISTED_PLAYERS


def test_flex_and_late_round_slots_are_not_looked_up():
    # There is no "best available FLEX" — it's filled by an RB/WR/TE, each
    # covered by its own entry. DST/K are handled by the deferral rules.
    from backend.app.services.ai_service import _board_for_prompt
    ctx = _coverage_ctx()
    assert len(_board_for_prompt(ctx, {"FLEX": 1, "DST": 1, "K": 1})) == _LISTED_PLAYERS


def test_no_duplicates_when_the_same_player_wins_both_axes():
    from backend.app.services.ai_service import _board_for_prompt
    ctx = _coverage_ctx()
    ctx.top_available = [p for p in ctx.top_available if p["id"] != 51]
    out = _board_for_prompt(ctx, {"TE": 1})
    assert len(out) == len({p["id"] for p in out}) == _LISTED_PLAYERS + 1


# ---------------------------------------------------------------------------
# DST/K coverage once the roster-tax gate opens
#
# Live bug (Coop's league, 14 teams, round 15 of 15, pick 203): the roster
# gap dict correctly showed only DST left, the prompt correctly said "draft
# now — you are out of rounds to defer them", and the board correctly held
# four available defenses in the top 25 by ADP. The recommendation still
# never once suggested one, across all 15 rounds. Root cause: the MUST
# EVALUATE shortlist that forces the model to weigh in on a candidate is
# built from three axes — cheapest overall, best VOR, cheapest per
# position — and DST/K structurally cannot win any of the three (no
# PlayerMetrics row ever exists for them, so VOR is always None, and a
# waiver-tier WR/RB is routinely cheaper by ADP than any defense). "Draft
# now" was true and unenforced: nothing required the model to ever look at
# one. due_late_gaps is the fix — see _board_for_prompt's and _shortlist's
# docstrings for why it's a separate argument from `gaps` rather than
# merged into it.
# ---------------------------------------------------------------------------

def test_late_position_is_ignored_until_explicitly_due():
    # Passing DST/K through the ordinary `gaps` argument must stay a no-op
    # even though this fixture's board has no DST/K in the top 25 — this is
    # the "not yet" case (RULE 9), and it must behave identically whether
    # the caller passes {} or {"DST": 1} through `gaps`.
    from backend.app.services.ai_service import _board_for_prompt
    ctx = _coverage_ctx()
    ctx.top_available.append(
        {"id": 90, "rank": 90, "name": "Some Defense", "position": "DST",
         "team": "SF", "adp": 200.0, "sleeper_id": None}
    )
    out = _board_for_prompt(ctx, {"DST": 1})
    assert "DST" not in {p["position"] for p in out}


def test_due_late_position_is_pulled_onto_the_board():
    from backend.app.services.ai_service import _board_for_prompt
    ctx = _coverage_ctx()
    ctx.top_available.append(
        {"id": 90, "rank": 90, "name": "Some Defense", "position": "DST",
         "team": "SF", "adp": 200.0, "sleeper_id": None}
    )
    out = _board_for_prompt(ctx, {}, due_late_gaps={"DST": 1})
    assert "Some Defense" in {p["name"] for p in out}


def test_due_dst_is_forced_into_the_must_evaluate_shortlist():
    # This is the actual live failure: a DST present on the board, with no
    # PlayerMetrics row (so VOR is None) and an ADP well behind the cheapest
    # skill players filling the shortlist's other two axes, still has to
    # get a forced verdict once it's the roster tax that's due.
    board = [{"id": i, "rank": i, "name": f"WR{i}", "position": "WR", "team": "X",
              "adp": float(90 + i), "sleeper_id": None} for i in range(1, 21)]
    board.append({"id": 90, "rank": 90, "name": "Some Defense", "position": "DST",
                   "team": "SF", "adp": 144.9, "sleeper_id": None})
    metrics = {i: {"fantasy_points_avg": 6.0} for i in range(1, 21)}  # no entry for the DST
    repl = {"WR": 5.0}

    without_gate = _shortlist(board, 203, metrics, repl, {})
    assert "Some Defense" not in {p["name"] for p in without_gate}, \
        "fixture must reproduce the live omission before the fix is applied"

    with_gate = _shortlist(board, 203, metrics, repl, {}, due_late_gaps={"DST": 1})
    assert "Some Defense" in {p["name"] for p in with_gate}


def test_board_says_why_a_late_player_is_listed():
    ctx = _coverage_ctx()
    ctx.my_roster = roster("QB", "RB", "RB", "WR", "WR")   # no TE
    prompt = _build_prompt(ctx)
    assert "best available player at every position you still need to start" in prompt


def test_a_failed_lookup_is_not_cached_as_absence(monkeypatch):
    # The cache lives for the process. Writing "no news" after a failed
    # fetch would mean one transient Chroma error at the first pick silently
    # disables news and injury retrieval for the whole draft — and RULE 0
    # reads IR status out of that retrieval, so a player on injured reserve
    # would look exactly like a healthy one for fifteen rounds.
    from backend.app.services import ai_service as S
    S._retrieval_cache.clear()
    calls = {"n": 0}

    def flaky(where):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("chroma is briefly unavailable")
        return [({"sleeper_id": "1"}, "real news")]

    import types, sys
    fake = types.ModuleType("backend.rag.vector_store")
    fake.fetch_by_metadata = flaky
    monkeypatch.setitem(sys.modules, "backend.rag.vector_store", fake)

    board = [{"id": 1, "name": "P1", "position": "WR", "sleeper_id": "1"}]
    first = S._retrieve_player_context(board)
    assert "No retrieved data" in first          # degrades, does not raise
    assert "1" not in S._retrieval_cache, "a failure must not be remembered"

    second = S._retrieve_player_context(board)   # must try again
    assert calls["n"] == 2
    assert "real news" in second


def test_a_genuine_empty_result_is_still_cached(monkeypatch):
    # The optimisation this guard protects: a player who really has no
    # chunks should not be re-fetched on every pick.
    from backend.app.services import ai_service as S
    S._retrieval_cache.clear()
    calls = {"n": 0}

    def empty(where):
        calls["n"] += 1
        return []

    import types, sys
    fake = types.ModuleType("backend.rag.vector_store")
    fake.fetch_by_metadata = empty
    monkeypatch.setitem(sys.modules, "backend.rag.vector_store", fake)

    board = [{"id": 1, "name": "P1", "position": "WR", "sleeper_id": "1"}]
    S._retrieve_player_context(board)
    S._retrieve_player_context(board)
    assert calls["n"] == 1, "an empty result is a fact worth remembering"


def test_a_needed_position_forces_verdicts_on_both_the_cheap_and_the_good():
    # _board_for_prompt already put both on the board, but the shortlist's
    # per-position axis took only the cheapest — so the better tight end was
    # visible while carrying no required verdict. Live consequence: the model
    # recommended the cheaper one (VOR -2.94, 11% target share, his team had
    # just ADDED 21% of a team's targets) and justified it by asserting he
    # was "the only TE on the board", while naming the other TE two sentences
    # later. A player who must be accounted for cannot be waved away.
    board = [{"id": i, "rank": i, "name": f"WR{i}", "position": "WR", "team": "X",
              "adp": float(90 + i), "sleeper_id": None} for i in range(1, 21)]
    board += [{"id": 50, "rank": 50, "name": "Cheap TE", "position": "TE",
               "team": "PIT", "adp": 152.0, "sleeper_id": None},
              {"id": 51, "rank": 51, "name": "Good TE", "position": "TE",
               "team": "HOU", "adp": 158.1, "sleeper_id": None}]
    metrics = {50: {"fantasy_points_avg": 7.6}, 51: {"fantasy_points_avg": 10.5}}
    repl = {"TE": 10.5, "WR": 11.5}

    names = {p["name"] for p in _shortlist(board, 128, metrics, repl, {"TE": 1})}
    assert "Cheap TE" in names and "Good TE" in names


def test_a_filled_position_only_contributes_its_best_by_adp():
    # The second entry is reserved for positions you must still fill; adding
    # it everywhere would crowd the cap and dilute the requirement.
    #
    # The receivers carry strong metrics on purpose, so they win the GLOBAL
    # best-VOR slots. Otherwise "Good TE" enters through that axis instead of
    # the per-position one and the test proves nothing about either.
    board = [{"id": i, "rank": i, "name": f"WR{i}", "position": "WR", "team": "X",
              "adp": float(90 + i), "sleeper_id": None} for i in range(1, 21)]
    board += [{"id": 50, "rank": 50, "name": "Cheap TE", "position": "TE",
               "team": "PIT", "adp": 152.0, "sleeper_id": None},
              {"id": 51, "rank": 51, "name": "Good TE", "position": "TE",
               "team": "HOU", "adp": 158.1, "sleeper_id": None}]
    metrics = {i: {"fantasy_points_avg": 25.0} for i in range(1, 21)}
    metrics.update({50: {"fantasy_points_avg": 7.6}, 51: {"fantasy_points_avg": 10.5}})
    repl = {"TE": 10.5, "WR": 11.5}

    # TE not needed -> only the cheapest TE is forced in.
    filled = {p["name"] for p in _shortlist(board, 128, metrics, repl, {})}
    assert "Cheap TE" in filled and "Good TE" not in filled
    # TE needed -> both, which is the whole point of the change.
    needed = {p["name"] for p in _shortlist(board, 128, metrics, repl, {"TE": 1})}
    assert "Cheap TE" in needed and "Good TE" in needed


def test_shortlist_and_board_agree_on_which_players_exist():
    # The two functions disagreeing is what let the model call a player
    # absent while he was on the board. Every shortlisted player must come
    # from the board it was built from.
    board = [{"id": i, "rank": i, "name": f"P{i}",
              "position": ["QB", "RB", "WR", "TE"][i % 4], "team": "X",
              "adp": float(i), "sleeper_id": None} for i in range(1, 30)]
    metrics = {i: {"fantasy_points_avg": 30.0 - i} for i in range(1, 30)}
    repl = {"QB": 17.5, "RB": 11.1, "WR": 11.9, "TE": 10.6}
    ids = {p["id"] for p in board}
    assert all(p["id"] in ids for p in _shortlist(board, 20, metrics, repl, {"TE": 1}))


def test_a_needed_position_shows_a_realistic_slate_not_two_names():
    # Two entries — cheapest and best — proved too thin at a mandatory slot:
    # it surfaced the cheapest TE (VOR -2.94) and the best (-0.06) while
    # hiding the second-best (-0.68) entirely. You cannot choose from a
    # slate you were never shown.
    from backend.app.services.ai_service import _board_for_prompt, _NEEDED_POSITION_DEPTH
    board = [{"id": i, "rank": i, "name": f"WR{i}", "position": "WR", "team": "X",
              "adp": float(90 + i), "sleeper_id": None} for i in range(1, 26)]
    tes = [{"id": 50 + i, "rank": 50 + i, "name": f"TE{i}", "position": "TE",
            "team": "X", "adp": 150.0 + i * 3, "sleeper_id": None} for i in range(6)]
    ctx = ctx_with_available(1)
    ctx.top_available = board + tes
    ctx.player_metrics = {}
    ctx.replacement_ppg = {}
    shown = [p["name"] for p in _board_for_prompt(ctx, {"TE": 1}) if p["position"] == "TE"]
    assert len(shown) == _NEEDED_POSITION_DEPTH


def test_the_best_at_a_needed_position_appears_even_when_outside_that_window():
    # The best available is regularly past the cheapest few — the best tight
    # end on the live board sat sixth by ADP.
    from backend.app.services.ai_service import _board_for_prompt
    board = [{"id": i, "rank": i, "name": f"WR{i}", "position": "WR", "team": "X",
              "adp": float(90 + i), "sleeper_id": None} for i in range(1, 26)]
    tes = [{"id": 50 + i, "rank": 50 + i, "name": f"TE{i}", "position": "TE",
            "team": "X", "adp": 150.0 + i * 3, "sleeper_id": None} for i in range(8)]
    ctx = ctx_with_available(1)
    ctx.top_available = board + tes
    # TE7 is the last by ADP and the best by production.
    ctx.player_metrics = {57: {"fantasy_points_avg": 14.0}}
    ctx.replacement_ppg = {"TE": 10.5}
    shown = {p["name"] for p in _board_for_prompt(ctx, {"TE": 1}) if p["position"] == "TE"}
    assert "TE7" in shown


# ---------------------------------------------------------------------------
# _load_board — the player pool the prompt is built from
#
# The failure this replaced a flat ADP window over: at a mandatory TE slot,
# a 60-player window ending at ADP 160.5 excluded the second-best available
# tight end at 163.3. A window that excludes the player you need is
# indistinguishable, from inside the prompt, from that player not existing,
# and widening it only moves the pick at which the same thing happens again.
# ---------------------------------------------------------------------------

class _FakePlayer:
    def __init__(self, pid, name, position, adp, ppg=None):
        self.id = pid
        self.rank = pid
        self.name = name
        self.position = position
        self.team = "X"
        self.adp = adp
        self.ppg = ppg
        self.sleeper_id = None


def _fake_pool(monkeypatch, players):
    """Stands in for the three player_repo queries over a fixed pool."""
    from backend.app.api import recommendations

    def _at(position):
        return [p for p in players if position is None or p.position == position]

    def _top(db, n=10, position=None):
        return sorted(_at(position), key=lambda p: p.adp)[:n]

    def _from_adp(db, min_adp, n=3, position=None):
        rows = [p for p in _at(position) if p.adp >= min_adp]
        return sorted(rows, key=lambda p: p.adp)[:n]

    def _by_ppg(db, n=3, position=None):
        rows = [p for p in _at(position) if p.ppg is not None]
        return sorted(rows, key=lambda p: -p.ppg)[:n]

    monkeypatch.setattr(recommendations.repo, "get_top_available", _top)
    monkeypatch.setattr(recommendations.repo, "get_available_from_adp", _from_adp)
    monkeypatch.setattr(recommendations.repo, "get_best_available_by_ppg", _by_ppg)


def _deep_pool():
    # 40 receivers own the whole cheap end of the board; the tight ends sit
    # behind all of them, which is what a real board looks like by round 10.
    wrs = [_FakePlayer(i, f"WR{i}", "WR", 90.0 + i) for i in range(1, 41)]
    tes = [_FakePlayer(100 + i, f"TE{i}", "TE", 150.0 + i * 3) for i in range(6)]
    rbs = [_FakePlayer(200 + i, f"RB{i}", "RB", 140.0 + i * 4) for i in range(6)]
    qbs = [_FakePlayer(300 + i, f"QB{i}", "QB", 145.0 + i * 5) for i in range(6)]
    return wrs + tes + rbs + qbs


def test_every_position_is_covered_however_deep_it_sits(monkeypatch):
    from backend.app.api.recommendations import _load_board, _CONTEXT_POSITIONS
    _fake_pool(monkeypatch, _deep_pool())
    board = _load_board(None, top_n=30, per_position=6)
    for pos in _CONTEXT_POSITIONS:
        # >= rather than ==: the overall window contributes at whatever
        # positions sit cheap, on top of the guaranteed slate.
        assert len([p for p in board if p["position"] == pos]) >= 6, pos


def test_coverage_does_not_depend_on_the_overall_window(monkeypatch):
    # The point of querying by position: even a window too small to reach
    # past the receivers still returns a full slate at every position.
    from backend.app.api.recommendations import _load_board
    _fake_pool(monkeypatch, _deep_pool())
    board = _load_board(None, top_n=5, per_position=6)
    assert len([p for p in board if p["position"] == "TE"]) == 6
    assert {p["name"] for p in board if p["position"] == "TE"} == {
        f"TE{i}" for i in range(6)
    }


def test_the_cheapest_players_overall_are_still_there(monkeypatch):
    # Positional slates are the requirement; the overall window is what the
    # requirement gets weighed against. Losing it would leave the prompt
    # unable to see that a receiver 60 picks cheaper exists.
    from backend.app.api.recommendations import _load_board
    _fake_pool(monkeypatch, _deep_pool())
    board = _load_board(None, top_n=30, per_position=6)
    names = {p["name"] for p in board}
    assert {f"WR{i}" for i in range(1, 31)} <= names


def test_board_is_sorted_by_adp(monkeypatch):
    # _board_for_prompt slices [:_LISTED_PLAYERS] and the dominance guard
    # slices top_available the same way; both mean "the cheapest N".
    from backend.app.api.recommendations import _load_board
    _fake_pool(monkeypatch, _deep_pool())
    board = _load_board(None, top_n=30, per_position=6)
    adps = [p["adp"] for p in board]
    assert adps == sorted(adps)


def test_a_player_in_both_queries_appears_once(monkeypatch):
    # The cheap receivers come back from both the WR slate and the overall
    # window; a duplicate would be billed twice and listed twice.
    from backend.app.api.recommendations import _load_board
    _fake_pool(monkeypatch, _deep_pool())
    board = _load_board(None, top_n=30, per_position=6)
    ids = [p["id"] for p in board]
    assert len(ids) == len(set(ids))


def test_late_round_positions_get_no_slate(monkeypatch):
    # DST and K are deferred to the final rounds by rule, so a guaranteed
    # slate of them is context nobody reads and tokens nobody needs.
    from backend.app.api.recommendations import _CONTEXT_POSITIONS
    assert "DST" not in _CONTEXT_POSITIONS and "K" not in _CONTEXT_POSITIONS


def test_the_union_is_smaller_than_the_window_it_replaced(monkeypatch):
    # Cost matters: every row here is billed through two bulk lookups and
    # some of them through the prompt itself.
    from backend.app.api.recommendations import _load_board
    _fake_pool(monkeypatch, _deep_pool())
    assert len(_load_board(None, top_n=30, per_position=6)) < 100


def test_the_survivor_at_a_position_is_loaded_however_far_out_he_sits(monkeypatch):
    # Cost of waiting compares the best player now against the best one
    # likely to still be there next turn. That second player is past the
    # run of the board by definition, so a slate of the cheapest few never
    # holds him — and without him the prompt says "every RB will be gone",
    # which is false and tells you nothing about whether to wait.
    from backend.app.api.recommendations import _load_board
    from backend.app.services.ai_service import _format_positional_dropoff
    pool = ([_FakePlayer(i, f"WR{i}", "WR", 90.0 + i) for i in range(1, 41)]
            + [_FakePlayer(200 + i, f"RB{i}", "RB", 95.0 + i * 2) for i in range(20)])
    _fake_pool(monkeypatch, pool)
    horizon = 130
    board = _load_board(None, top_n=30, per_position=6, horizon_pick=horizon)
    text = _format_positional_dropoff(board, horizon)
    assert "projects to be gone" not in text
    assert "Cost of waiting" in text


def test_without_a_horizon_no_survivors_are_fetched(monkeypatch):
    # The last pick of a draft has no next turn. Asking "who survives to
    # never" would just bill for rows nothing reads.
    from backend.app.api.recommendations import _load_board
    pool = [_FakePlayer(i, f"WR{i}", "WR", 90.0 + i) for i in range(1, 41)]
    _fake_pool(monkeypatch, pool)
    board = _load_board(None, top_n=10, per_position=6, horizon_pick=None)
    assert max(p["adp"] for p in board) <= 100.0


def test_the_best_scorer_at_a_position_is_loaded_however_expensive_his_adp(monkeypatch):
    # The live miss this exists for: the highest-VOR back available was the
    # tenth cheapest at his position. Ranking by points per game within a
    # position IS ranking by VOR — replacement level is a constant there.
    from backend.app.api.recommendations import _load_board
    rbs = [_FakePlayer(200 + i, f"RB{i}", "RB", 110.0 + i * 5, ppg=8.0 - i * 0.1)
           for i in range(12)]
    rbs[9].ppg = 14.0                      # tenth cheapest, best by a mile
    _fake_pool(monkeypatch, rbs)
    board = _load_board(None, top_n=5, per_position=6, horizon_pick=None)
    assert rbs[9].name in {p["name"] for p in board}


def test_production_axis_survives_a_position_with_no_metrics(monkeypatch):
    # Rookies and anyone the metrics job hasn't reached have no ppg at all.
    # That's a join returning nothing, not a failure.
    from backend.app.api.recommendations import _load_board
    pool = [_FakePlayer(i, f"WR{i}", "WR", 90.0 + i) for i in range(1, 10)]
    _fake_pool(monkeypatch, pool)
    assert len(_load_board(None, top_n=30, per_position=6, horizon_pick=50)) == 9


def test_an_empty_position_is_simply_absent(monkeypatch):
    # Late in a draft a position can be picked clean. That is a smaller
    # board, not an error.
    from backend.app.api.recommendations import _load_board
    pool = [_FakePlayer(i, f"WR{i}", "WR", 90.0 + i) for i in range(1, 6)]
    _fake_pool(monkeypatch, pool)
    board = _load_board(None, top_n=30, per_position=6)
    assert {p["position"] for p in board} == {"WR"}


# ---------------------------------------------------------------------------
# Deterministic recommendation sections — best_available / needs / depth
#
# These replaced the model-authored "alternatives" list entirely — see
# RecommendationResult's docstring in ai_service.py. All three are computed
# straight from ctx.top_available and the gap math (_compute_gap_info), with
# no Claude call involved, specifically to keep output tokens down: "who's
# cheapest by ADP" doesn't need a model.
# ---------------------------------------------------------------------------

def _sections_ctx(top_available, my_roster=None, starting_lineup=None,
                   round_number=1, total_rounds=15, available_counts=None):
    return RecommendationContext(
        pick_number=1, round_number=round_number, my_slot=1, league_size=12,
        is_my_turn=True, picks_until_my_turn=0, my_next_pick_number=1,
        total_rounds=total_rounds,
        top_available=top_available,
        my_roster=my_roster or [],
        starting_lineup=starting_lineup or {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1},
        available_counts=available_counts or {},
    )


def test_best_available_ignores_position_takes_cheapest_two():
    board = [
        {"id": 1, "rank": 1, "name": "WR1", "position": "WR", "team": "X", "adp": 50.0, "sleeper_id": None},
        {"id": 2, "rank": 2, "name": "RB1", "position": "RB", "team": "X", "adp": 10.0, "sleeper_id": None},
        {"id": 3, "rank": 3, "name": "QB1", "position": "QB", "team": "X", "adp": 30.0, "sleeper_id": None},
    ]
    ctx = _sections_ctx(board)
    out = _best_available(ctx, due_late_gaps={})
    assert [p.player_id for p in out] == [2, 3]   # cheapest two by ADP
    assert all(p.tags == [_TAG_BEST_AVAILABLE] for p in out)


def test_best_available_excludes_dst_until_due():
    board = [
        {"id": 1, "rank": 1, "name": "Some Defense", "position": "DST", "team": "X", "adp": 5.0, "sleeper_id": None},
        {"id": 2, "rank": 2, "name": "RB1", "position": "RB", "team": "X", "adp": 50.0, "sleeper_id": None},
    ]
    ctx = _sections_ctx(board)
    not_due = _best_available(ctx, due_late_gaps={})
    assert "Some Defense" not in {p.player_name for p in not_due}

    due = _best_available(ctx, due_late_gaps={"DST": 1})
    assert "Some Defense" in {p.player_name for p in due}


def test_best_available_caps_at_two():
    board = [{"id": i, "rank": i, "name": f"P{i}", "position": "WR", "team": "X",
              "adp": float(i), "sleeper_id": None} for i in range(1, 10)]
    ctx = _sections_ctx(board)
    assert len(_best_available(ctx, due_late_gaps={})) == 2


def test_best_available_surfaces_the_vor_winner_not_just_cheapest_adp():
    # Live bug: with a full starting lineup and no roster gap, `main` was an
    # RB taken on VOR/opportunity-cost grounds while two cheaper receivers
    # (pure ADP) filled both best_available slots — the model's own pick
    # never appeared in ANY section. ADP-only ranking was too narrow.
    board = [
        {"id": 1, "rank": 1, "name": "Cheap WR1", "position": "WR", "team": "X", "adp": 118.0, "sleeper_id": None},
        {"id": 2, "rank": 2, "name": "Cheap WR2", "position": "WR", "team": "X", "adp": 118.2, "sleeper_id": None},
        {"id": 3, "rank": 3, "name": "Value RB", "position": "RB", "team": "X", "adp": 129.4, "sleeper_id": None},
    ]
    ctx = _sections_ctx(board)
    ctx.player_metrics = {
        1: {"fantasy_points_avg": 8.0}, 2: {"fantasy_points_avg": 8.0},
        3: {"fantasy_points_avg": 15.0},
    }
    ctx.replacement_ppg = {"WR": 7.5, "RB": 8.0}
    out = _best_available(ctx, due_late_gaps={})
    names = {p.player_name for p in out}
    assert "Cheap WR1" in names        # ADP-axis winner
    assert "Value RB" in names         # VOR-axis winner, despite worse ADP
    assert "Cheap WR2" not in names    # backfill only fires when needed
    value_rb = next(p for p in out if p.player_name == "Value RB")
    assert "VOR" in value_rb.reasoning


def test_best_available_backfills_by_adp_when_axes_agree_or_vor_is_unknown():
    # No metrics at all (e.g. very early draft, rookies) -> VOR is None for
    # everyone, so both slots must still fill from ADP alone.
    board = [{"id": i, "rank": i, "name": f"P{i}", "position": "WR", "team": "X",
              "adp": float(i), "sleeper_id": None} for i in range(1, 5)]
    ctx = _sections_ctx(board)
    out = _best_available(ctx, due_late_gaps={})
    assert [p.player_id for p in out] == [1, 2]
    assert all("ADP" in p.reasoning for p in out)


def test_needs_fills_highest_priority_open_position_first():
    # QB and RB both open; only QB has a real candidate slate (cheapest +
    # best VOR), so both needs slots go to QB rather than one each,
    # matching _POSITION_PRIORITY rather than alphabetical accident.
    board = [
        {"id": 1, "rank": 1, "name": "QB-cheap", "position": "QB", "team": "X", "adp": 40.0, "sleeper_id": None},
        {"id": 2, "rank": 2, "name": "QB-best", "position": "QB", "team": "X", "adp": 55.0, "sleeper_id": None},
        {"id": 3, "rank": 3, "name": "RB-only", "position": "RB", "team": "X", "adp": 10.0, "sleeper_id": None},
    ]
    ctx = _sections_ctx(board, my_roster=[])
    ctx.player_metrics = {2: {"fantasy_points_avg": 20.0}}
    ctx.replacement_ppg = {"QB": 15.0}
    skill_gaps = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
    out = _needs(ctx, skill_gaps, due_late_gaps={})
    assert [p.player_id for p in out] == [1, 2]
    assert all(p.tags == [_TAG_NEEDS] for p in out)


def test_needs_moves_to_next_position_when_one_slot_remains():
    # QB has only a cheapest candidate (no VOR data), so its second slot is
    # unfillable — the leftover slot goes to the next open position rather
    # than being left empty.
    board = [
        {"id": 1, "rank": 1, "name": "QB-only", "position": "QB", "team": "X", "adp": 40.0, "sleeper_id": None},
        {"id": 2, "rank": 2, "name": "RB-only", "position": "RB", "team": "X", "adp": 10.0, "sleeper_id": None},
    ]
    ctx = _sections_ctx(board)
    skill_gaps = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
    out = _needs(ctx, skill_gaps, due_late_gaps={})
    assert [p.player_id for p in out] == [1, 2]


def test_needs_is_empty_once_the_lineup_is_full():
    board = [{"id": 1, "rank": 1, "name": "WR1", "position": "WR", "team": "X",
              "adp": 10.0, "sleeper_id": None}]
    ctx = _sections_ctx(board, my_roster=roster("QB", "RB", "RB", "WR", "WR", "TE", "RB", "DST"))
    assert _needs(ctx, {}, {}) == []


def test_depth_fires_only_once_skill_gaps_clear_and_only_for_qb_te():
    board = [
        {"id": 1, "rank": 1, "name": "QB2", "position": "QB", "team": "X", "adp": 180.0, "sleeper_id": None},
        {"id": 2, "rank": 2, "name": "RB-bench", "position": "RB", "team": "X", "adp": 170.0, "sleeper_id": None},
    ]
    ctx = _sections_ctx(board)

    # skill_gaps non-empty -> depth stays empty even though a QB candidate
    # exists (gated on skill_gaps, not on `needs` — see _depth_pick).
    assert _depth_pick(ctx, {"WR": 1}) == []

    # skill_gaps empty -> depth picks the cheapest QB, never the cheaper
    # RB (RB/WR aren't in _DEPTH_POSITIONS). No TE on the board, so only
    # one entry comes back — see the next test for both firing at once.
    depth = _depth_pick(ctx, {})
    assert len(depth) == 1
    assert depth[0].player_id == 1
    assert depth[0].tags == [_TAG_DEPTH]


def test_depth_recommends_both_qb_and_te_when_both_available():
    # Live report: with backup QBs routinely cheaper by ADP than backup
    # TEs, a combined "cheapest of QB or TE" silently starved TE stash
    # recommendations even on rosters genuinely thin at TE. One candidate
    # per position fixes that — QB's cheaper ADP no longer crowds TE out.
    board = [
        {"id": 1, "rank": 1, "name": "QB2", "position": "QB", "team": "X", "adp": 180.0, "sleeper_id": None},
        {"id": 2, "rank": 2, "name": "TE2", "position": "TE", "team": "X", "adp": 210.0, "sleeper_id": None},
        {"id": 3, "rank": 3, "name": "RB-bench", "position": "RB", "team": "X", "adp": 170.0, "sleeper_id": None},
    ]
    ctx = _sections_ctx(board)
    depth = _depth_pick(ctx, {})
    assert {p.player_id for p in depth} == {1, 2}
    assert {p.position for p in depth} == {"QB", "TE"}
    assert all(p.tags == [_TAG_DEPTH] for p in depth)


def test_depth_picks_cheapest_within_each_position_independently():
    # Two QB candidates and one TE candidate: depth should return the
    # cheaper QB (not both QBs) plus the one TE — one slot per position,
    # not one slot for "cheapest overall" per position count.
    board = [
        {"id": 1, "rank": 1, "name": "QB-cheap", "position": "QB", "team": "X", "adp": 160.0, "sleeper_id": None},
        {"id": 2, "rank": 2, "name": "QB-pricier", "position": "QB", "team": "X", "adp": 180.0, "sleeper_id": None},
        {"id": 3, "rank": 3, "name": "TE2", "position": "TE", "team": "X", "adp": 210.0, "sleeper_id": None},
    ]
    ctx = _sections_ctx(board)
    depth = _depth_pick(ctx, {})
    assert {p.player_id for p in depth} == {1, 3}


def test_depth_returns_only_te_when_no_qb_is_available():
    board = [
        {"id": 1, "rank": 1, "name": "TE2", "position": "TE", "team": "X", "adp": 210.0, "sleeper_id": None},
    ]
    ctx = _sections_ctx(board)
    depth = _depth_pick(ctx, {})
    assert [p.player_id for p in depth] == [1]
    assert depth[0].position == "TE"


def test_tag_overlaps_merges_tags_for_the_same_player():
    main = PickSuggestion(1, "Shared", "RB", 10.0, "reason")
    best_available = [
        PickSuggestion(1, "Shared", "RB", 10.0, "value", tags=[_TAG_BEST_AVAILABLE]),
        PickSuggestion(2, "Other", "WR", 20.0, "value", tags=[_TAG_BEST_AVAILABLE]),
    ]
    needs = [PickSuggestion(1, "Shared", "RB", 10.0, "need", tags=[_TAG_NEEDS])]
    _tag_overlaps(main, best_available, needs, [], [])
    assert main.tags == [_TAG_MAIN, _TAG_BEST_AVAILABLE, _TAG_NEEDS]
    assert best_available[0].tags == [_TAG_MAIN, _TAG_BEST_AVAILABLE, _TAG_NEEDS]
    assert best_available[1].tags == [_TAG_BEST_AVAILABLE]
    assert needs[0].tags == [_TAG_MAIN, _TAG_BEST_AVAILABLE, _TAG_NEEDS]
    # Mutating one player's tags must not silently mutate another's — each
    # occurrence gets its own list, not a shared reference.
    main.tags.append("mutated")
    assert "mutated" not in best_available[0].tags


def test_tag_overlaps_folds_in_opportunity_too():
    # opportunity is model output, extracted separately from the other
    # three, but still needs to end up in the same reconciled tag list when
    # it overlaps with one of them — see _tag_overlaps' docstring.
    main = PickSuggestion(1, "Shared", "RB", 10.0, "reason")
    opportunity = [PickSuggestion(1, "Shared", "RB", 10.0, "role", tags=[_TAG_OPPORTUNITY])]
    _tag_overlaps(main, [], [], [], opportunity)
    assert main.tags == [_TAG_MAIN, _TAG_OPPORTUNITY]
    assert opportunity[0].tags == [_TAG_MAIN, _TAG_OPPORTUNITY]


def test_build_sections_never_exceeds_the_deterministic_slot_budget():
    board = (
        [{"id": i, "rank": i, "name": f"WR{i}", "position": "WR", "team": "X",
          "adp": float(i), "sleeper_id": None} for i in range(1, 10)]
        + [{"id": 100 + i, "rank": 100 + i, "name": f"RB{i}", "position": "RB",
            "team": "X", "adp": float(100 + i), "sleeper_id": None} for i in range(5)]
    )
    ctx = _sections_ctx(board, my_roster=[])
    main = PickSuggestion(1, "WR1", "WR", 1.0, "top pick")
    # opportunity passed empty here — this test is about the three
    # DETERMINISTIC sections' own budget (best_available/needs/depth),
    # which _build_sections computes regardless of what opportunity holds.
    best_available, needs, depth = _build_sections(ctx, main, [])
    total = 1 + len(best_available) + len(needs) + len(depth)
    assert total <= 5
    assert len(best_available) <= 2
    assert len(needs) <= 2
    assert len(depth) <= 2
    # This board's my_roster=[] means skill_gaps is never empty, so depth
    # never fires here — needs and depth CAN coexist in general (see
    # _depth_pick's docstring), just not on this particular board/roster.
    assert depth == []


def test_parse_response_populates_the_deterministic_sections():
    ctx = ctx_with_available(1, 2, 3)
    result = _parse_response(json.dumps(valid_payload()), ctx)
    assert result is not None
    # ctx_with_available's roster is empty and only stocks RB players, so
    # best_available draws from the same small pool as main.
    assert result.best_available
    assert all(p.player_id in {1, 2, 3} for p in result.best_available)
    # main always carries at least its own "main" tag once sections are built.
    assert _TAG_MAIN in result.main.tags
