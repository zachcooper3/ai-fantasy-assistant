"""
Sleeper prefill — especially the mock-draft path (league_id: null), which
used to degrade to "enter roster settings manually" even though the draft
object's own settings carried the full roster shape. Payload shapes here
mirror a real league_mock response captured live on 2026-07-29.
"""

import asyncio

import pytest

from backend.app.services import sleeper_client, sleeper_prefill
from backend.app.services.sleeper_prefill import (
    _scoring_from_metadata,
    _slots_from_draft_settings,
    build_prefill,
)


# Trimmed real-world league_mock payload (draft 1388...328, 2026-07-29)
MOCK_DRAFT_INFO = {
    "draft_id": "1388332007859171328",
    "league_id": None,  # mocks have no top-level league
    "draft_order": {"872566687508131840": 5},
    "metadata": {
        "league_id": "1367214051372830720",  # ...but league mocks carry it here
        "scoring_type": "ppr",
        "type": "league_mock",
    },
    "settings": {
        "rounds": 15, "teams": 14, "pick_timer": 120,
        "slots_qb": 1, "slots_rb": 2, "slots_wr": 2, "slots_te": 1,
        "slots_flex": 2, "slots_def": 1, "slots_k": 1, "slots_bn": 5,
    },
    "status": "paused",
    "type": "snake",
}


# ---------------------------------------------------------------------------
# _slots_from_draft_settings
# ---------------------------------------------------------------------------

def test_slots_extracted_from_draft_settings():
    counts, unsupported = _slots_from_draft_settings(MOCK_DRAFT_INFO["settings"])
    assert counts == {
        "qb_slots": 1, "rb_slots": 2, "wr_slots": 2,
        "te_slots": 1, "flex_slots": 2, "dst_slots": 1,
    }
    assert unsupported == []


def test_explicit_zero_is_real_data():
    counts, _ = _slots_from_draft_settings({"slots_qb": 1, "slots_def": 0})
    assert counts["dst_slots"] == 0  # league starts no DST — not "unknown"


def test_no_slot_keys_means_unknown():
    assert _slots_from_draft_settings({"teams": 12, "rounds": 15}) == ({}, [])


def test_superflex_flagged_as_unsupported():
    counts, unsupported = _slots_from_draft_settings(
        {"slots_qb": 1, "slots_super_flex": 1}
    )
    assert unsupported == ["slots_super_flex"]
    assert counts["qb_slots"] == 1


# ---------------------------------------------------------------------------
# _scoring_from_metadata
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("ppr", "ppr"),
    ("half_ppr", "half_ppr"),
    ("std", "standard"),
    ("dynasty_ppr", "ppr"),
    ("2qb", None),          # unrecognized — don't guess
    ("", None),
    (None, None),
])
def test_scoring_from_metadata(raw, expected):
    assert _scoring_from_metadata({"scoring_type": raw}) == expected


# ---------------------------------------------------------------------------
# build_prefill — mock draft end to end (no league call needed)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_sleeper(monkeypatch):
    """Patches sleeper_client's async calls; get_league fails loudly so a
    test proves the mock-draft path never needed it."""
    calls = {"get_league": 0}

    async def fake_get_draft(draft_id):
        return MOCK_DRAFT_INFO

    async def fake_get_user(username):
        return {"user_id": "872566687508131840"}

    async def fake_get_league(league_id):
        calls["get_league"] += 1
        raise AssertionError("league lookup should not be needed")

    monkeypatch.setattr(sleeper_client, "get_draft", fake_get_draft)
    monkeypatch.setattr(sleeper_client, "get_user", fake_get_user)
    monkeypatch.setattr(sleeper_client, "get_league", fake_get_league)
    return calls


def test_mock_draft_fully_prefills_without_league(fake_sleeper):
    result = asyncio.run(build_prefill("1388332007859171328", "CoopersOk"))

    assert result.league_size == 14
    assert result.total_rounds == 15
    assert result.my_draft_position == 5
    assert (result.qb_slots, result.rb_slots, result.wr_slots) == (1, 2, 2)
    assert (result.te_slots, result.flex_slots, result.dst_slots) == (1, 2, 1)
    assert result.detected_scoring_format == "ppr"
    # The exact regression: no "enter roster settings manually" warning
    assert result.warnings == []
    assert fake_sleeper["get_league"] == 0


def test_league_fallback_when_draft_settings_lack_slots(monkeypatch):
    """Older/odd drafts without slots_* keys still use the league path."""
    draft_info = {
        "league_id": "L1",
        "settings": {"teams": 12, "rounds": 15},
        "metadata": {},
    }

    async def fake_get_draft(draft_id):
        return draft_info

    async def fake_get_league(league_id):
        return {
            "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "DEF", "BN"],
            "scoring_settings": {"rec": 0.5},
        }

    monkeypatch.setattr(sleeper_client, "get_draft", fake_get_draft)
    monkeypatch.setattr(sleeper_client, "get_league", fake_get_league)

    result = asyncio.run(build_prefill("123", None))
    assert result.rb_slots == 2 and result.flex_slots == 1 and result.dst_slots == 1
    assert result.detected_scoring_format == "half_ppr"
    assert any("half ppr" in w for w in result.warnings)  # non-PPR caveat


def test_no_roster_source_at_all_warns(monkeypatch):
    async def fake_get_draft(draft_id):
        return {"league_id": None, "settings": {"teams": 12, "rounds": 15}, "metadata": {}}

    monkeypatch.setattr(sleeper_client, "get_draft", fake_get_draft)

    result = asyncio.run(build_prefill("123", None))
    assert result.qb_slots is None
    assert any("roster settings" in w for w in result.warnings)


def test_unreachable_draft_warns_and_returns(monkeypatch):
    async def fake_get_draft(draft_id):
        raise RuntimeError("network down")

    monkeypatch.setattr(sleeper_client, "get_draft", fake_get_draft)

    result = asyncio.run(build_prefill("123", None))
    assert result.league_size is None
    assert any("double-check" in w for w in result.warnings)
