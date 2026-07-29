"""
Input validation at system boundaries — audit W5 (config cross-field
rules) and W6 (Sleeper ID path-injection guards).
"""

import pytest
from pydantic import ValidationError

from backend.app.api.sync import SyncStartRequest
from backend.app.schemas import DraftConfigRequest
from backend.app.services.sleeper_client import _validate_numeric_id


# ---------------------------------------------------------------------------
# W5 — DraftConfigRequest
# ---------------------------------------------------------------------------

def test_draft_position_at_league_size_boundary_is_valid():
    cfg = DraftConfigRequest(league_size=12, my_draft_position=12)
    assert cfg.my_draft_position == 12


def test_draft_position_beyond_league_size_rejected():
    with pytest.raises(ValidationError, match="my_draft_position"):
        DraftConfigRequest(league_size=12, my_draft_position=14)


def test_scoring_format_restricted_to_known_values():
    assert DraftConfigRequest(scoring_format="half_ppr").scoring_format == "half_ppr"
    with pytest.raises(ValidationError):
        DraftConfigRequest(scoring_format="superflex-madness")


def test_defaults_are_standard_ppr_lineup():
    cfg = DraftConfigRequest()
    assert (cfg.league_size, cfg.my_draft_position, cfg.total_rounds) == (12, 1, 15)
    assert (cfg.qb_slots, cfg.rb_slots, cfg.wr_slots, cfg.te_slots,
            cfg.flex_slots, cfg.dst_slots) == (1, 2, 2, 1, 1, 1)


# ---------------------------------------------------------------------------
# W6 — Sleeper ID guards
# ---------------------------------------------------------------------------

def test_numeric_ids_pass_through():
    assert _validate_numeric_id("1234567890", "draft_id") == "1234567890"


@pytest.mark.parametrize("bad", [
    "", "abc", "123/456", "../user/x", "123?x=1", "1 2", "123#frag", None,
])
def test_non_numeric_ids_rejected(bad):
    with pytest.raises(ValueError, match="Invalid Sleeper"):
        _validate_numeric_id(bad, "draft_id")


def test_sync_start_request_rejects_path_traversal():
    SyncStartRequest(draft_id="1234567890")  # valid
    with pytest.raises(ValidationError):
        SyncStartRequest(draft_id="../user/x")
    with pytest.raises(ValidationError):
        SyncStartRequest(draft_id="123?bust=1")
