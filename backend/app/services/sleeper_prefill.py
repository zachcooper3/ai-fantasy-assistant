"""
Best-effort Sleeper draft/league settings lookup, for pre-filling (not
silently overriding) the setup form.

Ties together three existing sleeper_client calls that, before this file,
were only ever used for live pick syncing (draft_sync.py):
  - get_draft(draft_id)   -> league_id, team count, round count, draft order
  - get_league(league_id) -> roster_positions, scoring_settings
  - get_user(username)    -> user_id, to resolve the caller's own draft slot

Design stance, matching ai_service.py's fallback philosophy elsewhere in
this app: setup should never be blocked or crashed by a flaky or wrong
Sleeper ID. Every external call is isolated in its own try/except; a
failure at any stage degrades that stage's fields to None (with a
human-readable warning) rather than aborting the whole lookup.

Author: Zach Cooper
"""

import logging

from backend.app.schemas import SleeperPrefillResponse
from backend.app.services import sleeper_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sleeper roster_positions code -> our DraftConfig field
# ---------------------------------------------------------------------------

# Direct, unambiguous mappings only. Sleeper's plain "FLEX" is RB/WR/TE,
# which matches this app's own _FLEX_ELIGIBLE_POSITIONS exactly (see
# ai_service.py) — safe to map 1:1.
_DIRECT_SLOT_MAP = {
    "QB": "qb_slots",
    "RB": "rb_slots",
    "WR": "wr_slots",
    "TE": "te_slots",
    "DEF": "dst_slots",
    "FLEX": "flex_slots",
}

# Not starting slots — safe to ignore rather than flag.
_IGNORED_SLOTS = {"BN", "IR", "TAXI", "K"}

# Slot types this app doesn't model (no superflex/2-QB/IDP support — a
# deliberate scope decision, see DraftConfigRequest's docstring). Counted
# and surfaced as a warning rather than guessed at, since e.g. mapping
# WRRB_FLEX or REC_FLEX onto our single generic flex_slots would silently
# misrepresent the league's actual eligibility rules.
_UNSUPPORTED_SLOTS = {"SUPER_FLEX", "WRRB_FLEX", "REC_FLEX", "IDP_FLEX", "DL", "LB", "DB"}


def _detect_scoring_format(scoring_settings: dict) -> str | None:
    """Maps Sleeper's reception-point setting to ppr/half_ppr/standard."""
    rec = scoring_settings.get("rec")
    if rec is None:
        return None
    if rec >= 1:
        return "ppr"
    if rec >= 0.5:
        return "half_ppr"
    return "standard"


def _slots_from_roster_positions(positions: list[str]) -> tuple[dict[str, int], list[str]]:
    """
    Returns ({field_name: count}, [unsupported codes found]).
    Only fields that appear at least once are included in the dict, so the
    caller can tell "this league has 0 TEs" (present, value 0 — not
    possible via roster_positions since 0 just means absent) apart from
    "we don't know" — in practice every code either appears >=1 time or
    not at all, so absent fields are left as None upstream to mean
    "not detected" is indistinguishable from "zero"; that's fine here
    because a real Sleeper lineup always includes >=1 of each core
    position, and if it doesn't, None (falling back to the form's current
    value) is the safer default over a confident-looking 0.
    """
    counts: dict[str, int] = {}
    unsupported: set[str] = set()
    for code in positions:
        field = _DIRECT_SLOT_MAP.get(code)
        if field:
            counts[field] = counts.get(field, 0) + 1
        elif code in _UNSUPPORTED_SLOTS:
            unsupported.add(code)
        elif code not in _IGNORED_SLOTS:
            # Unknown code we've never seen — treat like unsupported rather
            # than silently dropping it, so it's at least visible.
            unsupported.add(code)
    return counts, sorted(unsupported)


async def build_prefill(draft_id: str, username: str | None) -> SleeperPrefillResponse:
    """
    Builds a SleeperPrefillResponse from a Sleeper draft ID and optional
    username. Never raises — any failure degrades to None fields plus a
    warning explaining what couldn't be detected.
    """
    warnings: list[str] = []
    result = SleeperPrefillResponse()

    try:
        draft_info = await sleeper_client.get_draft(draft_id)
    except Exception as e:
        logger.warning(f"Sleeper prefill: get_draft({draft_id!r}) failed: {e}")
        warnings.append(
            "Couldn't reach Sleeper for that draft ID — double-check it and enter "
            "settings manually."
        )
        result.warnings = warnings
        return result

    settings = draft_info.get("settings") or {}
    result.league_size = settings.get("teams")
    result.total_rounds = settings.get("rounds")
    if result.league_size is None or result.total_rounds is None:
        warnings.append(
            "Sleeper didn't report a team count or round count for this draft — "
            "enter those manually."
        )

    # My draft slot — only attempted if a username was given.
    if username:
        draft_order = draft_info.get("draft_order") or {}
        try:
            user = await sleeper_client.get_user(username)
            user_id = user.get("user_id") if user else None
            slot = draft_order.get(user_id) if user_id else None
            if slot is not None:
                result.my_draft_position = slot
            else:
                warnings.append(
                    f"Couldn't find '{username}' in this draft's order — enter your "
                    "draft slot manually (the order may not be set until Sleeper "
                    "randomizes it, right before the draft starts)."
                )
        except Exception as e:
            logger.warning(f"Sleeper prefill: get_user({username!r}) failed: {e}")
            warnings.append(f"Couldn't look up Sleeper user '{username}' — enter your draft slot manually.")

    # Roster construction + scoring — requires the league_id from the draft.
    league_id = draft_info.get("league_id")
    if not league_id:
        warnings.append("Couldn't determine this draft's league — enter roster settings manually.")
        result.warnings = warnings
        return result

    try:
        league_info = await sleeper_client.get_league(league_id)
    except Exception as e:
        logger.warning(f"Sleeper prefill: get_league({league_id!r}) failed: {e}")
        warnings.append("Couldn't fetch this league's roster settings from Sleeper — enter them manually.")
        result.warnings = warnings
        return result

    roster_positions = league_info.get("roster_positions") or []
    if roster_positions:
        counts, unsupported = _slots_from_roster_positions(roster_positions)
        result.qb_slots = counts.get("qb_slots")
        result.rb_slots = counts.get("rb_slots")
        result.wr_slots = counts.get("wr_slots")
        result.te_slots = counts.get("te_slots")
        result.flex_slots = counts.get("flex_slots")
        result.dst_slots = counts.get("dst_slots")
        if unsupported:
            warnings.append(
                f"This league has roster slot(s) this app doesn't model yet "
                f"({', '.join(unsupported)}) — double-check the roster settings below; "
                "they may be incomplete."
            )
    else:
        warnings.append("Sleeper didn't report roster positions for this league — enter them manually.")

    scoring_settings = league_info.get("scoring_settings") or {}
    detected = _detect_scoring_format(scoring_settings)
    result.detected_scoring_format = detected
    if detected and detected != "ppr":
        warnings.append(
            f"This league uses {detected.replace('_', ' ')} scoring, but this app's "
            "rankings, ADP, and AI recommendations are currently built on PPR data — "
            "they may be less accurate for your league."
        )

    result.warnings = warnings
    return result
