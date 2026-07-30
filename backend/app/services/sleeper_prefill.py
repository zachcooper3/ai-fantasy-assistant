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

# ---------------------------------------------------------------------------
# Draft-settings slot keys — the draft object's OWN roster shape
# ---------------------------------------------------------------------------
# GET /draft/{id} carries the full lineup in settings (slots_qb, slots_rb,
# ...), which makes it the preferred source over the league's
# roster_positions: it needs no second HTTP call, and — decisively — it's
# the ONLY source available for mock drafts, which have league_id: null
# (confirmed live 2026-07-29 against a real league_mock draft; the old
# league-only path degraded every mock to "enter roster settings
# manually" even though the numbers were sitting right there in the
# response we'd already fetched).

_DRAFT_SETTINGS_SLOT_MAP = {
    "slots_qb": "qb_slots",
    "slots_rb": "rb_slots",
    "slots_wr": "wr_slots",
    "slots_te": "te_slots",
    "slots_flex": "flex_slots",   # Sleeper's plain flex = RB/WR/TE, same as ours
    "slots_def": "dst_slots",
}

# Bench and kicker aren't starting slots this app models — same stance as
# _IGNORED_SLOTS above.
_DRAFT_SETTINGS_UNSUPPORTED = {
    "slots_super_flex", "slots_wrrb_flex", "slots_rec_flex",
    "slots_idp_flex", "slots_dl", "slots_lb", "slots_db",
}

# Sleeper's metadata.scoring_type strings -> our scoring_format values.
_SCORING_TYPE_MAP = {
    "ppr": "ppr",
    "half_ppr": "half_ppr",
    "std": "standard",
    "standard": "standard",
    # "2qb", "dynasty_ppr", etc. exist — anything unrecognized falls
    # through to the league scoring_settings check instead of guessing.
    "dynasty_ppr": "ppr",
    "dynasty_half_ppr": "half_ppr",
    "dynasty_std": "standard",
}


def _slots_from_draft_settings(settings: dict) -> tuple[dict[str, int], list[str]]:
    """
    Returns ({field_name: count}, [unsupported slot keys with count > 0])
    from a draft object's settings. Empty dict means the settings carried
    no slots_* keys at all (fall back to the league's roster_positions).

    Unlike roster_positions, an explicit 0 here is real data ("this league
    starts no DST"), so zeros are passed through rather than dropped.
    """
    if not any(k.startswith("slots_") for k in settings):
        return {}, []

    counts = {
        field: settings[key]
        for key, field in _DRAFT_SETTINGS_SLOT_MAP.items()
        if isinstance(settings.get(key), int)
    }
    unsupported = sorted(
        key for key in _DRAFT_SETTINGS_UNSUPPORTED if settings.get(key)
    )
    return counts, unsupported


def _scoring_from_metadata(metadata: dict) -> str | None:
    """Maps the draft's own metadata.scoring_type to ppr/half_ppr/standard,
    or None if absent/unrecognized."""
    raw = (metadata.get("scoring_type") or "").strip().lower()
    return _SCORING_TYPE_MAP.get(raw)


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
    metadata = draft_info.get("metadata") or {}
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

    # Roster construction — the draft's OWN settings are the preferred
    # source (no second HTTP call, and the only source that exists for
    # mock drafts, whose league_id is null — see
    # _slots_from_draft_settings' docstring).
    roster_known = False
    counts, unsupported = _slots_from_draft_settings(settings)
    if counts:
        result.qb_slots = counts.get("qb_slots")
        result.rb_slots = counts.get("rb_slots")
        result.wr_slots = counts.get("wr_slots")
        result.te_slots = counts.get("te_slots")
        result.flex_slots = counts.get("flex_slots")
        result.dst_slots = counts.get("dst_slots")
        roster_known = True
        if unsupported:
            warnings.append(
                f"This draft has roster slot(s) this app doesn't model yet "
                f"({', '.join(unsupported)}) — double-check the roster settings below; "
                "they may be incomplete."
            )

    # Scoring — the draft's own metadata often says it outright.
    detected = _scoring_from_metadata(metadata)

    # League lookup — only for whatever's still missing. Mock drafts have
    # league_id: null at the top level, but league mocks carry the real
    # league in metadata.league_id (confirmed live), so check both before
    # giving up.
    league_id = draft_info.get("league_id") or metadata.get("league_id")

    if (not roster_known or detected is None) and league_id:
        try:
            league_info = await sleeper_client.get_league(league_id)
        except Exception as e:
            logger.warning(f"Sleeper prefill: get_league({league_id!r}) failed: {e}")
            league_info = None
            if not roster_known:
                warnings.append(
                    "Couldn't fetch this league's roster settings from Sleeper — "
                    "enter them manually."
                )

        if league_info is not None:
            if not roster_known:
                roster_positions = league_info.get("roster_positions") or []
                if roster_positions:
                    counts, unsupported = _slots_from_roster_positions(roster_positions)
                    result.qb_slots = counts.get("qb_slots")
                    result.rb_slots = counts.get("rb_slots")
                    result.wr_slots = counts.get("wr_slots")
                    result.te_slots = counts.get("te_slots")
                    result.flex_slots = counts.get("flex_slots")
                    result.dst_slots = counts.get("dst_slots")
                    roster_known = True
                    if unsupported:
                        warnings.append(
                            f"This league has roster slot(s) this app doesn't model yet "
                            f"({', '.join(unsupported)}) — double-check the roster "
                            "settings below; they may be incomplete."
                        )
                else:
                    warnings.append(
                        "Sleeper didn't report roster positions for this league — "
                        "enter them manually."
                    )
            if detected is None:
                detected = _detect_scoring_format(league_info.get("scoring_settings") or {})
    elif not roster_known and not league_id:
        warnings.append(
            "Couldn't determine this draft's roster settings — enter them manually."
        )

    result.detected_scoring_format = detected
    if detected and detected != "ppr":
        warnings.append(
            f"This league uses {detected.replace('_', ' ')} scoring, but this app's "
            "rankings, ADP, and AI recommendations are currently built on PPR data — "
            "they may be less accurate for your league."
        )

    result.warnings = warnings
    return result
