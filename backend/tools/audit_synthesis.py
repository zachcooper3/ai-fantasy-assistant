"""
Cross-checks every stored "what it means" ChromaDB note against the exact
data block Claude was given when it was generated, and flags any number in
the note that doesn't trace back to that block.

Why this exists: the live Opportunity pick hallucinated a PPG figure on
2026-08-11 (see ai_service.py RULE 10 history) despite being told to ground
every stat in the prompt. fetch_synthesis.py and fetch_rookie_synthesis.py
carry the exact same risk and have carried it since before this app's
RULE 10 discipline existed — one Claude call per player, asked to
"reference specific numbers," with no check afterward that it actually did.
Those notes aren't a one-off live response, either: they're written to
ChromaDB and retrieved as "Player News & Analysis" on every future
recommendation for that player, so a hallucinated number in a stored note
is a standing wrong answer, not a single bad pick.

Method: rebuild the literal ground-truth block each note was synthesized
from (format_metrics_prompt / format_draft_profile_prompt — the same
functions fetch_synthesis.py and fetch_rookie_synthesis.py call), extract
every number that appears in it, extract every number in the stored note,
and flag any note-number with no match in the ground truth within a small
rounding tolerance. This is a triage tool, not a verdict: it will flag
some real false positives (a rank like "WR2" or an enumeration), which is
why it prints the surrounding sentence for each hit rather than just a
number. Read-only — issues SELECTs and Chroma reads, writes nothing.

Run:
    py -m backend.tools.audit_synthesis
    py -m backend.tools.audit_synthesis --sleeper-id 4866
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlmodel import Session, select

from backend.db.database import engine
from backend.db.models import DraftProfile, Player, PlayerMetrics
from backend.ingestion.fetch_synthesis import format_metrics_prompt
from backend.ingestion.fetch_rookie_synthesis import format_draft_profile_prompt
from backend.rag.vector_store import get_collection

# A number closer than this to some ground-truth number counts as a match.
# Percentages in the ground truth are always whole numbers (see fetch_
# synthesis.py's _pct), so a wider tolerance there absorbs Claude writing
# "23.5%" for a value the block showed as "23%". Non-percent stats in the
# ground truth already carry decimals, so a tighter tolerance is enough to
# absorb ordinary rounding (10.95 -> "11.0") without absorbing a genuinely
# different number.
_PCT_TOLERANCE = 1.5
_ABS_TOLERANCE = 0.3

# Below this, a lone digit is almost always prose (an enumeration, "one of
# three factors", a jersey-adjacent mention) rather than a cited stat, and
# flagging it is pure noise.
_MIN_FLAG_VALUE = 3.0

# Comma group is required to have its OWN capture so "1,372" parses as one
# number (1372) rather than splitting at the comma and matching only "372"
# against the ground truth's "1372" — confirmed live: this was the single
# biggest source of false positives in the first version of this tool,
# flagging normal 4-digit college rushing/receiving yardage as unmatched.
_NUM_RE = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(%)?")


def _extract_numbers(text: str) -> list[tuple[float, bool]]:
    """[(value, is_percent)] for every number token in `text`."""
    out = []
    for m in _NUM_RE.finditer(text):
        whole, frac, pct = m.group(1), m.group(2), m.group(3)
        val = float(whole.replace(",", "") + (f".{frac}" if frac else ""))
        out.append((val, bool(pct)))
    return out


def _has_match(val: float, is_pct: bool, ground_truth: list[tuple[float, bool]]) -> bool:
    tol = _PCT_TOLERANCE if is_pct else _ABS_TOLERANCE
    return any(gpct == is_pct and abs(gval - val) <= tol for gval, gpct in ground_truth)


def _context(text: str, needle: str, width: int = 60) -> str:
    i = text.find(needle)
    if i == -1:
        return text[:120]
    start, end = max(0, i - width), min(len(text), i + len(needle) + width)
    return ("..." if start > 0 else "") + text[start:end].strip() + ("..." if end < len(text) else "")


def audit(sleeper_id: str | None = None) -> None:
    with Session(engine) as session:
        metrics_by_sid = {
            m.sleeper_id: m for m in session.exec(select(PlayerMetrics))
        }
        profile_by_sid = {
            d.sleeper_id: d for d in session.exec(select(DraftProfile)) if d.sleeper_id
        }
        player_by_sid = {
            p.sleeper_id: p for p in session.exec(select(Player)) if p.sleeper_id
        }

    where = {"chunk_type": "what_it_means"}
    if sleeper_id:
        where = {"$and": [where, {"sleeper_id": sleeper_id}]}
    results = get_collection().get(where=where, include=["documents", "metadatas"])
    docs = results.get("documents") or []
    metas = results.get("metadatas") or []

    print(f"Auditing {len(docs)} 'what it means' note(s)...\n")

    flagged = 0
    no_ground_truth = 0
    for note, meta in zip(docs, metas):
        sid = meta.get("sleeper_id")
        name = meta.get("player_name", "?")
        player = player_by_sid.get(sid)

        m = metrics_by_sid.get(sid)
        d = profile_by_sid.get(sid)
        if player and m:
            ground_truth_text = format_metrics_prompt(player, m)
        elif player and d:
            ground_truth_text = format_draft_profile_prompt(player, d)
        else:
            no_ground_truth += 1
            continue  # can't verify without knowing what it was grounded in

        ground_truth = _extract_numbers(ground_truth_text)
        note_numbers = _extract_numbers(note)

        bad = [
            (val, is_pct) for val, is_pct in note_numbers
            if val >= _MIN_FLAG_VALUE and not _has_match(val, is_pct, ground_truth)
        ]
        if not bad:
            continue

        flagged += 1
        print("=" * 72)
        print(f"{name}  (sleeper_id={sid})")
        print("-" * 72)
        for val, is_pct in bad:
            token = f"{val:g}{'%' if is_pct else ''}"
            print(f"  UNMATCHED: {token}")
            print(f"    ...{_context(note, token)}...")
        print("\n  Full note:")
        print(f"    {note}")
        print()

    print("=" * 72)
    print(
        f"{flagged}/{len(docs)} note(s) flagged for at least one unmatched number "
        f"({no_ground_truth} skipped — no matching PlayerMetrics/DraftProfile row "
        "to verify against, e.g. the source data has since changed)."
    )
    print(
        "A flag is not proof — ranks like 'WR2' and enumerations sometimes trip "
        "this. Read the context line before treating it as a hallucination."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flag stored synthesis notes containing numbers not traceable to their source data."
    )
    parser.add_argument("--sleeper-id", type=str, default=None, help="Only audit this one player.")
    args = parser.parse_args()
    audit(sleeper_id=args.sleeper_id)


if __name__ == "__main__":
    main()
