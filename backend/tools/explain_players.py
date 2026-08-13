"""
Shows exactly what the recommendation prompt says about specific players —
and, just as importantly, what it does NOT say.

This exists because "why did it pick X over Y?" is not answerable by reading
ai_service.py. The prompt is assembled from four independent sources (the
player row, PlayerMetrics, DraftProfile, and ChromaDB), each of which can be
silently empty for one player and rich for another. When that happens the
model isn't ignoring your instructions, it's reading a different amount of
evidence about each player than you assume — and the only way to see that is
to render the actual lines side by side.

The "NOT VISIBLE" block is the point of the tool. A field that is NULL for
every player in the DB (because an ingestion step failed) looks identical,
from inside the prompt, to a field that doesn't exist — the model can't tell
you it's missing, it just quietly reasons without it.

Read-only: issues SELECTs and nothing else.

Run:
    py -m backend.tools.explain_players "Emeka Egbuka" "Davante Adams"
"""

import argparse
import logging
import sys

from sqlmodel import Session, select

from backend.app.services.ai_service import (
    _format_draft_profile_line,
    _format_metrics_line,
    _format_metrics_section,
    _MAX_CHUNKS_PER_PLAYER,
)
from backend.db.database import engine
from backend.db.models import DraftProfile, Player, PlayerMetrics

# Fields worth calling out by name when they're NULL — these are the ones the
# prompt leans on hardest ("real usage", "rising role"), so their absence
# changes recommendations the most.
_KEY_METRIC_FIELDS = [
    "snap_pct", "target_share", "carry_share", "depth_chart_rank",
    "target_share_trend", "snap_pct_trend", "depth_chart_trend",
]


def _find(session: Session, name: str) -> list[Player]:
    """Case-insensitive substring match — you shouldn't have to type the
    apostrophe in "De'Zhaun Stribling" correctly to debug a pick."""
    stmt = select(Player).where(Player.name.ilike(f"%{name}%"))  # type: ignore[attr-defined]
    return list(session.exec(stmt))


# model_fields is read off the class, not the instance — Pydantic 2.11
# deprecated instance access and will remove it in v3.
def _metrics_dict(m: PlayerMetrics) -> dict:
    return {c: getattr(m, c) for c in type(m).model_fields}


def _profile_dict(d: DraftProfile) -> dict:
    return {c: getattr(d, c) for c in type(d).model_fields}


def _retrieved_chunks(sleeper_id: str | None) -> list[str] | str:
    """The same query the prompt makes. Returns a message rather than raising
    when the vector store isn't importable, matching how the prompt itself
    degrades (best-effort, never fatal)."""
    if not sleeper_id:
        return "no sleeper_id on this player — retrieval is skipped entirely for them"
    try:
        from backend.rag.vector_store import query as vector_query
    except Exception as e:
        return f"vector store unavailable ({e})"
    try:
        return vector_query(
            "",
            n_results=_MAX_CHUNKS_PER_PLAYER,
            where={"$and": [
                {"sleeper_id": sleeper_id},
                {"chunk_type": {"$in": ["what_happened", "what_it_means"]}},
            ]},
        )
    except Exception as e:
        return f"query failed ({e})"


def explain(names: list[str]) -> None:
    with Session(engine) as session:
        # Global NULL coverage first: a field empty for everyone is an
        # ingestion failure, not a fact about the player you're debugging,
        # and mistaking one for the other sends you looking in the wrong file.
        all_metrics = list(session.exec(select(PlayerMetrics)))
        total = len(all_metrics)
        print("=" * 72)
        print(f"DB-WIDE METRIC COVERAGE ({total} PlayerMetrics rows)")
        print("=" * 72)
        for f in _KEY_METRIC_FIELDS:
            n = sum(1 for m in all_metrics if getattr(m, f, None))
            flag = "  <-- EMPTY FOR EVERY PLAYER (ingestion gap)" if n == 0 else ""
            print(f"  {f:<26} populated: {n:>4}/{total}{flag}")
        print()

        resolved: list[Player] = []
        for name in names:
            matches = _find(session, name)
            if not matches:
                print(f"!! No player matching {name!r}\n")
                continue
            if len(matches) > 1:
                print(f"!! {name!r} matched {len(matches)}: "
                      f"{', '.join(p.name for p in matches)} — using the first\n")
            resolved.append(matches[0])

        for p in resolved:
            print("=" * 72)
            print(f"{p.name}  ({p.position}, {p.team})")
            print("=" * 72)
            print(f"  rank {p.rank} | ADP {p.adp} | "
                  f"{'AVAILABLE' if p.is_available else 'already drafted'} | "
                  f"sleeper_id {p.sleeper_id}")

            m = session.exec(
                select(PlayerMetrics).where(PlayerMetrics.player_id == p.id)
            ).first()
            d = session.exec(
                select(DraftProfile).where(DraftProfile.player_id == p.id)
            ).first()

            print("\n  -- LINE THE PROMPT ACTUALLY RENDERS --")
            section = _format_metrics_section(
                [{"id": p.id, "name": p.name, "position": p.position,
                  "team": p.team, "adp": p.adp, "sleeper_id": p.sleeper_id,
                  "rank": p.rank}],
                {p.id: _metrics_dict(m)} if m else {},
                {p.id: _profile_dict(d)} if d else {},
            )
            for line in section.splitlines():
                if line.startswith("- "):
                    print(f"    {line}")

            print("\n  -- DRAFT CAPITAL ON FILE --")
            if d is None:
                print("    none")
            else:
                dp_line = _format_draft_profile_line(_profile_dict(d))
                print(f"    {dp_line or 'row exists but nothing resolved'}")
                if m is not None:
                    # The suppression bug worth seeing plainly: draft capital
                    # is rendered only as a FALLBACK for players with no
                    # metrics, so a second-year player with first-round
                    # pedigree presents to the model as an anonymous veteran.
                    print("    ^^ NOT SHOWN IN THE PROMPT — this player has "
                          "PlayerMetrics, and _format_metrics_section only "
                          "renders draft capital when metrics are MISSING.")

            print("\n  -- NOT VISIBLE TO THE MODEL --")
            missing = []
            if m is None:
                missing.append("all PlayerMetrics (no row at all)")
            else:
                missing += [f for f in _KEY_METRIC_FIELDS
                            if not getattr(m, f, None)]
            # Nothing anywhere in the schema carries these.
            missing += ["age / birthdate (no column exists)",
                        "years of NFL experience (no column exists)",
                        "bye week (exists on Player but never reaches the prompt)"]
            for f in missing:
                print(f"    {f}")

            print("\n  -- RETRIEVED NEWS/ANALYSIS (ChromaDB) --")
            chunks = _retrieved_chunks(p.sleeper_id)
            if isinstance(chunks, str):
                print(f"    {chunks}")
            elif not chunks:
                print("    NONE — prompt shows an explicit 'no retrieved data' line")
            else:
                for ch in chunks:
                    print(f"    [{len(ch)} chars] {ch[:300]}"
                          f"{'...' if len(ch) > 300 else ''}")
            print()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Show what the recommendation prompt says about specific players."
    )
    ap.add_argument("names", nargs="+", help="Player names (substring match).")
    args = ap.parse_args()
    if not args.names:
        ap.print_help()
        sys.exit(1)
    explain(args.names)


if __name__ == "__main__":
    main()
