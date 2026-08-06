"""
Times every stage of a single recommendation so latency can be attributed
instead of guessed at.

Four things can dominate, and they call for opposite fixes:

  DB context build   — SQL, usually milliseconds
  ChromaDB retrieval — one embedding + query per candidate player, and the
                       FIRST call in a process also loads the embedding
                       model, which is seconds on its own
  Claude API call    — dominated by OUTPUT tokens, not input; generation is
                       sequential, prompt ingestion is not
  parsing            — microseconds

Guessing badly here is expensive in both directions. Trimming the prompt
when generation is the bottleneck buys nothing, and switching models when
retrieval is the bottleneck buys nothing either.

Run:
    py -m backend.tools.profile_recommendation              # full, real API call
    py -m backend.tools.profile_recommendation --no-api     # free, skips Claude
    py -m backend.tools.profile_recommendation --repeat 3   # see cache effects
"""

import argparse
import asyncio
import logging
import time
from contextlib import contextmanager

from sqlmodel import Session

logger = logging.getLogger(__name__)

_timings: dict[str, float] = {}


@contextmanager
def _timed(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        _timings[label] = time.perf_counter() - start


def _bar(seconds: float, total: float, width: int = 34) -> str:
    filled = int(width * seconds / total) if total else 0
    return "#" * filled + "." * (width - filled)


async def _run(use_api: bool, repeat: int) -> None:
    from backend.app.services import ai_service as S
    from backend.app.api.recommendations import _build_context
    from backend.app.services.draft_state import DraftConfig, DraftStateService
    from backend.db.database import engine

    svc = DraftStateService()
    svc.start_session(DraftConfig(league_size=12, my_draft_position=6, total_rounds=15))

    for run in range(1, repeat + 1):
        _timings.clear()
        print(f"\n{'=' * 62}\nRUN {run} of {repeat}\n{'=' * 62}")

        with Session(engine) as db:
            with _timed("1. build context (SQL)"):
                ctx = _build_context(svc, db)

        # Retrieval is inside _build_prompt; time it separately first so the
        # two can be told apart. This also warms the module-level cache, so
        # the _build_prompt figure below reflects formatting only.
        with _timed("2. ChromaDB retrieval"):
            news = S._retrieve_player_context(ctx.top_available)

        with _timed("3. build prompt (formatting)"):
            prompt = S._build_prompt(ctx)

        system = S._build_system_prompt(ctx.scoring_format)
        in_est = (len(system) + len(prompt)) // 4
        print(f"\n  board: {len(ctx.top_available)} players, "
              f"{S._LISTED_PLAYERS} shown, retrieval over {S._MAX_CONTEXT_PLAYERS}")
        print(f"  news section: {len(news):,} chars "
              f"({'EMPTY — ChromaDB unavailable' if not news else 'ok'})")
        print(f"  prompt: ~{in_est:,} input tokens")

        if use_api:
            svc_ai = S.AIService()
            if not svc_ai.is_configured:
                print("\n  no ANTHROPIC_API_KEY — skipping the API call")
            else:
                with _timed("4. Claude API call"):
                    response = await svc_ai._client.messages.create(
                        model=svc_ai.model_name,
                        max_tokens=S._MAX_RESPONSE_TOKENS,
                        temperature=S._TEMPERATURE,
                        system=system,
                        messages=[{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": "{"}],
                    )
                usage = getattr(response, "usage", None)
                raw = next((b.text for b in response.content if hasattr(b, "text")), "")
                with _timed("5. parse response"):
                    S._parse_response(S._restore_prefill(raw), ctx)
                if usage:
                    api = _timings.get("4. Claude API call", 0)
                    out = getattr(usage, "output_tokens", 0)
                    print(f"  actual tokens: {getattr(usage,'input_tokens',0):,} in, {out:,} out")
                    if out and api:
                        print(f"  generation rate: {out / api:.0f} output tok/sec")

        total = sum(_timings.values())
        print(f"\n  {'stage':<26}{'sec':>8}{'%':>7}")
        for label, secs in _timings.items():
            print(f"  {label:<26}{secs:>8.2f}{100 * secs / total:>6.0f}%  {_bar(secs, total)}")
        print(f"  {'TOTAL':<26}{total:>8.2f}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-api", action="store_true",
                    help="Skip the Claude call (free; times only local work).")
    ap.add_argument("--repeat", type=int, default=1,
                    help="Repeat N times — run 2+ shows what caching actually saves.")
    args = ap.parse_args()
    asyncio.run(_run(use_api=not args.no_api, repeat=args.repeat))


if __name__ == "__main__":
    main()
