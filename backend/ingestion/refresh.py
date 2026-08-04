"""
One command to refresh every data source this app draws on, in dependency
order.

    py -m backend.ingestion.refresh              # everything free
    py -m backend.ingestion.refresh --with-ai    # + the two Claude synthesis steps

The split is deliberate. Six of the eight steps hit free public data sources
and can be re-run as often as you like; the two synthesis steps call the
Claude API once per player and cost real money, so they never run unless you
ask for them by name. During draft week the free refresh is the one you want
daily — ADP moves, injuries move, and neither synthesis step's output changes
much day to day.

Ordering is not arbitrary and the steps are not independent:

    fetch_adp ............... rewrites the Player table wholesale
      -> sync_sleeper_ids ... needs those rows to attach sleeper_ids to
        -> fetch_metrics .... matches players by sleeper_id
        -> fetch_draft_profiles
          -> fetch_college_stats ... only enriches existing DraftProfile rows
        -> chunker .......... needs sleeper_ids to key chunks by
          -> fetch_synthesis ......... reads PlayerMetrics   [Claude]
          -> fetch_rookie_synthesis .. reads DraftProfile    [Claude]

Because fetch_adp truncates and reloads Player, running anything downstream
of it against a stale player table produces silently mismatched data rather
than an error — which is exactly the kind of failure this script exists to
make impossible to cause by hand.

Failure policy mirrors the rest of the app: a step that only enriches the
prompt is best-effort and the run continues without it, but a step everything
downstream depends on stops the run, because continuing would just write
mismatched rows on top of a broken foundation.

Author: Zach Cooper
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Steps are launched from here so relative paths like data/raw/... resolve
# the same way they do when you run each script by hand.
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Step:
    name: str           # short handle, used by --only
    module: str         # runnable via python -m
    label: str          # what it actually does, for the log
    critical: bool      # does the rest of the run depend on this succeeding?
    uses_claude: bool
    args: tuple[str, ...] = ()


# fetch_adp is listed instead of ingest_players because it re-ingests the
# database itself after writing the CSV (see its docstring) — running both
# would just do the same import twice.
_STEPS: list[Step] = [
    Step("adp", "backend.ingestion.fetch_adp",
         "Fresh PPR ADP + reload of the Player table", critical=True, uses_claude=False),
    Step("ids", "backend.ingestion.sync_sleeper_ids",
         "Sleeper ID crosswalk (everything below keys off this)",
         critical=True, uses_claude=False),
    Step("metrics", "backend.ingestion.fetch_metrics",
         "Prior-season usage, efficiency, durability", critical=False, uses_claude=False),
    Step("draft", "backend.ingestion.fetch_draft_profiles",
         "Draft capital for the last two classes", critical=False, uses_claude=False),
    Step("college", "backend.ingestion.fetch_college_stats",
         "Final-college-season production (needs CFBD_API_KEY)",
         critical=False, uses_claude=False),
    Step("news", "backend.ingestion.chunker",
         "Injury status + RotoWire news chunks", critical=False, uses_claude=False),
    Step("synthesis", "backend.ingestion.fetch_synthesis",
         "Claude 'what it means' analysis for veterans", critical=False, uses_claude=True),
    Step("rookies", "backend.ingestion.fetch_rookie_synthesis",
         "Claude analysis for rookies with no NFL snaps", critical=False, uses_claude=True),
]


def _plan(with_ai: bool, only: list[str] | None) -> list[Step]:
    steps = _STEPS if with_ai else [s for s in _STEPS if not s.uses_claude]
    if only:
        wanted = {n.lower() for n in only}
        unknown = wanted - {s.name for s in _STEPS}
        if unknown:
            raise SystemExit(
                f"Unknown step(s): {', '.join(sorted(unknown))}. "
                f"Valid: {', '.join(s.name for s in _STEPS)}"
            )
        # Filter the full list, not `steps`, so `--only synthesis` works
        # without also needing --with-ai. Asking for a step by name is
        # unambiguous consent to run it.
        steps = [s for s in _STEPS if s.name in wanted]
    return steps


def _run(step: Step) -> tuple[bool, float]:
    """Runs one step as a subprocess, streaming its output.

    Subprocess rather than importing and calling main(): each script owns its
    own argparse and logging setup, and a hard crash (or a sys.exit deep in
    one of them) stays contained instead of taking down the whole refresh.
    """
    cmd = [sys.executable, "-m", step.module, *step.args]
    print(f"\n{'=' * 72}\n>> {step.name}: {step.label}\n   {' '.join(cmd)}\n{'=' * 72}",
          flush=True)
    started = time.monotonic()
    try:
        result = subprocess.run(cmd, cwd=_REPO_ROOT)
        ok = result.returncode == 0
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"!! {step.name} could not be launched: {e}", flush=True)
        ok = False
    return ok, time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh every data source, in dependency order.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Steps: " + ", ".join(f"{s.name}" for s in _STEPS),
    )
    parser.add_argument(
        "--with-ai", action="store_true",
        help="Also run the two Claude synthesis steps. These call the Anthropic "
             "API once per player and cost money; everything else is free.",
    )
    parser.add_argument(
        "--only", nargs="+", metavar="STEP", default=None,
        help="Run only these steps (by name), skipping dependency ordering. "
             "For re-running a single step that failed.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan and exit without running anything.",
    )
    args = parser.parse_args()

    steps = _plan(args.with_ai, args.only)

    print(f"Refresh plan ({len(steps)} step(s)):")
    for i, s in enumerate(steps, 1):
        tag = "  [CLAUDE API — costs money]" if s.uses_claude else ""
        print(f"  {i}. {s.name:<10} {s.label}{tag}")
    if not args.with_ai and not args.only:
        print("\n  (skipping the Claude synthesis steps — add --with-ai to include them)")
    if args.dry_run:
        return

    results: list[tuple[Step, bool, float]] = []
    aborted = False
    try:
        for step in steps:
            ok, secs = _run(step)
            results.append((step, ok, secs))
            if not ok and step.critical:
                print(f"\n!! '{step.name}' failed and everything after it depends on "
                      f"it — stopping here rather than writing mismatched data on "
                      f"top of a broken player table.", flush=True)
                aborted = True
                break
            if not ok:
                print(f"\n.. '{step.name}' failed; it only enriches the prompt, so "
                      f"continuing without it.", flush=True)
    except KeyboardInterrupt:
        print("\n\nInterrupted — stopping. Already-completed steps are committed; "
              "re-run with --only to pick up where you left off.", flush=True)
        aborted = True

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    for step, ok, secs in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {step.name:<10} {secs:6.1f}s   {step.label}")
    skipped = [s for s in steps if s.name not in {r[0].name for r in results}]
    for s in skipped:
        print(f"  ----  {s.name:<10}    ---   not reached")

    failed = [s.name for s, ok, _ in results if not ok]
    total = sum(secs for _, _, secs in results)
    print(f"\n  {len(results) - len(failed)}/{len(steps)} succeeded in {total:.1f}s total")
    if failed:
        print(f"  failed: {', '.join(failed)}")
        print(f"  retry just those:  py -m backend.ingestion.refresh --only {' '.join(failed)}")
    if not args.with_ai and not args.only:
        print("  note: Claude synthesis was not refreshed (--with-ai to include it)")

    sys.exit(1 if failed or aborted else 0)


if __name__ == "__main__":
    main()
