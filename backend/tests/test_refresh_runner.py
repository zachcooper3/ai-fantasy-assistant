"""
backend.ingestion.refresh — step planning and failure policy.

The runner's whole value is that it can't be got wrong by hand, so the two
things worth pinning are: the Claude-API steps never run unless asked for
(they cost money), and a failure in a step everything depends on stops the
run instead of writing mismatched data on top of a broken player table.

Steps are exercised as real subprocesses against stdlib modules — `this`
always exits 0, a nonexistent module always exits 1 — so the actual
subprocess/returncode path is covered rather than mocked away.
"""

import pytest

from backend.ingestion import refresh
from backend.ingestion.refresh import Step, _plan, _STEPS

_OK = "this"                    # stdlib, exits 0
_FAIL = "no_such_module_xyz"    # does not exist, exits 1


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def test_claude_steps_are_excluded_by_default():
    names = [s.name for s in _plan(with_ai=False, only=None)]
    assert "synthesis" not in names and "rookies" not in names
    assert names, "the free steps must still be planned"


def test_with_ai_includes_the_claude_steps_last():
    plan = _plan(with_ai=True, only=None)
    names = [s.name for s in plan]
    assert names[-2:] == ["synthesis", "rookies"]
    # Synthesis reads PlayerMetrics and DraftProfile, so it can only run
    # after the steps that populate them.
    assert names.index("metrics") < names.index("synthesis")
    assert names.index("draft") < names.index("rookies")


def test_dependency_order_is_preserved():
    names = [s.name for s in _plan(with_ai=True, only=None)]
    # fetch_adp truncates and reloads Player; everything keys off sleeper_ids.
    assert names.index("adp") == 0
    assert names.index("ids") < names.index("metrics")
    assert names.index("ids") < names.index("news")
    # fetch_college_stats only enriches rows fetch_draft_profiles created.
    assert names.index("draft") < names.index("college")


def test_only_runs_the_named_steps():
    plan = _plan(with_ai=False, only=["metrics", "news"])
    assert [s.name for s in plan] == ["metrics", "news"]


def test_only_can_name_a_claude_step_without_with_ai():
    # Naming a step explicitly is unambiguous consent to run it; requiring
    # both flags would just be a papercut when retrying a failed step.
    plan = _plan(with_ai=False, only=["synthesis"])
    assert [s.name for s in plan] == ["synthesis"]


def test_only_is_case_insensitive():
    assert [s.name for s in _plan(False, ["METRICS"])] == ["metrics"]


def test_unknown_step_name_is_rejected_with_the_valid_list():
    with pytest.raises(SystemExit) as e:
        _plan(with_ai=False, only=["bogus"])
    assert "bogus" in str(e.value)
    assert "metrics" in str(e.value)  # tells you what IS valid


def test_every_step_is_individually_addressable():
    # --only is the documented retry path, so every step must be reachable
    # by name or a failure becomes unrecoverable without editing the file.
    for s in _STEPS:
        assert [x.name for x in _plan(True, [s.name])] == [s.name]


# ---------------------------------------------------------------------------
# Failure policy
# ---------------------------------------------------------------------------

def _run_with(monkeypatch, steps, argv=("refresh",)):
    monkeypatch.setattr(refresh, "_STEPS", steps)
    monkeypatch.setattr(refresh.sys, "argv", list(argv))
    with pytest.raises(SystemExit) as e:
        refresh.main()
    return e.value.code


def test_critical_failure_stops_the_run(monkeypatch, capsys):
    code = _run_with(monkeypatch, [
        Step("a", _OK, "fine", critical=True, uses_claude=False),
        Step("boom", _FAIL, "critical failure", critical=True, uses_claude=False),
        Step("downstream", _OK, "must not run", critical=False, uses_claude=False),
    ])
    out = capsys.readouterr().out
    assert code == 1
    assert "not reached" in out
    # The point: the downstream step was never executed, not merely reported.
    assert "OK    downstream" not in out


def test_optional_failure_does_not_stop_the_run(monkeypatch, capsys):
    code = _run_with(monkeypatch, [
        Step("a", _OK, "fine", critical=True, uses_claude=False),
        Step("soft", _FAIL, "optional", critical=False, uses_claude=False),
        Step("c", _OK, "must still run", critical=False, uses_claude=False),
    ])
    out = capsys.readouterr().out
    assert code == 1                 # still reports failure
    assert "OK    c" in out          # but the run continued
    assert "not reached" not in out


def test_clean_run_exits_zero(monkeypatch, capsys):
    code = _run_with(monkeypatch, [
        Step("a", _OK, "fine", critical=True, uses_claude=False),
        Step("b", _OK, "also fine", critical=False, uses_claude=False),
    ])
    assert code == 0
    assert "2/2 succeeded" in capsys.readouterr().out


def test_failure_summary_names_the_retry_command(monkeypatch, capsys):
    _run_with(monkeypatch, [
        Step("soft", _FAIL, "optional", critical=False, uses_claude=False),
    ])
    assert "--only soft" in capsys.readouterr().out


def test_dry_run_executes_nothing(monkeypatch, capsys):
    monkeypatch.setattr(refresh, "_STEPS", [
        Step("boom", _FAIL, "would fail if actually run",
             critical=True, uses_claude=False),
    ])
    monkeypatch.setattr(refresh.sys, "argv", ["refresh", "--dry-run"])
    refresh.main()  # must not raise SystemExit
    out = capsys.readouterr().out
    assert "Refresh plan" in out
    assert "SUMMARY" not in out


def test_plan_warns_that_claude_steps_were_skipped(monkeypatch, capsys):
    monkeypatch.setattr(refresh.sys, "argv", ["refresh", "--dry-run"])
    refresh.main()
    assert "--with-ai" in capsys.readouterr().out
