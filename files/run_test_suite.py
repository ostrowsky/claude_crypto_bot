"""Run the whole test suite and fail only on NEW breakage.

The other bot gates its restart on a green suite (`restart_full_stack.bat`
step 0/8). Ours cannot: discovery over `test_*.py` currently reports 757 tests
with 37 failures and 3 errors, all pre-existing. Gating on green would brick the
restart procedure; ignoring tests entirely is how a silent regression ships.

So this compares against a recorded baseline of known-failing test ids and exits
non-zero only when a test fails that was passing before — the same shape as the
pre-commit harness, which blocks staged files and merely warns about legacy debt.

    pyembed\\python.exe files\\run_test_suite.py            # regression gate
    pyembed\\python.exe files\\run_test_suite.py --update   # re-record baseline
    pyembed\\python.exe files\\run_test_suite.py --strict   # require green

The baseline is a debt register, not a target. `--update` after a change that
fixes tests shrinks it; growing it needs a reason stated in the commit.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Committed on purpose: this is a debt register, not runtime state. Kept out of
# .runtime/ so a fresh clone does not read an empty baseline and call all 40
# known failures new.
BASELINE = HERE / "test_baseline.json"


def run_suite() -> tuple[set[str], int]:
    """Return (failing test ids, total tests run).

    Runs with `files/` as the working directory. Everything here resolves data
    paths relative to it (CLAUDE.md §6), so discovering from the repo root
    turned 40 real failures into 148 phantom ones — the gate would then treat
    its own misconfiguration as project debt.
    """
    import contextlib
    import os

    loader = unittest.TestLoader()
    buf = io.StringIO()
    prev = os.getcwd()
    os.chdir(HERE)
    try:
        suite = loader.discover(start_dir=".", pattern="test_*.py", top_level_dir=".")
        runner = unittest.TextTestRunner(stream=buf, verbosity=0)
        # test modules print their own reports; that output is not the gate's
        with contextlib.redirect_stdout(io.StringIO()):
            result = runner.run(suite)
    finally:
        os.chdir(prev)
    failing = {str(test) for test, _ in result.failures}
    failing |= {str(test) for test, _ in result.errors}
    # Import-time explosions surface as _FailedTest entries; keep them, they are
    # the loudest kind of breakage.
    return failing, result.testsRun


def load_baseline() -> set[str]:
    if not BASELINE.exists():
        return set()
    try:
        data = json.loads(BASELINE.read_text(encoding="utf-8"))
        return set(data.get("failing", []))
    except (json.JSONDecodeError, OSError):
        return set()


def save_baseline(failing: set[str], total: int) -> None:
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps({
        "failing": sorted(failing),
        "n_failing": len(failing),
        "n_tests": total,
    }, indent=1, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="test suite regression gate")
    ap.add_argument("--update", action="store_true", help="re-record the baseline")
    ap.add_argument("--strict", action="store_true", help="require a fully green suite")
    args = ap.parse_args(argv)

    failing, total = run_suite()
    baseline = load_baseline()

    if args.update:
        save_baseline(failing, total)
        print(f"baseline recorded: {len(failing)} known-failing of {total} tests")
        return 0

    new = sorted(failing - baseline)
    fixed = sorted(baseline - failing)

    print(f"tests: {total}  failing: {len(failing)}  baseline: {len(baseline)}")
    if fixed:
        print(f"FIXED since baseline ({len(fixed)}) — run --update to bank them:")
        for name in fixed[:10]:
            print(f"  + {name}")
    if new:
        print(f"NEW FAILURES ({len(new)}):")
        for name in new:
            print(f"  - {name}")
        return 1
    if args.strict and failing:
        print(f"STRICT: {len(failing)} tests still failing")
        return 1
    print("no new failures" + (" (pre-existing debt unchanged)" if failing else ""))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
