"""Pipeline Stress Test — synthetic regression injection.

Раз в месяц подсовывает pipeline'у заведомо плохую "гипотезу" и проверяет,
что L3 validator её отклоняет. Если pipeline пропускает — он сломан и
теоретическая защита от регрессии = 0.

Тесты:
  A. extreme_floor_drop  — entry_score_floor 56 -> 5. Любой валидатор должен
                            отклонить (массовый рост FP).
  B. raise_locked_key    — попытка изменить config_key из do_not_touch.
                            L6 approve должен заблокировать.
  C. relax_protected_gate — попытка ослабить gate из do_not_touch.gates.
                            L6 должен заблокировать.

Тест НЕ модифицирует prod config — все hypothesis-файлы пишутся под префиксом
`_synthetic-` и удаляются после прогона.

Usage:
    pyembed\\python.exe files\\pipeline_stress_test.py
    pyembed\\python.exe files\\pipeline_stress_test.py --keep-artifacts
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import os

# Isolate stress test state in a namespace so synthetic runs never pollute
# production logs. Must be set BEFORE importing pipeline_lib so subprocesses
# we spawn inherit it.
os.environ["CRYPTOBOT_PIPELINE_NAMESPACE"] = "stress_test"

import pipeline_lib as PL

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SYNTH_PREFIX = "_synthetic-"


def make_hypothesis(slug: str, payload: dict) -> Path:
    hid = f"{SYNTH_PREFIX}{slug}"
    record = {
        "hypothesis_id":   hid,
        "schema_version":  1,
        "generated_at":    PL.utc_now_iso(),
        "window_days":     7,
        "rank":            1,
        "rule":            payload["rule"],
        "config_key":      payload["config_key"],
        "diff":            payload["diff"],
        "rationale":       f"SYNTHETIC stress test: {slug}",
        "expected_delta":  payload.get("expected_delta", {}),
        "risk":            "designed to be rejected by pipeline",
        "rollback":        "n/a — stress test",
        "validation_required": ["backtest_60d_pareto_sweep"],
        "source_flag":     "synthetic",
        "persistence":     {"days_red": 99, "values": []},
        "severity":        "critical",
        "status":          "pending_validation",
        "validation_report": None,
        "shadow_report":   None,
    }
    p = PL.HYPOTHESES / f"{hid}.json"
    PL.write_json(p, record)
    return p


def run(args: list[str]) -> dict:
    res = subprocess.run(args, cwd=PL.REPO_ROOT, capture_output=True,
                         text=True, timeout=300, encoding="utf-8", errors="replace")
    return {"ok": res.returncode == 0, "stdout": res.stdout, "stderr": res.stderr, "returncode": res.returncode}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extreme_floor_drop() -> dict:
    """L3 validator should REJECT lowering entry_score floor to 5.0."""
    p = make_hypothesis("extreme_floor_drop", {
        "rule": "entry_score_floor_relax",
        "config_key": "ENTRY_SCORE_FLOOR_GLOBAL",
        "diff": {"from": 56.0, "to": 5.0},
        "expected_delta": {"watchlist_top_early_capture_pct": "+0.10..+0.20"},
    })
    r = run([str(PL.PYEMBED), str(PL.FILES_DIR / "pipeline_validator.py"),
             "--hypothesis-file", str(p)])
    hyp_after = PL.read_json(p) or {}
    verdict = (hyp_after.get("validation_report") or {}).get("result", {}).get("verdict")
    return {
        "test": "extreme_floor_drop",
        "expected_verdict": "reject",
        "actual_verdict":   verdict,
        "pass": verdict == "reject",
        "artifact": str(p),
        "stdout_tail": "\n".join(r["stdout"].splitlines()[-5:]),
    }


def test_locked_config_key() -> dict:
    """L6 approve should REFUSE non-force approval of locked config_key."""
    locked = PL.load_do_not_touch().get("config_keys_locked", [])
    if not locked:
        return {"test": "locked_config_key", "skip": True, "reason": "no locked keys in do_not_touch.json"}
    key = locked[0]
    p = make_hypothesis("locked_key_attempt", {
        "rule": "tighten_proba_synth",
        "config_key": key,
        "diff": {"from": "X", "to": "Y"},
        "expected_delta": {},
    })
    # Manually pre-fill a fake validation_report so safety_checks only flags the lock
    h = PL.read_json(p) or {}
    h["validation_report"] = {"result": {"verdict": "accept", "reason": "synthetic"}}
    PL.write_json(p, h)

    r = run([str(PL.PYEMBED), str(PL.FILES_DIR / "pipeline_approve.py"),
             "--hypothesis-file", str(p), "--approve", "--reason", "stress test"])
    # In approve mode without --force, safety_checks must abort with exit code 2
    return {
        "test": "locked_config_key",
        "expected_returncode": 2,
        "actual_returncode":   r["returncode"],
        "pass": r["returncode"] == 2,
        "artifact": str(p),
        "stdout_tail": "\n".join(r["stdout"].splitlines()[-10:]),
    }


def test_protected_gate_relax() -> dict:
    """L3 validate_gate_threshold should REJECT relaxing a do_not_touch gate."""
    gates = PL.load_do_not_touch().get("gates", [])
    if not gates:
        return {"test": "protected_gate_relax", "skip": True, "reason": "no gates in do_not_touch"}
    gate = gates[0]["name"]
    p = make_hypothesis("protected_gate_relax", {
        "rule": f"relax_gate_{gate}",
        "config_key": f"GATE_{gate.upper()}_THRESHOLD",
        "diff": {"from": "current", "to": "+10% looser"},
    })
    r = run([str(PL.PYEMBED), str(PL.FILES_DIR / "pipeline_validator.py"),
             "--hypothesis-file", str(p)])
    hyp_after = PL.read_json(p) or {}
    verdict = (hyp_after.get("validation_report") or {}).get("result", {}).get("verdict")
    return {
        "test": "protected_gate_relax",
        "expected_verdict": "reject",
        "actual_verdict":   verdict,
        "pass": verdict == "reject",
        "artifact": str(p),
        "stdout_tail": "\n".join(r["stdout"].splitlines()[-5:]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-artifacts", action="store_true",
                    help="don't delete synthetic hypothesis files after run")
    args = ap.parse_args()

    print("[stress] running 3 synthetic regression tests...")
    results = [
        test_extreme_floor_drop(),
        test_locked_config_key(),
        test_protected_gate_relax(),
    ]

    n_pass = sum(1 for r in results if r.get("pass"))
    n_skip = sum(1 for r in results if r.get("skip"))
    n_fail = len(results) - n_pass - n_skip

    summary = {
        "run_id": f"stress-{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')}",
        "generated_at": PL.utc_now_iso(),
        "total": len(results), "passed": n_pass, "failed": n_fail, "skipped": n_skip,
        "tests": results,
        "verdict": "pipeline_healthy" if n_fail == 0 else "pipeline_broken",
    }
    out = PL.PIPELINE / "stress" / f"{summary['run_id']}.json"
    PL.write_json(out, summary)

    if not args.keep_artifacts:
        for r in results:
            art = r.get("artifact")
            if art and Path(art).exists():
                Path(art).unlink()
        print("[stress] cleaned synthetic artifacts")

    print()
    for r in results:
        status = "SKIP" if r.get("skip") else ("PASS" if r.get("pass") else "FAIL")
        print(f"  [{status}] {r['test']}: expected={r.get('expected_verdict') or r.get('expected_returncode')}  actual={r.get('actual_verdict') or r.get('actual_returncode')}")
    print()
    print(f"[stress] summary: {n_pass} pass / {n_fail} fail / {n_skip} skip → {summary['verdict']}")
    print(f"[stress] wrote {out}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
