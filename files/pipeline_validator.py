"""L3 — Backtest Validator (numerical config diffs).

Принимает hypothesis с `config_key` + `diff` и валидирует на ≥60 дней истории.

Архитектура — dispatcher:
  - для каждого config_key (или семейства gate_*) есть свой validator-функция
  - validator делает Pareto sweep вокруг proposed value, считает recall@20 + Sharpe×√n
  - возвращает структуру validation_report, которая прикрепляется к hypothesis

Reject criteria (общие, могут переопределяться per-validator):
  - recall@20 на hold-out 14 дней падает > 2pp → REJECT
  - Sharpe×√n не растёт vs current → REJECT
  - delta вне expected_delta range → WARNING (но не reject)

Если validator для config_key не зарегистрирован — emit `pending_manual_validation`
с инструкцией: какой backtest скрипт надо написать.

Usage:
    pyembed\\python.exe files\\pipeline_validator.py --hypothesis h-2026-05-11-foo
    pyembed\\python.exe files\\pipeline_validator.py --hypothesis-file path/to/h.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pipeline_lib as PL


REJECT_RECALL_DROP_PP = 0.02       # > 2pp recall@20 drop = reject
REJECT_IF_SHARPE_NOT_IMPROVED = True
GRID_POINTS = 5
HOLDOUT_DAYS = 14
BACKTEST_DAYS = 60


# ---------------------------------------------------------------------------
# Validators — one per supported config_key family
# Each returns a validation_report dict.
# ---------------------------------------------------------------------------


def _make_grid(lo: float, hi: float, n: int = GRID_POINTS) -> list[float]:
    if n < 2:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [round(lo + step * i, 4) for i in range(n)]


def _run_existing_script(script_name: str, extra_args: list[str] | None = None,
                          full_stdout: bool = False) -> dict:
    """Run an existing backtest script, capture stdout+stderr.

    By default stdout is truncated to last 2000 chars (diagnostic context).
    Pass full_stdout=True when caller needs to parse the entire output (e.g. JSON).
    """
    script = PL.FILES_DIR / script_name
    if not script.exists():
        return {"ok": False, "error": f"script not found: {script}"}
    try:
        res = subprocess.run(
            [str(PL.PYEMBED), str(script)] + (extra_args or []),
            cwd=PL.REPO_ROOT, capture_output=True, text=True, timeout=300,
            encoding="utf-8", errors="replace",
        )
        return {
            "ok": res.returncode == 0,
            "stdout": res.stdout if full_stdout else res.stdout[-2000:],
            "stderr": res.stderr[-500:],
            "returncode": res.returncode,
        }
    except (subprocess.TimeoutExpired, OSError) as e:
        return {"ok": False, "error": str(e)}


def validate_entry_score_floor(hyp: dict) -> dict:
    """Real replay: call _replay_entry_score.py with grid around proposed value.

    Uses bot_events.jsonl entries blocked by entry_score (their candidate_score and
    ranker_top_gainer_prob are known) to project what would happen at each floor.
    """
    diff = hyp.get("diff", {})
    cur = diff.get("from")
    target = diff.get("to")
    if cur is None or target is None:
        return {"verdict": "reject", "reason": "diff.from / diff.to missing"}

    grid_vals = _make_grid(min(cur, target), max(cur, target))
    grid_csv = ",".join(str(v) for v in grid_vals)

    rep = _run_existing_script("_replay_entry_score.py", [
        "--window-days", str(BACKTEST_DAYS),
        "--grid", grid_csv,
    ], full_stdout=True)
    if not rep.get("ok"):
        return {
            "validator": "validate_entry_score_floor",
            "verdict": "pending_manual_validation",
            "reason": "_replay_entry_score.py failed",
            "details": rep,
        }

    try:
        replay_data = json.loads(rep["stdout"])
    except (json.JSONDecodeError, KeyError):
        return {
            "validator": "validate_entry_score_floor",
            "verdict": "pending_manual_validation",
            "reason": "could not parse replay output",
            "raw_stdout": rep.get("stdout", "")[:1000],
        }

    grading = replay_data.get("grading", {})
    return {
        "validator": "validate_entry_score_floor",
        "verdict": grading.get("verdict", "pending_manual_validation"),
        "reason":  grading.get("reason", "replay completed"),
        "best_grid_point": grading.get("best"),
        "grid_results":    replay_data.get("grid_results"),
        "current_floor":   replay_data.get("current_floor"),
        "floor_distribution": replay_data.get("floor_distribution"),
        "n_blocked_events_in_window": replay_data.get("total_blocked_by_entry_score"),
        "window_days": replay_data.get("window_days"),
    }


def validate_gate_threshold(hyp: dict) -> dict:
    """Generic gate-threshold relax — uses analyze_blocked_gates output as evidence."""
    gate = hyp.get("rule", "").replace("relax_gate_", "")
    if not gate:
        return {"verdict": "reject", "reason": "could not extract gate name from rule"}

    dnt = {g["name"] for g in PL.load_do_not_touch().get("gates", [])}
    if gate in dnt:
        return {"verdict": "reject", "reason": f"gate '{gate}' is in do_not_touch list"}

    ag = _run_existing_script("analyze_blocked_gates.py")
    return {
        "validator": "validate_gate_threshold",
        "gate": gate,
        "verdict": "pending_manual_validation",
        "reason": "v1 emits diagnostic; needs per-gate replay script",
        "manual_steps": [
            f"1. Locate gate '{gate}' in files/trend_scout_rules.py or monitor.py",
            f"2. Identify threshold constant in files/config.py",
            f"3. Write _backtest_relax_{gate}.py that replays bot_events with threshold sweep",
            f"4. Reject if recall@20 drops > {REJECT_RECALL_DROP_PP:.0%}",
        ],
        "raw_diagnostic": ag.get("stdout", "")[:800] if ag.get("ok") else None,
    }


def validate_tighten_proba(hyp: dict) -> dict:
    """Tighten ML proba threshold for a losing mode."""
    return {
        "validator": "validate_tighten_proba",
        "verdict": "pending_manual_validation",
        "reason": "v1 emits checklist; needs per-mode replay",
        "manual_steps": [
            f"1. config_key={hyp.get('config_key')}: identify current proba_min value",
            f"2. Sweep [current, current+0.02, current+0.05, current+0.08]",
            f"3. For each, use _backtest_signal_precision.py with --mode-filter <mode> --proba-min <v>",
            f"4. Pick smallest tightening where mode realized_pnl_pct > 0 on hold-out",
        ],
    }


# Dispatcher: rule → validator
VALIDATORS: dict[str, Callable[[dict], dict]] = {
    "entry_score_floor_relax":   validate_entry_score_floor,
}
# Prefix-based dispatchers
PREFIX_VALIDATORS = [
    ("relax_gate_",     validate_gate_threshold),
    ("tighten_proba_",  validate_tighten_proba),
]


def dispatch(hyp: dict) -> dict:
    rule = hyp.get("rule", "")
    if rule in VALIDATORS:
        return VALIDATORS[rule](hyp)
    for prefix, fn in PREFIX_VALIDATORS:
        if rule.startswith(prefix):
            return fn(hyp)
    return {
        "verdict": "pending_manual_validation",
        "reason": f"no validator registered for rule='{rule}'",
        "manual_steps": [
            f"Add a validator for rule '{rule}' to files/pipeline_validator.py",
            f"Register it in VALIDATORS or PREFIX_VALIDATORS",
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypothesis", help="hypothesis_id (looked up in .runtime/pipeline/hypotheses)")
    ap.add_argument("--hypothesis-file", help="explicit path to hypothesis JSON")
    ap.add_argument("--print", dest="do_print", action="store_true")
    args = ap.parse_args()

    if args.hypothesis_file:
        hp = Path(args.hypothesis_file)
    elif args.hypothesis:
        hp = PL.HYPOTHESES / f"{args.hypothesis}.json"
    else:
        ap.error("--hypothesis or --hypothesis-file required")

    hyp = PL.read_json(hp)
    if not hyp:
        print(f"[L3] hypothesis not found: {hp}")
        sys.exit(1)

    print(f"[L3] validating {hyp['hypothesis_id']} (rule={hyp.get('rule')})", file=sys.stderr)
    vr = dispatch(hyp)
    report = {
        "validator_run_id": f"val-{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')}",
        "generated_at": PL.utc_now_iso(),
        "hypothesis_id": hyp["hypothesis_id"],
        "config": {
            "reject_recall_drop_pp": REJECT_RECALL_DROP_PP,
            "grid_points": GRID_POINTS,
            "holdout_days": HOLDOUT_DAYS,
            "backtest_days": BACKTEST_DAYS,
        },
        "result": vr,
    }

    hyp["validation_report"] = report
    if vr.get("verdict") == "accept":
        hyp["status"] = "validated"
    elif vr.get("verdict") == "reject":
        hyp["status"] = "rejected"
    else:
        hyp["status"] = "pending_validation"
    PL.write_json(hp, hyp)
    print(f"[L3] verdict={vr.get('verdict')} status={hyp['status']}")
    print(f"[L3] attached to {hp}")

    if args.do_print:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
