"""Pipeline Orchestrator — единственная точка входа для расписания.

Запускает все слои в правильной последовательности.

Режимы:
    --daily   (default): L1 health -> L4 shadow snapshot -> L5 critic -> L7 monitor
    --weekly:            всё что daily + L2 hypotheses -> L3 validate each
    --bootstrap:         как weekly, но с --min-persistent-days 1 (для bootstrap <7d данных)

Usage:
    pyembed\\python.exe files\\pipeline_run.py --daily
    pyembed\\python.exe files\\pipeline_run.py --weekly
    pyembed\\python.exe files\\pipeline_run.py --bootstrap

Exit codes:
    0 — все слои отработали (даже если L2 не выдал гипотез)
    1 — фатальная ошибка хоть в одном слое (script crash, не verdict)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pipeline_lib as PL

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ORCH_DIR = PL.PIPELINE / "orch"
ORCH_DIR.mkdir(parents=True, exist_ok=True)


def run_step(name: str, args: list[str], timeout: int = 300) -> dict:
    started = time.time()
    print(f"[orch] === {name} ===")
    print(f"[orch] cmd: {' '.join(args)}")
    try:
        res = subprocess.run(
            args, cwd=PL.REPO_ROOT, capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
        duration = time.time() - started
        ok = res.returncode == 0
        print(f"[orch] {name} returncode={res.returncode} duration={duration:.1f}s")
        # Print last lines of stdout to give operator a glimpse without flooding
        last = "\n".join(res.stdout.splitlines()[-10:])
        if last:
            print(f"[orch] tail-stdout:\n{last}")
        if res.stderr.strip():
            last_err = "\n".join(res.stderr.splitlines()[-5:])
            print(f"[orch] tail-stderr:\n{last_err}")
        return {
            "step": name, "ok": ok, "returncode": res.returncode,
            "duration_sec": round(duration, 2),
            "stdout_tail": last, "stderr_tail": res.stderr[-500:],
        }
    except subprocess.TimeoutExpired:
        return {"step": name, "ok": False, "error": "timeout", "duration_sec": timeout}
    except OSError as e:
        return {"step": name, "ok": False, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--daily", action="store_true", help="L1+L4+L5+L7")
    g.add_argument("--weekly", action="store_true", help="daily + L2 hypotheses + L3 validate")
    g.add_argument("--bootstrap", action="store_true", help="weekly with min-persistent=1")
    ap.add_argument("--run-evaluator", action="store_true",
                    help="also run _run_signal_evaluator.py (slow ~3 min)")
    args = ap.parse_args()

    if not (args.daily or args.weekly or args.bootstrap):
        args.daily = True

    started = datetime.now(timezone.utc)
    results = []

    py = str(PL.PYEMBED)
    files = str(PL.FILES_DIR)

    # L1 — daily health
    l1_args = [py, f"{files}/bot_health_report.py"]
    if args.run_evaluator:
        l1_args.append("--run-evaluator")
    results.append(run_step("L1 health", l1_args, timeout=600))

    # L4 — shadow snapshot
    results.append(run_step("L4 shadow", [py, f"{files}/pipeline_shadow.py", "--window-days", "7"]))

    # L5 — blind critic for any unevaluated decisions
    results.append(run_step("L5 critic", [py, f"{files}/pipeline_blind_critic.py", "--all-pending"]))

    # L4 sim refresh — re-run counterfactual sim for every pending hypothesis
    # daily, NOT just weekly. The dataset (critic_dataset.jsonl) grows every
    # day via the EOD learning job, so yesterday's verdict can flip. Skipping
    # this in daily means the operator sees stale numbers in Telegram.
    # Cost: ~5 sec/hypothesis × usually <=3 pending = under 20 sec.
    for hp in sorted(PL.HYPOTHESES.glob("h-*.json")):
        h = PL.read_json(hp) or {}
        if h.get("status") != "pending_validation":
            continue
        hid = h.get("hypothesis_id")
        if not hid:
            continue
        results.append(run_step(
            f"L4 refresh {hid}",
            [py, f"{files}/pipeline_shadow.py", "--simulate-from-hypothesis", hid,
             "--window-days", "60", "--min-events", "30"],
            timeout=300,
        ))

    # L7 — monitor (legacy naive before/after — keep for backward compat)
    results.append(run_step("L7 monitor", [py, f"{files}/pipeline_monitor.py", "--check-after-days", "7"]))

    # Attribution — normalised delta + bootstrap CI for every approved decision
    # that has both a pinned baseline and >= 4 post-apply health reports.
    # Skips silently for fresh decisions ("needs_data" verdict).
    results.append(run_step("attribution", [py, f"{files}/pipeline_attribution.py", "--all-due", "--days-after", "14"]))

    # Telegram delivery — sends today's L1 health.tg.txt + brief attribution
    # block. Idempotent by day via tg_send_dedup.json. Silently no-ops if
    # token/chat_ids/health report are missing — never blocks the orchestrator.
    results.append(run_step("notify", [py, f"{files}/pipeline_notify.py"]))

    if args.weekly or args.bootstrap:
        l2_args = [py, f"{files}/pipeline_hypothesis.py", "--window-days", "7", "--max", "3"]
        if args.bootstrap:
            l2_args += ["--min-persistent-days", "1"]
        results.append(run_step("L2 hypotheses", l2_args))

        # L3 — validate each fresh hypothesis (status=pending_validation)
        for hp in sorted(PL.HYPOTHESES.glob("h-*.json")):
            h = PL.read_json(hp) or {}
            if h.get("status") != "pending_validation":
                continue
            results.append(run_step(
                f"L3 validate {h.get('hypothesis_id')}",
                [py, f"{files}/pipeline_validator.py", "--hypothesis-file", str(hp)],
                timeout=600,
            ))

        # Weekly does NOT need to re-sim — the daily step above already
        # refreshes every pending hypothesis. This loop used to duplicate it.

    finished = datetime.now(timezone.utc)
    summary = {
        "run_id": f"orch-{started.strftime('%Y-%m-%dT%H%M%SZ')}",
        "mode": "weekly" if args.weekly else ("bootstrap" if args.bootstrap else "daily"),
        "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_at": finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_sec": round((finished - started).total_seconds(), 2),
        "n_steps": len(results),
        "n_failed": sum(1 for r in results if not r.get("ok")),
        "steps": results,
    }
    out = ORCH_DIR / f"{summary['run_id']}.json"
    PL.write_json(out, summary)
    print()
    print(f"[orch] done: {summary['n_steps']} steps, {summary['n_failed']} failed, {summary['duration_sec']}s")
    print(f"[orch] wrote {out}")
    sys.exit(1 if summary["n_failed"] > 0 else 0)


if __name__ == "__main__":
    main()
