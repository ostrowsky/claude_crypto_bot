"""Daily aggregator: runs all metric backtest scripts, parses METRIC_JSON
lines from their output, appends a single JSON row to
.runtime/metrics_daily.jsonl.

Designed to be invoked at end of daily_learning.py or as standalone.

2026-10-20: the row is now written whatever happens to the scripts. From
2026-10-12 to 10-19 the orchestrator's L0 step (600 s) killed this script
every morning: nine scripts ran in sequence with 180 s each (up to 1 620 s),
and the row was only written after the last one, so a kill meant no row at
all -- and the morning report showed the 10-11 North Star as today's, marked
verified. Now the scripts run against one overall budget that ends before the
caller's timeout, the North Star goes first, and a script the budget did not
reach is recorded as `skipped_budget`, not left out (TH-05: a metric must
know what it does not know).
"""
from __future__ import annotations
import argparse
import subprocess, json, sys, io, time
from pathlib import Path
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
PYEMBED = ROOT / "pyembed" / "python.exe"
OUT_PATH = ROOT / ".runtime" / "metrics_daily.jsonl"

# Overall wall-clock budget for all scripts. pipeline_run.py gives this step
# L0_METRICS_TIMEOUT_SEC; the budget must end before it with room to write the
# row (test_report_metrics_daily.py checks that).
DEFAULT_BUDGET_SEC = 1500
PER_SCRIPT_TIMEOUT_SEC = 300

SCRIPTS = [
    # (filename, args). It was a flat list of names and subprocess.run
    # passed no arguments, so `--use-zigzag` could not be set no matter
    # what anyone intended — the daily EX1 has always been the proxy.
    # A metric computable only one way because the CALLER cannot express
    # the other is a caller defect, not a metric defect.
    # The North Star first: if the budget runs out, it is the one that must exist.
    ("_compute_early_capture.py", []),
    ("_backtest_top20_coverage_funnel.py", []),
    ("_backtest_fast_reversal_by_mode.py", []),
    ("_backtest_signal_precision.py", []),
    ("_backtest_whipsaw_rate.py", []),
    ("_backtest_time_to_signal.py", []),
    ("_backtest_capture_ratio.py", []),
    ("_backtest_ex1_realized_potential.py", ["--use-zigzag"]),
    ("_backtest_net_realized_pnl.py", []),
]


def _run_subprocess(args: list[str], timeout: float) -> tuple[int, str]:
    out = subprocess.run(args, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=timeout)
    return out.returncode, (out.stdout or "") + (out.stderr or "")


def parse_metric(text: str) -> dict | None:
    metric = None
    for ln in text.splitlines():
        if ln.startswith("METRIC_JSON:"):
            try:
                metric = json.loads(ln[len("METRIC_JSON:"):])
            except Exception:
                pass
    return metric


def build_row(scripts=SCRIPTS, budget_sec: float = DEFAULT_BUDGET_SEC,
              per_script_timeout: float = PER_SCRIPT_TIMEOUT_SEC,
              runner=_run_subprocess, clock=time.monotonic,
              files_dir: Path = ROOT / "files", python: Path = PYEMBED) -> dict:
    """Run every script inside one budget; every script gets an entry."""
    started = clock()
    row = {"ts": datetime.now(timezone.utc).isoformat(), "budget_sec": budget_sec}
    for s, script_args in scripts:
        p = files_dir / s
        if not p.exists():
            row[s] = {"error": "missing"}
            continue
        remaining = budget_sec - (clock() - started)
        if remaining <= 1:
            row[s] = {"error": "skipped_budget"}
            continue
        t0 = clock()
        try:
            rc, text = runner([str(python), str(p), *script_args],
                              min(per_script_timeout, remaining))
            metric = parse_metric(text)
            if metric is None:
                row[s] = {"error": "no METRIC_JSON line", "exit": rc}
            else:
                row[s] = metric
        except subprocess.TimeoutExpired:
            row[s] = {"error": "timeout"}
        except Exception as ex:
            row[s] = {"error": str(ex)[:200]}
        row[s]["elapsed_sec"] = round(clock() - t0, 1)
    row["elapsed_sec"] = round(clock() - started, 1)
    return row


def append_row(row: dict, out_path: Path = OUT_PATH) -> None:
    out_path.parent.mkdir(exist_ok=True)
    with io.open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def print_summary(row: dict, out_path: Path, scripts=SCRIPTS) -> None:
    print(f"=== Metrics row appended: {out_path} ===")
    print(f"timestamp: {row['ts']}  elapsed: {row.get('elapsed_sec')}s of {row.get('budget_sec')}s")
    for s, _ in scripts:
        m = row.get(s, {})
        if "error" in m:
            print(f"  {s:<45} ERR: {m['error']}")
        else:
            # Pull a key value depending on metric type
            name = m.get("metric", "?")
            if "early_capture" in m: v = f"{m['early_capture']:.3f} (north-star)"
            elif "precision_pct" in m: v = f"prec={m['precision_pct']:.1f}% rate={m.get('raw_entries_per_day',0):.1f}/d"
            elif "median_h" in m: v = f"median lead {m['median_h']:+.2f}h, late {m.get('late_30m_pct',0):.0f}%"
            elif "overall_pct" in m: v = f"overall {m['overall_pct']:.1f}%"
            elif "top20" in m and isinstance(m.get("top20"), dict) and m["top20"]:
                t = m["top20"]
                if "share_ex1_ge_05" in t:
                    v = f"top20 EX1 median={t['median']:+.3f}, ex1>=0.5: {t['share_ex1_ge_05']:.0f}%"
                else:
                    v = f"top20 mean={t['mean']:+.3f}"
            else: v = "ok"
            print(f"  {s:<45} {name:<28} {v}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-sec", type=float, default=DEFAULT_BUDGET_SEC)
    args = ap.parse_args(argv)
    row = build_row(budget_sec=args.budget_sec)
    append_row(row)
    print_summary(row, OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
