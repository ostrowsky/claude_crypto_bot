"""Daily aggregator: runs all metric backtest scripts, parses METRIC_JSON
lines from their output, appends a single JSON row to
.runtime/metrics_daily.jsonl.

Designed to be invoked at end of daily_learning.py or as standalone.
"""
from __future__ import annotations
import subprocess, json, sys, io
from pathlib import Path
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
PYEMBED = ROOT / "pyembed" / "python.exe"

SCRIPTS = [
    "_backtest_top20_coverage_funnel.py",   # C1, C2
    "_backtest_fast_reversal_by_mode.py",   # Q1, Q3
    "_backtest_signal_precision.py",        # D1, D2
    "_backtest_whipsaw_rate.py",            # Q2
    "_backtest_time_to_signal.py",          # E1
    "_backtest_capture_ratio.py",           # E2
    "_compute_early_capture.py",            # north-star
]

row = {"ts": datetime.now(timezone.utc).isoformat()}
for s in SCRIPTS:
    p = ROOT / "files" / s
    if not p.exists():
        row[s] = {"error": "missing"}
        continue
    try:
        out = subprocess.run([str(PYEMBED), str(p)], capture_output=True,
                             text=True, encoding="utf-8", timeout=180)
        text = (out.stdout or "") + (out.stderr or "")
        # Find METRIC_JSON: lines
        metric = None
        for ln in text.splitlines():
            if ln.startswith("METRIC_JSON:"):
                try: metric = json.loads(ln[len("METRIC_JSON:"):])
                except: pass
        if metric is None:
            row[s] = {"error": "no METRIC_JSON line", "exit": out.returncode}
        else:
            row[s] = metric
    except subprocess.TimeoutExpired:
        row[s] = {"error": "timeout"}
    except Exception as ex:
        row[s] = {"error": str(ex)[:200]}

# Append
out_path = ROOT / ".runtime" / "metrics_daily.jsonl"
out_path.parent.mkdir(exist_ok=True)
with io.open(out_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(row) + "\n")

# Print compact summary
print(f"=== Metrics row appended: {out_path} ===")
print(f"timestamp: {row['ts']}")
for s in SCRIPTS:
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
        elif "top20" in m and m["top20"]: v = f"top20 mean={m['top20']['mean']:+.3f}"
        else: v = "ok"
        print(f"  {s:<45} {name:<28} {v}")
