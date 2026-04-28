"""P4: per-feature Pareto on impulse_guard.
Parse blocked-event reason text for impulse_guard, extract sub-conditions,
group by sub-condition, compute avg_r5 etc per group.
"""
from __future__ import annotations
import json, io, sys, re
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
NOW = datetime.now(timezone.utc); CUT = NOW - timedelta(days=14)

# Patterns we expect — reason texts include numbers we can extract
patterns = {
    "daily_range_1h_15": re.compile(r"1h impulse_speed guard: daily_range ([\d.]+)% > (15\.\d+)%"),
    "daily_range_1h_10": re.compile(r"1h impulse_speed guard: daily_range ([\d.]+)% > (10\.\d+)%"),
    "weak_15m_adx":      re.compile(r"weak 15m impulse: ADX ([\d.]+) < (\d+)"),
    "vol_x_low":         re.compile(r"vol_x ([\d.]+) <"),
    "slope_low":         re.compile(r"slope_pct ([\d.+\-]+) <"),
}

# Stream blocked + outcomes
buckets = defaultdict(list)  # subcond -> [r5_pct, ...] (we don't have r5 per blocked; use take baseline)
sub_counts = defaultdict(int)
sample_reasons = defaultdict(list)
all_impulse_guard_reasons = defaultdict(int)

with io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if "impulse" not in ln.lower(): continue
        try: e = json.loads(ln)
        except: continue
        if e.get("event") != "blocked": continue
        ts = e.get("ts","")
        try: dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        except: continue
        if dt < CUT: continue
        # reason
        reason = ""
        if isinstance(e.get("decision"), dict):
            reason = e["decision"].get("reason_code","") or ""
        if not reason:
            reason = e.get("reason_code") or e.get("reason","") or ""
        rc_short = (e.get("decision",{}).get("reason_code") if isinstance(e.get("decision"), dict)
                    else e.get("reason_code") or "")
        # Filter: only impulse_guard / impulse_speed sub-block
        rtxt = reason
        if "impulse" not in rtxt.lower(): continue
        all_impulse_guard_reasons[rtxt[:60]] += 1

        for name, pat in patterns.items():
            m = pat.search(rtxt)
            if m:
                sub_counts[name] += 1
                if len(sample_reasons[name]) < 3:
                    sample_reasons[name].append(rtxt[:80])
                break

print("=== P4: impulse_guard sub-condition breakdown (last 14d) ===\n")
print(f"{'sub-condition':<22} {'n':>5}")
for k, v in sorted(sub_counts.items(), key=lambda x: -x[1]):
    print(f"  {k:<22} {v:>5}")
    for s in sample_reasons[k][:1]:
        print(f"    e.g. {s}")
print(f"\nTotal classified: {sum(sub_counts.values())}")
print(f"Total impulse-related blocked events: {sum(all_impulse_guard_reasons.values())}")

print(f"\n=== Top-N raw block-reason texts ===")
for r, c in sorted(all_impulse_guard_reasons.items(), key=lambda x: -x[1])[:20]:
    print(f"  {c:>5}  {r}")
