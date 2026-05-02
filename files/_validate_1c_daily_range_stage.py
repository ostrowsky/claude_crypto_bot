"""1C: daily_range stage-aware gate.
For each top-20 winner blocked by daily_range guard, determine if it was:
  - early stage (first day this sym hit blocked-by-range threshold)
  - late stage (2nd+ consecutive day blocked-by-range)

Stage-aware gate would:
  - Allow early stage  → recovers winners
  - Block late stage   → keeps protection

Approach: for each (date, sym) blocked by daily_range, look back 3 days.
If sym was NOT blocked yesterday → early stage (allow).
"""
from __future__ import annotations
import json, io, sys, re
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
NOW = datetime.now(timezone.utc); CUT = NOW - timedelta(days=14)

# top-20
top20 = set()
with io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        try: e = json.loads(ln)
        except: continue
        ts_ms = e.get("ts");
        if not ts_ms: continue
        dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
        if dt < CUT: continue
        if e.get("label_top20") == 1:
            top20.add((dt.strftime("%Y-%m-%d"), e.get("symbol")))

# Find blocked-by-daily_range events per (date, sym)
range_pat = re.compile(r"daily_range ([\d.]+)% > ([\d.]+)%")
blocked_dr = defaultdict(list)  # (d, sym) -> [daily_range%]
with io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if "daily_range" not in ln: continue
        try: e = json.loads(ln)
        except: continue
        if e.get("event") != "blocked": continue
        ts = e.get("ts","")
        try: dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        except: continue
        if dt < CUT - timedelta(days=3): continue  # extra lookback
        sym = e.get("sym") or e.get("symbol") or ""
        if not sym: continue
        d = dt.strftime("%Y-%m-%d")
        reason = ""
        if isinstance(e.get("decision"), dict):
            reason = e["decision"].get("reason_code","") or ""
        if not reason: reason = e.get("reason_code") or e.get("reason","") or ""
        m = range_pat.search(reason)
        if m:
            blocked_dr[(d, sym)].append(float(m.group(1)))

# For each top-20 winner blocked by daily_range:
# - check yesterday and 2 days ago for same sym in blocked_dr
# - if NOT in blocked_dr yesterday → early stage
print("=== 1C: daily_range stage-aware ===")
early_stage = []; late_stage = []
for key in top20:
    d, sym = key
    if key not in blocked_dr: continue
    dt = datetime.fromisoformat(d)
    yesterday = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
    day2ago   = (dt - timedelta(days=2)).strftime("%Y-%m-%d")
    blocked_y  = (yesterday, sym) in blocked_dr
    blocked_y2 = (day2ago, sym)   in blocked_dr
    info = {"d": d, "sym": sym,
            "today_max_dr": max(blocked_dr[key]),
            "blocked_yesterday": blocked_y,
            "blocked_day2ago": blocked_y2}
    if not blocked_y:
        early_stage.append(info)
    else:
        late_stage.append(info)

print(f"\nTop-20 winners blocked by daily_range (14d): {len(early_stage)+len(late_stage)}")
print(f"  early stage (NOT blocked yesterday): {len(early_stage)}  → would be recovered")
print(f"  late stage  (blocked yesterday too):  {len(late_stage)}  → still blocked")

print(f"\nEARLY (would be recovered):")
for x in early_stage:
    print(f"  {x['d']}  {x['sym']:<10} dr={x['today_max_dr']:.1f}%  prev_blocked={x['blocked_yesterday']}/{x['blocked_day2ago']}")
print(f"\nLATE (correctly stay blocked):")
for x in late_stage:
    print(f"  {x['d']}  {x['sym']:<10} dr={x['today_max_dr']:.1f}%  prev_blocked={x['blocked_yesterday']}/{x['blocked_day2ago']}")

# Estimate net effect
n_top20 = len(top20)
recovered = len(early_stage)
print(f"\nEstimate: +{recovered} winners recovered out of {n_top20} top-20 winner-days "
      f"= +{100*recovered/max(1,n_top20):.1f} п.п. coverage")

metric = {
    "metric": "1C_daily_range_stage",
    "n_top20": n_top20,
    "blocked_by_dr": len(early_stage)+len(late_stage),
    "early_stage_recoverable": len(early_stage),
    "late_stage_kept": len(late_stage),
    "coverage_uplift_pp": 100*recovered/max(1,n_top20),
}
print("\nMETRIC_JSON:" + json.dumps(metric))
