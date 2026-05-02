"""E1: time-to-signal (TTS).
For each entered top-20 winner: hours between first 'move start' marker
(first intraday snapshot with tg_return_since_open >= 0.02) and entry_time.

Positive lead = bot entered BEFORE 2% move detected.
Negative lead = bot entered AFTER 2% threshold (chasing).

Outputs JSON line for daily aggregator at the end.
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
NOW = datetime.now(timezone.utc); CUT = NOW - timedelta(days=14)

# Snapshots: earliest with tg_return_since_open >= 0.02 per (date, sym)
move_start = {}  # (d, sym) -> earliest_dt
top20 = set()
with io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        try: e = json.loads(ln)
        except: continue
        ts_ms = e.get("ts");
        if not ts_ms: continue
        dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
        if dt < CUT: continue
        sym = e.get("symbol"); d = dt.strftime("%Y-%m-%d")
        feat = e.get("features") or {}
        tgret = feat.get("tg_return_since_open")
        if tgret is None: continue
        if float(tgret) >= 0.02:
            prev = move_start.get((d, sym))
            if prev is None or dt < prev:
                move_start[(d, sym)] = dt
        if e.get("label_top20") == 1:
            top20.add((d, sym))

# First entry per (date, sym)
first_entry = {}
with io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if '"event":"entry"' not in ln and '"event": "entry"' not in ln: continue
        try: e = json.loads(ln)
        except: continue
        if e.get("event") != "entry": continue
        ts = e.get("ts","")
        try: dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        except: continue
        if dt < CUT: continue
        sym = e.get("sym") or e.get("symbol") or ""
        if not sym: continue
        d = dt.strftime("%Y-%m-%d")
        prev = first_entry.get((d, sym))
        if prev is None or dt < prev:
            first_entry[(d, sym)] = dt

# Compute TTS for top-20 winners that bot entered
leads_h = []
for key in top20:
    if key not in first_entry: continue
    if key not in move_start: continue
    delta_h = (first_entry[key] - move_start[key]).total_seconds() / 3600
    leads_h.append(delta_h)

print("=== E1: Time-to-signal ===")
print(f"Entered top-20 winners with move_start signal: n={len(leads_h)}\n")
if leads_h:
    leads_h.sort()
    n = len(leads_h)
    median = leads_h[n//2]
    mean = sum(leads_h)/n
    p25 = leads_h[n//4]
    p75 = leads_h[3*n//4]
    early_count = sum(1 for x in leads_h if x <= 0)
    late_30m = sum(1 for x in leads_h if x > 0.5)
    print(f"  median lead = {median:+.2f}h  (negative = entered BEFORE 2% move)")
    print(f"  mean   lead = {mean:+.2f}h")
    print(f"  p25         = {p25:+.2f}h")
    print(f"  p75         = {p75:+.2f}h")
    print(f"  entered BEFORE 2% move (lead<=0): {early_count}/{n} ({100*early_count/n:.0f}%)")
    print(f"  late >30 min after 2% move:        {late_30m}/{n} ({100*late_30m/n:.0f}%)")
    # JSON line for aggregator
    metric = {
        "metric": "E1_time_to_signal",
        "n": n,
        "median_h": median,
        "mean_h": mean,
        "early_pct": 100*early_count/n,
        "late_30m_pct": 100*late_30m/n,
    }
    print("\nMETRIC_JSON:" + json.dumps(metric))
