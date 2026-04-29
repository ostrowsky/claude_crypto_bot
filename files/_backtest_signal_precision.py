"""D1: signal precision = (entries on top-20 winners) / (total entries).
Deduplicated to one entry per (date, sym).
Also: D2 tg_message_rate (entries/day).
"""
from __future__ import annotations
import json, io, sys
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

# entries per (date, sym), per day
entries_uniq = set()
entries_per_day = defaultdict(int)
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
        entries_uniq.add((d, sym))
        entries_per_day[d] += 1

n_entries = len(entries_uniq)
n_top20_entries = sum(1 for k in entries_uniq if k in top20)
total_entries_raw = sum(entries_per_day.values())
n_days = len(entries_per_day)

print("=== D1: signal_precision_top20 + D2: tg_message_rate ===\n")
print(f"Unique (date,sym) entries: {n_entries}")
print(f"  on top-20 winners:       {n_top20_entries} ({100*n_top20_entries/max(1,n_entries):.1f}%)")
print(f"\nRaw entries (incl re-entries): {total_entries_raw}")
print(f"  Across {n_days} days  → avg {total_entries_raw/max(1,n_days):.1f} entries/day")
print(f"  Unique-symbol entries/day:    {n_entries/max(1,n_days):.1f}")

metric = {
    "metric": "D1_D2_precision_msgrate",
    "n_unique_entries": n_entries,
    "n_top20_entries": n_top20_entries,
    "precision_pct": 100*n_top20_entries/max(1,n_entries),
    "total_raw_entries": total_entries_raw,
    "n_days": n_days,
    "raw_entries_per_day": total_entries_raw/max(1,n_days),
    "unique_entries_per_day": n_entries/max(1,n_days),
}
print("\nMETRIC_JSON:" + json.dumps(metric))
