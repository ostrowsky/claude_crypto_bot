"""H: bot enters top-20 winners LATE — after most of the move already happened.
For each (date, sym) where bot entered AND label_top20=1, compare:
  - entry hour-of-day (UTC)
  - eod_return_pct (full-day % move)
  - entry vs day_open price ratio (where in the candle did we enter)
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent

NOW = datetime.now(timezone.utc)
CUT = NOW - timedelta(days=14)

# 1) For each (date, sym) collect eod_return + day-open features from top_gainer_dataset
# Take EARLIEST snapshot of the day (gives intraday feature near 00:00 UTC)
day_open_data = {}  # (d, sym) -> earliest snapshot dict
day_eod_ret  = {}   # (d, sym) -> eod_return_pct
top20_set    = set()  # (d, sym)
with io.open(ROOT / "files" / "top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        try: e = json.loads(ln)
        except: continue
        ts_ms = e.get("ts")
        if not ts_ms: continue
        dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
        if dt < CUT: continue
        d = dt.strftime("%Y-%m-%d"); sym = e.get("symbol")
        if not sym: continue
        day_eod_ret[(d, sym)] = e.get("eod_return_pct")
        prev = day_open_data.get((d, sym))
        if prev is None or ts_ms < prev[0]:
            day_open_data[(d, sym)] = (ts_ms, e.get("features") or {})
        if e.get("label_top20") == 1:
            top20_set.add((d, sym))

# 2) First entry per (date, sym) from bot_events
first_entry = {}  # (d, sym) -> (dt, entry_price)
with io.open(ROOT / "files" / "bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if '"event":"entry"' not in ln and '"event": "entry"' not in ln: continue
        try: e = json.loads(ln)
        except: continue
        ev = e.get("event","")
        if ev != "entry": continue
        ts = e.get("ts","")
        try: dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        except: continue
        if dt < CUT: continue
        sym = e.get("sym") or e.get("symbol") or ""
        d = dt.strftime("%Y-%m-%d")
        if not sym: continue
        ep = float(e.get("price") or e.get("entry_price") or 0)
        if ep <= 0: continue
        prev = first_entry.get((d, sym))
        if prev is None or dt < prev[0]:
            first_entry[(d, sym)] = (dt, ep)

# 3) Cross
print(f"=== Entry lateness on EOD top-20 winners, last 14d ===\n")
rows = []
for key in sorted(first_entry.keys()):
    if key not in top20_set: continue
    d, sym = key
    dt, ep = first_entry[key]
    eod = day_eod_ret.get(key)
    feat = day_open_data.get(key, (None, {}))[1]
    tg_open = feat.get("tg_return_since_open")  # decimal
    rows.append({"d": d, "sym": sym, "hour": dt.hour, "eod": eod,
                 "tg_open": tg_open})

print(f"Entered top-20 winners: n={len(rows)}\n")
print(f"  {'date':<11} {'sym':<10} {'hr_UTC':>6} {'eod_ret%':>9} {'pre_entry_move%*':>17}")
print(f"  {'-'*11} {'-'*10} {'-'*6} {'-'*9} {'-'*17}")
for r in rows:
    eod = (r["eod"]*100) if (r["eod"] is not None and abs(r["eod"]) < 5) else (r["eod"] or 0)
    # tg_open is recorded at *snapshot time*, near day-open; weak proxy for "move at entry time"
    pre = (r["tg_open"]*100) if (r["tg_open"] is not None and abs(r["tg_open"]) < 5) else (r["tg_open"] or 0)
    print(f"  {r['d']:<11} {r['sym']:<10} {r['hour']:>6d} {eod:>+8.2f}% {pre:>+15.2f}%")

# Aggregate
hour_buckets = defaultdict(int)
for r in rows: hour_buckets[r["hour"]//4*4] += 1
print(f"\nHour-of-day distribution (UTC, 4h buckets):")
for h in sorted(hour_buckets):
    print(f"  {h:02d}-{h+3:02d}h: {'#'*hour_buckets[h]} ({hour_buckets[h]})")

late_count = sum(1 for r in rows if r["hour"] >= 12)
print(f"\nEntries at >=12 UTC (likely after Asia+EU saw the move): {late_count}/{len(rows)} ({100*late_count/max(1,len(rows)):.0f}%)")

print("""
*pre_entry_move% = tg_return_since_open at the EARLIEST daily snapshot.
  Note: this is logged at snapshot time (08:30/14:30/20:30 LOCAL), not at entry time.
  Treat as weak proxy — if it's already +5% by first snapshot and we only enter at 20:00 UTC,
  we captured the tail, not the head.
""")
