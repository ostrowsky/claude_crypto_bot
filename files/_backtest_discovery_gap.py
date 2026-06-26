"""Validate the discovery gap: the live scan only sees coins the PUMP detector
injects (>=2%/15min). Gradual uptrends (e.g. POL) are never scanned -> never
signalled. Quantify: of watchlist top-20 (day,sym), how many were DARK (no bot
event = never scanned), and how many a gradual TREND-discovery rule would have
surfaced (recall) vs how many coins/day it would add to the scan (flood/noise).

trend-discovery proxy (1h bars from 15m klines): a bar qualifies if
  close > MA20 AND MA20 rising (MA20[t] > MA20[t-3]) AND close 3h-slope > +1%.
A (day,sym) is "discoverable" if any bar that day qualifies.

Read-only, 60d klines.  pyembed\python.exe files\_backtest_discovery_gap.py
"""
from __future__ import annotations
import csv, io, json, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "history"
CUTd = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
WL = set(json.load(io.open(ROOT/"files"/"watchlist.json", encoding="utf-8")))


def load_1h(sym):
    """15m closes -> 1h closes (every 4th bar) with their day."""
    p = HIST / f"{sym}_15m.csv"
    if not p.exists(): return []
    rows = []
    with io.open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try: rows.append((datetime.fromisoformat(r["ts"]), float(r["close"])))
            except Exception: continue
    return rows[::4]   # ~1h


def trend_days(sym):
    """Set of YYYY-MM-DD where a gradual-uptrend bar fired."""
    bars = load_1h(sym)
    if len(bars) < 25: return set()
    cl = np.array([c for _, c in bars])
    ma20 = np.convolve(cl, np.ones(20)/20, mode="valid")  # aligned to idx>=19
    out = set()
    for i in range(20, len(cl)):
        m = ma20[i-19]; mprev = ma20[i-19-3] if i-19-3 >= 0 else None
        if mprev is None: continue
        slope3 = cl[i]/cl[i-3]-1.0
        if cl[i] > m and m > mprev and slope3 > 0.01:
            out.add(bars[i][0].strftime("%Y-%m-%d"))
    return out


# top-20 watchlist (day,sym)
top = set()
for ln in io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8", errors="replace"):
    if '"label_top20"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    ts = e.get("ts")
    if not ts: continue
    d = datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime("%Y-%m-%d")
    if d >= CUTd and e.get("symbol") in WL and e.get("label_top20") == 1:
        top.add((d, e.get("symbol")))

# scanned (day,sym) = had any bot event
scanned = set()
for ln in io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8", errors="replace"):
    if '"event"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    sym = e.get("sym") or e.get("symbol") or ""
    try: d = datetime.fromisoformat((e.get("ts","")).replace("Z","+00:00")).strftime("%Y-%m-%d")
    except: continue
    if d >= CUTd: scanned.add((d, sym))

# precompute trend-days per watchlist sym
td = {s: trend_days(s) for s in WL}

dark = [k for k in top if k not in scanned]
dark_discoverable = [k for k in dark if k[1] in td and k[0] in td[k[1]]]

# flood: distinct (day,sym) where trend fired, and top-20 share
fire = set()
for s in WL:
    for d in td.get(s, ()):
        if d >= CUTd: fire.add((d, s))
fire_top = [k for k in fire if k in top]
days = len({d for d, _ in fire}) or 1

print("=" * 70)
print("Discovery-gap validation (watchlist top-20, 30d)")
print("=" * 70)
print(f"watchlist top-20 (day,sym): {len(top)}")
print(f"  DARK (never scanned, no event): {len(dark)} ({100*len(dark)/max(1,len(top)):.0f}%)")
print(f"  of dark, trend-discovery WOULD surface: {len(dark_discoverable)} "
      f"({100*len(dark_discoverable)/max(1,len(dark)):.0f}% of dark)")
print()
print(f"trend-discovery firing across watchlist: {len(fire)} (day,sym) over ~{days} days")
print(f"  = ~{len(fire)/days:.0f} coins/day added to scan")
print(f"  of those, top-20: {len(fire_top)}  precision={100*len(fire_top)/max(1,len(fire)):.1f}%")
print()
print("Read: high 'dark' + high 'would surface' => trend-discovery recovers")
print("real missed movers. Watch coins/day (scan-load) and precision (noise).")
