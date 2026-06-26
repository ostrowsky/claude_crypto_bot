"""For the watchlist top-20 that were DARK (never promoted to hot_coins) yet a
gradual-trend marker fired: did they EVER form a full bullish 1h EMA stack
(close > EMA20 > EMA50 > EMA200) that day BEFORE the high? If yes, the bot had a
valid 1h-trend structure it didn't act on (a real over-reject). If no (like POL:
price reclaimed EMA20 but EMA20<EMA50<EMA200), the no-signal was CORRECT.

Read-only, 60d 15m klines -> 1h.  pyembed\python.exe files\_backtest_dark_stack_check.py
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


def ema(a, n):
    k = 2/(n+1); out = np.empty_like(a); out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = a[i]*k + out[i-1]*(1-k)
    return out


def bars_1h(sym):
    p = HIST / f"{sym}_15m.csv"
    if not p.exists(): return None
    rows = []
    with io.open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try: rows.append((datetime.fromisoformat(r["ts"]), float(r["close"]), float(r["high"])))
            except Exception: continue
    rows = rows[::4]
    if len(rows) < 210: return None
    return rows


# top-20 + dark + trend-marker (reuse the discovery-gap logic, condensed)
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

scanned = set()
for ln in io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8", errors="replace"):
    if '"event"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    sym = e.get("sym") or e.get("symbol") or ""
    try: d = datetime.fromisoformat((e.get("ts","")).replace("Z","+00:00")).strftime("%Y-%m-%d")
    except: continue
    if d >= CUTd: scanned.add((d, sym))

dark = [k for k in top if k not in scanned]

had_stack = 0; no_stack = 0; examples_stack = []; examples_nostack = []
checked = 0
for d, sym in dark:
    b = bars_1h(sym)
    if b is None: continue
    cl = np.array([c for _, c, _ in b]); hi = np.array([h for _, _, h in b])
    ts = [t for t, _, _ in b]
    e20, e50, e200 = ema(cl, 20), ema(cl, 50), ema(cl, 200)
    # bars on day d
    day_idx = [i for i, t in enumerate(ts) if t.strftime("%Y-%m-%d") == d and i >= 200]
    if not day_idx: continue
    checked += 1
    peak_i = max(day_idx, key=lambda i: hi[i])
    # full bullish stack BEFORE the peak that day
    stack = any(cl[i] > e20[i] > e50[i] > e200[i] for i in day_idx if i <= peak_i)
    if stack:
        had_stack += 1
        if len(examples_stack) < 6: examples_stack.append((d, sym))
    else:
        no_stack += 1
        if len(examples_nostack) < 6: examples_nostack.append((d, sym))

print("=" * 68)
print("Dark top-20: full bullish 1h EMA stack before the high? (30d)")
print("=" * 68)
print(f"dark top-20 checked (had 1h klines): {checked}")
print(f"  HAD full bullish stack (potential over-reject): {had_stack}")
print(f"  NO stack (correct reject, early reversal like POL): {no_stack}")
print(f"\n  over-reject examples: {examples_stack}")
print(f"  correct-reject examples: {examples_nostack}")
print("\nIf HAD-stack is small, the dark misses are mostly correct early-reversal")
print("rejects (no full 1h trend) -> no loosening warranted. If large, the full-")
print("stack entry condition is too strict for valid early trends.")
