"""Leading-signal test for EARLINESS: does a volume surge precede price moves
better than chance — early enough to alert before our price-confirmation gates?

If P(forward move | volume surge) >> base rate, a volume-surge "watch" tier
could fire earlier than the 1.5%/EMA confirmations. If ~base rate, volume does
not lead price here and earliness is at the frontier too (honest stop).

Method (history klines, watchlist, read-only): for each 15m bar compute
vol_x = volume / median(prior 16 bars). Forward move = max high over next M bars
/ close - 1. Report P(forward_move >= TH) by vol_x bucket vs base rate, and the
median LEAD (bars from the surge to first +TH touch). ASCII-only.
    pyembed\python.exe files\_backtest_volume_lead.py
"""
from __future__ import annotations
import csv, io, json, sys, statistics
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "history"
M = 16            # forward window bars (4h)
TH = 5.0          # "move" = +5% forward
BASE = 16         # bars for volume median


def klines(sym):
    p = HIST / f"{sym}_15m.csv"; b = []
    if p.exists():
        with io.open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try: b.append((float(r["high"]), float(r["low"]),
                               float(r["close"]), float(r["volume"])))
                except Exception: continue
    return b


WL = json.load(io.open(ROOT/"files"/"watchlist.json", encoding="utf-8"))

# bucket -> list of (followed_move:bool, lead_bars or None)
buckets = {"<2": [], "2-3": [], "3-5": [], ">5": []}
all_followed = []
for sym in WL:
    b = klines(sym)
    if len(b) < BASE + M + 5: continue
    vols = [x[3] for x in b]
    for i in range(BASE, len(b) - M):
        med = statistics.median(vols[i-BASE:i]) or 0.0
        if med <= 0: continue
        vx = vols[i] / med
        close_i = b[i][2]
        if close_i <= 0: continue
        # forward: first bar that touches +TH, and the max
        lead = None
        for j in range(1, M+1):
            if (b[i+j][0] / close_i - 1.0) * 100.0 >= TH:
                lead = j; break
        followed = lead is not None
        all_followed.append(followed)
        key = "<2" if vx < 2 else "2-3" if vx < 3 else "3-5" if vx < 5 else ">5"
        buckets[key].append((followed, lead))

base_rate = sum(all_followed)/len(all_followed)*100 if all_followed else 0
print("="*64)
print(f"Volume-lead test (watchlist, 15m, fwd {M}b/{M*15}min, move>=+{TH}%)")
print("="*64)
print(f"total bars: {len(all_followed)}   base rate P(move): {base_rate:.1f}%\n")
print(f"  {'vol_x bucket':<14}{'n':>8}{'P(move)%':>10}{'lift_vs_base':>13}{'median_lead_bars':>18}")
for key in ("<2", "2-3", "3-5", ">5"):
    rows = buckets[key]; n = len(rows)
    if not n: print(f"  {key:<14}{0:>8}"); continue
    p = sum(1 for f, _ in rows if f)/n*100
    leads = sorted(l for f, l in rows if f and l is not None)
    ml = leads[len(leads)//2] if leads else float("nan")
    print(f"  {key:<14}{n:>8}{p:>10.1f}{p-base_rate:>+13.1f}{ml:>18.0f}")
print("\nRead: if high vol_x buckets show P(move) clearly ABOVE base rate, volume")
print("leads price -> a surge 'watch' tier alerts earlier. median_lead_bars = how")
print("many 15m bars from the surge to the +%TH touch (the earliness headroom).")
print("If P(move) ~ base rate across buckets, volume does not lead here.")
