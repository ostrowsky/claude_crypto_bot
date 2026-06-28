"""Sweep TREND_15M_QUALITY_RSI_MAX (max period). C98 was blocked at RSI 72.7 vs
cap 72.0 (non-bull day). Question: does raising the cap admit eventual winners /
watchlist top-20, or just overheated junk?

For trend/15m candidates blocked specifically by the RSI guard (reason contains
'RSI'), bucket by the candidate's RSI (f.rsi) and report forward ret_5, win%, and
watchlist top-20 share per band. Raising the cap to T admits the bands above the
old cap up to T — relax only if those bands are net-positive / carry top-20.

Read-only.  pyembed\python.exe files\_backtest_trend_rsi_sweep.py
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
WL = set(json.load(io.open(ROOT/"files"/"watchlist.json", encoding="utf-8")))


def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def _avg(xs): return sum(xs)/len(xs) if xs else 0.0


top = set()
for ln in io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8", errors="replace"):
    if '"label_top20"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    ts = e.get("ts")
    if not ts or e.get("label_top20") != 1 or e.get("symbol") not in WL: continue
    top.add((datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime("%Y-%m-%d"), e.get("symbol")))

BANDS = [(68, 70), (70, 72), (72, 74), (74, 76), (76, 100)]
band_data = {b: {"r5": [], "win": 0, "t20": set(), "taken": 0} for b in BANDS}
taken_daysym = set()
take_r5 = []

for ln in io.open(ROOT/"files"/"critic_dataset.jsonl", encoding="utf-8", errors="replace"):
    if '"action"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    dec = e.get("decision", {}) or {}; lab = e.get("labels", {}) or {}
    a = dec.get("action"); sym = e.get("sym")
    ts = e.get("ts_signal"); d = str(ts)[:10] if ts else None
    r5 = _f(lab.get("ret_5"))
    if a == "take":
        if r5 is not None: take_r5.append(r5)
        if d and sym: taken_daysym.add((d, sym))
        continue
    if dec.get("reason_code") != "trend_quality":
        continue
    if "RSI" not in str(dec.get("reason", "")):
        continue
    feat = e.get("f", {}) or {}
    rsi = _f(feat.get("rsi"))
    if rsi is None:
        continue
    for lo, hi in BANDS:
        if lo <= rsi < hi:
            bd = band_data[(lo, hi)]
            if r5 is not None:
                bd["r5"].append(r5); bd["win"] += 1 if r5 > 0 else 0
            if d and sym:
                if (d, sym) in top: bd["t20"].add((d, sym))
            break

tk = _avg(take_r5)
print("=" * 72)
print(f"TREND_15M RSI-guard sweep (max period)  take baseline avg_r5={tk:+.3f}% "
      f"n={len(take_r5)}")
print(f"current non-bull cap = 72.0  (bull cap = 76.0)")
print("=" * 72)
print(f"{'RSI band':<12}{'n':>6}{'avg_r5%':>9}{'win%':>6}{'top20':>7}"
      f"{'  top20 MISSED (never taken)':>0}")
for b in BANDS:
    bd = band_data[b]
    n = len(bd["r5"]); avg = _avg(bd["r5"]); win = bd["win"]/max(1, n)*100
    miss = len([k for k in bd["t20"] if k not in taken_daysym])
    print(f"{b[0]}-{b[1]:<8}{n:>6}{avg:>+9.3f}{win:>6.0f}{len(bd['t20']):>7}   "
          f"MISSED={miss}")
print("-" * 72)
print("Cap 72->74 admits the 72-74 band; 72->76 admits 72-76. Relax ONLY if the")
print("newly-admitted band(s) have avg_r5 >= take baseline AND carry top-20 (else")
print("we'd just add overheated losers). Negative band = cap is correct there.")
