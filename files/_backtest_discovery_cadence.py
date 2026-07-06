"""ATOM-type miss: is the fix faster discovery CADENCE, or is 20-min already
enough (so the miss is elsewhere — criteria/gates)? For each DARK top-20 (never
scanned) measure the headroom between the FIRST impulse-signal bar and the day's
peak. Large headroom => a 20-min scan has ample time; the miss is the discovery
criterion/gates, not cadence. Small headroom => the move is too fast for a 20-min
cadence and a 5m first-alert tier (H5) would actually help.

impulse trigger (15m): 3-bar cumulative return >= 1.5% (mirrors impulse_speed).
Read-only, klines.  pyembed\python.exe files\_backtest_discovery_cadence.py
"""
from __future__ import annotations
import csv, io, json, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "history"
WL = set(json.load(io.open(ROOT/"files"/"watchlist.json", encoding="utf-8")))
IMP_TH = 1.5          # 3-bar % for impulse trigger
BAR_MIN = 15

# klines per sym: ts(ms), high, close
K = {}
for p in HIST.glob("*_15m.csv"):
    sym = p.name[:-8]
    if sym not in WL:
        continue
    ts = []; hi = []; cl = []
    with io.open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                ts.append(int(datetime.fromisoformat(r["ts"]).timestamp()*1000))
                hi.append(float(r["high"])); cl.append(float(r["close"]))
            except Exception:
                continue
    if len(cl) > 100:
        K[sym] = (np.array(ts), np.array(hi), np.array(cl))

# top-20 + dark (never scanned)
top = set()
for ln in io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8", errors="replace"):
    if '"label_top20"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    ts = e.get("ts")
    if not ts or e.get("label_top20") != 1 or e.get("symbol") not in WL: continue
    d = datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime("%Y-%m-%d")
    top.add((d, e.get("symbol")))
scanned = set()
for ln in io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8", errors="replace"):
    if '"event"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    sym = e.get("sym") or e.get("symbol")
    try: d = datetime.fromisoformat(e.get("ts", "").replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except: continue
    if sym in WL: scanned.add((d, sym))
dark = sorted(k for k in top if k not in scanned)

headrooms = []; had_trigger = 0; checked = 0; no_trigger = 0
for d, sym in dark:
    dd = K.get(sym)
    if dd is None: continue
    ts, hi, cl = dd
    day0 = int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()*1000)
    day1 = day0 + 86400_000
    idx = np.where((ts >= day0) & (ts < day1))[0]
    if len(idx) < 10: continue
    checked += 1
    peak_i = idx[np.argmax(hi[idx])]
    trig = None
    for j in idx:
        if j < 3: continue
        r3 = (cl[j] - cl[j-3]) / cl[j-3] * 100.0
        if r3 >= IMP_TH:
            trig = j; break
    if trig is None:
        no_trigger += 1; continue
    if trig <= peak_i:
        had_trigger += 1
        headrooms.append((peak_i - trig) * BAR_MIN)   # minutes from first signal to peak

hr = np.array(headrooms) if headrooms else np.array([0])
print("=" * 64)
print(f"DARK top-20 checked (had klines): {checked}")
print(f"  had impulse trigger BEFORE peak (catchable):   {had_trigger} "
      f"({100*had_trigger/max(1,checked):.0f}%)")
print(f"  NO impulse trigger all day (not momentum):     {no_trigger} "
      f"({100*no_trigger/max(1,checked):.0f}%)")
print(f"  headroom first-signal -> peak (minutes): median={np.median(hr):.0f} "
      f"p25={np.percentile(hr,25):.0f} p75={np.percentile(hr,75):.0f}")
fast = int(np.sum(hr <= 20))
print(f"  moves with headroom <= 20min (cadence would MISS): {fast}/{len(headrooms)} "
      f"({100*fast/max(1,len(headrooms)):.0f}%)")
print("=" * 64)
print("Large median headroom (>>20min) => 20-min discovery cadence is ENOUGH;")
print("the dark miss is the discovery CRITERION/gates, not speed. Only the")
print("<=20min slice would benefit from a 5m first-alert tier (H5).")
