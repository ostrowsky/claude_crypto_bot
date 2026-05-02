"""E2: realized capture ratio.
For paired (entry, exit) on top-20 winners:
  capture_ratio_realized = pnl_pct / eod_return_pct (clamped to [0, 1.5])

Higher = bot caught more of the daily move.
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
NOW = datetime.now(timezone.utc); CUT = NOW - timedelta(days=30)

# top-20 winners + eod_return
eod_ret = {}; top20 = set()
with io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        try: e = json.loads(ln)
        except: continue
        ts_ms = e.get("ts");
        if not ts_ms: continue
        dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
        if dt < CUT: continue
        sym = e.get("symbol"); d = dt.strftime("%Y-%m-%d")
        # eod_return_pct: take latest snapshot's value
        eod_ret[(d, sym)] = e.get("eod_return_pct")
        if e.get("label_top20") == 1: top20.add((d, sym))

# Pair entries with exits
entries = {}; pairs = []
with io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if '"event"' not in ln: continue
        try: e = json.loads(ln)
        except: continue
        ev = e.get("event","")
        if ev not in ("entry","exit"): continue
        ts = e.get("ts","")
        try: dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        except: continue
        if dt < CUT: continue
        sym = e.get("sym") or e.get("symbol") or ""
        if not sym: continue
        d = dt.strftime("%Y-%m-%d")
        if ev == "entry":
            entries[sym] = (dt, d, float(e.get("price") or e.get("entry_price") or 0))
        else:
            ent = entries.pop(sym, None)
            if not ent: continue
            ex_p = float(e.get("exit_price") or e.get("price") or 0)
            if ent[2] <= 0 or ex_p <= 0: continue
            pnl = (ex_p - ent[2]) / ent[2] * 100
            pairs.append({"d": ent[1], "sym": sym, "pnl": pnl,
                          "is_top20": (ent[1], sym) in top20,
                          "eod": eod_ret.get((ent[1], sym))})

# Capture ratio for top-20 winners with valid eod
caps = []
caps_all = []
for r in pairs:
    if r["eod"] is None: continue
    eod = float(r["eod"])
    # eod_return_pct in dataset can be in % or decimal — observed values up to +475 suggesting % units.
    # Heuristic: if abs(eod) > 5 -> already in %; else decimal.
    if abs(eod) <= 5: eod = eod * 100
    if abs(eod) < 1.0: continue  # tiny denominator
    cr = r["pnl"] / eod  # both in %; cr is unitless ratio
    cr = max(-0.5, min(1.5, cr))  # clamp
    caps_all.append(cr)
    if r["is_top20"]:
        caps.append(cr)

print("=== E2: realized capture ratio ===")
def stats(lst, label):
    if not lst: print(f"{label}: empty"); return None
    lst = sorted(lst); n = len(lst)
    mean = sum(lst)/n; median = lst[n//2]
    pos = sum(1 for x in lst if x > 0)
    print(f"{label}: n={n}, mean={mean:+.3f}, median={median:+.3f}, positive={100*pos/n:.0f}%")
    return {"n": n, "mean": mean, "median": median, "positive_pct": 100*pos/n}

s_t = stats(caps,    "TOP-20 only           ")
s_a = stats(caps_all,"all paired (top+non)  ")

metric = {
    "metric": "E2_capture_ratio",
    "top20": s_t,
    "all": s_a,
}
print("\nMETRIC_JSON:" + json.dumps(metric))
