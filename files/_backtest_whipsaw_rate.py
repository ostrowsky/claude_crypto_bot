"""Q2: whipsaw rate.
Whipsaw = paired trade exiting via TRAIL stop in <=5 bars at PnL in [-1.5%, 0%].
Per mode, and overall.
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
NOW = datetime.now(timezone.utc); CUT = NOW - timedelta(days=30)

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
        if ev == "entry":
            entries[sym] = {
                "price": float(e.get("price") or e.get("entry_price") or 0),
                "mode": e.get("mode","?"), "tf": e.get("tf","?"),
            }
        else:
            ent = entries.pop(sym, None)
            if not ent: continue
            ex_p = float(e.get("exit_price") or e.get("price") or 0)
            if ent["price"] <= 0 or ex_p <= 0: continue
            pnl = (ex_p - ent["price"]) / ent["price"] * 100
            bars = int(e.get("bars_held") or 0)
            reason = (e.get("reason") or "")
            trail_hit = ("ATR-трейл" in reason) or ("trail" in reason.lower())
            pairs.append({"mode": ent["mode"], "tf": ent["tf"],
                          "pnl": pnl, "bars": bars, "trail_hit": trail_hit})

def is_whipsaw(t): return t["trail_hit"] and t["bars"] <= 5 and -1.5 <= t["pnl"] <= 0

groups = defaultdict(list)
for t in pairs: groups[f"{t['mode']}/{t['tf']}"].append(t)

print("=== Q2: whipsaw rate ===")
print(f"Total paired trades 30d: {len(pairs)}\n")
print(f"  {'mode/tf':<22} {'n':>4} {'whipsaws':>9} {'%':>6}")
total_w = 0
for k, rows in sorted(groups.items(), key=lambda x: -len(x[1])):
    n = len(rows); w = sum(1 for r in rows if is_whipsaw(r))
    total_w += w
    if n >= 5:
        print(f"  {k:<22} {n:>4d} {w:>9d} {100*w/n:>5.1f}%")

overall = 100*total_w/max(1, len(pairs))
print(f"\nOVERALL whipsaws: {total_w}/{len(pairs)} ({overall:.1f}%)")

metric = {
    "metric": "Q2_whipsaw_rate",
    "n_total": len(pairs),
    "n_whipsaw": total_w,
    "overall_pct": overall,
    "per_mode": {k: {"n": len(v), "whipsaw_pct": 100*sum(1 for r in v if is_whipsaw(r))/len(v)}
                 for k, v in groups.items() if len(v) >= 5},
}
print("\nMETRIC_JSON:" + json.dumps(metric))
