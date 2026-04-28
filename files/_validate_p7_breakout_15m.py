"""P7: should breakout/15m mode be disabled?
Check: does breakout/15m EVER enter a top-20 winner? If yes, killing it kills coverage.
Also: count its entries, exits with PnL, fast-reversal share.
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
NOW = datetime.now(timezone.utc); CUT = NOW - timedelta(days=30)

# Top-20 winners
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

# Breakout/15m entries
brk_entries = []
brk_exits_paired = []
entries_open = {}
with io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        try: e = json.loads(ln)
        except: continue
        ev = e.get("event","")
        if ev not in ("entry","exit"): continue
        ts = e.get("ts","")
        try: dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        except: continue
        if dt < CUT: continue
        sym = e.get("sym") or e.get("symbol") or ""
        mode = e.get("mode","?"); tf = e.get("tf","?")
        if ev == "entry":
            if mode == "breakout" and tf == "15m":
                d = dt.strftime("%Y-%m-%d")
                brk_entries.append({"d": d, "sym": sym, "dt": dt,
                                    "is_top20": (d, sym) in top20})
                entries_open[sym] = (dt, float(e.get("price") or e.get("entry_price") or 0), d)
            else:
                # forget any open breakout for sym (different mode entry)
                pass
        else:  # exit
            if sym in entries_open:
                edt, eprice, ed = entries_open.pop(sym)
                exit_price = float(e.get("exit_price") or e.get("price") or 0)
                if eprice > 0 and exit_price > 0:
                    pnl = (exit_price - eprice) / eprice * 100
                    bars = int(e.get("bars_held") or 0)
                    reason = (e.get("reason") or "")[:60]
                    brk_exits_paired.append({"d": ed, "sym": sym, "pnl": pnl,
                                             "bars": bars, "reason": reason,
                                             "is_top20": (ed, sym) in top20})

print("=== P7: breakout/15m utility ===\n")
n = len(brk_entries)
n_top20 = sum(1 for x in brk_entries if x["is_top20"])
print(f"breakout/15m entries (30d): {n}")
print(f"  on top-20 winners: {n_top20} ({100*n_top20/max(1,n):.1f}%)")
print(f"  on non-winners:    {n - n_top20}")

paired = brk_exits_paired
n_p = len(paired)
if n_p:
    win = sum(1 for r in paired if r["pnl"] > 0) / n_p * 100
    avg = sum(r["pnl"] for r in paired) / n_p
    fr1 = sum(1 for r in paired if r["bars"] <= 3 and r["pnl"] <= -0.3)
    print(f"\nPaired entry→exit: n={n_p}, avg_pnl={avg:+.2f}%, win={win:.0f}%, FR_v1={fr1} ({100*fr1/n_p:.1f}%)")
    # Top-20 vs non
    wins_t20 = [r for r in paired if r["is_top20"]]
    if wins_t20:
        avg_t = sum(r["pnl"] for r in wins_t20) / len(wins_t20)
        print(f"  on top-20: n={len(wins_t20)}, avg_pnl={avg_t:+.2f}%")
    nws = [r for r in paired if not r["is_top20"]]
    if nws:
        avg_n = sum(r["pnl"] for r in nws) / len(nws)
        print(f"  on non-winners: n={len(nws)}, avg_pnl={avg_n:+.2f}%")

# Sample
print("\nSample (last 8 paired):")
for r in paired[-8:]:
    tag = " (top-20)" if r["is_top20"] else ""
    print(f"  {r['d']} {r['sym']:<10} pnl={r['pnl']:+.2f}% bars={r['bars']:>2}  {r['reason']}{tag}")
