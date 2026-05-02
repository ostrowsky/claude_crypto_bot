"""H: 53.7% alignment entries close within 3 bars at avg P&L -0.348% (per CLAUDE.md §4a).
Verify across all modes/tfs on last 30d paired trades.

Definition of fast-reversal: exit within 3 bars after entry AND pnl <= -0.3%.
Stricter alt: exit via trail-stop within 3 bars (whipsaw).
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent

NOW = datetime.now(timezone.utc)
CUT = NOW - timedelta(days=30)

entries = {}
trades = []

with io.open(ROOT / "files" / "bot_events.jsonl", encoding="utf-8") as f:
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
                "dt": dt,
                "price": float(e.get("price") or e.get("entry_price") or 0),
                "mode": e.get("mode","?"),
                "tf": e.get("tf","?"),
            }
        else:
            ent = entries.pop(sym, None)
            if not ent: continue
            ex_price = float(e.get("exit_price") or e.get("price") or 0)
            if ex_price <= 0 or ent["price"] <= 0: continue
            pnl = (ex_price - ent["price"]) / ent["price"] * 100
            bars = int(e.get("bars_held") or 0)
            reason = e.get("reason","") or ""
            trail_hit = ("ATR-трейл" in reason) or ("trail" in reason.lower())
            trades.append({
                "mode": ent["mode"], "tf": ent["tf"],
                "pnl": pnl, "bars": bars, "trail_hit": trail_hit,
            })

print(f"=== Fast-reversal analysis, last 30d ===")
print(f"Total paired trades: {len(trades)}\n")

# Per mode/tf: fast-reversal rate (3 def's)
groups = defaultdict(list)
for t in trades:
    groups[f"{t['mode']}/{t['tf']}"].append(t)

print(f"  {'mode/tf':<22} {'n':>4} {'FR_v1':>7} {'FR_v2':>7} {'FR_v3':>7} {'avg_pnl_FR':>10}")
print(f"  {'='*22} {'='*4} {'='*7} {'='*7} {'='*7} {'='*10}")
print(f"   v1 = bars<=3 & pnl<=-0.3%   v2 = v1 & trail_hit   v3 = bars<=3 (any pnl)\n")

modes = sorted(groups.keys(), key=lambda k: -len(groups[k]))
overall_fr1 = overall_fr1_pnl_sum = overall_fr1_n = 0
for k in modes:
    rows = groups[k]
    n = len(rows)
    if n < 5: continue
    fr1 = [r for r in rows if r["bars"] <= 3 and r["pnl"] <= -0.3]
    fr2 = [r for r in rows if r["bars"] <= 3 and r["pnl"] <= -0.3 and r["trail_hit"]]
    fr3 = [r for r in rows if r["bars"] <= 3]
    pct = lambda lst: 100*len(lst)/n
    fr1_pnl = (sum(r["pnl"] for r in fr1)/len(fr1)) if fr1 else 0.0
    print(f"  {k:<22} {n:>4d} {pct(fr1):>6.1f}% {pct(fr2):>6.1f}% {pct(fr3):>6.1f}% {fr1_pnl:>+9.2f}%")
    overall_fr1 += len(fr1); overall_fr1_pnl_sum += sum(r["pnl"] for r in fr1)
    overall_fr1_n += n

print()
total_fr1 = overall_fr1
print(f"  OVERALL:  total trades {overall_fr1_n}, FR_v1 = {total_fr1} ({100*total_fr1/max(1,overall_fr1_n):.1f}%), avg_pnl_FR = {overall_fr1_pnl_sum/max(1,total_fr1):+.2f}%")

# What % of all $ losses come from fast-reversals?
all_neg_pnl = sum(t["pnl"] for t in trades if t["pnl"] < 0)
fr_neg_pnl = sum(t["pnl"] for t in trades if t["bars"] <= 3 and t["pnl"] <= -0.3)
print(f"\n  Total negative pnl pool: {all_neg_pnl:+.2f}% (sum of losers)")
print(f"  Fast-reversal pool:      {fr_neg_pnl:+.2f}% ({100*fr_neg_pnl/min(-0.0001,all_neg_pnl):.0f}% of total drag)")

# METRIC_JSON for daily aggregator
metric = {
    "metric": "Q1_Q3_fast_reversal",
    "n_total_pairs": overall_fr1_n,
    "n_fr_v1": overall_fr1,
    "fr_v1_overall_pct": 100*overall_fr1/max(1, overall_fr1_n),
    "fr_drag_pct_of_total_neg": 100*fr_neg_pnl/min(-0.0001, all_neg_pnl),
}
print("\nMETRIC_JSON:" + json.dumps(metric))
