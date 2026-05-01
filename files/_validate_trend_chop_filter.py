"""Hypothesis: stricter ADX + slope thresholds on trend mode would
filter out chop-range entries like STRKUSDT 2026-05-01 (ADX 20.2,
slope +0.70%) without losing real trends.

Filters tested on trend/15m and trend/1h entries (30d):
  - baseline (current production thresholds, no extra filter)
  - ADX >= 22, slope >= 1.0
  - ADX >= 25, slope >= 1.0
  - ADX >= 25, slope >= 1.2
  - ADX >= 25, slope >= 1.5
  - ADX >= 28, slope >= 1.5
  - vol_x >= 1.5 (alone)
  - combined ADX>=25, slope>=1.2, vol_x>=1.3

Metrics per filter:
  - n_kept (entries that pass)
  - n_top20 / precision_pct
  - paired avg_pnl
  - FR_v1 rate (≤3 bars, pnl ≤ -0.3%)
  - recall vs trend-mode baseline
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
NOW = datetime.now(timezone.utc); CUT = NOW - timedelta(days=30)

# top-20 winners
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

# Entries + paired exits (for trend mode only)
open_trades = {}  # sym -> entry record
trend_entries = []  # all entries on trend mode

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
        mode = e.get("mode","?"); tf = e.get("tf","?")
        if ev == "entry":
            if mode == "trend":
                rec = {
                    "d": d, "sym": sym, "tf": tf, "mode": mode,
                    "adx": float(e.get("adx") or 0),
                    "slope_pct": float(e.get("slope_pct") or 0),
                    "rsi": float(e.get("rsi") or 0),
                    "vol_x": float(e.get("vol_x") or 0),
                    "daily_range": float(e.get("daily_range") or 0),
                    "ml_proba": float(e.get("ml_proba") or 0),
                    "price": float(e.get("price") or 0),
                    "is_top20": (d, sym) in top20,
                    "entry_dt": dt,
                    "pnl": None, "bars": None, "fr_v1": False, "trail_hit": False,
                }
                open_trades[sym] = rec
                trend_entries.append(rec)
        else:  # exit
            ent = open_trades.pop(sym, None)
            if ent is None: continue
            ex_p = float(e.get("exit_price") or e.get("price") or 0)
            if ent["price"] <= 0 or ex_p <= 0: continue
            pnl = (ex_p - ent["price"]) / ent["price"] * 100
            bars = int(e.get("bars_held") or 0)
            reason = (e.get("reason") or "")
            ent["pnl"] = pnl
            ent["bars"] = bars
            ent["fr_v1"] = bars <= 3 and pnl <= -0.3
            ent["trail_hit"] = ("ATR-трейл" in reason) or ("trail" in reason.lower())

# Print baseline
def stats(rows, label, baseline_top20=None):
    n = len(rows)
    if n == 0:
        print(f"  {label:<32} n=0"); return None
    paired = [r for r in rows if r["pnl"] is not None]
    n_t = sum(1 for r in rows if r["is_top20"])
    n_t_paired = sum(1 for r in paired if r["is_top20"])
    avg_pnl = sum(r["pnl"] for r in paired) / max(1, len(paired))
    win = sum(1 for r in paired if r["pnl"] > 0) / max(1, len(paired)) * 100
    fr = sum(1 for r in paired if r["fr_v1"]) / max(1, len(paired)) * 100
    prec = 100*n_t/max(1, n)
    recall = (100*n_t/baseline_top20) if baseline_top20 else None
    rec_str = f"{recall:>5.0f}%" if recall is not None else "  —  "
    print(f"  {label:<32} n={n:>3d}  paired={len(paired):>3d}  "
          f"prec={prec:>4.1f}%  rec={rec_str}  "
          f"avg_pnl={avg_pnl:+5.2f}%  win={win:>4.0f}%  FR={fr:>4.1f}%")
    return {"n": n, "precision_pct": prec, "recall_pct": recall,
            "avg_pnl": avg_pnl, "win_pct": win, "fr_v1_pct": fr,
            "paired_n": len(paired)}

print(f"=== Trend-mode chop-filter validation (30 d) ===\n")
print(f"Total trend entries: {len(trend_entries)}")
print(f"  trend/15m: {sum(1 for r in trend_entries if r['tf']=='15m')}")
print(f"  trend/1h:  {sum(1 for r in trend_entries if r['tf']=='1h')}")

baseline_n_top20 = sum(1 for r in trend_entries if r["is_top20"])
print(f"  on top-20 winners: {baseline_n_top20}\n")

filters = [
    ("baseline (no filter)",        lambda r: True),
    ("ADX>=22 & slope>=1.0",        lambda r: r["adx"] >= 22 and r["slope_pct"] >= 1.0),
    ("ADX>=25 & slope>=1.0",        lambda r: r["adx"] >= 25 and r["slope_pct"] >= 1.0),
    ("ADX>=25 & slope>=1.2",        lambda r: r["adx"] >= 25 and r["slope_pct"] >= 1.2),
    ("ADX>=25 & slope>=1.5",        lambda r: r["adx"] >= 25 and r["slope_pct"] >= 1.5),
    ("ADX>=28 & slope>=1.5",        lambda r: r["adx"] >= 28 and r["slope_pct"] >= 1.5),
    ("vol_x>=1.5 only",             lambda r: r["vol_x"] >= 1.5),
    ("ADX>=25 & slope>=1.2 & vol>=1.3", lambda r: r["adx"] >= 25 and r["slope_pct"] >= 1.2 and r["vol_x"] >= 1.3),
    ("ADX>=22 & slope>=1.0 & vol>=1.3", lambda r: r["adx"] >= 22 and r["slope_pct"] >= 1.0 and r["vol_x"] >= 1.3),
]

# Per TF
for tf in ["15m", "1h"]:
    rows_tf = [r for r in trend_entries if r["tf"] == tf]
    if not rows_tf: continue
    base_top20_tf = sum(1 for r in rows_tf if r["is_top20"])
    print(f"\n── trend/{tf}  (total n={len(rows_tf)}, top20={base_top20_tf}) ──")
    print(f"  {'filter':<32} {'stats'}")
    for label, fn in filters:
        kept = [r for r in rows_tf if fn(r)]
        stats(kept, label, baseline_top20=base_top20_tf)

# Pareto-optimal: precision >= +5pp AND recall >= 70% AND avg_pnl >= 0
print(f"\n=== Pareto-viable filters (per TF) ===")
print(f"Criteria: precision_uplift >= +5pp, recall >= 70%, avg_pnl >= 0")
for tf in ["15m", "1h"]:
    rows_tf = [r for r in trend_entries if r["tf"] == tf]
    if not rows_tf: continue
    base = stats(rows_tf, "baseline_internal", baseline_top20=sum(1 for r in rows_tf if r["is_top20"]))
    print(f"\n  trend/{tf}:")
    for label, fn in filters[1:]:  # skip baseline
        kept = [r for r in rows_tf if fn(r)]
        if not kept: continue
        n = len(kept); paired = [r for r in kept if r["pnl"] is not None]
        n_t = sum(1 for r in kept if r["is_top20"])
        prec = 100*n_t/n
        recall = 100*n_t/max(1, base["n"]*base["precision_pct"]/100) if base else 0
        avg_pnl = sum(r["pnl"] for r in paired) / max(1, len(paired))
        if prec >= base["precision_pct"] + 5 and recall >= 70 and avg_pnl >= 0:
            print(f"    {label:<32} prec={prec:.1f}% (+{prec-base['precision_pct']:.1f}pp), "
                  f"recall={recall:.0f}%, avg_pnl={avg_pnl:+.2f}%")

# Sample STRK-like cases that current bot let through but proposed filter blocks
print(f"\n=== STRK-like rejected entries (ADX 20-22, slope 0.5-0.9) ===")
strk_like = [r for r in trend_entries
             if 19 <= r["adx"] <= 23 and 0.4 <= r["slope_pct"] <= 1.1]
print(f"  total: {len(strk_like)} (current bot took all these)")
top20_in_strk = sum(1 for r in strk_like if r["is_top20"])
paired_strk = [r for r in strk_like if r["pnl"] is not None]
if paired_strk:
    avg = sum(r["pnl"] for r in paired_strk)/len(paired_strk)
    fr = sum(1 for r in paired_strk if r["fr_v1"])
    print(f"  on top-20: {top20_in_strk}/{len(strk_like)} = {100*top20_in_strk/max(1,len(strk_like)):.1f}%")
    print(f"  paired: n={len(paired_strk)}  avg_pnl={avg:+.2f}%  FR={fr} ({100*fr/len(paired_strk):.0f}%)")
print(f"  → these would be SKIPPED by ADX>=25 & slope>=1.2 filter")

metric = {"metric": "trend_chop_filter_sweep",
          "total_trend_entries": len(trend_entries),
          "baseline_top20": baseline_n_top20,
          "strk_like_count": len(strk_like) if strk_like else 0,
          "strk_like_top20_share": 100*top20_in_strk/max(1, len(strk_like)) if strk_like else 0}
print("\nMETRIC_JSON:" + json.dumps(metric))
