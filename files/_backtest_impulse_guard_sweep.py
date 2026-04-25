"""
Targeted Pareto sweep for impulse_guard parameters.

Two candidate changes:
  A) Lower ADX floor (IMPULSE_SPEED_15M_ADX_MIN: 20→14, IMPULSE_SPEED_1H_ADX_MIN: 18→14)
  B) Raise RSI_MAX on non-bull days (IMPULSE_SPEED_1H_RSI_MAX: 76→78 or 80)

Uses critic_dataset.jsonl: pull all impulse_guard blocked rows, separate
by sub-check, sweep thresholds, report delta vs baseline (take rows).
"""
from __future__ import annotations
import json, io, sys, math
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
FILES = Path(__file__).resolve().parent

# ── Load all rows with features + labels + block reason ───────────────
rows = []
with io.open(FILES / "critic_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        try: e = json.loads(ln)
        except: continue
        labels = e.get("labels") or {}
        ret5 = labels.get("ret_5")
        if ret5 is None: continue
        dec = e.get("decision") or {}
        action = dec.get("action", "")
        feat = e.get("f") or {}
        if not feat: continue
        adx = feat.get("adx")
        rsi = feat.get("rsi")
        vol_x = feat.get("vol_x")
        if adx is None or rsi is None: continue
        rows.append({
            "sym": e.get("sym"), "tf": e.get("tf"),
            "sig": e.get("signal_type"), "is_bull": bool(e.get("is_bull_day")),
            "action": action,
            "reason_code": dec.get("reason_code", ""),
            "reason_text": dec.get("reason", ""),
            "adx": float(adx), "rsi": float(rsi),
            "vol_x": float(vol_x or 0),
            "daily_range": float(feat.get("daily_range") or 0),
            "ret5": float(ret5),
        })

take_rows = [r for r in rows if r["action"] == "take"]
imp_blocked = [r for r in rows
               if "impulse_guard" in r["reason_code"]
               and r["sig"] == "impulse_speed"]

print(f"Total rows: {len(rows)}  take: {len(take_rows)}  impulse_guard blocked: {len(imp_blocked)}")

def agg(lst, name):
    if not lst:
        print(f"  {name:58s} n=0"); return None
    n = len(lst); rets = [r["ret5"] for r in lst]
    wins = sum(1 for v in rets if v>0); tot=sum(rets); avg=tot/n
    sd = (sum((v-avg)**2 for v in rets)/n)**0.5 if n>1 else 0
    sh = (avg/sd*math.sqrt(n)) if sd>0 else 0
    print(f"  {name:58s} n={n:>4d}  win={wins/n*100:>5.1f}%  avg={avg:+.3f}%  sum={tot:+7.1f}%  sh={sh:+.2f}")
    return {"n":n,"avg":avg,"win":wins/n*100,"sh":sh}

take_15m = [r for r in take_rows if r["tf"]=="15m" and r["sig"]=="impulse_speed"]
take_1h  = [r for r in take_rows if r["tf"]=="1h"  and r["sig"]=="impulse_speed"]
blk_15m  = [r for r in imp_blocked if r["tf"]=="15m"]
blk_1h   = [r for r in imp_blocked if r["tf"]=="1h"]

print("\n=== BASELINE (take decisions) ===")
agg(take_15m, "take 15m impulse_speed")
agg(take_1h,  "take 1h  impulse_speed")

print("\n=== IMPULSE_GUARD BLOCKED — by sub-check keyword ===")
for kw in ["ADX", "RSI", "range", "ext"]:
    b15 = [r for r in blk_15m if kw.lower() in r["reason_text"].lower()]
    b1h = [r for r in blk_1h  if kw.lower() in r["reason_text"].lower()]
    if b15: agg(b15, f"15m  blocked by {kw}")
    if b1h: agg(b1h, f"1h   blocked by {kw}")

# ── A) ADX floor sweep ─────────────────────────────────────────────────
# Rows blocked by ADX floor = reason text contains "ADX" and adx < current floor
adx_blk_15m = [r for r in blk_15m if "ADX" in r["reason_text"]]
adx_blk_1h  = [r for r in blk_1h  if "ADX" in r["reason_text"]]

print(f"\n=== A) ADX FLOOR SWEEP ===")
print(f"  ADX-blocked: 15m n={len(adx_blk_15m)}, 1h n={len(adx_blk_1h)}")
print(f"\n  15m: if we set floor=X, rows with ADX in [X, current_20) get unblocked:")
for floor in [8, 10, 12, 14, 16, 17, 18, 19]:
    unblocked = [r for r in adx_blk_15m if r["adx"] >= floor]
    kept_blocked = [r for r in adx_blk_15m if r["adx"] < floor]
    ub = agg(unblocked, f"    floor={floor:>2d}  unblocked (ADX>={floor})")
    if ub:
        # Combine: current take + newly unblocked -> new baseline
        new_n = len(take_15m) + ub["n"]
        new_avg = (sum(r["ret5"] for r in take_15m) + sum(r["ret5"] for r in unblocked)) / new_n
        print(f"          new combined avg_ret5={new_avg:+.4f}%  delta vs take={new_avg - (sum(r['ret5'] for r in take_15m)/len(take_15m) if take_15m else 0):+.4f}%")

print(f"\n  1h: if we set floor=X:")
for floor in [8, 10, 12, 14, 15, 16, 17]:
    unblocked = [r for r in adx_blk_1h if r["adx"] >= floor]
    ub = agg(unblocked, f"    floor={floor:>2d}  unblocked (ADX>={floor})")
    if ub and take_1h:
        new_avg = (sum(r["ret5"] for r in take_1h) + sum(r["ret5"] for r in unblocked)) / (len(take_1h)+ub["n"])
        base_avg = sum(r["ret5"] for r in take_1h)/len(take_1h)
        print(f"          new combined avg_ret5={new_avg:+.4f}%  delta={new_avg-base_avg:+.4f}%")

# ── B) RSI MAX sweep (non-bull day only) ────────────────────────────────
rsi_blk_15m = [r for r in blk_15m if "RSI" in r["reason_text"] and not r["is_bull"]]
rsi_blk_1h  = [r for r in blk_1h  if "RSI" in r["reason_text"] and not r["is_bull"]]

print(f"\n=== B) RSI_MAX SWEEP (non-bull day, 1h impulse_speed) ===")
print(f"  RSI-blocked non-bull: 15m n={len(rsi_blk_15m)}, 1h n={len(rsi_blk_1h)}")

take_1h_nb = [r for r in take_1h if not r["is_bull"]]
print(f"  take 1h non-bull baseline: n={len(take_1h_nb)}")
if take_1h_nb:
    base = agg(take_1h_nb, "take 1h non-bull baseline")

for rsi_max in [76, 77, 78, 79, 80, 82, 85]:
    unblocked = [r for r in rsi_blk_1h if r["rsi"] <= rsi_max]
    if not unblocked: continue
    ub = agg(unblocked, f"    rsi_max={rsi_max}  unblocked (RSI<={rsi_max})")
    if ub and take_1h_nb:
        all_rows = take_1h_nb + unblocked
        new_avg = sum(r["ret5"] for r in all_rows) / len(all_rows)
        base_avg = sum(r["ret5"] for r in take_1h_nb) / len(take_1h_nb)
        print(f"          combined avg={new_avg:+.4f}%  delta={new_avg-base_avg:+.4f}%")

# ── C) Combined best: ADX floor=14 + RSI_max=78 ─────────────────────
print(f"\n=== C) COMBINED BEST (1h): ADX floor=14 + RSI_max=78, non-bull ===")
unblocked_adx = [r for r in adx_blk_1h if r["adx"] >= 14]
unblocked_rsi = [r for r in rsi_blk_1h  if r["rsi"] <= 78]
all_unblocked = {id(r): r for r in unblocked_adx + unblocked_rsi}.values()
if take_1h_nb:
    combined = list(take_1h_nb) + list(all_unblocked)
    new_avg = sum(r["ret5"] for r in combined)/len(combined)
    base_avg = sum(r["ret5"] for r in take_1h_nb)/len(take_1h_nb)
    print(f"  total unblocked: {len(list(all_unblocked))}  combined n={len(combined)}")
    print(f"  base_avg={base_avg:+.4f}%  new_avg={new_avg:+.4f}%  delta={new_avg-base_avg:+.4f}%")

# ── D) 15m: sanity — what ADX distribution looks like in take rows ────
print(f"\n=== D) ADX distribution in current 15m impulse_speed TAKE rows ===")
for lo, hi in [(0,10),(10,15),(15,17),(17,20),(20,25),(25,30),(30,40),(40,100)]:
    sub = [r for r in take_15m if lo <= r["adx"] < hi]
    if sub: agg(sub, f"  ADX [{lo:>2d},{hi:>3d})  take 15m")

print(f"\n=== E) ADX distribution in current 1h impulse_speed TAKE rows ===")
for lo, hi in [(0,10),(10,15),(15,18),(18,20),(20,25),(25,30),(30,40),(40,100)]:
    sub = [r for r in take_1h if lo <= r["adx"] < hi]
    if sub: agg(sub, f"  ADX [{lo:>2d},{hi:>3d})  take 1h")
