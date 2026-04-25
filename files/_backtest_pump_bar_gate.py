"""
Backtest proposed 'pump-bar noise' gate using critic_dataset.jsonl.

Hypothesis (from BNT example):
  Single-bar pump with long upper wick + weak ML -> garbage entry.

Critic dataset has:
  f.vol_x            - relative volume multiple
  f.upper_wick_pct   - upper wick as % of bar range (0..1)
  f.body_pct         - body as % of range
  labels.ret_5       - 5-bar forward return (the outcome)

Proposed gate variants:
  V1: vol_x >= 10 AND upper_wick_pct >= 0.40          (purely shape-based)
  V2: vol_x >= 10 AND upper_wick_pct >= 0.40 AND body_pct <= 0.50
  V3: vol_x >= 15 AND upper_wick_pct >= 0.35          (stricter vol, looser wick)
  V4: vol_x >= 8 AND upper_wick_pct >= 0.50           (looser vol, strict wick)

For each variant report:
  - KEPT:    trades NOT blocked → win%, avg ret_5
  - BLOCKED: trades blocked → count and avg ret_5 (to confirm they were indeed bad)

Goal: find variant where BLOCKED subset has clearly negative avg ret_5,
      and KEPT subset has avg ret_5 >= baseline ('take' decisions) avg.

Critical constraint: verify on 'take' decisions only (real bot entries),
  to not regress DOGS-like approved signals.
"""
from __future__ import annotations
import json, io, sys, math
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
FILES = Path(__file__).resolve().parent

rows = []
stats = {"total": 0, "take": 0, "has_feat": 0, "has_label": 0, "kept_for_backtest": 0}

print("Streaming critic_dataset.jsonl...")
with io.open(FILES / "critic_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        stats["total"] += 1
        try: e = json.loads(ln)
        except: continue
        dec = e.get("decision") or {}
        if dec.get("action") != "take": continue
        stats["take"] += 1
        feat = e.get("f") or {}
        if "vol_x" not in feat: continue
        stats["has_feat"] += 1
        labels = e.get("labels") or {}
        ret5 = labels.get("ret_5")
        if ret5 is None: continue
        stats["has_label"] += 1
        rows.append({
            "sym": e.get("sym"), "tf": e.get("tf"),
            "sig": e.get("signal_type"),
            "vol_x": float(feat.get("vol_x") or 0),
            "upper": float(feat.get("upper_wick_pct") or 0),
            "body":  float(feat.get("body_pct") or 0),
            "lower": float(feat.get("lower_wick_pct") or 0),
            "rsi": float(feat.get("rsi") or 0),
            "adx": float(feat.get("adx") or 0),
            "ret5": float(ret5),
            "is_bull": bool(e.get("is_bull_day")),
            "reason_code": dec.get("reason_code", ""),
        })
        stats["kept_for_backtest"] += 1

print(f"Stats: {stats}\n")
print(f"Backtesting on {len(rows)} 'take' decisions with outcome labels\n")

def sumr(lst, name):
    if not lst:
        print(f"  {name:55s} n=0"); return None
    n = len(lst); rets = [t["ret5"] for t in lst]
    wins = sum(1 for r in rets if r > 0); tot = sum(rets); avg = tot/n
    sd = (sum((r-avg)**2 for r in rets)/n)**0.5 if n>1 else 0
    sh = (avg/sd*math.sqrt(n)) if sd>0 else 0
    print(f"  {name:55s} n={n:>5d}  win={wins/n*100:>5.1f}%  avg_ret5={avg:+7.3f}%  sum={tot:+8.1f}  sharpe={sh:+.2f}")
    return {"n": n, "avg": avg, "win": wins/n*100, "sum": tot, "sharpe": sh}

print("=== BASELINE (all 'take' decisions) ===")
baseline = sumr(rows, "baseline")

print("\n=== by vol_x bucket ===")
def vb(v):
    if v<2: return "<2"
    if v<5: return "2-5"
    if v<10: return "5-10"
    if v<15: return "10-15"
    if v<20: return "15-20"
    return ">=20"
bybuck = defaultdict(list)
for t in rows: bybuck[vb(t["vol_x"])].append(t)
for k in ["<2","2-5","5-10","10-15","15-20",">=20"]:
    if k in bybuck: sumr(bybuck[k], f"vol_x {k}")

print("\n=== by upper_wick_pct bucket (vol_x >= 5 only) ===")
hi_vol = [t for t in rows if t["vol_x"] >= 5]
print(f"  (subset n={len(hi_vol)}, mean ret5={sum(t['ret5'] for t in hi_vol)/len(hi_vol):+.3f}%)")
def wb(w):
    if w<0.2: return "<0.20"
    if w<0.3: return "0.20-0.30"
    if w<0.4: return "0.30-0.40"
    if w<0.5: return "0.40-0.50"
    return ">=0.50"
byw = defaultdict(list)
for t in hi_vol: byw[wb(t["upper"])].append(t)
for k in ["<0.20","0.20-0.30","0.30-0.40","0.40-0.50",">=0.50"]:
    if k in byw: sumr(byw[k], f"upper_wick {k}  (vol_x>=5)")

print("\n=== GATE VARIANTS — block if condition TRUE ===")
variants = [
    ("V1 vx>=10 & wick>=0.40",        lambda t: t["vol_x"]>=10 and t["upper"]>=0.40),
    ("V2 vx>=10 & wick>=0.40 & body<=0.50", lambda t: t["vol_x"]>=10 and t["upper"]>=0.40 and t["body"]<=0.50),
    ("V3 vx>=15 & wick>=0.35",        lambda t: t["vol_x"]>=15 and t["upper"]>=0.35),
    ("V4 vx>=8 & wick>=0.50",         lambda t: t["vol_x"]>=8 and t["upper"]>=0.50),
    ("V5 vx>=12 & wick>=0.30",        lambda t: t["vol_x"]>=12 and t["upper"]>=0.30),
    ("V6 vx>=10 & wick>=0.50",        lambda t: t["vol_x"]>=10 and t["upper"]>=0.50),
    ("V7 vx>=20 & any",               lambda t: t["vol_x"]>=20),
]
for name, pred in variants:
    kept = [t for t in rows if not pred(t)]
    blocked = [t for t in rows if pred(t)]
    print(f"\n  --- {name} ---")
    k = sumr(kept, "KEPT")
    b = sumr(blocked, "BLOCKED")
    if k and b and baseline:
        d = k["avg"] - baseline["avg"]
        print(f"      delta KEPT vs baseline: {d:+.4f}%  |  blocked {b['n']}/{len(rows)} ({b['n']/len(rows)*100:.2f}%)")

print("\n=== GATE VARIANTS by tf ===")
for tf in ["15m", "1h"]:
    sub = [t for t in rows if t["tf"] == tf]
    if not sub: continue
    print(f"\n--- tf = {tf} (n={len(sub)}) ---")
    base = sumr(sub, f"{tf} baseline")
    for name, pred in variants:
        kept = [t for t in sub if not pred(t)]
        blocked = [t for t in sub if pred(t)]
        if not blocked: continue
        k = sumr(kept, f"KEPT  [{name}]")
        b = sumr(blocked, f"BLOCKED [{name}]")
        if k and b and base:
            d = k["avg"] - base["avg"]
            print(f"      delta={d:+.4f}%  block_rate={b['n']/len(sub)*100:.2f}%")

print("\n=== GATE by signal_type (V5 = vx>=12 & wick>=0.30) ===")
gate = lambda t: t["vol_x"]>=12 and t["upper"]>=0.30
by_sig = defaultdict(list)
for t in rows: by_sig[t["sig"]].append(t)
for sig, lst in sorted(by_sig.items(), key=lambda kv: -len(kv[1])):
    kept = [t for t in lst if not gate(t)]
    blocked = [t for t in lst if gate(t)]
    if len(lst) < 10: continue
    k = sumr(kept, f"KEPT   {sig}")
    b = sumr(blocked, f"BLOCKED {sig}")

print("\n=== 'DOGS-like' sanity check: impulse_speed with vol_x in [2,5] ===")
dogs_like = [t for t in rows if t["sig"]=="impulse_speed" and 2<=t["vol_x"]<5]
sumr(dogs_like, "impulse_speed vol_x in [2,5]")
print("\n=== 'BNT-like': impulse with vol_x>=15 ===")
bnt_like = [t for t in rows if t["sig"]=="impulse" and t["vol_x"]>=15]
sumr(bnt_like, "impulse vol_x>=15")
