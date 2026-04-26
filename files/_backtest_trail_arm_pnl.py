"""How does bandit's trail-arm choice affect P&L per mode?
H: tight/very_tight arms cause premature whipsaw exits on impulse_speed.
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

# Pair entries -> exits, attach trail_k that was used
entries = {}  # sym -> open entry dict
trades = []   # list of {sym, mode, tf, trail_k_used, base_trail_k_inferred, pnl, bars, dr}

def base_trail_k(mode: str, tf: str) -> float:
    if mode in ("strong_trend", "impulse_speed"):
        return 2.5
    if mode == "impulse":
        return 2.0
    if mode == "alignment":
        return 2.0
    if mode == "retest":
        return 1.5
    if mode == "breakout":
        return 1.5
    if mode == "trend":
        return 2.0
    return 2.0

def arm_name(trail_k_used: float, base: float) -> str:
    if base <= 0: return "?"
    mult = trail_k_used / base
    # Map to nearest arm
    candidates = [
        (0.70, "very_tight"),
        (0.85, "tight"),
        (1.00, "default"),
        (1.20, "wide"),
        (1.40, "very_wide"),
    ]
    return min(candidates, key=lambda c: abs(mult - c[0]))[1]

with io.open(ROOT / "files" / "bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if '"event"' not in ln: continue
        try: e = json.loads(ln)
        except: continue
        ev = e.get("event", "")
        if ev not in ("entry", "exit"): continue
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
                "trail_k": float(e.get("trail_k") or 0),
                "mode": e.get("mode", "?"),
                "tf": e.get("tf", "?"),
                "dr": float(e.get("daily_range") or 0),
                "ml_proba": float(e.get("ml_proba") or 0),
            }
        elif ev == "exit":
            ent = entries.pop(sym, None)
            if not ent: continue
            ex_price = float(e.get("exit_price") or e.get("price") or 0)
            if ex_price <= 0 or ent["price"] <= 0: continue
            pnl = (ex_price - ent["price"]) / ent["price"] * 100
            bars = int(e.get("bars_held") or 0)
            reason = e.get("reason", "") or ""
            base = base_trail_k(ent["mode"], ent["tf"])
            trades.append({
                "sym": sym,
                "mode": ent["mode"],
                "tf": ent["tf"],
                "trail_k_used": ent["trail_k"],
                "base_trail_k": base,
                "arm": arm_name(ent["trail_k"], base),
                "pnl": pnl,
                "bars": bars,
                "dr": ent["dr"],
                "trail_hit": "ATR-трейл" in reason or "trail" in reason.lower(),
                "ml_proba": ent["ml_proba"],
            })

print(f"=== Trade-by-arm breakdown, last 30d ===")
print(f"Total paired trades: {len(trades)}\n")

# Per (mode/tf, arm)
by_key = defaultdict(list)
for t in trades:
    key = (f"{t['mode']}/{t['tf']}", t["arm"])
    by_key[key].append(t)

# Group by mode for printing
modes = sorted(set(f"{t['mode']}/{t['tf']}" for t in trades))
arms_order = ["very_tight", "tight", "default", "wide", "very_wide", "?"]

for mode_tf in modes:
    rows = [t for t in trades if f"{t['mode']}/{t['tf']}" == mode_tf]
    if len(rows) < 5: continue
    print(f"\n── {mode_tf}  (n={len(rows)}) ──")
    print(f"  {'arm':<12} {'n':>4} {'avg_pnl':>8} {'win%':>6} {'avg_bars':>8} {'trail_exit%':>10}  avg_dr")
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    for arm in arms_order:
        ar = by_arm.get(arm, [])
        if not ar: continue
        n = len(ar)
        avg_pnl = sum(r["pnl"] for r in ar) / n
        win = sum(1 for r in ar if r["pnl"] > 0) / n * 100
        avg_bars = sum(r["bars"] for r in ar) / n
        trail_exits = sum(1 for r in ar if r["trail_hit"]) / n * 100
        avg_dr = sum(r["dr"] for r in ar) / n
        print(f"  {arm:<12} {n:>4d} {avg_pnl:>+7.2f}% {win:>5.0f}% {avg_bars:>7.1f}  {trail_exits:>9.0f}%  {avg_dr:.2f}")

# Special view: high-volatility days (daily_range >= 5)
print(f"\n\n=== High-volatility (daily_range >= 5%) on impulse_speed/strong_trend ===")
high_vol = [t for t in trades if t["dr"] >= 5.0 and t["mode"] in ("impulse_speed", "strong_trend")]
print(f"  Total: {len(high_vol)}")
by_arm_hv = defaultdict(list)
for t in high_vol:
    by_arm_hv[t["arm"]].append(t)
print(f"  {'arm':<12} {'n':>4} {'avg_pnl':>8} {'win%':>6} {'avg_bars':>8} {'trail_exit%':>10}")
for arm in arms_order:
    ar = by_arm_hv.get(arm, [])
    if not ar: continue
    n = len(ar)
    avg_pnl = sum(r["pnl"] for r in ar) / n
    win = sum(1 for r in ar if r["pnl"] > 0) / n * 100
    avg_bars = sum(r["bars"] for r in ar) / n
    trail_exits = sum(1 for r in ar if r["trail_hit"]) / n * 100
    print(f"  {arm:<12} {n:>4d} {avg_pnl:>+7.2f}% {win:>5.0f}% {avg_bars:>7.1f}  {trail_exits:>9.0f}%")

# Trail-hit losses by arm — these are the ones our floor would help
print(f"\n=== Trail-hit losses (pnl<0 AND trail-stop): would wider trail help? ===")
losses_by_arm = defaultdict(list)
for t in trades:
    if t["trail_hit"] and t["pnl"] < 0:
        losses_by_arm[t["arm"]].append(t)
for arm in arms_order:
    ar = losses_by_arm.get(arm, [])
    if not ar: continue
    n = len(ar)
    avg_loss = sum(r["pnl"] for r in ar) / n
    avg_bars = sum(r["bars"] for r in ar) / n
    is_high_vol_modes = sum(1 for r in ar if r["mode"] in ("impulse_speed","strong_trend"))
    print(f"  {arm:<12} {n:>4d} losses, avg={avg_loss:>+5.2f}%, avg_bars={avg_bars:.1f}, "
          f"impulse/strong: {is_high_vol_modes}")
