"""Pair ALGO entries with their exits, compute hold time, P&L, exit reason.
Goal: confirm trail-stop is too tight on fresh impulse_speed entries."""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent

NOW = datetime.now(timezone.utc)
CUT = NOW - timedelta(days=14)

events = []
with io.open(ROOT / "bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if "ALGO" not in ln: continue
        try: e = json.loads(ln)
        except: continue
        sym = e.get("sym") or e.get("symbol") or ""
        if sym != "ALGOUSDT": continue
        ts = e.get("ts","")
        try: dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        except: continue
        if dt < CUT: continue
        e["_dt"] = dt
        events.append(e)

events.sort(key=lambda e: e["_dt"])

print(f"=== ALGOUSDT trades, last 14d ===\n")
print(f"{'ts':<12} {'type':6s} {'mode':14s}/{'tf':3s}  {'price':>8s}  {'P&L':>7s}  {'hold':>7s}  reason")

cur_entry = None
for e in events:
    typ = e.get("event","")
    if typ not in ("entry","exit","cooldown_start"):
        continue
    ts = e["_dt"].strftime("%m-%d %H:%M")
    tf = e.get("tf","?")
    mode = e.get("mode","?")

    if typ == "entry":
        price = e.get("entry_price") or e.get("price") or 0
        cur_entry = (e["_dt"], float(price) if price else 0, mode, tf)
        print(f"{ts:<12} {typ:6s} {mode:14s}/{tf:3s}  {price:>8}  {'-':>7s}  {'-':>7s}")
    elif typ == "exit":
        price = e.get("exit_price") or e.get("price") or 0
        try: price_f = float(price)
        except: price_f = 0
        if cur_entry and cur_entry[1] > 0 and price_f > 0:
            pnl = (price_f - cur_entry[1]) / cur_entry[1] * 100
            hold = e["_dt"] - cur_entry[0]
            hold_str = f"{int(hold.total_seconds()//60)}min"
        else:
            pnl = 0; hold_str = "?"
        reason = (e.get("reason") or "")[:70]
        # ATR trail in Russian — translate
        reason = reason.replace("ATR-трейл пробит", "ATR-trail hit")
        print(f"{ts:<12} {typ:6s} {mode:14s}/{tf:3s}  {price:>8}  {pnl:>+6.2f}%  {hold_str:>7s}  {reason}")
        cur_entry = None

# Now look at ATR_TRAIL_K config
print(f"\n=== Trail-stop config snapshot ===")
import sys as _sys; _sys.path.insert(0, str(ROOT / "files"))
import config
keys = [
    "ATR_TRAIL_K", "ATR_TRAIL_K_STRONG",
    "ATR_TRAIL_K_15M_IMPULSE", "ATR_TRAIL_K_15M_IMPULSE_BULL",
    "ATR_TRAIL_K_IMPULSE_SPEED_15M", "ATR_TRAIL_K_IMPULSE_SPEED_1H",
    "ATR_TRAIL_K_BULL_DAY",
    "MAX_HOLD_BARS",
    "FAST_LOSS_EXIT_ENABLED",
]
for k in keys:
    v = getattr(config, k, "<missing>")
    print(f"  {k:38s} = {v}")
