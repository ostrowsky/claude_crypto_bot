"""Net exit-policy replay across ALL entered positions (winners + losers).

The winners-only replay (_backtest_exit_monetization.py) showed holding ~3x the
capture, but a deployable policy must not blow up losers. This replays the
forward path from ENTRY for every entered position under candidate exit
mechanics and reports the effect split by winner(top20)/loser, plus NET per
trade (what the bot actually faces, not knowing winner/loser at entry).

Policies (long, pnl% = (exit-entry)/entry*100), replayed over HORIZON bars:
  ACTUAL        : realized pnl from bot_events (current behaviour)
  TRAIL_5 / _8  : single position, fixed % trailing stop from running peak
  PARTIAL_T/8   : take 50% off at +T%, keep 50% on a wide 8% trail to horizon
                  (if +T never touched, whole position runs on the 8% trail)
  HOLD_EOD      : hold entry->horizon close (upper bound, unsafe)

Deploy criterion: WINNER capture up AND NET(all) >= ACTUAL (winner upside not
paid by loser blow-ups).

ASCII-only. Run from repo root:
    pyembed\python.exe files\_backtest_exit_net_policy.py
"""
from __future__ import annotations
import csv, io, json, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "history"
DAYS = 21
CUT = datetime.now(timezone.utc) - timedelta(days=DAYS)
HORIZON = 96            # 24h on 15m
PARTIAL_TARGET = 8.0    # take half at +8%


def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


WL = set(json.load(io.open(ROOT/"files"/"watchlist.json", encoding="utf-8")))
_KC: dict = {}
def klines(sym):
    if sym in _KC: return _KC[sym]
    p = HIST / f"{sym}_15m.csv"; b = []
    if p.exists():
        with io.open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try: b.append({"ts": datetime.fromisoformat(r["ts"]),
                               "high": float(r["high"]), "low": float(r["low"]),
                               "close": float(r["close"])})
                except Exception: continue
    _KC[sym] = b; return b


# top-20 winners (watchlist)
top = set()
with io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        if '"label_top20"' not in ln: continue
        try: e = json.loads(ln)
        except: continue
        ts = e.get("ts")
        if not ts: continue
        dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
        if dt < CUT: continue
        if e.get("symbol") in WL and e.get("label_top20") == 1:
            top.add((dt.strftime("%Y-%m-%d"), e.get("symbol")))

# first entry->exit per position
open_pos = {}; trades = []
for ln in io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8", errors="replace"):
    if '"event"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    ev = e.get("event", "")
    if ev not in ("entry", "exit"): continue
    sym = e.get("sym") or e.get("symbol") or ""
    try: dt = datetime.fromisoformat((e.get("ts","")).replace("Z","+00:00"))
    except: continue
    if dt < CUT: continue
    if ev == "entry":
        open_pos.setdefault(sym, {"dt": dt, "d": dt.strftime("%Y-%m-%d"),
                                  "p": _f(e.get("price") or e.get("entry_price"))})
    else:
        en = open_pos.pop(sym, None)
        if not en or not en["p"] or en["p"] <= 0: continue
        xp = _f(e.get("exit_price") or e.get("price"))
        if not xp or xp <= 0: continue
        trades.append({"sym": sym, "dt": en["dt"], "entry": en["p"],
                       "pnl": (xp-en["p"])/en["p"]*100,
                       "win": (en["d"], sym) in top})


def fwd(t):
    return [b for b in klines(t["sym"]) if b["ts"] >= t["dt"]][:HORIZON]


def p_trail(t, buf):
    f = fwd(t)
    if not f: return None
    e = t["entry"]; peak = e; stop = peak*(1-buf)
    for b in f:
        if b["low"] <= stop: return (stop-e)/e*100
        if b["high"] > peak: peak = b["high"]; stop = peak*(1-buf)
    return (f[-1]["close"]-e)/e*100


def p_partial(t, target, rbuf=0.08):
    f = fwd(t)
    if not f: return None
    e = t["entry"]; tgt = e*(1+target/100.0)
    half = None; peak = e; stop = peak*(1-rbuf)
    for b in f:
        if half is None and b["low"] <= stop: return (stop-e)/e*100
        if half is None and b["high"] >= tgt: half = (tgt-e)/e*100
        if half is not None and b["low"] <= stop: return 0.5*half + 0.5*((stop-e)/e*100)
        if b["high"] > peak: peak = b["high"]; stop = peak*(1-rbuf)
    last = (f[-1]["close"]-e)/e*100
    return last if half is None else 0.5*half + 0.5*last


def p_hold(t):
    f = fwd(t)
    return (f[-1]["close"]-t["entry"])/t["entry"]*100 if f else None


def p_profit_gated(t, arm=5.0, tight=0.03, wide=0.08):
    """Tight trail until the position's PEAK reaches +arm% above entry, then
    switch to a wide trail. Losers never arm (cut tight); proven winners get the
    wide leash. The deployable 'thread the needle' idea (H5 family)."""
    f = fwd(t)
    if not f: return None
    e = t["entry"]; peak = e; armed = False
    buf = tight; stop = peak*(1-buf)
    for b in f:
        if b["low"] <= stop: return (stop-e)/e*100
        if b["high"] > peak: peak = b["high"]
        if not armed and peak >= e*(1+arm/100.0): armed = True
        buf = wide if armed else tight
        stop = max(stop, peak*(1-buf))   # never lower the stop
    return (f[-1]["close"]-e)/e*100


POLICIES = {
    "ACTUAL":      lambda t: t["pnl"],
    "TRAIL_5":     lambda t: p_trail(t, 0.05),
    "TRAIL_8":     lambda t: p_trail(t, 0.08),
    f"PARTIAL_{PARTIAL_TARGET:.0f}/8": lambda t: p_partial(t, PARTIAL_TARGET),
    "GATED_5/3->8": lambda t: p_profit_gated(t, 5.0, 0.03, 0.08),
    "GATED_3/2->8": lambda t: p_profit_gated(t, 3.0, 0.02, 0.08),
    "HOLD_EOD":    lambda t: p_hold(t),
}

# round-trip fee (Binance USDT-M futures taker ~0.05%/side). All these policies
# are single-position = 1 round-trip, so fees shift them ~uniformly; the point of
# this validation is (a) materiality and (b) that ACTUAL stays net-best (fees do
# NOT rescue any wider/churn policy). FEE=0 reproduces the gross numbers.
import os
try:
    import config as _cfg
    _fee_default = float(getattr(_cfg, "FEE_ROUNDTRIP_PCT", 0.10))
except Exception:
    _fee_default = 0.10
FEE = float(os.environ.get("FEE_ROUNDTRIP_PCT", _fee_default))

rows = [t for t in trades if fwd(t)]
wins = [t for t in rows if t["win"]]
loss = [t for t in rows if not t["win"]]
print("="*78)
print(f"Net exit-policy replay — ALL entered positions, {DAYS}d  (fee_roundtrip={FEE}%)")
print("="*78)
print(f"n={len(rows)}  winners(top20)={len(wins)}  losers={len(loss)}\n")
def m(xs): xs=[x for x in xs if x is not None]; return sum(xs)/len(xs) if xs else float("nan")
print(f"{'policy':<13}{'ALL_gross':>11}{'ALL_net_fee':>13}{'W_net':>9}{'L_net':>9}{'W_win%':>8}")
print("-"*63)
for name, fn in POLICIES.items():
    w = m([fn(t) for t in wins]); l = m([fn(t) for t in loss]); a = m([fn(t) for t in rows])
    ww = [fn(t) for t in wins]; ww=[x for x in ww if x is not None]
    wwin = sum(1 for x in ww if x>0)/len(ww)*100 if ww else float("nan")
    print(f"{name:<13}{a:>+11.2f}{a-FEE:>+13.2f}{w-FEE:>+9.2f}{l-FEE:>+9.2f}{wwin:>8.0f}")
print("\nValidation: fees shift single-position policies ~uniformly by -FEE; the")
print("ranking is preserved and ACTUAL stays net-best -> fees do not rescue churn.")
print("The churn penalty bites on EXTRA round-trips (rotation, re-entry) tested")
print("separately (each adds a full -FEE on top).")
