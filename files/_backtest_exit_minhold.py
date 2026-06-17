"""Validate (max period) a MIN-HOLD grace before the ATR-trail exit, to stop
1-bar whipsaws (e.g. LQTY 2026-06-17: 1h impulse_speed BUY then trail-SELL 99s
later, -4.8%). The trail currently can fire on the forming bar's intra-poll tick
with no grace.

Replays each entry's forward path under a % trailing stop that is DISABLED for
the first K bars (min-hold), then active. Reports whipsaw rate (exit <=3 bars at
a loss) and net pnl by K, split winners/losers — so a min-hold is only adopted
if it cuts whipsaws WITHOUT letting real reversals run materially deeper (net).

Read-only, history klines (60d).  pyembed\python.exe files\_backtest_exit_minhold.py
"""
from __future__ import annotations
import csv, io, json, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "history"
CUT = datetime.now(timezone.utc) - timedelta(days=60)
HORIZON = 96
BUF = 0.05            # 5% trailing buffer (mode-agnostic proxy)
MODE = None          # None = all modes; or "impulse_speed"

def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return None

_KC = {}
def klines(sym):
    if sym in _KC: return _KC[sym]
    p = HIST / f"{sym}_15m.csv"; b = []
    if p.exists():
        with io.open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try: b.append((datetime.fromisoformat(r["ts"]), float(r["high"]), float(r["low"]), float(r["close"])))
                except Exception: continue
    _KC[sym] = b; return b

WL = set(json.load(io.open(ROOT/"files"/"watchlist.json", encoding="utf-8")))
top = set()
for ln in io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8", errors="replace"):
    if '"label_top20"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    ts = e.get("ts")
    if not ts: continue
    dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
    if e.get("symbol") in WL and e.get("label_top20") == 1:
        top.add((dt.strftime("%Y-%m-%d"), e.get("symbol")))

opn = {}; trades = []
for ln in io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8", errors="replace"):
    if '"event"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    et = e.get("event", ""); sym = e.get("sym") or e.get("symbol") or ""
    if et not in ("entry", "exit"): continue
    try: dt = datetime.fromisoformat((e.get("ts","")).replace("Z","+00:00"))
    except: continue
    if dt < CUT: continue
    if et == "entry":
        opn[sym] = {"dt": dt, "d": dt.strftime("%Y-%m-%d"), "p": _f(e.get("price") or e.get("entry_price")), "mode": e.get("mode","?")}
    else:
        en = opn.pop(sym, None)
        if not en or not en["p"] or en["p"] <= 0: continue
        if MODE and en["mode"] != MODE: continue
        trades.append({"sym": sym, "dt": en["dt"], "entry": en["p"], "mode": en["mode"], "win": (en["d"], sym) in top})

def fwd(t): return [b for b in klines(t["sym"]) if b[0] >= t["dt"]][:HORIZON]

def sim(t, minhold):
    f = fwd(t)
    if not f: return None, None
    e = t["entry"]; peak = e; stop = peak*(1-BUF)
    for k, b in enumerate(f):
        # trail can only FIRE after minhold bars; peak still tracks meanwhile
        if k >= minhold and b[2] <= stop:
            return (stop-e)/e*100, k
        if b[1] > peak: peak = b[1]; stop = peak*(1-BUF)
    return (f[-1][3]-e)/e*100, len(f)

rows = [t for t in trades if fwd(t)]
wins = [t for t in rows if t["win"]]; loss = [t for t in rows if not t["win"]]
print("="*70)
print(f"Min-hold-before-trail validation ({'all modes' if not MODE else MODE}, 60d, buf {BUF*100:.0f}%)")
print("="*70)
print(f"n={len(rows)}  winners={len(wins)}  losers={len(loss)}\n")
def m(xs): xs=[x for x in xs if x is not None]; return sum(xs)/len(xs) if xs else float('nan')
print(f"  {'min_hold':>9}{'whipsaw%':>10}{'ALL_net':>9}{'W_net':>9}{'L_net':>9}")
for K in (0, 1, 2, 3):
    res = [(t, sim(t, K)) for t in rows]
    pnls = [(t, p) for t, (p, k) in res if p is not None]
    whip = sum(1 for t,(p,k) in res if p is not None and k is not None and k<=3 and p<0)/len(res)*100
    alln = m([p for _,p in pnls])
    wn = m([p for t,p in pnls if t["win"]]); ln_ = m([p for t,p in pnls if not t["win"]])
    print(f"  {K:>9}{whip:>10.1f}{alln:>+9.2f}{wn:>+9.2f}{ln_:>+9.2f}")
print("\nwhipsaw% = exits <=3 bars at a loss (the 1-bar flips). Adopt smallest K")
print("that cuts whipsaw% materially while ALL_net does not drop vs K=0.")
