"""Validate (max period) the cooldown re-alert mechanism's RATE and PRECISION
before wiring it: if we fire an info re-alert when a coin we exited breaks
+TRIGGER% above exit during cooldown, how many alerts/day would that add, and
what share land on actual watchlist top-20 movers (precision)? Recall = of
top-20 cooldowns, how many re-alert.

Keeps the channel from being spammed: only worth wiring if rate is modest and
precision clearly beats the base top-20 share of cooldowns.

Read-only, history klines (60d).  pyembed\python.exe files\_backtest_realert_precision.py
"""
from __future__ import annotations
import csv, io, json, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "history"
try:
    import config as _cfg
    CD_BARS = int(getattr(_cfg, "COOLDOWN_BARS", 19))
except Exception:
    CD_BARS = 19
TFMIN = {"15m": 15, "1h": 60}


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
                try: b.append((datetime.fromisoformat(r["ts"]), float(r["high"])))
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

last_exit = {}
cooldowns = []
for ln in io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8", errors="replace"):
    if '"event"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    et = e.get("event", ""); sym = e.get("sym") or e.get("symbol") or ""
    try: dt = datetime.fromisoformat((e.get("ts","")).replace("Z","+00:00"))
    except: continue
    if et == "exit":
        last_exit[sym] = (dt, _f(e.get("exit_price") or e.get("price")), e.get("tf", "15m"))
    elif et == "cooldown_start":
        ex = last_exit.get(sym)
        if ex and ex[1] and ex[1] > 0:
            cooldowns.append({"sym": sym, "dt": ex[0], "px": ex[1], "tf": ex[2],
                              "win": (ex[0].strftime("%Y-%m-%d"), sym) in top})

# only cooldowns with kline coverage
cov = []
for c in cooldowns:
    dur = CD_BARS * TFMIN.get(c["tf"], 15)
    end = c["dt"] + timedelta(minutes=dur)
    bars = [b for b in klines(c["sym"]) if c["dt"] < b[0] <= end]
    if bars:
        c["maxhigh"] = max(b[1] for b in bars); cov.append(c)

days = sorted({c["dt"].strftime("%Y-%m-%d") for c in cov})
ndays = max(1, len(days))
n_top_cd = sum(1 for c in cov if c["win"])
base_prec = n_top_cd / len(cov) * 100 if cov else 0
print("=" * 70)
print(f"Re-alert rate & precision (max period, COOLDOWN_BARS={CD_BARS})")
print("=" * 70)
print(f"cooldowns w/ coverage: {len(cov)}  days: {ndays}  span {days[0]}..{days[-1]}")
print(f"base top-20 share of cooldowns: {base_prec:.0f}%  (precision to beat)\n")
print(f"  {'trigger':>8}{'realerts':>10}{'per_day':>9}{'precision%':>12}{'recall_top20%':>15}")
for tr in (3.0, 5.0, 8.0):
    fired = [c for c in cov if (c["maxhigh"]-c["px"])/c["px"]*100 >= tr]
    n = len(fired)
    top_fired = sum(1 for c in fired if c["win"])
    prec = top_fired/n*100 if n else 0
    recall = top_fired/n_top_cd*100 if n_top_cd else 0
    print(f"  {tr:>7.0f}%{n:>10}{n/ndays:>9.1f}{prec:>12.0f}{recall:>15.0f}")
print("\nRead: per_day = added info-alerts/day (channel-noise budget). precision =")
print("share that were real top-20 movers (vs base). recall = of top-20 cooldowns,")
print("how many we'd re-surface. Wire only if rate modest AND precision >> base.")
