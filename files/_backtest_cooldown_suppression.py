"""Validate (max available period): how often does the post-exit COOLDOWN mute a
coin that KEEPS RUNNING — i.e. a continuing move we don't re-alert on?

Objective is early alerts, not P&L. We exit + cooldown (COOLDOWN_BARS), and if
the coin continues up during that window the channel hears nothing. This
quantifies how often that happens and how big the muted continuation is, to
decide whether to decouple alert-cooldown from trade-cooldown.

Method (read-only): for each exit->cooldown_start in bot_events, take the exit
price and scan history klines over the cooldown window (COOLDOWN_BARS * tf). A
'muted continuation' = max high during cooldown rises >= TH above exit. Report
share and size, split by whether the day/sym was a watchlist top-20.

ASCII-only.  pyembed\python.exe files\_backtest_cooldown_suppression.py
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
                try: b.append((datetime.fromisoformat(r["ts"]), float(r["high"]), float(r["close"])))
                except Exception: continue
    _KC[sym] = b; return b


WL = set(json.load(io.open(ROOT/"files"/"watchlist.json", encoding="utf-8")))

# top-20 winner days (watchlist) — full dataset span
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

# exits with the prior exit price, paired to cooldown_start
last_exit_px = {}
cooldowns = []
for ln in io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8", errors="replace"):
    if '"event"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    et = e.get("event", "")
    sym = e.get("sym") or e.get("symbol") or ""
    try: dt = datetime.fromisoformat((e.get("ts","")).replace("Z","+00:00"))
    except: continue
    if et == "exit":
        last_exit_px[sym] = (dt, _f(e.get("exit_price") or e.get("price")), e.get("tf", "15m"))
    elif et == "cooldown_start":
        ex = last_exit_px.get(sym)
        if ex and ex[1] and ex[1] > 0:
            cooldowns.append({"sym": sym, "dt": ex[0], "px": ex[1], "tf": ex[2],
                              "win": (ex[0].strftime("%Y-%m-%d"), sym) in top})


def cont_pct(c):
    """Max % the coin rose above exit price during the cooldown window."""
    dur_min = CD_BARS * TFMIN.get(c["tf"], 15)
    end = c["dt"] + timedelta(minutes=dur_min)
    bars = [b for b in klines(c["sym"]) if c["dt"] < b[0] <= end]
    if not bars: return None
    mx = max(b[1] for b in bars)
    return (mx - c["px"]) / c["px"] * 100.0


rows = [(c, cont_pct(c)) for c in cooldowns]
rows = [(c, p) for c, p in rows if p is not None]
if not rows:
    print("no cooldowns with kline coverage"); sys.exit(0)

days = sorted({c["dt"].strftime("%Y-%m-%d") for c, _ in rows})
print("=" * 66)
print(f"Cooldown-suppression validation (COOLDOWN_BARS={CD_BARS})")
print("=" * 66)
print(f"cooldowns with kline coverage: {len(rows)}   span {days[0]}..{days[-1]}\n")

def share(rs, th):
    n = len(rs)
    return sum(1 for _, p in rs if p >= th)/n*100 if n else 0

for label, subset in (("ALL cooldowns", rows),
                      ("on top-20 days", [(c, p) for c, p in rows if c["win"]]),
                      ("non-top-20", [(c, p) for c, p in rows if not c["win"]])):
    n = len(subset)
    if not n:
        print(f"{label:<16} n=0"); continue
    med = sorted(p for _, p in subset)[n//2]
    print(f"{label:<16} n={n:<5} median_cont={med:+.2f}%   "
          f">=+3%: {share(subset,3):.0f}%   >=+5%: {share(subset,5):.0f}%   "
          f">=+10%: {share(subset,10):.0f}%")
print("\nRead: 'cont' = max rise above our exit price DURING the cooldown window.")
print(">=+5% share = how often cooldown muted a still-running coin a re-alert")
print("would have surfaced. High share on top-20 days = decouple alert-cooldown.")
