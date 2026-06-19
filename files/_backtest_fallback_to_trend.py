"""Fallback-to-trend backtest: when impulse_speed is curtailed, is it better to
RECLASSIFY the candidate as 'trend' (enter, tighter behavior) than to HARD-BLOCK
it? And how does that compare to entering AS impulse_speed?

Context: impulse_speed curtailment hard-blocked 266-813 candidates/day in the
altseason (most are fast trends silently upgraded to impulse_speed). Blocking
protects realized pnl but forgoes coverage of fast movers. Fallback-to-trend
keeps the entry with trend exit behavior (min-buffer 0 vs 1.5%, shorter
max_hold), so we measure the three-way tradeoff on REAL impulse_speed entries.

Replays the forward price path (history/<sym>_15m.csv) under ATR-based trailing
for each policy:
  AS_IMPULSE  : buffer=max(k*ATR, 1.5%*price), max_hold=48 bars
  AS_TREND    : buffer=max(k*ATR, 0),          max_hold=16 bars
  BLOCK       : no trade (pnl 0, no coverage)
k = ATR_TRAIL_K (2.0). Reports realised pnl, win%, and big-mover (ret proxy:
fwd_max >= +8%) coverage/capture per policy.

ASCII-only. Run from repo root:
    pyembed\python.exe files\_backtest_fallback_to_trend.py
"""
from __future__ import annotations
import csv, io, json, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "history"
CUT = datetime.now(timezone.utc) - timedelta(days=25)
K = 2.0
BIG_MOVE = 8.0   # fwd peak >= +8% from entry = a 'big mover' worth covering

_KC: dict = {}
def klines(sym):
    if sym in _KC: return _KC[sym]
    p = HIST / f"{sym}_15m.csv"; b = []
    if p.exists():
        with io.open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    b.append({"ts": datetime.fromisoformat(r["ts"]),
                              "high": float(r["high"]), "low": float(r["low"]),
                              "close": float(r["close"])})
                except Exception:
                    continue
    _KC[sym] = b
    return b


def atr_at(bars, idx, n=14):
    if idx < 1: return None
    lo = max(1, idx - n + 1)
    trs = []
    for j in range(lo, idx + 1):
        h, l, pc = bars[j]["high"], bars[j]["low"], bars[j-1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs) / len(trs) if trs else None


def fwd_from(sym, entry_dt):
    bars = klines(sym)
    idx = None
    for i, b in enumerate(bars):
        if b["ts"] >= entry_dt:
            idx = i; break
    if idx is None or idx < 14:
        return None, None
    return bars, idx


def sim(bars, idx, entry, buf_floor_pct, max_hold):
    atr = atr_at(bars, idx) or 0.0
    fwd = bars[idx: idx + max_hold]
    if not fwd: return None, None
    peak = entry
    buf = max(K * atr, buf_floor_pct * entry)
    stop = peak - buf
    fwd_max = entry
    for b in fwd:
        fwd_max = max(fwd_max, b["high"])
        if b["low"] <= stop:
            return (stop - entry) / entry * 100, (fwd_max - entry) / entry * 100
        if b["high"] > peak:
            peak = b["high"]; stop = peak - max(K * atr, buf_floor_pct * entry)
    return (fwd[-1]["close"] - entry) / entry * 100, (fwd_max - entry) / entry * 100


# pair impulse_speed entries
opn = {}; trades = []
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
        opn[sym] = {"p": float(e.get("price") or 0), "mode": e.get("mode","?"), "dt": dt}
    else:
        en = opn.pop(sym, None)
        if not en or en["mode"] != "impulse_speed": continue
        if en["p"] <= 0: continue
        trades.append((sym, en["dt"], en["p"]))

rows = []
for sym, dt, entry in trades:
    bars, idx = fwd_from(sym, dt)
    if bars is None: continue
    imp = sim(bars, idx, entry, 0.015, 48)
    trd = sim(bars, idx, entry, 0.0, 16)
    if imp[0] is None or trd[0] is None: continue
    rows.append({"imp": imp[0], "trd": trd[0], "fwd_max": imp[1]})

print("="*68)
print(f"Fallback-to-trend backtest (impulse_speed entries, 25d, n={len(rows)})")
print("="*68)
if not rows:
    sys.exit(0)
big = [r for r in rows if r["fwd_max"] >= BIG_MOVE]

def line(name, key, entered=True):
    if not entered:
        print(f"  {name:<12} avg_pnl=+0.000  win=  -   (no trade)  "
              f"big_movers covered=0/{len(big)}")
        return
    n = len(rows)
    avg = sum(r[key] for r in rows)/n
    win = sum(1 for r in rows if r[key] > 0)/n*100
    bavg = (sum(r[key] for r in big)/len(big)) if big else float("nan")
    print(f"  {name:<12} avg_pnl={avg:+.3f}  win={win:4.0f}%   "
          f"big_movers avg_pnl={bavg:+.2f}% covered={len(big)}/{len(big)}")

print(f"big movers in set (fwd peak>=+{BIG_MOVE}%): {len(big)}\n")
line("AS_IMPULSE", "imp")
line("AS_TREND", "trd")
line("BLOCK", "imp", entered=False)
print()
n = len(rows)
print(f"net per-trade: AS_IMPULSE {sum(r['imp'] for r in rows)/n:+.3f}  "
      f"AS_TREND {sum(r['trd'] for r in rows)/n:+.3f}  BLOCK +0.000")
print("\nRead: BLOCK = 0 pnl but 0 coverage. Fallback-to-trend is worth it if")
print("AS_TREND keeps big-mover coverage at >= breakeven net (so we don't pay")
print("to catch movers) and ideally beats AS_IMPULSE on net.")
