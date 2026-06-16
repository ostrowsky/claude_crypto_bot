"""Exit-monetization replay (priority A): of the watchlist top-20 winners the
bot ENTERED, how much of the move did we give back by exiting early, which exit
class leaks most, and would suppressing soft exits (RSI/EMA/MACD weakness) while
in profit capture more?  SHADOW only — no production SELL changes.

Motivating case: UNI 2026-06-16 (top-20, +15% eod) — bot entered, exited 22min
later on "RSI overbought 85", then UNI ran without us; re-entry blocked.

Method (read-only):
  - Winners = watchlist top-20 (label_top20) days, last N days.
  - For each ENTERED winner (first entry->its exit from bot_events): actual
    realized pnl, exit-reason class, entry/exit time+price.
  - Forward path from history klines: MFE_after_exit = max high in HORIZON bars
    after exit  -> giveback% left on table.
  - Policy replay from ENTRY price over HORIZON bars:
      ACTUAL        : realized pnl (entry->actual exit)
      SUPPRESS_SOFT : if actual exit was a SOFT class (rsi/ema/macd) AND we were
                      in profit, ignore it; exit instead on an ATR-style trail
                      (buffer=TRAIL%*price) or horizon close.
      HOLD_EOD      : exit at the horizon close (hold through).
  Report per exit-class: n, actual pnl, giveback; and policy means + capture
  vs eod_return.

ASCII-only. Run from repo root:
    pyembed\python.exe files\_backtest_exit_monetization.py
"""
from __future__ import annotations
import csv, io, json, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "history"
DAYS = 21
CUT = datetime.now(timezone.utc) - timedelta(days=DAYS)
HORIZON_BARS = 96          # 24h on 15m
TRAIL = 0.05               # 5% trailing buffer for the SUPPRESS_SOFT replay


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
                try:
                    b.append({"ts": datetime.fromisoformat(r["ts"]),
                              "high": float(r["high"]), "low": float(r["low"]),
                              "close": float(r["close"])})
                except Exception: continue
    _KC[sym] = b; return b


def classify(reason: str) -> str:
    r = (reason or "").lower()
    if "atr" in r or "трейл" in r or "trail" in r: return "atr_trail"
    if "max_hold" in r or "время" in r or "лимит" in r or "hold" in r: return "time_hold"
    if "ema" in r: return "ema_weak"
    if "rsi" in r or "перекупл" in r: return "rsi"
    if "macd" in r: return "macd"
    if "rotation" in r or "ротац" in r: return "rotation"
    return "other"


SOFT = {"rsi", "ema_weak", "macd"}


# top-20 winners (watchlist) + eod
top, eod = set(), {}
with io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        if '"label_top20"' not in ln: continue
        try: e = json.loads(ln)
        except: continue
        ts = e.get("ts")
        if not ts: continue
        dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
        if dt < CUT: continue
        sym = e.get("symbol")
        if sym not in WL: continue
        d = dt.strftime("%Y-%m-%d")
        if e.get("label_top20") == 1: top.add((d, sym))
        ev = _f(e.get("eod_return_pct"))
        if ev is not None:
            ev = ev if abs(ev) > 5 else ev*100
            eod[(d, sym)] = max(eod.get((d, sym), -999), ev)


# first entry->exit per (day,sym) from bot_events
open_pos = {}; trades = {}
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
    d = dt.strftime("%Y-%m-%d")
    if ev == "entry":
        if sym not in open_pos:
            open_pos[sym] = {"dt": dt, "d": d, "p": _f(e.get("price") or e.get("entry_price"))}
    else:
        en = open_pos.pop(sym, None)
        if not en or not en["p"] or en["p"] <= 0: continue
        xp = _f(e.get("exit_price") or e.get("price"))
        if not xp or xp <= 0: continue
        key = (en["d"], sym)
        if key in top and key not in trades:   # first entry on a winner-day
            trades[key] = {
                "sym": sym, "entry_dt": en["dt"], "entry": en["p"],
                "exit_dt": dt, "exit": xp,
                "pnl": (xp-en["p"])/en["p"]*100,
                "cls": classify(e.get("reason") or ""),
                "eod": eod.get(key),
            }


def fwd_bars(sym, dt):
    return [b for b in klines(sym) if b["ts"] >= dt][:HORIZON_BARS]


def replay_from_entry(t, suppress_soft):
    """Return pnl% under policy. suppress_soft=False -> just measure HOLD_EOD."""
    fwd = fwd_bars(t["sym"], t["entry_dt"])
    if not fwd: return None
    entry = t["entry"]
    if suppress_soft:
        # honor the actual exit UNLESS it was soft & in profit; then trail to horizon
        if t["cls"] in SOFT and t["pnl"] > 0.5:
            peak = entry; stop = peak*(1-TRAIL)
            for b in fwd:
                if b["low"] <= stop: return (stop-entry)/entry*100
                if b["high"] > peak: peak = b["high"]; stop = peak*(1-TRAIL)
            return (fwd[-1]["close"]-entry)/entry*100
        return t["pnl"]   # non-soft or not-in-profit: keep actual exit
    else:
        return (fwd[-1]["close"]-entry)/entry*100   # HOLD_EOD


def mfe_after_exit(t):
    fwd = fwd_bars(t["sym"], t["exit_dt"])
    if not fwd: return None
    mx = max(b["high"] for b in fwd)
    return (mx - t["exit"])/t["exit"]*100


rows = list(trades.values())
print("="*72)
print(f"Exit-monetization replay — watchlist top-20 winners entered, {DAYS}d")
print("="*72)
print(f"entered winners with data: {len(rows)}")
if not rows: sys.exit(0)

# per exit-class: actual pnl + giveback
by = defaultdict(list)
for t in rows:
    t["giveback"] = mfe_after_exit(t)
    by[t["cls"]].append(t)
print(f"\n{'exit_class':<12}{'n':>4}{'avg_pnl':>9}{'avg_giveback':>14}{'avg_eod':>9}")
for cls, ts in sorted(by.items(), key=lambda x: -len(x[1])):
    n = len(ts)
    ap = sum(x["pnl"] for x in ts)/n
    gb = [x["giveback"] for x in ts if x["giveback"] is not None]
    agb = sum(gb)/len(gb) if gb else float("nan")
    ed = [x["eod"] for x in ts if x["eod"] is not None]
    aed = sum(ed)/len(ed) if ed else float("nan")
    print(f"{cls:<12}{n:>4}{ap:>+9.2f}{agb:>+14.2f}{aed:>+9.2f}")

# policy comparison
def mean(xs): xs=[x for x in xs if x is not None]; return sum(xs)/len(xs) if xs else float("nan")
actual = [t["pnl"] for t in rows]
hold = [replay_from_entry(t, False) for t in rows]
supp = [replay_from_entry(t, True) for t in rows]
print(f"\n-- policy mean realized pnl over entered winners (n={len(rows)}) --")
print(f"  ACTUAL          {mean(actual):+.2f}%")
print(f"  SUPPRESS_SOFT   {mean(supp):+.2f}%   (ignore RSI/EMA/MACD exit when pnl>0.5%, trail {TRAIL*100:.0f}%)")
print(f"  HOLD_EOD        {mean(hold):+.2f}%   (hold entry->{HORIZON_BARS}b)")
soft_rows = [t for t in rows if t["cls"] in SOFT and t["pnl"] > 0.5]
print(f"\nsoft-exit-in-profit winners (the addressable leak): {len(soft_rows)}")
if soft_rows:
    a = mean([t["pnl"] for t in soft_rows])
    s = mean([replay_from_entry(t, True) for t in soft_rows])
    print(f"  on those: ACTUAL {a:+.2f}%  ->  SUPPRESS_SOFT {s:+.2f}%  (delta {s-a:+.2f}pp)")
print("\nRead: giveback = % the coin rose ABOVE our exit within 24h (left on table).")
print("If SUPPRESS_SOFT > ACTUAL with big giveback on rsi/ema classes, early soft")
print("exits are the capture leak; H5 trailing-only / re-entry is the fix to test.")
