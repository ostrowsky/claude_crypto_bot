"""Re-entry replay (priority A, structurally-sound lever): after the bot EXITS a
position, if the coin RESUMES momentum (breaks +R% above our exit within W bars)
we re-enter and trail. Re-entry is selective by construction — a position that
does NOT resume never re-triggers — so unlike holding/wide-stops it should not
bleed the (5:1) losers. Test that across ALL exits (winners + losers).

Motivating case: UNI 2026-06-16 — exited at 01:28 (RSI 85), then ran +15%;
cooldown(19 bars)+gates blocked re-entry.

Method (read-only, shadow): for each first entry->exit (last N days, watchlist):
  - after exit_dt, scan forward klines; if high >= exit*(1+R%) within W bars,
    RE-ENTER at that trigger price; then trail TRAIL% to horizon -> reentry pnl.
  - aggregate: # re-entry triggers, mean reentry pnl, split winner(top20)/loser,
    NET per re-entry. Also how often a re-entry triggers at all.

ASCII-only. Run from repo root:
    pyembed\python.exe files\_backtest_reentry.py
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
W_BARS = 24          # 6h window to resume after exit
HORIZON = 96         # 24h hold cap after re-entry
TRAIL = 0.05


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

open_pos = {}; exits = []
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
        open_pos.setdefault(sym, {"d": dt.strftime("%Y-%m-%d")})
    else:
        en = open_pos.pop(sym, None)
        if not en: continue
        xp = _f(e.get("exit_price") or e.get("price"))
        if not xp or xp <= 0: continue
        exits.append({"sym": sym, "exit_dt": dt, "exit": xp,
                      "win": (en["d"], sym) in top})


def reentry(t, R):
    """If price breaks +R% above exit within W_BARS, re-enter there, trail to
    horizon. Returns (triggered, pnl%)."""
    bars = [b for b in klines(t["sym"]) if b["ts"] > t["exit_dt"]]
    if not bars: return False, None
    win = bars[:W_BARS]
    trig = t["exit"] * (1 + R/100.0)
    ti = None
    for i, b in enumerate(win):
        if b["high"] >= trig:
            ti = i; break
    if ti is None: return False, None
    e = trig                      # re-enter at trigger price
    path = bars[ti: ti+HORIZON]
    peak = e; stop = peak*(1-TRAIL)
    for b in path:
        if b["low"] <= stop: return True, (stop-e)/e*100
        if b["high"] > peak: peak = b["high"]; stop = peak*(1-TRAIL)
    return True, (path[-1]["close"]-e)/e*100


print("="*66)
print(f"Re-entry replay — after exit, re-enter on +R% resume ({DAYS}d)")
print("="*66)
print(f"exits in window: {len(exits)}  (winners={sum(1 for x in exits if x['win'])}"
      f" losers={sum(1 for x in exits if not x['win'])})")
print(f"window={W_BARS}b/6h  trail={TRAIL*100:.0f}%  hold cap={HORIZON}b\n")
print(f"{'R%trigger':>10}{'triggers':>10}{'trig%':>7}{'mean_pnl':>10}"
      f"{'W_mean':>9}{'L_mean':>9}{'NET/exit':>10}")
print("-"*65)
for R in (2.0, 3.0, 5.0, 8.0):
    res = [(t, reentry(t, R)) for t in exits]
    trig = [(t, pnl) for t, (ok, pnl) in res if ok and pnl is not None]
    n = len(trig)
    if not n:
        print(f"{R:>9.0f}%{0:>10}"); continue
    mean = sum(p for _, p in trig)/n
    w = [p for t, p in trig if t["win"]]; l = [p for t, p in trig if not t["win"]]
    wm = sum(w)/len(w) if w else float("nan")
    lm = sum(l)/len(l) if l else float("nan")
    # NET per exit = total re-entry pnl spread over ALL exits (cost of re-entry program)
    net = sum(p for _, p in trig)/len(exits)
    print(f"{R:>9.0f}%{n:>10}{100*n/len(exits):>6.0f}%{mean:>+10.2f}"
          f"{wm:>+9.2f}{lm:>+9.2f}{net:>+10.3f}")
print("\nRead: triggers = exits where the coin resumed +R% (re-entry fired).")
print("Re-entry is selective; want mean_pnl>0 and W_mean strong with L_mean not")
print("deeply negative. NET/exit = avg added pnl per exited position.")
