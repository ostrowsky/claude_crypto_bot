"""Is impulse_speed lateness the discriminator between winners and losers?

RF_losing_mode_impulse_speed (2026-06-05): mode loses, median_lateness ~62%.
Hypothesis: impulse_speed enters too deep into the move — late entries have
little upside left but full downside. If lateness cleanly separates losers,
a lateness gate (block when the move is already >X% done) removes the bad tail
without touching early entries.

Method (read-only):
  - For each impulse_speed entry (bot_events, last N days), load history klines,
    run zigzag_labeler.detect_uptrends, find the uptrend covering the entry.
  - lateness_pct = (entry - trend_start) / (trend_peak - trend_start) * 100
    = how much of the up-move was ALREADY done at buy time.
  - remaining_upside_pct = (trend_peak - entry) / entry * 100.
  - Pair with realised pnl (exit event). Bucket by lateness; show avg pnl /
    win% / remaining upside per bucket, and what a lateness<=T gate would do.

ASCII-only. Run from repo root:
    pyembed\python.exe files\_backtest_impulse_speed_lateness.py
"""
from __future__ import annotations
import csv, io, json, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "history"
sys.path.insert(0, str(ROOT / "files"))
from zigzag_labeler import detect_uptrends

DAYS = 35
CUT = datetime.now(timezone.utc) - timedelta(days=DAYS)
MODE = "impulse_speed"

_KC: dict = {}
def klines(sym, tf="15m"):
    if sym in _KC: return _KC[sym]
    p = HIST / f"{sym}_{tf}.csv"; bars = []
    if p.exists():
        with io.open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    bars.append({"ts": datetime.fromisoformat(r["ts"]),
                                 "high": float(r["high"]), "low": float(r["low"]),
                                 "close": float(r["close"]), "open": float(r["open"]),
                                 "volume": float(r["volume"])})
                except Exception:
                    continue
    _KC[sym] = bars
    return bars


def lateness_for(sym, entry_dt, entry_price, tf="15m"):
    bars = klines(sym, tf)
    if not bars: return None
    trends = detect_uptrends(bars, symbol=sym, swing_pct=4.0,
                             max_drawdown_pct=2.0, min_duration_bars=4)
    best = None
    for t in trends:
        if t.end_ts < entry_dt or t.start_ts > entry_dt + timedelta(hours=6):
            continue
        if best is None or abs((t.start_ts - entry_dt).total_seconds()) < \
                           abs((best.start_ts - entry_dt).total_seconds()):
            best = t
    if best is None: return None
    start_p = getattr(best, "start_price", None)
    peak_p = getattr(best, "peak_price", getattr(best, "end_price", None))
    if start_p is None or peak_p is None or peak_p <= start_p:
        return None
    late = (entry_price - start_p) / (peak_p - start_p) * 100
    rem = (peak_p - entry_price) / entry_price * 100
    return max(0.0, min(100.0, late)), rem


# pair impulse_speed entries -> exits
opn = {}; rows = []
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
        opn[sym] = {"p": float(e.get("price") or e.get("entry_price") or 0),
                    "mode": e.get("mode","?"), "dt": dt}
    else:
        en = opn.pop(sym, None)
        if not en or en["mode"] != MODE: continue
        xp = float(e.get("exit_price") or e.get("price") or 0)
        if en["p"] <= 0 or xp <= 0: continue
        lr = lateness_for(sym, en["dt"], en["p"])
        if lr is None: continue
        rows.append({"sym": sym, "pnl": (xp-en["p"])/en["p"]*100,
                     "late": lr[0], "rem": lr[1]})

print("="*70)
print(f"impulse_speed lateness vs outcome ({DAYS}d, zigzag-matched)")
print("="*70)
print(f"matched entries: {len(rows)}")
if not rows:
    sys.exit(0)


def agg(rs, label):
    n = len(rs)
    if n == 0: print(f"  {label:<16} n=0"); return
    pnl = sum(r["pnl"] for r in rs)/n
    win = sum(1 for r in rs if r["pnl"] > 0)/n*100
    rem = sum(r["rem"] for r in rs)/n
    print(f"  {label:<16} n={n:<4} avg_pnl={pnl:+.2f}%  win={win:4.0f}%  "
          f"avg_remaining_upside={rem:+.1f}%")

import statistics
print(f"median lateness: {statistics.median(r['late'] for r in rows):.1f}%")
print("\nBy lateness bucket (% of up-move already done at buy):")
buckets = [("<=30%", 0, 30), ("30-50%", 30, 50), ("50-70%", 50, 70), (">70%", 70, 101)]
for lab, lo, hi in buckets:
    agg([r for r in rows if lo <= r["late"] < hi], lab)

print("\nLateness gate simulation (keep entries with lateness <= T):")
print(f"  {'T':>6}{'kept_n':>8}{'kept_avg_pnl':>14}{'kept_win%':>11}{'cut_n':>7}{'cut_avg_pnl':>13}")
for T in (40, 50, 60, 70):
    keep = [r for r in rows if r["late"] <= T]
    cut = [r for r in rows if r["late"] > T]
    ka = sum(r["pnl"] for r in keep)/len(keep) if keep else float("nan")
    kw = sum(1 for r in keep if r["pnl"]>0)/len(keep)*100 if keep else float("nan")
    ca = sum(r["pnl"] for r in cut)/len(cut) if cut else float("nan")
    print(f"  {T:>5}%{len(keep):>8}{ka:>+13.2f}%{kw:>10.0f}%{len(cut):>7}{ca:>+12.2f}%")
print("\nRead: if late buckets (>60-70%) have negative avg_pnl and low remaining")
print("upside while early buckets are positive, a lateness<=T gate removes the")
print("losing tail. Pick T where cut_avg_pnl<0 and kept_avg_pnl improves.")
