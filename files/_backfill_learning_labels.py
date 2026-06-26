"""Backfill the two learning labels over critic_dataset history (steps 1-3),
using critic_dataset's own locked _rewrite_records (safe vs live appends):

  (B) label_fast_reversal  — realized-outcome based: a TAKE that exited within
      ~1h at a loss (bars_held <= 4(15m)/1(1h) AND trade_exit_pnl < threshold).
      Replaces the dead RM-3 ATR-threshold label (None on all takes).
  (A) label_premature_exit — exited, then the coin ran >= PREMATURE_EXIT_PCT
      within the cooldown window (the re-alert condition) = false early exit.
      Computed from history klines (records older than the kline window stay None).

Run from files/.  pyembed\python.exe files\_backfill_learning_labels.py
"""
from __future__ import annotations
import csv, io, sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "history"

import config as cfg
import critic_dataset as cd

MAXB_15M = int(getattr(cfg, "FAST_REVERSAL_MAX_BARS_15M", 4))
MAXB_1H = int(getattr(cfg, "FAST_REVERSAL_MAX_BARS_1H", 1))
PMAX = float(getattr(cfg, "FAST_REVERSAL_PNL_MAX", -0.3))
PRE_PCT = float(getattr(cfg, "PREMATURE_EXIT_PCT", 5.0))
CD_BARS = int(getattr(cfg, "COOLDOWN_BARS", 19))

# preload 15m klines (ms ts, high, close) per symbol
_K = {}
for p in HIST.glob("*_15m.csv"):
    sym = p.name[:-8]
    ts, hi, cl = [], [], []
    with io.open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                import datetime as _dt
                ts.append(int(_dt.datetime.fromisoformat(r["ts"]).timestamp()*1000))
                hi.append(float(r["high"])); cl.append(float(r["close"]))
            except Exception:
                continue
    if len(cl) > 50:
        _K[sym] = (np.array(ts), np.array(hi), np.array(cl))

BAR15 = 15*60*1000
stats = {"fr": 0, "fr1": 0, "pe": 0, "pe1": 0}


def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def premature(sym, bar_ts, bars_held, tf):
    d = _K.get(sym)
    if d is None or bar_ts is None: return None
    ts, hi, cl = d
    bar_ms = BAR15 if tf == "15m" else 4*BAR15
    exit_ts = int(bar_ts) + int(bars_held) * bar_ms
    j = np.searchsorted(ts, exit_ts)
    if j >= len(cl): return None
    exit_close = cl[j]
    if exit_close <= 0: return None
    win_bars = CD_BARS * (4 if tf == "1h" else 1)   # cooldown window in 15m bars
    fwd_hi = hi[j+1:j+1+win_bars]
    if len(fwd_hi) == 0: return None
    rise = (float(fwd_hi.max()) - exit_close) / exit_close * 100.0
    return 1 if rise >= PRE_PCT else 0


def mutate(rec):
    dec = rec.get("decision", {}) or {}
    if str(dec.get("action", "")) != "take":
        return False
    lab = rec.setdefault("labels", {})
    bh = lab.get("trade_bars_held"); pnl = _f(lab.get("trade_exit_pnl"))
    tf = str(rec.get("tf", "15m"))
    changed = False
    if bh is not None and pnl is not None:
        maxb = MAXB_15M if tf == "15m" else MAXB_1H
        fr = 1 if (int(bh) <= maxb and pnl < PMAX) else 0
        if lab.get("label_fast_reversal") != fr:
            lab["label_fast_reversal"] = fr; changed = True
        stats["fr"] += 1; stats["fr1"] += fr
        if lab.get("label_premature_exit") is None:
            pe = premature(rec.get("sym"), rec.get("bar_ts"), bh, tf)
            if pe is not None:
                lab["label_premature_exit"] = pe; changed = True
                stats["pe"] += 1; stats["pe1"] += pe
    return changed


print(f"klines loaded: {len(_K)} syms; rewriting critic_dataset (locked)...")
cd._rewrite_records(mutate)
print("=" * 56)
print("Backfill done.")
print(f"  fast_reversal labeled (resolved takes): {stats['fr']}  "
      f"==1: {stats['fr1']} ({100*stats['fr1']/max(1,stats['fr']):.0f}%)")
print(f"  premature_exit labeled (had klines):    {stats['pe']}  "
      f"==1: {stats['pe1']} ({100*stats['pe1']/max(1,stats['pe']):.0f}%)")
