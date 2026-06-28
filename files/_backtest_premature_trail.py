"""Exit-side decision: does WIDENING the trail on high proba_premature_exit
entries recover capture (NS sub-metric: realized vs potential) without a net
loss? Unlike the entry guard, holding longer costs no top-20 COVERAGE — but a
wider trail DOES bleed more on true reversals, so this is the real tradeoff.

Forward price replay per labelled TAKE (entry features -> proba_premature_exit;
klines -> forward path). For each entry simulate a trailing stop with buffer
atr_pct*k over <=max_hold bars under k_current vs k_wide. Conditioning rule:
use k_wide only when proba_premature_exit > T. Reports, per threshold T, the
aggregate realized pnl and capture vs the all-current baseline.

FINDING (2026-06, OOS holdout >=2026-05-16): the in-sample gain (d_real +0.378
@0.55) EVAPORATES out-of-sample — d_real ~0 (mostly slightly negative), d_cap
negative at every threshold. proba_premature_exit (AUC 0.596) is too weak to
drive a profitable trail-widening rule that generalises. => NOT actionable; like
the entry guard, keep dormant. (Mode-specific trail widening — EX1 impulse_speed
1.5%->8% — is a separate, validated lever and stays.)

  pyembed\python.exe files\_backtest_premature_trail.py
"""
from __future__ import annotations
import csv, io, json, sys
from pathlib import Path
from datetime import datetime
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config
config.FAST_REVERSAL_LEARNING_ENABLED = True
import offline_rl as orl
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "history"
FEATS = ["close_vs_ema20", "close_vs_ema50", "close_vs_ema200", "ema20_vs_ema50",
         "ema50_vs_ema200", "slope", "rsi", "adx", "vol_x", "macd_hist_norm",
         "atr_pct", "daily_range", "body_pct", "upper_wick_pct", "lower_wick_pct",
         "btc_vs_ema50", "btc_momentum_4h", "market_vol_24h"]
K_CUR, K_WIDE = 2.0, 4.0          # current vs wider trail multiplier
MAXH_15M, MAXH_1H = 48, 24

pm = CatBoostClassifier(); pm.load_model(str(ROOT/"files"/"premature_exit_catboost.cbm"))


def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def proba(f, tf, mode):
    try:
        row = [float(f.get(k, 0.0) or 0.0) for k in FEATS] + [str(tf), str(mode)]
        return float(pm.predict_proba([row])[0][1])
    except Exception:
        return 0.0


# klines: sym -> (ts_ms[], high[], low[], close[])
K = {}
for p in HIST.glob("*_15m.csv"):
    sym = p.name[:-8]; ts = []; hi = []; lo = []; cl = []
    with io.open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                ts.append(int(datetime.fromisoformat(r["ts"]).timestamp()*1000))
                hi.append(float(r["high"])); lo.append(float(r["low"])); cl.append(float(r["close"]))
            except Exception:
                continue
    if len(cl) > 50:
        K[sym] = (np.array(ts), np.array(hi), np.array(lo), np.array(cl))


def replay(sym, bar_ts, atr_pct, k, maxh):
    """Trailing stop from entry bar; buffer = atr_pct% * k below running peak.
    Returns (realized_ret_pct, potential_ret_pct)."""
    d = K.get(sym)
    if d is None or atr_pct is None or atr_pct <= 0:
        return None
    ts, hi, lo, cl = d
    j = int(np.searchsorted(ts, bar_ts))
    if j >= len(cl) - 2:
        return None
    entry = cl[j]
    if entry <= 0:
        return None
    buf = atr_pct/100.0 * k
    peak = entry; realized = None
    end = min(j + maxh, len(cl) - 1)
    for t in range(j+1, end+1):
        peak = max(peak, hi[t])
        stop = peak * (1 - buf)
        if lo[t] <= stop:
            realized = (stop - entry)/entry*100.0
            break
    if realized is None:
        realized = (cl[end] - entry)/entry*100.0
    potential = (float(np.max(hi[j+1:end+1])) - entry)/entry*100.0
    return realized, potential


# OOS only: the premature model trained on <2026-05-16 (70% split). Evaluate the
# trail decision on the holdout so the proba is out-of-sample (not overfit).
HOLDOUT_CUTOFF = "2026-05-16"
rows = []
for rec in orl._load_critic_dataset(max_records=25000):
    if rec.get("decision", {}).get("action") != "take":
        continue
    if str(rec.get("ts_signal", ""))[:10] < HOLDOUT_CUTOFF:
        continue
    lab = rec.get("labels", {}) or {}
    if lab.get("label_premature_exit") is None:
        continue
    f = rec.get("f", {}) or {}
    atr_pct = _f(f.get("atr_pct"))
    bar_ts = rec.get("bar_ts"); sym = rec.get("sym"); tf = rec.get("tf", "15m")
    if atr_pct is None or not bar_ts or not sym:
        continue
    maxh = MAXH_15M if tf == "15m" else MAXH_1H
    cur = replay(sym, bar_ts, atr_pct, K_CUR, maxh)
    wide = replay(sym, bar_ts, atr_pct, K_WIDE, maxh)
    if cur is None or wide is None:
        continue
    p = proba(f, tf, rec.get("signal_type", "trend"))
    is_top = (lab.get("ret_10", 0.0) or 0.0) >= 3.0
    rows.append((p, cur[0], cur[1], wide[0], wide[1], is_top))

n = len(rows)
base_real = np.mean([r[1] for r in rows]); base_cap = np.mean([r[1]/r[2] if r[2] > 0 else 0 for r in rows])
print(f"replayed takes={n}   trail k_cur={K_CUR} vs k_wide={K_WIDE}")
print(f"baseline (all current): realized={base_real:+.3f}%  capture={base_cap:.3f}")
print(f"{'thr':>6}{'n_wide':>8}{'realized%':>11}{'capture':>9}{'d_real':>9}{'d_cap':>8}{'  top20_real%':>0}")
for T in [0.40, 0.50, 0.55, 0.60, 0.70]:
    real = []; cap = []; tr = []
    for p, cr, cp, wr, wp, is_top in rows:
        use_wide = p > T
        rr = wr if use_wide else cr; pp = wp if use_wide else cp
        real.append(rr); cap.append(rr/pp if pp > 0 else 0)
        if is_top: tr.append(rr)
    nw = sum(1 for r in rows if r[0] > T)
    mr = np.mean(real); mc = np.mean(cap); mt = np.mean(tr) if tr else 0
    print(f"{T:>6.2f}{nw:>8}{mr:>+11.3f}{mc:>9.3f}{mr-base_real:>+9.3f}{mc-base_cap:>+8.3f}"
          f"   top20_real={mt:+.2f}%")
print("\nWiden trail only when proba_premature_exit>T. d_real/d_cap vs all-current")
print("baseline. POSITIVE d_cap with d_real>=0 => recovers capture cost-free ->")
print("worth wiring (no coverage tradeoff). NEGATIVE d_real => wider trail bleeds")
print("more on reversals than it recovers -> not worth it.")
