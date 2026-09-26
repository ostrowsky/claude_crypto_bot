"""Can a rocket be told apart early, without false alarms? Hypotheses R-1..R-9 on the full history.

Input: .runtime/backtests/rocket_events.jsonl from _rocket_dataset.py (one row per
coin-day and trigger: first 15m close >= open * (1 + 2.5% / 5% / 7.5%)).
Split by TIME (TH-03): thresholds and models are fitted on days before
2026-03-01, judged on days from 2026-03-01. Spec: docs/specs/features/rocket-segment-spec.md

  1. base rates and every feature's AUC on train and on test (stable = same side)
  2. the ceiling: a CatBoost model per trigger level, precision of its top alerts
  3. the economics: alerts traded with the calibrated live trail (P-1 engine)
  4. simple threshold rules (2-3 conditions) chosen on train, judged on test
"""
import io
import itertools
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config as cfg  # noqa: E402
import indicators as I  # noqa: E402
import _backtest_trend_start_detector as TD  # noqa: E402

SPLIT = "2026-03-01"
LABELS = {"sym", "day", "t", "trig", "day_max", "day_close", "rocket", "big", "left_day", "left_24h", "dd_before", "eod", "top20"}
MODEL_F = ["hrs", "ret_t", "bar_ret", "rvol", "up_share", "range7", "atr_pct", "e20_slope", "log_qv", "compress", "excess",
           "hrs_since_first", "pull_hi", "above_e20_since", "min_since_first", "rsi", "vol_x"]
R = [json.loads(l) for l in io.open(FILES.parent / ".runtime/backtests/rocket_events.jsonl", encoding="utf-8")]


def auc(y, s):
    m = np.isfinite(s)
    y, s = y[m], s[m]
    r = np.argsort(np.argsort(s)) + 1
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def col(rows, f):
    return np.array([np.nan if r.get(f) is None else float(r[f]) for r in rows])


def X(rows, fs):
    return np.column_stack([col(rows, f) for f in fs])


# ---------------- 1 ----------------
E25 = [r for r in R if r["trig"] == 0.025]
y = np.array([r["rocket"] for r in E25])
tr = np.array([r["day"] < SPLIT for r in E25])
days = sorted({r["day"] for r in E25})
print("=== 1. events at the first +2.5%% close: %d on %d days (%s .. %s)" % (len(E25), len(days), days[0], days[-1]))
print("  rocket (>=10%% from open AND close >= 60%% of it): %.1f%% = %d, %.2f per day; train %.1f%%, test %.1f%%" % (
    100 * y.mean(), y.sum(), y.sum() / len(days), 100 * y[tr].mean(), 100 * y[~tr].mean()))
print("  %-16s %6s %6s  %s" % ("feature", "train", "test", "verdict"))
for f in [k for k in E25[0] if k not in LABELS]:
    s = col(E25, f)
    a1, a2 = auc(y[tr], s[tr]), auc(y[~tr], s[~tr])
    stable = (a1 - 0.5) * (a2 - 0.5) > 0 and min(abs(a1 - 0.5), abs(a2 - 0.5)) >= 0.05
    print("  %-16s %.3f  %.3f  %s" % (f, a1, a2, "STABLE" if stable else ("flips" if (a1 - 0.5) * (a2 - 0.5) < 0 else "weak")))

# ---------------- 2 ----------------
from catboost import CatBoostClassifier  # noqa: E402
print("\n=== 2. ceiling: CatBoost per trigger, trained < %s, judged >= %s" % (SPLIT, SPLIT))
models = {}
for trig in (0.025, 0.05, 0.075):
    S = [r for r in R if r["trig"] == trig]
    a = [r for r in S if r["day"] < SPLIT]
    b = [r for r in S if r["day"] >= SPLIT]
    ya, yb = np.array([r["rocket"] for r in a]), np.array([r["rocket"] for r in b])
    cont = np.array([r["left_day"] >= 0.05 and r["eod"] > 0 for r in b])
    left, eod = col(b, "left_day"), col(b, "eod")
    nd = len({r["day"] for r in b})
    m = CatBoostClassifier(iterations=400, depth=4, learning_rate=0.05, verbose=0, random_seed=1).fit(X(a, MODEL_F), ya)
    p = m.predict_proba(X(b, MODEL_F))[:, 1]
    models[trig] = (b, p)
    print("  trigger +%.1f%%: test n=%d, base rocket %.1f%%, base continuation (+5%% more and EOD above) %.1f%%, AUC %.3f" % (
        100 * trig, len(b), 100 * yb.mean(), 100 * cont.mean(), auc(yb, p)))
    for q in (0.8, 0.9, 0.95, 0.98):
        sel = p >= np.quantile(p, q)
        print("     top %4.1f%%: %.2f alerts/day  rocket %4.1f%% (%.1fx)  continuation %4.1f%%  median left %.1f%%  mean EOD %+.2f%%" % (
            100 * (1 - q), sel.sum() / nd, 100 * yb[sel].mean(), yb[sel].mean() / yb.mean(), 100 * cont[sel].mean(),
            100 * np.median(left[sel]), 100 * eod[sel].mean()))

# ---------------- 3 ----------------
print("\n=== 3. economics at +2.5%%: alerts traded with the calibrated live trail (P-1), test period")
b, p = models[0.025]
top = set(np.argsort(-p)[: len(b) // 10])
cache = {}


def trail(sym, t, k, fl, max_bars=96):
    if sym not in cache:
        bb = TD.bars_15m(sym)
        h = np.array([x[2] for x in bb]); lo = np.array([x[3] for x in bb]); c = np.array([x[4] for x in bb])
        cache[sym] = ({x[0]: i for i, x in enumerate(bb)}, c, I._atr(h, lo, c, cfg.ATR_PERIOD))
    idx, c, atr = cache[sym]
    ie = idx.get(datetime.fromisoformat(t + "+00:00") - timedelta(minutes=15))
    if ie is None:
        return None
    ep = c[ie]
    stop = ep - max(k * atr[ie], fl * ep)
    last = min(len(c) - 1, ie + max_bars)
    if last <= ie:
        return None
    for j in range(ie + 1, last + 1):
        if np.isfinite(atr[j]):
            stop = max(stop, c[j] - max(k * atr[j], fl * c[j]))
        if c[j] < stop:
            return (c[j] / ep - 1) * 100
    return (c[last] / ep - 1) * 100


rnd = random.Random(1)
for k, fl, name in ((2.0, 0.0, "trend trail k2"), (2.5, 0.015, "k2.5 + 1.5% floor"), (2.0, 0.08, "8% floor")):
    for gname, grp in (("all events", range(len(b))), ("model top 10%", sorted(top))):
        res = [(trail(b[i]["sym"], b[i]["t"], k, fl), b[i]["rocket"]) for i in grp]
        res = [x for x in res if x[0] is not None]
        v = [x[0] for x in res]
        bs = sorted(np.mean([rnd.choice(v) for _ in v]) for _ in range(1000))
        print("  %-18s %-14s n=%5d mean %+.2f%% [%+.2f, %+.2f]  rockets %+.1f%%  others %+.2f%%" % (
            name, gname, len(v), np.mean(v), bs[25], bs[-26], np.mean([x[0] for x in res if x[1]]),
            np.mean([x[0] for x in res if not x[1]])))

# ---------------- 4 ----------------
print("\n=== 4. simple rules (2-3 thresholds at train quantiles), chosen on train, judged on test")
DIR = {"hrs": -1, "rvol": 1, "atr_pct": 1, "range7": 1, "ret_t": 1, "bar_ret": 1, "up_share": 1, "e20_slope": 1,
       "compress": 1, "excess": 1, "log_qv": 1}
A = {f: col(E25, f) for f in DIR}
cont = np.array([r["left_day"] >= 0.05 and r["eod"] > 0 for r in E25])
ntr = len({r["day"] for r in E25 if r["day"] < SPLIT})
nte = len({r["day"] for r in E25 if r["day"] >= SPLIT})
conds = []
for f, sg in DIR.items():
    for q in (0.5, 0.7, 0.8, 0.9):
        th = np.nanquantile(A[f][tr], q if sg > 0 else 1 - q)
        conds.append((f, "%s%s%.4g" % (f, ">=" if sg > 0 else "<=", th), (A[f] >= th) if sg > 0 else (A[f] <= th)))
cands = []
for kk in (2, 3):
    for combo in itertools.combinations(conds, kk):
        if len({c[0] for c in combo}) < kk:
            continue
        m = np.logical_and.reduce([c[2] for c in combo])
        if (m & tr).sum() / ntr < 0.3:
            continue
        cands.append((y[m & tr].mean(), " & ".join(c[1] for c in combo), m))
cands.sort(key=lambda x: -x[0])
print("  %d rules with >= 0.3 alerts/day on train; the 10 best on train:" % len(cands))
for ptr, name, m in cands[:10]:
    print("   train %4.1f%% -> test %4.1f%% (%.1fx)  continuation %4.1f%%  %.2f alerts/day  | %s" % (
        100 * ptr, 100 * y[m & ~tr].mean(), y[m & ~tr].mean() / y[~tr].mean(), 100 * cont[m & ~tr].mean(),
        (m & ~tr).sum() / nte, name))
