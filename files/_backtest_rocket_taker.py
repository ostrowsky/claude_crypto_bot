"""R-10: does aggressive buying (taker-buy share) tell a rocket apart at its first move?

Taker-buy base volume per 15m bar (.runtime/backtests/taker_15m, _fetch_taker_15m.py)
joined to the rocket events (_rocket_dataset.py) at the alert bar, using only
bars up to and including it:

  tk_bar      taker share of the crossing bar
  tk_1h       last 4 bars
  tk_session  since the UTC open
  tk_pre4h    the 16 bars BEFORE the session's first +2.5% close (accumulation)
  tk_rel      tk_session minus the coin's own 7-day taker share (it varies by coin)

Judged like R-1..R-9: AUC on train (< 2026-03-01) and test, then whether adding
them lifts the CatBoost ceiling and the rules. Spec: docs/specs/features/rocket-segment-spec.md
"""
import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SPLIT = "2026-03-01"
TK = FILES.parent / ".runtime" / "backtests" / "taker_15m"
R = [json.loads(l) for l in io.open(FILES.parent / ".runtime/backtests/rocket_events.jsonl", encoding="utf-8")]
BASE_F = ["hrs", "ret_t", "bar_ret", "rvol", "up_share", "range7", "atr_pct", "e20_slope", "log_qv", "compress", "excess",
          "hrs_since_first", "pull_hi", "above_e20_since", "min_since_first", "rsi", "vol_x"]
TK_F = ["tk_bar", "tk_1h", "tk_session", "tk_pre4h", "tk_rel"]

series = {}


def load(sym):
    if sym not in series:
        fp = TK / (sym + ".json")
        if not fp.exists():
            series[sym] = None
            return None
        d = json.loads(fp.read_text(encoding="utf-8"))
        ts = np.array(sorted(int(k) for k in d))
        v = np.array([d[str(t)][0] for t in ts])
        tb = np.array([d[str(t)][1] for t in ts])
        series[sym] = (ts, v, tb, {int(t): i for i, t in enumerate(ts)})
    return series[sym]


def share(v, tb, a, b):
    s = v[a:b].sum()
    return tb[a:b].sum() / s if s > 0 else np.nan


first_close = {}
for r in R:
    if r["trig"] == 0.025:
        first_close[(r["sym"], r["day"])] = r["t"]
n_ok = 0
for r in R:
    s = load(r["sym"])
    for f in TK_F:
        r[f] = None
    if s is None:
        continue
    ts, v, tb, idx = s
    bar_open = datetime.fromisoformat(r["t"] + "+00:00") - timedelta(minutes=15)
    i = idx.get(int(bar_open.timestamp() * 1000))
    day0 = int(datetime.fromisoformat(r["day"] + "T00:00+00:00").timestamp() * 1000)
    ds = idx.get(day0)
    f0 = datetime.fromisoformat(first_close[(r["sym"], r["day"])] + "+00:00") - timedelta(minutes=15)
    i0 = idx.get(int(f0.timestamp() * 1000))
    if i is None or ds is None or i0 is None or i < 672 or i0 < 16:
        continue
    r["tk_bar"] = share(v, tb, i, i + 1)
    r["tk_1h"] = share(v, tb, i - 3, i + 1)
    r["tk_session"] = share(v, tb, ds, i + 1)
    r["tk_pre4h"] = share(v, tb, i0 - 16, i0)
    r["tk_rel"] = r["tk_session"] - share(v, tb, ds - 672, ds)
    n_ok += 1
print("events with taker features: %d of %d" % (n_ok, len(R)))


def auc(y, s):
    m = np.isfinite(s)
    y, s = y[m], s[m]
    rk = np.argsort(np.argsort(s)) + 1
    n1 = y.sum()
    n0 = len(y) - n1
    return (rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def col(rows, f):
    return np.array([np.nan if r.get(f) is None else float(r[f]) for r in rows])


from catboost import CatBoostClassifier  # noqa: E402
for trig in (0.025, 0.05, 0.075):
    S = [r for r in R if r["trig"] == trig and r.get("tk_session") is not None]
    y = np.array([r["rocket"] for r in S])
    tr = np.array([r["day"] < SPLIT for r in S])
    print("\n=== trigger +%.1f%%: n=%d, base rocket train %.1f%% / test %.1f%%" % (100 * trig, len(S), 100 * y[tr].mean(), 100 * y[~tr].mean()))
    for f in TK_F:
        x = col(S, f)
        qs = np.nanpercentile(x, [20, 40, 60, 80])
        qi = np.digitize(x, qs)
        rates = " ".join("%4.1f" % (100 * y[qi == q].mean()) for q in range(5))
        print("  %-10s AUC train %.3f  test %.3f   rocket %% by quintile: %s" % (f, auc(y[tr], x[tr]), auc(y[~tr], x[~tr]), rates))
    a = [r for r, t in zip(S, tr) if t]
    b = [r for r, t in zip(S, tr) if not t]
    yb = y[~tr]
    nd = len({r["day"] for r in b})
    for name, fs in (("without taker", BASE_F), ("with taker", BASE_F + TK_F)):
        Xa = np.column_stack([col(a, f) for f in fs])
        Xb = np.column_stack([col(b, f) for f in fs])
        m = CatBoostClassifier(iterations=400, depth=4, learning_rate=0.05, verbose=0, random_seed=1).fit(Xa, y[tr])
        p = m.predict_proba(Xb)[:, 1]
        out = []
        for q in (0.9, 0.95):
            sel = p >= np.quantile(p, q)
            out.append("top %d%%: %.2f/day rocket %.1f%%" % (round(100 * (1 - q)), sel.sum() / nd, 100 * yb[sel].mean()))
        print("  model %-13s test AUC %.3f | %s" % (name, auc(yb, p), " | ".join(out)))
