"""Can funding tell the START of a trend from its MIDDLE, where price could not?

Price and volume cannot: five independent attacks (forward label, start label at
four window widths, two RSI constraints, and 15m resolution at two window
scalings) all landed a median 40-52% into the move, against ~50% for a random
alert placed inside a trend. See trend-start-detector-spec.md.

Funding measures something price does not -- crowd positioning. The hypothesis
this script tests, stated so it can fail:

    A trend's opening hours run on neutral or negative funding (nobody is
    positioned yet); its middle runs on elevated funding (the crowd has piled
    in). If so, funding is the missing start-vs-middle discriminator.

THE TEST IS DELIBERATELY NARROW. It does not ask "does funding predict trends"
-- that would be answered by the base rate of trends and would drown the
question. It restricts the population to bars ALREADY INSIDE a +20% ZigZag
trend and asks a single binary question there:

    is this bar in the first `--start-hours` of the trend, or later in it?

That is the discrimination every price-based attempt failed at, isolated from
everything else. An AUC near 0.50 on this task is a clean negative and closes
the funding hypothesis; meaningfully above 0.50 means the signal exists and is
worth building on.

Funding posts every 8h, so each hourly bar takes the most recent funding value
STRICTLY BEFORE it -- never the one that settles later in the same interval.

Reported alongside: the same comparison against random non-trend bars, so a
positive result can be checked for being merely "funding is different during
big moves" rather than "funding is different at their start".
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import sys
from bisect import bisect_left
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HISTORY = ROOT / "history"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _backtest_continuation_signal as CS      # noqa: E402
import _backtest_trend_start_detector as TD     # noqa: E402
import _diag_uptrend_population as UP           # noqa: E402

_FUND: dict = {}


def funding(sym):
    """(timestamps, rates) for a symbol, ascending. Empty when not backfilled."""
    if sym in _FUND:
        return _FUND[sym]
    ts, rate = [], []
    fp = HISTORY / ("%s_funding.csv" % sym)
    if fp.exists():
        with io.open(fp, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    ts.append(datetime.fromisoformat(r["ts"]))
                    rate.append(float(r["funding_rate"]))
                except (KeyError, ValueError, TypeError):
                    continue
    _FUND[sym] = (ts, rate)
    return _FUND[sym]


def feats_at(sym, when):
    """Funding features visible at `when`, using only settlements before it.

    Returns None when fewer than 6 settlements precede the bar, so early rows
    cannot fake a 'no positioning yet' reading out of missing history.
    """
    ts, rate = funding(sym)
    if not ts:
        return None
    j = bisect_left(ts, when) - 1
    if j < 6:
        return None
    cur = rate[j]
    prev = rate[j - 1]
    win = rate[max(0, j - 5):j + 1]
    mean6 = sum(win) / len(win)
    span = max(win) - min(win)
    return {
        "funding": cur * 10000.0,                 # basis points, readable
        "funding_d1": (cur - prev) * 10000.0,
        "funding_mean6": mean6 * 10000.0,
        "funding_vs_mean6": (cur - mean6) * 10000.0,
        "funding_range6": span * 10000.0,
        "funding_sign_flips6": float(sum(
            1 for a, b in zip(win, win[1:]) if (a > 0) != (b > 0))),
    }


FEATS = ["funding", "funding_d1", "funding_mean6", "funding_vs_mean6",
         "funding_range6", "funding_sign_flips6"]


def q(v, p):
    v = sorted(v)
    return v[int(p * (len(v) - 1))] if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=float, default=20.0)
    ap.add_argument("--give-back", type=float, default=2.0)
    ap.add_argument("--start-hours", type=int, default=6,
                    help="hours after the trend low that count as its START")
    args = ap.parse_args()

    syms = [s for s in UP.watchlist()
            if len(TD.load_bars(s, "1h")) >= 400 and funding(s)[0]]
    print("=" * 90)
    print("FUNDING as a START-vs-MIDDLE discriminator")
    print("population: bars INSIDE +%.0f%% trends only -- the question price could "
          "not answer" % args.run)
    print("symbols with both klines and funding: %d" % len(syms))
    print("=" * 90)

    start_rows, mid_rows, out_rows = [], [], []
    n_trends = 0
    for s in syms:
        bars = TD.load_bars(s, "1h")
        idx = {b[0]: i for i, b in enumerate(bars)}
        inside = set()
        for t in TD.trends_tf(s, args.run, args.give_back, "1h"):
            st = UP.attr(t, "start_ts", "start", "low_ts")
            en = UP.attr(t, "end_ts", "end", "high_ts")
            if st is None or en is None:
                continue
            a, b = idx.get(st), idx.get(en)
            if a is None or b is None or b <= a:
                continue
            n_trends += 1
            for i in range(a, b + 1):
                inside.add(i)
                f = feats_at(s, bars[i][0])
                if f is None:
                    continue
                f["_day"] = bars[i][0].strftime("%Y-%m-%d")
                (start_rows if i <= a + args.start_hours else mid_rows).append(f)
        rng = random.Random(hash(s) & 0xffff)
        pool = [i for i in range(200, len(bars)) if i not in inside]
        for i in rng.sample(pool, min(60, len(pool))):
            f = feats_at(s, bars[i][0])
            if f:
                f["_day"] = bars[i][0].strftime("%Y-%m-%d")
                out_rows.append(f)

    print()
    print("trends %d   start-bars %d   middle-bars %d   non-trend bars %d"
          % (n_trends, len(start_rows), len(mid_rows), len(out_rows)))
    if len(start_rows) < 50 or len(mid_rows) < 50:
        print("too few rows to judge")
        return

    print()
    print("%-22s%12s%12s%12s" % ("feature (bp)", "START", "MIDDLE", "NON-TREND"))
    print("-" * 60)
    for k in FEATS:
        print("%-22s%12.3f%12.3f%12.3f" % (
            k, q([r[k] for r in start_rows], .5),
            q([r[k] for r in mid_rows], .5),
            q([r[k] for r in out_rows], .5) if out_rows else float("nan")))

    # The decisive number, and the split it needs.
    #
    # A RANDOM split scores 0.846 here and means nothing: funding posts every 8h,
    # so every bar in an 8h window of one symbol carries the SAME value, and
    # adjacent hours of one trend land on both sides of the cut. The model then
    # memorises (symbol, funding value) -> label. Splitting by TIME, with whole
    # days kept on one side, is the only version of this number worth reading
    # (TH-03).
    rows = ([dict(r, _y=1) for r in start_rows] + [dict(r, _y=0) for r in mid_rows])
    rows.sort(key=lambda r: r["_day"])
    days = sorted(set(r["_day"] for r in rows))
    cut_day = days[int(len(days) * 0.7)]
    tr = [r for r in rows if r["_day"] < cut_day]
    te = [r for r in rows if r["_day"] >= cut_day]
    print()
    print("time split at %s: train %d / test %d  (%d days)"
          % (cut_day, len(tr), len(te), len(days)))
    if len(te) < 100 or len(set(r["_y"] for r in te)) < 2:
        print("holdout too small or single-class")
        return
    try:
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
                               verbose=0, random_seed=0, allow_writing_files=False)
        m.fit([[r[k] for k in FEATS] for r in tr], [r["_y"] for r in tr])
        p = list(m.predict_proba([[r[k] for k in FEATS] for r in te])[:, 1])
        auc = CS.auc([r["_y"] for r in te], p)
        nulls = []
        for sd in range(5):
            y = [r["_y"] for r in tr]
            random.Random(100 + sd).shuffle(y)
            mm = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
                                    verbose=0, random_seed=sd, allow_writing_files=False)
            mm.fit([[r[k] for k in FEATS] for r in tr], y)
            nulls.append(CS.auc([r["_y"] for r in te],
                                list(mm.predict_proba([[r[k] for k in FEATS]
                                                       for r in te])[:, 1])))
        nm = sum(nulls) / len(nulls)
        nsd = (sum((v - nm) ** 2 for v in nulls) / max(len(nulls) - 1, 1)) ** 0.5
        print()
        print("START vs MIDDLE, funding features only")
        print("   base rate (share of in-trend bars that are START) %.4f"
              % (len(start_rows) / (len(start_rows) + len(mid_rows))))
        print("   AUC                 %.4f" % auc)
        print("   shuffled-label null %.4f +- %.4f   -> z = %.2f"
              % (nm, nsd, (auc - nm) / nsd if nsd else float("nan")))
        print()
        print("   Read 0.50 as: funding does not know where in a trend it is,")
        print("   and the funding hypothesis is closed. Above it, the signal is")
        print("   real and the next step is adding these to the detector.")
    except ImportError:
        print("catboost unavailable")


if __name__ == "__main__":
    main()
