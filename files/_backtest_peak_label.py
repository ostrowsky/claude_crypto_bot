"""Should the signal model be trained on the SIZE of the move ahead, not its sign?

THE PROBLEM THIS ADDRESSES

The live label is `ret_5 > 0` — did the close rise over five bars. Measured on the
deployed model, its largest weights are NEGATIVE on momentum sequences
(seq_trend_slope -0.198, seq_trend_macd_hist_norm -0.094, seq_trend_rsi -0.076).
Trained on a five-bar coin flip it learned mean reversion, which is a real
property of five-bar returns and the opposite of the operator's goal — naming the
coins that will grow most.

Two consequences already measured this week:

* On 2026-08-20 the model scored an entire rising market near zero and the gate
  admitted 0 of 4486 candidates while XRP ran +19%, ORDI +18.8%, ENA +17.7%.
* A percentile floor built on its ranking catches big movers at or BELOW its own
  admission rate (top-20% admits 21.0% and catches 14% of >=3% movers on bull
  days) — the ranking puts the biggest movers last.

THE CHANGE UNDER TEST

Exactly one thing moves. The horizon stays at five bars; the population, the
features, the estimator and the split all stay. Only the question changes:

    OLD   y = 1 if close(t+5) > close(t)              "did it end up"
    NEW   y = 1 if max(high[t+1..t+5]) / close(t) - 1 >= T%   "was there a move"

Peak, not close, because the target is the day's largest MOVE — a run that is
given back was still a run the bot should have caught. The threshold T is chosen
from the measured distribution rather than assumed, which is what stage 1 below
reports.

WHAT WOULD MAKE THIS A FAILURE

If the new label produces a model that ranks big movers no better than the old
one, the label was not the problem and this is a negative result to record.
Stage 2 answers exactly that, on a time split, with the old label as the control.
"""
from __future__ import annotations

import argparse
import collections
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import _backtest_trend_start_detector as TD  # noqa: E402
import ml_signal_model as M  # noqa: E402

_BARS15: dict = {}


def bars_for(sym: str, tf: str):
    """Kline series matching the row's timeframe."""
    if tf == "15m":
        if sym not in _BARS15:
            _BARS15[sym] = TD.bars_15m(sym)
        return _BARS15[sym]
    return TD.load_bars(sym, "1h")


def peak_ahead(sym: str, tf: str, when: datetime, horizon: int):
    """(entry_close, peak_pct) over the next `horizon` bars, or None.

    The entry price is the close of the signal's own bar and the peak is taken
    from the bars strictly AFTER it, so the label never contains the bar the
    features describe.
    """
    bars = bars_for(sym, tf)
    if len(bars) < 30:
        return None
    idx = None
    for i, b in enumerate(bars):
        if b[0] == when:
            idx = i
            break
    if idx is None or idx + horizon >= len(bars):
        return None
    entry = bars[idx][4]
    if entry <= 0:
        return None
    fut = bars[idx + 1: idx + 1 + horizon]
    if len(fut) < horizon:
        return None
    return entry, (max(b[2] for b in fut) / entry - 1.0) * 100.0


def build(rows, horizon):
    """Attach a measured forward peak to every row that can be resolved."""
    out = []
    miss = collections.Counter()
    for r in rows:
        sym, tf = r.get("sym"), str(r.get("tf") or "1h")
        ts = r.get("ts_signal")
        try:
            d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            miss["bad_ts"] += 1
            continue
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        got = peak_ahead(sym, tf, d, horizon)
        if got is None:
            miss["no_bars"] += 1
            continue
        out.append((r, got[1]))
    return out, miss


def q(v, p):
    v = sorted(v)
    return v[int(p * (len(v) - 1))] if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5,
                    help="bars ahead; 5 keeps the live label's horizon so only "
                         "the question changes")
    ap.add_argument("--thresholds", default="1,2,3,5",
                    help="candidate T values, in percent")
    args = ap.parse_args()

    rows = M.load_training_rows(M.ROOT / "critic_dataset.jsonl")
    print("labelled rows in critic_dataset: %d" % len(rows))
    paired, miss = build(rows, args.horizon)
    print("resolvable against klines: %d   (dropped: %s)"
          % (len(paired), dict(miss)))
    if len(paired) < 2000:
        print("too few resolvable rows to judge")
        return

    peaks = [p for _, p in paired]
    olds = [M._safe_float((r.get("labels") or {}).get("ret_5")) for r, _ in paired]

    print()
    print("=" * 84)
    print("STAGE 1 — the distribution the threshold has to be chosen from")
    print("=" * 84)
    print("forward PEAK over %d bars:  p25 %.2f%%  median %.2f%%  p75 %.2f%%  "
          "p90 %.2f%%" % (args.horizon, q(peaks, .25), q(peaks, .5),
                          q(peaks, .75), q(peaks, .9)))
    print("live label ret_5 (close):   p25 %.2f%%  median %.2f%%  p75 %.2f%%"
          % (q(olds, .25), q(olds, .5), q(olds, .75)))
    print()
    old_base = sum(1 for x in olds if x > 0) / len(olds)
    print("%-14s%12s%14s" % ("threshold", "base rate", "n positive"))
    print("-" * 42)
    print("%-14s%11.1f%%%14d" % ("OLD ret_5>0", 100 * old_base,
                                 sum(1 for x in olds if x > 0)))
    for t in [float(x) for x in args.thresholds.split(",")]:
        n = sum(1 for x in peaks if x >= t)
        print("%-14s%11.1f%%%14d" % ("peak>=%.0f%%" % t, 100 * n / len(peaks), n))

    print()
    print("A base rate near the old one keeps the classifier in the same regime;")
    print("far below it and the model sees too few positives to fit 60 features.")

    # ---- stage 2: does the new label rank big movers better? ----------------
    print()
    print("=" * 84)
    print("STAGE 2 — does a model trained on it RANK the big movers better?")
    print("=" * 84)
    print("time split, both labels trained on identical rows and features;")
    print("scored by where the actual big movers land in each model's ranking")
    print()

    paired.sort(key=lambda rp: rp[0]["_dt"])
    cut = int(len(paired) * 0.7)
    tr, te = paired[:cut], paired[cut:]
    names = M.safe_feature_names()

    def mat(part):
        X = np.zeros((len(part), len(names)), dtype=float)
        for i, (r, _) in enumerate(part):
            fmap = M.build_feature_dict(r)
            X[i] = np.array([M._safe_float(fmap.get(n)) for n in names], dtype=float)
        return X

    Xtr, Xte = mat(tr), mat(te)
    sc = M.StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    te_peaks = np.asarray([p for _, p in te], dtype=float)

    print("train %d / test %d   (%s .. %s)"
          % (len(tr), len(te), tr[0][0]["_dt"].date(), te[-1][0]["_dt"].date()))
    print()
    print("%-20s%10s%12s%12s%12s" % (
        "label trained on", "AUC", "top10% avg", "top10% >3%", "top10% >5%"))
    print("-" * 70)

    labels = [("OLD ret_5>0",
               np.asarray([1.0 if M._safe_float((r.get("labels") or {}).get("ret_5")) > 0
                           else 0.0 for r, _ in tr]))]
    for t in [float(x) for x in args.thresholds.split(",")]:
        labels.append(("NEW peak>=%.0f%%" % t,
                       np.asarray([1.0 if p >= t else 0.0 for _, p in tr])))

    for name, ytr in labels:
        if len(np.unique(ytr)) < 2:
            print("%-20s  degenerate label, skipped" % name)
            continue
        m = M.LogisticModel(Xtr.shape[1]).fit(Xtr, ytr)
        s = np.asarray(m.predict_proba(Xte), dtype=float)
        # graded against the SAME truth for every row: the real forward peak
        order = np.argsort(-s)
        k = max(1, len(order) // 10)
        top = te_peaks[order[:k]]
        y_true = (te_peaks >= 3.0).astype(float)
        auc = M.roc_auc_score_np(y_true, s)
        print("%-20s%10s%11.2f%%%11.0f%%%11.0f%%" % (
            name, ("%.4f" % auc) if auc is not None else "n/a",
            float(top.mean()),
            100.0 * float((top > 3).mean()), 100.0 * float((top > 5).mean())))

    base3 = 100.0 * float((te_peaks > 3).mean())
    base5 = 100.0 * float((te_peaks > 5).mean())
    print("-" * 70)
    print("%-20s%10s%11.2f%%%11.0f%%%11.0f%%" % (
        "NO MODEL (base)", "0.5000", float(te_peaks.mean()), base3, base5))

    # ---- stage 3: does it hold at other cuts, or only at 70/30? -------------
    print()
    print("=" * 84)
    print("STAGE 3 — stability across time cuts")
    print("=" * 84)
    print("One split is one observation. An inversion that only exists at 70/30")
    print("is a property of that boundary, not of the label.")
    print()
    print("%-8s%9s%14s%14s%14s%14s" % (
        "cut", "test n", "OLD auc", "NEW auc", "OLD top10%", "NEW top10%"))
    print("-" * 74)
    for frac in (0.50, 0.60, 0.70, 0.80):
        c = int(len(paired) * frac)
        a, b = paired[:c], paired[c:]
        if len(a) < 3000 or len(b) < 1500:
            continue
        Xa, Xb = mat(a), mat(b)
        s2 = M.StandardScaler().fit(Xa)
        Xa, Xb = s2.transform(Xa), s2.transform(Xb)
        pk = np.asarray([p for _, p in b], dtype=float)
        truth = (pk >= 3.0).astype(float)
        res = {}
        for nm, yv in (("OLD", np.asarray(
                          [1.0 if M._safe_float((r.get("labels") or {}).get("ret_5")) > 0
                           else 0.0 for r, _ in a])),
                       ("NEW", np.asarray([1.0 if p >= 2.0 else 0.0 for _, p in a]))):
            if len(np.unique(yv)) < 2:
                res[nm] = (float("nan"), float("nan"))
                continue
            mm = M.LogisticModel(Xa.shape[1]).fit(Xa, yv)
            sc2 = np.asarray(mm.predict_proba(Xb), dtype=float)
            o = np.argsort(-sc2)
            k2 = max(1, len(o) // 10)
            au = M.roc_auc_score_np(truth, sc2)
            res[nm] = (au if au is not None else float("nan"),
                       float(pk[o[:k2]].mean()))
        print("%-8s%9d%14.4f%14.4f%13.2f%%%13.2f%%" % (
            "%.0f/%.0f" % (frac * 100, (1 - frac) * 100), len(b),
            res["OLD"][0], res["NEW"][0], res["OLD"][1], res["NEW"][1]))
    print("-" * 74)
    print("NEW is trained on peak>=2%% at every cut; OLD is the live label.")

    print()
    print("READ THIS")
    print("  Every row is graded against the SAME truth -- its real forward peak --")
    print("  whatever label the model was trained on, so the comparison is about")
    print("  the label and not about scoring each model on its own exam.")
    print("  AUC here is against 'peak >= 3%%', the operator's kind of outcome.")
    print("  'top10%%' is what the model's most confident decile actually did: the")
    print("  number that decides, because that decile is what a gate admits.")


if __name__ == "__main__":
    main()
