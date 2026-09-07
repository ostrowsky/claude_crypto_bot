"""Where do the gate floors land if the model is retrained on the peak label?

WHY THIS HAS TO BE COMPUTED BEFORE ANYTHING IS SWITCHED

Changing the training label changes the model's output DISTRIBUTION, and every
gate floor in config.py was tuned against the old one. That exact mismatch caused
the 2026-08-20 blackout: a per-segment model shifted the level, the fixed 0.22
floor did not move with it, and the gate admitted 0 of 4486 candidates while the
market ran +19%. Switching the label without re-deriving the floors would repeat
that failure deliberately.

So this reports, for both labels on the same holdout:

    the output distribution              where the probabilities actually sit
    admit rate at each floor             how much the gate would let through
    what the admitted candidates DID     average forward peak, share above 3/5%
    recall of the big movers             of all >=3% movers, how many got in

and finds the floor on the NEW scale that reproduces the live admit rate, plus
the floor that maximises big-mover recall at a workable volume.

HONEST SCOPE

ml_zone is one gate of several. Everything measured here is what the ml floor
alone would admit; trend_quality, trend_chop, mode_range_quality and rotation all
still apply downstream, so live entry counts will be lower than the admit rates
below. This says where the ML floor belongs, not how many trades result.
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

_IDX: dict = {}


def index_for(sym: str, tf: str):
    """Timestamp -> bar index, built once per symbol.

    The first version of this join scanned the whole series for every one of
    45 075 rows and took ten minutes; the label work should not cost that on
    every rerun.
    """
    key = (sym, tf)
    if key in _IDX:
        return _IDX[key]
    bars = TD.bars_15m(sym) if tf == "15m" else TD.load_bars(sym, "1h")
    _IDX[key] = (bars, {b[0]: i for i, b in enumerate(bars)})
    return _IDX[key]


def peak_ahead(sym, tf, when, horizon):
    bars, idx = index_for(sym, tf)
    i = idx.get(when)
    if i is None or i + horizon >= len(bars):
        return None
    entry = bars[i][4]
    if entry <= 0:
        return None
    fut = bars[i + 1: i + 1 + horizon]
    if len(fut) < horizon:
        return None
    return (max(b[2] for b in fut) / entry - 1.0) * 100.0


def load_paired(horizon):
    rows = M.load_training_rows(M.ROOT / "critic_dataset.jsonl")
    out, miss = [], 0
    for r in rows:
        try:
            d = datetime.fromisoformat(str(r.get("ts_signal")).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            miss += 1
            continue
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        p = peak_ahead(r.get("sym"), str(r.get("tf") or "1h"), d, horizon)
        if p is None:
            miss += 1
            continue
        out.append((r, p))
    out.sort(key=lambda rp: rp[0]["_dt"])
    return out, miss, len(rows)


def q(v, p):
    v = np.asarray(v, dtype=float)
    return float(np.quantile(v, p)) if len(v) else float("nan")


def table(name, scores, peaks, floors, live_rate=None):
    print()
    print("=" * 92)
    print(name)
    print("=" * 92)
    print("output distribution:  p10 %.4f  p25 %.4f  median %.4f  p75 %.4f  p90 %.4f  max %.4f"
          % (q(scores, .10), q(scores, .25), q(scores, .5),
             q(scores, .75), q(scores, .90), float(np.max(scores))))
    big3 = peaks >= 3.0
    big5 = peaks >= 5.0
    print("holdout: %d rows, %d moved >=3%% (%.1f%%), %d moved >=5%% (%.1f%%)"
          % (len(peaks), int(big3.sum()), 100 * big3.mean(),
             int(big5.sum()), 100 * big5.mean()))
    print()
    print("%-10s%10s%12s%11s%11s%13s%13s" % (
        "floor", "admits", "admit rate", "avg peak", ">3%", "recall>=3%", "recall>=5%"))
    print("-" * 92)
    best = None
    for f in floors:
        m = scores >= f
        n = int(m.sum())
        if n < 20:
            continue
        r3 = float((big3 & m).sum()) / max(1, int(big3.sum()))
        r5 = float((big5 & m).sum()) / max(1, int(big5.sum()))
        rate = float(m.mean())
        mark = ""
        if live_rate is not None and best is None and rate <= live_rate:
            mark = "   <- matches the live admit rate"
            best = f
        print("%-10.3f%10d%11.1f%%%10.2f%%%10.0f%%%12.0f%%%12.0f%%%s" % (
            f, n, 100 * rate, float(peaks[m].mean()),
            100 * float((peaks[m] > 3).mean()), 100 * r3, 100 * r5, mark))
    print("-" * 92)
    print("%-10s%10d%11.1f%%%10.2f%%%10.0f%%%12.0f%%%12.0f%%" % (
        "no gate", len(peaks), 100.0, float(peaks.mean()),
        100 * float((peaks > 3).mean()), 100.0, 100.0))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--label-threshold", type=float, default=2.0,
                    help="peak%% that counts as a positive for the NEW label")
    ap.add_argument("--cut", type=float, default=0.7)
    args = ap.parse_args()

    paired, miss, total = load_paired(args.horizon)
    print("critic rows %d, resolvable %d, dropped %d" % (total, len(paired), miss))
    if len(paired) < 3000:
        print("too few rows")
        return

    cut = int(len(paired) * args.cut)
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
    peaks = np.asarray([p for _, p in te], dtype=float)
    print("train %d / test %d  (%s .. %s)"
          % (len(tr), len(te), tr[0][0]["_dt"].date(), te[-1][0]["_dt"].date()))

    y_old = np.asarray([1.0 if M._safe_float((r.get("labels") or {}).get("ret_5")) > 0
                        else 0.0 for r, _ in tr])
    y_new = np.asarray([1.0 if p >= args.label_threshold else 0.0 for _, p in tr])

    s_old = np.asarray(M.LogisticModel(Xtr.shape[1]).fit(Xtr, y_old)
                       .predict_proba(Xte), dtype=float)
    s_new = np.asarray(M.LogisticModel(Xtr.shape[1]).fit(Xtr, y_new)
                       .predict_proba(Xte), dtype=float)

    live_floor = 0.10
    live_rate = float((s_old >= live_floor).mean())

    table("OLD LABEL (ret_5 > 0) — the scale the live floors were tuned on",
          s_old, peaks,
          [0.02, 0.05, 0.10, 0.15, 0.22, 0.28, 0.35, 0.45])
    print()
    print("the live floor of %.2f admits %.1f%% of candidates on this scale"
          % (live_floor, 100 * live_rate))

    matched = table(
        "NEW LABEL (peak >= %.0f%%) — where the floor has to move to" % args.label_threshold,
        s_new, peaks,
        [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
        live_rate=live_rate)

    print()
    print("=" * 92)
    print("WHAT TO SET")
    print("=" * 92)
    if matched is not None:
        m = s_new >= matched
        b3 = peaks >= 3.0
        print("floor %.2f on the NEW scale reproduces today's admit rate (%.1f%%)"
              % (matched, 100 * live_rate))
        print("   and would catch %.0f%% of the >=3%% movers, against %.0f%% today"
              % (100 * float((b3 & m).sum()) / max(1, int(b3.sum())),
                 100 * float((b3 & (s_old >= live_floor)).sum()) / max(1, int(b3.sum()))))
    else:
        print("no floor on the new scale is as strict as the live one --")
        print("the new distribution sits lower; pick from the table by recall.")
    # ---- stability: is the floor a property of the label or of one boundary? -
    print()
    print("=" * 92)
    print("STABILITY ACROSS TIME CUTS")
    print("=" * 92)
    print("A floor that only works at 70/30 is a property of that boundary. Each")
    print("cut refits both labels on everything before it and grades what follows.")
    print()
    print("%-9s%9s%12s%12s%14s%14s" % (
        "cut", "test n", "NEW .15 adm", "NEW .15 rec", "NEW .20 rec", "OLD live rec"))
    print("-" * 72)
    for frac in (0.50, 0.60, 0.70, 0.80):
        c2 = int(len(paired) * frac)
        a, b = paired[:c2], paired[c2:]
        if len(a) < 4000 or len(b) < 1500:
            continue
        Xa, Xb = mat(a), mat(b)
        s2 = M.StandardScaler().fit(Xa)
        Xa, Xb = s2.transform(Xa), s2.transform(Xb)
        pk = np.asarray([p for _, p in b], dtype=float)
        b3 = pk >= 3.0
        if b3.sum() < 30:
            continue
        ya = np.asarray([1.0 if p >= args.label_threshold else 0.0 for _, p in a])
        yo = np.asarray([1.0 if M._safe_float((r.get("labels") or {}).get("ret_5")) > 0
                         else 0.0 for r, _ in a])
        sn = np.asarray(M.LogisticModel(Xa.shape[1]).fit(Xa, ya)
                        .predict_proba(Xb), dtype=float)
        so = np.asarray(M.LogisticModel(Xa.shape[1]).fit(Xa, yo)
                        .predict_proba(Xb), dtype=float)
        m15, m20 = sn >= 0.15, sn >= 0.20
        # the old model judged at the admit rate the new one has at 0.15, so the
        # two are compared at equal volume rather than at equal threshold
        thr_o = float(np.quantile(so, 1.0 - float(m15.mean())))
        mo = so >= thr_o
        print("%-9s%9d%11.1f%%%11.0f%%%13.0f%%%13.0f%%" % (
            "%.0f/%.0f" % (frac * 100, (1 - frac) * 100), len(b),
            100 * float(m15.mean()),
            100 * float((b3 & m15).sum()) / int(b3.sum()),
            100 * float((b3 & m20).sum()) / int(b3.sum()),
            100 * float((b3 & mo).sum()) / int(b3.sum())))
    print("-" * 72)
    print("OLD is graded at the SAME admit rate as NEW .15, not the same number,")
    print("so the comparison is about ordering and not about where a floor sits.")

    print()
    print("READ THIS")
    print("  Matching the admit rate is the SAFE swap: same volume, different")
    print("  ordering. Raising recall means admitting more, which other gates")
    print("  and MAX_OPEN then have to absorb.")
    print("  ml_zone is one gate of several -- trend_quality, trend_chop,")
    print("  mode_range_quality and rotation still apply after it, so live entry")
    print("  counts stay well below these admit rates.")


if __name__ == "__main__":
    main()
