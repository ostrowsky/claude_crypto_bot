"""Which ml_zone gate policy should the bot run? Walk-forward over the max window.

WHY THIS EXISTS

On 2026-08-20 the gate admitted ZERO of 4486 candidates while the market rose
all day. It blocked XRP (+19% after the block), ORDI (+18.8%), ENA (+17.7%);
51% of the 83 blocked coins rose more than 3%. The model was not degenerate --
on its own training population it still produces a median 0.4053 and 75% of rows
clear the 0.22 floor. Live it produced a median 0.0569 and nothing cleared.

The mechanism: the model's largest weights are NEGATIVE on momentum-sequence
features (seq_trend_slope -0.198, seq_trend_macd_hist_norm -0.094,
seq_trend_rsi -0.076). It has learned "already rising => less likely to rise
again". In a broad rally that describes every coin at once, so the whole market
is scored near zero and a FIXED floor becomes a blackout.

Measured against its own target that day it was simply wrong: it put ~6% on
these candidates rising, and 35 of 37 rose (median +1.79%). But its ORDERING
survived -- corr(proba, ret_5) = +0.270, top half +2.06% vs bottom half +1.31%
(n=37, weak). That split -- level broken, order usable -- is what motivates a
percentile floor: it consumes only the ranking and is immune to the level
drifting under retrains and regime changes.

WHAT IS COMPARED

    NO GATE           admit everything (the ceiling on coverage)
    CURRENT           0.22 on bull days, 0.28 otherwise -- what runs today
    FIXED 0.18/0.15/0.12/0.10
    PERCENTILE top 10/20/30/40%   threshold from a TRAILING window of the last
                                  `--window` scored candidates, so it is causal
                                  and never sees the fold's own future

HOW IT IS SCORED

Walk-forward: at each cut the model is retrained on everything before it and
applied to what follows, mirroring the nightly retrain. Per policy the report
gives admit rate, the return of what was admitted, and -- the number that
matters under the operator's goal of catching the biggest movers -- RECALL OF
BIG WINNERS: of all candidates that went on to gain >= 3% and >= 5%, what share
the gate let through. A gate that admits little but misses the winners is the
failure this whole exercise is about (CLAUDE.md section 7: a filter blocking
>80% of eventual winners is broken).

Everything is also broken out by regime, because the blackout is regime-specific
and an average over both would hide exactly the thing being fixed.
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import ml_signal_model as M  # noqa: E402


def pct(v, p):
    v = sorted(v)
    return v[int(p * (len(v) - 1))] if v else float("nan")


def fit_fold(train_rows):
    """Same estimator, features and scaler the live trainer uses."""
    bundle = M.build_dataset(train_rows, positive_ret_threshold=0.0)
    X = np.vstack([bundle.X_train, bundle.X_val])
    y = np.concatenate([bundle.y_train, bundle.y_val])
    scaler = M.StandardScaler().fit(X)
    model = M.LogisticModel(X.shape[1]).fit(scaler.transform(X), y)
    return scaler, model


def build_payloads(train_rows):
    """A live-shaped payload plus a global-only twin.

    Live scoring routes each candidate to a per-segment model when one exists
    for its (signal_type, bull/nonbull) pair, and only falls back to the global
    model otherwise. Scoring the backtest with the global model alone was the
    first version of this script and it reported the CURRENT gate admitting
    99.9% -- against 0% observed live on 2026-08-20. The gap is entirely the
    routing, so the comparison has to carry it.
    """
    scaler, model = fit_fold(train_rows)
    names = M.safe_feature_names()
    glob = {
        "model_name": "logistic",
        "feature_names": names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "threshold": 0.5,
        "positive_ret_threshold": 0.0,
        "model": model.to_dict(),
    }
    seg_reports = M.train_segment_models(train_rows, positive_ret_threshold=0.0)
    kept = {}
    for key, rep in (seg_reports or {}).items():
        pay = rep.get("model_payload") or rep.get("payload")
        if not pay:
            continue
        # same keep rule as build_live_model_payload: only segments that beat
        # the baseline on their own validation survive into production
        d = rep.get("delta") or rep.get("improvement_delta") or {}
        if float(d.get("ret5_avg_delta", 0.0)) > 0.0:
            kept[key] = pay
    live = dict(glob)
    if kept:
        live["segment_model_payloads"] = kept
    return live, glob, sorted(kept.keys())


def score_payload(rows, payload):
    return [M.predict_proba_from_payload(payload, r) for r in rows]


def admits(probs, rows, policy, window):
    """Admission mask for one policy, evaluated causally in time order."""
    kind, arg = policy
    out = []
    hist = deque(maxlen=window)
    for p, rec in zip(probs, rows):
        if kind == "none":
            out.append(True)
        elif kind == "current":
            floor = 0.22 if rec.get("is_bull_day") else 0.28
            out.append(p >= floor)
        elif kind == "fixed":
            out.append(p >= arg)
        elif kind == "pctile":
            # Threshold from what came BEFORE this candidate, never from the
            # fold's own future. Before the window fills, admit -- the live
            # system has the same warm-up and hiding it would flatter the policy.
            if len(hist) < 50:
                out.append(True)
            else:
                out.append(p >= float(np.quantile(list(hist), 1.0 - arg)))
        hist.append(p)
    return out


def evaluate(rows, probs, mask):
    rets = [M._safe_float((r.get("labels") or {}).get("ret_5")) for r in rows]
    adm = [r for r, m in zip(rets, mask) if m]
    big3 = [i for i, r in enumerate(rets) if r >= 3.0]
    big5 = [i for i, r in enumerate(rets) if r >= 5.0]
    return {
        "n": len(rets),
        "admitted": len(adm),
        "rate": len(adm) / len(rets) if rets else 0.0,
        "mean": sum(adm) / len(adm) if adm else float("nan"),
        "median": pct(adm, .5),
        "win": sum(1 for x in adm if x > 0) / len(adm) if adm else float("nan"),
        "rec3": sum(1 for i in big3 if mask[i]) / len(big3) if big3 else float("nan"),
        "rec5": sum(1 for i in big5 if mask[i]) / len(big5) if big5 else float("nan"),
        "n3": len(big3),
        "n5": len(big5),
    }


POLICIES = [
    ("NO GATE", ("none", None)),
    ("CURRENT .22/.28", ("current", None)),
    ("FIXED 0.18", ("fixed", 0.18)),
    ("FIXED 0.15", ("fixed", 0.15)),
    ("FIXED 0.12", ("fixed", 0.12)),
    ("FIXED 0.10", ("fixed", 0.10)),
    ("PCTILE top40%", ("pctile", 0.40)),
    ("PCTILE top30%", ("pctile", 0.30)),
    ("PCTILE top20%", ("pctile", 0.20)),
    ("PCTILE top10%", ("pctile", 0.10)),
]


def report(title, per_policy):
    print()
    print("=" * 104)
    print(title)
    print("=" * 104)
    print("%-17s%9s%8s%10s%10s%8s%11s%11s" % (
        "policy", "admitted", "rate", "mean r5", "med r5", "win%",
        "recall>=3%", "recall>=5%"))
    print("-" * 104)
    for name, m in per_policy:
        print("%-17s%9d%7.1f%%%9.3f%%%9.3f%%%7.0f%%%10.0f%%%11.0f%%" % (
            name, m["admitted"], 100 * m["rate"], m["mean"], m["median"],
            100 * m["win"], 100 * m["rec3"], 100 * m["rec5"]))
    if per_policy:
        m = per_policy[0][1]
        print("-" * 104)
        print("population: %d candidates, of which %d gained >=3%% and %d gained >=5%%"
              % (m["n"], m["n3"], m["n5"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--window", type=int, default=500,
                    help="trailing candidates the percentile threshold is read from")
    ap.add_argument("--min-train", type=int, default=3000)
    args = ap.parse_args()

    rows = M.load_training_rows(M.ROOT / "critic_dataset.jsonl")
    print("labelled signal rows: %d  (%s .. %s)"
          % (len(rows), rows[0]["_dt"].date(), rows[-1]["_dt"].date()))

    # Walk forward: train on the past, gate the future, repeat. This mirrors the
    # nightly retrain; scoring rows the model was fitted on would make every
    # ranking-based policy look better than it can be live (TH-03).
    n = len(rows)
    cuts = [int(n * f) for f in np.linspace(0.5, 0.85, args.folds)]
    all_rows, all_probs, all_glob = [], [], []
    for k, cut in enumerate(cuts, 1):
        end = cuts[k] if k < len(cuts) else n
        tr, te = rows[:cut], rows[cut:end]
        if len(tr) < args.min_train or len(te) < 200:
            continue
        live_pay, glob_pay, kept = build_payloads(tr)
        p_live = score_payload(te, live_pay)
        p_glob = score_payload(te, glob_pay)
        all_rows.extend(te)
        all_probs.extend(p_live)
        all_glob.extend(p_glob)
        print("  fold %d: train %d -> test %d  (%s .. %s)  median proba: "
              "routed %.4f | global %.4f   segments kept: %s"
              % (k, len(tr), len(te), te[0]["_dt"].date(), te[-1]["_dt"].date(),
                 pct(p_live, .5), pct(p_glob, .5), ",".join(kept) or "none"))
    if not all_rows:
        print("no usable folds")
        return

    res = [(nm, evaluate(all_rows, all_probs,
                         admits(all_probs, all_rows, pol, args.window)))
           for nm, pol in POLICIES]
    report("ALL REGIMES, out-of-sample walk-forward", res)

    for flag, label in ((True, "BULL DAYS ONLY"), (False, "NON-BULL DAYS ONLY")):
        idx = [i for i, r in enumerate(all_rows) if bool(r.get("is_bull_day")) == flag]
        if len(idx) < 200:
            continue
        sub = [all_rows[i] for i in idx]
        subp = [all_probs[i] for i in idx]
        res = [(nm, evaluate(sub, subp, admits(subp, sub, pol, args.window)))
               for nm, pol in POLICIES]
        report(label + " -- the blackout is regime-specific, so an average hides it",
               res)

    print()
    print("READ THIS")
    print("  The operator's goal is naming the coins that will grow most, so the")
    print("  column that decides is recall of the >=3% and >=5% movers. A gate")
    print("  with a good mean return and poor recall is discarding the winners")
    print("  and keeping the quiet survivors.")
    print("  NO GATE is the ceiling on recall, not a proposal: it admits")
    print("  everything and its mean return is what the gate has to beat.")


if __name__ == "__main__":
    main()
