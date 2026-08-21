"""Do positioning states cluster, and do the clusters differ in what happens next?

THE IDEA, AND WHY IT IS BETTER THAN HAND-CUT GROUPS

The flow classes used so far ("longs opening", "short covering") come from
thresholds a human picked: OI above 0.5%, price above 0.3%, taker above 1.1.
Nothing in the data chose those numbers, and they use two features out of the
twelve recorded. Clustering removes both limitations at once.

THE TRAP, WHICH KILLS THIS APPROACH IF IGNORED

Clustering finds structure in FEATURE space. It has no idea what happened next
and will happily return beautifully separated clusters whose forward returns are
identical. Separation is not the result; a difference in OUTCOME is. Worse, if
one tries k = 2..10 across several algorithms, some partition will show an
outcome spread by chance alone -- the same multiple-comparison trap that produced
nine refuted hypotheses in this repository.

So the decisive number here is NOT silhouette, inertia or cluster shape. It is:

    does the outcome spread between clusters exceed what RANDOM partitions
    of the same sizes produce on the same rows?

That null is computed by shuffling the outcome column and re-measuring, many
times. A clustering that cannot beat its own shuffled null is describing the
feature space and nothing else.

WHY THIS IS ALSO NOT THE ONLY TOOL

If the question is "which positioning states precede large moves", that is a
SUPERVISED question and a supervised model answers it more directly. Clustering
earns its place when interpretable regimes are wanted -- states a human can name
and reason about -- which is exactly what a trading operator needs from a screen.
Both are worth running; this file is the unsupervised half.

SAMPLE SIZE

Twelve features on 87 rows is fitting noise. The script refuses to draw a
conclusion below --min-rows and says how long the recorder must run to get
there. Twelve snapshots a day of ~87 coins is ~1000 rows a day, but rows from
one snapshot are NOT independent -- the whole market moves together -- so the
count that matters is DAYS, and the split is by day for the same reason.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STORE = HERE / "positioning_history.jsonl"

FEATS = ["move", "chg", "rngpos", "vs_ma25", "oi_1h", "oi_4h", "oi_24h",
         "px_4h", "taker", "taker_trend", "retail", "top", "funding_bp"]


def load(min_resolved=True):
    if not STORE.exists():
        return []
    out = []
    with io.open(STORE, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if min_resolved and not (r.get("resolved") and r.get("out_peak_pct") is not None):
                continue
            if any(r.get(f) is None for f in FEATS):
                continue
            out.append(r)
    return out


def matrix(rows):
    X = np.array([[float(r[f]) for f in FEATS] for r in rows], dtype=float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd


def kmeans(X, k, seed=0, iters=100):
    """Plain Lloyd's algorithm with k-means++ seeding — no sklearn dependency."""
    rng = np.random.RandomState(seed)
    n = len(X)
    centres = [X[rng.randint(n)]]
    for _ in range(k - 1):
        d = np.min(np.stack([((X - c) ** 2).sum(axis=1) for c in centres]), axis=0)
        probs = d / d.sum() if d.sum() > 0 else np.ones(n) / n
        centres.append(X[rng.choice(n, p=probs)])
    C = np.array(centres)
    lab = np.zeros(n, dtype=int)
    for _ in range(iters):
        dist = np.stack([((X - c) ** 2).sum(axis=1) for c in C], axis=1)
        new = dist.argmin(axis=1)
        if (new == lab).all():
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(axis=0)
    return lab, C


def spread(labels, y):
    """How far apart the clusters are in OUTCOME.

    Weighted mean absolute deviation of cluster means from the overall mean, so
    a partition that isolates three rows with a wild return cannot dominate the
    number by sitting in a corner.
    """
    y = np.asarray(y, dtype=float)
    gm = y.mean()
    tot, n = 0.0, len(y)
    for j in np.unique(labels):
        m = labels == j
        tot += m.sum() * abs(y[m].mean() - gm)
    return tot / n if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="2,3,4,5,6")
    ap.add_argument("--nulls", type=int, default=200)
    ap.add_argument("--min-rows", type=int, default=2000)
    ap.add_argument("--min-days", type=int, default=20)
    args = ap.parse_args()

    rows = load()
    days = sorted({r["ts"][:10] for r in rows})
    print("resolved rows with a full feature vector: %d over %d day(s)"
          % (len(rows), len(days)))

    if len(rows) < args.min_rows or len(days) < args.min_days:
        need_days = max(0, args.min_days - len(days))
        print()
        print("NOT ENOUGH DATA — refusing to cluster.")
        print("  have %d rows / %d days, want >= %d rows / >= %d days."
              % (len(rows), len(days), args.min_rows, args.min_days))
        print("  Rows from one snapshot are not independent: the whole market")
        print("  moves together, so ~87 coins in a snapshot is closer to ONE")
        print("  observation than to 87. Days are what count.")
        if need_days:
            print("  At 12 snapshots a day the recorder needs ~%d more day(s)."
                  % need_days)
        print()
        print("  Running anyway on this sample would find clusters -- it always")
        print("  does -- and they would be noise wearing the shape of a result.")
        return

    X = matrix(rows)
    y = np.array([r["out_peak_pct"] for r in rows], dtype=float)
    cut = days[int(len(days) * 0.7)]
    tr = np.array([r["ts"][:10] < cut for r in rows])
    te = ~tr
    print("time split at %s: train %d / test %d" % (cut, tr.sum(), te.sum()))

    print()
    print("=" * 96)
    print("CLUSTER OUTCOME SPREAD vs ITS OWN SHUFFLED NULL")
    print("Fit on train, applied to test. A clustering that cannot beat the null")
    print("has found shape in the features and nothing about what follows.")
    print("=" * 96)
    print("%-5s%9s%12s%14s%9s%s" % ("k", "test n", "spread", "null mean±sd", "z", "  verdict"))
    print("-" * 96)

    rng = random.Random(0)
    for k in [int(x) for x in args.k.split(",")]:
        lab_tr, C = kmeans(X[tr], k, seed=0)
        d = np.stack([((X[te] - c) ** 2).sum(axis=1) for c in C], axis=1)
        lab_te = d.argmin(axis=1)
        if len(np.unique(lab_te)) < 2:
            print("%-5d%9d   collapsed to one cluster on test" % (k, te.sum()))
            continue
        s = spread(lab_te, y[te])
        nulls = []
        yy = list(y[te])
        for _ in range(args.nulls):
            rng.shuffle(yy)
            nulls.append(spread(lab_te, np.array(yy)))
        nm, nsd = float(np.mean(nulls)), float(np.std(nulls, ddof=1))
        z = (s - nm) / nsd if nsd > 0 else float("nan")
        verdict = "  real separation" if z > 3 else ("  marginal" if z > 2 else "  indistinguishable from random")
        print("%-5d%9d%11.3f%%%9.3f±%-5.3f%9.2f%s" % (k, te.sum(), s, nm, nsd, z, verdict))

    print()
    print("READ THIS")
    print("  z is computed against shuffled OUTCOMES on the SAME clusters, so it")
    print("  asks exactly one question: given these groups, is the difference in")
    print("  what happened next bigger than chance? Silhouette and inertia are")
    print("  deliberately absent -- they measure how round the clusters are, which")
    print("  is not the question.")
    print("  Several k values are tried, so a single k at z just above 2 is a")
    print("  multiple-comparison artefact, not a finding.")


if __name__ == "__main__":
    main()
