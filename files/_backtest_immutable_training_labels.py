"""`top_gainer_model` trained on leaky vs immutable labels, full dataset.

The old label is written from the same rolling-24h snapshot that produced the
features, so `tg_return_since_open` is an input AND very nearly the answer. This
trains both ways and publishes AUC, recall@0.3, base rate and holdout size per
tier.

Base rate travels with every ratio (TH-01): the two label sets do NOT share one,
so AUC alone would hide that the question changed. Neither does the universe —
the immutable labels are watchlist-scoped, so its tiers mean "top-N within the
watchlist" while the old ones mean "top-N on the exchange".

Writes nothing but stdout; the live model blob is untouched.

  pyembed\\python.exe files\\_backtest_immutable_training_labels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

import immutable_labels as IL  # noqa: E402
import train_top_gainer as TT  # noqa: E402

TIERS = (5, 10, 20, 50)
VAL_RATIO = 0.2


def _fit(X, labels, tag: str) -> dict:
    """Time-sorted holdout, same splitter for both arms so the split is not a
    second moving part (TH-04)."""
    out = {}
    split = int(len(X) * (1 - VAL_RATIO))
    for n in TIERS:
        tier = f"top{n}"
        y = np.asarray(labels[tier], dtype=float)
        y_tr, y_va = y[:split], y[split:]
        if y_tr.sum() < 10 or y_va.sum() < 5:
            out[tier] = None                  # too few positives to score
            continue
        _, m = TT.train_gradient_boosting(X[:split], y_tr, X[split:], y_va,
                                          f"{tag}:{tier}")
        out[tier] = {"auc": m["auc"], "recall_at_03": m["recall_at_03"],
                     "base_rate": float(y_va.mean()), "n_val": int(len(y_va))}
    return out


def main() -> int:
    X, labels = TT.load_dataset(TT.DATASET_FILE, min_samples=100)
    if len(X) == 0:
        print("empty dataset")
        return 1

    order = np.argsort(labels["ts"])
    X = X[order]
    for k in ("top5", "top10", "top20", "top50", "return", "ts"):
        labels[k] = labels[k][order]
    labels["symbol"] = [labels["symbol"][i] for i in order]

    days = TT._utc_days(labels["ts"])
    import config as _c
    floor = float(getattr(_c, 'TRAIN_IMMUTABLE_LABEL_MIN_PCT', 0.0))
    keep, new_labels, stats = IL.tier_labels(days, labels["symbol"],
                                             tiers=TIERS, floor=floor)
    idx = np.asarray(keep, dtype=int)

    print("=" * 78)
    print("top_gainer_model · leaky snapshot labels vs immutable later-EOD")
    print("=" * 78)
    print(f"rows in dataset      {len(X)}")
    print(f"rows the store knows {stats['n_labelled']} "
          f"({100*stats['n_labelled']/len(X):.1f}%)  "
          f"dropped {stats['dropped_unlabelled']}")
    print(f"floor +{floor:.0f}%   window {days[0]}..{days[-1]}")
    print()

    leaky = _fit(X, labels, "leaky")
    imm = _fit(X[idx], new_labels, "immutable")

    print(f"  {'tier':<7}{'AUC leaky':>11}{'base':>8}"
          f"{'AUC immut':>11}{'base':>8}{'dAUC':>9}{'n_val imm':>11}")
    for n in TIERS:
        tier = f"top{n}"
        a, b = leaky[tier], imm[tier]
        if a is None or b is None:
            print(f"  {tier:<7}{'too few positives to score':>50}")
            continue
        print(f"  {tier:<7}{a['auc']:>11.4f}{100*a['base_rate']:>7.2f}%"
              f"{b['auc']:>11.4f}{100*b['base_rate']:>7.2f}%"
              f"{b['auc']-a['auc']:>+9.4f}{b['n_val']:>11}")

    print()
    print("Reading it: a LARGE fall in AUC is the metric becoming honest, not")
    print("the model getting worse — the old number was earned by reading the")
    print("answer out of tg_return_since_open. AUC still near 0.99 on the")
    print("immutable side would mean the leak is NOT gone and something in the")
    print("new path still sees the outcome: stop and investigate, not ship.")
    print()
    print("The two arms answer different questions. Old tiers are top-N on the")
    print("exchange; immutable tiers are top-N within the watchlist with a")
    print(f"+{floor:.0f}% floor. dAUC is not a like-for-like improvement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
