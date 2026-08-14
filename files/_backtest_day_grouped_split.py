"""Maximum-period evidence for the day-grouped training split (TH-04).

Trains `top_gainer_model` both ways on the full dataset and publishes the
holdout numbers side by side. The question is not "which model is better" — it
is "how much of the reported AUC was borrowed from the validation day".

Expected direction: AUC should FALL. The old split cuts by row index, so the
boundary lands inside a UTC day and part of that day trains the model while the
rest validates it; the tier labels are per-day ranks, so knowing part of a day
tells you about the rest. A drop here is the metric becoming honest, not the
model becoming worse — and a report that does not say so will be read as a
regression and reverted.

If AUC RISES, that is a surprise and nothing should be flipped until it is
understood.

Read-only: trains in memory, writes no model artifact.

    pyembed\\python.exe files\\_backtest_day_grouped_split.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

import train_top_gainer as T  # noqa: E402
from day_split import split_indices_by_day  # noqa: E402

VAL_RATIO = 0.2
EMBARGO_DAYS = 1
DAY_MS = 86_400_000
TIERS = ("top5", "top10", "top20", "top50")


def _evaluate(X, labels, train_idx, val_idx, tag: str) -> dict:
    out = {}
    for tier in TIERS:
        y_tr = labels[tier][train_idx]
        y_va = labels[tier][val_idx]
        if len(set(y_tr.tolist())) < 2 or len(set(y_va.tolist())) < 2:
            out[tier] = None                      # a degenerate side is not a score
            continue
        _, metrics = T.train_gradient_boosting(X[train_idx], y_tr,
                                               X[val_idx], y_va, tier)
        out[tier] = metrics
    out["_n_train"] = len(train_idx)
    out["_n_val"] = len(val_idx)
    out["_tag"] = tag
    return out


def main() -> int:
    X, labels = T.load_dataset(T.DATASET_FILE, min_samples=200)
    if len(X) == 0:
        print("empty dataset")
        return 1

    order = np.argsort(labels["ts"])
    X = X[order]
    for key in ("top5", "top10", "top20", "top50", "return", "ts"):
        labels[key] = labels[key][order]

    ts = labels["ts"]
    days = sorted({int(t // DAY_MS) for t in ts})
    print("=" * 78)
    print("TH-04 · day-grouped split vs row-index split · maximum available period")
    print("=" * 78)
    print(f"rows {len(X)} · UTC days {len(days)} · val_ratio {VAL_RATIO} · "
          f"embargo {EMBARGO_DAYS}d")

    cut = int(len(X) * (1 - VAL_RATIO))
    old_train = list(range(cut))
    old_val = list(range(cut, len(X)))
    straddling = ({int(ts[i] // DAY_MS) for i in old_train} &
                  {int(ts[i] // DAY_MS) for i in old_val})
    print(f"row-index cut at row {cut}: "
          f"{'day ' + str(straddling) + ' STRADDLES the boundary' if straddling else 'no straddle'}")

    new_train, new_val = split_indices_by_day(ts, train_frac=1 - VAL_RATIO,
                                              embargo_days=EMBARGO_DAYS)
    print(f"day-grouped: train {len(new_train)} · holdout {len(new_val)} · "
          f"embargoed {len(X) - len(new_train) - len(new_val)}")
    print()

    old = _evaluate(X, labels, old_train, old_val, "row-index (current)")
    new = _evaluate(X, labels, new_train, new_val, "day-grouped (flagged)")

    print(f"  {'tier':<8}{'AUC row-index':>15}{'AUC day-grouped':>18}{'delta':>10}")
    honest_drop = 0
    for tier in TIERS:
        a, b = old.get(tier), new.get(tier)
        if not a or not b:
            print(f"  {tier:<8}{'n/a':>15}{'n/a':>18}{'':>10}  degenerate side")
            continue
        auc_a, auc_b = a.get("auc", 0.0), b.get("auc", 0.0)
        delta = auc_b - auc_a
        honest_drop += 1 if delta < 0 else 0
        print(f"  {tier:<8}{auc_a:>15.4f}{auc_b:>18.4f}{delta:>+10.4f}")

    print()
    print(f"  holdout rows: row-index {old['_n_val']} · day-grouped {new['_n_val']}")
    print()
    deltas = [abs(new[t]["auc"] - old[t]["auc"])
              for t in TIERS if old.get(t) and new.get(t)]
    largest = max(deltas) if deltas else 0.0
    aucs = [new[t]["auc"] for t in TIERS if new.get(t)]
    saturated = bool(aucs) and min(aucs) > 0.95

    if largest < 0.005 and saturated:
        # The important case, and the one a careless report would call "no
        # regression, ship it". Neither model is honest: AUC ~0.99 comes from
        # tg_return_since_open encoding the label (TH-03). A day-boundary leak
        # is undetectable through a bigger leak, because a model that can read
        # the answer has no need to peek at neighbouring rows.
        print("VERDICT: NO MEASURABLE DIFFERENCE, and that is the finding.")
        print(f"  largest AUC delta {largest:.4f}; all tiers sit above 0.95 because")
        print("  tg_return_since_open encodes the label (TH-03). The smaller leak")
        print("  cannot be measured through the bigger one.")
        print("  The split fix is still correct — a UTC day does straddle the old")
        print("  boundary — but flipping the flag now buys nothing observable.")
        print("  Recommendation: flip it together with the label replacement, so")
        print("  one measurable change is attributable instead of two invisible ones.")
    elif largest < 0.005:
        print("VERDICT: no measurable difference. The split fix is correct but")
        print("inert on this dataset; flipping is safe and uninformative.")
    elif honest_drop >= 3:
        print("VERDICT: AUC falls on most tiers, as predicted. The old number was")
        print("partly borrowed from rows of the validation day. The lower figure is")
        print("the honest one — this is not a regression.")
    elif honest_drop == 0:
        print("VERDICT: AUC rose. That contradicts the leak hypothesis — do not")
        print("flip the flag until it is understood.")
    else:
        print("VERDICT: mixed. Inspect per tier before flipping anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
