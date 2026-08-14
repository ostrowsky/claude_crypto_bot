# Day-grouped training split (TH-04)

- **Slug:** `day-grouped-training-split`
- **Status:** spec → implementation
- **Created:** 2026-08-14
- **Parent:** [`immutable-label-store`](immutable-label-store-spec.md) (ships the splitter)
- **Truth-harness invariants:** TH-03, TH-04, TH-06/TH-07 (flagged behaviour change)
- **Flag:** `TRAIN_DAY_GROUPED_SPLIT_ENABLED`, **default `False`** — current
  behaviour is preserved until the operator flips it
- **Rollback:** set the flag to `False`; the next nightly retrain restores the
  previous split

## Problem

`train_top_gainer.train_and_save` sorts rows by timestamp and then cuts by row
index:

```python
split_idx = int(len(X) * (1 - val_ratio))
X_train, X_val = X[:split_idx], X[split_idx:]
```

The sort is chronological, so this looks like a walk-forward split. It is not.
The cut lands **inside** a UTC day, so some rows of that day train the model and
others validate it. Rows from one day share market beta, and the tier labels
(`top5/10/20/50`) are per-day *ranks* — a rank is a property of the whole day,
so knowing part of a day tells you about the rest of it. The reported AUC is
therefore inflated by an amount nobody has measured.

This is the TH-04 blocking finding, and it is why `evaluation_scope` in the
emitted metrics already reads `time_sorted_row_holdout_same_snapshot_label`.

## Change

Use `day_split.split_indices_by_day` — the splitter shipped with the label
store — so the boundary falls between whole UTC days, with an optional embargo
that drops the boundary days from training entirely.

```
TRAIN_DAY_GROUPED_SPLIT_ENABLED = False   # default: unchanged behaviour
TRAIN_SPLIT_EMBARGO_DAYS       = 1        # days withheld at the boundary
```

`evaluation_scope` becomes `day_grouped_holdout_same_snapshot_label` when the
flag is on — still naming the *other* open defect, because the labels remain
same-snapshot (TH-03) until the immutable store replaces them. Fixing the split
does not fix the label, and the scope string must not imply it did.

## Why the default is `False`

`top_gainer_model` feeds `ranker_top_gainer_prob` into the candidate ranker's
hard veto, so retraining it on different rows changes live gating indirectly.
This project's own invariant is that a behaviour change ships behind a flag
whose default is the current behaviour, with a stated rollback. A correctness
argument is not an exemption — the previous version of this loop shipped changes
on conviction and the 8% trail widen cost 54.9% in five days.

So: measure both, publish both, recommend, and let the operator flip.

## Expected direction

Reported AUC should **fall**. The old number was earned partly by seeing rows
from the validation day during training; removing that removes the borrowed
signal. A drop here is the metric becoming honest, not the model becoming worse,
and the report must say so — otherwise the next reader will treat it as a
regression and revert the fix.

If AUC *rises*, that is a surprise worth investigating before flipping anything.

## Verification

`test_day_grouped_split.py`:

1. `split_indices_by_day` returns index arrays whose days do not intersect;
2. the boundary is a real day boundary — no day appears on both sides for any
   fraction;
3. the embargo drops boundary days from training without moving them to holdout;
4. with the flag off, the trainer's split is byte-identical to today's;
5. with the flag on, `evaluation_scope` changes and still names the label defect;
6. a dataset spanning fewer than two days is refused rather than split.

**Maximum-period evidence** (`_backtest_day_grouped_split.py`): train both ways
on the full available dataset and publish AUC, recall@0.3 and holdout size per
tier, side by side.

**Shadow/canary: не применимо.** The flag default leaves production untouched;
there is no second live behaviour to stage. When the operator flips it, the
nightly retrain is the change, and the rollback is the flag.

## Maximum-period evidence — and it refuted the expectation

`_backtest_day_grouped_split.py`, full dataset, 120 088 rows over 310 UTC days,
`val_ratio = 0.2`, embargo 1 day. The row-index cut does straddle a day (day
20654 appears on both sides), so the defect is real.

```
tier      AUC row-index   AUC day-grouped     delta
top5             0.9927            0.9927   +0.0000
top10            0.9945            0.9945   +0.0000
top20            0.9938            0.9939   +0.0001
top50            0.9876            0.9875   -0.0001
train rows        96 070            95 730
holdout rows      24 018            23 434
```

**No measurable difference — and that is the finding, not a disappointment.**

AUC sits at ~0.99 on every tier because `tg_return_since_open` is among the top
features while the label is "did the day's return land in the top-N". The label
is encoded in the inputs (TH-03). A model that can *read* the answer has no need
to peek at neighbouring rows of the same day, so the day-boundary leak
contributes nothing detectable. **The smaller leak cannot be measured through
the bigger one.**

A less careful process would have reported "split fixed, AUC unchanged, no
regression, shipped" — three true statements adding up to a false impression.

**Recommendation: leave the flag off** and flip it together with the label
replacement, so one measurable change is attributable instead of two invisible
ones. Re-run this backtest at that point; only then does it have the power to
say anything.

## Findings from the architecture review

**The first comparison was invalid and looked fine.** Splitting 80/20 by *day
count* produced a 49/51 split by *rows* — early days carry a handful of rows,
recent ones carry ~105 symbols × 4 snapshots. The two models were trained on
58k and 96k rows, so the AUC difference measured training-set size, not the
split. `split_indices_by_day` now takes `by="rows"` and places the boundary at
the day nearest the target row mass; the corrected run compares 95 730 against
96 070.

**A test fixture never exercised its own case.** `timestamps(days=1, per_day=50)`
used a fixed hour step, so "one day" silently spanned three and the single-day
refusal was never tested. Rows are now spaced to fit inside the day.

**The verdict logic called a null result "mixed".** Tiny deltas of both signs
were reported as ambiguous when the honest reading is "no measurable difference,
because a larger leak saturates the metric". Saturation is now detected
explicitly and named.

## Not in scope

The same-snapshot label (TH-03) — that is the label store's job, and this change
deliberately keeps `evaluation_scope` naming it so the remaining defect stays
visible.
