# `top_gainer_model` on immutable training labels (TH-03)

- **Slug:** `top-gainer-immutable-training-labels`
- **Status:** spec → implementation
- **Created:** 2026-08-15
- **Parent:** [`north-star-immutable-labels`](north-star-immutable-labels-spec.md)
- **Consumes:** [`immutable-label-store`](immutable-label-store-spec.md)
- **Truth-harness invariants:** TH-01 (base rate beside every ratio), TH-03
  (label provenance), TH-04 (comparable windows), TH-07 (flagged behaviour change)
- **Flag:** `TRAIN_IMMUTABLE_LABELS_ENABLED`, **default `False`**
- **Rollback:** set the flag to `False`; the next nightly retrain restores the
  previous labels

## Problem

`train_top_gainer` reads `label_top5/10/20/50` straight out of
`top_gainer_dataset.jsonl`. Those labels are written from the same rolling-24h
leaderboard snapshot that produced the row's features, so `tg_return_since_open`
is an input *and* very nearly the answer. The model scores AUC ≈ 0.99 on every
tier and the emitted `label_timing` already confesses
`same_snapshot_current_24h_leaderboard`.

This is the last consumer of the leaky label that changes live behaviour:
`top_gainer_model` produces `ranker_top_gainer_prob`, which feeds the candidate
ranker's hard veto. It is also what keeps TH-03 red and the North Star
`provisional`.

## What the store can and cannot reproduce

Measured before designing, because the answer changed the design:

```
dataset rows                120 088     distinct symbols 478   median 103/day
rows the store can label     65 660     (54.7%)
store universe                   98 symbols, median 95/day, 199 well-covered days
```

**The existing tiers are global, not universe-relative.** `label_top20` has a
base rate of **6.31%** on ~103 rows/day. If it meant "top-20 of the day's
universe" it would be ~19%. It means top-20 *of all Binance*, intersected with
whatever the row's universe was. The store holds only watchlist symbols, so
**the global rank is not reproducible from it at all.**

That is the binding constraint, and it forces a redefinition rather than a
drop-in replacement. Two consequences must be stated rather than absorbed:

1. **45.3% of rows become unlabellable** and are dropped, not labelled 0. A
   missing label is not a negative; counting it as one teaches the model that
   every symbol outside the store failed to move.
2. **The tier semantics change** from "top-N on the exchange" to "top-N within
   the watchlist". A different question, on a smaller and easier universe.

## Change

A row's tier label becomes: **rank ≤ N by `eod_return_pct` within the store's
universe for that UTC day, AND `eod_return_pct ≥ TRAIN_IMMUTABLE_LABEL_MIN_PCT`.**

```
TRAIN_IMMUTABLE_LABELS_ENABLED = False   # default: unchanged behaviour
TRAIN_IMMUTABLE_LABEL_MIN_PCT  = 5.0     # the floor is load-bearing, see below
```

### Why the floor is not optional

A pure rank label mints exactly N winners a day whatever the market does — the
base rate is fixed by construction, so the label carries no information about
whether it was a day worth trading. This repo has already paid for that lesson
once: the entry bandit's rank-only label scored **lift 1.02×** (indistinguishable
from random) and only the +3% floor took it to 4.07×.

Measured here, over 199 well-covered days / 18 905 labelled rows:

| tier | no floor | floor +3% | floor +5% | current (leaky, global) |
|------|---------:|----------:|----------:|------------------------:|
| top5  |  5.26% |  4.02% | **2.96%** |  1.49% |
| top10 | 10.53% |  6.31% | **4.10%** |  3.15% |
| top20 | 21.05% |  8.99% | **5.08%** |  6.31% |
| top50 | 52.63% | 12.62% | **6.28%** | 15.34% |

Without a floor `top50` is a **52.6%** base rate — a coin flip wearing the name
of a gainer tier. `+5%` is chosen because it is the qualification threshold the
project already uses for a MoveEvent, not because it fit the table best; it also
happens to land the tier base rates back in the range of the ones they replace.

### The tiers lose their separation, and that is reported

Under the floor, `top20` (5.08%) and `top50` (6.28%) nearly coincide: inside a
95-symbol universe, few days have 50 coins up more than 5%, so the floor binds
long before the rank does. The four tiers are no longer four distinct questions.
This is a real loss of resolution and it belongs in the report, not in a
footnote — a reader comparing per-tier AUC across the flag needs to know the
tiers themselves changed.

## Expected direction

AUC should **fall sharply**, likely from ~0.99 into the 0.6–0.8 range. The old
number was earned by reading the answer out of the features. A large drop is the
metric becoming honest; it is not the model getting worse, and the report must
say so or the next reader will revert the fix.

If AUC stays near 0.99, the leak is not gone and something in the new path still
sees the outcome — that is a stop-and-investigate result, not a success.

## Verification

`test_train_top_gainer_labels.py`:

1. with the flag off, the label vectors are identical to today's, element for
   element;
2. with the flag on, a row whose `(symbol, day)` the store does not know is
   **dropped**, never labelled 0;
3. the floor is applied — a rank-qualifying row below the threshold is a
   negative;
4. a day whose store universe is too thin contributes no positives;
5. the emitted report carries `label_timing = "immutable_later_eod_close"`,
   the base rate per tier, and the row count actually trained on;
6. `evaluation_scope` names both the split and the label, so neither defect can
   hide behind the other being fixed.

**Maximum-period evidence** (`_backtest_immutable_training_labels.py`): train
both ways over the full dataset and publish per tier the AUC, recall@0.3, base
rate and holdout size. Base rate beside every ratio (TH-01), because the two
label sets have different base rates and AUC alone would not show it.

**Shadow/canary:** the flag default leaves production untouched, so there is no
second live behaviour to stage. When the operator flips it, the nightly retrain
is the change and the rollback is the flag. `ranker_top_gainer_prob` shifts as
soon as the retrained blob loads, so the first 24h of `bot_events.jsonl` must be
compared for veto rate before walking away.

## Maximum-period evidence

`_backtest_immutable_training_labels.py`, full dataset, 120 088 rows,
2025-10-07..2026-08-14, same time-sorted splitter on both arms so the split is
not a second moving part:

```
rows the store can label  65 660 (54.7%)   dropped 54 428
floor +5%

tier     AUC leaky    base   AUC immut    base      dAUC   n_val
top5        0.9927   1.14%      0.8493   2.54%   -0.1434   13 132
top10       0.9945   2.51%      0.8349   2.99%   -0.1596   13 132
top20       0.9938   4.79%      0.8293   3.03%   -0.1645   13 132
top50       0.9876  10.36%      0.8051   3.03%   -0.1825   13 132
```

**AUC 0.99 → 0.83, the predicted direction and magnitude.** The old number was
read out of `tg_return_since_open`; 0.83 is what the features are actually worth
when the answer is not among them. That residual is not leakage — a coin already
up 10% by 08:00 genuinely does end the day higher more often, and that is known
at snapshot time.

### The tiers collapse into one label — this blocks the flip

The holdout base rates are suspiciously equal, and they are equal for a reason:

```
pair            identical rows   positives
top5  / top10          99.56%    334 / 392
top10 / top20          99.95%    392 / 398
top20 / top50         100.00%    398 / 398
```

`top20` and `top50` are **byte-identical** on the holdout. Inside a 95-symbol
watchlist, rarely do more than ~10 coins gain over 5% in a day, so the floor
binds long before the rank does and "rank ≤ 20" and "rank ≤ 50" select the same
rows. The four-tier structure was only meaningful because the original label was
**global** — top-50 of ~400 exchange symbols is a real tier; top-50 of 95 is not.

This is not cosmetic. `top_gainer_model.py` builds a tier ladder on fixed
thresholds (0.3 / 0.3 / 0.35 / 0.4) calibrated against base rates of
1.49 / 3.15 / 6.31 / 15.34%. Four near-identical heads at ~3% base rate make the
ladder return the same answer at every rung and leave every threshold
mis-calibrated at once.

### Recommendation: implement, do not flip

The label path is correct and shipped behind the flag. Flipping it needs one
more decision first, and there are two coherent ways to take it:

1. **Collapse to a single head** — under the floor the rank contributes almost
   nothing above N=5, so the honest model is one "qualifying mover" classifier,
   and the ladder consumers change with it.
2. **Extend the label store past the watchlist** so global rank is reproducible
   and the tiers keep their original meaning. More work, preserves every
   consumer, and is the better long-run answer.

Either way that is a separate change with its own evidence. Flipping the flag
today would swap a leaky model for an honest one whose four outputs are the same
number, and the ranker would read it through thresholds meant for something else.

## Findings from the review

**A second copy of the pipeline had drifted out of sync.** `main()` duplicated
the whole train/split/emit sequence and referenced `n_train` and `scope`, neither
of which was ever defined in it — the CLI path raised `NameError` before writing
anything, and nobody noticed because `daily_learning` calls `train_and_save`. It
was also where the leaky `label_timing` string survived the relabelling. `main()`
now delegates, so one path cannot disagree with itself.

**A test asserted the wrong arithmetic.** The floor test expected four rows above
`+5%` from returns `+9,+8,+7,+6,+5`; the boundary is inclusive and the answer is
five. The code was right and the test was corrected — worth recording, because
the opposite reflex would have loosened a threshold to make a test pass.

## Not in scope

Extending the label store beyond the watchlist so the global rank becomes
reproducible. That would preserve the original tier semantics and is the better
long-run answer, but it means fetching klines for ~400 symbols over ~200 days and
is a separate change with its own evidence.
