# ML gate follow-ups: goal label as traded, and the training source (2026-10-12)

- **Slug:** `ml-label-followups-1012`
- **Status:** research. Nothing ships from this spec.
- **Parent:** [`goal-aligned-ml-label-1012`](goal-aligned-ml-label-1012-spec.md)
- **Truth harness:** TH-01, TH-03, TH-04, TH-06, TH-08, TH-10, TH-11, TH-12
- **Script:** `files/_backtest_ml_label_followups.py` (uses the parent's helpers).
  **Tests:** `files/test_ml_label_followups_1012.py`

Shared with the parent:
- the population and the timing;
- the walk-forward by month with a 1-day embargo;
- the estimator (live `CatBoostModel`);
- matched admission per test month (primary 60%, secondary 40%);
- the winner-day set and the reachable-early-day goal metric;
- the calibrated live trail with `trail_k` = 2.00.

## Test 1 — the GOAL label, with trades counted the way the bot takes them

**Status: EXPLORATORY.** It reuses the data the parent already graded, so it
**cannot turn the parent's "refuted" into "supported"** (TH-10). The only
question it answers is whether a forward SHADOW of the GOAL model is worth
building. A shadow would be judged on data nobody has seen.

**Why the trade test is redefined.** The parent counted every admitted candidate
as a trade. The bot does not trade that way: it holds one position per coin and
waits `COOLDOWN_BARS` (19) bars after an exit. Consecutive candidates of one coin
are therefore counted many times over, while the bot would take only one of them.

**Primary trade estimand (fixed here).** For each arm separately, per coin, in
time order:
- an admitted row opens a trade if no trade of that arm is open on that coin;
- a trade stays open from its entry to its exit, plus 19 bars of its own tf;
- only rows with a complete live-trail trade count.

The statistic is the mean pnl of these trades, GOAL minus control. It uses a
95% day-cluster bootstrap with 2000 resamples. 95% is right because this test
has one arm.

**Also reported.**
- The parent's row-level delta, now with a 95% interval.
- The decomposition into rows only GOAL admits (added) and rows only the control
  admits (removed): their counts and mean pnl.

**Outcome.**

| result | condition |
|---|---|
| `shadow-worthy` | position-deduplicated trade delta lower bound ≥ −0.10 pp |
| `not worth a shadow` | that delta's upper bound < 0 |
| `undetermined` | anything else |

## Test 2 — what the live model learns from

**Observation behind it.** The live model trains on the signal rows of
`ml_dataset.jsonl` (`load_training_rows`, `signal_type != none`).

**Correction from the run:** the spec as first written said "about 7.9k rows, 85%
of them from March–April". That was an estimate taken from all rows. The run
counted **5 482 signal rows, 55% of them from March–April** (1 967 + 1 022). The gate scores the bot's candidates, which
`critic_dataset.jsonl` holds with the same features: 57.6k rows, every month.
On ICP 2026-10-11 13:00, a model with the live label trained on the candidate
population admitted the row. The live model scored it 0.023.

**Arms.** Both use the live label `peak5_2.0` (computed from the 15m store with
the parent's timing), the same 60 features and the same estimator.

| arm | training rows |
|---|---|
| `SRC_MLDS` (control, status quo) | `ml_dataset.jsonl` signal rows that resolve against the 15m store |
| `SRC_CRITIC` | the parent's critic candidates (exactly the parent's `C_peak5`) |

Training for both uses rows whose decision is more than 1 day before the test
month. Both are graded on the same critic test rows. A fold is skipped for both
arms if `SRC_MLDS` has fewer than 500 training rows.

**Decision rule (one comparison, so 95% intervals).** At the primary point:

- **supported** iff all four hold:
  - goal delta ≥ +0.5 pp;
  - its lower bound > 0;
  - the position-deduplicated trade delta lower bound ≥ −0.10 pp;
  - goal delta ≥ 0 in at least half of the test months.
- **refuted** iff goal delta ≤ 0, or the trade lower bound < −0.10 pp.
- **inconclusive** otherwise.

The secondary point, the row-level trade delta and the AUCs are context only.

**Limits.** The live pipeline also:
- picks a model family (logistic / MLP / CatBoost) on a 70/15/15 split;
- applies the promotion gate.

Both arms fix CatBoost (the live incumbent's family) and use every prior row, so
this tests the training POPULATION, not the full nightly pipeline.

## Results (run 2026-10-12)

**Reproduction.** The parent's numbers reproduce exactly: control 26.9%, GOAL
33.9%, row-level trade +0.033 pp. Population and timing are as in the parent.

**Data.**
- 45 684 out-of-sample rows over 6 months; 454 winner-days.
- `ml_dataset` training rows per fold: 2 970 → 5 291.
- Critic training rows per fold: 11 758 → 52 431.

### Test 1 (exploratory) — outcome: **shadow-worthy**

| admission | goal Δ pp [95%] | trades as the bot takes them: Δ pp [95%] | every admitted row: Δ pp [95%] |
|---|---|---|---|
| 60% | +7.05 [+4.19, +9.91], 6/6 months | **+0.045 [−0.076, +0.154]** (−0.032% n=7 255 vs −0.077% n=6 777) | +0.033 [−0.162, +0.197] |
| 40% | +10.35 [+6.61, +14.10], 6/6 months | +0.105 [−0.096, +0.288] | +0.104 [−0.180, +0.359] |

**Changed rows at 60%.** Rows only GOAL admits: 10 870, mean +0.276%. Rows only
the control admits: 10 774, mean +0.195%.

**Why the interval narrowed.** Once overlapping candidates of one coin are
collapsed into the single trade the bot would take, the lower bound clears the
−0.10 bar. This is exploratory and uses the parent's data, so the parent's
verdict stands. It justifies a forward shadow and nothing more.

**All arms lose per trade.** Means are −0.03..−0.09% as the bot takes them,
which is the known baseline.

### Test 2 (pre-registered) — verdict: **refuted**

| admission | goal Δ pp [95%] | months ≥ 0 | trades as taken: Δ pp [95%] |
|---|---|---|---|
| 60% | **−0.88 [−2.64, +0.66]** | 50% | −0.005 [−0.049, +0.037] |
| 40% | +0.88 [−1.10, +2.86] | 83% | −0.007 [−0.070, +0.059] |

AUC against the live label: 0.759 (candidates) vs 0.743 (`ml_dataset`).

**Reading.** Training the live label on the 10× larger candidate population
ranks a little better on its own label and admits no more winner-days early. The
ICP row was an anecdote: its month percentile is 27% under `SRC_MLDS` and 42%
under `SRC_CRITIC`. The training population is not the lever. The label is.
