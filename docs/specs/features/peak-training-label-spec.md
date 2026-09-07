# Training label: the size of the move ahead, not its sign

- **Slug:** `peak-training-label`
- **Status:** DEPLOYED 2026-09-07 (live restart 21:31, verified in place)
- **Truth-harness invariants:** TH-01 (base rate beside every ratio), TH-03
  (time split, four cuts), TH-06 (the bot's own candidates), TH-10, TH-11 (the
  proxy is graded against the real outcome, not against itself), TH-13
- **Flags:** `ML_PEAK_LABEL_ENABLED = True`, `ML_PEAK_LABEL_HORIZON = 5`, `ML_PEAK_LABEL_THRESHOLD_PCT = 2.0`, `ML_PEAK_LABEL_MIN_RESOLVED = 0.60`
- **Rollback:** `ML_PEAK_LABEL_ENABLED = False`, restart, retrain. The ml
  floors are DERIVED from this flag (`0.15 if enabled else 0.10`), so they
  revert with it and cannot be left behind — which is the failure mode this
  coupling exists to prevent. Previous artifacts kept as
  `ml_signal_model.pre_peak_label.json`, `bandit_entry_state.pre_peak_label.json`,
  `ml_candidate_ranker.pre_peak_label.json`.

## The defect

The live label is `ret_5 > 0`: did the close rise over five bars. The deployed
model's largest weights are **negative on momentum sequences**
(`seq_trend_slope` −0.198, `seq_trend_macd_hist_norm` −0.094,
`seq_trend_rsi` −0.076). It learned mean reversion — which is a true property of
five-bar closes, and the opposite of the operator's goal.

The distribution says why. Over 37 112 resolvable candidates:

```
forward PEAK over 5 bars   p25 0.23%   median 0.59%   p75 1.28%   p90 2.45%
live label ret_5 (close)   p25 -0.66%  median -0.10%  p75 0.47%
```

The typical candidate rises and then gives it back. A label built on the close
teaches the model to predict the giving-back, and momentum is what precedes it.

## The change under test

One thing moves. Horizon, population, features, estimator and split all stay:

```
OLD   y = 1 if close(t+5) > close(t)
NEW   y = 1 if max(high[t+1..t+5]) / close(t) - 1 >= T%
```

Peak rather than close, because the target is the day's largest **move** and a
run that is given back was still a run the bot should have caught.

The peak is taken from bars strictly **after** the signal bar, so the label never
contains the bar its features describe.

## Result: the old label is inverted, at every cut

Both models are graded against **the same truth** — each row's real forward peak
— so the comparison is about the label rather than each model being marked on its
own exam.

| cut | test n | OLD AUC | NEW AUC | OLD top-decile | NEW top-decile |
|---|---|---|---|---|---|
| 50/50 | 18 556 | 0.3261 | 0.8024 | 0.79% | 2.65% |
| 60/40 | 14 845 | 0.2994 | 0.8046 | 0.72% | 2.72% |
| 70/30 | 11 134 | 0.2758 | 0.8102 | 0.69% | 2.81% |
| 80/20 | 7 423 | 0.2672 | 0.8056 | 0.67% | 3.08% |

AUC is against "peak ≥ 3%". Base rate for the decile column is **1.06%**.

**The old label does not merely fail to rank big movers — it ranks them last.**
An AUC of 0.27–0.33 is a consistent inversion, and its most confident decile
averages 0.67–0.79% against a coin-blind 1.06%. Every floor raised on that scale
makes the admitted set *worse*: at 0.45 the old model admits 31.4% of candidates
and keeps 13% of the ≥3% movers.

This single fact explains three separate failures measured earlier this week: the
2026-08-20 blackout, the percentile floor scoring below random, and the bandit's
rejects outperforming its own entries.

The threshold barely matters — AUC 0.8102 / 0.8146 / 0.8175 / 0.8196 at
T = 2/3/1/5%. **T = 2% is chosen** for its base rate (13.9%, 5 142 positives),
enough to fit 60 features without the class being rare.

## Where the gate floors land on the new scale

Changing the label changes the output distribution, and every floor in
`config.py` was tuned against the old one. Not re-deriving them would repeat the
August blackout deliberately.

```
OLD scale   p10 0.3706  median 0.4309  p90 0.4787  max 0.5544   (a narrow band)
NEW scale   p10 0.0829  median 0.1317  p90 0.3011  max 0.9997   (spread out)
```

The live floor of 0.10 admits **100%** on the old scale — the ml gate currently
does nothing at all. On the new scale it becomes usable:

| floor | admit rate | avg peak | share >3% | recall of ≥3% movers |
|---|---|---|---|---|
| 0.10 | 73.3% | 1.28% | 10% | 96% |
| **0.15** | **41.6%** | **1.70%** | 16% | **84%** |
| 0.20 | 24.8% | 2.08% | 21% | 67% |
| 0.30 | 10.1% | 2.81% | 31% | 40% |
| 0.50 | 2.8% | 4.42% | 49% | 17% |
| none | 100% | 1.06% | 8% | 100% |

Stability of the recommended floor, with the old model graded at the **same admit
rate** rather than the same number, so only ordering is compared:

| cut | NEW 0.15 admits | NEW 0.15 recall | NEW 0.20 recall | OLD at equal volume |
|---|---|---|---|---|
| 50/50 | 46.0% | 86% | 71% | 26% |
| 60/40 | 41.7% | 83% | 66% | 19% |
| 70/30 | 41.6% | 84% | 67% | 18% |
| 80/20 | 41.5% | 83% | 66% | 17% |

At equal volume the new label recalls **four times** as many big movers.

## What switching requires — not done yet

1. `peak_5` written into `critic_dataset.jsonl` for new rows, and backfilled for
   existing ones from klines (the join used here, indexed, takes seconds).
2. `build_dataset` labelling on it instead of `ret_5 > 0`.
3. `ML_GENERAL_HARD_BLOCK_MIN` / `..._BULL_DAY_MIN` moved from 0.10 to the new
   scale — 0.15 keeps 84% of big movers at 41.6% admit.
4. `ML_CANDIDATE_RANKER` and the bandit context both consume `ml_proba`; their
   own thresholds and learned weights are calibrated to the old distribution and
   must be re-derived or reset, not left to drift.

Item 4 is the one that can go wrong quietly, and it is the reason this is written
down before anything is switched rather than after.

## Honest limits

- All of this replays **the bot's own candidates** (TH-06). It says nothing about
  candidates the gates never admitted.
- The new label and the grading criterion are both peak-based, so they share a
  definitional affinity. That is why the decisive column is the top decile's
  **actual** forward move (2.65–3.08% against a 1.06% base), which does not
  depend on how the label is defined.
- 7 963 of 45 075 rows could not be resolved against klines and were dropped;
  they are older rows outside the cached window, so the sample tilts recent.

## Deployment, 2026-09-07

### Maximum-period backtest

The evidence above IS the maximum-period backtest: every labelled candidate the
bot has produced, 45 075 rows of which 37 112 resolve against klines, spanning
2026-03-24 to 2026-09-06, evaluated at four time cuts. No sub-window was chosen.

### Shadow / canary decision

**No shadow period.** A shadow run of a training label would have to train a
second model, score it in parallel and wait for outcomes — which is exactly what
the four-cut walk-forward already did, on 37 112 rows instead of a day's worth.
Shadow adds nothing the replay has not given.

What replaces it is a **24-hour watch with the rollback trigger fixed in advance**
(below), and the fact that the ml floor is one gate of several: `trend_quality`,
`trend_chop`, `mode_range_quality` and rotation all still apply, so a wrong call
here shows up as a changed candidate mix rather than as unguarded trading.

### Acceptance and rollback trigger, set before the fact

Read after 24 hours of live decisions:

- **Keep** if `ml_zone` admits between 30% and 60% of candidates, and the coins
  it admits show a higher forward peak than those it rejects.
- **Roll back** if `ml_zone` admits under 10% (a blackout is re-forming) or over
  90% (the gate is inert again and the floor is wrong for the new scale), or if a
  watchlist coin reaching the day's Binance top-20 is rejected by `ml_zone`
  alone.

Stated now so the decision cannot be re-argued from whichever number looks better
tomorrow.

### Verified in place, not assumed

After the restart the deployed model reproduces the backtest:

| | measured in backtest | live payload |
|---|---|---|
| p10 / median / p90 | 0.0829 / 0.1317 / 0.3011 | 0.0837 / 0.1352 / 0.3167 |
| admit at 0.15 | 41.6% | 43.8% |
| avg peak admitted | 1.70% | 1.66% |
| recall of ≥3% movers | 84% | 86% |

`label_version = peak5_2.0`, and no `ML LABEL MISMATCH` line appeared in the log.

### Stale-artifact detection

The ranker consumes `ml_proba` as a feature and the bandit as context element 0;
both were fitted to the old distribution. The payload now carries
`label_version`, and `monitor._check_label_version` logs one loud line if the
deployed model answers a different question than config expects. The bandit is
independently disabled (see `_backtest_gate_overblocking.py`), so its stale state
is not in the live path today, but the archived copy and the warning mean
re-enabling it cannot happen silently.

### Known incomplete

The training-time selection criterion still scores families and thresholds by
`selected_ret5_avg` — the **close**. That is why the retrain reported negative
numbers for all three families: they were graded on the outcome the label no
longer targets. Family choice is therefore still decided by a metric pointing the
old way. It is deliberately not bundled here so the label change can be read on
its own.
