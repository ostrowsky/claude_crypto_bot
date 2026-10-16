# ML promotion at matched admission — SHADOW (2026-10-09)

Status: built, **SHADOW** (`ML_PROMOTION_MATCHED_ADMISSION_ENABLED = False`). Readout
`ML-ADMISSION-MATCH` (window from 2026-10-10, due 2026-10-30) decides whether it goes live.
Script: `files/_backtest_ml_promotion_matched.py` · result: `.runtime/ml_promotion_matched.json`.
Code: `files/ml_promotion_gate.py`, `files/ml_signal_model.py` (main), `files/monitor.py`
(`_ml_floor_admission`), `files/readouts.py` (`readout_ml_admission_match`).
Tests: `files/test_ml_admission_match.py`. Truth-harness rules: TH-01, TH-03, TH-07, TH-10, TH-13.

## Why

`ml_promotion_gate.py` (2026-09-29) replaces the live ML signal model each night unless
the new one is clearly worse on AUC, recall or precision **at the live floor** (0.15).
From 10-02 to 10-08 it kept the 10-01 model seven nights in a row, each time for "clearly
worse on precision at the live floor". So the nightly retrain did not reach the bot for a
week. That is the §0 failure mode: the learning loop runs, but nothing it learns is used.

The log shows the same pattern every night: incumbent = CatBoost, candidate = **logistic**
(it wins the trainer's own validation), **AUC equal** (10-08: 0.654 vs 0.655, interval
[−0.025, +0.026]). But the logistic's scores sit higher, so the fixed floor admits **~84%**
of rows instead of **~66%**. Precision then falls mechanically (0.298 → 0.259, base 0.239).
A fixed floor compares score scales, not models.

## Offline check (fixed before the run)

Rows: the trainer's own test split, rebuilt the same way: 891 rows, 2026-08-26..10-08,
peak-label base rate 0.240, 122 rows on winner-days (immutable later-EOD top-20).
Neither model trained on them. Paired bootstrap, 500 resamples, seed 29.

| comparison | admitted | precision | recall | winner rows admitted |
|---|---|---|---|---|
| incumbent, live floor 0.15 | 0.668 | 0.296 | 0.822 | 0.893 |
| A: candidate, live floor 0.15 | 0.847 | 0.258 | 0.911 | 0.951 |
| A diff, 95% | | [−0.054, −0.020] | [+0.038, +0.146] | [+0.009, +0.106] |
| B: candidate, matched floor 0.182 | 0.668 | 0.291 | 0.808 | 0.877 |
| B diff, 95% | | [−0.023, +0.012] | [−0.066, +0.042] | [−0.068, +0.033] |
| B′: floor 0.185 chosen on 1st half, judged on 2nd (n=446): incumbent | 0.778 | 0.303 | 0.868 | 0.931 |
| B′: candidate | 0.758 | 0.293 | 0.818 | 0.879 |
| B′ diff, 95% | | [−0.033, +0.012] | [−0.130, +0.018] | [−0.128, +0.018] |

Reading:
- **At an equal admitted share the candidate is neither better nor worse.** The freeze is a
  floor artifact, not a model problem. It also cost nothing measurable this week: the
  models are equivalent.
- A matched floor picked on earlier rows held the admitted share on later rows within 2 pp
  (0.758 vs 0.778). This is one split of 446 rows. The point estimates of recall and
  winner reach are lower (−5 pp), with intervals reaching −13 pp. Hence shadow, not live.
- Option A (promote the logistic at 0.15) would admit 18 pp more, reach more winner-day
  rows (+5.8 pp) and lose 3.8 pp of precision. That is a loosening of the gate, the
  same kind of operator trade-off as `mode_range_quality` OFF. It is not part of this change.

## Maximum-period walk-forward (2026-10-09)

Script: `files/_backtest_ml_admission_walkforward.py`. Results:
`.runtime/ml_admission_walkforward.json` and `..._0.55_0.65_0.75_0.85.json` (plus `.log`).
Setup:
- all 5 939 ml_dataset rows (2026-03-24..10-08, base 0.161);
- at each cut the production trainer fits on the rows before it, and the gate decides
  against the model in use;
- the model in use is applied to the next block, rows no model saw.

Two policies run side by side: fixed (today's gate) and matched (this change).

| run | policy | promotions | admitted (block range) | precision | recall | winner-days reached |
|---|---|---|---|---|---|---|
| cuts 0.35..0.85 (6 blocks) | fixed | 5 | 0.555 (0.11..0.75) | 0.279 | 0.765 | **27/42** |
| | matched | 5 | 0.288 (0.11..0.42) | 0.356 | 0.506 | **10/42** |
| cuts 0.55..0.85 (4 blocks) | fixed | 3 | 0.676 (0.56..0.75) | 0.281 | 0.867 | **27/41** |
| | matched | 4 | 0.690 (0.55..0.77) | 0.275 | 0.868 | **27/41** |

Transfer, i.e. a new model at its matched floor against the model in use on the forward
block. Admitted-share difference per cut:
- long run: −0.002, +0.015, +0.017, −0.045, +0.069;
- short run: +0.012, −0.030, +0.035.

The mechanism holds the share as designed.

**What the long run shows is the main risk of this change.** Matching does not choose an
operating point; it freezes the incumbent's. In the long run the anchor was the first
model, trained on one week of rows, which admitted 11%. The chain kept 16–42% while the
fixed floor's share grew with the base rate (0.17 → 0.24), and it lost 17 of 42
winner-days. With a sensible anchor (the short run, first model trained on ~3 300 rows)
both policies are equal on the goal, and matching promotes one model more (the 10-08
case).

The live anchor is today's CatBoost at ~66% (0.62–0.69 on the logged nights), inside the
fixed policy's 0.56–0.75. So the readout has an extra **ANOMALY**: any night whose
incumbent share falls outside [0.50, 0.85]. A deliberate move of the operating point
remains an operator decision.

## Design

Every night, in addition to the existing comparison:
1. **Matched floor.** The candidate's floor is the score at which it admits the same share
   of tonight's held-out rows as the incumbent admits at its effective floor, computed
   separately for bull and other days. With the flag off, the effective floor is the config
   floor. With the flag on, it is the incumbent's own `admission_floor`, if it carries one.
   The candidate payload always carries `admission_floor` (plus `holdout_to`,
   `incumbent_share`).
2. **Matched decision.** Same `decide()` rule (AUC / recall / precision, 95% upper bound
   < 0 = keep the incumbent), with the candidate judged at its matched floor. It is logged
   as `matched` in `.runtime/ml_promotion_log.jsonl`.
3. **Out-of-sample transfer.** Rows newer than yesterday's candidate's `holdout_to` are rows
   its floor was not fitted on. For each such row the log stores
   `[ts, sym, y, admitted_by_incumbent, admitted_by_yesterday's_candidate_at_its_floor]`.
   Since 10-09 the candidate is written to `ml_signal_model.candidate.json` every night,
   promoted or not, so "yesterday's candidate" always exists.
4. **Flag ON** (`ML_PROMOTION_MATCHED_ADMISSION_ENABLED = True`, then restart): the gate
   decides by the matched comparison (the fixed-floor result is kept as
   `fixed_floor_decision`). The bot's ML gate reads the loaded model's `admission_floor`
   (`monitor._ml_floor_admission`, clamped to `ML_FLOOR_CALIBRATED_MIN/MAX`), ahead of
   `ML_FLOOR_CALIBRATION_ENABLED`. A model without one (today's CatBoost incumbent) keeps
   the config floor. The two halves are switched by ONE flag on purpose: promoting at
   matched admission while the bot kept the fixed floor would widen the gate.
5. Any exception in the matched part is logged as `matched_error` and leaves the nightly
   decision unchanged.

Rollback: `ML_PROMOTION_MATCHED_ADMISSION_ENABLED = False` + restart.

## Readout ML-ADMISSION-MATCH (pre-registered)

The criteria were revised once, on 10-09, before the window opened: the winner-row
criterion became a guard and the due date moved from 10-23 to 10-30. Reason: about
20 fresh rows a night, of which ~10% are on winner-days, so a "lower 95% ≥ −5 pp" target
on winner rows could never be met.

Rows: the pooled `transfer` rows from 2026-10-10, deduplicated by (ts, sym). The bootstrap
is over rows. Verdict at or after 2026-10-30, needing ≥ 14 nights with fresh rows and
≥ 350 rows:
- **KEEP** (switch it on): the 95% interval of the admitted-share difference (candidate at
  its stored floor − incumbent at the live floor) lies inside [−0.07, +0.07], precision
  difference lower 95% ≥ −0.03, and winner-day rows admitted not clearly worse (upper 95%
  ≥ 0).
- **ROLLBACK_SUGGESTED** (keep the fixed floor): the share interval lies entirely outside
  ±0.07, or precision upper 95% < −0.03, or winner-day rows clearly worse.
- otherwise **INCONCLUSIVE**. **ANOMALY**: any night with `matched_error`.

## Limits

- `ml_proba` also enters the entry score linearly (`ML_GENERAL_SCORE_WEIGHT`). A model
  with a shifted score scale moves `candidate_score` even at a matched floor. This was
  true of every past family switch too. It is not addressed here.
- The offline evidence is one test split and a single half-split. The question the shadow
  answers is narrow: does the matched floor hold the admitted share on new rows? It is not
  whether the new model is better; this week it is not.
- Matching keeps the gate's operating point (the share it admits) as the data drifts,
  instead of a score number. A deliberate change of that share stays an operator decision.
