# Goal-aligned training label for the ML gate (2026-10-12)

- **Slug:** `goal-aligned-ml-label-1012`
- **Status:** research. Nothing ships from this spec. A supported arm becomes a
  separate, flagged change with its own floor re-derivation.
- **Trigger:** ICPUSDT 2026-10-11. The gate scored the 13:00 UTC candidate at
  0.023 (floor 0.15). ICP then rose +13% from that price in a slow climb, and the
  model's scores stayed below 0.06 until 15:00. The model was right about its own
  label: over the next 5 bars ICP made only +1.34%, so `peak5_2.0` was 0. The
  label asks "+2% within 75 minutes". That is a volatility question, and a slow
  day-long climb is not in it. SHAP: volatility features (daily range, ATR in
  four forms) −1.06 logit; BTC above EMA50 +0.25.
- **Truth harness:** TH-01, TH-02, TH-03, TH-06, TH-08, TH-10, TH-11, TH-12
- **Script:** `files/_backtest_goal_label.py`. **Tests:** `files/test_goal_label_1012.py`

## Question

Would a model trained on a label that measures the move the GOAL cares about
admit more winner-days before their +2.5% crossing than the live label, at the
same admission rate and without worse trades? The live label rewards fast spikes.

## Fixed before the run

**Population (TH-06).** The bot's own candidates in `critic_dataset.jsonl`,
all stages. Rows are deduplicated per (sym, tf, bar open). A row is used only
when the 15m long store (`history/<sym>_15m_419d.csv`) holds its decision bar
and 96 bars after it. Maximum period: 2026-03-24 .. the last resolvable day.

**Timing.** The decision is made at the close of the signal bar. For 15m rows
that is the 15m bar at `ts_signal`; for 1h rows it is the 15m bar at
`ts_signal + 45m`. Every forward quantity uses bars strictly after that one.

**Arms.** Same rows, same 60 features (`safe_feature_names`), same estimator
(the live `CatBoostModel`, 300 trees of depth 4), same scaler fitted on train
only. Only the label changes.

| arm | label = 1 when |
|---|---|
| `C_peak5` (control, live) | max high over the next 5 bars of the row's tf ≥ +2% |
| `D24_5` | max high over the next 24h ≥ +5% |
| `D24_10` | max high over the next 24h ≥ +10% |
| `DVOL_2` | max high over the next 24h ≥ 2 × the coin's median daily range over the 14 prior full UTC days |
| `GOAL` | the row's (UTC day, coin) is an immutable global-top-20 winner-day and the decision falls before that day's first +2.5% crossing |

`GOAL` shares a definition with the grading metric (TH-02). That is why the
per-trade non-inferiority test exists, and why `GOAL` is reported as the
label-affine arm.

**Walk-forward (TH-03).** Each test month runs from 2026-05 to the last month.
Training uses every row whose decision is more than 1 day before that month
starts (24h embargo = the longest label horizon). Testing uses the rows of that
month. The out-of-sample predictions of all test months are pooled.

**Operating point.** Inside each test month, every arm admits the same share of
rows, its top-A by score.
- Primary A = 0.60. That is the share the live gate let through on its own
  holdout (0.67–0.69 in `.runtime/ml_promotion_log.jsonl`), rounded down.
- Secondary A = 0.40.
- The script also prints the live admit share measured on rows that carry
  `ml_proba` since 2026-09-07. It is printed for context only and is not used.

**Goal (TH-11).** Winner-days are immutable global-top-20 days of watchlist
coins with the bot up all day and a known crossing (`goal_validator.winner_days`),
taken inside the test months.

- **Reachable early day:** at least one admitted row of that coin with
  `day open <= decision < crossing`.
- **Goal metric:** reachable early days / winner-days. Arm minus control is
  reported in pp, with a paired bootstrap over winner-days (2000 resamples).

**This is a ceiling.** Gates downstream of the ML gate still apply, so a
reachable day is not an entered day.

**Per trade.** Every admitted row is replayed through the calibrated live trail
(`goal_validator.live_trail`).
- The entry is at the decision bar, on the row's tf.
- `trail_k` = median of the logged entries since 2026-05-01. The trail floor
  comes from the row's mode.
- `max_bars` is 96 on 15m and 48 on 1h.

The test statistic is the mean pnl of admitted rows, arm minus control. The
bootstrap resamples days.

**Decision rule.** With 4 arms against one control the intervals are Bonferroni
intervals at 98.75%. An arm is **supported** only if all three hold at primary A:
1. Goal delta ≥ +0.5 pp, and its 98.75% lower bound > 0.
2. Per-trade delta lower bound ≥ −0.10 pp.
3. Goal delta ≥ 0 in at least half of the test months.

An arm is **refuted** if its goal delta ≤ 0 or its per-trade lower bound
< −0.10 pp. Anything else is **inconclusive**. Secondary A and the AUCs are
context and never change the verdict.

**Also printed (context).**
- Base rate of each label.
- Precision of admitted rows on "early winner row".
- The mean 24h peak of each arm's top decile.
- The ICP 2026-10-11 rows: first admitted time per arm. This is an illustration
  and not evidence.

## Results (run 2026-10-12)

**Data.**
- 57 735 deduplicated candidates; 57 588 resolvable, from 2026-03-24 to 2026-10-12.
- Out-of-sample: 45 684 rows over 6 test months (2026-05 .. 2026-10).
- 454 winner-days.
- `trail_k` = 2.00, the median of 3 906 entries.
- Live gate since 09-07: 49.8% of 11 546 rows passed the ML stage (context only).

**Base rates.** `C_peak5` 14.7%, `D24_5` 24.8%, `D24_10` 9.1%, `DVOL_2` 7.9%, `GOAL` 0.9%.

**Primary point, top 60% admitted:**

| arm | reachable early days | Δ pp [98.75% CI] | trade mean | Δ pp [98.75% CI] | months Δ ≥ 0 | verdict |
|---|---|---|---|---|---|---|
| `C_peak5` | 26.9% | — | +0.050% | — | — | control |
| `D24_5` | 26.4% | −0.44 [−3.30, +2.42] | +0.078% | +0.028 [−0.075, +0.132] | 50% | **refuted** (goal ≤ 0) |
| `D24_10` | 26.7% | −0.22 [−2.86, +2.42] | +0.036% | −0.014 [−0.142, +0.099] | 50% | **refuted** |
| `DVOL_2` | 26.7% | −0.22 [−3.08, +2.64] | −0.009% | −0.059 [−0.246, +0.087] | 67% | **refuted** |
| `GOAL` | 33.9% | **+7.05 [+3.52, +10.57]** | +0.083% | +0.033 [**−0.220**, +0.233] | 100% | **refuted** (per-trade lower bound < −0.10) |

**Secondary point, top 40% admitted** (context):
- `GOAL` +10.35 pp [+5.51, +14.98]; trade delta +0.104, CI [−0.278, +0.419].
- `D24_5` −0.66, `D24_10` −1.32, `DVOL_2` −1.98.

**Reading.**

1. **The "size of the move over the day" labels are refuted.** They do not move
   the goal at all (−0.2 to −0.4 pp at 60%, worse at 40%). Their models pick
   bigger 24h movers (top-decile 24h peak: 7.37% / 7.65% against the control's
   7.20%). But a bigger move later in the day is not an entry before the
   crossing.
2. **The `GOAL` label finds early winner rows.** AUC on "early winner row" is
   0.683 against 0.485 for the control. Precision of admitted rows is 1.43% against
   0.93%, on a 0.97% base. It gains winner-days in **all 6 months**.
   - Its trade mean is no worse as a point estimate. But the per-trade interval is
     too wide to show non-inferiority, so under the rule fixed above it is
     **refuted**.
   - Its top decile has the *smallest* 24h peak (4.09%, against a base of 4.30%).
     It prefers quiet coins early in the day, which is the ICP shape.
   - Affinity caveat (TH-02): the label and the grading metric share a
     definition. The result is still out-of-sample (walk-forward, 1-day embargo).
3. **Ceiling.** "Reachable" ignores the gates downstream of ML, so +7 pp of
   reachable days is not +7 pp of entries.

**ICP 2026-10-11 (illustration).** At 60% admission, the control retrained on this
population already admits ICP's 13:00 candidate. The live model, trained on
`ml_dataset.jsonl` (mostly March–April signal rows), scored it 0.023 and
blocked it. The 10-11 day is not in the graded set: its crossing is not yet in
the label store.

**Not done, and not to be done by reading these numbers.** Changing the per-trade
test after the fact (for example, pairing only the rows that differ) would be a
new test. If wanted, it must be pre-registered separately.
