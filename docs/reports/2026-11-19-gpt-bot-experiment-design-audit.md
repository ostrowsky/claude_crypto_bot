# gpt_crypto_bot experiment-design audit — rejected hypotheses worth re-testing here (2026-11-19)

- **Status:** research, read-only audit of `D:\Projects\gpt_crypto_bot`. Nothing in
  that project was changed, and none of its processes were touched.
- **Lens:** this repo's truth-harness checklist
  (`skills/crypto-bot-truth-harness/references/audit-checklist.md`, TH-01..TH-12).
- **Scope:**
  - 31 rejected entries in its `docs/FEATURE_SPEC_INDEX.md`;
  - the 19 reports in `docs/reports/`;
  - the specs of the entry-, exit- and ranking-quality hypotheses: impulse-entry
    CatBoost, turnover economics, capacity CatBoost, external Top-50 screen,
    rank velocity, trend-start, slow persistent trend, early-block rescue,
    post-block discriminator, delayed confirmation, objective ranking, observable
    tail selector, exit-action advantage, joint direction-amplitude.
- **Purpose:** find hypotheses that were rejected because of how the test was
  built, not because the idea failed, and turn them into hypotheses for this
  bot's signal quality.

The neighbour's protocols are strict: pre-registration, time splits and
independent verifiers. The defects below are in the acceptance criteria and
measurement design, not in its honesty.

## Recurring design defects

### E1 — positive-skew policies judged by median and worse-rate

"Let the winner run" policies lose often and a little, and win rarely and a lot.
Gates that require a non-negative median, a worse-rate ≤ 30–35% or a
non-decreasing win rate reject them by construction.

| Hypothesis | Mean effect | Reason for rejection |
|---|---|---|
| hold 5 bars after a WEAK sell | **+0.366%** per case | worse 41% |
| 50% tail after a 1h-confirmed WEAK | validation **+0.49 pp**, holdout **+0.34 pp** | median < 0, worse 61% |
| exit-action model, TEST selected actions (n = 33) | mean **+0.58 pp**, median **+0.61 pp** | hurt 42% > 35% ceiling |
| learned tail, sell 50% / hold 10 bars | selected +0.17 pp, overall +0.04 pp | median < 0, worse 65% |

The reports state the reason in so many words: "positive means are driven by a
minority of large winners". For a goal of holding day-leaders, that minority is
the target.

### E2 — trend decisions measured on a fixed short horizon

- Every entry model's target is the 75-minute return; the median holding time
  is 90 minutes.
- The live tail-selector check used a fixed T+10 (2.5 h on 15m); forward it read
  −0.12%.
- Delayed confirmation was judged on 1–4 h and end-of-day returns, although its
  MFE was **8.7%** against 3.7% for the base.

A trailing or leader exit was never applied to these cohorts.

### E3 — portfolio gate with no power, over a broken control

- **The control loses −96% over 186 days.** Raw price: −1,309 USDT. Fees plus
  slippage: −8,317 USDT. That is about 59 trades a day on 10 slots.
- **The test cannot detect a real effect.** The paired 95% intervals are ±60–100
  bp a day over 38 TEST days. Three October models had positive TEST point
  estimates (+3.4 / +7.2 / +9.5 pp) and were all rejected.
- **The gate requires everything at once.** Six to seven conditions must hold
  together, one of them "no fewer captured / early Top-15 pairs" with zero
  tolerance, which any filter almost surely fails.
- **The rejections are not independent.** Every test uses the same control, so
  one weak control produces many rejections.

### E4 — the label contains the baseline (TH-03)

- **The baseline already contains the answer.** In the Top-50 screen and
  rank-velocity studies the label is the rolling-24h return at 23:00. The
  baseline `static_target` is the part of that window already known at 12:15, so
  its 3.87× lift is largely mechanical.
- **The hypotheses were never really tested.** Momentum, volume acceleration and
  3h rank velocity were judged against that baseline. Any re-ranking away from
  the known part loses mechanically. Whether they predict the remaining
  12:15 → 23:00 move was never tested.

### E5 — absolute threshold instead of matched admission

The impulse_speed CatBoost admitted only candidates with forecast net return
> 0. That was 417 of 48,327 (0.9%): the "filter" all but switched the mode off,
and was then judged on early capture.

### E6 — selection on the test, then a forward reversal

- `non_ema_mfe150`: picked from a rule grid, +0.28 on test, −0.12 forward.
- The post-block discriminator: its "best holdout variant" was chosen on a
  7-day holdout of 526 rows, then rejected for selecting too few candidates.

### Minor defects

- **Wrong population (TH-06).** The slow-persistent-trend detector was validated
  on every 1h bar of the market since 2017, not on the bot's candidates. It was
  judged on a fixed 24h return (36h was positive) and an incident deadline.
- **Proxy PnL (TH-11).** Trend-start used summed per-trade percentages; its
  baseline swung from +31.8% to −259% after a measurement fix.
- **Arbitrary harm credit.** The gate-harm audit credited each blocked non-winner
  with a fixed 1%.
- **Short samples.** Several studies had 27–43 label days.

## Check against this repo

- **E1 — not repeated.** Our readouts judge by means with intervals, not by
  worse-rate. For example WEAK-HOLD-SHADOW: rocket-day lower 95% > 0 and
  all-trade lower ≥ −0.10.
- **E2 — repeated once.** Our ML gate label `peak5_2.0` is the same 75-minute
  question; it is being addressed by the GOAL-label shadow
  (`ml-goal-shadow-1012-spec.md`).
- **E3-class defect found.** `goal_validator.validate_gate_off` reported p = 1.0
  for `correlation_guard` (`corr-guard-goal-1016-spec.md`). The cause:
  `botlog.log_blocked(sym, tf, 0.0, ...)` logs portfolio-limit blocks with price
  0.0, and `pipeline_replay_validator._iter_rows` drops rows with price ≤ 0.

## Hypotheses for this bot, by relevance to the goal

1. **Rotation when the portfolio is full.** It addresses late entry. When slots
   were full, the neighbour's rejected candidate beat the weakest incumbent 57%
   of the time (+0.15 pp ret5), and 79 objective pairs were missed in 29 days;
   its ranker test was killed by E3/E5.
   - **Our rotation went dead on its own.** ML-gated rotation needs
     `ml_proba ≥ 0.62`. Since the 2026-09-07 label switch, the share of
     portfolio-blocked candidates reaching that level fell from 32% (April) to
     0–2% (August–November). Only 5 ML evictions appear in the logs kept since
     09-29.
   - **Score replacements fell too:** from about 100 a month to about 5.
   - **Result: REFUTED for this bot** (`rotation-goal-1119-spec.md`).
     - Neither arm gained a single winner-day out of 599.
     - The ceiling is 2 days (0.33 pp): only 3 winner-days ever had a
       portfolio-blocked candidate before the crossing, and 31 of the 34 such
       days saw it after.
     - The neighbour's capacity regret does not transfer: our bot rarely fills
       its portfolio before leaders cross.
2. **5m scanning.** The neighbour accepted the problem (46% of entered winners
   had an earlier blocked signal, median lag 49 min). Neither repo has tested
   the remedy.
3. **50% partial exit with a trailing tail.** Judge it by capture on rocket-days
   and by means with intervals. It was rejected only through E1/E2.
4. **Rank velocity or volume acceleration on the correct target** (the remaining
   move from observation to end of day). The neighbour never tested it (E4). Our
   own five price/volume trend-start attacks were negative, so priority is low.

**Do not re-test.** Delayed-confirmation entries (consistent with our three
refutations of crossing entry), the trend-start profile, and BTC-led timing.
