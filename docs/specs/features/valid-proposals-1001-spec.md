# Report steps researched on the maximum period (2026-10-01)

- **Slug:** `valid-proposals-1001`
- **Status:** research done; evidence store updated; **no trading behaviour changed**
- **Owner:** operator + Claude (operator: «исследуй все валидные предложения на макс периоде по стандартному флоу»)
- **Truth harness:** TH-01, TH-02 (features that already contain the answer), TH-03
  (time split), TH-06 (the bot's own decisions), TH-08 (negative results kept),
  TH-10 (lock evidence refreshed), TH-11 (goal, not a proxy), TH-13
- **Rollback:** `do_not_touch.json` before the write is kept by the operator's
  backup; nothing in config or live decisions changed — no flag, no shadow needed

Of the morning report's steps, two were valid: (1) re-verify the `do_not_touch`
gate locks (126 days past a 30-day budget) and (2) time-held-out ranking at a fixed
alert budget. The EX1-reference step was judged low value (capture is already
measured by the North Star decomposition and the exit validator).

## 1. Gate locks re-judged by the goal — `files/_replay_locked_gates_goal.py`

The locks were verified 2026-05-28 by a proxy (5-bar return of the blocked bucket).
New: `goal_validator.validate_gate_off` — every row a gate blocked on the day is
admitted (downstream pass rate p), judged by the L3 criterion: winner-days entered
before the +2.5% crossing, per-trade non-inferiority (−0.10 pp), months. Maximum
period 2026-05-01.. (412 winner-days); the ML gate only on the current model's era
(peak label, 2026-09-07..), since its blocked population depends on the model.

| lock | goal if removed | trades: combined − current | verdict |
|---|---|---|---|
| trend_1h_chop | 10.9% → 12.2% (+1.26 pp) | −0.752 [−2.501, +0.687] | **confirmed** |
| open_cluster_cap | +0.00 pp | +0.005 [−0.003, +0.014] | **confirmed** |
| ml_proba_zone (09-07..) | 9.6% → 12.2% (+2.62 pp, 94 days) | −0.122 [−0.274, +0.032] | **confirmed** |
| ml_proba_zone (05-01.., old models) | +1.80 pp | +0.021 [−0.013, +0.058] | not the gate running now |
| mode_range_quality | 10.9% → 14.4% (**+3.43 pp**) | −0.002 [−0.049, +0.045], months 60% | **unsupported by the goal** |
| mtf, cooldown | — | — | not replayable: never logged as blocked events |

Cost the goal criterion does not see (`files/_gate_off_cost.py`, operator wants
no junk): removing mode_range_quality adds 1 418 new coin-days (11.8/day) of
which, with p = 0.56, ~**+6.6 messages/day**, and only **1.1%** of those new
coin-days are winners (current entries: 10.9%). Its goal gain comes mostly from
turning LATE entries into early ones on coins the bot enters anyway. Hence it is
moved to `contested` in `do_not_touch.json` with its evidence and **left
enabled**; the report shows a P1 operator decision. The confirmed locks got
`last_verified = 2026-10-01`, `verified_via = goal_validator.validate_gate_off`.

Regression of the refactor: the 09-29 chop hypothesis (22 → 20) still rejects —
goal +0.00 pp, −0.094 [−0.433, +0.292] on 412 winner-days (was −0.100 on 401).

## 2. Alert-budget ranking — `files/_backtest_alert_budget.py`: REFUTED

Online rule: send an entry message only if a score known at entry ≥ T; T and the
score chosen on TRAIN (05-01..07-31, 64 days) for a budget B; judged on TEST
(08-01.., 52 days) against all entries (28.8 msg/day, precision 0.111, 18 early
winner-days, mean pnl −0.027%). Pre-registered success at B = 10: precision ≥ 2×
with Wilson lower bound above the current, ≤ 10 msg/day, ≥ 50% of early winner
catches kept, pnl lower bound ≥ current − 0.10 pp.

| B | chosen on TRAIN | TEST msg/day | precision [95%] | early winner-days kept | mean pnl |
|---|---|---|---|---|---|
| 3 | logit of all scores | 3.6 | **0.466** [0.394, 0.540] | **0 of 18** | −0.049% |
| 5 | logit | 5.3 | 0.362 [0.306, 0.422] | 0 of 18 | −0.041% |
| 10 | logit | 10.4 | 0.248 [0.212, 0.289] | 1 of 18 (6%) | +0.128% [−0.265, +0.519] |

**Refuted** on the early-catch condition at every budget. What it shows (TH-02):
the scores that buy precision — day rank, return since the open, the ranker's
top-gainer probability — already contain the answer: they recognise a winner that
is moving, i.e. late. The scores that keep early catches (ranker EV, ML) have
precision BELOW the current stream (train: 0.06 / 0.10–0.12). With the features
available at entry, precision and earliness pull in opposite directions — the
same conclusion as late-entry-0930-spec.md from the other side.

The result is written to `.runtime/backtests/alert_budget_result.json`; the
report no longer offers the step while a result younger than 30 days exists.

## Tests

`files/test_valid_proposals_1001.py` (8): gate-off admits only rows the gate
blocked (not rows stopped earlier), a stage-less gate is not replayable, the ML
lock's era, contested lock → operator step, a tested budget step is not
re-offered, pre-registration constants, Wilson / logit sanity.
