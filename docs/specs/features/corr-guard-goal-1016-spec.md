# correlation_guard re-judged by the goal (2026-10-16)

- **Slug:** `corr-guard-goal-1016`
- **Status:** research. No behaviour change. The decision belongs to the operator.
- **Trigger:** INJUSDT on 2026-10-15/16. The move ran from 7.061 to 9.357
  (+32.5%). The guard blocked three early candidates:
  - 7.424 (16% into the move);
  - 8.44 (60%);
  - 8.863 (78%).

  Each time the coin's cluster of open positions was full (2/2). The bot entered at
  9.249, 95% into the move. The ML gate passed all three candidates.
- **Truth harness:** TH-01, TH-06, TH-08, TH-10, TH-11, TH-13
- **Scripts:**
  - `goal_validator.validate_gate_off(["correlation_guard"])`
  - `files/_backtest_corr_guard_goal.py` (re-judgement and tail check)
- **Tests:** `files/test_corr_guard_goal_1016.py`

## Prior evidence

The guard was enforced on 2026-07-07 (`CORR_GUARD_SHADOW=False`) after an 11-day
shadow: 43 would-blocks, "0 top-20 lost". It was never judged by the goal
criterion.

## Results — maximum period of enforcement, 2026-07-07 .. 2026-10-16

**Goal.** Measured on 293 winner-days, as the share entered before the +2.5%
crossing.

| run | p (downstream pass rate) | now → guard off | Δ pp [95%] |
|---|---|---|---|
| `validate_gate_off` as built | 1.0 | 9.22% → 12.63% | +3.41 |
| p measured after the guard (portfolio limits, late rotation) | 0.851 | 9.22% → 12.17% | **+2.95 [+1.21, +4.93]** |
| same, clone_signal_guard also counted after the guard | 0.791 | 9.22% → 11.99% | **+2.77 [+1.15, +4.60]** |

10 winner-days become newly reachable; about 8 of them are expected to be
entered.

**Validator defect found (TH-13).** `validate_gate_off` measured p = 1.0 here.
`_check_portfolio_limits` runs AFTER the guard, but its blocks have no pipeline
stage, so `_iter_rows` drops them from the population instead of counting them as
candidates that passed the guard and were not entered. The run's verdict was
"accept"; with the measured p it is still positive.

**Per trade** (calibrated live trail, as in the validator). There are 910 added
trades.
- Their mean is −0.011%, against −0.000% for the 2 416 current trades.
- Combined minus current: −0.003 pp [−0.089, +0.078]. That is non-inferior.
- Months agree 2 of 4.

**Tail — the guard's own purpose.**

- **Pre-registered criterion.** The worst 10% of days by the current trades'
  summed P&L (9 days). On those days the added trades average −1.48% against
  −2.42% for the current ones (+0.94 pp). The guard is not needed there, so this
  criterion is **met in favour of removal**.
- **Exploratory: the criterion was mis-specified.** It conditions on the days the
  CURRENT trades are bad. The guard's risk shows on the days its BLOCKED trades
  cluster, which are different days:

| day | added trades | added sum | current sum | combined |
|---|---|---|---|---|
| 2026-08-27 | 60 | −66.0% | −47.7% | **−113.7%** |
| 2026-08-22 | 26 | −57.1% | −40.3% | −97.3% |
| 2026-08-23 | 56 | −46.2% | −28.4% | −74.6% |
| 2026-09-30 | 27 | −46.2% | −45.3% | −91.5% |
| 2026-09-29 | 38 | −40.4% | +3.7% | −36.7% |

| daily summed P&L (equal-weight alerts) | current | guard off |
|---|---|---|
| worst day | −94.8% | **−113.7%** |
| 5th percentile | −71.7% | −89.8% |
| max drawdown of the cumulative sum | −588 | **−767 (+30%)** |
| total | −0.9 | −11.0 |

## Reading

Removing the guard gains about +2.8–3.0 pp of winner-days entered early (about
8 of 293 days in 3.3 months). The mean trade is unchanged. The cost is clustered
losing days: the guard did what it was built for on 08-22/23/27 and 09-29/30.
This is a trade-off between the goal (earliness) and tail risk, not a free win.
It is the operator's call, and the drawdown numbers are exploratory because the
pre-registered tail test missed them.

**Not tested yet.** These middle grounds each need their own pre-registered run
before any change:
- `CORR_MAX_PER_CLUSTER` 2 → 3;
- exempting candidates that come before their day's crossing;
- exempting the day's leader.
