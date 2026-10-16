# correlation_guard compromises: cluster cap 3, pre-crossing exemption (2026-10-16)

- **Slug:** `corr-guard-compromise-1016`
- **Status:** research, pre-registered. No behaviour change.
- **Parent:** [`corr-guard-goal-1016`](corr-guard-goal-1016-spec.md). Removing the
  guard entirely gives +2.8–3.0 pp of early winner-days, and its max drawdown is
  +30% (exploratory).
- **Truth harness:** TH-01, TH-06, TH-07, TH-08, TH-10, TH-11
- **Script:** `files/_backtest_corr_guard_compromise.py`. **Tests:** `files/test_corr_guard_compromise_1016.py`

## Question

Can the guard keep most of its tail protection while giving back most of the
early entries?

## Fixed before the run

**Population.** `correlation_guard` blocks in `bot_events.jsonl` over the whole
enforcement period: 2026-07-07 .. 2026-10-16, 15 290 blocks, all of which carry a
parsable `cluster size n/cap`. Rows are deduplicated per coin and bar, as in
`goal_validator`.

**Arms.** A guard-blocked row is ADMITTED under:

| arm | admitted when |
|---|---|
| `CAP3` | the logged cluster size n < 3 (`CORR_MAX_PER_CLUSTER` 2 → 3) |
| `PRECROSS` | the coin's return from its UTC day open, at the logged price, is < +2.5%. This is known at decision time: the day open is the first 15m bar of the UTC day in the long store |
| `OFF` | always (reference, the parent's arm) |

**Measurement** follows the parent.

- **Goal.** Winner-days entered before the +2.5% crossing; expected gain
  Σ(1−(1−p)^k).
  - p is the share of coin-hours that passed the guard (entry or a later
    portfolio block) and ended with an entry.
  - It is measured on `critic_dataset` with clone_signal_guard counted after the
    guard: p = 0.791, the conservative of the parent's two values.
- **Per trade.** Combined minus current mean with the added trades at weight p,
  using the calibrated live trail. Trades are resampled as in
  `goal_validator.bootstrap` (2000 resamples), and the interval level is the
  97.5% fixed below.
- **Tail, specified on the right days this time.** Daily summed P&L of
  equal-weight trades, with added trades weighted by p, over all days. Two
  statistics:
  - the max drawdown of the cumulative daily sum;
  - the worst single day.

**Decision per arm (2 arms against the current guard, Bonferroni: 97.5%
intervals).**

| verdict | condition |
|---|---|
| **supported** | goal Δ ≥ +0.5 pp AND its lower bound > 0 AND per-trade lower bound ≥ −0.10 pp AND max drawdown no more than 10% deeper than the current one AND worst day no more than 10% worse than the current one |
| **refuted** | goal Δ ≤ 0, OR per-trade lower bound < −0.10 pp, OR either tail statistic more than 10% worse |
| **inconclusive** | otherwise |

**Known limits.**
- Admitting more trades fills clusters and the portfolio sooner, which would
  block later candidates. This interaction is only approximated by p.
- The tail statistic treats every alert as one equal unit. The bot does not size
  positions.

## Results (run 2026-10-16)

**Setup.** 293 winner-days and 1 875 guard-blocked bars; p = 0.791.

**Current guard.** 2 416 trades, mean −0.000%, max drawdown −588.1, worst day −94.8.

| arm | admitted bars / added trades | goal Δ pp [97.5%] | per trade [97.5%] | max drawdown | worst day | verdict |
|---|---|---|---|---|---|---|
| **CAP3** | 716 / 446 (mean −0.154%) | **+1.08 [+0.27, +2.43]** (9.22% → 10.29%) | −0.020 [−0.067, +0.028] | −646.1 (**+9.9%**) | −97.7 (+3%) | **supported (marginal)** |
| PRECROSS | 751 / 463 (mean −0.342%) | +2.77 [+1.08, +4.86] | −0.045 [−0.095, +0.001] | −736.4 (+25%) | −97.7 (+3%) | refuted (tail) |
| OFF | 1 875 / 910 (mean −0.011%) | +2.77 [+1.08, +5.00] | −0.002 [−0.088, +0.075] | −725.2 (+23%) | −99.9 (+5%) | refuted (tail) |

**Reading.**

- **PRECROSS.** It takes the whole early-entry gain of OFF (+2.77 pp; by
  construction the early candidates are the pre-crossing ones), but its added
  trades are the worst of the three (−0.34%) and its drawdown is +25%. On the days
  the guard protects against, the candidates it blocks are exactly the early
  ones.
- **CAP3.** It passes every criterion, but only just: the drawdown is +9.9%
  against a 10% bound. That number is a point estimate with no interval, so a
  slightly different period could flip it.
- **Trade-off.** CAP3 recovers about 1/3 of the early-entry gain (+1.08 of
  +2.77 pp) for under 10% more drawdown.
- **One cluster cap does not tell early from late.** The guard blocks early
  candidates on good days and on shock days alike.

**Recommendation.** If the operator wants it, ship `CORR_MAX_PER_CLUSTER = 3` as
a reversible config change with a readout. A shadow cannot add information here:
the would-be admits (blocks at cluster size 2/2) are already logged. Read out:
- the early winner-days gained;
- the live max drawdown and worst day against the current guard's levels.

Rollback is to set the cap back to 2.
