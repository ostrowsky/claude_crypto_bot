# Leader exit across the UTC day reset — REFUTED (2026-10-05)

Status: **refuted, nothing deployed, no config change.** Live leader exit (X-9c) unchanged; its
readout continues to 2026-10-12.
Script: `files/_backtest_leader_midnight.py` · result: `.runtime/leader_midnight_backtest.json` (+ `.log`).
Truth-harness rules: TH-04 (comparable windows: same trades, same engine), TH-06 (bot's own trades),
TH-08 (negative result committed), TH-10, TH-13 (mechanism checked in code before claiming it).

## Observation that started it (X-9c readout, 2026-09-29 .. 10-04, 6 days)

- Readout X-9c: COLLECTING (6 of 14 days). Rocket-day 15m trades n=21, mean +0.74 % [−0.85, +2.22]
  vs keep threshold +1.70 % and backtest expectation +3.3 %; switched share 0.34 (expected 0.256);
  no anomalies.
- 58 leader-mode exits; 48 "out of the day's top-10". 13 of the 48 in the 00 UTC hour (no other hour
  above 5). The 15 that crossed a midnight after the switch closed at +1.59 % on average.
- Mechanism (`files/leader_exit.py:101-104`): after 4 bars in leader mode the position exits on the
  FIRST bar the coin is outside the day's top-10, ranked by return since the UTC open — which restarts
  from ~0 at midnight.

## Design (fixed before the replay)

Engine `exit_validator` (every real 15m trade with an exit since 2026-03-01, 4 313; same switch rule
top-5 at +3 %; same 8 % wide trail; immutable labels). The current arm reproduces
`exit_validator.leader_arm` on all 4 313 trades (asserted in the script). Only the lost-lead check changes:

- **A** — skipped during the first 8 bars (2 h) of a UTC day;
- **B** — ranked by the trailing 24 h return (96 bars) instead of the return since the UTC open.

Verdict rule = `exit_validator.decide` (rocket lower95 > 0, all lower95 ≥ −0.10 pp, ≥ 60 % months ≥ 0).

## Result

The effect is real in history: of 1 038 lost-lead exits of the current rule, **428 (41 %) fall in the
00 UTC hour**. Removing it does not pay:

| variant | changed trades | rocket-days (n=375), pp | all trades (n=4 313), pp | winner-days (n=446), pp | months ≥ 0 | verdict |
|---|---|---|---|---|---|---|
| A: skip first 2 h | 440 | −0.01 [−0.29, +0.30] | −0.003 [−0.028, +0.026] | −0.06 [−0.30, +0.19] | 38 % of 8 | not accepted |
| B: 24 h rank | 932 | −0.33 [−0.82, +0.24] | −0.000 [−0.055, +0.060] | −0.36 [−0.84, +0.13] | 38 % of 8 | not accepted |

Rocket-day mean trade: current 3.29 %, A 3.27 %, B 2.96 %. Median hold after the switch: 5.25 h,
6.0 h, 4.25 h.

## Reading

By midnight a day-leader's move is mostly done; whether the position is closed by the noisy rank
right after the reset or held a few hours longer, the next day's continuation averages out. The
midnight exits look arbitrary but do not leak capture. Do not re-test day-reset variants of the
lost-lead rule on the same data. Capture remains the weak factor (0.22–0.25 of the North Star);
the remaining exit-side evidence is in `rocket-segment-spec.md` (exits on WEAK RSI divergence,
RSI overbought, micro-weakness), to be judged after the X-9c readout on 2026-10-12.
