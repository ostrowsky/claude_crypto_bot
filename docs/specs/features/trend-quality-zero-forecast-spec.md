# trend_quality on a forecast that does not exist — measured, refuted

- **Slug:** `trend-quality-zero-forecast`
- **Status:** REFUTED as a lever 2026-09-18. No gate change ships.
- **Truth-harness invariants:** TH-01 (base rate beside every ratio), TH-05
  (absence of data is not evidence — the question this answers), TH-06 (the
  bot's own candidates), TH-08 (negative result committed), TH-13
- **Flags:** none switched. Rule under test: `TREND_15M_QUALITY_GUARD_ENABLED`,
  `TREND_15M_QUALITY_FORECAST_MIN = 0.25`, alt path `vol_x ≥ 1.2`, `ADX ≥ 24`,
  `slope ≥ 0.35`.
- **Backtest:** `files/_backtest_trend_quality_zero_forecast.py`
- **Tests:** `files/test_trend_quality_zero_forecast.py`

## What prompted it

STRK rose **+34.9%** from 0.02608 (09-16 17:00 UTC) to 0.03519 (09-18 08:00).
The bot produced one signal: 09-18 01:01 at 0.02903, exited 7 bars later at
0.02943 (+1.38%) on `WEAK: RSI divergence`. Every other attempt was stopped by
`trend_quality` — 61 blocks, no other gate involved.

64 of STRK's 96 `weak 15m trend` reasons carried **forecast exactly 0.000**. The
alt path then failed on ADX (54), vol+ADX (32), vol (10) — ADX lags at the
start of a move.

## Why 0.000 means "no data", not "bad"

`strategy.py:1932`: `forecast_return_pct = max(expected_return over THIS coin's
rule signals since 00:00 UTC)`, and when fewer than `TODAY_MIN_SIGNALS = 2` are
evaluable it is **0.0**. A coin that has just started moving has no signals
today, so it has no forecast, so the gate scores it as a bad forecast, so it
does not signal. It resets for every coin at UTC midnight. Across the log, 78%
of 32 826 such blocks since 2026-04-09 carried 0.000.

That is CLAUDE.md §0a rule 5 in the code. The question is whether it COSTS.

## Result

Max period 2026-04-01 … 2026-09-18, 12 594 deduplicated 15m decisions.

| 4h forward peak | n | median | mean | ≥3% | ≥5% | vs PASSED |
|---|---|---|---|---|---|---|
| passed trend_quality | 8729 | 0.95% | 1.62% | 15.1% | **5.9%** | — |
| rejected: NO DATA | 2877 | 0.98% | 1.53% | 14.0% | **4.4%** | 0.94× |
| rejected: LOW FCST | 981 | 0.96% | 1.47% | 14.3% | 5.3% | 0.90× |

5 bars: 0.91 / 0.84 (0.92×) / 0.79 (0.86×); ≥5% 1.8 / 1.3 / 1.0.

**The conflation is wrong in principle and costs nothing measurable.** The
population it mis-scores has a thinner big-mover tail than what the gate passes;
admitting it would dilute the flow. September — the regime that raised the
question — sits at parity (1.74% vs 1.69%) and does not reverse it. The midnight
reset is visible: 00–05 UTC holds a third of NO DATA rejects, and they are
weakest there (0.72% vs 0.87%).

Comparability caveat: blocked events do not record the entry mode, so PASSED is
the 15m pass-through of the whole pipeline, not a matched trend-only control.

## A data defect this file found — bigger than its own question

`_backtest_trend_start_detector.bars_15m` reads only
`history/<sym>_15m_419d.csv`, a one-off backfill that **ends 2026-08-20**. The
daily task writes a different file, `history/<sym>_15m.csv`, a **rolling 30-day**
window. The first run of this backtest lost 4104 of 12 592 rows and all of
September to it.

The same loader feeds the peak TRAINING label (`ml_signal_model._peak_bars`).
Resolution of 15m rows: 100% through July, **48% in August, 0% in September** —
those rows silently fall back to the old inverted `ret_5 > 0` label, and the
overall 92% hides it from `ML_PEAK_LABEL_MIN_RESOLVED`.

And the gap grows: once the rolling window's start passes 2026-08-20 08:00
(the 06:00 run of 2026-09-20), bars between the two files stop existing on disk.
Fix is tracked separately; it is a behaviour change to the gate model's labels.
