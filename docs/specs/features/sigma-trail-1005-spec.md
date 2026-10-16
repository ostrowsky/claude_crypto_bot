# Volatility-forecast trail (σ̂ instead of ATR) — REFUTED (2026-10-05)

Status: **refuted, nothing deployed, no config change.**
Script: `files/_backtest_sigma_trail.py` · result: `.runtime/sigma_trail_backtest.json` (+ `.log`).
Truth-harness rules: TH-03 (out-of-sample only), TH-06 (bot's own trades), TH-08 (negative result committed), TH-10.

## Step 0 — baseline re-measured (same day)

`files/_compute_early_capture.py --days N`, full-uptime days only (down/partial days excluded, not scored):

| window | NS_EarlyCapture_top20_v2 | winner-days | coverage | capture | lead (clock) | v3 move-lead |
|---|---|---|---|---|---|---|
| 30 d | 0.122 | 122 | 0.81 | 0.25 | 0.60 | 0.014 |
| 60 d | 0.108 | 212 | 0.74 | 0.24 | 0.60 | 0.012 |
| 120 d | 0.108 | 316 | 0.73 | 0.24 | 0.60 | 0.013 |
| 240 d | 0.100 | 558 | 0.73 | 0.22 | 0.62 | 0.009 |

Reading: coverage is already 0.73–0.81; **capture (0.22–0.25) is the weakest factor** of the
product, and the exit trail decides capture. That is why the first tool tried was on the exit side.

## Hypothesis

Replace the volatility estimate inside the live trail
`stop = close − max(k · ATR, floor · close)` by a realized-volatility **forecast** σ̂ of the next
hour (HAR on 15m bars: log mean squared return over 15m / 1h / 4h / 1d / 1w, Parkinson range
over 15m / 1h, intraday Fourier profile; pooled OLS). On BTC 1m data a HAR of this kind reached
R² 0.75 for the next hour's log volatility, so a forward-looking buffer might stop the trail
exiting winners on noise.

## Design (fixed before the replay)

- Only the volatility source changes: `ATR_hat = κ · σ̂ · close`; `trail_k` per trade and the
  per-mode floor are untouched.
- Same width by construction: `κ = median(ATR / (σ̂ · close))` over every coin and bar before
  2026-03-01. What is compared is the SHAPE of the buffer, not wider vs narrower.
- σ̂ fitted only on 2025-06-27 .. 2026-02-28 (1.07M coin-bars); replay on every real 15m trade
  from 2026-03-01 (4 313 trades) — `exit_validator` engine (plain P-1 trail, 96 bars), its labels
  and its verdict rule: rocket lower95 > 0 AND all lower95 ≥ −0.10 pp AND ≥ 60 % of months ≥ 0.
- Primary arm σ̂ 1 h ahead; secondary σ̂ 4 h ahead.

## Result

Diagnostic first — does σ̂ forecast the next period's realized volatility better than ATR,
out of sample (≈1.95M coin-bars after 2026-03-01, both linearly calibrated on the pre-period)?

| horizon | R² σ̂ (HAR) | R² ATR |
|---|---|---|
| next 1 h | 0.330 | 0.328 |
| next 4 h | 0.562 | 0.546 |

On 15m bars ATR (a 14-bar average range) already carries almost all of the forecastable
volatility; the HAR adds 0.002–0.016 of R². The research result on BTC 1m data does not transfer
because there the naive baseline was the last 15 minutes, not a 3.5-hour average.

Replay (paired per trade, σ̂ − ATR, percentage points of trade P&L):

| arm | changed trades | rocket-days (n=375) | all trades (n=4 313) | winner-days (n=446) | months ≥ 0 | verdict |
|---|---|---|---|---|---|---|
| **σ̂ 1 h (primary)** | 1 204 | −0.04 [−0.36, +0.25] | **−0.044 [−0.082, −0.006]** | −0.07 [−0.36, +0.20] | 38 % of 8 | needs_review → **not accepted** |
| σ̂ 4 h | 1 202 | −0.14 [−0.48, +0.15] | −0.046 [−0.098, +0.003] | −0.21 [−0.54, +0.09] | 25 % of 8 | needs_review → not accepted |

- No improvement on rocket-days; all trades slightly worse (interval entirely below zero for the
  primary arm, though inside the −0.10 pp non-inferiority margin).
- Width did not stay matched after March: median buffer at entry 2.26 % (σ̂) vs 2.01 % (ATR).
  The shape change alone, at a slightly wider width, still did not help.
- `alignment` (1 255 trades) is untouched: its 8 % floor always exceeds k · ATR.

## Verdict and what follows

Refuted: a better volatility forecast is not available on 15m bars, so the trail cannot gain from
it. Do not re-test σ̂ / HAR / GARCH-style buffers on the same 15m data. Capture is still the
weakest factor; remaining exit-side levers are behavioural (when to switch to the wide leader
trail, which signals end a hold), not the volatility estimate. Next in the plan: drift / IC /
run-manifest monitoring (step 2), then seasonality-adjusted activity for early detection (step 3).
