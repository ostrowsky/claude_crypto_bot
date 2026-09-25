# trend_quality price_edge / daily_range / RSI caps — measured, refuted

- **Slug:** `trend-quality-caps`
- **Status:** REFUTED 2026-09-25. No change ships.
- **Truth-harness invariants:** TH-01, TH-04 (current vs relaxed thresholds on
  the same logged inputs, not the day's decision), TH-06, TH-08, TH-11 (peak
  reported beside trough and a trailed outcome), TH-13
- **Backtest:** `files/_backtest_trend_quality_caps.py`
- **Tests:** `files/test_trend_quality_caps.py`
- **Related:** `trend-quality-zero-forecast-spec.md` (the forecast/alt path of the
  same guard), `lateness-caps-and-rsi-partial-exit-spec.md` (the strategy caps)

## Why

After QNT's +29% day (2026-09-24) every later candidate was stopped by
`trend_quality` on `price edge 4.6–6.8% > 3.20%` and `daily_range 34% > 10%`.
The guard (`monitor._trend_entry_quality_guard_reason`, 15m trend candidates;
impulse_speed falls back to trend while curtailed) checks, in order:
price_edge > 3.2 (bull 4.0) → daily_range > 10 (bull 14) → RSI > 72 (bull 76)
→ forecast/alt path.

## Method

Each logged rejection (2026-03-10 … 09-25, 6 240 after dedup) is re-judged by
applying today's thresholds and a relaxed set to the same logged inputs, with
the later checks still applied. Relaxed rows are compared with trend/15m entries
(653) by 4h peak, 4h trough and an ATR trail from the row's price. The forecast,
logged only on `weak` rows, is taken as 0.0 elsewhere (78% of known rows).

## Result

| cap | rows freed | trailed vs entries | notes |
|---|---|---|---|
| price_edge (lifted) | 17 of 381 | — | 232 then stopped by daily_range, 113 by RSI |
| RSI 72 → 80 | 683 | −0.12% / +0.09% mean | winner-day lift 1.0× / 0.8× |
| daily_range 10 → 20 | 115 | +0.62%, 95% CI [+0.01, +1.38] | median −0.46%; without its 3 best trades +0.02%; one of 12 grid points |
| all three lifted | 1 109 | +0.14%, CI [−0.05, +0.34] | 4 of 5 months ≥ entries |

No cap removes demonstrably better candidates. The daily_range band is the only
lead, and it rests on three trades.

## Shadow / canary and rollback

Not applicable (не применимо): nothing changes. Maximum-period evidence above.

## Pre-registered re-test (written 2026-09-25, before any new data exists)

The daily_range 10 -> 20 band is the only lead this measurement produced, and it
rests on three trades. It is re-read ONCE, on rows logged after 2026-09-25 only,
never on the rows above:

- change tested: `TREND_15M_QUALITY_DAILY_RANGE_MAX` 10 -> 20, nothing else;
- method: this backtest, unchanged; control = trend/15m entries of the same period;
- read no earlier than when the new band holds n >= 100 rows;
- accept only if ALL hold: trailed mean minus control > 0 with the 95% interval
  excluding zero; band median >= control median; mean without its 3 best trades
  still > control mean.

Anything else is a refutation, recorded here.
