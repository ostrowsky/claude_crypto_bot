# Lateness caps and RSI partial exit — two refuted levers (QNT, 2026-09-24)

- **Slug:** `lateness-caps-and-rsi-partial-exit`
- **Status:** REFUTED 2026-09-25. No behaviour change ships.
- **Truth-harness invariants:** TH-01 (base rate beside every ratio), TH-04
  (same daily_range bucket = comparable rows), TH-06, TH-08 (negative results
  committed with their numbers), TH-11 (the forward peak is a proxy — downside
  and a trailed outcome are reported beside it), TH-13
- **Backtests:** `files/_backtest_lateness_caps.py` (replay),
  `files/_backtest_lateness_caps_analyse.py`, `files/_backtest_lateness_caps_downside.py`,
  `files/_backtest_rsi_exit_partial.py`
- **Tests:** `files/test_lateness_and_partial_exit.py`
- **Rows:** `.runtime/backtests/lateness_rows.jsonl` (126 456 rows)

## What prompted it

QNTUSDT 2026-09-24: +29% on the day (70.69 → 91.40), +58.9% from the 09-22 low
to 104.68 on 09-25. The bot entered at 73.32 (12:19 UTC), exited at 77.00 on
`RSI перекуплен (87.1)` (+5.02%), and then produced no candidate for 33
consecutive 15m bars while the price went 78.96 → 89.48. A replay with the
bot's own rule functions (matching the live log at 11:30, 12:00 and 22:00)
showed no rule firing on any of those bars: trend on volume/RSI, alignment on
`daily_range > 9%`, impulse on the 1-bar move, impulse_speed on slope/MACD,
ema_cross on volume. Cooldown was not the cause — nothing fired without it.

Two hypotheses followed: the "too late" caps cut the biggest moves; and a
partial exit at RSI-overbought would keep the tail the RSI exit throws away.

## 1. Lateness caps — REFUTED

Candidates no rule produces never reach `bot_events.jsonl`, so the strategy was
replayed on raw klines: 101 watchlist coins, 2025-06-29 … 2026-09-25, five live
rules evaluated three times — caps as coded (7%), bull-day caps (10%), every
daily_range cap lifted.

| band | n | 4h peak | 4h trough | trailed mean | trailed median |
|---|---|---|---|---|---|
| FIRE (today) | 120 542 | 1.67% | −1.51% | +0.01% | −0.34% |
| BULLBAND | 3 135 | 3.18% | −2.66% | +0.04% | −0.67% |
| LATE (caps lifted) | 2 744 | 4.83% | −3.92% | −0.01% | −1.07% |

The bigger peak is volatility: the trough grows with it, and inside the same
daily_range bucket LATE matches or trails FIRE (15–25%: +0.27% vs +0.29%;
25%+: −0.62% vs −0.33%). Winner-days: a rule already fires on 819 of 829
immutable top-20 winner-days (98.8%) under today's caps; lifting all caps adds
4. The rule layer fires on 38 212 coin-days at lift 1.00 — it is not where
winners are lost. QNT 09-24 has no LATE row: lifting the caps would not have
produced a signal in its silent afternoon.

A 07:30 UTC alignment row for QNT turned out to be a data artefact (vol_x 1.01 in
the long store vs 0.94 on live spot; non-bull floor 1.00; `is_bull` False all
morning) — live was right to stay silent.

## 2. RSI-overbought partial exit — NOT SUPPORTED

The RSI exit is the most profitable exit class (121 exits, mean +7.04%). On the
103 with pnl ≥ 2%: a full switch to the ATR trail gives +0.614% per trade,
95% CI [−0.76, +2.22]; a 50/50 split gives exactly half, +0.307% [−0.38, +1.09],
positive in 3 of 7 months, and lowers the worst trade from +2.12% to −0.66%. The
trail wins on 35% of trades; a few very large wins carry the mean.

## Shadow / canary and rollback

Not applicable (не применимо): nothing changes in the bot. Maximum-period
evidence is the backtests above.

## What remains open

The day's biggest moves are reached by the rules but lost downstream — gates,
the exit, and the re-entry after it. For QNT the live exit came on RSI at +5%
and every later candidate hit `trend_quality` (`price_edge`, `daily_range`),
whose own caps (`TREND_15M_QUALITY_*`) are a separate question from the ones
refuted here.
