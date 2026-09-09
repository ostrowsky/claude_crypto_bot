# WEAK exits above break-even — measured, and refuted

- **Slug:** `weak-exit-above-breakeven`
- **Status:** REFUTED 2026-09-09. No code change ships. Recorded so the
  hypothesis is not re-invented (TH-08).
- **Truth-harness invariants:** TH-01 (base rate beside every ratio), TH-06
  (graded on the bot's OWN trades), TH-08 (negative result committed with the
  numbers that killed it), TH-10 (proxy is not the outcome), TH-13
- **Flags:** none — nothing was switched. The rule under test is the existing
  `H5_TRAILING_ONLY_AFTER_BREAK_EVEN_ENABLED` / `H5_BREAK_EVEN_PCT = 0.5`.
- **Backtest:** `files/_backtest_weak_exit_above_breakeven.py`
- **Tests:** `files/test_weak_exit_backtest.py`

## What prompted it

ATOM ran three days, 1.559 to 1.967 (+26.2%). The bot booked **+2.50%** of it:

| time (UTC) | mode | in → out | result |
|---|---|---|---|
| → 09-06 00:15 | alignment | 1.559 → 1.563 | +0.26%, `WEAK: RSI divergence` |
| 09-07 07:45 → 10:30 | alignment | 1.621 → 1.616 | −0.31%, first close below EMA20 |
| 09-08 11:30 → 13:30 | alignment | 1.687 → 1.730 | +2.55%, `WEAK: volume exhaustion` |
| 09-09 09:09 | impulse_speed | 1.965 | open |

Entries were **early** — 1.559 is the base of the move — so goals 1 and 2 held
and goal 3 (exit only before the trend ends) failed. `trend_quality` also blocked
ATOM 91 times and `ml_zone` 23 times during the run.

The suspect was visible in `monitor.py::_h5_should_suppress`: H5 hands control to
the ATR trail past break-even, except it returns False for every WEAK reason. The
+2.55% exit was five times the 0.5% floor and still fired.

## The question, framed to be answerable both ways

For every WEAK exit that fired **past break-even**, what would the ATR trail have
returned instead? The trail is not "hold forever" — it exits on price rather than
on a pattern, so this is a substitution, not the removal of a stop.

The counter-hypothesis, stated first: WEAK is a genuine early warning that fires
before a reversal the trail only catches after giving back its buffer.

## Method

- **Population:** exits whose reason is WEAK and whose realized `pnl_pct` ≥
  `H5_BREAK_EVEN_PCT` — the exact set H5 would suppress.
- **Counterfactual:** trade stays open at the exit bar; ATR trail runs with
  `width = max(trail_k × ATR%, TRAIL_MIN_BUFFER_PCT_<MODE>)`, peak seeded with
  the running high since entry (`bars_held` back).
- **Ordering:** the stop is tested against the bar's LOW **before** the peak
  absorbs that bar's HIGH. The other order lets a bar's own spike widen the stop
  it then fails to hit, inflating every result silently.
- **Control:** WEAK exits *below* break-even — H5 leaves those alone, so any
  gain there is the market rising, not the rule working.
- **Cap:** 96 bars. A longer cap flatters the counterfactual.

## Result — REFUTED

Maximum period 2026-03-03 … 2026-09-09: 4898 exits, **1271 WEAK (26%)**, 846
above break-even, 699 resolvable against klines.

| group | n | actual med | trail med | delta med | trail better | delta avg |
|---|---|---|---|---|---|---|
| WEAK, above break-even | 699 | 1.36% | 0.86% | **−0.79%** | 34% | **−0.21%** |
| WEAK, below (control) | 370 | 0.17% | −0.51% | −0.33% | 38% | −0.11% |

The trail made **460 of 699 (66%)** trades worse: −931.5% of damage against
+786.7% of gains, **net −144.8%, or −0.207% per trade**. Negative in all five
entry modes (impulse_speed −0.34, trend −0.40, retest −0.16, impulse −0.08,
alignment −0.02) and in six months of seven.

**WEAK is an early warning that works. The exclusion in `_h5_should_suppress` is
correct and stays.**

## The dissenting cell, and why it does not rescue the idea

2026-09 alone read +1.97% per trade, 54% better — the very regime that raised the
question. Tested against the RM-22 step-A precedent, where gate-blocked forward
returns flip sign across regime cells (spread 0.548pp):

| regime | n | delta avg |
|---|---|---|
| btc_up | 442 | −0.22% |
| btc_dn | 186 | −0.54% |

Same sign, no flip. September's `btc_up` cell is **n=16**, below the n ≥ 20 bar
the script fixed for itself before any number was seen. 2026-05 `btc_up` was also
positive (+0.75%, n=46) while its `btc_dn` was −0.84% — inconsistent, not
regime-conditional.

## Honest limits

- Graded on trades the bot actually took (TH-06); it says nothing about exits on
  positions the gates never opened.
- ATOM really did give up ~24pp. Refuting this substitution does not make the
  leak imaginary — it removes one candidate fix.
- The 96-bar cap bounds the counterfactual in both directions.

## Do not re-test without

new evidence of a different kind: a longer horizon, an exit rule that is not the
ATR trail, or a population beyond the bot's own trades.
