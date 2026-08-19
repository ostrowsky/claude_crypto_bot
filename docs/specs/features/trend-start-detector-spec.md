# Trend-start detector: catching the move, not the close

- **Slug:** `trend-start-detector`
- **Status:** evidence 2026-08-19 — POSITIVE at the +20% target, negative below it; nothing deployed
- **Created:** 2026-08-19
- **Truth-harness invariants:** TH-01 (base rate and lift beside every ratio),
  TH-03 (holdout split by time), TH-06 (the population the bot actually sees —
  here deliberately *upstream* of it), TH-10 (results committed with the numbers)
- **Flag:** none — this is measurement
- **Rollback:** не применимо

## The target changed, and everything before it answered a different question

The operator's definition, stated 2026-08-19 and now in durable memory:

> **Top-20 is the list of coins that had the maximum MOVE during the day**, not
> the coins that closed highest. A coin may collapse by EOD and still belong on
> that list. What is wanted is early entry into that move and a late but still
> profitable exit.

Every measurement surface in this repo scores where a coin **closes**:
`eod_return_pct` in the label store, the `EarlyCapture@top20` denominator and
its `winners_by_day`, the `top_gainer_model` tier ladder, the early-ranking
shadow. Those numbers are not wrong — they answer a question that was not asked.

This spec is the first measurement built on the move itself.

## What "an uptrend" means here

The ZigZag definition the operator confirmed on two of their own charts —
INJ 18–19 Aug (+11.98% over 37h, internal drawdown 1.11%, R² 0.816) and
TIA 19 Aug (+7.00% over 9h, drawdown 0.65%, R² 0.729):

> a run of `RUN_PCT` that ends when price gives back `GIVE_BACK_PCT` from its
> running peak.

**Duration is not bounded.** Multi-week trends are targets too, confirmed
explicitly. An earlier 48-bar horizon trained the model on a one-day question
while the catch report graded it against trends of any length; that mismatch was
the model's problem, not the target's.

### The label cannot be "the swing low"

That bar is identifiable only in hindsight, and a detector firing exactly on it
is not a thing that can exist. So the label is stated forward from **every** bar:
from here, does price gain `RUN_PCT` before giving back `GIVE_BACK_PCT`? Bars
where neither happens before the data runs out are `None` and dropped — folding
"nothing yet" into "no trend" teaches the model that stagnation looks like
absence, which is the flattering error.

One ordering detail carries the positive rate: the give-back is tested against
the peak **before** the current bar's high is folded in. Updating the peak first
would let a single bar both set a new high and be forgiven its own drawdown.

## Features: what the operator's charts actually show

Not a generic dump — the evidence a human reads off those two screenshots:
distance to MA25 and MA99 and their crossing, MACD histogram with its increment
and bars since it crossed zero upward, RSI and its 6-bar rise, volume against
its 20-bar mean, and the **base**: `bars_in_base` (consecutive hours within ±3%
of the current close), base tightness against ATR, current volume against the
base's median, and distance to the base's high.

`bars_in_base` was the missing one. Both charts show two to three days of flat
range before the move, and the earlier feature set could only measure how
*tight* a fixed window was, never how *long* the quiet lasted.

## Population

Every bar of every watchlist symbol with hourly klines — **957 975 rows, 99
symbols, 419 days** — and deliberately **not** the bot's event log. Every
negative result earlier in this work was measured on the bot's own entries and
therefore reflects whatever the upstream gates admitted; this target sits
upstream of the bot entirely, so those results do not transfer.

## Results

Split by time at 2026-04-14, null from 5 shuffled refits, alert budget 2% of bars.

| target | base | AUC | null ± sd | z | lift@0.5% | trends | caught | **random caught** |
|---|---|---|---|---|---|---|---|---|
| +5% | 0.0821 | 0.5250 | 0.4944 ± 0.0079 | 3.88 | **0.55×** | 4048 | 3.8% | **21.7–24.2%** |
| +10% | 0.0104 | 0.5122 | 0.4989 ± 0.0141 | **0.95** | 2.69× | 590 | 28.1% | **19.5–29.3%** |
| **+20%** | 0.0011 | **0.6921** | 0.4912 ± 0.0219 | **9.17** | **11.92×** | 94 | **95.7%** | 13.8–28.7% |

**The random-alert baseline decides two of the three rows.** A long trend is
easy to hit by accident: firing on 2% of bars at random lands inside a 100-bar
trend about 87% of the time from its length alone. Measured
(`_diag_catch_random_baseline.py`), that baseline is 13.8–29.3% depending on
target — so:

- **+5% is worse than random** (3.8% caught) and its top decile ranks *below*
  chance (lift 0.55×), despite a nominally significant AUC. Separation without a
  usable ranking, exactly the failure the continuation work was built to catch.
- **+10% is indistinguishable from random** on both checks independently:
  z = 0.95, and 28.1% sits inside the 19.5–29.3% band. Its 2.69× lift is noise.
- **+20% clears both**: z = 9.17, and 95.7% is more than three times the upper
  edge of the random band.

### A correction worth recording

Removing the time horizon was proposed as a fix and **made the +5% target
worse**: z fell 6.29 → 3.88 and top-decile lift 1.39× → 0.55×. For short targets
the horizon was part of the definition, not a mismatch. The change was right for
multi-week trends and harmful for intraday ones.

## What the +20% detector actually is

At the 2% budget, threshold `p >= 0.003`:

```
still ahead at the first alert   median 18.98%   (p25 12.07%, p75 26.52%)
how far into the move            median 40%
alerts fired                     5702, of which 98.4% outside any trend
```

Sample of the holdout catches (17 distinct symbols in the top 25, spread across
April–August — the result is **not** concentrated in a few names or one period):

| symbol | alert | trend | ahead | into |
|---|---|---|---|---|
| DOGSUSDT | 05-04 21:00 | 99.7% | **86.7%** | 13% |
| DOGSUSDT | 05-06 22:00 | 88.2% | 78.3% | 11% |
| APEUSDT | 04-24 13:00 | 81.3% | 76.4% | 6% |
| COTIUSDT | 07-27 19:00 | 89.3% | 53.2% | 40% |
| ORDIUSDT | 04-16 21:00 | 62.2% | 45.9% | 26% |
| BNTUSDT | 04-19 15:00 | 51.7% | 45.4% | 12% |
| AEVOUSDT | 08-01 12:00 | 29.2% | 28.3% | **3%** |

**It is an attention screen, not an entry trigger, and the distinction is not
cosmetic.** The largest catches fired at *low* confidence (p = 0.004–0.008), and
98.4% of alerts hit nothing: roughly 63 false alerts per catch. The mechanism is
not "the model pointed at a coin" but "the model spread alerts across the
regimes where large trends occur and covered them". That is real information —
the random control rules out duration as the explanation — but it is knowledge
of **where**, not of **when**.

## Not yet established

94 trends and roughly 300 positive rows in the holdout is thin. Three checks are
outstanding, any of which could overturn this:

1. **Stability across time cuts.** One split is one observation; 3–4 needed.
2. **A tighter operating point.** Top-0.5% carries lift 11.92× — if 1376 alerts
   catch nearly as many trends, the noise falls fourfold.
3. **Duration of the caught trends.** An unbounded +20% target can run for
   weeks. If the median caught trend spans weeks, "exit before it ends" becomes
   a different problem at a different scale.

## Verification

`test_trend_start_detector.py` — 20 tests, aimed at the ways this number could
be wrong while still printing: the label's race semantics and its unbounded
form, a bar being unable to forgive its own drawdown, feature rows provably
unchanged by appending later bars, `bars_in_base` responding to duration rather
than tightness, alerts credited only inside the trend window, remaining move
measured from the alert price, and the random baseline existing at the same
budget.
