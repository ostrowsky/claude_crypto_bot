# Trend-start detector: catching the move, not the close

- **Slug:** `trend-start-detector`
- **Status:** evidence 2026-08-20 — POSITIVE on WHICH coin; NEGATIVE on WHEN at 1h AND 15m; nothing deployed
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

## The three outstanding checks — all now run

1. **Stability across time cuts.** Repeated at 55/45, 65/35, 70/30 and 80/20.
   AUC 0.600 / 0.662 / 0.692 / 0.686 and catch 90.4% / 91.6% / 95.7% / 97.7% —
   stable, and AUC *rises with training size* and plateaus, which is a learning
   curve rather than a boundary artefact. **Do not read the z column of that
   table**: it used 3 shuffled seeds, so the null's own sd is unstable and z
   swung 1.93-15.34 across near-identical AUCs. Stability lives in AUC and catch
   rate, not in z.
2. **A tighter operating point — passed, and it is the one to use.**

   | budget | alerts | caught | still ahead |
   |---|---|---|---|
   | 0.5% | 1 425 | 86.2% | 14.51% |
   | 1.0% | 2 851 | 91.5% | 16.30% |
   | 2.0% | 5 702 | 95.7% | 18.98% |
   | 5.0% | 14 257 | 97.9% | 21.03% |

   Cutting alerts fourfold costs 9 trends of 90 and drops the noise from ~63
   false per catch to ~16. **0.5% is the operating point**, not 2%.
3. **Duration — the premise was wrong.** Caught trends run p25 5h, median **7h**,
   p75 11h, max 37h, and **0%** last longer than a week. At a 2% give-back a
   weekly wave is cut into a dozen segments, so multi-week trends do not exist in
   this population at all. Wanting them means widening the give-back, which is a
   decision about the target rather than about the model.

   This also refutes the concern that "exit before it ends" would be a
   position-management problem at weekly scale. It is not.

## The exit, measured while the entry question was still open

`_backtest_trend_exit_rule.py`, 76 episodes over 36 symbols, entering at the
trend start (generous — it isolates the exit from the detector's timing):

| policy | median | capture | exited before the peak |
|---|---|---|---|
| ideal (exit at the peak) | 19.10% | 1.00 | 0% |
| **2% give-back from the peak** | **15.13%** | **0.78** | 7% |
| slope over 6h ≤ 0 | 10.85% | 0.65 | 17% |
| no new high for 6h | 12.27% | 0.62 | 14% |
| below MA12 | 12.05% | 0.60 | 11% |
| slope over 24h ≤ 0 | 8.98% | 0.46 | 0% |

The operator proposed selling on a plateau or a turn down. **The simplest rule —
wait for a 2% give-back — beats every plateau rule** on capture, on median gain
and on premature exits. The `exited before the peak` column says why: on a
7-hour trend the hourly slope flattens several times, and a plateau rule sells
on the first pause.

For contrast, the live bot captures 19.8% of the remaining move with a median
trade of −0.50%. The difference is not the exit rule; it is what is being exited
from.

## Verification

`test_trend_start_detector.py` — 20 tests, aimed at the ways this number could
be wrong while still printing: the label's race semantics and its unbounded
form, a bar being unable to forgive its own drawdown, feature rows provably
unchanged by appending later bars, `bars_in_base` responding to duration rather
than tightness, alerts credited only inside the trend window, remaining move
measured from the alert price, and the random baseline existing at the same
budget.

## The earliness question, answered separately — and NEGATIVELY

The detector above finds trends. It does not find their beginnings: the first
alert lands a median **40-48% into the move**, while a random alert placed
inside a trend lands at ~50%. That is close to no timing advantage at all, and
it is not a shortfall of the model — the forward label ("from here there is
+20% ahead") is satisfied by a bar in the middle of a move exactly as well as by
one at its start. The model was never asked for the start.

Three independent attacks on that, all failing on timing while agreeing with
each other:

| approach | caught | still ahead | **into the move** |
|---|---|---|---|
| forward label | 84.2% | 14.90% | **48%** |
| start label, 1h window | 55.8% | 17.40% | **46%** |
| start label, 2h window | 68.4% | 17.47% | **40%** |
| start label, 3h window | 83.2% | 17.28% | **43%** |
| start label, 6h window | 90.5% | 17.47% | **45%** |
| forward label, alerts banned at RSI ≥ 65 | 21.1% | 21.35% | 32% |
| forward label, alerts banned at RSI ≥ 55 | 4.2% | 22.68% | 20% |

**The start label raises ranking sharply and moves the timing not at all.**
AUC 0.694 → 0.822-0.841 and lift 12.5× → 21-61×, stable across every window
width — yet `into%` stays flat at 40-46% whether the label calls the first hour
of a trend positive or the first six. Trained *exclusively* on opening bars, the
model's highest-scoring bars are still mid-move ones.

The RSI constraint prices the same wall from the other side. Banning alerts
above RSI 65 removes only **7.2%** of test bars from eligibility and costs
**three quarters** of all catches — the score is concentrated almost entirely on
bars whose momentum has already turned. Buying `into%` down to 20% costs 96% of
the catches.

### What this means, stated as narrowly as the evidence allows

On **hourly** bars with these features, the start of a strong trend is not
separable from its middle. This is a statement about resolution as much as about
features: the median caught trend runs **7 hours**, so its opening hour is one
observation in seven and has almost no shape to describe. The same experiments
on 15m bars would give that trend 28 observations instead of 7.

What survives intact: the detector answers **which coin** with AUC 0.82-0.84 and
lift 21-61×, stable across four label widths and four time cuts. It does not
answer **when**.

### A live confirmation, and its limits

On 2026-08-19 the bot sat out a market-wide rally: every gate rejected it
(`ml_proba` 0.04-0.19 against a 0.22 floor on eight coins, the bandit preferring
SKIP at ucb 1.72 vs 1.20, `trend_quality` on RSI 77.8 > 76, an explicit "late 1h
continuation" guard). Scored on the same hours, trained only on data before
08-18, the detector fired 9 alerts of 93 symbols at the 0.5% budget:

```
ETHUSDT  +8.84% since alert (peak +11.88%)    WLDUSDT  +8.78% (peak +11.19%)
NEARUSDT +5.66%   SNXUSDT +2.00%   CRVUSDT +1.57%   LDOUSDT +1.46%
BNTUSDT  +1.15%   ARBUSDT -0.45%   ENSUSDT -0.70%
median +1.57%, 7 of 9 positive; six of the nine fired at 15:00, four hours
before the rally accelerated
```

**Nine alerts on one evening is an anecdote, not evidence** — on a day when
nearly everything rose, 7-of-9 would occur by chance often enough to prove
nothing. The statistical case is the backtest, not this. And of the twelve coins
the operator was watching, only ETH fired; INJ ran 4.24 → 4.61 and scored
0.0028. That is a miss and is recorded as one.

## A run that reported the wrong label — recorded on purpose

A patch adding the start label matched on an anchor that had since drifted, so
`start_bars` was defined and never called. The patch script printed
"start-label mode wired" without verifying. Three runs then printed
`label: this bar is within 6h of the start` above numbers produced by the
forward label; the only clue was **identical AUC to four decimal places across
supposedly different labels**.

`test_trend_start_detector.py` now asserts the function is called and its result
used. The failure is kept in this file because a spec that records only what was
intended is the failure mode the whole document exists to prevent.

## Resolution was not the constraint either — 15m, and it is WORSE

The 1h negative result carried one honest escape: a median caught trend runs 7
hours, so its opening hour is one observation in seven and may simply have had
no shape to describe. 419 days of 15m klines for the whole watchlist were
backfilled to close that escape (`_backfill_watchlist_15m_419d.py`, 101 symbols,
40 224 bars each, 747 MB). The escape is now closed, in the unhelpful direction.

| run | rows | AUC | lift@0.5% | alerts | caught | ahead | **into** |
|---|---|---|---|---|---|---|---|
| 1h reference | 958 452 | 0.8379 | 22.2× | 1 419 | **59.6%** | 17.07% | **42%** |
| 15m, native windows | 3 912 901 | 0.8674 | 20.4× | 5 724 | 42.2% | 14.12% | **52%** |
| 15m, windows ×4 | 3 877 261 | 0.8340 | 18.6× | 5 677 | 37.2% | 15.47% | **48%** |

The alert budget is a share of BARS, and 15m has four times as many, so both 15m
runs fired **four times more alerts in absolute terms** — and still caught fewer
of the same trends, and landed later inside each. Native windows let the
features react 4× faster in wall-clock time; scaled windows (×4) give them the
same physical horizon the 1h run had. Neither helps.

Worth noting against §0a rule 11: **15m-native has the best AUC of the three
(0.8674) and the worst timing (52%)**. Separation improved while the thing that
matters got worse — the same pattern that killed the +5% target earlier in this
spec, and another reason AUC cannot be the acceptance metric here.

### The target was held fixed, on purpose

A finer grid does not resample the ZigZag population, it replaces it: at a 2%
give-back the 15m grid finds **99** trends where 1h finds **342**, because a
give-back is now detected on intra-hour lows the hourly bar hides, so runs are
cut before reaching +20%. The survivors are the unusually smooth ones (median
10.8h vs 7.0h). Widening the give-back to 3% restores the count (351) and still
not the population — those run a median 18.8h.

No parameter makes them identical, so the experiment does not try. `--trend-grid`
holds the target on the 1h grid — the same 342 trends every committed number was
scored against — and only the detector's input resolution varies. The universe is
likewise pinned to the 99 symbols the 1h runs used (`--match-1h-universe`); the
15m backfill covers 101, and letting BAKE and MKR in would have put a population
change inside a resolution comparison.

## Verdict on earliness: it is the features, not the model, the label, or the grid

Five independent attacks, all agreeing:

| attack | into the move |
|---|---|
| forward label, 1h | 48% |
| start label, 1h, windows 1/2/3/6h | 40-46%, flat in the window width |
| RSI < 65 constraint | 32%, at the cost of 75% of all catches |
| RSI < 55 constraint | 20%, at the cost of 96% |
| **15m resolution, native and scaled** | **52% and 48%, with 4× the alerts** |

**On price and volume alone, the start of a strong trend is not separable from
its middle at any resolution tested.** The information that identifies a trend
arrives with the trend's own momentum; asking for it earlier does not relocate
the signal, it discards it.

What survives, unchanged and strong: the detector answers **which coin** —
AUC 0.82-0.87, lift 18-61×, stable across four time cuts, four label widths and
two timeframes. It does not answer **when**.

The next honest step is not another model or another grid. It is data of a
different kind — order book depth and imbalance, funding rates, liquidation
flow, exchange inflows — none of which this repo currently records. That is a
collection task before it is a modelling one (§0 rule 1).

## Engineering defects found while building this, recorded so they are not repeated

Each of these would have produced either a crash or quietly wrong numbers:

- **The window scaling reached only half of each range.** Replacements ran with
  `n=1`, so `lo24` was scaled and `hi24` left at 23 bars; `base_range_24` would
  have divided a `24*sc`-bar high by a 24-bar low. A test now scans the whole of
  `feature_table` for surviving literal lookbacks.
- **An `int8` label overflowed inside `auc()`** — `len(y) - sum(y)` with a numpy
  int8 sum. The memory win was in the 22 float columns; the label is int64 now.
- **A numpy array was tested for truth** (`if ps:`) in the null loop.
- **The dict-per-row dataset needed 6.8 GB against 2.3 GB free** and could not
  run at 15m at all. Columns brought it to ~700 MB.
- **`min_duration_bars` had to scale with the timeframe** — `4` means "four
  hours" on 1h and would have meant one hour on 15m.

### The refactor was proven neutral rather than assumed to be

Converting the dataset to columns changed the 1h AUC from a remembered 0.6944 to
0.6725, and `float32` storage was the obvious suspect. It was wrong: float64
reproduced 0.6725 **exactly**, digit for digit. Running the committed
pre-refactor script (`git show 10fd3d4:...`) on the same cache also returned
0.6725 / 13.62× / 87.2% / 45% — identical. The refactor is neutral; the 0.6944
belonged to an earlier snapshot of a 1h cache the live bot keeps appending to.
A remembered number from a moving dataset is not a baseline.

## Alternative data: what exists, what was tested, what must be collected

The 1h and 15m negatives pointed outside price and volume. Every alternative
source Binance serves was probed for how far back it actually reaches
(2026-08-20), because a source that cannot cover the max period cannot be
validated to this project's standard:

| source | history | step | status |
|---|---|---|---|
| **funding rate** | **420 days** | 8h | testable now; backfilled for 93 symbols |
| open interest | 30 days, HTTP 400 beyond | 5m-4h | collection only |
| taker buy/sell volume | 30 days | 1h | collection only |
| long/short ratios (3 kinds) | 30 days | 1h | collection only |
| order book depth | **none — live snapshot only** | — | collection only |

Thirty days holds roughly two dozen +20% trends; a result on that is
indistinguishable from noise. So those sources are **untestable today** and the
only useful action is to start accumulating them — the §0 rule 1 case, where
work blocked by missing data becomes data-collection work.

### Funding: a real narrow signal that does not translate

Restricted to bars already INSIDE a +20% trend, asking only "is this the first
6h or later", funding features alone score **AUC 0.6059, null 0.5238 ± 0.0303,
z = 2.71** on a time split. The medians move the way the hypothesis predicted:

```
                    START    MIDDLE   NON-TREND
funding (bp)        0.212     0.500       0.493
funding_mean6       0.020     0.314       0.266
```

Non-trend funding matches the MIDDLE, not the start — so the separation is
"trend beginnings are unusual", not merely "trends are unusual". That is the
first thing in this whole line of work to score above null on the start-vs-
middle question at all.

It does not survive contact with the detector. Matched universes, both arms on
the same 91 symbols and the same 73 holdout trends:

| | AUC | lift@0.5% | caught | ahead | into |
|---|---|---|---|---|---|
| price only | 0.8514 | 23.14× | 52 (71.2%) | 17.28% | 47% |
| price + funding | 0.8439 | 25.87× | 53 (72.6%) | 17.40% | 40% |

AUC slightly **down**, one extra trend caught, and `into` 40% against 47% — the
same price-only configuration has printed 42%, 45%, 46% and 47% across data
snapshots, so 40% sits inside the observed run-to-run spread. **Funding does not
measurably help the detector.** The narrow signal is real and does not convert
into earlier alerts.

### Two attractive numbers destroyed by their own base checks

Recorded because both looked like findings and cost nothing but the check:

1. **`4h_leader_watch` firing at `into` 13%** — the best earliness figure this
   project has produced, and computed on **n = 3**. The rule catches 3 of 342
   trends (0.9% recall) and fires 0.62 times per symbol-year. Its forward edge
   is real (44% of fires exceed +5% against 21% for a random bar) but it is a
   rare high-precision alert with negligible coverage. Two components are dead
   weight: dropping the 4h context score entirely changes nothing (71 fires vs
   78, identical metrics), and the bare strength gate — up 10% today on 3x
   volume — beats the full rule on forward move (6.06% vs 4.19% median) while
   entering at `into` 65% instead of 13%.
2. **Funding scoring AUC 0.846 on start-vs-middle** — from a RANDOM split.
   Funding posts every 8h, so every bar in an 8h window of one symbol carries
   the same value and adjacent hours of one trend land on both sides of the cut;
   the model memorises (symbol, funding value) → label. A time split gives
   0.6059. The leakage was worth 0.24 of AUC (TH-03).

A third confound was caught in the A/B itself: 8 of 99 symbols have no funding
history, and dropping them silently removed 21 trends from the denominator,
turning a population change into a fake +13pp of recall. `--funding-universe`
now forces both arms onto the same symbol set (TH-04).
