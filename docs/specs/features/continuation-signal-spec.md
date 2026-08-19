# Continuation signal: does the information exist at decision time?

- **Slug:** `continuation-signal`
- **Status:** evidence 2026-08-19 — ranking POSITIVE, exit policy NEGATIVE, no behaviour change
- **Created:** 2026-08-19
- **Truth-harness invariants:** TH-01 (a ratio without its base rate is not
  evidence), TH-03 (holdout, split by time), TH-06 (validate on the population
  the bot actually sees), TH-10 (a negative result is committed with the numbers
  that killed it)
- **Flag:** none — nothing ships from this
- **Rollback:** не применимо

## Which goal this serves

**Goals 2 and 3, jointly.** They are one question asked at two moments.

Goal 2 wants entry as early in the move as possible; goal 3 wants exit only just
before the move ends. Both need the same predicate — *does this move continue
from here* — evaluated before the move for entry and at exit time for exit. The
current state of each:

| goal | where it stands |
|---|---|
| 2 — early entry | move-relative lead **0.02** of 1.0 |
| 3 — exit before the end | 19.8% of the remaining move captured; every fixed trail width is worse than the current exits |

`_backtest_exit_timing.py` closed the threshold-tuning route: a width is a
constant, and the question is conditional. That is why this experiment asks
whether the conditional information exists at all, rather than proposing another
policy.

## What is asked, precisely

At a bar the bot is holding through, using only bars up to and including that
one: does price gain `+up_pct` before it loses `dn_pct`, within the next
`horizon` bars?

Phrased as a **race**, not a terminal return, because a stop experiences path
order and a mean does not. A window where neither side is touched is labelled
`None` and dropped — folding "nothing happened" into "the move ended" is a
different claim, and the flattering one for any model that learns to predict
stagnation.

## Population

Every bar from entry to entry+48, for **every** entry in the event log — not
only the bars actually held, and not only winner-days.

Both restrictions were rejected for the same reason. Restricting to entry..exit
measures the current policy's own window and calls it the world; restricting to
winner-days conditions the sample on the outcome the policy cannot know, which
is how this experiment's predecessor produced "an 8% trail triples capture" —
a result live trading had already rolled back in June (TH-06).

## Why the exit-time features had to be reconstructed

The exit event records `bars_held`, `entry_price`, `exit_price`, `mode`,
`pnl_pct`, `reason`, `sym`, `tf`, `trail_k`, `ts` — and **no market state at
all**. Every indicator the entry event carries (`adx`, `rsi`, `slope_pct`,
`vol_x`, `ml_proba`, `macd_hist`, …) is absent at exit.

So a predictor cannot be built from the logs: at the moment of decision there is
nothing to feed it. Features here are recomputed from hourly klines, which is
legitimate because they were available to the bot at that instant.

When this was written, instrumenting the exit event looked like a hard
prerequisite for shipping anything (§0 — a change blocked by missing logging is
a logging-fix-first task). Step 2 then found there is nothing to ship, so the
instrumentation is **demoted to measurement hygiene**: still worth doing, so the
next person asking this question does not have to rebuild the state from klines
again, but no longer blocking anything.

The hourly cache also had to be repaired first: `_1h_365d` stops on 2026-06-20
while `_1h` reached today for only a quarter of symbols, so "use whichever file
covers this trade" would have made cache staleness a selection rule on the
sample. Both are now merged on timestamp, and `_backfill_klines_history.py
--days 200 --tf 1h` refreshed 99 of 102 watchlist symbols (`MKRUSDT`,
`BAKEUSDT`, `SNTUSDT` return empty — candidates for the phantom filter).

## Controls

An AUC on its own is not evidence, so four things travel with it:

1. **A null with width.** The null is estimated by refitting on shuffled labels
   across several seeds, not once. The first run of this script printed a single
   shuffled 0.4851 beside a real 0.5124 and treated the gap as signal — but the
   null's own spread was larger than the 0.012 being claimed. A label passes
   only if it clears the null mean by more than 2 sd.
2. **Lift at the operating point**, with the base rate beside it. The same first
   run had a CI that formally excluded 0.5 while top-decile lift sat at
   **0.99×** — the ranking ordered nothing usable, and the verdict logic called
   it a success anyway. The verdict now requires separation **and** lift ≥ 1.10.
3. **A bootstrap clustered by trade.** 48 bars of one trade are one observation
   with 48 rows; a row-level interval would be far too narrow.
4. **A split by time on day boundaries**, never at random.

## The label grid, and why there is one

| label | question it asks |
|---|---|
| +2 / −2 in 12h | symmetric — nearly a coin flip on these names |
| +5 / −2 in 12h | asymmetric — a real move against a small give-back |
| +5 / −3 in 24h | the same, over a day |
| +10 / −3 in 24h | only large continuations count |
| +3 / −1.5 in 6h | short-horizon, tight give-back |

Concluding "no signal" from a single parameterisation would be the mirror image
of concluding "signal" from one. A trailing policy cares about the asymmetric
race — keeping a large move while giving back little — so the symmetric label is
the least relevant of the five even though it is the most natural to write.

Each row carries its own base rate, because these are different questions and
their positives are not comparable.

## Verification

`test_continuation_signal.py` — 19 tests. The failure they exist to catch is not
a crash but an off-by-one in the feature window that lets a bar see its own
future, producing a high AUC, a plausible story and a policy built on nothing:

- the label is a race, honours asymmetric thresholds, reads an
  ambiguous both-sides-in-one-bar as a stop fill, and returns `None` rather than
  0 when neither side is touched;
- appending future bars does not change a computed feature row, and the caller's
  slices (`b[max(0, k - WARMUP):k + 1]` for features, `b[k + 1:k + 1 + horizon]`
  for the label) provably do not overlap;
- no day appears on both sides of the split;
- `lift_at` cannot return a precision without its base rate;
- the bootstrap resamples trades rather than rows;
- the null is refit across seeds, and the verdict requires both criteria;
- the population is not restricted to winner-days.

## Result

The verdict criteria below were fixed before any number was seen.

```
population: bars 0..48 after every entry, 4 344 trades, 99 symbols
split by time at 2026-06-28   train 110 days / test 48 days

label               base     AUC    95% CI (trade)   null mean+-sd  lift@10%     z
+2 / -2  in 12h    0.507  0.5124   [0.5002,0.5250]   0.5032+-0.0068    0.99x  1.36
+5 / -2  in 12h    0.168  0.5291   [0.5082,0.5488]   0.4993+-0.0155    1.27x  1.92
+5 / -3  in 24h    0.312  0.4869   [0.4681,0.5065]   0.5010+-0.0113    0.91x -1.25
+10 / -3 in 24h    0.097  0.5822   [0.5411,0.6223]   0.5036+-0.0244    1.55x  3.22
+3 / -1.5 in 6h    0.242  0.5238   [0.5089,0.5367]   0.5007+-0.0090    1.10x  2.56
```

**The signal grows with the asymmetry and the size of the move.** A symmetric
±2% race is indistinguishable from a coin flip; "a large continuation against a
small give-back" is not. That is the shape a trailing policy needs, and it is
the opposite of what the most natural label to write would have shown.

`+5 / −3 in 24h` landed *below* the null (AUC 0.487). That is not a failure of
the run — it is evidence the pipeline does not manufacture positives.

### Multiplicity

Five labels were tested, so five chances. Under Bonferroni the strongest
(z = 3.22, p ≈ 0.0006) survives at p ≈ 0.003; `+3 / −1.5 in 6h` (z = 2.56,
p ≈ 0.005) becomes p ≈ 0.026 — **not confirmed, merely not refuted**. Only the
`+10 / −3 in 24h` label is treated as a finding.

### The clock was noise, not the carrier

`hour_utc` was the top feature of the first run, which would have made this a
time-of-day effect wearing a trend-continuation costume. Refitting without it
makes the signal **stronger** — z 3.22 → 4.57, lift 1.55× → 1.54× — so the clock
was absorbing variance, not supplying the answer. It is dropped.

### The volatility tautology, and why it does not explain the result

`atr_pct` is the top feature, and the label is "+10% within 24h". A coin that
routinely swings 10% satisfies that label more often whether or not its current
move continues — the same shape of defect as `tg_return_since_open` scoring 0.99
on "was today a top gainer" (TH-02). So the model was refit twice:

| feature set | AUC | null ± sd | z | lift@10% |
|---|---|---|---|---|
| volatility only (`atr_pct`, `range_pct`) | — | — | 3.20 | **1.11×** |
| everything except volatility | 0.5790 | 0.5042 ± 0.0165 | **4.53** | **1.60×** |
| all features | 0.5822 | 0.5036 ± 0.0244 | 3.22 | 1.55× |
| all features, no `hour_utc` | — | — | 4.57 | 1.54× |

**Volatility carries the separation but ranks nothing usable.** It reproduces
the AUC almost exactly and then delivers a top decile barely above the base
rate. The usable ranking comes from trend shape — `dist_ema50`,
`dd_from_run_max`, `rsi`, `vol_ratio`, `pnl_since_entry` — which alone beats the
full model. This is the single most important line in this file: had only the
AUC been reported, the honest conclusion would have been "we rediscovered
volatility".

Stated plainly: at a base rate of 9.7%, the top decile of the shape-only
ranking runs at **15.5%**.

## What this does NOT establish

- **It is not a policy.** A ranking with 1.6× lift on a 9.7% base rate is an
  ordering, not an exit rule. Fixed-width trails also looked good until they
  were replayed on every trade and charged for it; the same replay is the next
  step and it may well kill this too.
- **It is not measured on the live decision surface.** These features were
  reconstructed from klines. Live, the exit path has none of them, so any
  deployment is gated on instrumenting the exit event first.
- **1.6× is modest.** It says the information exists, not that it is enough.
  Goal 3 currently captures 19.8% of the remaining move; nothing here yet says
  how much of the other 80% this could recover.

## Step 2 — the replay: the ranking does NOT become an exit policy

`_backtest_continuation_exit_policy.py`. Hold from entry, score P(continues) at
each hourly bar, leave the first time it falls below a threshold; time stop at
48 bars. Model trained on the earlier days with an embargo **on the bar, not
just on the trade** — a trade entered before the cut runs on for 48 bars, so its
later bars sit on the test days. (The embargo changed the answer by 0.007pp, so
it mattered for correctness rather than for the conclusion.)

Compared against the actual exits **of the same 905 trades**, not against all
4 174: the test window is a different market, and comparing across windows is
the error that produced a fake "СТАЛО ХУЖЕ" after the outage.

```
policy                         n    median      mean       win      beats    sum pnl   hold
ACTUAL exits (bot today)     905    -0.499    -0.335     35.7%       0.0%     -303.4      -
continuation p<0.03          905    -0.966    -0.700     39.1%      43.0%     -633.1     47
continuation p<0.05          905    -0.848    -0.644     38.7%      43.9%     -582.6     47
continuation p<0.08          905    -0.391    -0.598     33.6%      46.1%     -540.7     12
continuation p<0.12          905    -0.257    -0.233     34.9%      55.7%     -210.7      2
continuation p<0.20          905    -0.143    -0.152     39.2%      60.0%     -137.8      1
fixed trail 1.5%             905    -1.136    -0.437     27.1%      37.9%     -395.8      6
fixed trail 3.0%             905    -1.660    -0.509     30.9%      35.7%     -460.9     19
fixed trail 8.0%             905    -0.934    -0.458     39.1%      43.4%     -414.5     47
hold to 48 bars              905    -0.967    -0.706     38.9%      43.0%     -638.5      -
```

Two thresholds beat the actual exits on median **and** head-to-head — `p<0.20`
at −0.143 vs −0.499, winning 60% of pairs. That looks like the result this file
was built to find.

**The duration control refutes it.** Random exits drawn from the *same*
holding-time distribution give a median in **[−0.148, −0.116]**, which brackets
the policy's −0.143. At equal holding time, the ranking adds nothing. And the
`hold` column says what the "policy" actually learned: `p<0.20` holds a median
of **one bar**, `p<0.12` two. The threshold did not learn when to leave; it
learned to leave immediately.

This is why the control existed. Without it the honest-looking conclusion would
have been "the continuation model improves exits by 0.36pp per trade".

## Why the ranking is real and still useless here

Both are true and they do not conflict. The model orders bars by P(continue)
better than chance (z = 4.53). An exit policy needs something else: the bar
where continuation *stops*, which is one specific bar per trade, not an
ordering over all of them. A 1.60× lift on a 9.7% base leaves the top decile
wrong 84.5% of the time — enough to rank, nowhere near enough to time a single
irreversible decision.

## The finding underneath: the loss is not at the exit

Every exit rule tested — the bot's own, five thresholds, three trail widths,
hold-to-48 — lands between −0.14% and −1.66% median. The best is the one that
holds for one hour. That pattern only has one explanation, and
`_diag_forward_drift.py` confirms it directly on all 4 392 trades:

```
bar      n    median      mean    win%
  1   4389    -0.078    -0.057    42.7
  4   4384    -0.289    -0.199    40.7
 12   4378    -0.549    -0.260    39.8
 24   4371    -0.665    -0.331    41.0
 47   4359    -0.799    -0.384    43.0
```

**The median trade is already negative one hour after entry, and gets worse
monotonically.** Win rate sits near 40% at every horizon. An exit rule chooses
*when* to realise a path; it cannot change the path. On a population with
negative drift from bar 1, the optimal exit converges on "immediately", which is
the same statement as "do not enter".

This agrees with `portfolio-alpha` (negative on every window: 30d −6.24%,
60d −14.56%, 90d −11.84%, 180d −15.97%) and with `weekly-steering-pair`
(early-alert lift 1.04× vs all-alert 3.96× — the bot confirms moves rather than
predicting them), both reached by unrelated routes.

**Goal 3 is not where the loss is.** Work aimed at capturing more of the
remaining move is optimising the smaller term. The binding constraint is which
candidates get entered at all — goals 1 and 2.

## Next step, in order

Step 2 is done and it came back negative, which changes the order of what is
left.

1. **Nothing more on exits until entries change.** With median drift negative
   from bar 1, an exit improvement is bounded by a term that is not the binding
   one. Re-open this only if the entered population stops losing from the first
   hour.
2. **Instrumenting the exit event stays worth doing**, but as measurement
   hygiene rather than as a prerequisite for a policy: without market state at
   exit, the next person to ask this question also has to reconstruct it from
   klines. Demoted from blocking to useful.
3. **The live question is goals 1 and 2** — which candidates get entered at all.
   The early-ranking shadow is accumulating days toward a verdict; it addresses
   exactly this and needs no new hypothesis.
