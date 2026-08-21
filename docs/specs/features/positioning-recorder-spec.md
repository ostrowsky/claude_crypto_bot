# Positioning recorder: building the history the API will not give back

- **Slug:** `positioning-recorder`
- **Status:** COLLECTING since 2026-08-21 09:53 UTC. No decision, no gate, no
  model consumes it yet — and none should until it can be validated.
- **Truth-harness invariants:** TH-01 (base rate beside every ratio), TH-03
  (holdout split by time), TH-08 (negative results committed), TH-10 (numbers
  travel with the claim), TH-13 (never assert without proof)
- **Flags:** none. Recording only; `CryptoBot_PositioningRecorder` every 2h.
- **Rollback:** `Unregister-ScheduledTask -TaskName CryptoBot_PositioningRecorder`.
  The store is append-only and inert; deleting the task stops growth and breaks
  nothing.

## Why

Open interest, taker flow and long/short positioning describe something price
cannot: who is positioned, how heavily, and which side is paying to hold. The
bot uses none of it — every feature it computes comes from price and volume.

Binance serves all of it for **30 days only**. Measured against the API on
2026-08-20: `openInterestHist`, `takerlongshortRatio` and both long/short ratio
endpoints return HTTP 400 beyond 30 days; only `fundingRate` reaches back 420
days. Thirty days holds roughly two dozen +20% moves across the watchlist, which
is far too few to separate a relationship from a coincidence.

That is the whole problem, and it has exactly one solution: start writing them
down. A snapshot taken today can be scored eight hours later and never expires.
Everything not recorded today is gone permanently.

Funding — the one source deep enough to test properly — was tested, and it did
**not** help (AUC 0.606 on the narrow start-vs-middle question, no improvement
to the detector: see `trend-start-detector-spec.md`). That result is the reason
this spec makes no claim about the others. They are unproven, which is not the
same as promising.

## What is recorded

One row per watchlist coin per snapshot, into `files/positioning_history.jsonl`:

| field | meaning |
|---|---|
| `move`, `chg`, `rngpos` | the day's MOVE, its close change, and where price sits in the day's range |
| `vs_ma25` | extension above the 25h mean |
| `oi_1h`, `oi_4h`, `oi_24h` | open interest across three windows — 24h alone cannot say whether money is arriving now or leaving after arriving yesterday |
| `px_4h` | price over the same 4h window the OI class uses |
| `taker`, `taker_trend` | taker buy/sell ratio and where it sits against its own 6h mean |
| `retail`, `top` | all accounts vs largest accounts, long/short |
| `funding_bp` | funding in basis points |
| `flow` | the derived class, below |

Resolution after `--horizon` hours (default 8) adds `out_peak_pct`,
`out_close_pct` and `out_top50` — the last being membership in the day's top-50
by MOVE across the whole futures universe, which is the operator's own yardstick
rather than a plain return.

## The flow classes, and why OI needs price

Open interest rising says money arrived, not which side it took: every contract
has a long and a short. Price over the same window resolves it.

```
OI up   + price up     longs_opening    money betting on a rise
OI up   + price down   shorts_opening   money betting on a fall
OI down + price up     short_covering   the rise is bets against it closing,
                                        which ends when they are done
OI down + price down   longs_closing    holders leaving
```

`short_covering` is the distinction worth having: a rise carried by closures has
no new demand behind it, and reads identically to a healthy rally on a chart.

## The claim on record

Made 2026-08-21 09:53 UTC, deliberately falsifiable: the group **`longs_opening`
with `taker > 1.1`** — 26 of 87 coins that day — should beat the all-coin pool on
the share rising more than 3%.

If it does not, the signal does not work and that is recorded exactly as plainly
as the reverse. One day cannot settle it either way: the market had been rising
three days and nearly everything was up, which is precisely the condition under
which any group looks good. Only the SPREAD against the pool carries information,
and only once enough days — including falling ones — have accumulated.

## Reporting discipline

`report` prints the all-coin pool row **first**, refuses to score groups below
`--min-n` (default 30), and prints a warning while fewer than 20 days exist. This
is TH-01 applied mechanically: a ratio without its base rate is not evidence, and
the tool is built so the base rate cannot be omitted by accident.

## Clustering: available, and refusing to run

`_cluster_positioning.py` answers the natural next question — can the groups be
discovered rather than hand-cut? The flow thresholds (0.5%, 0.3%, 1.1) were
chosen by a human and use two features of thirteen; k-means uses all of them and
finds its own boundaries.

The trap it is built around: **clustering finds structure in feature space and
knows nothing about outcomes.** It will return beautifully separated clusters
whose forward returns are identical. So the verdict is not silhouette or inertia
— neither is computed — but:

> does the outcome spread between clusters exceed what random partitions of the
> same sizes produce on the same rows?

The null comes from shuffling the outcome column 200 times with the clusters
held fixed. A clustering that cannot beat its own shuffled null has described the
shape of a point cloud.

Three further guards, each earned by a past failure in this repository:

1. **It refuses below 20 days / 2000 rows.** Rows within one snapshot are not
   independent — the whole market moves together, so 87 coins in a snapshot is
   closer to one observation than to 87. Days are the unit.
2. **The split is by time**, clusters fitted on early days and applied to later
   ones. A cluster that only exists in its own training window is a description
   of the past.
3. **Several k are tried and the output says so.** A single k at z just above 2
   is a multiple-comparison artefact. Nine hypotheses died that way here already.

Expectation, stated before the data lands: probably negative. `CLAUDE.md` §7
records a multivariate logistic over 23 features scoring 0.60 in training and
**0.50 out of sample** — the signature of non-stationarity. Structure found in
August may simply not exist in October.

## Scheduling

`CryptoBot_PositioningRecorder`, every 2 hours, running
`run_positioning_recorder.cmd`: snapshot first, then resolve whatever has come
due. That order matters — `resolve` rewrites the whole store, so running it first
would operate on a file the snapshot is about to append to.

Both battery flags are **off**. §5 of `CLAUDE.md` records every CryptoBot task
being created with them on, which meant that on battery the weekly pipeline
silently never ran: eleven days of an empty hypothesis queue with no explanation
and `LastTaskResult 2147946720` as the only trace. The job log is appended, never
truncated, because that log is the evidence the task actually ran.

## Verification

`test_positioning_recorder.py` — 24 tests. The ones that matter: OI is never
classified without price, small moves are called flat rather than forced into a
direction, the store is append-only, snapshot precedes resolve in the scheduled
job, failures are written to the log rather than swallowed, the report cannot
omit the pool row, clustering refuses below its threshold and counts days rather
than rows, its verdict comes from the shuffled null, and the spread metric is
weighted so a one-row cluster with a wild return cannot carry it.

## What would make this worth using

A class whose outcome spread beats its null across at least 20 days including
falling ones, holding on a temporal holdout. Until then this is a recorder, and
nothing downstream should read it.
