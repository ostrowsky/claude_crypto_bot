# Global label universe (TH-03 / TH-05)

- **Slug:** `global-label-universe`
- **Status:** shipped 2026-08-17
- **Created:** 2026-08-17
- **Parent:** [`immutable-label-store`](immutable-label-store-spec.md)
- **Unblocks:** [`top-gainer-immutable-training-labels`](top-gainer-immutable-training-labels-spec.md),
  [`north-star-immutable-labels`](north-star-immutable-labels-spec.md),
  [`day-grouped-training-split`](day-grouped-training-split-spec.md)
- **Truth-harness invariants:** TH-03 (label provenance), TH-04 (comparable
  denominators), TH-05 (a metric must know what it does not know)
- **Rollback:** the daily-resolution records are additive and tagged; deleting
  them restores the watchlist-only store exactly

## Problem

The immutable label store holds only watchlist symbols, so the tiers it can
build are "top-N **within the watchlist**". The original labels are global —
`backfill_top_gainer_dataset` ranks **all USDT pairs** from
`/api/v3/ticker/24hr` and marks the top 5/10/20/50. Two consequences already
measured and blocking:

- `train_top_gainer` on watchlist-scoped immutable labels collapses the tiers:
  `top20` and `top50` are byte-identical on the holdout, which would
  mis-calibrate the live tier ladder's fixed thresholds at once;
- the North Star's two published values carry different denominators and are
  marked non-comparable, so the immutable one cannot replace the leaky one.

Both are the same missing thing: the store cannot reproduce a global rank.

## Change

Add a **global ranking tier** to the store, built from daily klines.

```
universe   all USDT pairs (currently 734), plus every symbol seen in
           top_gainer_dataset (469 valid) to recover delisted ones
interval   1d — one request per symbol returns 1000 days, far past the window
fields     open/high/low/close, eod_return_pct, max_move_pct, qualifies_move5
missing    anchor_ts / early_deadline_ts — intraday crossing times do not exist
           in a daily bar and are NOT approximated
```

Records carry `resolution: "1d"`; hourly records carry no such field and are
read as `"1h"` through `label_store.resolution_of`. The hourly builder is left
byte-identical and `BUILDER_VERSION` is NOT bumped — see the findings below.
The daily builder gets its own `DAILY_BUILDER_VERSION`.

`MIN_BARS_COMPLETE = 20` is a count of **hourly** bars; a daily record has one
bar and would be `complete=False`, which would drop it from every consumer. The
daily builder decides completeness from whether the UTC day has closed, not from
a bar count — the same question, asked correctly for the resolution.

### What must not change

`weekly_steering` computes MoveEvents from `early_deadline_ts`, which daily
records do not have. It already filters to the watchlist, so the new records are
outside its scope — but that is currently an accident of the filter, not a
stated invariant. It becomes explicit: **the MoveEvent path consumes
`resolution == "1h"` records only**, and a test asserts a daily record cannot
enter it. A `None` deadline silently classifying every alert as early or late is
exactly the failure this guard exists to prevent.

## What must be reported, not hidden

**Survivorship bias is real and cannot be fully removed.** The universe list is
fetched today, so pairs delisted since a past day are absent from that day's
reconstructed ranking, and a coin that was a genuine global top-20 gainer before
being delisted will not appear. Unioning in the 478 symbols the dataset itself
saw recovers those the exchange still serves klines for; the rest are
unrecoverable. Every artifact built on these labels states the resolved-symbol
count per day and the count that failed to resolve — the metric must know what
it does not know (TH-05).

**The reconstruction is not the snapshot.** The original label ranks a
**rolling 24h** window at the moment of the snapshot; this one ranks a **closed
UTC day**. Those are different windows, and a day's top-20 will differ between
them even with a perfect universe. The point is that only the second one can
serve as ground truth for features computed during the day — not that it
reproduces the first.

## Verification

`test_global_label_universe.py`:

1. a daily record is `complete` when its UTC day has closed, and incomplete for
   today, regardless of bar count;
2. a daily record carries `resolution == "1d"` and `anchor_ts is None` — the
   intraday fields are absent, never zero or approximated;
3. the MoveEvent path refuses daily records, so `weekly_steering` cannot consume
   one even if the watchlist filter is removed;
4. hourly and daily records for the same `(symbol, day)` do not overwrite each
   other, and the store's immutability still raises on a conflicting rewrite;
5. `immutable_labels.tier_labels` over a global universe restores tier
   separation — `top20` and `top50` are no longer the same label;
6. per-day resolved and unresolved symbol counts are emitted with the build.

**Maximum-period evidence**: re-run `_backtest_immutable_training_labels.py` and
`_backtest_immutable_ns.py` on the global store, and publish tier base rates
against the original global ones (1.49 / 3.15 / 6.31 / 15.34%). The immutable
base rates should land near those, because the same rule over the same universe
should mint a similar number of winners. A large gap means the universe is not
reconstructed, not that the label is better.

**Shadow/canary: не применимо** for the store build — it writes new records and
changes no behaviour. Flipping the consumers is the behaviour change, and each
already has its own flag.

## Result

```
universe fetched     734 USDT pairs (live + 469 valid dataset symbols)
resolved              530     failed 204 (no klines in the 240-day window)
records             19 502 hourly (watchlist)  +  88 703 daily (global)
per-day universe    ~440-497 symbols, 240 days (2025-12-20..2026-08-16)
```

**Tier collapse is gone.** Under the watchlist-only store `top20` and `top50`
were byte-identical on the holdout; over the global universe every pair is
distinct:

```
pair            identical rows   positives      (was, watchlist-only)
top5  / top10          98.46%    386 / 713           99.56%
top10 / top20          97.92%    713 / 1156          99.95%
top20 / top50          97.37%   1156 / 1717         100.00%
```

**The floor is no longer needed and the default drops to 0.0.** It existed only
because a rank inside 95 symbols put `top50` at a 52.6% base rate. Over ~500
symbols the rank is discriminative again, and rank-only reproduces the original
label — top-N of all USDT pairs:

```
floor      top5    top10    top20    top50
none      1.39%    2.70%    5.07%   11.57%
+3%       1.39%    2.70%    4.95%    9.45%
+5%       1.39%    2.67%    4.61%    7.06%
original  1.49%    3.15%    6.31%   15.34%   <- the target
```

**Row coverage rose from 54.7% to 86.9%** (106 507 of 122 545 labelled).

### The North Star denominator is reproducible again

`winners_by_day(rank_before_filter=True)` ranks the global universe and *then*
intersects the watchlist — the North Star's own definition. It yields **3.08
winners/day** against the original label's ~3.8. The two values may now be read
against each other; what still differs is the window (closed UTC day vs rolling
24h) and the universe (delisted pairs are unrecoverable).

## Findings from the review

**Adding a field to the hourly builder would have broken immutability.**
`LabelStore._identity` hashes every field plus `provenance.builder_version`, so
putting `resolution: "1h"` into `build_day_record` — or bumping
`BUILDER_VERSION` — would make a rebuild of any of the 19 502 written records
raise `ImmutableLabelError`. The field is read through a normaliser instead of
backfilled into records the store forbids touching.

**The MoveEvent guard was an accident of a filter.** `weekly_steering` computes
eligibility as `alert_ts is not None and (deadline is None or alert_ts < deadline)`.
A daily record has no deadline, so a single one entering that loop would classify
**every** alert as early and inflate Coverage@move. Only the watchlist filter
kept them out. `is_move_event_source` now says so explicitly, with a test.

**One malformed symbol killed a 734-symbol fetch.** `top_gainer_dataset` contains
a row whose `symbol` is not ASCII; `urllib` raised `UnicodeEncodeError` when it
reached the URL. Because Python sorts non-ASCII last, the crash landed after all
valid pairs — the data was complete and the run still exited 1. Symbols are now
validated against `^[A-Z0-9]{2,20}USDT$` and one bad symbol costs that symbol.

**204 of 734 pairs resolved to nothing** in the 240-day window — newly listed or
long delisted. They are counted and named in the build summary rather than
quietly shrinking the universe.

## Not in scope

Backfilling intraday resolution for non-watchlist symbols. Hourly bars for 700+
symbols is ~200× the data for no current consumer: the bot only alerts on
watchlist coins, so time-lead and MoveEvents are watchlist questions.
