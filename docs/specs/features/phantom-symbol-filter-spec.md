# Phantom symbols: dead tickers with live-looking features

- **Slug:** `phantom-symbol-filter`
- **Status:** shipped 2026-08-19
- **Created:** 2026-08-19
- **Truth-harness invariants:** TH-03 (a feature that cannot be true is not a
  feature), TH-05 (absence of data is not a value), TH-06 (validate on the
  population the bot actually sees)
- **Flag:** `PHANTOM_SYMBOL_FILTER_ENABLED`, default **True** — it removes rows
  that describe instruments which have not traded for months
- **Rollback:** flip the flag; the filter is one predicate in one place

## Which goal this serves

**Goal 1 — spot the day's winners early.** Directly: the early ranking spent
three of its ten picks on instruments whose last candle is from 2024 or 2025.
Those slots cannot ever be right, so removing them is a straight gain in the
only thing a ten-name list has — its ten names.

It also protects goals 2 and 3 indirectly, because the same phantom rows are in
the training data every model here learns from.

## How it was found

A live RENDER/USDT breakout was blocked by `trend quality guard: RSI 79.1 > 76.0`
— the confirmation catch-22 already measured. Checking whether the early path had
named it revealed that it had: **RNDRUSDT, third at 0.552.** But the watchlist
carries *both* `RNDRUSDT` and `RENDERUSDT`, and RNDR was renamed to RENDER in
2024. The pick was the dead half of a rename.

## The mechanism

`/api/v3/ticker/24hr` **keeps returning a row for delisted pairs**, with a
non-zero quote volume:

```
symbol        24h quote vol      last daily candle
RNDRUSDT          1 634 692      2024-07-22
EOSUSDT             924 392      2025-05-26
ACAUSDT             371 371      2026-02-13
RENDERUSDT        1 311 399      2026-08-19   <- the live instrument
SOLUSDT         119 933 615      2026-08-19
```

The snapshot builds features from that ticker, so a dead pair arrives with a
complete, plausible feature row. `EOSUSDT` carried
`tg_return_since_open = 6.79` — a coin "up 6.8% since the open" that has not
printed a candle since May 2025. The model ranks it first.

**Three of the ten early picks were phantoms**: EOS (#1, 0.765), RNDR (#3,
0.552), ACA (#8, 0.442).

## Scope, stated precisely

Three watchlist symbols are affected, not thirty-eight. The label store holds 38
stale symbols, but 35 of them are in the global ranking universe added on
2026-08-17 and were never tradeable here. Conflating the two counts would
overstate this by an order of magnitude.

## Change

One predicate, `is_live(symbol)`: the immutable label store has a complete record
for that symbol within `PHANTOM_MAX_LABEL_AGE_DAYS` (default 14). The store is
built from klines, so a symbol that has not printed a candle cannot have a recent
label — the check needs no extra network call and cannot be fooled by a ticker.

Applied in two places:

1. **the early ranking** — a phantom cannot be named, because a name that cannot
   be graded is a wasted slot;
2. **the snapshot writer** — a phantom row is not appended to
   `top_gainer_dataset`, because every model here trains on that file.

The watchlist itself loses the three dead entries. §14 calls the watchlist
immutable and says **do not expand** it; removing an instrument that has not
traded since 2024 is the opposite of expanding, and the operator asked for it
directly. `RENDERUSDT` is already present, so the rename loses no coverage.

## What must be reported, not hidden

**Existing dataset rows are not rewritten.** `top_gainer_dataset.jsonl` already
contains phantom rows for the whole period they were live-looking; the filter
stops new ones. Anything trained on history still sees them until the file is
rebuilt, and that is a separate change with its own evidence.

**A symbol can be live and unlabelled.** A newly listed pair has no label yet and
would fail `is_live`. That is the right call for ranking — an ungradeable name is
a wasted slot — but it means the filter is slightly conservative, and the count
of symbols it excludes is reported with every run rather than left implicit.

## Verification

`test_phantom_symbol_filter.py`:

1. a symbol whose newest complete label is older than the threshold is not live;
2. a symbol with a label from yesterday is live;
3. a symbol absent from the store is not live, and is counted separately from
   one that is present but stale — the two are different facts;
4. the early ranking excludes phantoms and reports how many it dropped;
5. with the flag off, nothing is filtered;
6. the three known phantoms resolve as dead and `RENDERUSDT` as live.

## Maximum-period evidence — it never drops a live winner

The risk worth testing is the opposite of the bug: a filter that removes a
symbol which was about to win. Evaluated over the whole label window, judging
each symbol **by what the store knew on that day** rather than by what it knows
now:

```
label window   2025-12-20..2026-08-16   221 days   681 winner-days

removed symbol   winner-days in window   newest label
EOSUSDT                     0            2025-05-20
RNDRUSDT                    0            2024-07-29
ACAUSDT                     4            2026-02-13

winner-days the filter would have dropped, as of each day:  0 of 681
```

`ACAUSDT` did win four times — all of them before it stopped printing candles,
and on those days the store had a fresh label for it, so the filter kept it. The
as-of evaluation is the whole point: judging staleness with today's knowledge
would have retroactively deleted those four winners and made the filter look
harmful when it is not.

## Shadow/canary decision: не применимо, and why that is not a dodge

There is no live decision to stage. The three removed symbols produced **zero
events in the entire event log** — no entry, no exit, not even a blocked
candidate. The live path needs klines to analyse a symbol, and a delisted pair
has none, so it never reached a gate.

Their only reach was the feature dataset and the early ranking, and both are
measurement surfaces. The behaviour change is that the bot stops polling three
instruments that cannot trade.

The filter's own risk — standing between the ranking and a live symbol — is
covered by the fail-open guard rather than by a canary: above a 25% drop it
stands down, because that is a broken label store rather than a dead market.

## The second application landed a commit late

The change section promised the filter in two places. The first commit shipped
only the early ranking, and the spec said otherwise for one commit — worth
recording, because a spec that describes what was intended rather than what
exists is the failure mode this file is supposed to prevent.

Both snapshot sites in `backfill_top_gainer_dataset.py` are now guarded. The
watchlist cleanup made this redundant *today* — the snapshot iterates the
watchlist and the watchlist is clean — but the next rename would leak straight
into `top_gainer_dataset.jsonl`, and that file is what every model here trains
on. A wasted slot in a ten-name list is cheap; a phantom row in the training
data is not.

The writer logs what it dropped rather than silently shrinking the universe, and
if the helper cannot be imported it writes the raw watchlist and says so: losing
a day of collection to a failed import is worse than a few phantom rows.
