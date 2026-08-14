# Immutable label store (Phase 0a)

- **Slug:** `immutable-label-store`
- **Status:** spec → implementation
- **Created:** 2026-08-14
- **Parent:** [`continuous-improvement-agent`](continuous-improvement-agent-spec.md) Phase 0a
- **Truth-harness invariants:** TH-03 (label provenance), TH-04 (day-grouped
  splits), TH-05 (unknown stays unknown)
- **Rollback:** delete `.runtime/labels/`; nothing reads it yet that cannot fall
  back to the existing datasets

## Problem

Two blocking harness findings share one root cause, and one operator question
depends on fixing them.

**TH-03 — the ground truth measures itself.** `label_top20` is computed from the
same rolling-24h snapshot that produces the features, so a coin already up 14%
at the snapshot is labelled a winner *because* it is up 14%. That is how
`top_gainer_model` scored AUC 0.99 and the bandit reached "recall@20 = 100%".
The North Star inherits it and is marked provisional.

**TH-04 — a UTC day can straddle the split.** `train_top_gainer.py` splits sorted
rows by row index. Rows from one day share market beta and, with T+N labels,
share outcomes; putting some on each side leaks.

**And the operator's question has no instrument.** Asked whether trading results
have improved, the honest answer today is that nothing can tell — the metric
that could is provisional, and the metric that could resolve weekly does not
exist. Building immutable labels is step one of both.

## Design

### Source: exchange klines, never the bot's own snapshots

Labels come from 1h OHLCV pulled from the exchange (`_hourly_ohlcv_long.json`,
98 symbols × 200 days). The bot's datasets are not consulted. That is the whole
point: a label derived from the same snapshot as the features cannot be ground
truth for those features.

### One record per (symbol, UTC day)

Strict UTC. `Europe/Budapest` produces 23- and 25-hour days at DST boundaries
and disagrees with the exchange day — a silent denominator defect in exactly the
metric built to be trustworthy.

```
symbol · utc_day
open · high · low · close                 from the day's bars
eod_return_pct        = close/open - 1    the immutable later-EOD label
max_move_pct          = high/open - 1     the best the day offered
qualifies_move5       = max_move_pct >= 5.0
anchor_ts             first bar whose high >= open x 1.05   (nullable)
early_deadline_ts     first bar whose high >= open x 1.025  (nullable)
bars_used             hours actually present
complete              bars_used >= 20     partial days are UNKNOWN, not zero
label_mature_at       UTC day close
provenance            {source, source_sha256, builder_version, built_at}
```

`MoveEvent v2` from the parent spec is exactly this record: the qualifying test
is on the whole UTC day, the early deadline is a **fixed** +2.5% crossing rather
than half the realised move, so it cannot be computed with hindsight.

### Honest resolution limit

Hourly bars place crossings to the hour, not the minute. An alert inside the
same hour as `early_deadline_ts` cannot be ordered against it, and will be
counted **late** — conservative against the bot. Sharpening this needs 15m bars
and is not claimed here.

### Immutability

A record for a (symbol, day) is written once. Rebuilding must reproduce it
byte-identically; a mismatch raises rather than overwrites. Only the last day in
the window may legitimately change, because it is still forming — so a day is
written only when `complete` and its close has passed.

### Day-grouped splitting

`day_split.split_by_day(rows, day_key, train_frac)` returns train/holdout such
that **no UTC day appears on both sides**, with an optional embargo of N days
between them for label maturity. This is the TH-04 fix as a reusable function;
wiring it into `train_top_gainer.py` is a separate change with its own
before/after evidence, because it alters what the model trains on.

## Verification

`test_label_store.py`:

1. rebuilding an existing day reproduces it byte-identically;
2. a changed source hash is refused, not silently overwritten;
3. incomplete days are `complete=false` and excluded from qualification;
4. `early_deadline_ts <= anchor_ts` whenever both exist — the deadline can never
   follow the anchor it precedes (the v1 MoveEvent bug, pinned);
5. labels are computed from klines only — the module imports no bot dataset;
6. every record carries provenance with a source hash;
7. `split_by_day` never puts one day on both sides, at any fraction;
8. an embargo removes the boundary days from training;
9. splitting is deterministic.

## Findings from the architecture review

**The first `status()` reported a rate that was not a rate.** It divided
qualifying events by the union of all days in the store — but three of 98
symbols (`AUDIOUSDT`, `EOSUSDT`, `RNDRUSDT`) are delisted or renamed on the
exchange, so the fetch returned 2024–2025 history for them. Their labels are
real; pooling them with 2026 days mixed eras and universes. The published figure
was **5.18 qualifying events per day** where the honest figure over the
well-covered window is **16.27** — a threefold understatement produced by a
denominator nobody had looked at.

Fixed by scoping every rate to days where at least 80% of the universe is
present, reporting `days_any_coverage` and `well_covered_window` separately, and
**naming the stale symbols** instead of averaging them away. When no day is
well covered, rates are withheld rather than computed.

That this appeared in the very module built to end label dishonesty is the
useful part: the failure mode is not a lack of care, it is that denominators are
invisible until something forces them into the open.

## Current contents

`as_of 2026-08-14`, source `_hourly_ohlcv_long.json`:

```
records              19 502          symbols            98
days_any_coverage       660          full_window        2023-11-11..2026-08-12
well_covered_days       199          well_covered_window 2026-01-26..2026-08-12
qualifying_move5      3 237          qualifying_per_day 16.27
base_rate_pct         17.12          stale_symbols      3, named
```

16 qualifying events per day is the number that matters for the weekly report:
it is ~114/week against ~20 top-20 winners, which is why the steering pair was
put on this event and not on the mission metric.

## Explicitly not claimed

This ships labels and a splitter. It does **not** yet recompute the North Star,
retrain any model, or produce a weekly report — those consume the store and come
next. No trading behaviour changes.
