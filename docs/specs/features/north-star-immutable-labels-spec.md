# North Star on immutable labels (TH-03)

- **Slug:** `north-star-immutable-labels`
- **Status:** shipped; immutable value PRIMARY since 2026-08-17
- **Created:** 2026-08-14
- **Consumes:** [`immutable-label-store`](immutable-label-store-spec.md)
- **Truth-harness invariants:** TH-03 (label provenance), TH-04 (comparable
  windows), TH-11 (proxy is not the outcome)
- **Flags:** `NS_IMMUTABLE_LABELS_ENABLED` (metric, default **True**),
  `TRAIN_IMMUTABLE_LABELS_ENABLED` (model, default **False**)
- **Rollback:** flip either flag; both paths remain in the code

## Problem

`label_top20` marks a coin a winner using the same rolling-24h snapshot that
produced its features. A coin already up 14% at the snapshot is labelled a
winner *because* it is up 14%. Every number resting on that label inherits it:

- `top_gainer_model` reports AUC ≈ 0.99 with `tg_return_since_open` among its
  top features;
- the bandit once reported "recall@20 = 100%";
- the North Star is marked **provisional** and cannot prove progress;
- and yesterday's day-grouped-split evidence came back at **zero effect**,
  because a model that can read the answer has no need to peek across a day
  boundary. The smaller leak is unmeasurable through the bigger one.

Fixing the label is therefore not one improvement among several. It is the
precondition for the others being measurable at all.

## Change

### Winners

A winner is now a `(UTC day, symbol)` in the day's **top 20 by
`eod_return_pct`**, taken from the immutable label store — close ÷ open on
exchange klines, computable only after the day closes. The bot's datasets are
not consulted.

```
NS_IMMUTABLE_LABELS_ENABLED = True    # metric only; changes no behaviour
```

`_compute_early_capture` grows a second loader. The old one stays, and **both
values are published side by side** so the change is visible rather than
substituted. The old figure is labelled `leaky_same_snapshot`; the new one
carries `label_provenance = "immutable_later_eod_klines"`.

### Model labels

`train_top_gainer` can label rows by joining `(symbol, utc_day)` to the store
and ranking on `eod_return_pct`. Default **off**: this model feeds
`ranker_top_gainer_prob` into the hard veto, so relabelling changes live gating
indirectly, and a behaviour change ships with current behaviour as its default.

## What must be reported, not hidden

**The two universes differ.** The store covers 98 symbols, three of them stale
(delisted or renamed), so the immutable top-20 is drawn from ~95 current symbols
against the watchlist's 105. The window also differs: the store begins
2026-01-26. Any comparison of the two North Star values must state both.

**The number is expected to change, possibly a lot.** The old winner set was
"already moved by the snapshot"; the new one is "finished the day highest". They
overlap but are not the same population. A different value is the point — it is
not evidence that either the bot or the metric got worse.

**`provisional` is not lifted by this alone.** It is lifted when the North Star
is computed on immutable labels *and* the harness stops flagging TH-03. Until
`train_top_gainer` also switches, the model is still trained on leaky labels and
`evaluation_scope` must keep saying so.

## Verification

`test_north_star_labels.py`:

1. immutable winners come from the store, never from `top_gainer_dataset` (AST
   and behavioural);
2. exactly 20 winners per fully covered day, fewer when the universe is thin;
3. a day whose coverage is below the store's threshold yields no winners rather
   than a short list;
4. the emitted metric carries `label_provenance` and both values;
5. the old and new winner sets are reported with their overlap, not silently
   merged;
6. the model labeller returns `None` for a `(symbol, day)` the store does not
   know, so a missing label is never silently a zero.

**Maximum-period evidence** (`_backtest_immutable_ns.py`): recompute the North
Star both ways over the longest window both label sources cover, and publish
value, coverage, capture, time-lead, winner-set overlap and n.

**Shadow/canary: не применимо** for the metric — it changes no behaviour. The
model flag stays off, so there is no second live behaviour to stage.

## Maximum-period evidence

`_backtest_immutable_ns.py`, 117 common UTC days (2026-04-17..2026-08-12), rule
held constant at "the day's top-20 by EOD return", varying only the data source:

```
winner set                            n    per day
snapshot-derived returns           2340       20.0
immutable (top-20 by EOD close)    2340       20.0
in both                            1452         62%

snapshot returns      EC=0.0243  n=1920  cov=0.52  cap=0.07  lead=0.59
immutable later-EOD   EC=0.0236  n=1920  cov=0.52  cap=0.08  lead=0.55
```

**The winner sets disagree on 38% of day-symbol pairs, and the aggregate barely
moves.** Both facts matter and neither cancels the other. The composition change
is large — a third of the days' winners are different coins — while the North
Star is insensitive to it, because coverage and lead average over a set whose
*size* is fixed by construction. So this change does not improve the number; it
replaces where the number comes from. That is the point: the value is now usable
as ground truth for features computed during the day, which the old one was not.

### The first version of this comparison was invalid

It compared `label_top20` (global top-20 intersected with the watchlist, 3.8
winners/day) against the watchlist's own top-20 (20/day). Overlap read 11% and
EC fell 0.070 → 0.024, and it would have been reported as the leakage effect. It
is not: two changes moved at once — provenance *and* denominator — and the
denominator dominated. The script now ranks the snapshot returns by the same rule
before comparing. **A leakage experiment that also changes the population is not
an experiment.**

### The same trap survives in the daily report, and is labelled rather than hidden

`_compute_early_capture` cannot apply the same fix: the label store holds only
watchlist symbols, so the global top-20 denominator is unreproducible from it.
The two values are therefore emitted with different denominators (n=39 vs n=340
over the same 19 full days) and the report says so in three places — a printed
warning naming both denominators, `immutable_denominator` in the JSON, and
`immutable_comparable_to_primary = False` so a downstream consumer cannot
difference them by accident. The same-rule answer lives in the backtest.

## Update 2026-08-17 — the two values are comparable now

With the global store, `winners_by_day(rank_before_filter=True)` reproduces the
North Star's own denominator (`watchlist ∩ global-top20`) at **3.08 winners/day**
against the original label's ~3.8. `immutable_comparable_to_primary` flipped to
`True` and the printed warning became a statement of what still differs.

Recomputed over the maximum period both cover (109 days, 2026-04-19..08-16):

```
                          n   per day    EC      cov    cap    lead
snapshot rolling-24h    405       3.7   0.0727   0.62   0.17   0.67
immutable later-EOD     319       2.9   0.1041   0.75   0.22   0.63
in both                 175        55%
```

Last 30 days, 18 full days: leaky **0.072** vs immutable **0.129** (cov 0.58 vs
0.82).

**The honest North Star is higher than the leaky one, and the reason is
mechanical.** The old label ranks a rolling 24h window at snapshot time, so it
counted coins that spiked *overnight, before the day being measured* — moves the
bot could not have caught, scored as misses. Ranking a closed UTC day asks only
about moves that happened while the bot was watching. Coverage 0.62 → 0.75 is
that correction, not an improvement in the bot.

Still far from the 0.40 target and the 0.25 floor. What changed is that the
number can now be trusted and compared over time.

## 2026-08-17 — the immutable value becomes primary, versioned

`NS_EarlyCapture_top20` → **`NS_EarlyCapture_top20_v2`**. The metric is versioned
rather than silently redefined: a reader comparing today against last month sees
two names, not one number whose meaning changed. The old value keeps travelling
in the same payload as `legacy_metric` / `legacy_early_capture` / `legacy_n`, so
the historical series stays reconstructable (TH-04).

```
metric            NS_EarlyCapture_top20_v2
label_provenance  immutable_later_eod_klines
denominator       global_top20_intersect_watchlist_from_label_store
30d               0.129  (n=28, cov 0.82)
legacy 30d        0.072  (n=38, cov 0.58)
```

Rollback: `NS_IMMUTABLE_LABELS_ENABLED=False` restores the old value as primary.

### The harness checks could not see their own repair

`TH03_TOP_GAINER_TARGET` and `TH04_DAY_GROUP_SPLIT` were pure source-string
checks — "train_top_gainer splits by row index", "the label comes from the same
snapshot". Both stayed red after the fixes shipped behind flags, because the
strings they matched are still in the file (the legacy paths remain on purpose).
A check that cannot observe its own repair is worse than no check: it keeps
reporting a solved finding until the reader learns to skip the red line.

They now read the **deployed model blob** — `label_timing`, `evaluation_scope` —
which is a stricter witness than the flag, since a flag can be on while the model
in production predates it. `TH03_NORTH_STAR_TARGET` reads which value the metric
publishes as primary, not whether the legacy loader exists.

Blocking findings dropped 5 → 2. What remains is genuinely open: canonical ZigZag
EX1 (TH-11) and two gate locks without current-epoch evidence (TH-10).

**And the re-pointed check was wrong on its first run.** `_trained_model_facts`
caught bare `Exception` and used `io`, which `truth_harness` does not import, so
every field came back `None` and the checks reported "no immutable label" about a
model that had one. A swallowed error returning a plausible answer is the exact
failure this file exists to prevent; the handler is now narrow and an unreadable
blob is reported as `_unreadable` rather than as an absent field.

## Not in scope

Retiring the old label from `top_gainer_dataset` (it still feeds the bandit and
the critic), and flipping the model flag — that is a separate change whose
evidence is a retrain comparison, and it should be flipped together with the
day-grouped split so one measurable change is attributable.
