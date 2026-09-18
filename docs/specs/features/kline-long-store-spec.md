# Long kline store — history that keeps growing

- **Slug:** `kline-long-store`
- **Status:** DEPLOYED 2026-09-18 (data repaired in place, daily append wired)
- **Truth-harness invariants:** TH-04 (comparable windows need the data to
  exist), TH-05 (a metric must know what it does not know), TH-11 (the training
  label is only as good as its klines), TH-13
- **Flags:** none — data plumbing, no decision logic changes.
- **Code:** `files/_backfill_klines_history.py` (`extend_long_store`, `extend_all`,
  `--extend-only`)
- **Tests:** `files/test_kline_long_store.py`
- **Rollback:** restore `history/_backup_15m_419d_20260918/` and
  `history/_backup_1h_365d_20260918/` over `history/`.

## The defect

Two kinds of kline file live in `history/`:

| file | written by | window |
|---|---|---|
| `<sym>_15m_419d.csv`, `<sym>_1h_365d.csv` | one-off backfills | fixed, **frozen** |
| `<sym>_15m.csv`, `<sym>_1h.csv` | daily tasks | rolling 30 / 60 days |

Every reader that wants *history* — the backtests and, since 2026-09-07, the peak
**training label** in `ml_signal_model._peak_bars` — reads the long file. The 15m
long file ended **2026-08-20**; the 1h long file ended 2026-06-20 and the rolling
1h file only started 2026-07-20, leaving a month in neither.

Nothing errored. What it cost, measured:

| rows of ml_dataset | peak label resolved |
|---|---|
| 15m, 2026-08 | **48%** |
| 15m, 2026-09 | **0%** |
| 1h, 2026-06 | 63% |
| 1h, 2026-07 | 44% |

Unresolved rows fall back to the old inverted `ret_5 > 0` label, and the overall
92% resolution cleared `ML_PEAK_LABEL_MIN_RESOLVED = 0.60`, so the newest data was
being taught with the wrong label in silence. A backtest built on the same
loader lost 4104 of 12 592 rows and all of September on its first run.

## The fix

1. **Daily append.** After each fetch, the backfill appends bars newer than the
   long file's last timestamp from the rolling file onto the long one, for 15m
   and 1h. Append-only: bars already in the long file are never rewritten, so
   results computed on it stay reproducible. A gap between the files is printed
   as `GAP` — the bars in between must be re-fetched — never papered over.
2. **One-off repair, 2026-09-18.** 15m: 258 957 bars appended across 102
   symbols, 0 gaps. 1h: the 2026-06-20 … 2026-07-20 hole re-fetched from Binance
   for all 427 symbols (336 941 bars), 0 symbols left with a hole > 1h.

## Verified

- 15m series continuous to 2026-09-18 08:15 for every checked symbol, 0 irregular steps.
- Peak label resolution **98.5% → 5487 / 5490 (100%)**; every month and timeframe 97–100%.
- A retrain on the repaired data chooses the same family (logistic) and scores
  the same on the recent holdout (AUC 0.8099 vs 0.8096 live, top-decile 3.76%
  both). Expected, not a null result: the repaired rows are Aug–Sep, which fall
  in the validation/test tail of the 70/15/15 time split. They correct model
  selection now and enter training as they age.

## Not covered

Delisted pairs cannot be re-fetched (TH-05). The rolling files remain the
freshness source for live checks; the long files are for history.
