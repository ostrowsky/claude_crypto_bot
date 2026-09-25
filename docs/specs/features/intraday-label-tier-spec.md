# Intraday label tier — hourly timing for the move-relative North Star

- **Slug:** `intraday-label-tier`
- **Status:** DEPLOYED 2026-09-25 (built in place, nightly append wired)
- **Truth-harness invariants:** TH-03 (labels from exchange klines only), TH-05
  (a missing or partial day is skipped, never scored), TH-08, TH-11, TH-12, TH-13
- **Code:** `files/label_store.py` (`build_intraday_from_store`,
  `intraday_deadlines`, `INTRADAY_FILE`), `files/_compute_early_capture.py`,
  `files/daily_learning.py` (`refresh_label_store`), `files/artifact_freshness.py`
- **Tests:** `files/test_intraday_label_tier.py`
- **Rollback:** delete `.runtime/labels/move_events_1h_v1.jsonl` and revert the
  commit; the main store is not touched by this change.

## Why

`EarlyCapture@move_lead` scores how early the bot alerted relative to the coin's
own move: the deadline is the hour the price first crossed +2.5% from the UTC
open. Hourly labels stopped on 2026-08-12 (their source was never refreshed), and
every later day came from daily klines, which carry no time. The metric had
nothing to score for six weeks — and until the fix in
`measurement-integrity-0925-spec.md` it printed 0.000 instead of "not
computable".

The daily records are immutable and cannot be upgraded in place, so the timing
lives in its own immutable file, `.runtime/labels/move_events_1h_v1.jsonl`,
built with the same `build_day_record` from the repaired long 1h kline store.
Only the move-lead metric reads it; the top-20 ranking stays on the main store.

## Rules

- a day is written only after it closes and with ≥ 20 hourly bars;
- the provenance hash is of that day's bars, so rebuilding an unchanged day is a
  no-op and a changed one raises instead of overwriting;
- nightly, after the daily labels, for the watchlist, last 30 days. The 1h store
  refreshes at 06:20 local, after the 02:30 cycle, so the tier lags one day.

## Built 2026-09-25

4 003 records, 2026-08-13 … 09-24, 93–94 symbols a day (the rest of the
watchlist has no candles). A rebuild wrote 0 and raised 0. The day's return from
the hourly records equals the daily record's on all 4 003 days (median and p99
difference 0.0000 pp).

## What it shows

| window | winner-days with timing | coverage | lead | EarlyCapture@move_lead |
|---|---|---|---|---|
| 14 days | 43 | 0.67 | 0.09 | 0.009 |
| 30 days | 95 | 0.75 | 0.09 | 0.021 |
| 60 days | 147 | 0.64 | 0.08 | 0.015 |

Over 60 days the bot entered 113 winner-days that carry timing, and only **15
(13%) before the +2.5% crossing**; the median entry came 3.0 hours after it
(p25 +1.0 h, p75 +9.1 h). Goal 2 — signal the entry early relative to the move —
is now measurable, and it is where the North Star is lost.

## Shadow / canary

Not applicable (не применимо): a measurement input, no decision reads it.
Maximum-period evidence is the build above.
