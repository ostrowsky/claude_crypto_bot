# Measurement integrity — five defects found while closing open hypotheses (2026-09-25)

- **Slug:** `measurement-integrity-0925`
- **Status:** DEPLOYED 2026-09-25. The logging fix in `monitor.py` takes effect on
  the next bot restart.
- **Truth-harness invariants:** TH-04 (a field must mean one thing), TH-05
  (absence of data is not a miss / a zero), TH-08, TH-11, TH-12, TH-13
- **Code:** `files/_backfill_klines_history.py`, `files/_compute_early_capture.py`,
  `files/pipeline_monitor.py`, `files/monitor.py` (`_build_block_context`),
  `files/pipeline_replay_validator.py`
- **Tests:** `files/test_measurement_integrity.py`, `files/test_kline_long_store.py`
- **Rollback:** revert the commit. Store backups:
  `history/_backup_15m_419d_20260925/`, `history/_backup_1h_365d_20260925/`,
  `history/_backup_peak_label_cache_20260925.json`.

None of these changes a trading decision. Each made a number say something the
data did not.

## 1. Unclosed klines frozen into the long store (introduced 2026-09-18)

Binance returns the still-forming bar last. `fetch_paginated` kept it, the 06:00
local run wrote it into the rolling file at 04:00 UTC, and `extend_long_store`
(added 2026-09-18) appended it and never revisited it. Compared with the exchange:
one bar per symbol per day with 97–100% of its volume missing and closes off by
up to 12%, plus the manual 09-18 08:15 run and the 419-day backfill's last bar
(08-20 08:00/08:15). Every 15m bar in the 20 after it carried a wrong `vol_x`.
That produced the QNT 2026-09-24 07:30 "alignment fire" (store `vol_x` 1.01 vs
live 0.94): an artefact, since found and dismissed.

Fix: `fetch_paginated` keeps only bars whose `close_time` has passed; the append
replaces an overlapping bar that differs from the (closed-only) rolling file — a
closed bar never changes on the exchange, so a difference means it was captured
while forming. Repair: 737 15m bars and 651 1h bars replaced, exactly at the
diagnosed times; the store now matches the exchange bar-for-bar (3 953 bars × 3
symbols checked, 0 differing). The peak-label cache was moved aside to rebuild.
The live bot never reads these files. Replays of 2026-09-25 had at most 1.6% of
rows within 20 bars of a bad bar; no conclusion rests on that margin.

## 2. The `ml_proba` field of blocked events carried the ranker

`monitor._build_block_context` wrote a parameter named `ranker_proba` into
`ml_proba`. Four block sites passed the ML score; seven (trend_quality onward)
passed the candidate ranker's `quality_proba`. In 30 102 recent blocked events
`ml_proba == ranker_quality_proba` exactly, while 546 of 546 entries carry the
true score. This is the "bimodal live ml_proba" (the ranker sits near 0.40); a
replay of the same candidates through the ML model gives a smooth 0.13–0.86.
A second, shadowed definition carried the same defect and was removed.

Fix: an explicit `ml_proba` argument at all 11 sites; the ranker travels only in
its `ranker_*` fields. Decisions never used this field — only analyses did.

## 3. `EarlyCapture@move_lead` scored only misses

A winner-day without an intraday deadline was skipped only when the bot had
entered it, so on 2026-09-25 all 36 entered winner-days were dropped and the 14
missed ones kept: 0.000. Exclusion no longer depends on the outcome, and an empty
sample prints "not computable" and publishes `null`. It is not computable in the
14-day window (all labels there are daily); over 60 days, 11 winner-days with
timing give coverage 0.73 and lead 0.03 — entries after the +2.5% crossing.

## 4. L7 counted unmeasured decisions as misses

A metric no health report carries set `overall_hit = False`. Of the 11 "misses"
behind eleven weeks of `hit_rate = 0.0`, 8 were unmeasurable and each produced a
weekly rollback recommendation. A metric now needs 2 readings on each side;
decisions with none are `unmeasurable` and stay out of `hit_rate`. Today: 8
unmeasurable, 3 needs_data, 3 measured misses — all three on the legacy
rolling-24h metric, approved within a day of each other, one shared window.

## 5. The L3 ML replay graded the ranker against an ML floor

Consequence of 2. Rows whose `ml_proba` equals `ranker_quality_proba` are now
unreplayable. The 2026-09-18 verdict on tightening the ML floor 0.15 → 0.30
("reject, 1.12×") was computed on the ranker and is void. On clean rows:
0.15 → 0.20 removes 171 rows at 0.77× mean 4h peak and 0.47× the ≥5% share;
0.15 → 0.30 removes 322 at 0.93× / 0.83×. One month of post-switch data — a lead
for the operator, not a change.

## Maximum-period evidence and shadow decision

Measurement and logging fixes; the evidence for each is the full-log count
above. Shadow / canary: not applicable (не применимо) — no decision path changes.
