# mode_range_quality gate OFF — operator decision (2026-10-01)

- **Slug:** `mrq-off-1001`
- **Status:** SHIPPED 2026-10-01, `MODE_RANGE_QUALITY_GUARD_ENABLED = False`
- **Decision:** operator, chat 2026-10-01: «Снять: +3.4 п.п. ранних входов, но ~+6.6
  мусорных сообщений в день» — the trade-off was put to the operator with both numbers
- **Truth harness:** TH-06 (bot's own decisions), TH-07 (flag + rollback; the guard
  keeps running in shadow), TH-10 (pre-registered live readout), TH-11 (goal)
- **Rollback:** `MODE_RANGE_QUALITY_GUARD_ENABLED = True` + restart

## Evidence (maximum period, valid-proposals-1001-spec.md)

Gate removed entirely, judged by the L3 goal criterion over 412 winner-days since
2026-05-01: entered before the +2.5% crossing **10.9% → 14.4% (+3.43 pp)**; trades
changed −0.07% (n = 2 182) vs current −0.06% (n = 3 555); combined − current
**−0.002 pp [−0.049, +0.045]** (non-inferior); months agree 60% of 5. Cost the goal
criterion does not see: ~1 418 new coin-days blocked by it (11.8/day), with the
downstream pass rate 0.56 **~+6.6 messages/day**, only **1.1%** of those new
coin-days winners (current entries 10.9%). The gain comes mostly from earlier
entries on coins the bot enters anyway.

The original reason for the gate (2026-04-24, 60 days, 2 197 entries) was
precision on quiet days — a proxy, not the goal.

## What changes

- `monitor.py`: the guard is evaluated with `ignore_flag=True`; when the flag is
  off and it would block, the candidate is NOT blocked and an event
  `mode_range_shadow` (sym, tf, mode, price, reason) is written, so every entry the
  removal lets through is identifiable.
- `do_not_touch.json`: `mode_range_quality` leaves `contested` for `released`,
  with the decision; the report's P1 step disappears.
- `decisions.jsonl`: an approved record (operator, chat) with a pinned baseline, so
  the 14-day attribution measures it.

## Readout `MRQ-OFF` (pre-registered, readout_registry.json)

Window from 2026-10-02, due 2026-10-16, ≥ 14 comparable days, ≥ 20 added trades.
Added entries = entries preceded by a `mode_range_shadow` of the same coin and
timeframe within one bar. Measured: added entries/day (expected ~6.6), realised
pnl of added vs other entries, early winner-days from the added entries.
- ANOMALY at once: > 20 added entries/day (3× the expectation)
- KEEP: combined − other, lower 95% ≥ −0.10 pp
- ROLLBACK_SUGGESTED: upper 95% < −0.10 pp
- otherwise INCONCLUSIVE (keep, re-read)

Honest limit: +3.43 pp of 412 winner-days is ~14 days in 120, i.e. ~1–2 early
winner-days in a 14-day window — the readout can check the cost side (messages,
pnl) with power, the gain side only as a count.

## Tests

`files/test_mrq_off_1001.py` (9).
