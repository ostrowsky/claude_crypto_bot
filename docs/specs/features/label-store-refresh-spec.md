# Immutable label store — kept current, and its absence made loud

- **Slug:** `label-store-refresh`
- **Status:** DEPLOYED 2026-09-21 (store refilled in place, nightly append wired)
- **Truth-harness invariants:** TH-03 (the honest label), TH-04 (comparable
  windows — the metric changed meaning without saying so), TH-05 (a metric must
  know what it does not know), TH-08, TH-11, TH-12, TH-13
- **Code:** `files/daily_learning.py` (`refresh_label_store`, Step 1b),
  `files/_compute_early_capture.py` (`primary_degraded`),
  `files/artifact_freshness.py` (`label_store`), `files/contextual_bandit.py`
  and `daily_learning.py` (pending-decision path removed)
- **Tests:** `files/test_label_store_refresh.py`
- **Rollback:** remove the `refresh_label_store()` call in `run_full_cycle`;
  the pre-refresh store is kept as
  `.runtime/labels/move_events_v1.pre_refresh_20260921.jsonl`.

## The defect

`build_global_labels.py` was run once, on 2026-08-17. The store it fills ended
**2026-08-16** and nothing ever extended it. Two consumers read it, and both
degraded without an error:

| consumer | what happened | evidence |
|---|---|---|
| `top_gainer_model` training | joins immutable labels onto the dataset; rows with no label are dropped | `immutable labels: 106507/…` **identical on every one of 34 nights, 2026-08-18 … 09-21**, while dropped rows grew 17 019 → 26 534. The model retrained nightly on frozen data. |
| North Star primary value | needs winner-days inside its 14-day window | from 2026-08-30 there were none; `primary = res_imm or res_top20` published the leaky rolling-24h value under the same job name |

`metrics_daily.jsonl` shows the swap: `label_provenance` flips from
`immutable_later_eod_klines` to `rolling_24h_same_snapshot` on 2026-08-30 and
`NS_EarlyCapture_top20_v2` becomes byte-identical to `legacy_early_capture`.
CLAUDE.md §1 says the primary metric is v2 on immutable labels; for three weeks
that was untrue and nothing said so. `artifact_freshness` did not list the store.

## The fix

1. **Nightly append.** `daily_learning.refresh_label_store()` runs
   `build_global_labels.py --days 30` before anything trains on the labels. The
   builder never touches an existing `(symbol, day)` record and skips an
   unfinished day, so the run is idempotent — it only appends what has newly
   closed. A failure is logged and the cycle continues; the freshness table then
   reports the store as stale.
2. **Freshness.** `label_store` is declared with a 36 h interval.
3. **The metric no longer substitutes silently.** When
   `NS_IMMUTABLE_LABELS_ENABLED` is on and the honest value cannot be computed,
   the artifact carries `primary_degraded: true`, `primary_degraded_reason` and
   `immutable_store_last_day`, and the console prints a banner naming the leaky
   label. The metric name is unchanged for backward compatibility, so the flag
   is the signal. When the flag is off the fallback is intended and stays quiet.
4. **Repaired in place, 2026-09-21.** `build_global_labels.py --days 60`:
   17 020 records written, 12 024 skipped as existing, store now ends
   2026-09-20 with 483–491 symbols per day (489 before the gap).

## Verified

- `NS_EarlyCapture_top20_v2` is primary again: **0.092** (n=39, coverage 0.72,
  capture 0.26, lead 0.53) against the leaky legacy value 0.077 (n=65);
  `primary_degraded: false`.
- Training join: **138 789** labelled rows, up from 106 507 (+32 282, +30%),
  reaching 2026-09-20 (counted on the raw dataset, before the phantom filter).
- 14 tests.

## Also removed: the pending-decision path

`daily_learning` called `resolve_pending_decisions`, which raised
`operands could not be broadcast together with shapes (21,21) (18,18)` on every
night from 2026-06-01 — 109 nights, 7 892 decisions never resolved. Contexts
were stored at 18 features; the bandit grew to 20 that day (RM-22 step B) and is
21 now. Its reward was "the coin is a top
gainer", the leaky label the offline path replaced on 2026-08-13, and the entry
bandit is rebuilt from scratch every night (`BANDIT_REBUILD_ON_TRAIN`), which
erases anything the resolver wrote. `should_enter` also rewrote the whole growing
JSON on every decision. The resolver, the buffer helpers, the writer in
`should_enter` and the "Pending decisions resolved" report line are gone;
`bandit_pending.json` was moved out of `files/`.

## Evidence, shadow decision and rollback — for the pending-decision removal

The removal of `resolve_pending_decisions` is the only part of this change that
deletes behaviour, so it gets the evidence the rest does not need.

**Maximum-period evidence.** `bot_learning.log` is the whole record, 161 cycle
nights, 2026-04-08 … 2026-09-21. The resolver logged a success on **52** of them,
2026-04-08 through 2026-05-31. On 2026-06-01 the bandit's feature vector grew
(18 → 20, later 21) while the buffered contexts stayed at 18, and it has raised
`operands could not be broadcast together with shapes (21,21) (18,18)` on **all
109 cycle nights since — no night in that span is without the error, and none
succeeded**. The buffer it reads holds 7 892 decisions spanning 2026-05-31 …
2026-08-21. So the resolver worked for seven weeks, on the leaky top-gainer
reward that the offline path replaced on 2026-08-13, and has produced nothing for
the last fifteen; there is no current behaviour to regress.

**Shadow / canary: not applicable** (не применимо). A shadow run measures what a
change would have done next to what happens today; here what happens today is an
exception on line one. Nor does the live path move: `BANDIT_ENABLED = False`
since 2026-09-07, so `should_enter` — the buffer's only writer — is not called,
and the entry bandit is rebuilt from scratch each night
(`BANDIT_REBUILD_ON_TRAIN`), so nothing the resolver wrote would have survived.

**Rollback.** Revert the commit to restore the resolver and the buffer helpers;
`bandit_pending.json` is preserved at
`history/_backup_bandit_pending_20260921.json`. Re-enabling the bandit does not
need it: its reward comes from the offline path (`forward_top10_min3`).

## Not covered

- The refresh takes ~16 min (750 requests plus reading the 147 MB dataset for the
  universe) and lengthens the nightly cycle by that much. Inside the 30-minute
  limit; a tighter universe would cut it.
- Delisted pairs cannot be ranked (TH-05).
- `EarlyCapture@top20_move_lead` prints 0.000 on n=11 winner-days that carry an
  intraday deadline. Not investigated here.
