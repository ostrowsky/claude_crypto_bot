# metrics_daily: the row is always written, and a stale row never reads as today (2026-10-20)

- **Slug:** `metrics-daily-stale-1020`
- **Status:** SHIPPED 2026-10-20
- **Truth harness:** TH-05 (a metric must know what it does not know), TH-13
  (claims checked against the system), TH-10 (the report states no conclusion
  its data cannot support)
- **Rollback:** revert the commit. Reporting only: no trading decision reads
  these fields.
- **Maximum period / shadow:** not applicable (reporting and scheduling only).

## Defect

`.runtime/metrics_daily.jsonl` got no row after 2026-10-11T07:00. The morning
report kept printing the 10-11 North Star (0.204) as today's value with status
`verified` from 10-12 through 10-20.

Evidence (`.runtime/pipeline/orch/orch-*.json`, step `L0 metrics`):

- every run from 2026-10-12T07:00 to 10-20T07:02 ended `error: timeout` after
  600 s (10 runs);
- before that the step was intermittent: 137–599 s when it finished, and
  timeouts on 09-23, 09-26, 10-02, 10-05 (x2), 10-06, 10-09 and 10-11 03:30;
- run alone on 10-20 the nine scripts took **428 s** in total. The slowest were
  EX1 zigzag at 108 s, `_compute_early_capture` at 75 s and the funnel at 54 s.
  The 07:00 run shares the CPU with other jobs on a throttled laptop, so 600 s
  had no margin.

The causes:

1. `report_metrics_daily.py` ran nine scripts in sequence, each with a 180 s
   limit (up to 1 620 s), against a 600 s orchestrator timeout.
2. The row was appended only after the last script, so a kill meant **no row**,
   not a partial one.
3. `pipeline_run.run_step` swallowed `TimeoutExpired` without printing, so
   `cron-daily.log` showed `cmd:` followed by the next step, and the step looked
   like it had run.
4. `bot_health_report.collect_metrics_daily_latest` took the last row with no
   age check, so a nine-day-old value read as current and `verified`.

## Change

- `report_metrics_daily.py`:
  - all scripts share one wall-clock budget (`DEFAULT_BUDGET_SEC = 1500`), and
    each script gets `min(300, remaining)`;
  - the North Star script runs first;
  - a script the budget does not reach is recorded as
    `{"error": "skipped_budget"}`;
  - per-script `elapsed_sec` is stored in the row;
  - the row is always appended.
- `pipeline_run.py`: `L0_METRICS_TIMEOUT_SEC = 1800`, above the budget plus
  start-up. Timeouts and start failures are now printed.
- `bot_health_report.py`:
  - every metric carries `_as_of` (the ts of its row) and `_stale`, which is set
    when the row is older than `METRICS_DAILY_MAX_AGE_HOURS = 36` or its ts
    cannot be read;
  - a metric missing from the latest row (it timed out) is taken from the newest
    row that has it, still marked stale by its own age;
  - scorecard entries get status `stale` and `as_of`;
  - the North Star in that state gets the headline «ДАННЫЕ УСТАРЕЛИ», a line
    naming the date the value is from, a P0 step `restore_metrics_daily` and a
    critical red flag `RF_metrics_daily_stale`;
  - none of the "labels are provisional" statements appear for a stale value,
    because its labels are immutable; it is only old.
- `truth_harness.py`:
  - `TH05_STALE_METRICS` fails a report that presents a > 36 h old
    metrics_daily row as anything but `stale`;
  - `TH05_METRICS_DAILY_NOT_REFRESHED` fails when `metrics_daily.jsonl` itself
    has no row in the last 36 h;
  - `stale` is accepted as disclosure by `TH03_NORTH_STAR_DISCLOSURE`.

## Tests

- `files/test_metrics_daily_staleness.py`: the budget contract (slow scripts
  still produce a row with every script present, within the budget), North Star
  first, budget below the orchestrator timeout, timeout printed, fresh row →
  `verified`, 9-day row → `stale` on every metrics_daily-derived entry,
  unreadable ts → stale, fallback from an older row → stale, the P0 step and
  the Telegram line.
- `files/test_truth_harness.py`: three TH-05 cases.

Disclosed: the `report_metrics_daily.py` refactor was drafted before its tests.
The defect itself was reproduced from the orchestrator records above. The tests
for the stale guard and the harness checks were written first and failed before
the code.
