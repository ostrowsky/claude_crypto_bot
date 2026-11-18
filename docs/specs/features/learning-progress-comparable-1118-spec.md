# Learning-progress analyzer: trends only on comparable windows, and Q5 from the target specs (2026-11-18)

- **Slug:** `learning-progress-comparable-1118`
- **Status:** SHIPPED 2026-11-18
- **Truth harness:** TH-01 (a ratio travels with its n, base rate and lift),
  TH-02 (a proxy is not progress), TH-04 (out-of-sample only), TH-05 (only
  windows with the same definition and comparable denominator are compared),
  TH-10 (with too little evidence the answer is "рано судить"), TH-13 (every
  defect below was reproduced on the live files before the fix).
- **Rollback:** revert the commit. Reporting and logging only: no trading
  decision reads the analyzer, and the three new `learning_progress.jsonl`
  fields are read by nothing else.
- **Maximum period / shadow:** not applicable (reporting only).

## 1. Defects

`files/analyze_learning_progress.py` is what the `crypto-progress` skill runs for
every "отчёт по обучению", so its verdicts reach the operator as written. Three
defects were found while writing the 2026-11-18 report; each was reproduced on
the live `.runtime` files.

### D1. Trends compared points across label changes (TH-05)

- **North Star.** `IMPROVING, first=0.0743 last=0.1242`. The first point is
  2026-04-29, metric `NS_EarlyCapture_top20` on the rolling-24h label with no
  `days_window`. The last is `NS_EarlyCapture_top20_v2` on immutable labels.
  The series also held 2026-08-30..09-21 rows where the primary value silently
  fell back to the leaky label, and 08-20..08-30 rows computed while the label
  store was frozen at 08-16 (n fell 19 → 1 as the window left the store).
  In total metrics_daily holds 7 definitions of this number.
- **recall@20.** `DEGRADING, 0.7541 → 0.1333`. The first point (2026-08-14) is
  out-of-sample, but on a different bandit training population. The label store
  froze on 08-16, and `phantom_filter.drop_phantom_rows` judges liveness from
  that store, so bandit training rows fell 11 110 → 6 816 (08-23..09-21) and
  jumped back to 11 618 on 09-22 after the refresh (0573da3). The operating
  point is also different: ENTER 34.8% vs 1.2%, base rate 8.1% vs 2.25%, lift
  2.17 vs 11.1. No n was printed for any point.
- **AUC top20.** Always `UNKNOWN`. The filter asked for
  `model_label_timing == "immutable_later_eod"`, but the rows say
  `"immutable_later_eod_close"`.
- **Overlap.** Consecutive daily NS rows share 13 of 14 days. The old verdict
  compared means of the first and last thirds of all those overlapping points.

### D2. The UCB separation line contradicted itself

`DEGRADING first=0.0473 last=0.1046 delta=-0.0887`. Two bugs:

1. `delta` was the mean of the last third minus the mean of the first third
   (0.0738 − 0.1625). It was printed next to the single first and last points,
   so the printed numbers could not agree with each other.
2. UCB was the only series with no scope filter. Its first point (2026-04-07)
   is the pre-rebuild bandit state that was re-ingested ~188× (CLAUDE.md §4:
   "UCB separation growing measured nothing").

`bandit n_signal` had the same rendering defect (`first 5888 last 25000
delta +9515`). It is also a capped counter (25 000), not a metric.

### D3. Q5 read the legacy matrix as the plan

Q5 parsed only `auto-improvement-loop-spec.md`. That spec is LEGACY / AS-BUILT,
last updated 2026-08-13, and its RM-4 / RM-5 / RM-18 items are retired
(CLAUDE.md §0). Q6 then recommended "Next roadmap items in priority order:
RM-21, RM-16, RM-17". The plan lives in `continuous-improvement-agent-spec.md`
(signal bot, target) and `autonomous-trading-roadmap-1010-spec.md` (trader).

## 2. Rules

### 2.1 Comparable points

- **North Star.** A metrics_daily row is a point only if all of these hold:
  - its `_compute_early_capture.py` block has the same `metric`, `days_window`,
    `label_provenance`, `denominator` and `lead_definition` as the newest row
    with metric `NS_EarlyCapture_top20_v2` on `immutable_later_eod_klines`;
  - `primary_degraded` is not true;
  - `immutable_store_last_day` is at least the row's date minus one day. A row
    that cannot show its labels covered its window is excluded (TH-10). So
    every row before 2026-09-22, when the field was added, is out.
  - `early_capture` and `n` are present.

  One point per UTC date (the last row of the date). Excluded rows are counted
  by reason and printed.
- **Bandit (recall@20, UCB, n_signal).** Rows with `bandit_evaluation_scope ==
  out_of_sample_time_holdout`, the newest row's `bandit_label`, and a date on or
  after the bandit epoch start, **2026-09-22** (the label-store refresh; see D1).
- **Model (AUC).** Rows with the newest row's `model_evaluation_scope` and
  `model_label_timing`, dated on or after **2026-09-22**. The model had trained
  on frozen labels for 34 nights before that (0573da3), and AUC fell
  0.898 → 0.848 that night.

The epoch starts are constants in the analyzer, each with its evidence. A future
definition change that leaves no field in the row must add one.

### 2.2 Non-overlapping windows

Each point has a window span. For the NS it is `[date − days_window, date − 1]`.
For recall it is the logged evaluation span (see §3). Walking back from the
newest point, the next point taken is the newest one whose span ends before the
current one's starts. This gives a chain of disjoint windows.

### 2.3 Verdict

For bounded metrics (NS, recall), whose per-unit values lie in [0, 1]:

1. Fewer than two windows in the chain → `INSUFFICIENT`.
2. Split the chain in time into an early and a late half with the same number of
   windows. With an odd count the middle window is left out.
3. Any window in either half without `n` → `INSUFFICIENT`.
4. Pool each half as an n-weighted mean, which is exact because the windows are
   disjoint. If either pooled n is below **20** → `INSUFFICIENT`. 20 is the
   per-window unit at which `continuous-improvement-agent-spec.md` §11.1 puts
   the NS minimum detectable difference at 0.127, 2.1× the metric's level.
   Below it not even that is reached.
5. Approximate 95% CI of `late − early`:
   `±1.96·sqrt(p₁(1−p₁)/n₁ + p₂(1−p₂)/n₂)`. For a mean of values in [0, 1],
   p(1−p) is an upper bound on the variance, so this is conservative in the
   variance. Units are treated as independent (not day-clustered), which is
   optimistic. Both are printed with the interval.
6. CI above 0 → `IMPROVING`; below 0 → `DEGRADING`; otherwise `INCONCLUSIVE`,
   printed with «рано судить».

`FLAT` is gone. An interval that spans zero does not show that the value is flat.

UCB separation and AUC have no logged n or variance. They get `DIAGNOSTIC`:
first and last of the comparable epoch, with `last − first` computed from those
same two printed points, and no verdict. n_signal and total updates are printed
as counters, first → last.

### 2.4 Output additions

- Q1 prints the chain, one line per window: date, value, n, full days. It also
  prints the excluded-row counts, and for the newest point:
  - the decomposition: coverage, capture and lead (capture and lead are means
    over entered winners, so their product is not the NS);
  - v3 `move_lead_early_capture` with `move_lead_n` and the winners without a
    deadline;
  - `days_full / days_window` and `days_down_or_partial`.
- Q2 prints, for each comparable recall point, the ENTER rate, base rate, lift
  and precision (TH-01), and the n when it is logged.
- Q5 reports:
  - the target spec: its `Status` line, its `Revised` date, the date of its last
    commit and its §13 rollout phases. The spec has no per-phase status, so the
    reached phase is printed as **not computed**;
  - the trader roadmap: the date of its last commit, and each §7 row as
    `#, spec, stage, verdict`, where the verdict is the first bold phrase of the
    status cell;
  - the legacy matrix, as a one-line **STALE** note with its age and the retired
    items. Its RM todo list is no longer printed and no longer feeds Q6.

## 3. Logging fix (§0: missing data is part of the work)

`daily_learning.save_progress` did not record how many positives the recall was
computed on, nor which days the evaluation covered. Three fields are added from
the evaluation result that already holds them:

- `bandit_n_positive` (`total_positive`);
- `bandit_eval_first_day`;
- `bandit_eval_last_day` (min and max of `daily[].day`).

Rows written before 2026-11-19 lack them, so recall will read `INSUFFICIENT`
until enough disjoint 7-day windows have accumulated. That is the honest state.

## 4. Tests (written before the fix)

`files/test_learning_progress_comparable.py`, on synthetic rows:

1. A legacy NS row plus v2 rows: the legacy row is excluded and the chain uses
   v2 only.
2. v2 rows without `immutable_store_last_day`, or with a stale store, are
   excluded with a reason.
3. Overlapping daily rows reduce to disjoint windows, and two rows on one date
   reduce to the last.
4. Small pooled n gives `INSUFFICIENT`. A difference inside the CI gives
   `INCONCLUSIVE`. A large difference with large n gives `IMPROVING`.
5. Recall rows before the bandit epoch, or in another scope, are excluded.
   Recall points without n give `INSUFFICIENT`.
6. UCB: the printed `last − first` equals last minus first. Rows before the
   epoch are not used.
7. AUC rows with `immutable_later_eod_close` are used.
8. Q5 reads the target spec's status and phases and the trader roadmap's §7
   rows, and calls the legacy matrix stale. Q6 no longer suggests `RM-` items.
9. The rendered Q1 shows the decomposition, v3 and `days_down_or_partial`.
10. `save_progress` writes the three new fields.

The existing `test_learning_progress_truth.py` must still pass.

## 5. Verification

On the live files of 2026-11-18, before and after: see the commit message.

## 6. Pipeline blocker count (2026-11-18, second fix)

### D4. Five May records reported as a structural blocker (TH-05, TH-10)

Q3 and Q6 counted every decision with `validation_verdict == pending_manual_validation`.
Five records matched, and the report called them «5 decisions stuck», a structural blocker.
They are not open work:

| record | stage | what happened |
|---|---|---|
| `d-2026-05-11T112800Z-31ae5d` ENTRY_SCORE_FLOOR_GLOBAL | approved | superseded the same day (`rolled_back`, demo of the rollback path) and on 05-28 (`deferred`, verdict `reject`) |
| `d-2026-05-11T213540Z-ce9fdf` WATCHLIST_MATCH_SCORE_TOLERANCE | rejected | closed |
| `d-2026-05-11T213541Z-f0df87` GATE_RANKER_HARD_VETO_THRESHOLD | rejected | closed |
| `d-2026-05-11T213714Z-4d6cb5` MODE_IMPULSE_SPEED_ENABLED | approved | superseded on 05-28 |
| `d-2026-05-28T132755Z-f37987` MODE_IMPULSE_SPEED_ENABLED | deferred | the key never existed in `config.py` (05-28 record). The question was answered in June by curtailment with fallback to trend (`cur001`, `rbcur1`, `fbt001`). |

None of the four keys is defined in `files/config.py` (checked 2026-11-18).

### 6.1 Rule

- **Thread.** Records are grouped by `hypothesis_id`. Without it they fall back to `config_key`,
  and without that to `decision_id`. The latest record (by `ts`, file order on ties) is the
  thread's current state.
- **Blocker.** A record counts only when all of these hold:
  - its verdict is `pending_manual_validation`;
  - it is the latest record of its thread;
  - its stage is not terminal (`rejected`, `rolled_back`);
  - its `config_key` is defined in `files/config.py`.
- **Not blocking.** Every other pending record is still shown, with its reason:
  - `superseded` — a later record of the same thread exists;
  - `closed_without_verdict` — terminal stage;
  - `unknown_key` — the key is not in `config.py`.
- **Config unreadable.** The `unknown_key` class is not applied, a latest non-terminal record
  counts as a blocker, and the output says `keys_verified = false`. The count is never lowered
  without evidence.

### 6.2 Output

- `pipeline.pending_manual_validation` now holds the blocker count. Q3/Q6 and the skill read
  this key.
- New fields: `pending_verdict_records` (the raw count), `pending_not_blocking`
  ({reason: count}), `pending_detail` ([{decision_id, config_key, stage, reason}]) and
  `keys_verified`.
- **Q3:** «(!) pipeline blocker» only when blockers > 0. Otherwise one line: pending records,
  none blocking, with the counts by reason.
- **Q6:** «Unblock pipeline» only when blockers > 0.

### 6.3 Tests (written before the fix, `files/test_learning_progress_comparable.py`)

1. The five live-shaped records plus their later records give 0 blockers: superseded 2,
   closed 2, unknown key 1, raw 5.
2. A latest pending record, stage `approved`, with a known key is one blocker.
3. Pending, then a later record of the same thread is `superseded`.
4. Unreadable config: no `unknown_key` class, the latest non-terminal record counts,
   `keys_verified = false`.
5. Records without `hypothesis_id` are grouped by `config_key`.
6. Rendering at 0 blockers: no «pipeline blocker» line in Q3, no «Unblock pipeline» in Q6, and
   the «none blocking» line is shown. At 1 blocker both appear.
