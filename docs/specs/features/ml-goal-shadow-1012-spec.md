# ML gate: the GOAL-label model in SHADOW (2026-10-12)

- **Slug:** `ml-goal-shadow-1012`
- **Status:** SHADOW. Logging only; no decision of the bot reads it.
- **Flag / rollback:** `ML_GOAL_SHADOW_ENABLED = True`. Rollback is `False`; the
  live path takes effect on the next restart and the nightly training on the
  next night.
- **Evidence (maximum-period backtest):** both runs below used the whole
  candidate history (57 588 candidates, 2026-03-24 .. 2026-10-12, walk-forward
  by month, 454 winner-days).
  - [`goal-aligned-ml-label-1012`](goal-aligned-ml-label-1012-spec.md): +7.05 pp
    of winner-days reachable before the crossing, refuted on the per-trade bound
    under row-level counting.
  - [`ml-label-followups-1012`](ml-label-followups-1012-spec.md) Test 1
    (exploratory): trades counted as the bot takes them gave +0.045 pp
    [−0.076, +0.154], which is shadow-worthy.
- **Truth harness:** TH-03, TH-05, TH-06, TH-07, TH-10, TH-11, TH-12
- **Readout:** `ML-GOAL-SHADOW` in `files/readout_registry.json`. The window runs
  from 2026-10-13 to the due date 2026-11-23.

## Why a shadow and not a switch

Both backtests used data that had already been looked at. The second one changed
how trades are counted after the first had failed. Neither modelled the gates
downstream of ML, so the +7 pp is a ceiling. The only honest confirmation is data
nobody has seen yet, with the downstream gates in place. A shadow produces that
data without changing what the bot sends.

## Components

1. **`files/ml_goal_shadow.py` — nightly training.**
   - **Rows.** The bot's candidates in `critic_dataset.jsonl`, deduplicated per
     (sym, tf, `ts_signal`). Features come from `ml_signal_model.build_feature_dict`,
     the same 60 the live gate uses.
   - **Label `goal_early_top20_v1`.** It is 1 when the decision (the close of the
     signal bar) lies in `[day open, first +2.5% crossing)` of an immutable
     global-top-20 winner-day of that coin (`immutable_labels.winners_by_day(...,
     rank_before_filter=True)`, `label_store.intraday_deadlines()`).
   - **Eligibility.** A row trains only if its decision day is RESOLVED: the day
     appears both in the EOD returns and in the crossing deadlines, and the
     decision is more than 1 day old. An unresolved day would turn every
     candidate on it into a negative.
   - **Model.** `StandardScaler` + the live `CatBoostModel` (300 trees, depth 4).
   - **Output.** Written atomically to `files/ml_goal_shadow_model.json` in the
     live payload format, so `predict_proba_from_payload` reads it unchanged. It
     also records `label_version`, `trained_at`, `n_rows`, `n_pos` and
     `first_day` / `last_day`.
   - **Invocation.** `daily_learning` calls it as a subprocess right after the
     live ML retrain, only when the flag is on. It is best-effort: a failure logs
     an error and never breaks the cycle.
2. **Live scoring.** `ml_goal_shadow.score(...)` builds the record exactly as the
   live gate does (`build_runtime_record` with the candidate's mode, bull-day
   flag and `btc_vs_ema50`).
   - The model file is re-read when its mtime changes, checked at most every
     `ML_GOAL_SHADOW_RELOAD_CHECK_SEC`.
   - It returns `None` (fails open) if the flag is off, the file is missing or
     unusable, or anything raises.
3. **Logging.** In `monitor.py`, at the ML gate, the shadow is scored in a worker
   thread once the floor is known and right before the live block decision. The
   event is `ml_goal_shadow` (`botlog.log_ml_goal_shadow`) with these fields:
   - `sym`, `tf`, `bar_ts`, `mode`, `price`;
   - `goal_score` (null if scoring failed);
   - `ml_proba`, `ml_min`, `ml_max`, `live_pass`;
   - `model_trained_at`.

   Only rows that reach the ML hard-block decision are logged. The shadow cannot
   change the fate of rows blocked earlier, so they are not its population
   (TH-06).

## Readout `ML-GOAL-SHADOW` — fixed now, before the data exists

**Population.** `ml_goal_shadow` events in the window, one per (sym, tf, bar_ts),
with the decision at bar open + tf.

**Matched admission.** For each ISO week, the shadow admits the top
`round(s·n)` rows by `goal_score`, where `s` is the share the live gate admitted
that week. Rows with a null score are never admitted by the shadow. The live gate
and the shadow therefore pass the same volume.

**Goal** (relax/tighten combined, as in `goal_validator`). Winner-days come from
`goal_validator.winner_days` on comparable days of the window.
- *now* = days with a live entry before the crossing.
- *lost* = such days whose every early entry came from a row the live gate
  admitted and the shadow would not. An entry with no logged shadow row is kept.
- *gained* = for days without an early entry, `1-(1-p)^k`. Here k is the number of
  shadow-only rows inside `[open, crossing)` while not in a position, and p is the
  share of live-admitted rows that became an entry.
- **Δ = (Σgained − lost) / winner-days**, with a 95% bootstrap over winner-days
  (1000 resamples).

**Per trade.** Every trade uses the calibrated live trail from the decision bar,
with the trail floor of its mode and `max_bars` 96 (15m) / 48 (1h).
- **current:** live entries, each with its own `trail_k` (median if missing).
- **removed:** entries the shadow would not admit.
- **added:** shadow-only rows taken as the bot would (one position per coin,
  `COOLDOWN_BARS` after exit, never inside a live position), at the median
  `trail_k`.

The statistic is `(Σcur − Σrem + p·Σadd) / (n_cur − n_rem + p·n_add) − mean(cur)`,
with a 95% day-cluster bootstrap.

**Verdict at the due date** (needs ≥ 28 comparable days and ≥ 20 added trades,
otherwise TOO_EARLY and re-read daily):

| verdict | meaning | condition |
|---|---|---|
| **KEEP** | propose switching the gate to the GOAL label, as a separate flagged change with its own floor | Δ ≥ +0.5 pp AND its lower bound > 0 AND the per-trade lower bound ≥ −0.10 pp |
| **ROLLBACK_SUGGESTED** | drop the GOAL-label line, flag off | Δ ≤ 0 OR the per-trade upper bound < −0.10 pp |
| **INCONCLUSIVE** | — | otherwise |

**Anomaly** (reported at once, before the due date):
- more than 5% of shadow rows have a null score; or
- 2 or more comparable days have zero shadow rows; or
- the model is older than 3 days at the latest event.

## Smoke test and power, written before the window opens

Run on 2026-10-12 with SYNTHETIC shadow events: the bot's real candidates that
reached the ML gate between 2026-09-15 and 10-10, live pass taken from the logged
stage, and a RANDOM `goal_score`. The readout ran end to end:
- 10 125 rows; the live gate admitted 49.2%; 126 winner-days, 11 entered early
  now; p = 0.19.
- Goal Δ −0.42 pp [−3.70, +2.43]. A random model shows no gain, so the readout
  does not manufacture one.
- Per trade −0.093 pp [−0.555, +0.278]; 943 current, 472 removed, 868 added.

**Power caveat (TH-10).** Over 25 days the per-trade interval is about ±0.42 pp.
KEEP needs a lower bound ≥ −0.10, so it requires a clearly positive point
estimate or more days. A real model overlaps the live gate more than a random one
does and removes fewer entries, which should narrow the interval, but by how much
is unknown. **INCONCLUSIVE at the due date is a plausible honest outcome.** In
that case the window is extended with a new registry entry and the criteria
unchanged. These criteria are NOT loosened.

## Tests

`files/test_ml_goal_shadow_1012.py`:
- the label and the resolved-day filter;
- the payload format and that `predict_proba_from_payload` reads it;
- fail-open scoring and the flag gate;
- reload on mtime;
- the botlog event fields;
- the readout's matched weekly admission;
- the goal arithmetic on synthetic events;
- the verdict and anomaly rules;
- the config defaults and the registry entry.
