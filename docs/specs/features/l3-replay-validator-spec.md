# L3 replay validator — the hypothesis loop could never close

- **Slug:** `l3-replay-validator`
- **Status:** DEPLOYED 2026-09-18
- **Truth-harness invariants:** TH-01 (base rate beside every ratio), TH-04
  (comparable windows — the two mistakes below), TH-05, TH-06 (the bot's own
  decisions), TH-08, TH-12, TH-13
- **Code:** `files/pipeline_replay_validator.py` (new), `files/pipeline_validator.py`
  (`dispatch` fallback), `files/pipeline_hypothesis.py` (L2 told what L3 can
  test), `files/bot_health_report.py` (failed LLM calls shown as failures)
- **Tests:** `files/test_pipeline_replay_validator.py`
- **Rollback:** revert `dispatch` to return `pending_manual_validation` — the
  replay module is only reached through it.

## Why the loop did not work

The operator's report said the LLM agent was last called 2026-09-06 and asked
why the learning loop was not working. Three separate causes, measured:

1. **The LLM agent had no API credit since 2026-08-30.** Both calls since
   (08-30, 09-06) returned `Your credit balance is too low to access the
   Anthropic API`. The report counted them as calls — "last call 09-06, 34
   total" — so the one line meant to show the agent's state said it was fine.
   Operator action: top up credits. Fixed here: the block now shows the last
   SUCCESSFUL call (2026-08-23) and a separate failure line with the cause.
2. **The 2026-09-13 weekly run never happened.** Windows booted at 20:46 that
   day (System log, Kernel-General id 12); at 05:30 the machine was off, and the
   task had `StartWhenAvailable = False`, so a missed start was never made up.
   Set to `True` on 2026-09-18 with operator approval.
3. **Even when everything ran, L3 never produced a verdict.** Eleven weekly runs,
   2026-06-21 … 09-06: every L3 verdict was `pending_manual_validation`, and L7
   read `hit_rate = 0.0` every week. `dispatch` looked validators up by `rule`,
   a free-form name L2 invents; three names were registered. The one hypothesis
   in the queue had waited since 2026-08-16.

Cause 3 is structural: fixing 1 and 2 alone would restart generation into a
queue that cannot drain.

## The fix

L2 always supplies `config_key` and `diff`. For a threshold whose compared value
is logged in `bot_events.jsonl`, the change can be replayed on the bot's own
decisions: apply the CURRENT and the PROPOSED value to the same logged inputs,
take the rows that change side, and grade their forward 4h PEAK against rows
that pass today — over the whole log, by month.

Decision rule, fixed in the module before any live hypothesis was graded:

| | accept | reject |
|---|---|---|
| relax | n ≥ 60, mean ≥ 1.0×, ≥5% tail ≥ 1.0×, months agree ≥ 60% | mean < 1.0×, tail < 1.0×, months agree < 60% |
| tighten | n ≥ 60, mean < 1.0×, tail < 1.0×, months ≤ 40% | mean ≥ 1.0×, tail ≥ 1.0×, months ≥ 60% |

Otherwise `needs_review`; n < 30 is `needs_data`, re-run next week. Accept means
the evidence supports it — approval stays with the operator. Nothing parks: an
unknown key or one the log cannot replay is **rejected with the reason**, and L2
now receives `validatable_config_keys` with live values and a rule to propose
only those.

Replayable today: `ML_GENERAL_HARD_BLOCK_MIN` / `_BULL_DAY_MIN` (relax and
tighten), `TREND_15M_QUALITY_*` forecast and alt thresholds (relax),
`TREND_1H_CHOP_*` thresholds (relax).

## Two mistakes the first version made, caught before any live verdict

1. **It used the day's decision instead of the logged value.** Floors and the
   model's scale moved over the period, so rows blocked under an old rule looked
   "admitted" by the new value — raising the ml floor 0.15 → 0.30 was graded a
   RELAX and accepted. Both sides are now judged by applying current and
   proposed values to the same inputs, and ml rows are limited to after
   2026-09-07 19:31, when the peak-label model's scale went live.
2. **It compared against an unmatched control.** Chop rejects are all trend/1h;
   the rows that pass are every 1h mode. Relaxing chop ADX 25 → 20 read as
   ACCEPT (1.27×). Only entries record their mode, so the control for the trend
   gates is now trend ENTRIES — which also passed every later gate, biasing
   against relaxing. Same change: 1.12×, months split 50% → `needs_review`.

The decision rule was also made symmetric after the first run: reject had
ignored month consistency while accept used it.

## Verdicts on the first run

| change | band n | mean vs pass | ≥5% tail | months | verdict |
|---|---|---|---|---|---|
| `TREND_15M_QUALITY_FORECAST_MIN` 0.25 → 0.15 (the stuck one) | 107 | 0.80× | 0.69× | 0% of 2 | **reject** |
| `ML_GENERAL_HARD_BLOCK_MIN` 0.15 → 0.10 | 238 | 0.57× | 0.33× | 0% of 1 | reject |
| `ML_GENERAL_HARD_BLOCK_MIN` 0.15 → 0.30 | 136 | 1.12× | 1.25× | 100% of 1 | reject (would remove better rows) |
| `TREND_1H_CHOP_ADX_MIN` 25 → 20 | 255 | 1.12× | 1.25× | 50% of 6 | needs_review |
| `TREND_1H_CHOP_VOL_MIN` 1.3 → 1.1 | 76 | 0.76× | 0.15× | 0% of 1 | reject |

The hypothesis waiting since 08-16 was closed by running L3 on it: `rejected`.
Note its own rationale did not hold — forecast 0.000 stays below 0.15, so the
AAVE/ENA rows it cited would have stayed blocked.

## Not covered

- Keys whose compared value is not logged (cooldowns, lateness caps, exits)
  cannot be replayed and are rejected as unvalidatable; each needs its own spec
  or its inputs logged.
- L7 `hit_rate = 0.0` for eleven weeks is unexplained here; it may be re-counting
  the same eleven old decisions. Unverified.
