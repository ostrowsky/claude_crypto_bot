# Model health in the morning report (2026-10-05)

Status: **deployed, reporting only** (`MODEL_HEALTH_ENABLED = True`; rollback = `False`).
Code: `files/model_health.py`, tests `files/test_model_health.py` (9). Wiring: pipeline step
"model health" (before notify, `pipeline_run.py`), block in `pipeline_notify.build_agent_blocks`,
run manifests around the nightly retrains in `daily_learning.run_full_cycle`.
Outputs: `.runtime/pipeline/model_health/{health.jsonl, run_manifests.jsonl, backfill.jsonl}`.
Truth-harness rules: TH-01 (base rates), TH-02 (label partly visible), TH-05 (knows what it
cannot see), TH-06 (bot's own candidates), TH-10, TH-11 (proxy vs the model's own target).

## Why

The existing drift step (`pipeline_drift.py`, RM-14) carried no information:

| fact (166 daily runs, 2026-05-19 .. 10-05) | value |
|---|---|
| runs that said `drift_detected` | **164 of 166** |
| `ranker_proba` runs with any data | **0 of 166** — critic_dataset never logs it |
| consumers of its output | none (not in health report, not in notify) |

A two-sample KS test on thousands of rows calls every small shift significant, and it tracked
model OUTPUTS, which move whenever a model is retrained (nightly). Nothing measured whether
`ml_proba` still does its job, and nothing recorded what a retrain changed.

## What is measured

Population: critic rows that reached the ML gate (`decision.ml_proba` not None).

- **Quality against the model's own target.** Since 2026-09-07 `ml_signal_model` is trained on
  the peak label (high within `ML_PEAK_LABEL_HORIZON` = 5 bars ≥ `ML_PEAK_LABEL_THRESHOLD_PCT` =
  2 %). Per day: AUC of `ml_proba` for that label (resolved with `ml_signal_model._peak_ahead`, its
  cache read-only); 14 days pooled with a day bootstrap. Alert when the 95 % interval is below
  0.5; watch when it contains 0.5.
- **Reference only, never alerted on:** Spearman IC with the forward 10-bar return (`labels.ret_10`)
  and AUC for "the coin ends the day in the immutable global top-20" with its base rate.
- **Input drift.** PSI (reference-quantile bins) of the market features the models read — ATR,
  daily range, volume ratio, RSI, ADX, slope, EMA50 distance, candle body, BTC vs EMA50 — last 7
  days vs the 30 before. Shift at PSI ≥ 0.25; **alert only when a shift persists 3 days**.
  `btc_vs_ema50` is a regime indicator: reported, never alerts.
- **Run manifests.** One line per nightly retrain: model sha256 before/after (changed?), git HEAD
  and dirty-file count, `config.py` sha256, dataset bytes/mtime, the trainer's summary.

## Validation (backfill 2026-05-20 .. 10-05, 139 days, 32 430 ML-gate rows, 32 248 peak-resolved)

Development path, in order — each step was judged on this backfill:

| version | alert days | why it changed |
|---|---|---|
| PSI on model outputs + IC vs `ret_10` | 77 / 139 (55 %) | output drift = model changes (09-07 label switch: PSI 2.16); `ret_10` is not the target |
| PSI on inputs (3-day persistence) + AUC vs own target | 108 / 139 (78 %) | `btc_vs_ema50` alone held the alert 99 days |
| **same, `btc_vs_ema50` reported but not alerting** | **40 / 139 (29 %)** | final |

The final input-drift alerts fall in four episodes (10–12 Jun, 16–26 Jul, 24–30 Aug, 11–23 Sep),
each a volatility-regime change (daily range / ATR). A rule requiring two features at once would
have cut alerts to 14 %; it was NOT adopted because it would be tuned on the same history.

Quality, current model (14-day windows fully after the 09-07 switch): **AUC 0.72–0.74 against
its own target over 15 days from 09-21, intervals within [0.67, 0.79], target base rate 17–21 %** — the model does what it is
trained for. Before 09-07 the live model was trained on another label, so its backfilled
"inverted / no signal" days against the peak label are not interpretable as failures.

What the plan's acceptance asked ("alerts fire on known incidents, e.g. 08-20") is only partly
met: the 08-20 ML blackout falls inside a quality "inverted" stretch (08-20 .. 08-27, except 08-25), but that
period predates the peak label, so the alert cannot be attributed to the blackout. The 09-07
label switch is a model change and shows in run manifests, not in input drift — by design.

## Finding worth acting on (reference metric)

IC of `ml_proba` with the forward 10-bar return is **−0.14 [−0.21, −0.08]** over the last 14
days: the gate prefers candidates with a large upside excursion (its target) whose price two and
a half hours later is, on average, worse than for low-scored candidates. This ties directly to the
North Star's weak factor (capture 0.22–0.25) and is an input for the exit / leader work, not a
reason to change the gate by itself.

## Open

- `ranker_proba` is not logged in critic_dataset (logging gap; `pipeline_drift.py` has been blind
  to it since May). `pipeline_drift.py` is left running unchanged; retiring it is a separate decision.
- `test_pipeline_notify.RollbackBlockTests.test_none_when_no_log` fails independently of this
  change: it reads the real rollback log instead of a temp tree.
