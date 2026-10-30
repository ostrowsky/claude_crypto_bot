# CLAUDE.md — Guide for Claude working in this repo

Auto-injected into Claude Code's system prompt. Read first, trust it, don't re-discover.
**Companion:** `PROJECT_CONTEXT.md` — the long-form project dossier. Keep both in sync: when architecture, filters, schedules, or known issues change, update BOTH files.
**Map:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — one page covering all four planes, what is built vs designed vs broken, and the spec index.

---

## 0. Founding principle — continuous learning is P0

**The bot's trading algorithms must continuously learn from data in order to
improve the bot's purpose metrics (see §1). If the data needed to learn is
missing, collecting it is part of the work, not a separate effort.**

Continuous self-improvement of the algorithms is the project's **first-priority
task**. Every subsequent change — config tweaks, refactors, new features —
**must be validated for alignment with this priority before merging**. The
specific question to ask before any change:

| Question | Verdict |
|----------|---------|
| Does this change improve the bot's ability to learn from data? | ✅ preferred |
| Does this change leave the learning loop unchanged? | 🟡 allowed |
| Does this change reduce, slow, or fragment learning? | 🚨 must justify or refactor |

Concrete consequences:

1. **A change blocked by missing logging is a logging-fix-first task.** Don't
   speculate from incomplete state; instrument the bot.
2. **Bypassing the pipeline (manual config edits without an approved
   hypothesis, model retrains decoupled from pipeline decisions, ad-hoc
   filter tweaks) weakens the loop.** Do it only with documented justification.
3. **Training metrics that diverge from deployment metrics are the most
   important diagnostic.** Never collapse them into a single number; never
   hide the gap.
4. **Memory is mandatory.** What was tried, what was approved, what was
   rolled back — these go in `decisions.jsonl` and `already_tried.jsonl`.
   Skipping the write is a regression.

Two documents cover the improvement loop, and they have different jobs:

- **As-built / legacy:** [`auto-improvement-loop-spec.md`](docs/specs/features/auto-improvement-loop-spec.md)
  — what is wired up today and how each piece was validated. Still living: every
  PR touching a loop component updates its status row and the North Star table.
  Its RM-4/RM-5/RM-18 auto-execution and canary items are **retired**.
- **Target architecture:** [`continuous-improvement-agent-spec.md`](docs/specs/features/continuous-improvement-agent-spec.md)
  — what to build toward. **Where the two disagree, the target wins.**

One-page map of the whole system: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## 0a. Truth harness — how to not fool ourselves

Every number in this project has at some point looked good and meant nothing.
These rules exist because each was violated and cost real time. **A change is not
done until it answers them.**

### The rules

1. **A ratio without its base rate is not evidence.** `recall@20 = 100%` was
   celebrated for months; the bandit reaches it by firing ENTER on **73% of
   everything** — lift 1.36. Always publish base rate and lift next to any
   recall / coverage / hit-rate.
2. **Name the features that already contain the answer.** `top_gainer_model`
   scores AUC 0.99 while `tg_return_since_open` is an input and the label is "was
   the day's return top-20". That model confirms a move in progress; it does not
   predict one. If a feature could encode the label, say so in the same breath as
   the score.
3. **In-sample numbers are never an achievement.** Report holdout only, split by
   TIME, never at random — rows of the same day leak into each other.
4. **Compare only comparable windows.** A 5-working-day window against a 10-day
   one produced a fake "СТАЛО ХУЖЕ" after the outage. Endpoints must match in
   sample size, or the answer is "рано судить".
5. **A metric must know what it does not know.** Days the bot was down counted as
   misses until 2026-08-05 — an outage read as a performance collapse. Absence of
   data is not evidence of failure.
6. **Validate a gate on the bot's OWN entries, not on the market.** The
   weak-extension filter was strong across all market episodes and worthless on
   the bot's entries: the gates upstream mean it never samples that population.
7. **Behaviour changes ship behind a flag with a stated rollback**, and are
   measured in SHADOW first whenever the shadow can answer the question.
8. **Negative results are committed**, with the numbers that killed them. Nine
   hypotheses in this repo were refuted; re-testing them is waste.
9. **Every claim in this file must be machine-checkable.** `_audit_md_vs_config.py`
   verifies flags and dataset sizes; a claim that cannot be checked should be
   phrased so it can be.
10. **A report may not state a conclusion the data does not support.** "Рано
    судить" is a valid answer and is preferred over an invented trend.
11. **Proxy is not the business outcome.** Training recall/AUC cannot answer
    whether the bot catches moves or earns money; per-mode P&L cannot replace
    canonical portfolio alpha.
12. **Every change is traceable to evidence.** Spec, focused tests, validation
    result and applicable `TH-01`…`TH-12` IDs must travel together.

13. **Never assert without proof — verify, then speak.** A claim about what
    is running, what a flag does, or why something behaves as it does is not
    knowledge until it has been checked against the system itself. Three
    failures on 2026-08-20 alone: "the version banner is just stale git
    metadata" (asserted twice before checking process start time against file
    mtime — it was stale, but only proving it made the answer worth anything),
    "segment routing is now off" (the flag was set in a function the live path
    never reaches; the live median proba had not moved), and a top-40 ranking
    computed on the futures universe and presented as the operator's spot
    screen. The cost of checking is a single command; the cost of not checking
    is a wrong diagnosis acted on. If a check is impossible, say the claim is
    unverified rather than stating it flat.

### Enforcement

- `pyembed\python.exe files\truth_harness.py full` — full current-state audit;
  `_harness_check.py` remains an additional MD/config/live-state checker.
- `.githooks/pre-commit` runs `truth_harness.py change --staged` plus
  `git diff --cached --check`. Enable it with
  `git config core.hooksPath .githooks` in every worktree.
- Project skill: `skills/crypto-bot-truth-harness/SKILL.md`; use it for the
  judgment checks that cannot be established by static code. The lighter
  `.claude/skills/md-compliance/SKILL.md` remains available for MD-only audits.
- Normative contract: `docs/specs/features/truth-harness-spec.md`.
- Every PR/commit touching behaviour or metrics states which rule it satisfies
  (usually 1, 3, 6, 7).
- When a rule is violated on purpose, say so in the commit message with the
  reason — silence is the failure mode this section exists to prevent.

---

## 1. Purpose & success metric

Telegram bot scanning Binance Futures/Spot in real time, generating early BUY signals on coins that will be in the daily top gainers by EOD.

**Main metric (North Star):** earliest possible BUY on a coin that ends up in
the immutable later-EOD top-20 daily gainers, before the main move.

`EarlyCapture@top20 = mean(coverage × realized_capture × time_lead)` over
winner-days in `watchlist ∩ global-top20`; target **0.40**, acceptable floor
**0.25**. The older binary `watchlist_top_early_capture_pct` from critic reports
is a separate deployment diagnostic, not the North Star. Until immutable
later-EOD labels replace the current rolling-24h same-snapshot labels, the
computed North Star must be marked **provisional** and cannot prove progress.

Since 2026-08-14 the metric is also published on immutable later-EOD labels from
exchange klines (`NS_IMMUTABLE_LABELS_ENABLED`, `files/immutable_labels.py`) —
**beside** the old value, never instead of it. Between 08-14 and 08-17 it carried
`immutable_comparable_to_primary = False`, because the store held only watchlist
symbols and the two values had different denominators. That is no longer the case
— see the update below. The remaining condition for lifting `provisional` was
that `train_top_gainer` also stop using the snapshot label; that flag went **ON
2026-08-17**, so the condition is met: the primary metric is now
`NS_EarlyCapture_top20_v2` on immutable labels and the paragraph above describes
history, not the current state.

**The primary metric is `NS_EarlyCapture_top20_v2` since 2026-08-17** — computed
on immutable later-EOD labels, `label_provenance = immutable_later_eod_klines`,
denominator `global_top20_intersect_watchlist_from_label_store`. It is
**versioned, not redefined**: the old value keeps travelling in the same payload
as `legacy_early_capture` / `legacy_n` so the historical series stays
reconstructable (TH-04). Current 30d reading: **v2 = 0.129 (n=28, cov 0.82)**
against legacy 0.072 (n=38, cov 0.58). Rollback: `NS_IMMUTABLE_LABELS_ENABLED=False`
restores the old value as primary.

**Updated 2026-08-17.** The label store now covers the **global** USDT universe
(734 pairs fetched, 530 resolved, ~497/day over 240 days; `build_global_labels.py`),
so `watchlist ∩ global-top20` is reproducible from exchange klines and the two
North Star values are comparable. `immutable_comparable_to_primary = True`.
Maximum period (109 days): leaky **0.0727** (cov 0.62) vs immutable **0.1041**
(cov 0.75); last 30d 0.072 vs 0.129. The honest number is HIGHER because the old
label ranked a rolling 24h window and counted overnight spikes — moves the bot
could not catch — as misses. Still far from the 0.40 target; what changed is that
it can be trusted. Delisted pairs remain unrecoverable (TH-05).

### Goal update: daily revenue — in a separate trader app (operator, 2026-10-10)

The operator's next phase: a bot that trades a Binance **demo** account by itself (buy,
exchange-side TP/SL, sell) with **daily revenue** as its target metric. It is built as a
**separate application, `trader/`**, and must not modify this signal bot (operator,
2026-10-10). This bot's metrics and its North Star are unchanged. Baseline that frames the
trader's work (`trader/_backtest_daily_revenue_baseline.py`, 5 753 real signals of this bot,
03-03..10-10): trading them as-is loses −0.83%/day after 0.20% fees, all 8 months negative; the
E1 equity replay with 8 bps/side costs gives −1.17%/day on Jul–Oct. Roadmap:
`docs/specs/features/autonomous-trading-roadmap-1010-spec.md`.

### Goal update: leaders only (operator, 2026-09-27)

"Catch one, at most three day-leaders and hold them as long as it takes,
collecting their rise; no more junk signals." Measurable form: at most 3
entries a day, each into a coin that ends the day among the leaders, share of
the leader's move captured (entry to exit vs day open to peak). The literal 97%
is not reachable by any rule (it needs the day's low and high in hindsight);
it is recorded as the direction. First test (`leader-mode-spec.md`): entering
the CURRENT day-leader and holding it on a wide trail gives ~1 trade a day,
38-50% of them real leaders, median trade -2.7..-3.9%, not ready.

### The target segment: rockets (operator, 2026-09-26)

**A rocket is the bot's real target, not an exception.** Definition: a watchlist
coin that rises >= +10% from the UTC open within the day AND holds it — the day
closes at >= 60% of that rise. Examples: FIL 2026-09-26 (+16.8%, close +15.8%),
QNT 09-24 (+29.3% / +27.5%), STRK 09-18 (+56.2% / +52.6%). All three gave
hours between the first +2.5% and +10%. The bot must catch these early, hold
them while the trend lasts, and not raise false alarms.

What the data says so far (`docs/specs/features/rocket-segment-spec.md`, 455
days, 101 coins, split by time): 1 321 rockets, ~3.1 a day; at a coin's first
+2.5% close only **7.7%** of such days become rockets. Ten hypotheses (R-1..R-10:
timing, relative volume, volatility, compression, breakouts, decoupling,
momentum history, the path after +2.5%, a model over all of them, taker flow)
reach at most ~13% precision at +2.5% on the holdout (2x the base), and no rule
survives the time split (train 36-42% -> test 10-14%). With price, volume,
time and taker flow a rocket is **not** separable early without false alarms,
and alerts are not profitable either. The route left is new data (derivatives
positioning is being recorded by `positioning_recorder.py`) — and, on the exit
side, a wide trail: on rockets an 8% floor keeps +9.6% vs +2.6% for k=2.

- Project root: `D:\Projects\claude_crypto_bot\`
- **DO NOT TOUCH** `D:\Projects\gpt_crypto_bot\` — separate bot, separate Telegram token, independent process. Always check cmdline before killing any python PID.

---

## 2. Architecture

### Signal pipeline
```
Binance API -> Indicators -> Strategy (7 entry modes)
    -> ML gating (CatBoost)
    -> Ranker (candidate score)
    -> Contextual Bandit (enter/skip)
    -> Guards (trend quality, impulse speed, ranker veto, correlation guard)
    -> Portfolio rotation (ML-gated weak-leg eviction)
    -> Entry signal -> Telegram
```

### Monitored set — the ENTIRE watchlist (2026-08-12)

`MONITOR_FULL_WATCHLIST = True`: every watchlist coin is polled, so "the coin was
not being watched" is no longer a possible cause of a miss.

Before this the set was curated and `bot.py::_update_hot_coins` **rewrote**
`state.hot_coins` on every auto-reanalyze (30 min), wiping every promotion
(discovery / decoupling / soft). Observed: 18 coins -> 9 within 11 minutes, and
POL went 15 days without a single scan. The scan already analyses the whole
watchlist (`in_play + skipped` = all coins), so keeping them all costs no extra
request; symbols the scan lost to a transient `fetch_klines` None are re-added as
stub `CoinReport`s (`_poll_coin` fetches its own klines).

Cost is bounded by ROTATION, not by set size: `MAX_POLL_PER_CYCLE = 45` coins per
60s tick -> full sweep of ~105 in ~3 cycles (~3 min). **Coins with an open
position are polled every cycle** regardless of rotation — trailing stops cannot
wait. Rollback: `MONITOR_FULL_WATCHLIST = False`.

Superseded by this and disabled to avoid dead work: `SOFT_PROMOTE_ENABLED=False`,
decoupling promote-to-scan skipped (decoupling scoring/shadow logging untouched).

### 7 entry modes
`trend` (15m) · `strong_trend` (1h) · `retest` (1h) · `alignment` (1h MTF) · `impulse` (15m) · `impulse_speed` (1h) · `breakout`

### Key source files

| File | Role |
|------|------|
| `files/bot.py` | Telegram UI, entry point. `_sorted_position_items()` L469–506 = 10-field portfolio sort. |
| `files/monitor.py` | Main live loop. Rotation wiring ~L4072. |
| `files/strategy.py` | All 7 entry modes. |
| `files/config.py` | 300+ tunables. Single source of truth. Edit → restart. |
| `files/rotation.py` | ML-gated weak-leg eviction. `should_rotate`, `find_weakest_leg`, `evict_position` (`trail_stop = price*1.001`). |
| `files/correlation_guard.py` | Pearson log-return clustering (Union-Find). |
| `files/trend_scout_rules.py` | Scout `BlockRule` registry. New gates go here. |
| `files/contextual_bandit.py` | LinUCB bandit (entry + trail_k). |
| `files/offline_rl.py` | Offline bandit training. |
| `files/daily_learning.py` | Daily learning orchestrator. |
| `files/rl_headless_worker.py` | Background RL worker (ranker training). |
| `files/ml_candidate_ranker.py` | CatBoost ranker. Model blob: `ml_candidate_ranker.json`. |
| `files/ml_signal_model.py` | Signal classifier. Trains logistic, MLP and CatBoost and keeps whichever wins on validation — **not** CatBoost-only, as this line said until 2026-08-20. Currently MLP wins. |
| `files/top_gainer_critic.py` | Evaluates quality of top-gainer predictions. |
| `files/train_top_gainer.py` | Trains `top_gainer_model`. |
| `files/indicators.py` | Technical indicators. |
| `files/report_rl_daily.py` | Daily RL + bandit metrics report. |

### Runtime files

| File | Role |
|------|------|
| `.runtime/bot_bg.json` | Bot PID |
| `.runtime/rl_worker_bg.json` | RL worker PID |
| `.runtime/tg_send_dedup.json` | Telegram dedup state |
| `.runtime/learning_progress.jsonl` | Daily learning metrics history |
| `files/top_gainer_dataset.jsonl` (~147 MB) | Features of ALL watchlist coins × N daily snapshots |
| `files/critic_dataset.jsonl` (~185 MB) | Bot signals with outcomes (nested: `decision.action`, `decision.reason_code`, `labels.ret_5`, `labels.label_5`) |
| `files/bot_events.jsonl` (~150 MB) | All bot events (entry / blocked / exit) |
| `files/ml_dataset.jsonl` (~115 MB) | Raw ML training data |

### Tests / backtests
- `files/test_rotation.py` — 21 unit tests for rotation.
- `files/backtest_portfolio_rotation.py` — scenario sweep.
- `files/backtest_portfolio_rotation_grid.py` — Pareto grid (score × ml_proba). **Reuse this pattern** for any new gate sweep.

---

## 3. DO NOT read these directly

Large jsonls, model blobs, logs, and `pyembed/` are **blocked at the harness level** via `.claude/settings.local.json` deny-rules. Paths denied include:
`ml_dataset.jsonl`, `critic_dataset.jsonl`, `top_gainer_dataset.jsonl`, `bot_events.jsonl`, `ml_candidate_ranker.json`, `top_gainer_model.json`, `pyembed/**`.

To analyze them: **write a small Python script** that streams/aggregates and prints only the summary. Mirror the style of `backtest_portfolio_rotation_grid.py`.

Rule of thumb: if a file is >1 MB or lives in `.runtime/`, use scripted aggregation, not `Read`.

---

## 4. ML / RL system

### Contextual Bandit (LinUCB)
- **Entry bandit** — 2 arms: SKIP=0, ENTER=1 (main gate). Alpha=2.0 (aggressive exploration).
- **Trail bandit** — 5 arms, picks `trail_k` + `max_hold` after ENTER.
- Context: `slope_pct, adx, rsi, vol_x, ml_proba, daily_range, macd_hist, btc_vs_ema50, is_bull_day, mode, tf`.

### Asymmetric reward scheme

"Worth entering" = the row is in its day's top-`BANDIT_FORWARD_TOP_N` by the
move **still ahead of the snapshot** AND that move clears
`BANDIT_FORWARD_MIN_PCT`. See "Training-label leak fix" below for why it is no
longer "the coin was a top-20 gainer".

| Situation | Reward |
|-----------|--------|
| ENTER + worth entering | +1.0 |
| SKIP + worth entering | **-0.8** (miss penalty) |
| SKIP + nothing left to catch | +0.10 |
| ENTER + nothing left to catch | -0.12 |
| ENTER + fast reversal (≤3 bars, P&L < -0.5%) | **-0.6** (planned, see § 4a) |
| SKIP + fast reversal | **+0.30** (planned) |

### Training-label leak fix (2026-08-13) — TH-01/TH-03/TH-04

The bandit was paid +1.0 for ENTER whenever a row carried `label_top20`, and
`use_earliest_snapshot=True` picks the earliest record of a calendar day — which
is the 00 UTC one, i.e. the EOD resolution of a day already **over** (41% of
trained rows). Measured on the rows it trained on: 34.8% had
`|eod_return − return_since_open| < 0.5%` (nothing left to decide) and the median
move still ahead was **+1.75%**. A fresh bandit trained on that label scored
**lift 0.65×** on a temporal holdout — worse than random. That, not skill, is
where "recall@20 = 100%" (§0a rule 1) came from.

Two defects in the same path had to go with it:
- `_load_top_gainer_dataset` read the **first** 50k of 118 625 lines, freezing
  the training window at 2026-06-05 for 69 days. Now reads the tail.
- `train_entry_bandit` did `batch_update` onto the saved state every run: 8.39M
  updates from ~44.6k unique samples (~188× re-ingestion), which baked the old
  label into `A/b` and shrank the exploration bonus without new evidence — so
  "UCB separation growing" measured nothing either. Now rebuilds from scratch.

Label comparison (309 days, split by time, holdout 9765 rows; target = the coin
still has ≥ +3% ahead, base rate 2.0%) — `files/_backtest_bandit_leak_fix.py`:

| training label | ENTER | caught | precision | lift |
|---|---|---|---|---|
| old `label_top20` | 80.3% | 52% | 1.3% | **0.65×** |
| rank top-20, no floor | 98.0% | 100% | 2.1% | 1.02× |
| rank top-20 + floor +3% | 43.5% | 99% | 4.6% | 2.28× |
| **rank top-10 + floor +3%** | **24.3%** | **99%** | **8.3%** | **4.07×** |
| floor +3%, no rank | 80.0% | 100% | 2.5% | 1.25× |

The floor is load-bearing: a pure rank label mints exactly N winners a day
whatever the market does and collapses to "ENTER on everything" (1.02×).

Deployed: label + rebuild + tail window, state rebuilt (37 144 updates,
window 2026-06-23..08-13, base rate 7.5%, 6 289 already-decided rows dropped).
Old state kept at `files/bandit_entry_state.pre_leakfix.json`.
Rollback = `BANDIT_FORWARD_REWARD_ENABLED=False` + restore that file.

Caveat (§0a rule 6): measured on the dataset population — every watchlist coin
at a daily snapshot. Live, the bandit judges candidates that already passed the
upstream gates, so the live effect must be read off decision logs, not assumed.

### 4a. Anti-fast-reversal training requirement (2026-04-25)

The model must **avoid emitting BUY when a quick SELL is likely**. Currently 53.7% of `alignment` entries close within 3 bars at avg P&L -0.348% — these are unactionable for a human reader of the Telegram channel.

**Required pieces:**
1. **Label**: `label_fast_reversal` in `critic_dataset.jsonl` and `top_gainer_dataset.jsonl` — `1` if next 3 bars after entry hit `entry × (1 - ATR_pct × trail_k)`.
2. **Model output**: `proba_fast_reversal ∈ [0,1]` (separate head or new CatBoost classifier).
3. **Guard**: `_fast_reversal_guard_reason()` in `monitor.py`. Block if `proba_fast_reversal > FAST_REVERSAL_PROBA_MAX` (default `0.55`). Reason code: `fast_reversal_risk`.
4. **Bandit context**: add `proba_fast_reversal` to LinUCB context vector.
5. **Reward update**: see table above.

Implementation order: label → train → backtest (60d) → wire guard. Do NOT enable guard before backtest confirms recall@top20 stays ≥ current.

### Training sources (entry bandit)
1. **Primary:** `top_gainer_dataset.jsonl` — ALL ~105 watchlist coins × N daily
   snapshots. Read as the TAIL (`BANDIT_TG_MAX_RECORDS`), graded by
   `offline_rl._build_universal_rows` — the same helper the evaluation uses, so
   training and grading cannot drift apart.
2. **Secondary:** `critic_dataset.jsonl` — real bot signals with outcomes.
   Still labelled by per-day rank of `ret_10` (8.4% of records marked positive);
   rank-only labelling has the base-rate-by-construction weakness described
   above, so this source is the next one to re-examine.

### Top Gainer Model (CatBoost)
Tier classifiers: top5 / top10 / top20 / top50. Retrained daily via `daily_learning.py`. Metrics: AUC, recall@0.3.

**The reported AUC of ~0.99 is leakage, not skill.** The tier labels are written
from the same rolling-24h snapshot that produces the features, so
`tg_return_since_open` is an input and very nearly the answer (TH-03). Trained on
immutable later-EOD labels instead, the same features score **0.83** — that is
the honest level (`files/_backtest_immutable_training_labels.py`, 120 088 rows).

**`TRAIN_IMMUTABLE_LABELS_ENABLED` and `TRAIN_DAY_GROUPED_SPLIT_ENABLED` are ON
since 2026-08-17**, together, so one measurable change is attributable. Live AUC
0.925/0.921/0.898/0.837, `evaluation_scope = day_grouped_holdout_immutable_later_eod_label`.
Rollback: flip either flag and restore `files/top_gainer_model.pre_immutable.json`.

**`top_gainer_model.json` does NOT feed the ranker hard veto** — an earlier
version of this file said it did. `ranker_top_gainer_prob` comes from a
top-gainer model trained inside `ml_candidate_ranker.py` and serialised into
`ml_candidate_ranker.json`. `top_gainer_model.json` is loaded by
`enhanced_signals` and becomes `top_gainer_bonus`, added to the entry score.
(Over 24h `ranker_hard_veto` fired 0 times in 2 573 blocks — the four gates above
it take everything.)

**The bonus ladder is calibrated per model since 2026-08-17.** `score_bonus` used
fixed cuts (0.3/0.3/0.35/0.4) implicitly tuned to the leaky model's peaked
probabilities; on the honest model those same cuts fired on 37.1% of candidates
instead of 12.2% — a 3× loosening of the entry gate that would have shipped
silently at the 02:30 retrain. `_compute_metrics` now emits a per-tier
`bonus_threshold` at the base-rate quantile and the ladder reads it. Net effect
+0.7pp. Evidence: `docs/specs/features/top-gainer-immutable-training-labels-spec.md`.

### Learning progress (rebased 2026-08-13)

Everything before this date was measured against `label_top20` and is **not
comparable** with what follows: those recall figures (99–100% from Apr 07 on),
the UCB-separation growth and the top20 AUC of 0.93 were produced by the leaky
label plus repeated re-ingestion of the same rows. They are kept out of this
table on purpose — deriving a trend across the label change would be exactly the
incomparable-windows failure of §0a rule 4.

Current, from an honest time holdout (`evaluate_bandit_accuracy` now fits a fresh
bandit on the earlier days and grades the last 7):

| Date | scope | recall | ENTER rate | base | lift |
|------|-------|--------|-----------|------|------|
| Aug 13 | out-of-sample, 45 train days / 7 eval | 74.2% | 35.3% | 8.1% | **2.10×** |
| Aug 13 | in-sample echo (same live bandit) | 85.5% | 57.8% | 8.1% | 1.48× |

The two rows exist to stay visible against each other: the in-sample echo reads
higher on recall and worse on lift, which is the signature §0 rule 3 asks never
to be collapsed into one number.

`bandit_n_signal` once hit a hard cap of `8000` and stuck → `BANDIT_CRITIC_MAX_RECORDS=25_000`
in `config.py`, wired into `offline_rl.py`.

### Pareto sweep of Scout gates (`files/analyze_blocked_gates.py`)
Key find: actual `take` entries avg_r5 = **-0.016%**, but multiple `blocked` buckets are positive:

| gate | n | avg_r5 | win% | verdict |
|------|---|--------|------|---------|
| `impulse_guard` | 550 | **+0.641%** | 40.0 | over-blocking HARD |
| `entry_score` | 1374 | +0.123% | 50.3 | over-blocking (largest n) |
| `ranker_hard_veto` | 787 | +0.128% | 46.3 | still over-blocking |
| `clone_signal_guard` | 475 | +0.128% | 51.4 | over-blocking |
| `trend_quality` | 433 | +0.127% | 46.7 | mild over-blocking |
| `ml_proba_zone` | 1340 | +0.009% | 44.1 | neutral (vs take -0.02%) |
| `open_cluster_cap` | 136 | -0.360% | 35.3 | **working correctly** |
| `mtf` | 227 | -0.208% | 37.0 | **working correctly** |

To re-run: `pyembed\python.exe files\analyze_blocked_gates.py` from repo root.

---

## 5. Scheduled tasks (Windows)

| Task | Time | Action |
|------|------|--------|
| `CryptoBot_DailyLearning_EOD` | 02:30 local (00:30 UTC) | Full cycle: snapshot → **refresh immutable label store** → train bandit → retrain model → report. The label store step was added 2026-09-21: it had been built once (2026-08-17), never extended, and the North Star's primary value plus `top_gainer` training both degraded silently (see `docs/specs/features/label-store-refresh-spec.md`). The old "resolve pending bandit decisions" step is gone. |
| `CryptoBot_IntradaySnapshot` | 08:30 / 14:30 / 20:30 local | Feature snapshot → `top_gainer_dataset` |
| `CryptoBot_KlinesBackfill_Daily` | 06:00 local | `history/<sym>_15m.csv` refresh (`--days 30 --tf 15m`) |
| `CryptoBot_KlinesBackfill_1h_Daily` | 06:20 local | `history/<sym>_1h.csv` refresh (`--days 60 --tf 1h`) — **added 2026-08-17** |
| `CryptoBot_EarlyRankingShadow_Daily` | 02:50 local | Shadow top-k for goals 1-2 from the 00 UTC snapshot; logs only — **added 2026-08-19** |


**Nothing refreshed 1h until 2026-08-17.** The 06:00 task runs `--tf 15m` only,
so `history/<sym>_1h.csv` froze at 2026-06-20 — 58 days stale — while the 15m
cache stayed current. Nothing errored: a task never asked to do 1h cannot fail at
it. The cost showed up two layers away, in EX1's canonical mode matching **0 of
136** trades on 1h, because no uptrend can overlap a trade happening two months
after the data ends. The freshness check itself is fine (`cache_covers_window`
re-fetches anything whose last bar is over 6h old), so `--skip-existing` will not
re-introduce staleness — the gap was purely that no schedule invoked it.

**Scheduler gotcha (fixed 2026-08-13):** every CryptoBot task was created with
`DisallowStartIfOnBatteries` / `StopIfGoingOnBatteries` = True, so on battery the
weekly pipeline simply never ran (`LastTaskResult 2147946720`) — the hypothesis
generator was silent 11 days and the morning report showed an empty queue with no
explanation. All tasks now have both flags off. If a task reports that code again,
check power settings before debugging the code.

---

## 6. Process / bot lifecycle

### Start / restart
```
restart_bot.bat          - full restart (bot + RL worker)
start_bot_bg.ps1         - bot only
start_rl_worker_bg.ps1   - RL worker only
```

### Stop
```
stop_bot_bg.ps1          - stop bot (PID from .runtime/bot_bg.json)
stop_rl_headless.bat     - stop RL worker (PID from .runtime/rl_worker_bg.json)
```

### Python / cwd
- Bot uses **embedded Python:** `pyembed\python.exe`.
- Working dir for bot: `files\`. All imports relative to it. Run Python scripts from `files\`, not repo root.
- Logs: `bot_stdout.log`, `bot_stderr.log` (repo root).
- Only ONE main bot instance at a time (Telegram token conflict).
- Windows logs are cp1251 — keep `.bat` and printed strings ASCII-only.

### Restarting from a Claude session: start detached (2026-09-28)
A bot started from a Claude Code tool call lives in the Claude desktop app's
process tree and dies with it: on 2026-09-28 Windows updated the app (20:19) and
the bot stopped at 20:18:48 with no traceback and no crash event, while the RL
worker (started from a user cmd window) survived. From a Claude session, start
the bot through the WMI service so it has no Claude ancestor:
`Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File D:\Projects\claude_crypto_bot\start_bot_detached.ps1'}`.
The script reads the token itself (runner file or `files\.env`), calls
`start_bot_bg.ps1`, logs to `.runtime\start_bot_detached.log`. Back up
`bot_stderr.log` first — `start_bot_bg.ps1` deletes it. `restart_bot.bat` from a
user cmd window stays the normal manual path.

---

## 7. Known issues & fixes

### Positioning at the leader alert and at the crossing — no signal yet; one rule pre-registered (2026-10-20)

Scheduled test `crypto-bot-leader-positioning-test`. Positioning store: 60 days,
671 snapshots, 87 coins. Report: `docs/reports/2026-10-18-leader-positioning-readout.md`.

- **A. Leader continuation — too early.** 151 replayed alerts; replay = live 50/50.
  Base 0.261 on train, 0.205 on test, but only 9 test positives. Only `oi_1h` / `oi_4h`
  point the same way in both halves (AUC 0.62–0.70). Trading all test alerts on the
  k=3 / 8% trail: −3.44% [−5.21, −1.57].
- **B. Rockets earlier — REFUTED (R-11).** Positioning adds +0.022 AUC to R-9
  [−0.027, +0.064]; positioning alone scores 0.483. Do not re-test on less than ~6 months.
- **D. Crossing entry + positioning — too early.** Fourth look. The train rule gave
  test n = 27 < 60; it also fails the pnl bound and the reach.
- **C. Leader alert — readout ROLLBACK_SUGGESTED.** 53% vs 69%. The operator kept the
  alert (10-21); the message now quotes the live numbers (commit 8b50aca).

Next look: readout `LEADER-POSITIONING-OI`, due 2026-12-16, computed by
`_readout_leader_positioning_oi.py` (pipeline step before readouts). It tests ONE frozen
rule (`oi_1h >=` the 08-21..10-19 median) on days from 10-20 only. Spec:
`docs/specs/features/leader-positioning-oi-1021-spec.md`.

### metrics_daily went stale for 9 days; the report called it verified (2026-10-20, FIXED)

From 10-12 to 10-20 the orchestrator's `L0 metrics` step hit its 600 s timeout
on every run. The nine metric scripts ran in sequence (180 s each), and the row
was appended only at the end. They take 428 s when run alone, and the 07:00 run
shares the CPU with other jobs.

So `.runtime/metrics_daily.jsonl` stopped at 10-11. `run_step` printed nothing
on a timeout, and the morning report showed the 10-11 North Star as today's
with status `verified`.

What changed:
- `report_metrics_daily.py` runs everything inside one budget
  (`DEFAULT_BUDGET_SEC = 1500`). The North Star runs first, a script the budget
  does not reach is recorded `skipped_budget`, and the row is always written.
- `pipeline_run.L0_METRICS_TIMEOUT_SEC = 1800`, and timeouts are now printed.
- `bot_health_report`: any metrics_daily value older than 36 h
  (`METRICS_DAILY_MAX_AGE_HOURS`) gets status `stale` with its `as_of` date.
  The report then shows «ДАННЫЕ УСТАРЕЛИ», the P0 step `restore_metrics_daily`
  and the red flag `RF_metrics_daily_stale`.
- The truth harness checks `TH05_STALE_METRICS` and
  `TH05_METRICS_DAILY_NOT_REFRESHED`.

Spec: `docs/specs/features/metrics-daily-stale-1020-spec.md`.

### Decoupling signal — SHADOW (2026-05-07)
Validated (365d/1h, train/holdout, no lookahead): a watchlist coin that is
BOTH high-vol AND decoupled from the market (low trailing corr to the
equal-weight basket) is ~2× more likely to make a forward idiosyncratic
big move (top-gainer rocket). Vol-controlled lift +3.14pp (holdout
high-vol tercile); strict-rocket flag LIFT ×1.87 holdout, recall 22%,
precision ~10% (PRIORITY signal, ~8 misses/hit — NOT a detector). Cluster
lead-lag hypotheses were all NULL/marginal (one market); decoupling is the
only signal with a holdout-stable NS-relevant edge.
Helper `decoupling_signal.py` (pure `scores_from_rets` + self-test);
computed every `DECOUPLING_REFRESH_MIN` in `monitor.py` loop →
`config._decoupling_scores`; logged on entry events
(`decoupling_score/flag/corr` in `bot_events.jsonl`). SHADOW only — no
decision impact until shadow-replay confirms NS lift.
```python
DECOUPLING_SHADOW_ENABLED = True       # rollback = False
DECOUPLING_WINDOW_BARS = 168           # ~7d on 1h
DECOUPLING_VOL_PCTILE_MIN = 0.66
DECOUPLING_CORR_MAX = 0.60
DECOUPLING_GATE_ENABLED = False        # reserved; needs shadow-replay first
```
Specs/reports: `docs/specs/features/decoupling-signal-spec.md`,
`docs/reports/2026-05-07-decoupling-validation.md`,
`docs/reports/2026-05-07-cluster-lead-lag.md`,
`docs/reports/2026-05-07-universe-clusters.md`. Backtests:
`_backtest_decoupling.py`, `_backtest_decoupling_recall.py`,
`_backtest_cluster_lead_lag.py`, `_backtest_sector_lead_lag.py`,
`_backtest_btc_lead_15m.py`.

### impulse_speed regime curtailment + trail rollback (2026-06-05)

Live regression caught: the 8% trail widen (below) backtested +net on 35d but
live (5d, n=89) made impulse_speed LOSE (avg/trade +0.02->-0.62%, sum -54.9%) —
**rolled back** to 0.015 via a `rolled_back` decision (RM-4 runtime override
removed; `_config_runtime_overrides.py` reads decisions.jsonl, so logging a
decision auto-applies its `diff.to` as an in-memory override — rollback needs a
superseding `rolled_back` record, not just a config.py edit).

Root cause hunt (4 analyses, ALL negative — impulse_speed enters ~62% late):
1. lateness (zigzag, hindsight) separates outcomes but is tautological
   (late=near-peak=low remaining upside) — not real-time gateable.
2. extension static gate (close_vs_ema50>3 etc.) over-blocks: big movers are
   ALSO extended, recall 23-34% — fails the §7 over-block guard.
3. extension feature in the entry bandit: AUC 0.48, admitted set unchanged.
4. full multivariate logistic over 23 features: **train AUC 0.60, OOS 0.50** —
   no generalizing entry-time signal (the non-stationarity signature).

Conclusion: impulse_speed winners/losers are indistinguishable at entry with
available features; profitability is regime-driven (profitable Mar-early May,
negative mid-May-Jun). Lever is mode-level, not entry-level.

DEPLOYED — regime curtailment with auto-revive (`impulse_speed_curtail.py`):
pause the mode while its trailing-14d mean realized pnl < 0, auto-revive when
positive. Backtest (`_backtest_impulse_speed_curtail.py`, robust across
window/threshold grid): kept avg/trade +0.025 -> +0.162, ~-93 realized losses
removed, cost ~27% of late low-capture big-mover catches on bad-regime days.
`compute_and_write()` runs daily in `daily_learning.py` -> state file
`.runtime/impulse_speed_curtail.json`; live entry path calls `is_curtailed()`
in `_impulse_speed_entry_guard` (monitor.py), which FAILS OPEN.

**Hard-block was a live regression (2026-06-12), now FALLBACK-TO-TREND.**
Hard-block curtailment became the #1 block (266-813/day) and starved entries to
1-10/day during a broad altseason (distinct top-20/day 4-8 -> 25-32), because
`get_entry_mode` silently upgrades a fast trend (3-bar >=1.5%) to impulse_speed
and curtail hard-blocked those would-be-trend entries. Fix DEPLOYED same day:
when curtailed, **reclassify impulse_speed -> trend** (tighter stop, ENTER)
instead of blocking — `monitor.py` at the preview_mode block (~L3675) calls
`is_curtailed()` and downgrades to "trend"; the downstream curtail guard then
no-ops. Backtest (`_backtest_fallback_to_trend.py`, 25d, n=324): AS_TREND net
-0.221 beats AS_IMPULSE -0.290 with SAME big-mover coverage (13/13), vs BLOCK
0/13 — coverage-safe for altseason. Decisions: hard-block disabled via `rbcur1`
(rolled_back), re-enabled in fallback mode via `fbt001` (approved). Config:
```python
IMPULSE_SPEED_REGIME_CURTAIL_ENABLED = True       # rollback = False (off)
IMPULSE_SPEED_CURTAIL_FALLBACK_TO_TREND = True     # False = revert to hard-block
IMPULSE_SPEED_CURTAIL_WINDOW_DAYS = 14
IMPULSE_SPEED_CURTAIL_PNL_THRESHOLD = 0.0
IMPULSE_SPEED_CURTAIL_MIN_TRADES = 8
```
Known residual: auto-revive still self-freezes (fallback routes candidates to
trend, so no new impulse_speed realized pnl feeds the trailing window) — benign
now since fallback keeps coverage; a regime/shadow revive is the next refinement.

### EX1 capture fix — wider impulse_speed trail (2026-06-01)

Morning-report diagnosis: top-20 capture (NS sub-metric) stuck at ~0.15 —
binding leak is `atr_trail` prematurely stopping `impulse_speed` winners.
EX1 worst cases (DYDX -13.7%/+237%, LDO -9.1%/+445%, WIF -5.8%/+471%) all
exit at a LOSS on a deep retrace, then the coin runs without us.
`time_max_hold` exits captured ~20x more than `atr_trail` (EX1 +0.044 vs
-0.002).

Backtest (`_backtest_exit_policy_impulse.py`, 35d, 658 impulse_speed trades,
105 winners / 553 losers) replays the forward price path under exit policies.
Result is non-monotonic: tight 1.5% protects net but kills capture; mid
3-5% is worst of both (winners cut early, losers still bleed); **wide 8% is
the sweet spot** — winner mean pnl +0.94->+2.83%, capture +0.004->+0.015
(x4), and NET per-trade across ALL impulse_speed entries IMPROVES
-0.14->-0.06 (winner upside not paid by loser blow-ups). Pure HOLD_24h
helped winners but losers bled (net -0.25) — a stop is still needed, just
wide. PARTIAL_25/8 marginally best (net -0.05) but more complex; deferred.

Deploy: `TRAIL_MIN_BUFFER_PCT_IMPULSE_SPEED 0.015 -> 0.08` (live 2026-06-01,
bot restarted). Reversible = revert number. Honest caveat: capture still
~0.015 absolute (potential is +200..+470%) — meaningful step, not a silver
bullet. Watch EX1 atr_trail bucket over next 14d.

Also fixed: `pipeline_notify.py::build_next_step_block` crashed the morning
notify step (`TypeError: NoneType.__format__`) when an L3 counterfactual was
`available=True` but its numbers were `None`. Now requires all of
blocked/take/n before rich phrasing.

### RM-22 regime-conditional learned gating (2026-06-01)

Premise (operator): markets are non-stationary, so re-tuning *fixed* gate
thresholds is a losing game — the bot should LEARN to gate by market regime.
Three steps, each backtest-gated before deploy:

- **Step A (premise validation, `_backtest_regime_gate.py`)**: blocked-gate
  forward returns FLIP sign across regime cells (`bull/flat day × btc up/dn`).
  `entry_score` blocked pool: bull_day/btc_dn **+0.473** (Sharpe 2.88),
  bull_day/btc_up +0.093, flat_day/btc_dn +0.033, flat_day/btc_up -0.075
  (gate correct only here). Spread 0.548pp → premise confirmed.
- **Step B (regime interaction features, `contextual_bandit.py`)**: added
  `regime_bull_btc_up`, `regime_bull_btc_dn` (N_FEATURES 18→20) behind
  `BANDIT_REGIME_INTERACTION_ENABLED`. Backtest NEUTRAL (delta AUC +0.0005,
  critic_dataset) — the bull/btc_dn cell is too rare yet. **Held OFF**
  (`= False`). Bandit `load()` now dim-migrates saved A/b (identity/zero pad)
  so the 2.6M prior updates survive the dimension bump. Revisit once Step C
  populates the cell.
- **Step C (soft gate, DEPLOYED `= True` 2026-06-01)**: when
  `REGIME_SOFT_GATE_ENABLED`, an `entry_score`-below-floor candidate is no
  longer hard-blocked — it logs `entry_score_soft_pass` and falls through to
  the downstream regime-aware entry bandit (`should_enter` ~L5078), which
  decides enter/skip. Backtest (`_backtest_soft_gate.py`, temporal split):
  top-gainer coverage **17.7%→20.6% (+2.9pp)**, 32 admitted entries avg_r5
  **+0.274** / win 46.9% / Sharpe 0.90 — selective (RELAX-all additions were
  -0.031). POSITIVE → operator-approved deploy. Rollback = flag False.
  Wired in `monitor.py` entry_score floor block (~L4191) as an `elif` before
  the hard-block `else`.

Spec: `docs/specs/features/auto-improvement-loop-spec.md` (Sprint 0, RM-22).

### Event-loop freeze + self-inflicted death (2026-08-04)

Two stacked bugs made the bot die silently and stay dead (07-23: 8 days, zero
signals; 08-04: 7 hours, surfaced by /menu timing out).

1. **Label writes froze the loop.** `_fill_trade_outcome_labels` /
   `_fill_forward_labels` (monitor.py ~L3078) called `fill_labels` /
   `fill_forward_label`, and each of those rewrites the ENTIRE dataset file
   (`critic_dataset.jsonl` ~128 MB, `ml_dataset.jsonl` ~114 MB) inline in the
   asyncio loop → stalls of 130-300s, Telegram UI timeouts, WS reconnect storms.
   Fix: both helpers now enqueue onto a single-worker `_LABEL_EXECUTOR` thread
   (serialised, so no concurrent 128 MB passes; backlog warns at 25/100/250).
   Measured: 130-300s stalls every few minutes → **0 lag warnings**.
   Real fix still open: stop rewriting whole files (append-only patch +
   periodic compaction) — rewrite cost grows with the datasets.

2. **UI watchdog force-exited into nothing.** On a stall it called `os._exit(2)`
   "so the wrapper restarts" — but `.runtime/bot_bg_runner.cmd` runs python ONCE
   with no restart loop, so every force-exit was permanent.
   Fix: `UI_WATCHDOG_FORCE_EXIT_AFTER_WARNS 3 → 0` (disabled) plus a
   `force_exit_after_warns > 0` guard — without it 0 meant "exit on the FIRST
   warning" (`warn_count >= 0`), the opposite of disabling.
   Re-enable ONLY together with a real restart wrapper.

### Hypothesis pipeline produces unusable proposals (2026-08-05, OPEN)

All 16 pending L2 hypotheses reference `config_key`s that **do not exist** in
`config.py` (`MAX_LATENESS_PCT_IMPULSE_SPEED`, `ML_PROBA_MIN_IMPULSE_SPEED`,
`ENTRY_SCORE_MIN_5M`, ...), and two propose position sizing, which this bot does
not have (it is an alert system). None has a registered validator, so L3 cannot
check them either. The morning report still advertises the "best" one with a
ready-to-run `pipeline_approve.py` command — **do not run it**, it would target a
nonexistent parameter.

Consequence: no decision has come through the pipeline since 2026-06-17; every
change applied since then came from manual analysis. Minimum fix: validate
`config_key` against `config.py` at generation time and drop hypotheses whose key
is unknown. Status report: `docs/reports/2026-08-05-roadmap-status.md`.

### RSI-divergence hold — SHADOW, readout WEAK-HOLD-SHADOW (2026-10-21)

Operator-approved after X-9c = KEEP. The rule accepted on the Jul–Oct half (below): a 15m
exit on "WEAK: RSI дивергенция" with profit >= +0.5% at the signal (FIXED for all modes, not
the H5 per-mode thresholds) would be held on the trail. `RSI_DIV_HOLD_SHADOW_ENABLED = True`:
the bot still exits; `monitor._rsi_div_hold_decision` + event `weak_exit_shadow` record the
would-hold decision. No live hold flag exists yet. Readout `readouts.readout_weak_exit_shadow`:
window from 2026-10-06 (unseen by the replay), due 2026-11-30, >= 42 days and >= 40 changed
rocket-day trades; the hold arm is the offline P-1 trail plus the live leader-mode switch,
unresolved holds are pending, criterion = `exit_validator.decide` unchanged. Leader-mode
positions never reach the RSI exit (an event with `leader_mode=True` is an ANOMALY).
Rollback: False + restart. Spec: `docs/specs/features/weak-exit-shadow-1020-spec.md`.

### Weakness exits vs holding on the trail — REJECTED by the margin (2026-10-05)

Replay of 4 313 trades since 03-01: replacing WEAK RSI divergence / RSI overbought / quality
recheck / EMA fan / volume exhaustion exits (1 049 trades changed) by holding on the P-1 trail
gives **rocket-days +1.10 pp [+0.62, +1.63], 8/8 months positive, winners +1.04 pp**, but all
trades −0.035 pp [−0.105, +0.039] — the non-inferiority bound misses −0.10 by 0.005, so the
pre-registered rule rejects it and the margin is not relaxed. This is where rocket capture
leaks. **Time split (same day): of 12 narrowings chosen on Mar–Jun, only "WEAK RSI divergence
with profit at the signal ≥ +0.5% (fixed, all modes) → hold on the trail" passed; on Jul–Oct it
is ACCEPTED: rocket-days +0.89 pp [+0.27, +1.65], all trades +0.011 [−0.075, +0.105], 4/4
months.** The candidate grid was written after seeing the full-period table (disclosed), so the
clean confirmation is a shadow readout after X-9c (10-12). Live H5 thresholds are per mode
(0.3/0.5/0.7%) — the tested rule is a fixed 0.5%. Spec: `docs/specs/features/weak-exits-1005-spec.md`.

### Leader exit across the UTC day reset — REFUTED (2026-10-05)

The lost-lead exit ranks by return since the UTC open, so ranks are noise right after
midnight: in the replay since 03-01, **428 of 1 038** lost-lead exits fall in the 00 UTC
hour (live 09-29..10-04: 13 of 48). Removing it does not pay: skipping the check for the
first 2 h of the day — rocket-days −0.01 pp [−0.29, +0.30]; ranking on the trailing 24 h —
−0.33 pp [−0.82, +0.24]; all trades ~0; months agree 38%. A leader's move is mostly done by
midnight. Do not re-test day-reset variants of the lost-lead rule. X-9c readout (6/14 days on
10-05: rocket-day trades +0.74% [−0.85, +2.22], n=21) continues to 10-12. Spec:
`docs/specs/features/leader-midnight-1005-spec.md`.

### Model health in the morning report (2026-10-05)

`pipeline_drift.py` said "drift_detected" on 164 of 166 daily runs since May (KS on model
outputs), never had a single `ranker_proba` row (not logged in critic_dataset) and fed nothing.
`files/model_health.py` (flag `MODEL_HEALTH_ENABLED`, reporting only) adds: AUC of `ml_proba`
against its OWN target (peak ≥ 2% within 5 bars) — **0.72–0.75 over 15 days since 09-21, base
17–21%**; input-feature drift by PSI, alert only after 3 days (`btc_vs_ema50` = regime, never
alerts) — 40 of 139 backfilled days, four volatility-regime episodes; one run manifest per
nightly retrain. Reference only: IC of `ml_proba` with the forward 10-bar return is **−0.14
[−0.21, −0.08]** — the gate picks big upside excursions that end worse 2.5 h later (ties to the
weak capture factor). Spec: `docs/specs/features/model-health-1005-spec.md`.

### Demo trader — a SEPARATE application in `trader/` (2026-10-10)

The demo-account trader does not touch the signal bot: no import from `files/`, no
`config.py` flag, no runtime file of the bot. It READS three bot things only:
`files/bot_events.jsonl`, `history/*.csv`, `files/.env` (the two `BINANCE_DEMO_*` variables);
it writes only `trader/runtime/`. Its own brief is `trader/README.md`, its config
`trader/trader_config.py`, its tests `trader/run_tests.py` (a trader test asserts that no
bot file imports a trader module). Scheduled task `CryptoBot_MicrostructureRecorder` runs
`trader/run_recorder.cmd` every 5 min (read-only + public data). Do not add trading code to `files/`.
Since 2026-10-12 a second task, `CryptoBot_TraderShadow`, runs `trader/run_shadow.cmd`
every 5 min: E7 paper portfolios P0 (bot exits) vs P4 (X4, 8% trail), readout no earlier than 2026-11-09
(`docs/specs/features/e7-shadow-portfolio-1012-spec.md`).
Audit (2026-10-12): every Binance request of the trader and every trader decision (enter/skip with a
reason code, exits, day marks) go to `trader/runtime/audit/{exchange,decisions}/YYYY-MM-DD.jsonl`,
append-only, no secrets; summary `trader/audit_report.py` (`docs/specs/features/trader-audit-log-1012-spec.md`).
Phase D (2026-10-13): task `CryptoBot_TraderDaily` (02:20 local) runs `trader/run_daily.cmd`: ledger of demo fills
(myTrades), the day's revenue from equity marks with ledger reconciliation and the USDC drift split off, benchmarks,
report `trader/runtime/reports/YYYY-MM-DD.txt`; Claude task `crypto-trader-daily-report` posts it to the chat at 09:00.
No Telegram. Direction words need ≥ 10 counted days (`daily_revenue.MIN_DAYS_DIRECTION`)
(`docs/specs/features/demo-ledger-d-1013-spec.md`).
Since 2026-10-29 the report also carries block «6. Демо против бумаги» (`trader/parity.py`, measurement only).
- Per executor trade, d = demo ret − paper ret of the same key, split into entry vs the model's 8 bp, fees
  (BNB 0.075% vs 0.10%) and exit-and-rest.
- It also shows coverage (skips, paper exits without a position, exchange stops) and 7/30/all windows.
- First reading: +0.05 pp per trade over 8 days, too early (`docs/specs/features/demo-paper-parity-1029-spec.md`).
W1 (2026-10-28) — the trader must see ALL Binance spot coins (operator decision #2). Until W1, the executor and E7 read only
the bot's 93 coins. Task `CryptoBot_TraderWideScan` runs `trader/run_wide_scan.cmd` every 5 min, limit 10 min, IgnoreNew.
It calls `wide_scan.py step` (flag `WIDE_SCAN_ENABLED`; public GET only, no order). On every closed 15m bar it:
- evaluates every pair of the universe — 508 on 10-28;
- keeps their candles in `runtime/scan/bars/`;
- writes `wide_gen` (the bot's entry rules) and `leader` / `surge` flags, the flags for measurement only;
- runs the paper book `x4_wide` on the `wide_gen` entries;
- appends learning rows to `runtime/learning/wide_candidates.jsonl`.
The executor still trades the bot's 93 coins (operator, 10-28). Readout W-SHADOW no earlier than 2026-12-01: `x4_wide`
vs E7 `x4_floor8` (`docs/specs/features/wide-scan-shadow-w1-1028-spec.md`).
LA (2026-10-28) — exchange announcements as information that comes before the price move.
- Task `CryptoBot_TraderAnnouncements` runs `trader/run_announcements.cmd` every 5 min (limit 5 min) and calls
  `announcements.py step` (flag `ANNOUNCEMENTS_ENABLED`; public read-only GET, no order).
- Sources: Binance CMS catalogs 48 and 161, OKX new listings, Bybit new_crypto. Store: `runtime/announcements/`.
- History since 2023-01 is downloaded: Binance 1 606, OKX 988, Bybit 1 724.
- Labelling passed a control audit: 8% errors on 50 events.
- **The money test is REJECTED on train, 0 of 40; the 2026 test is not opened.**
  - Announcements FOLLOW the move: Binance new-perp coins are +14.7% in the 24 h before the announcement, 51% >= +2.5%.
  - After the announcement they fall.
  - Do not re-test longs on these announcement types (`docs/specs/features/listing-announcements-la-1028-spec.md` §8).
Phase C (2026-10-13): `trader/executor.py` executes the E7 book X4 at `TRADING_SCALE` 0.02 with a
catastrophe stop 3% below the policy stop on the exchange; task `CryptoBot_TraderExecutor` (every 5 min,
IgnoreNew) keeps it running. **C2 ON since 2026-10-20** (`TRADING_ENABLED = True`, `TRADING_MODE = "demo"`, operator): real orders on the DEMO account only; gate C after 7 days (>= 10-27). **PAUSE since 2026-10-29** (operator): no new X4 entries; open positions run to their exits; remove the `PAUSE` file to resume. Rollback: `TRADING_ENABLED = False`. Control by files in
`trader/runtime/executor/control/` (PAUSE, FLATTEN, RESET_NOTED). C0 smoke and C2 demo need the operator
(`docs/specs/features/demo-execution-c-1013-spec.md`).

XSMOM (2026-10-21, `docs/specs/features/xsmom-daily-money-1021-spec.md`): daily/weekly cross-sectional momentum on the
top-100 liquid spot pairs (2023-03..2025-12, 668 pairs incl. delisted, 24 candidates, money as E1) is **REJECTED on
train, test half never opened**: 0 of 24 admitted; best MF_L14_H7_K5 −0.014 %/day, vs the equal-weight market
+0.016 [−0.114, +0.151] with years −/+/−; pure momentum loses 0.10–0.29 %/day to the market and in 9 of 12 is worse
than ≥ 88% of matched-turnover random portfolios — a train-only observation that looks like 1–4 week reversal. The
first run was invalid (at 10k USDT the market benchmark stopped buying under the 10 USDT minimum); all arms now run
at 1M. Do not re-test this momentum grid; the 2026 window is still unopened for momentum AND reversal.

S1 reversal (2026-10-21, `docs/specs/features/s1-reversal-universe-1021-spec.md`, one spec replacing the S1 draft):
buying the K = 10 WORST pairs over 1/7/14/28 days (16 candidates, pair age >= 60 days) is **REJECTED on train, test
never opened**: 0 of 16; all lose 0.09–0.54 %/day to the equal-weight market and rank 0–1 of 50 matched-turnover random
portfolios. With XSMOM: BOTH extremes of past return lose to random picks (train-only observation, possibly a volatility
effect). Reference on 2023–2025: BTC buy-and-hold +0.158 %/day (+278%) while the equal-weight alt market lost 51% —
hindsight, not a strategy. The 2026 window for ranking by past return is still unopened.

FC funding carry (2026-10-22, `docs/specs/features/carry-funding-money-1021-spec.md`, spot long + perp short, L = 1):
on train (2023-03..2025-12) **passive BTC+ETH carry is the first trader hypothesis with a positive CI**: +0.0111 %/day
[+0.0095, +0.0131], ~4 %/yr, max DD −0.17 %, 0 liquidations, but declining (2025: +0.0067). Selecting pairs by trailing
funding is REJECTED (0 of 8: all worse than passive, liquidations in 7 of 8). **2026 test (opened once 2026-10-22, k = 1):
FC_PASSIVE ACCEPTED — +0.0024 %/day [+0.0015, +0.0038], ~0.9 %/yr, max DD −0.19 %, 0 liquidations; Feb–Apr negative.**
The first trader hypothesis profitable on its test, but small (train ~4 %/yr). Demo funding differs from live, so demo
money is a mechanics check only. **Next step (operator-approved 2026-10-22, `fc-passive-next-1022-spec.md`):**
`trader/fc_shadow.py` runs a paper FC_PASSIVE book on live public data as a step of `run_shadow.cmd` after E7
(`FC_SHADOW_ENABLED`, own log `runtime/fc_shadow/fc_shadow.log`), recomputing `carry.simulate` on [S, today) every
step; started 2026-10-22, S = 2026-10-23; readout FC-SHADOW (`READOUT_FC.json`) gives no verdict before 2027-01-21. **F0 = GO on 2026-10-22** (`_smoke_demo_futures.py --go`,
order/test only, never a position; run by Claude at the operator's request): futures wallet 5 000 USDT, BTCUSDT/ETHUSDT
ISOLATED + leverage 1, one-way; demo taker 0.04% (model 0.05%, kept); demo funding is the flat 0.0001, not live. `experiment_fc.judge` computes the interval at 1 − α.
**FC-DEMO (2026-10-30, operator 10-29 «все»; `fc-demo-execution-1030-spec.md`).** `trader/fc_demo.py` runs
FC_PASSIVE on the demo account: spot BTC+ETH long + USD-M perp short, L = 1 isolated, N = 2 500 per pair.
- Margin stop every cycle; a failed second leg unwinds the first; PAUSE / FLATTEN / kill switch.
- It runs as a step of `run_shadow.cmd`, log `runtime/fc_demo/fc_demo.log`.
- It skips the pre-registered KEEP of FC-SHADOW; the deviation is disclosed. Demo funding is not live funding, so
  demo money checks mechanics only.
- **DEMO MODE LIVE since 2026-10-30 09:26 UTC** (`FC_DEMO_MODE = "demo"`, N = 2 400 per pair):
  - BTC spot +0.030849 against perp −0.0308 at 77 694.8;
  - ETH spot +1.077521 against perp −1.077 at 2 224.98.
- Blockers fixed with tests before the switch:
  - X4 cannot trade FC pairs (`fc_reserved`), because it sells the free balance of a coin;
  - the X4 kill switch excludes the FC spot value;
  - daily revenue books the FC spot leg outside the policy revenue, and both FC legs as `R_fc_pct`;
  - a step lock prevents a double entry.
- **Incident at the first demo step (09:18 UTC).** The demo `fut_order` RESULT has no `avgPrice`, and the code
  read the filled perp as rejected. It unwound the spot legs and left naked shorts.
  - Claude closed both shorts by hand the same minute (reduceOnly).
  - Fix: price comes from `GET /fapi/v1/order`; a failed second leg now checks and closes the real perp position.
  - Realized −1.69 USDT.
- `exchange_client.fut_order` / `fut_get_order` / `fut_positions` exist since then.

Hypotheses of 2026-10-28 (`docs/specs/features/trader-hypotheses-1028-spec.md` §5), all checked on train 2023-03..2025-12
only; the 2026 window was not opened for any of them:
- **Verdict audit 2026-10-30** (`docs/reports/2026-10-30-verdict-audit.md`).
  - Price-only tests used 2023–2025 only, though Binance data goes back to 2017-08 (spot) and 2020-01 (futures and
    funding). That breaks the max-period rule.
  - A daily series of ~828 days cannot detect effects below ~0.11 %/day.
  - **Not shown, NOT refuted:** DIR 1d, DPOS 1d, TM, N8, RV. Their CI reaches +0.06..+0.29 %/day. "Do not re-test"
    does not apply to them; their frozen rules go to the unseen 2019–2022 backward holdout.
  - **Refuted for real:** DIR 4h, DPOS 4h, QB_T0 (vs FC), EX-RL1 A1.
  - **Rejected on other grounds:** XSMOM (random control), SN (liquidations), RB (theoretical ceiling).
- **BH19, backward holdout 2018–2022** (`backward-holdout-bh19-1030-spec.md`): frozen rules, one look, α = 0.01;
  418 spot pairs downloaded, stablecoins and fiat excluded as in the live universe.
  - **FC_PASSIVE CONFIRMED** on 2020-02..2022-12: +0.0247 %/day [+0.0196, +0.0303], 0 liquidations, max DD −0.36 %.
    It is now positive on three separate periods, but shrinking: 2022 +0.002, 2026 +0.0024.
  - **DIR_1d, TM, N8, RV NOT confirmed. Their verdict is final: REJECTED, do not re-test.** Each has R > 0 in the
    2019–2021 market, none beats its benchmark:
    - TM vs constant exposure: +0.071 [−0.053, +0.181];
    - N8 vs equal weight: −0.083;
    - RV vs 50/50: +0.019, and it is below the random p95;
    - DIR_1d top decile: −0.082 [−0.429, +0.270], AUC 0.538.
- **TM: BTC/ETH spot timing (trend, vol-managed, funding overheat) is REJECTED, 0 of 20.** None beats a circular-shift
  random timing of the same schedule, none beats buy-and-hold. Best vs constant exposure: T_sma_56 +0.027 [−0.039, +0.093]
  (`btc-eth-timing-tm-1028-spec.md`).
- **FS: FC carry switched on by funding is REJECTED, 0 of 9.** Every candidate is worse than always-on FC_PASSIVE; keep the
  carry on (`fc-switch-n4-1028-spec.md`).
- **N7: seasonality is REJECTED.** No hour or weekday survives a 74-window correction, and a 2026 test would have 3–7% power
  (`seasonality-n7-1028-spec.md`).
- **N8, low-volatility alts: REJECTED on its single 2026 look** (operator-approved, k = 1). R −0.065 [−0.261, +0.088];
  vs equal weight −0.037 (`lowvol-alts-n8-1028-spec.md`).
- **MD, multi-day spot holds (2–10 days): REJECTED on train in both rows** (`multiday-spot-md-1028-spec.md`).
  - Bot entries on the X4 book: 0 of 9, all R < 0. Best is the 8% trail held 10 days, −0.071 %/day against X4 −0.328.
  - Breakout / dip / volume events on the top-100 spot universe: 0 of 9, all worse than equal weight, PBO 0.97.
- **RV, BTC↔ETH rotation: REJECTED on its single 2026 look** (operator-approved 10-29, k = 1). It was admitted on train on the
  edge (1 of 6). On 2026 RV_mom_56 gives R +0.0066 [−0.318, +0.294] and +0.009 [−0.062, +0.083] vs 50/50; it is below the p95
  of random rotations (+0.063) (`btc-eth-rotation-rv-1028-spec.md` §7).
- Shorts and futures, allowed by the operator on 10-29 once spot was checked:
  - **QB, BTC/ETH quarterly basis: REJECTED.** On train it makes ~2.2 %/yr with CI > 0, but stops on PBO.
    The single 2026 look at QB_T0 gives +0.0004 %/day [−0.0039, +0.0046], below FC_PASSIVE (`quarterly-basis-qb-1029-spec.md`).
  - **N6 / SN, shorting new Binance perps after launch (L = 1, 1–30 days): REJECTED on train, 0 of 8, PBO 0.61.**
    - The median trade is +3..+17%, but 1–18% of new perps double within the hold and wipe out the margin.
    - The CI lower bound of R is < 0 for every candidate, and the short pays funding.
    - Do not re-test (`short-new-perps-n6-1029-spec.md`).
- **DIR, supervised direction forecast at 4h / 1d: REJECTED on train** (10-29, after the RL/analogues report found 15m/1h
  cannot pay fees).
  - CatBoost on 17 causal price / volume / BTC / market features, purged walk-forward, top-decile long on spot at 36 bp.
  - OOS AUC is 0.517 (4h) and 0.529 (1d), CI above 0.5, but accuracy barely beats "always down".
  - The selected longs rise only 49% / 48% of the time against the 61% / 55% needed. Net −0.195 % per 4h period, −0.257 %/day.
  - Not better than the universe; 2024–2025 negative (`direction-forecast-dir-1029-spec.md`).
- **DPOS, DIR plus derivatives positioning: REJECTED on train** (10-29).
  - Data: the public data.binance.vision USD-M metrics archive — 5-min OI, long/short, taker ratio — plus funding.
    361 perps, 2023–2025, stored in `trader/runtime/metrics_um/`.
  - The archive equals the live REST exactly (PIT-2b, 57 600 rows): archive time T is REST time T + 5 min, and REST
    ratios have 4 decimals.
  - Adding 10 positioning features lowers the 4h AUC: −0.0056 [−0.0094, −0.0015]. On 1d the change is
    −0.0035 [−0.015, +0.008].
  - The top decile still loses after costs. Do not re-test the 4h part; 1d is only "not shown" (verdict audit 10-30)
    (`dir-positioning-dpos-1029-spec.md`).
- **EX-RL1, entry execution (limit at the bid vs market), phase 0 on the aggTrades tape: not a profitability tool.**
  - The simulator is calibrated on 78 demo fills (median 9.5 bp). 1 964 X4 trades were replayed.
  - A limit with a market chase after 2 min loses 3.7 bp [−5.0, −2.7] per trade.
  - A 15-min limit without a chase gains +22.9 bp, but +21.8 of it comes from skipping 36% of a losing book. A random
    skip of the same share gives +18.4 (p95 +28.7), so the selection is no better than random. The price itself is
    worth only ~1–2 bp.
  - No phase 1, and no learned execution bandit (`execution-bandit-exrl1-1029-spec.md`).
- **RB, rebalancing BTC / ETH / USDT to fixed weights vs holding them: REJECTED on train, 0 of 6, PBO 0.57.**
  - Best: BTC/ETH weekly +0.0033 [−0.0241, +0.0392] %/day.
  - BTC/USDT is worse than holding by −0.02 %/day (`rebalancing-rb-1029-spec.md`).

### ML promotion at matched admission — SHADOW (2026-10-09)

The promotion gate kept the 10-01 ML model **7 nights running** (10-02..10-08), "clearly worse on
precision at the live floor". Each candidate was a logistic model with the SAME AUC (10-08: 0.655 vs
0.654) whose scores sit higher, so the fixed floor 0.15 admitted ~84% of rows instead of ~66%.
Offline on the trainer's test split (891 rows, 08-26..10-08): at an equal admitted share the
candidate is neither better nor worse (precision −0.5 pp [−2.3, +1.2], winner-day rows −1.6 pp
[−6.8, +3.3]). The freeze is a floor artifact and cost nothing measurable this week. A floor
chosen on the first half held the share on the second half within 2 pp. Recall and winner reach
there were −5 pp, intervals reaching −13 pp, hence SHADOW. Now every night the gate gives the
candidate its own `admission_floor` (the score at the incumbent's admitted share, bull/other
days), logs the matched decision, and logs how yesterday's candidate at its stored floor did on
rows newer than its holdout. Every candidate is now saved to `ml_signal_model.candidate.json`.
`ML_PROMOTION_MATCHED_ADMISSION_ENABLED = False`. True switches BOTH halves: the gate decides at
matched admission, and the bot's ML gate reads the model's `admission_floor` (ahead of
`ML_FLOOR_CALIBRATION_ENABLED`). Readout ML-ADMISSION-MATCH is due 2026-10-30. Promoting the
logistic at 0.15 instead (+5.8 pp winner rows, −3.8 pp precision) would loosen the gate, which is
an operator choice and not part of this change. Maximum-period walk-forward (5 939 rows, 03-24..10-08):
with a sensible anchor matched = fixed on the goal (27/41 winner-days both); anchored on a one-week model
the chain froze an 11% share and reached 10/42 vs 27/42 -- matching freezes the incumbent's operating
point, so the readout flags ANOMALY if that share leaves [0.50, 0.85]. Spec:
`docs/specs/features/ml-admission-match-1009-spec.md`.

### Volatility-forecast trail (σ̂ instead of ATR) — REFUTED (2026-10-05)

Baseline re-measured first: NS v2 0.122 (30d, n=122) / 0.100 (240d, n=558); coverage
0.73–0.81, **capture 0.22–0.25 is the weak factor**. Hypothesis: replace ATR in the trail
buffer by a HAR forecast of next-hour volatility (fit before 2026-03-01, width matched).
On 15m bars σ̂ predicts next-hour log volatility no better than ATR (OOS R² 0.330 vs
0.328; 4h 0.562 vs 0.546). Replay of 4 313 trades since 03-01 (exit_validator engine):
rocket-days −0.04 pp [−0.36, +0.25], all −0.044 pp [−0.082, −0.006], months agree 38% —
not accepted. Do not re-test σ̂/HAR/GARCH buffers on 15m data. Spec:
`docs/specs/features/sigma-trail-1005-spec.md`.

### correlation_guard cluster cap 2 → 3 — LIVE (2026-10-16)

Operator-approved after the compromise test below. `CORR_MAX_PER_CLUSTER = 3`
applies to entry AND prune; pruning was not in the backtest. New events (no
decision reads them):
- `corr_cap_relaxed`: a candidate the old cap would have blocked;
- `corr_prune`: an enforced prune.

Readout `CORR-CAP3` due 2026-11-13: keep if ≥ 1 winner-day is gained, trades are
non-inferior, and max drawdown and worst day are each no more than 10% deeper
with the relaxed trades. Rollback: set the cap back to 2. Spec:
`docs/specs/features/corr-cap3-live-1016-spec.md`.

### correlation_guard re-judged by the goal — a trade-off, operator's call (2026-10-16)

Trigger: INJ 10-15. The guard blocked three early candidates (16% / 60% / 78%
into a +32.5% move), and the bot entered at 95%. Over the enforcement period
(07-07 .. 10-16, 293 winner-days), removing the guard gives:
- **+2.77 to +2.95 pp** of early winner-days (CI lower bound > +1.1), with the
  downstream p measured at 0.79–0.85;
- an unchanged mean trade (−0.003 pp [−0.089, +0.078]);
- clustered losing days: worst day −94.8% → −113.7% and max drawdown +30%
  (exploratory; the pre-registered tail test conditioned on the wrong days).

Validator defect: `validate_gate_off` reported p = 1.0 because blocks from the
portfolio limit (which runs after the guard) have no stage and fall out of the
population. Spec: `docs/specs/features/corr-guard-goal-1016-spec.md`.

Compromises (pre-registered, `corr-guard-compromise-1016-spec.md`):
- **cluster cap 3:** +1.08 pp [+0.27, +2.43], trades non-inferior, drawdown
  +9.9%. Supported, but marginal against the 10% bound.
- **pre-crossing exemption:** refuted. The drawdown is +25%: the candidates
  the guard blocks on shock days are exactly the early ones.

### GOAL-label ML model in SHADOW (2026-10-12)

The next step after the two tests below: `ml_goal_shadow.py` trains the GOAL-label
model every night after the live retrain. `monitor.py` scores every candidate
that reaches the ML hard-block decision and logs `ml_goal_shadow` (shadow score,
live score, floor, live pass). No decision reads it.

Readout `ML-GOAL-SHADOW` runs on unseen days, due 2026-11-23, with these
criteria fixed in advance:
- the shadow admits the same weekly share as the live gate;
- KEEP if winner-days entered before the crossing gain ≥ +0.5 pp with a lower
  bound > 0, and the per-trade lower bound is ≥ −0.10 pp;
- ROLLBACK if the gain is ≤ 0.

Rollback: `ML_GOAL_SHADOW_ENABLED=False`. Spec:
`docs/specs/features/ml-goal-shadow-1012-spec.md`.

### ML gate label "size of the day's move" — REFUTED; goal label promising, not shown safe (2026-10-12)

Trigger: ICP 10-11. The gate scored 0.023 a candidate that then climbed +13% slowly.
`peak5_2.0` asks "+2% in 75 min", a volatility question. Walk-forward by month
over 57 588 bot candidates (03-24..10-12, 454 winner-days), with the same features
and CatBoost, at matched 60% admission:
- 24h-peak labels (≥5%, ≥10%, ≥2× the coin's usual daily range) change early
  winner-days by −0.2..−0.4 pp. They pick bigger later moves, not earlier entries.
- The goal label itself (early row of a top-20 winner-day) reaches **+7.05 pp
  [+3.52, +10.57]**, gaining in all 6 months. Its per-trade lower bound is −0.22,
  under −0.10, so by the pre-registered rule it is refuted.
- Ceiling: the downstream gates are ignored.

Do not re-test the 24h-move labels. Spec:
`docs/specs/features/goal-aligned-ml-label-1012-spec.md`.

Follow-ups the same day (`ml-label-followups-1012-spec.md`):
- **Exploratory, same data.** Counting trades the way the bot takes them (one
  position per coin plus the cooldown), the GOAL label's trade delta is +0.045 pp
  [−0.076, +0.154]. That is non-inferior, so the label is worth a forward SHADOW.
  This does not reverse the verdict above.
- **Pre-registered, refuted.** Training the live label on the bot's candidates
  instead of `ml_dataset.jsonl` gives −0.88 pp [−2.64, +0.66]. The training
  population is not the lever.

### "BTC leads SOL/ETH with a bigger move" — REFUTED (2026-10-03)

462 days, 93 coins: SOL/ETH beta to BTC 1.24–1.37 (not "several times"); they move
in the SAME 15m bar (corr 0.80/0.86, next-bar lead ~0). After a BTC +0.5% hour
(414 events) SOL had already made +1.08% inside it; 4h after: P(up) 55% vs base
50%; trend-end 55–57% vs 49–50%; buying at the event loses after fees; BTC event
days do not change which coins become top-20 winners. Do not re-test BTC-led
entries/exits on price data. Spec: `docs/specs/features/btc-leads-alts-1003-spec.md`.

### mode_range_quality gate OFF — operator decision (2026-10-01)

`MODE_RANGE_QUALITY_GUARD_ENABLED = False` (operator chose +3.43 pp early
winner-days over ~+6.6 msg/day at 1.1% precision). The guard still runs in shadow
and writes `mode_range_shadow`; readout MRQ-OFF (due 2026-10-16) compares the
added entries with the rest. Rollback: True + restart. Spec:
`docs/specs/features/mrq-off-1001-spec.md`.

### Gate locks re-judged by the goal; alert budget refuted (2026-10-01)

`goal_validator.validate_gate_off` (gate removed entirely, L3 criterion): chop,
open_cluster_cap and ml_proba_zone (current model era, 09-07..) CONFIRMED;
**mode_range_quality UNSUPPORTED by the goal** (+3.43 pp early winner-days,
trades non-inferior) but removing it adds ~6.6 msg/day at 1.1% precision of new
coin-days -> moved to `contested`, left ON, operator decision (report P1). mtf and
cooldown never appear as blocked events -> not replayable. Alert-budget ranking
REFUTED: at 3/5/10 msg/day precision rises to 0.47/0.36/0.25 (vs 0.11) but early
winner catches drop to 0-1 of 18 -- the precise scores already contain the answer
(late). Spec: `docs/specs/features/valid-proposals-1001-spec.md`.

### Morning report: invalid statements removed (2026-10-01)

One status mismatch ("measured" written, "verified" checked) printed four false
statements daily: "МЕТРИКА ПРЕДВАРИТЕЛЬНАЯ", a rolling-24h ground truth, a critical
red flag and a P0 step to build labels that exist since 08-17. Also fixed: the
coverage funnel, signal precision and time-to-signal ran on the leaky
`label_top20` while printed as watchlist∩global-top20 -> v2 on immutable labels
(funnel 80% = North Star coverage; precision 13.6%; time-to-signal +3.5 h after
the +2.5% crossing, measured from the crossing instead of ~6-hourly snapshots),
legacy values kept as `legacy_*`; a decision with NO measured metric was reported
as "просело: realert_rate"; the training<->live gap compared the switched-off
bandit with live buys. The trend verdict uses immutable-label rows only. Spec:
`docs/specs/features/report-validity-1001-spec.md`.

### Late entry: anatomy, and the crossing-entry fix REFUTED (2026-09-30)

531 winner-days (03-01..09-28): early 11.3%, **late 55.6%**, never 26.2%, held 7%.
Late entries come 3.1 h after the +2.5% crossing with 46% of the move done (early:
30%); in 70% of late days NO entry rule fired before the crossing (detection, not
gates). Entering AT the crossing when the coin is rank <= 3 of the day doubles
precision (0.202 vs the bot's 0.102) and enters ~10 points of the move earlier,
but loses -0.37%/trade (every month but September); a leader-mode exit and a
regime filter chosen on Mar-Jun / judged on Jul-Sep were refuted too (three looks,
counted). Do not re-test crossing entries on price features; next lever is
positioning at the crossing (added to the 2026-10-18 test). Spec:
`docs/specs/features/late-entry-0930-spec.md`.

### Improvement agent sees cases and is judged by the goal (2026-09-29)

Before: L2 got aggregate red flags only and could propose nothing but "move a
threshold" (all 3 proposals of 09-29 rejected, one on a 4-month-old premise); L3
judged by a 4h-peak proxy; readouts of live changes lived in free-text one-off
tasks. Now (spec `docs/specs/features/agent-tasks-0929-spec.md`):
- `files/incident_analyst.py` (daily, before notify): for each of yesterday's
  immutable top-20 on the watchlist, the FIRST place the bot lost it (bot down /
  held / caught early / caught late / blocked + first gate / heartbeat stages) and
  the exit; LLM only groups cases with ids, no remedies. Morning block "Лидеры".
  First 4 days (09-25..28): 25 winner-days, caught before +2.5% 6 (24%), **late
  17 (68%)**, blocked 1 — the dominant loss is late entry, not a gate.
- `files/goal_validator.py` decides L3 for replayable keys: winner-days entered
  before the +2.5% crossing + per-trade non-inferiority (-0.10 pp), maximum period;
  the peak replay is kept beside it. Fixed in both validators: rows the gate never
  judged (blocked later by impulse_guard) were counted as admitted — 9 of 46 rows
  of the chop 22->20 hypothesis.
- `files/readouts.py` + `files/readout_registry.json`: E-4, X-9c, leader alert read
  out daily against pre-registered criteria (COLLECTING / TOO_EARLY / KEEP /
  ROLLBACK_SUGGESTED / INCONCLUSIVE; ANOMALY at once). P-2 and positioning stay
  Claude tasks.
- L2 receives `incidents_14d`, must cite case_ids, writes unreachable losses to
  `decisions/l2_gaps.jsonl`.
- `files/exit_validator.py` (task 3): L3 for `LEADER_EXIT_*` and
  `TRAIL_MIN_BUFFER_PCT_<MODE>`, replayed on every real 15m trade since 03-01
  (4 286); paired rocket-day difference > 0 + all-trade non-inferiority + months.
  Reproduces X-9c vs X-7b: +0.45 pp [+0.22, +0.73] on rocket-days (spec +0.42).
  L2 is now offered these keys.
- trend-scout (RL worker, every 4h) was the ONLY loop that edits `config.py` by
  itself, on a proxy (ret5 >= 0, win >= 40% over 4h); CLONE_SIGNAL_GUARD_MAX_SIMILAR
  drifted 4 -> 23 that way. `TREND_SCOUT_AUTO_APPLY_REQUIRES_L3`: it now applies
  only on an L3 `accept` (goal/exit validator, maximum period); unreplayable keys
  are held and logged. Takes effect after an RL-worker restart.
- `files/ml_promotion_gate.py` (`ML_PROMOTION_GATE_ENABLED`): the nightly ML model
  replaces the live one unless clearly worse (95% upper bound < 0) on AUC, recall
  or precision at the live floor, on the trainer's test rows. First run: AUC
  0.661 -> 0.673, but admitted 62% -> 82% (score scale moved against the fixed 0.15
  floor), precision 0.308 -> 0.271 -> incumbent kept. Precision was added after that
  run (stated in the spec). Open defect: the floor is not calibrated per model.
- Per-model ML floor, SHADOW (`ML_FLOOR_CALIBRATION_ENABLED = False`,
  `ML_FLOOR_TARGET_RECALL = 0.80`): walk-forward (4 folds, 5 800 rows) did NOT
  support it -- fixed 0.15 kept recall 0.80-0.86, calibrated floors missed their
  target by 10-14 pp, goal +1 winner-day of 27. The model now carries a
  `calibration` block, the bot logs `ml_floor_calibration` at load; readout
  ML-FLOOR-SHADOW (due 2026-10-21) decides.
- ML model hot reload (`ML_MODEL_HOT_RELOAD_ENABLED`, spec §8): until 2026-09-29
  the bot loaded the model once per process, so the nightly retrain reached the
  gate only at a restart. Now a changed, settled, complete `ml_signal_model.json`
  replaces the model in use (checked every 60 s); event `ml_model_reload`. Readout
  ML-HOT-RELOAD flags a blackout day (>= 85% of >= 50 candidates blocked by the ML
  gate -- 08-20 was 88.6%, no other day since June above 79.1%).
Flags: `L3_GOAL_VALIDATOR_ENABLED`, `INCIDENT_ANALYST_ENABLED`,
`INCIDENT_ANALYST_LLM_ENABLED`, `READOUTS_ENABLED`, `READOUTS_LLM_SUMMARY_ENABLED`.

### Morning report ignores downtime (2026-08-05, OPEN)

The North Star metric is a 14-day window with no notion of the bot being down, so
the 8-day outage (07-23..07-31) counted every top-20 of those days as a miss and
the report announced "СТАЛО ХУЖЕ ... ~2 из 100". Restricted to days the bot was
actually alive: coverage 67%, silent-miss 8% — the best figures on record. Mark
days without data as "no data" instead of misses before trusting trend claims.

### Critical bugs (fixed)

| Date | Problem | Fix |
|------|---------|-----|
| 2026-04-13 | `is_bull_day_now` used before definition in `monitor.py` (~L3125 vs L3296) → silent crash each cycle, 0 signals overnight | Defined before `_impulse_speed_entry_guard` call |
| 2026-04-06 | 0 entries despite 12 top gainers (+5–59%) — all blocked | See "Filter overtightening" below |

### Filter overtightening (2026-04-06) — ИСТОРИЧЕСКИЙ РАЗБОР

Значения ниже — то, что поставили ТОГДА. Часть с тех пор дотюнена (`ML_GENERAL_HARD_BLOCK_MIN` сейчас 0.28, `..._BULL_DAY_MIN` 0.22). Текущие значения всегда в `config.py`, здесь — причина правки.

All filters were blocking **100% of top gainers**. Corrected values:
```python
ML_GENERAL_HARD_BLOCK_MAX = 1.01                        # was 0.65 (!) — blocked high confidence
ML_GENERAL_HARD_BLOCK_MIN = 0.35                        # was 0.55
ML_GENERAL_HARD_BLOCK_BULL_DAY_MIN = 0.28
ML_CANDIDATE_RANKER_HARD_VETO_15M_FINAL_MAX = -2.50     # was -0.75
TREND_15M_QUALITY_RSI_MAX = 72.0                        # was 68.0
TREND_15M_QUALITY_RSI_MAX_BULL_DAY = 76.0
TREND_15M_QUALITY_PRICE_EDGE_MAX_PCT = 3.20             # was 2.40
TREND_15M_QUALITY_PRICE_EDGE_MAX_BULL_DAY_PCT = 4.00
```


**Rule:** If a filter blocks >80% of eventual top gainers — it's broken. Verify via `bot_events.jsonl`.

### Bull-day relaxation (2026-04-13)
```python
TREND_15M_QUALITY_DAILY_RANGE_MAX = 10.0
TREND_15M_QUALITY_DAILY_RANGE_MAX_BULL_DAY = 14.0       # TAO 12% was being blocked
ML_CANDIDATE_RANKER_HARD_VETO_1H_TOP_GAINER_MAX = 0.25  # veto only if final bad AND TG prob low
```

### H5 trailing-only after break-even (2026-05-02) · v2.8.0
Когда `pos.pnl >= 0.5 %`, soft EMA-pattern exits подавляются и
управление передаётся ATR-trail’у. Цель: атаковать главный leak
из EX1 baseline — exit_class `ema20_weakness` (median EX1 −0.010).
Backtest 30 d: 4/982 H5-eligible exits, на одном APEUSDT 04-30
оставлено +471 % potential на столе. Default НА МОМЕНТ ВНЕДРЕНИЯ: SHADOW=True, ENABLED=False.
**Сейчас в config: ENABLED=True, SHADOW=False** — см. «H5 ACTIVATED» ниже.
Acceptance перед flip → True: 7 d shadow с ≥3 events.
Helper: `monitor.py::_h5_should_suppress` (~ L1745).
Wired в exit-pipeline после WEAK / TREND_HOLD overrides (~ L5325).
Configs:
```python
H5_TRAILING_ONLY_AFTER_BREAK_EVEN_ENABLED = True    # активировано 2026-05-05
H5_TRAILING_ONLY_SHADOW = False                     # shadow снят там же
H5_BREAK_EVEN_PCT = 0.5
```
Spec: `docs/specs/features/h5-trailing-only-break-even-spec.md`.

### H5 ACTIVATED + hybrid skill→scout (2026-05-05) · v2.12.0
H5 переведён из shadow-only в production: подавление soft EMA-pattern
exits при `pnl ≥ 0.5 %`. Триггер: ICPUSDT 05-05 — бот вышел на +2.5 %
по WEAK RSI-divergence, цена пошла на +6.8 % (left +4.3 %).
```python
H5_TRAILING_ONLY_AFTER_BREAK_EVEN_ENABLED = True   # was False
H5_TRAILING_ONLY_SHADOW = False                    # was True
```
Гибрид skill↔scout: weekly evaluator пишет
`evaluation_output/skill_missed_trends.json` (335 trends за 7d на тесте),
trend_scout читает с `--source skill`, прогоняет через свой
backtest+auto-apply pipeline. На smoke-тесте: 5 proposals, 2 approve,
2 reject. CLI: `pyembed/python.exe files/trend_scout.py --source skill`.
Spec: `docs/specs/features/signal-evaluator-integration-spec.md` §8a.

### Skill integration Phase B+D (2026-05-04) · v2.10.0
ZigZag-labeler выделен в `files/zigzag_labeler.py` — bot-internal модуль
(не дублирует skill, тот же алгоритм). Используется EX1 backtest’ом
с `--use-zigzag` флагом для замены proxy-knowledge intraday-high
на честный `matched_uptrend.gain_pct`.
Wrapper `_run_signal_evaluator.py` поддерживает `--per-mode` —
запускает skill отдельно на каждом signal_mode (alignment, trend,
impulse_speed, …) с separate output dir. Видно сразу: `impulse_speed`
alpha +6.93 %, `alignment` −0.33 %, `impulse` −4.34 % за 3 d.
Phase C (verdict-reward для бандита) — drafted, не реализовано.

Canonical-metrics spec: `docs/specs/features/metrics-canonical-spec.md`
— один canonical metric per business question. Защита от metric-soup
после добавления skill (всего теперь 13 framework + 8 skill метрик).

### H3 trend-surge precedence (2026-05-02) · v2.7.0
Detector `check_trend_surge_conditions` существовал, но в основном
pipeline шёл ПОСЛЕ `entry_ok` — фактически dead-code. Добавлен
config-флаг `TREND_SURGE_PRECEDENCE_ENABLED` (default **False**).
Когда True: surge ловится ПЕРЕД entry_ok, новый `sig_mode = trend_surge`,
trail_k = `ATR_TRAIL_K_TREND_SURGE` (2.5).
TG-метка: 🌱 Старт тренда (slope-ускорение).
Acceptance перед flip → True: ≥5 reclassifications за 7 d shadow.
Spec: `docs/specs/features/trend-surge-precedence-spec.md`.

### EX1 realized-to-potential capture (2026-05-02) · v2.7.0
Первая exit-side метрика. Baseline median **+0.001** на top-20 winners
— практически ноль capture от potential. Худший exit-class:
`ema20_weakness` (median EX1 −0.010, worst cases −10 % pnl при +170+%
potential). Скрипт: `_backtest_ex1_realized_potential.py`. Интегрировано
в `report_metrics_daily.py`.
Spec: `docs/specs/features/ex1-realized-potential-spec.md`.

### Trend/1h chop-filter (2026-05-01) · v2.6.0
Reason of birth: live STRKUSDT signal at ADX 20.2 / slope +0.70 % /
vol_x 1.61 — pure chop-range, not a trend.
Guard: блокирует `trend/1h` candidates если
`(ADX<25) OR (slope_pct<1.2) OR (vol_x<1.3)`.
Backtest 30 d (`_validate_trend_chop_filter.py`):
- baseline trend/1h: precision 1.2 %, avg_pnl −0.17 %, FR_v1 22.6 %.
- after filter: precision **16.7 %** (+15.5 pp), recall 100 % (1/1
  top-20 winner сохранён), avg_pnl **+1.58 %**, FR 16.7 %.
- 80 of 86 entries cut → ~14× меньше шума на TG.
Helper: `monitor.py::_trend_1h_chop_guard_reason` (~ L1700).
Wired right after `trend_quality` guard в monitor pipeline.
Reason-code: `trend_1h_chop`.
Configs:
```python
TREND_1H_CHOP_FILTER_ENABLED = True
TREND_1H_CHOP_ADX_MIN   = 25.0
TREND_1H_CHOP_SLOPE_MIN = 0.7    # relaxed from 1.2 (slow-build trends were cut)
TREND_1H_CHOP_VOL_MIN   = 1.3
TREND_1H_CHOP_USE_BULL_DAY_RELAX = True   # ON since 2026-08-06 (see below)
TREND_1H_CHOP_ADX_MIN_BULL_DAY   = 22.0   # bull-day relax: validated, max period —
TREND_1H_CHOP_SLOPE_MIN_BULL_DAY = 1.0    # bull ADX22-25 band +0.097% & 3 recovered
TREND_1H_CHOP_VOL_MIN_BULL_DAY   = 1.2    # top-20; the SAME band is -0.822% off bull
```
Rollback: `TREND_1H_CHOP_FILTER_ENABLED=False`.
Spec: `docs/specs/features/trend-1h-chop-filter-spec.md`.

### Entry-event logger fix (2026-04-30)
Added ranker outputs to entry-event payload in `bot_events.jsonl`:
`ranker_top_gainer_prob`, `ranker_ev`, `ranker_quality_proba`,
`ranker_final_score`, `signal_mode`, `candidate_score`, `score_floor`.
Unblocks post-hoc validation of 1A (top_gainer_prob mega-trigger)
and 4A (precision-prune). All fields nullable, no schema break.
Patch sites: `botlog.py::log_entry`, `monitor.py` ~ L4605.
Spec: `docs/specs/features/entry-event-logger-fix-spec.md`.

### Trail-stop min buffer floor (2026-04-26)
Whipsaw exits on `impulse_speed`/`strong_trend`/`impulse` when bandit picks
`tight`/`very_tight` arm on high-volatility names (ALGOUSDT 04-22, 04-26).
Fix: `buffer = max(trail_k*ATR, min_pct(mode)*price)` at both init_trail
and trail-update sites in `monitor.py` (helpers `_trail_min_buffer_pct`,
`_compute_trail_buffer` ~ L1705). Defaults:
```python
TRAIL_MIN_BUFFER_PCT_ENABLED = True
TRAIL_MIN_BUFFER_PCT_IMPULSE_SPEED = 0.015   # 8% was ROLLED BACK 2026-06-05 (live regression)
TRAIL_MIN_BUFFER_PCT_STRONG_TREND  = 0.015
TRAIL_MIN_BUFFER_PCT_IMPULSE       = 0.012
# trend / alignment / retest / breakout / default = 0.0
```
Rollback: `TRAIL_MIN_BUFFER_PCT_ENABLED=False`.
Spec: `docs/specs/features/trail-min-buffer-spec.md`.

---

## 8. Portfolio rotation (2026-04-17)

Problem: portfolio stuck at 10/10 with weak legs (EV=-0.5), blocking 22 trending candidates. Naive MAX_OPEN+1 gave only avg_r5=+0.04%. ML-gated rotation gave **avg_r5=+0.241%, Sharpe=+2.75** (Pareto winner: `ml_proba >= 0.62`, no score filter, n=211).

Config params (all in `config.py`):
```python
ROTATION_ENABLED: bool = True
ROTATION_ML_PROBA_MIN: float = 0.62
ROTATION_WEAK_EV_MAX: float = -0.40
ROTATION_WEAK_BARS_MIN: int = 3
ROTATION_PROFIT_PROTECT_PCT: float = 0.5
ROTATION_MAX_PER_POLL: int = 1
COOLDOWN_BARS: int = 19   # was 24 (Scout APPROVE)
```

Eviction mechanism: `evict_position` sets `trail_stop = last_price * 1.001` so the next ATR poll closes naturally — no forced market sell.

---

## 9. Analysis patterns

- **Blocked-event diagnosis:** load `bot_events.jsonl` in Python, group by `decision.reason_code`, sort by count. Pattern in `backtest_portfolio_rotation_grid.py`.
- **Gate tuning:** reuse the Pareto grid script — swap the filter lambda, keep the Sharpe-proxy `avg_r5 / sd_r5 * sqrt(n)`.
- **Rotation decisions:** `should_rotate` returns `RotationDecision(allowed, weak_sym, reason)` — log `reason` for every non-entry.

---

## 10. Scout rule placement

New gates → `trend_scout_rules.py` as `BlockRule(name, predicate, reason_code)`. Threshold → `config.py` (`UPPER_SNAKE_CASE`). Never hardcode numbers in `monitor.py`.

Live rules include: `ranker_hard_veto`, `correlation_guard`, `ml_proba_zone`, `trend_quality`. **Re-tighten only after a Pareto sweep shows no over-blocking.** Historical lesson: three of these blocked 100% of top gainers simultaneously.

---

## 11. Config change workflow

1. Edit `config.py`. Default must match current live behavior.
2. Gate risky changes behind a boolean (`ROTATION_ENABLED`, `CORR_GUARD_ENABLED`, `BANDIT_ENABLED`) — rollback = flip flag.
3. Restart bot (section 6).
4. Watch first 30 min of events before walking away.

### Important flags
```python
BANDIT_ENABLED = False   # OFF since 2026-09-07: _backtest_gate_overblocking.py
                         # showed the entry bandit's REJECTS outperforming its
                         # own entries, and its LinUCB state is calibrated to the
                         # pre-peak-label ml_proba scale. Rollback = True, but not
                         # before the state is rebuilt on the new scale.
ML_GENERAL_HARD_BLOCK_MAX = 1.01              # 1.01 = no upper cap
TOP_GAINER_CRITIC_ENABLED = True
RL_TRAIN_TELEGRAM_REPORTS_ENABLED = False           # live: off
TOP_GAINER_CRITIC_TELEGRAM_REPORTS_ENABLED = False  # live: off
TREND_15M_QUALITY_GUARD_ENABLED = True
ML_CANDIDATE_RANKER_HARD_VETO_ENABLED = True
ROTATION_ENABLED = True
REGIME_SOFT_GATE_ENABLED = True               # RM-22 Step C (deployed 2026-06-01)
BANDIT_REGIME_INTERACTION_ENABLED = False     # RM-22 Step B (neutral, held OFF)
DECOUPLING_SHADOW_ENABLED = True              # decoupling shadow logging (2026-05-07)
DECOUPLING_GATE_ENABLED = False               # reserved; needs shadow-replay first
POLL_HEARTBEAT_ENABLED = True                 # per-coin per-bar poll log, logging only (2026-09-25)
TREND_15M_QUALITY_ZERO_FORECAST_AS_NO_DATA = True  # forecast 0.000 = no data, not a block (2026-09-25, p0-validation-0925)
LEADER_ALERT_ENABLED = True                   # info alert: coin held top-3 of the day 2h (backtest 69% end top-3, live 53% -> readout ROLLBACK_SUGGESTED, kept by operator 10-21 with live numbers in the text), not a buy (2026-09-27)
LEADER_EXIT_ENABLED = True                    # X-7b/X-9c: held coin top-5 of the day at +3% -> wide 8% trail, exit on losing top-10 (2026-09-28)
COOLDOWN_PERSIST_ENABLED = True               # post-exit cooldowns survive a restart (.runtime/cooldowns.json) (2026-09-28)
L3_GOAL_VALIDATOR_ENABLED = True              # L3 judged by the goal, not the 4h peak (goal_validator.py, 2026-09-29)
INCIDENT_ANALYST_ENABLED = True               # daily: where each winner was lost (incident_analyst.py, 2026-09-29)
READOUTS_ENABLED = True                       # pre-registered readouts of live changes (readouts.py, 2026-09-29)
L3_EXIT_VALIDATOR_ENABLED = True              # L3 can test exit / leader-mode keys (exit_validator.py, 2026-09-29)
TREND_SCOUT_AUTO_APPLY_REQUIRES_L3 = True     # scout edits config only on an L3 accept (2026-09-29)
TREND_SCOUT_TG_ONLY_ACTIONABLE = True         # scout writes to Telegram only for L3-backed changes (2026-10-02, scout-tg-actionable-1002-spec.md)
ML_GOAL_SHADOW_ENABLED = True                 # GOAL-label ML model scored in SHADOW at the ML gate, no decision reads it (2026-10-12, ml-goal-shadow-1012-spec.md)
ML_PROMOTION_GATE_ENABLED = True              # nightly ML model replaces the live one unless clearly worse (ml_promotion_gate.py, 2026-09-29)
ML_FLOOR_CALIBRATION_ENABLED = False          # per-model ML floor: SHADOW until readout ML-FLOOR-SHADOW (2026-09-29)
ML_MODEL_HOT_RELOAD_ENABLED = True            # retrained ML model picked up without a restart (2026-09-29)
MODE_RANGE_QUALITY_GUARD_ENABLED = False      # OFF by operator 2026-10-01; shadow-logged (mrq-off-1001-spec.md)
MODEL_HEALTH_ENABLED = True                   # morning-report model health: own-target AUC, input PSI, run manifests (2026-10-05, model-health-1005-spec.md)
ML_PROMOTION_MATCHED_ADMISSION_ENABLED = False  # SHADOW: ML promotion + live floor at the incumbent's admitted share (2026-10-09, ml-admission-match-1009-spec.md)
RSI_DIV_HOLD_SHADOW_ENABLED = True            # SHADOW: logs weak_exit_shadow when a 15m RSI-divergence exit has >= +0.5% (readout WEAK-HOLD-SHADOW 11-30, weak-exit-shadow-1020-spec.md)
```

---

## 12. Telegram dedup

File: `.runtime/tg_send_dedup.json`

| Report | Cooldown | Key |
|--------|----------|-----|
| RL train complete | 30 min | `train_session` |
| Top-gainer critic midday | 60 min | `top_gainer_midday_YYYY-MM-DD` |
| Top-gainer critic final | 60 min | `top_gainer_final_YYYY-MM-DD` |

Rationale: multiple worker instances were each sending independent reports.

---

## 13. Secrets

`TELEGRAM_BOT_TOKEN` and similar never go into committed files, settings, or scripts. `.claude/settings.local.json` has deny-rules matching known token prefixes as a safety rail. If a token leaks anywhere, rotate it in @BotFather immediately.

Runtime token storage: `.runtime\bot_bg_runner.cmd` (runtime-generated, gitignored).

---

## 14. Watchlist

93 Binance USDT coins. File: `files/watchlist.json`.

**Nine more delisted pairs removed 2026-10-01** (BAKE, SNT, MKR, LRC, MDT, OXT,
TRU, TON, PYR — status BREAK, dead 45-533 days; backup
`files/watchlist.pre_dead_cleanup_1001.json`); `DEFAULT_WATCHLIST` cleaned of them
and of RNDR/EOS/ACA. `files/watchlist_liveness.py` runs daily and puts a P1 step in
the morning report when a pair stops trading. Spec:
`docs/specs/features/watchlist-dead-1001-spec.md`.

**Delisted entries were removed 2026-08-19** (`RNDRUSDT`, `EOSUSDT`, `ACAUSDT`;
backup at `files/watchlist.pre_phantom_cleanup.json`). This is not an exception
to the rule above — the rule forbids **expanding**, and these three had not
printed a candle since 2024-07, 2025-05 and 2026-02. `RNDR` was renamed to
`RENDER` and both halves were listed, so the dead half cost a slot and nothing
else.

They were not inert. `/api/v3/ticker/24hr` **keeps answering for delisted pairs**
with a non-zero volume, so the snapshot built full feature rows for them:
`EOSUSDT` carried `tg_return_since_open = 6.79` with no candle since May 2025,
and the early ranking put it **first**. Three of ten picks were phantoms.
`phantom_filter.is_live()` now guards against the next rename — liveness comes
from the immutable label store, which is built from klines and cannot be faked
by a ticker. It fails OPEN: if more than 25% of the universe looks stale that is
a broken store, not a dead market, so the filter stands down and says so.

**IMMUTABLE — DO NOT EXPAND.** These are the specific coins the user has access to for trading. Adding other symbols is pointless regardless of backtest results. If a top-gainer today is not in this list — that's expected and acceptable, not a bug to fix.

**Scope (operator decision #2, 2026-10-14):** this rule covers the SIGNAL BOT's watchlist only. The demo trader (`trader/`)
trades ALL pairs tradable on the Binance spot market. Its universe is `trader/runtime/trading/universe.json` (`trader/trading_universe.py`),
not `files/watchlist.json`. The signal bot's watchlist stays 93 coins and is not changed by this decision.

---

## 15. Session memory references

User's persistent memory tracks: project overview & metrics, Process Management (don't kill gpt_crypto_bot), Filter Overtightening lesson, Daily Learning Pipeline. Defer to those files for "why did we decide X" before proposing changes.

---

## 16. Maintenance of this file

**When anything in sections 2 / 4 / 5 / 7 / 8 / 11 changes — update BOTH `CLAUDE.md` and `PROJECT_CONTEXT.md` in the same commit.** They duplicate on purpose: `CLAUDE.md` is Claude's working brief, `PROJECT_CONTEXT.md` is the human-readable dossier.
