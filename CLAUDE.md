# CLAUDE.md — Guide for Claude working in this repo

Auto-injected into Claude Code's system prompt. Read first, trust it, don't re-discover.
**Companion:** `PROJECT_CONTEXT.md` — the long-form project dossier. Keep both in sync: when architecture, filters, schedules, or known issues change, update BOTH files.

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

The full state of the auto-improvement loop — what's implemented, what's not,
what the North Star metric is doing — lives in
[`docs/specs/features/auto-improvement-loop-spec.md`](docs/specs/features/auto-improvement-loop-spec.md).
**That spec is a living document. Every PR that touches a loop component
must update its status row and the North Star progress table.**

---

## 1. Purpose & success metric

Telegram bot scanning Binance Futures/Spot in real time, generating early BUY signals on coins that will be in the daily top gainers by EOD.

**Main metric (North Star):** earliest possible BUY on a coin that ends up in top-20 daily gainers (before the main move). Operationalised as
`watchlist_top_early_capture_pct = #(bot bought top-20 with capture_ratio ≥ 0.35) / #(top-20 in watchlist)`.

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
| `files/ml_signal_model.py` | CatBoost signal classifier. |
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
| `files/top_gainer_dataset.jsonl` (~29 MB) | Features of ALL watchlist coins × N daily snapshots |
| `files/critic_dataset.jsonl` (~36 MB) | Bot signals with outcomes (nested: `decision.action`, `decision.reason_code`, `labels.ret_5`, `labels.label_5`) |
| `files/bot_events.jsonl` (~17 MB) | All bot events (entry / blocked / exit) |
| `files/ml_dataset.jsonl` (~103 MB) | Raw ML training data |

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

| Situation | Reward |
|-----------|--------|
| ENTER + top-20 gainer | +1.0 |
| SKIP + top-20 gainer | **-0.8** (miss penalty) |
| SKIP + not top gainer | +0.10 |
| ENTER + not top gainer | -0.12 |
| ENTER + fast reversal (≤3 bars, P&L < -0.5%) | **-0.6** (planned, see § 4a) |
| SKIP + fast reversal | **+0.30** (planned) |

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
1. **Primary:** `top_gainer_dataset.jsonl` — ALL ~105 watchlist coins × N daily snapshots.
2. **Secondary:** `critic_dataset.jsonl` — real bot signals with outcomes.

### Top Gainer Model (CatBoost)
Tier classifiers: top5 / top10 / top20 / top50. Retrained daily via `daily_learning.py`. Metrics: AUC, recall@0.3.

### Learning progress (as of 2026-04-13)

| Date | Recall@20 | UCB Sep | Updates |
|------|-----------|---------|---------|
| Apr 07 | 99.0% | +0.047 | 86,751 |
| Apr 13 | 100.0% | **+0.112** | 362,594 |

UCB separation growing (+138% in a week) — bandit is learning to separate winners.

### Learning progress (as of 2026-04-17 / analysed 2026-04-18)
- Recall@20 = 100% since Apr 10.
- UCB sep: 0.015 → **0.136** (x9).
- Model AUC top20: 0.61 → **0.93** (jump on Apr 15→16: 0.68→0.89).
- `bandit_n_signal` hit hard cap `8000` on Apr 15 and stuck → fixed: `BANDIT_CRITIC_MAX_RECORDS=25_000` in `config.py`, wired into `offline_rl.py`.

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
| `CryptoBot_DailyLearning_EOD` | 02:30 local (00:30 UTC) | Full cycle: snapshot → resolve → train bandit → retrain model → report |
| `CryptoBot_IntradaySnapshot` | 08:30 / 14:30 / 20:30 local | Feature snapshot → `top_gainer_dataset` |

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

---

## 7. Known issues & fixes

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

### Critical bugs (fixed)

| Date | Problem | Fix |
|------|---------|-----|
| 2026-04-13 | `is_bull_day_now` used before definition in `monitor.py` (~L3125 vs L3296) → silent crash each cycle, 0 signals overnight | Defined before `_impulse_speed_entry_guard` call |
| 2026-04-06 | 0 entries despite 12 top gainers (+5–59%) — all blocked | See "Filter overtightening" below |

### Filter overtightening (2026-04-06)

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
оставлено +471 % potential на столе. Default: SHADOW=True (logging),
ENABLED=False (поведение не меняется до acceptance).
Acceptance перед flip → True: 7 d shadow с ≥3 events.
Helper: `monitor.py::_h5_should_suppress` (~ L1745).
Wired в exit-pipeline после WEAK / TREND_HOLD overrides (~ L5325).
Configs:
```python
H5_TRAILING_ONLY_AFTER_BREAK_EVEN_ENABLED = False
H5_TRAILING_ONLY_SHADOW = True
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
TREND_1H_CHOP_SLOPE_MIN = 1.2
TREND_1H_CHOP_VOL_MIN   = 1.3
TREND_1H_CHOP_USE_BULL_DAY_RELAX = False  # opt-in
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
TRAIL_MIN_BUFFER_PCT_IMPULSE_SPEED = 0.08    # 8% (was 1.5%) — EX1 capture fix 2026-06-01
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
BANDIT_ENABLED = True
ML_GENERAL_HARD_BLOCK_MAX = 1.01              # 1.01 = no upper cap
TOP_GAINER_CRITIC_ENABLED = True
RL_TRAIN_TELEGRAM_REPORTS_ENABLED = True
TOP_GAINER_CRITIC_TELEGRAM_REPORTS_ENABLED = True
TREND_15M_QUALITY_GUARD_ENABLED = True
ML_CANDIDATE_RANKER_HARD_VETO_ENABLED = True
ROTATION_ENABLED = True
REGIME_SOFT_GATE_ENABLED = True               # RM-22 Step C (deployed 2026-06-01)
BANDIT_REGIME_INTERACTION_ENABLED = False     # RM-22 Step B (neutral, held OFF)
DECOUPLING_SHADOW_ENABLED = True              # decoupling shadow logging (2026-05-07)
DECOUPLING_GATE_ENABLED = False               # reserved; needs shadow-replay first
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

~105 Binance USDT perpetual futures coins. File: `files/watchlist.json`.

**IMMUTABLE — DO NOT EXPAND.** These are the specific coins the user has access to for trading. Adding other symbols is pointless regardless of backtest results. If a top-gainer today is not in this list — that's expected and acceptable, not a bug to fix.

---

## 15. Session memory references

User's persistent memory tracks: project overview & metrics, Process Management (don't kill gpt_crypto_bot), Filter Overtightening lesson, Daily Learning Pipeline. Defer to those files for "why did we decide X" before proposing changes.

---

## 16. Maintenance of this file

**When anything in sections 2 / 4 / 5 / 7 / 8 / 11 changes — update BOTH `CLAUDE.md` and `PROJECT_CONTEXT.md` in the same commit.** They duplicate on purpose: `CLAUDE.md` is Claude's working brief, `PROJECT_CONTEXT.md` is the human-readable dossier.
