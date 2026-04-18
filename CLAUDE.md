# CLAUDE.md — Guide for Claude working in this repo

Auto-injected into Claude Code's system prompt. Read first, trust it, don't re-discover.
**Companion:** `PROJECT_CONTEXT.md` — the long-form project dossier. Keep both in sync: when architecture, filters, schedules, or known issues change, update BOTH files.

---

## 1. Purpose & success metric

Telegram bot scanning Binance Futures/Spot in real time, generating early BUY signals on coins that will be in the daily top gainers by EOD.

**Main metric:** earliest possible BUY on a coin that ends up in top-20 daily gainers (before the main move).

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

---

## 15. Session memory references

User's persistent memory tracks: project overview & metrics, Process Management (don't kill gpt_crypto_bot), Filter Overtightening lesson, Daily Learning Pipeline. Defer to those files for "why did we decide X" before proposing changes.

---

## 16. Maintenance of this file

**When anything in sections 2 / 4 / 5 / 7 / 8 / 11 changes — update BOTH `CLAUDE.md` and `PROJECT_CONTEXT.md` in the same commit.** They duplicate on purpose: `CLAUDE.md` is Claude's working brief, `PROJECT_CONTEXT.md` is the human-readable dossier.
