# claude_crypto_bot — architecture

- **Updated:** 2026-08-14
- **Companions:** [`CLAUDE.md`](../CLAUDE.md) (working brief),
  [`PROJECT_CONTEXT.md`](../PROJECT_CONTEXT.md) (dossier),
  [`docs/specs/README.md`](specs/README.md) (per-feature specs)

This is the map: what the system is made of, which parts are real, which are
designed, and which are known to be broken. It is written to be read before
touching anything, and it states status honestly — a map that flatters the
territory is worse than no map.

---

## 1. What the bot is for

A Telegram bot that scans ~105 Binance futures symbols in real time and emits
**early BUY alerts** on coins that will end the day among the top gainers.

It is an **alert system, not a trader**: no position sizing, no capital, no
execution. That single fact decides many arguments downstream — profitability
metrics, portfolio alpha and per-trade P&L are diagnostics here, never the
objective.

**Objective:** `EarlyCapture@top20 = mean(coverage × realized_capture ×
time_lead)` over winner-days in `watchlist ∩ top-20`; goal 0.40, floor 0.25.

> **Status: provisional, and unfit for short-horizon steering.** Ground truth
> still comes from the same rolling-24h snapshot that produces the features, so
> the metric partly measures itself. And at ~20 winners/week its minimum
> detectable weekly difference is 2.1× its own value — a weekly comparison can
> only see it tripling or collapsing. Current reading: **0.070**, marked
> provisional. See [§7](#7-measurement-and-why-it-is-not-yet-trustworthy).

---

## 2. Status at a glance

| Plane | State |
|---|---|
| Runtime (signal → alert) | **operational** |
| Learning (bandit, models, nightly cycle) | **operational**, training label rebuilt 2026-08-13 |
| Evidence & observability | **mostly new**, shipped 2026-08-13/14 |
| Improvement loop (hypotheses → validation → promotion) | **dead**; redesigned, not built |
| Measurement integrity | **6 blocking harness findings**; NS provisional |
| Dataset write path | **broken** — whole-file rewrite fails while the bot runs |

---

## 3. Runtime plane — from candle to alert

```
Binance REST + WS
   → indicators.py            technical features
   → strategy.py              7 entry modes
   → ml_signal_model.py       CatBoost gate (ML proba zone)
   → ml_candidate_ranker.py   CatBoost ranker + hard veto
   → contextual_bandit.py     LinUCB enter/skip  ← the online decision
   → guards                   trend quality · 1h chop · impulse-speed ·
                              correlation · clone · cluster cap · cooldown
   → rotation.py              ML-gated weak-leg eviction
   → bot.py                   Telegram alert
```

**Entry modes:** `trend` (15m) · `strong_trend` (1h) · `retest` (1h) ·
`alignment` (1h MTF) · `impulse` (15m) · `impulse_speed` (1h) · `breakout`.

**Monitored set — the entire watchlist.** `MONITOR_FULL_WATCHLIST = True`, so
"the coin was not being watched" is no longer a possible cause of a miss. Cost
is bounded by rotation, not by set size: `MAX_POLL_PER_CYCLE = 45` per 60-second
tick sweeps ~105 symbols in ~3 minutes, and **symbols with an open position are
polled every cycle** regardless.

**Live behaviour flags** (each is a rollback switch):

| Flag | Value | Meaning |
|---|---|---|
| `BANDIT_ENABLED` | True | LinUCB is the online gate |
| `BANDIT_FORWARD_REWARD_ENABLED` | True | reward follows the move still ahead (§4) |
| `ROTATION_ENABLED` | True | weak-leg eviction |
| `REGIME_SOFT_GATE_ENABLED` | True | below-floor entry_score falls through to the bandit |
| `TREND_1H_CHOP_FILTER_ENABLED` | True | chop guard on `trend/1h` |
| `IMPULSE_SPEED_REGIME_CURTAIL_ENABLED` | True | + `..._FALLBACK_TO_TREND=True` — curtailed mode is reclassified, not blocked |
| `H5_TRAILING_ONLY_AFTER_BREAK_EVEN_ENABLED` | True | soft EMA exits suppressed above +0.5% |
| `DECOUPLING_SHADOW_ENABLED` / `..._GATE_ENABLED` | True / False | scored and logged, no decision impact |
| `CORR_GUARD_SHADOW` | False | correlation guard enforcing |
| `FAST_REVERSAL_LEARNING_ENABLED` | False | §4a work not enabled |
| `TREND_SURGE_PRECEDENCE_ENABLED` | False | shadow acceptance not met |
| `BANDIT_REGIME_INTERACTION_ENABLED` | False | backtest neutral, held off |
| `UI_WATCHDOG_FORCE_EXIT_AFTER_WARNS` | 0 | disabled — it force-exited into no restart wrapper |

---

## 4. Learning plane

**Entry bandit (LinUCB, 2 arms).** Context: slope, ADX, RSI, vol_x, ml_proba,
daily range, MACD hist, BTC vs EMA50, bull-day, mode, tf (+2 reserved regime
interactions).

**The training label was rebuilt on 2026-08-13** and this is the single most
consequential change in the learning plane. Reward previously followed
`label_top20` on the earliest snapshot of a day — which is the 00 UTC record,
i.e. the EOD resolution of a day already over. 34.8% of trained rows had nothing
left to decide, and a fresh bandit on that label scored **lift 0.65× — below
random**. Reward now follows the move **remaining after the snapshot**: the
day's top-`BANDIT_FORWARD_TOP_N=10` by forward move, and at least
`BANDIT_FORWARD_MIN_PCT=3.0`%.

| training label | ENTER | caught | lift |
|---|---|---|---|
| old `label_top20` | 80.3% | 52% | **0.65×** |
| rank top-20, no floor | 98.0% | 100% | 1.02× |
| **rank top-10 + floor +3%** | **24.3%** | **99%** | **4.07×** |

Two defects in the same path went with it: the dataset was read as the **first**
50k of 118 625 lines, freezing the training window at 2026-06-05 for 69 days;
and every run did `batch_update` onto saved state, accumulating 8.39M updates
from ~44.6k unique samples. State is now rebuilt from scratch each run
(`BANDIT_REBUILD_ON_TRAIN`), reading the tail (`BANDIT_TG_MAX_RECORDS`).

**Honest current reading** (`evaluate_bandit_accuracy`, fresh bandit on earlier
days grading the last 7):

| scope | recall | ENTER rate | base | lift |
|---|---|---|---|---|
| out-of-sample (45 train days / 7 eval) | 74.2% | 35.3% | 8.1% | **2.10×** |
| in-sample echo (same live bandit) | 85.5% | 57.8% | 8.1% | 1.48× |

Both are printed side by side on purpose: the echo reads higher on recall and
worse on lift, and collapsing them into one number is exactly what §0 rule 3
forbids.

**Other learners:** trail bandit (5 arms), CatBoost top-gainer tiers
(top5/10/20/50, nightly), CatBoost candidate ranker (RL worker, hourly), signal
model. **Everything before 2026-08-13 in the learning-progress history was
measured against the leaky label and is not comparable with what follows.**

---

## 5. Data layer

### 5.1 Journals

| File | Size | Role |
|---|---:|---|
| `top_gainer_dataset.jsonl` | 149 MB | all watchlist coins × daily snapshots |
| `critic_dataset.jsonl` | 140 MB | bot signals with outcomes |
| `ml_dataset.jsonl` | 115 MB | raw ML rows |
| `bot_events.jsonl` | 99 MB | every entry / block / exit / shadow event |

Never read these directly — write a script that streams and prints a summary, or
query the store below.

### 5.2 Event store (new)

`files/event_store.py`: the JSONL stays authoritative and is never rewritten; a
SQLite mirror is synced from the last consumed byte offset. Full sync of
245 967 rows: 40.9s. Re-sync with nothing appended: **0.02s**.

Guarantees: primary key `(source_file, byte_offset)` so a replayed sync cannot
double-count; an offset past the current size means the source was rotated or
truncated, so its rows are dropped rather than spliced; a line without its
trailing newline is a writer mid-append and waits.

Byte offsets are for **incremental reading only** — they identify a position,
not content, and this repo rewrites its datasets. Content-addressed provenance
(hash chain over prefixes) is designed, not built.

### 5.3 The write path is broken

Every dataset update rewrites the whole file. On Windows, a file the live bot
holds open cannot be replaced:

```
PermissionError: [WinError 5]
  critic_dataset.jsonl.backfill.tmp -> critic_dataset.jsonl
```

The evidence: **19 orphaned `.tmp` files, 1.09 GB**, owned by dead PIDs, the
oldest 40 days, cleared on 2026-08-13. Label backfills therefore require a
stopped bot — one was run that way the same day, filling 7 136 rows and bringing
labels to 99.9–100%. Fixing this is the open half of roadmap item G1.

---

## 6. Evidence and observability plane (new)

| Component | Answers |
|---|---|
| `why_no_signal.py` | "почему нет сигнала по X?" — reads events backwards from EOF; POL over 3 days = 7 804 lines scanned, verdict `blocked:trend_quality` 100% |
| `block_reasons.py` | one stable code per gate across two languages, 310 free-text templates and 22 short codes |
| `artifact_freshness.py` | declared max age per learning input; flag-gated artifacts report `disabled`, not `stale` |
| `truth_harness.py` | TH-01…TH-12, full and staged-change profiles, pre-commit hook |
| `run_test_suite.py` | regression gate — fails only on NEW failures against a recorded baseline |
| `restart_full_stack.bat` | tests → stop → start → **verify**, non-zero exit if a worker did not come up |

Two of these exist because of specific silent failures: a backfill lock held
**58 days** (11 908 rows unlabelled, nothing in any report), and a restart that
reported success while `bot.py` never started.

---

## 7. Measurement, and why it is not yet trustworthy

`truth_harness full` today: **6 blocking findings, 1 warning.**

| Finding | What it means |
|---|---|
| TH-03 ×2 | North Star and top-gainer targets use same-snapshot labels |
| TH-04 | model validation can split one UTC day across train and holdout |
| TH-11 ×2 | no canonical portfolio alpha, no canonical ZigZag EX1 |
| TH-10 | gate evidence 77 days old against a 30-day budget |
| TH-08 (warn) | 47 legacy backtests with no durable verdict |

Metrics now name their own denominator (`denominator`,
`label_provenance` in the emitted artifacts) and gates are ranked by **harm**
— winners lost — rather than by how often they fired:

```
trend_quality   2 winners  7.7% of all top-20
ml_proba_zone   2 winners  7.7% of all top-20
```

**The truth harness (CLAUDE.md §0a) is the rule set the whole system is judged
by.** Its twelve invariants exist because each was violated here and cost real
time: a ratio without its base rate, a feature that contains the answer, an
in-sample number reported as achievement, incomparable windows, a metric that
does not know what it does not know, a gate validated on the market instead of
the bot's own entries, behaviour changed without a flag, negative results
discarded, unverifiable claims, reports that conclude past their data, proxies
sold as outcomes, and changes untraceable to evidence.

---

## 8. Improvement plane — designed, not built

The loop that is supposed to make the bot better is **dead**: L1–L7 are wired
and scheduled, yet all 16 pending hypotheses referenced config keys that do not
exist, none had a registered validator, and **no decision has come through the
pipeline since 2026-06-17**. Every change since came from manual analysis.

The replacement is designed in
[`continuous-improvement-agent-spec.md`](specs/features/continuous-improvement-agent-spec.md)
(v2). Its shape: an LLM proposes, deterministic code disposes; four separate
stores so no agent can reach the execution channel; a validation service with a
sealed holdout, purge/embargo and placebo runs; judges that are advisory only;
and a promotion path of shadow → symbol-subset canary → flagged live with
permanent operator approval.

**Phase 0 is repairing measurement, not launching agents** — immutable
later-EOD labels, day-grouped splits, provenance fields, until the harness is
green.

---

## 9. Operational topology

| Process | Purpose |
|---|---|
| `bot.py` | Telegram UI + monitoring loop |
| `rl_headless_worker.py` | ranker training, backfill, critic |

Scheduled: `CryptoBot_DailyLearning_EOD` 02:30 local (full nightly cycle),
`CryptoBot_IntradaySnapshot` 08:30 / 14:30 / 20:30. All CryptoBot tasks had
`DisallowStartIfOnBatteries` set, which silently skipped them for 11 days —
check power settings before debugging a silent scheduler.

Restart with `restart_full_stack.bat` (verifies) rather than `restart_bot.bat`
(interactive, and its `cmd /k` self-relaunch hides the exit code).

**Never touch `D:\Projects\gpt_crypto_bot\`** — a separate bot with its own
Telegram token. Always check the command line before stopping any python PID.

---

## 10. Invariants that hold everywhere

1. The watchlist is immutable (105 symbols).
2. Behaviour changes ship behind a flag whose default is current behaviour, with
   a stated rollback.
3. Logging a decision auto-applies its `diff.to` as a runtime override, so a
   rollback needs a superseding record — not a `config.py` edit.
4. Every ratio is published with its base rate and lift.
5. Holdout only, split by time; in-sample numbers are never an achievement.
6. Days the bot was down are `no data`, never misses.
7. A gate is validated on the bot's own entries, not on the market.
8. Negative results are committed with the numbers that killed them.
9. Secrets never enter committed files, scripts, or printed output.
10. `CLAUDE.md` and `PROJECT_CONTEXT.md` change together.

---

## 11. Known defects, ranked

| # | Defect | Consequence |
|---|---|---|
| 1 | Whole-file dataset rewrites | label backfill needs a stopped bot; failure leaves GB of orphans |
| 2 | North Star labels are same-snapshot | the objective partly measures itself; unfit for weekly steering |
| 3 | Improvement loop dead | every change is manual; no compounding |
| 4 | Test suite red at baseline | 40 of ~790 failing; new breakage is caught, old debt persists |
| 5 | Model validation splits by row index | one UTC day can straddle the split |
| 6 | No profitability evidence | TH-11 open; deliberate for an alert product, but undocumented as such |
| 7 | 47 backtests without durable verdicts | refuted work will be re-done |

---

## 12. Map — spec to component

| Area | Spec |
|---|---|
| End-to-end pipeline | `signal-pipeline` |
| Bandit | `contextual-bandit` |
| Ranker / models | `ml-candidate-ranker`, `top-gainer-model` |
| Nightly cycle | `daily-learning-pipeline` |
| Metrics | `metrics-framework`, `metrics-canonical`, `health-report-integrity` |
| Integrity | `truth-harness` |
| Diagnostics & restart | `operational-diagnostics` |
| Event store | `event-store` |
| Improvement loop | `continuous-improvement-agent` |
| Transfer roadmap | [`gpt-bot-transfer`](roadmaps/gpt-bot-transfer-roadmap.md) |
