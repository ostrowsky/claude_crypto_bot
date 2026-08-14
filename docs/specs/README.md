# Specs index

Spec-first workflow. См. [`../../AGENTS.md`](../../AGENTS.md) для процесса.
Обзор всей системы — [`../ARCHITECTURE.md`](../ARCHITECTURE.md).

## Templates

- [`templates/feature-spec.md`](./templates/feature-spec.md) — feature/fix template

## Features

| Slug | Status | Summary |
|------|--------|---------|
| [`signal-pipeline`](./features/signal-pipeline-spec.md) | shipped | Сквозной pipeline (indicators → strategy → ML → ranker → bandit → guards → rotation → entry) и 7 entry-режимов. |
| [`contextual-bandit`](./features/contextual-bandit-spec.md) | shipped | LinUCB entry (2 arms) + trail (5 arms). Async reward, источники, training pipeline. |
| [`ml-candidate-ranker`](./features/ml-candidate-ranker-spec.md) | shipped | CatBoost ranker (quality, EV, expected_return/drawdown, TG-prob) + hard veto. |
| [`top-gainer-model`](./features/top-gainer-model-spec.md) | shipped | Daily CatBoost top-N классификаторы (top5/10/20/50) + intraday snapshots + critic. |
| [`health-report-integrity`](./features/health-report-integrity-spec.md) | shipped 2026-08-13 | Completed-day critic fallback, data-quality alerts, comparable North-Star trend, and honest attribution wording. |
| [`truth-harness`](./features/truth-harness-spec.md) | shipped 2026-08-13 | TH-01…TH-12, full/staged compliance profiles, pre-commit enforcement and audit skill; current bot has open blocking findings. |
| [`operational-diagnostics`](./features/operational-diagnostics-spec.md) | shipped 2026-08-13 | `why_no_signal` answers "где сигналы по X?" from the logs; restart stack fails loudly with a test regression gate and `-FailIfNotRunning` status checks. |
| [`event-store`](./features/event-store-spec.md) | shipped 2026-08-13 (read path) | JSONL stays the journal; SQLite mirror synced from the last byte offset. Full sync 40.9s, re-sync 0.02s, aggregates verified identical. Write path still rewrites whole files. |
| [`control-plane-walking-skeleton`](./features/control-plane-walking-skeleton-spec.md) | shipped 2026-08-14 | Phase −1 of the target architecture: one attempt from `OBSERVED` to a verified terminal state in 1.4s, a corrupted result rejected before the governor, isolation enforced by AST checks. Proves the protocol; explicitly not market evidence. |
| [`four-store-split`](./features/four-store-split-spec.md) | shipped 2026-08-14 | P0: research memory severed from the execution channel. `config.py` now reads only the release store, written solely by `release_overrides.py` from signed approvals. Live gating verified identical across the migration; 7 legacy overrides carried as visible unsigned debt. |
| [`immutable-label-store`](./features/immutable-label-store-spec.md) | shipped 2026-08-14 | Phase 0a: later-EOD labels from exchange klines only (19 502 records, 199 well-covered days, 16.3 qualifying events/day), immutable with provenance, plus a day-grouped splitter with embargo. Fixes the TH-03 root cause; TH-04 helper ready to wire. |
| [`weekly-steering-pair`](./features/weekly-steering-pair-spec.md) | shipped 2026-08-14 | `Coverage@move` + `Precision@alert` on one MoveEvent definition, day-clustered CIs, downtime excluded. First reading: coverage 0.051, early-alert lift 1.04× vs all-alert 3.96× — the bot confirms moves rather than predicting them. |
| [`day-grouped-training-split`](./features/day-grouped-training-split-spec.md) | shipped flagged-off 2026-08-14 | TH-04: the row-index cut straddles a UTC day. Splitter wired behind `TRAIN_DAY_GROUPED_SPLIT_ENABLED=False`. Max-period evidence: **no measurable AUC difference**, because TH-03 label leakage saturates the metric — flip it with the label fix. |
| [`north-star-immutable-labels`](./features/north-star-immutable-labels-spec.md) | shipped 2026-08-14 | TH-03: the North Star is published a second time on exchange-kline later-EOD labels, beside the old value and marked non-comparable. Same-rule max-period evidence: winner sets differ on 38% of pairs, EC 0.0243 → 0.0236 — provenance changes, the number does not. Model flag `TRAIN_IMMUTABLE_LABELS_ENABLED` stays off. |
| [`top-gainer-immutable-training-labels`](./features/top-gainer-immutable-training-labels-spec.md) | shipped flagged-off 2026-08-15 | TH-03: `top_gainer_model` can train on immutable later-EOD labels (rank within the store universe AND a +5% floor). Max-period evidence: AUC 0.99 -> 0.83 across all four tiers, the honest level. **Flip blocked:** watchlist-scoped tiers collapse — top20 and top50 are byte-identical on the holdout, so the live tier ladder's thresholds would all be mis-calibrated at once. |
| [`continuous-improvement-agent`](./features/continuous-improvement-agent-spec.md) | design v2 | LLM proposes, deterministic code disposes. v2 after external review: research memory split from the executable channel, hash-chained provenance instead of byte offsets, CI-vs-threshold instead of MDE, one MoveEvent for both weekly metrics, defined canary, liveness. Phase 0 is repairing measurement. |
| [`portfolio-rotation`](./features/portfolio-rotation-spec.md) | shipped 2026-04-17 | ML-gated weak-leg eviction через soft-trail (`trail_stop = price × 1.001`). |
| [`correlation-guard`](./features/correlation-guard-spec.md) | shipped | Pearson log-return clustering (Union-Find) + cap позиций в кластере. |
| [`trend-quality-guard`](./features/trend-quality-guard-spec.md) | shipped | RSI / price-edge / daily-range cap для 15m `trend` (с bull-day relaxation). |
| [`daily-learning-pipeline`](./features/daily-learning-pipeline-spec.md) | shipped | EOD orchestrator: snapshot → resolve → train (bandit/TG/ranker/signal) → report. |
| [`trail-min-buffer`](./features/trail-min-buffer-spec.md) | shipped 2026-04-26 | Per-mode % floor на ATR-trail buffer для борьбы с whipsaw на impulse_speed/strong_trend. |
| [`anti-fast-reversal`](./features/anti-fast-reversal-spec.md) | draft | Label / model / guard / reward для отсечения быстрых разворотов (≤3 баров). |
| [`ml-signal-blindspot-recovery`](./features/ml-signal-blindspot-recovery-spec.md) | draft | Audit + oversampling weight для blind-spot syms (TRU/BLUR/MDT/ORDI/AUDIO). 24 % top-20 проходят через ML-block. |
| [`breakout-15m-disable`](./features/breakout-15m-disable-spec.md) | draft | Отключение `breakout/15m`: даже на 5/45 top-20 entries `avg_pnl=+0.03 %`. |
| [`eod-health-alert`](./features/eod-health-alert-spec.md) | draft | TG-алерт при `n_collected=0` / AUC drop / bandit stall в EOD-цикле. |
| [`metrics-framework`](./features/metrics-framework-spec.md) | provisional | 13 метрик в 4 слоях + North Star `EarlyCapture@top20`; immutable later-EOD ground truth ещё требуется. |
| [`entry-event-logger-fix`](./features/entry-event-logger-fix-spec.md) | shipped 2026-04-30 | Добавлены `ranker_top_gainer_prob`, `ranker_ev`, `ranker_quality_proba`, `signal_mode`, `candidate_score` в entry-event payload. Разблокирует валидацию 1A и 4A. |
| [`dynamic-max-hold`](./features/dynamic-max-hold-spec.md) | draft | Продление `max_hold_bars` если ADX растёт + price > EMA20 + pnl>0. Validated +0.039 NS (capture 0.16→0.24). Whitelist: impulse_speed, strong_trend, trend, retest. |
| [`trend-1h-chop-filter`](./features/trend-1h-chop-filter-spec.md) | shipped 2026-05-01 v2.6.0 | Block `trend/1h` if `(ADX<25) OR (slope<1.2) OR (vol_x<1.3)`. Backtest: precision 1.2 %→16.7 %, recall 100 %, avg_pnl −0.17 %→+1.58 %. |
| [`trend-surge-precedence`](./features/trend-surge-precedence-spec.md) | shipped 2026-05-02 v2.7.0 (flagged off) | H3: surge_ok идёт ПЕРЕД entry_ok. Активация `TREND_SURGE_PRECEDENCE_ENABLED=True`. Acceptance: ≥5 reclassifications за 7 d shadow. |
| [`ex1-realized-potential`](./features/ex1-realized-potential-spec.md) | shipped 2026-05-02 v2.7.0 | Exit-side метрика. Baseline median EX1 = +0.001 на top-20. `_backtest_ex1_realized_potential.py` + интеграция в daily aggregator. |
| [`signal-evaluator-integration`](./features/signal-evaluator-integration-spec.md) | Phase A+B+D shipped v2.10.0 | Phase A wrapper + Phase B `files/zigzag_labeler.py` module + Phase D EX1 `--use-zigzag` + per-mode wrapper option. Phase C (bandit verdict-reward) — drafted. |
| [`metrics-canonical`](./features/metrics-canonical-spec.md) | shipped 2026-05-04 v2.10.0 | Canonical map: 1 metric per business question. Защита от metric-soup при добавлении skill. |
| [`h5-trailing-only-break-even`](./features/h5-trailing-only-break-even-spec.md) | shipped 2026-05-02 v2.8.0 (shadow-on, enabled-off) | Suppress soft EMA-pattern exits when pnl >= +0.5 %. Backtest: 4 eligible / 30 d, 1 top-20 worth +471 % left on table. Acceptance: 7 d shadow с ≥3 events. |

## How to add a new spec

```
cp docs/specs/templates/feature-spec.md docs/specs/features/<slug>-spec.md
$EDITOR docs/specs/features/<slug>-spec.md
git add docs/specs/features/<slug>-spec.md
# … затем implement, verify, commit с `Spec: docs/specs/features/<slug>-spec.md` в commit body
```
