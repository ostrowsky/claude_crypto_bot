# Specs index

Spec-first workflow. См. [`../../AGENTS.md`](../../AGENTS.md) для процесса.

## Templates

- [`templates/feature-spec.md`](./templates/feature-spec.md) — feature/fix template

## Features

| Slug | Status | Summary |
|------|--------|---------|
| [`signal-pipeline`](./features/signal-pipeline-spec.md) | shipped | Сквозной pipeline (indicators → strategy → ML → ranker → bandit → guards → rotation → entry) и 7 entry-режимов. |
| [`contextual-bandit`](./features/contextual-bandit-spec.md) | shipped | LinUCB entry (2 arms) + trail (5 arms). Async reward, источники, training pipeline. |
| [`ml-candidate-ranker`](./features/ml-candidate-ranker-spec.md) | shipped | CatBoost ranker (quality, EV, expected_return/drawdown, TG-prob) + hard veto. |
| [`top-gainer-model`](./features/top-gainer-model-spec.md) | shipped | Daily CatBoost top-N классификаторы (top5/10/20/50) + intraday snapshots + critic. |
| [`portfolio-rotation`](./features/portfolio-rotation-spec.md) | shipped 2026-04-17 | ML-gated weak-leg eviction через soft-trail (`trail_stop = price × 1.001`). |
| [`correlation-guard`](./features/correlation-guard-spec.md) | shipped | Pearson log-return clustering (Union-Find) + cap позиций в кластере. |
| [`trend-quality-guard`](./features/trend-quality-guard-spec.md) | shipped | RSI / price-edge / daily-range cap для 15m `trend` (с bull-day relaxation). |
| [`daily-learning-pipeline`](./features/daily-learning-pipeline-spec.md) | shipped | EOD orchestrator: snapshot → resolve → train (bandit/TG/ranker/signal) → report. |
| [`trail-min-buffer`](./features/trail-min-buffer-spec.md) | shipped 2026-04-26 | Per-mode % floor на ATR-trail buffer для борьбы с whipsaw на impulse_speed/strong_trend. |
| [`anti-fast-reversal`](./features/anti-fast-reversal-spec.md) | draft | Label / model / guard / reward для отсечения быстрых разворотов (≤3 баров). |
| [`ml-signal-blindspot-recovery`](./features/ml-signal-blindspot-recovery-spec.md) | draft | Audit + oversampling weight для blind-spot syms (TRU/BLUR/MDT/ORDI/AUDIO). 24 % top-20 проходят через ML-block. |
| [`breakout-15m-disable`](./features/breakout-15m-disable-spec.md) | draft | Отключение `breakout/15m`: даже на 5/45 top-20 entries `avg_pnl=+0.03 %`. |
| [`eod-health-alert`](./features/eod-health-alert-spec.md) | draft | TG-алерт при `n_collected=0` / AUC drop / bandit stall в EOD-цикле. |
| [`metrics-framework`](./features/metrics-framework-spec.md) | draft | 13 метрик в 4 слоях (Coverage / Earliness / Quality / Discrimination) + north-star `EarlyCapture@top20`. Привязка каждой инициативы к target-метрике. |
| [`entry-event-logger-fix`](./features/entry-event-logger-fix-spec.md) | shipped 2026-04-30 | Добавлены `ranker_top_gainer_prob`, `ranker_ev`, `ranker_quality_proba`, `signal_mode`, `candidate_score` в entry-event payload. Разблокирует валидацию 1A и 4A. |
| [`dynamic-max-hold`](./features/dynamic-max-hold-spec.md) | draft | Продление `max_hold_bars` если ADX растёт + price > EMA20 + pnl>0. Validated +0.039 NS (capture 0.16→0.24). Whitelist: impulse_speed, strong_trend, trend, retest. |
| [`trend-1h-chop-filter`](./features/trend-1h-chop-filter-spec.md) | shipped 2026-05-01 v2.6.0 | Block `trend/1h` if `(ADX<25) OR (slope<1.2) OR (vol_x<1.3)`. Backtest: precision 1.2 %→16.7 %, recall 100 %, avg_pnl −0.17 %→+1.58 %. |
| [`trend-surge-precedence`](./features/trend-surge-precedence-spec.md) | shipped 2026-05-02 v2.7.0 (flagged off) | H3: surge_ok идёт ПЕРЕД entry_ok. Активация `TREND_SURGE_PRECEDENCE_ENABLED=True`. Acceptance: ≥5 reclassifications за 7 d shadow. |
| [`ex1-realized-potential`](./features/ex1-realized-potential-spec.md) | shipped 2026-05-02 v2.7.0 | Exit-side метрика. Baseline median EX1 = +0.001 на top-20. `_backtest_ex1_realized_potential.py` + интеграция в daily aggregator. |

## How to add a new spec

```
cp docs/specs/templates/feature-spec.md docs/specs/features/<slug>-spec.md
$EDITOR docs/specs/features/<slug>-spec.md
git add docs/specs/features/<slug>-spec.md
# … затем implement, verify, commit с `Spec: docs/specs/features/<slug>-spec.md` в commit body
```
