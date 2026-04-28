# Signal pipeline & 7 entry modes

- **Slug:** `signal-pipeline`
- **Status:** shipped (retroactive)
- **Created:** 2026-04-26
- **Owner:** core
- **Related:** `files/strategy.py`, `files/monitor.py`, `CLAUDE.md` §2

---

## 1. Problem

Бот должен в реальном времени просматривать ~105 USDT-перпетуалов Binance и
выдавать ранний BUY-сигнал на монеты, которые к концу дня окажутся в top-20
gainers. Источник правды по сетапам — `strategy.py`.

## 2. Success metric

**Primary:** earliest BUY на coin, попадающий в top-20 daily gainers.
**Secondary:** низкая доля fast-reversal-выходов (см. `anti-fast-reversal-spec`).

## 3. Scope

### In scope
- 7 entry-режимов и порядок их проверки.
- Сквозной pipeline: indicators → strategy → ML → ranker → bandit → guards → rotation → entry.

### Out of scope
- Reward-схема бандита (см. `contextual-bandit-spec`).
- ML-обучение (см. `daily-learning-pipeline-spec`).

## 4. Behaviour / design

```
Binance API
  └─► indicators.py  (EMA, ADX, RSI, ATR, MACD, vol_x, slope_pct, daily_range)
       └─► strategy.py  (7 modes, выдаёт candidate)
            └─► ml_signal_model  (proba ≥ ML_GENERAL_HARD_BLOCK_MIN)
                 └─► ml_candidate_ranker  (final_score, EV, top_gainer_prob)
                      └─► contextual_bandit  (entry arm: SKIP / ENTER)
                           └─► guards  (trend_quality, impulse_speed, ranker_hard_veto,
                                        correlation_guard, ml_proba_zone, mtf,
                                        clone_signal_guard, open_cluster_cap)
                                └─► rotation  (если 10/10, ML-gated eviction слабого)
                                     └─► entry  → Telegram
```

### 7 режимов

| Mode | TF | Файл/функция | Идея |
|------|----|----|------|
| `trend` | 15m | `strategy.py::trend_entry` | EMA20-pullback на восходящем тренде |
| `strong_trend` | 1h | `strategy.py::strong_trend_entry` | сильный 1h-тренд + ADX |
| `retest` | 1h | `strategy.py::retest_entry` | ретест ключевого уровня |
| `alignment` | 1h MTF | `strategy.py::alignment_entry` | согласование 1h/4h |
| `impulse` | 15m | `strategy.py::impulse_entry` | свечной импульс |
| `impulse_speed` | 1h / 15m | `strategy.py::impulse_speed_entry` | импульс с подтверждением скорости |
| `breakout` | 15m | `strategy.py::breakout_entry` | пробой high/low |

Бар-полл: `monitor.py::main_loop` → `_eval_symbol` → последовательная проверка
режимов до первого положительного.

### Контекст для бандита (input vector)

`slope_pct, adx, rsi, vol_x, ml_proba, daily_range, macd_hist, btc_vs_ema50,
is_bull_day, mode, tf` (см. `contextual-bandit-spec`).

## 5. Config flags & rollback

| Flag | Effect |
|------|--------|
| `BANDIT_ENABLED` | выключает entry-bandit, остаются только гарды |
| `ML_GENERAL_HARD_BLOCK_MAX/MIN` | ML proba-zone |
| `TREND_15M_QUALITY_GUARD_ENABLED` | trend_quality guard |
| `ML_CANDIDATE_RANKER_HARD_VETO_ENABLED` | ranker hard veto |
| `ROTATION_ENABLED` | портфельная ротация |
| `CORR_GUARD_ENABLED` | correlation guard |

Откат любого слоя — флипом флага без миграции данных.

## 6. Risks

- **Слой-каскад блокировок:** три гарда подряд однажды блокировали 100 % top
  gainers (см. CLAUDE.md §7 «Filter overtightening»). Правило: `>80 %`
  блокировки top-gainer’ов = гард сломан.
- **Race на 10/10:** сильный кандидат не попадает; покрывается `rotation-spec`.

## 7. Verification

- `files/analyze_blocked_gates.py` — Pareto sweep по reason_code.
- `files/test_rotation.py` — 21 unit-тест.
- Live: `bot_events.jsonl` агрегируется по `decision.reason_code`.

## 8. Follow-ups

- Расщепить spec по каждому из 7 режимов, если добавляются новые гарды per-mode.
- Спека на `anti-fast-reversal` (отдельно).
