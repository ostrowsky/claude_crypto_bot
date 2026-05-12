# Canonical metrics — single source of truth per business question

- **Slug:** `metrics-canonical`
- **Status:** shipped 2026-05-04 v2.10.0
- **Created:** 2026-05-04
- **Owner:** core
- **Related:** `metrics-framework-spec.md`,
  `signal-evaluator-integration-spec.md`.

---

## 1. Problem

Бот сейчас имеет ≥ 13 + 8 (skill) метрик. Без явной canonical map
команда читает противоречивые числа на одном вопросе и принимает
неоптимальные решения.

Пример: «бот зарабатывает?» можно ответить через:
- avg_pnl (наивно, без fees, без opportunity cost)
- EX1 mean (capture vs proxy)
- EarlyCapture@top20 (агрегат, размытый)
- skill alpha_vs_buy_and_hold (pure)

Цифры **не совпадают**. Без явного primary всем приходится держать
все четыре в голове.

## 2. Решение

Каждый бизнес-вопрос → **один** canonical metric. Остальные —
diagnostic / supporting, в primary не участвуют.

## 3. Canonical map

| Бизнес-вопрос | Canonical metric | Источник | Cadence |
|---------------|------------------|----------|---------|
| **«Бот зарабатывает деньги?»** | `alpha_vs_buy_and_hold_pct` | skill `report.json.summary` | weekly |
| **«Бот видит победителей?»** | `recall@top20` | `learning_progress.jsonl` (existing) | daily |
| **«Качество отдельной сделки?»** | per-trade `verdict` | skill `report.json.trade_verdicts[]` | per trade |
| **«Где утекает прибыль?»** | EX1 (Phase D ZigZag mode) | `_backtest_ex1_realized_potential.py --use-zigzag` | weekly |
| **«Бандит учится?»** | `bandit_ucb_separation` | `report_rl_daily.py` | daily |
| **«Сигналы в TG не шум?»** | `signal_precision_top20` (D1) | `_backtest_signal_precision.py` | daily |
| **«Канал не спамит?»** | `tg_message_rate` (D2) | `_backtest_signal_precision.py` | daily |
| **«Раннее срабатывание?»** | `time_to_signal` median (E1) | `_backtest_time_to_signal.py` | daily |
| **«Уверенно ли торгуем?»** | `whipsaw_rate` (Q2) | `_backtest_whipsaw_rate.py` | daily |
| **«Fast-reversal под контролем?»** | `fast_reversal_rate` (Q1) | `_backtest_fast_reversal_by_mode.py` | daily |

### Holistic single-number (для дашборда / TG-digest)

`EarlyCapture@top20` (north-star) — единственное число, агрегирующее
coverage × capture × time_lead. **Не canonical** для конкретных
вопросов, **canonical** для тренда «куда движемся».

## 4. Что DEPRECATED после Phase D

| Метрика | Replacement | Когда удалить |
|---------|-------------|---------------|
| EX1 в proxy-mode | EX1 `--use-zigzag` | после 30 d klines backfill |
| E2 capture_ratio (proxy) | EX1 (после Phase D) | через 7 d shadow |
| `_backtest_capture_ratio.py` в `report_metrics_daily.py` | удалить из aggregator | то же |
| Standalone `top_gainer_critic` отчёты в #public TG | переместить в #admin | сразу |

## 5. Что НЕ deprecate

- Все Phase 1 metric scripts остаются доступны для ad-hoc анализа.
- Они **не canonical**, но полезны как diagnostic surface.

## 6. Реализация

- [x] Эта спека подписана как single source of truth.
- [x] `report_metrics_daily.py` дополнен ссылкой в docstring на эту таблицу.
- [ ] TG-digest (отдельная спека): берёт **только** canonical metrics
  и постит в `#weekly-review`.
- [ ] Удалить `_backtest_capture_ratio.py` из `SCRIPTS` в
  `report_metrics_daily.py` после Phase D shadow.

## 7. Antipatterns

- **Не добавлять новую метрику без updating этой таблицы.** Если
  метрика не отвечает на новый бизнес-вопрос или не заменяет
  существующий canonical — она **diagnostic**, и попадает в
  «остальные», не в canonical.
- **Не публиковать non-canonical метрики в TG для пользователей.**
  Только canonical уровни. Diagnostic — для разработчика.
- **Не использовать `recall@top20` как «бот зарабатывает».** Recall
  меряет видимость в bandit-датасете; `alpha` меряет деньги.

## 8. Follow-ups

- TG-digest spec: canonical-metrics weekly summary в TG.
- Phase D klines backfill: prefetch на 30 d ВСЕХ ~105 watchlist
  coin (отдельная background-задача, ~2-4 ч API-вызовов).
- Через 30 d — review: если canonical metric набор оказался
  неполный, добавить **с явным review** этой таблицы.
