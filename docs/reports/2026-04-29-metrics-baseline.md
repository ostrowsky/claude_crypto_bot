# Metrics baseline — 2026-04-29

Первое полное измерение всех 13 метрик из
`docs/specs/features/metrics-framework-spec.md`. Источники: 7 backtest-скриптов
+ daily aggregator `report_metrics_daily.py`. Окно: 14 d (Coverage/Earliness)
и 30 d (Quality/Discrimination), указано per-метрика.

---

## North-star: `EarlyCapture@top20`

```
EarlyCapture@top20 = 0.074    (target 0.40, gap −82 %)

decomposition (для 66 winner-days, 14 d):
  coverage         = 0.697      (target 0.90, gap −22 %)
  capture_ratio    = 0.161      (target 0.50, gap −68 %)  ← главный отстающий
  time_lead_score  = 0.685      (target 0.85, gap −19 %)
```

**Capture — катастрофический отстающий.** Median realized capture-ratio
для top-20 winner = **0.002** (т.е. ловим почти ноль от дневного движа).
Это на порядок хуже, чем я предполагал в исходном estimate (0.50).

---

## Полная таблица метрик

| ID | Метрика | Текущее | Цель | Δ | Status |
|----|---------|--------:|-----:|--:|:-------|
| **NS** | EarlyCapture@top20 | **0.074** | 0.40 | −0.33 | 🔴 |
| C1 | coverage_funnel_top20 (raw) | 70 % | 90 % | −20 п.п. | 🟡 |
| C1' | coverage_funnel adjusted (holding) | ~80 % | 90 % | −10 п.п. | 🟡 |
| C2 | silent_miss_rate | 7.9 % | 1 % | +7 п.п. | 🔴 |
| C3 | model_recall@top20 | 100 % | ≥95 % | OK | 🟢 |
| **E1** | **time_to_signal** median | **+4.94h late** | ≤+0.5h | −4.4h | 🔴 |
| E1b | TTS p25 / p75 | +0.71 / +10.96h | tight | spread | 🔴 |
| E1c | early entries (lead ≤ 0) | 17 % | ≥50 % | −33 п.п. | 🔴 |
| **E2** | **capture_ratio mean (top-20)** | **0.052** | 0.50 | −0.45 | 🔴 |
| E2b | capture median (top-20) | 0.002 | ≥0.20 | −0.20 | 🔴 |
| E3 | premove_lead_time_sim | +4 h median | ≥2h live | TBD | ⚫ TBD |
| Q1 | fast_reversal_rate (overall) | 15 % | ≤8 % | +7 п.п. | 🟡 |
| Q1a | FR rate alignment/1h | 31 % | ≤12 % | +19 п.п. | 🔴 |
| Q1b | FR rate impulse/1h | 20 % | ≤12 % | +8 п.п. | 🟡 |
| Q1c | FR rate breakout/15m | 23 % | ≤12 % | +11 п.п. | 🔴 |
| Q1d | FR rate strong_trend/15m | 0 % | — | OK | 🟢 |
| Q2 | whipsaw_rate (overall) | 12.5 % | ≤5 % | +7.5 п.п. | 🔴 |
| Q2a | whipsaw breakout/15m | 41 % | ≤10 % | +31 п.п. | 🔴 |
| Q2b | whipsaw impulse_speed/15m | 9.7 % | ≤6 % | +3.7 п.п. | 🟡 |
| Q3 | fast_reversal_drag | 28 % | ≤12 % | +16 п.п. | 🔴 |
| Q4 | actionable_signal_rate | 85 % | ≥92 % | −7 п.п. | 🟡 |
| **D1** | **signal_precision_top20** | **9 %** | ≥35 % | −26 п.п. | 🔴 |
| D2 | tg_message_rate (raw/d) | 41.5 | 8–12 | +30 | 🔴 |
| D2b | tg_msg_rate (uniq syms/d) | 34.1 | 8–12 | +22 | 🔴 |
| D3 | bandit_ucb_separation | 0.154 | ≥0.20 | −0.046 | 🟡 |

Цвет: 🟢 OK · 🟡 нужно улучшить · 🔴 критический отстающий · ⚫ TBD

---

## Главные insights

### 1. Capture — главная пропасть

**E2 = 0.052, median 0.002.** Когда мы заходим в top-20 winner, мы ловим
**0.2 % дневного движа в среднем (медианно)**. Top-1 победитель `BLURUSDT
04-19`: capture = 1.00 (поймали полностью). Топ-5 имеют capture > 0.4.
Но 50 % entries имеют capture ≤ 0.002 — практически точка-в-точку
со входом «на хаях».

**Корень:** entry часто срабатывает, когда движение уже закончилось
(daily_range > 5 % к моменту нашего сигнала). Это согласуется с E1.

### 2. TTS — мы хронически опаздываем

**E1 = +4.94h late.** Median — почти 5 часов после того, как coin прошёл
2 % с открытия. **80 % entries** входят более чем через 30 минут после
2 %-движа. Только 17 % сигналов — **до** этого порога.

Это объясняет E2: к моменту входа значительная часть движа сделана.

### 3. Spam в TG

**D2 = 41.5 entries/day raw, 34 уникальных символа/день.** При target
8–12. Канал заваливает читателя сигналами, **из них только 9 % на
top-20 winners** (D1 = 9 %).

Подписчик не может отфильтровать — все BUY выглядят одинаково.

### 4. Whipsaw breakout/15m — катастрофа

**Q2a = 41 %.** Каждая 2.5-я сделка `breakout/15m` — это whipsaw
(trail-stop в ≤5 баров с loss до −1.5 %). Подтверждает спеку
`breakout-15m-disable-spec.md` — режим нужно выключить.

### 5. Coverage не такой плохой, как казалось

**C1' ≈ 80 %** с поправкой на «бот держит позицию». Real silent miss
~ 2 в неделю. Не главная боль.

---

## Влияние Phase 1 (instrumentation) на KPI

Прямого влияния на бизнес-KPI **нет** — это чистая измерительная фаза.
Никаких изменений в `bot.py` / `monitor.py` / `strategy.py`. Бот ведёт
себя как до инструментации.

**Что изменилось:**

| Дальше | До инструментации | После |
|--------|------|------|
| Главная цель | recall@top20 / AUC (proxy) | EarlyCapture@top20 (direct) |
| Видимость earliness | не измерялась | E1, E2, E3 в JSONL |
| Видимость quality | partial (avg_pnl) | Q1–Q4 per mode |
| Видимость TG-spam | не измерялась | D1, D2 daily |
| Daily snapshot | recall + AUC | 13 метрик в одной строке |
| Anti-metrics | implicit | explicit, в каждой rollout-спеке |

**Главное открытие:** мы оцениваем себя по recall@20 = 100 % и
AUC = 0.989, **но реальный north-star = 0.074**. Разрыв возник потому,
что recall меряет «бандит видит выигравших coins», а не «бот выдаёт
полезные сигналы вовремя». Без этой инструментации никакая дальнейшая
оптимизация не имела бы guarantee, что движется в правильную сторону.

---

## Re-prioritisation после baseline

Validated roadmap из `2026-04-28-improvements-validated.md` уже
правильно расставлен — но baseline уточняет **относительные** размеры
выигрышей:

| Initiative | Driver | Ожидаемый Δ NS | Confidence |
|-----------|--------|--------------:|:-----------|
| **P5 ML blind-spot** | C1 +5–10 п.п. | +0.05 | high |
| **P3-redesign screener** | E1 −2..3h | **+0.10** | medium |
| **P2 anti-FR** | Q1 −7 п.п. | +0.02 | high (UX) |
| **P7 disable breakout/15m** | Q2 −2 п.п., D2 −2/d | +0.01 | high |
| **D1/D2 prune (новая)** | precision +10 п.п. | +0.03 | medium |

**Новый кандидат: D1/D2 pruning.** При 41 entries/day и 9 % precision,
агрессивное ужесточение гардов (например, `ranker_ev > 0` обязательно)
может повысить precision до 25 % при потере 30 % entries — net win.
Достоен спеки.

---

## Operational changes

- ✅ `report_metrics_daily.py` запускается успешно. Завести scheduled
  task `CryptoBot_MetricsDailyReport` cron 03:50 local (после
  health-alert).
- ✅ `.runtime/metrics_daily.jsonl` — append-only history. Всё для
  трендов на N дней.
- 🔲 Расширение в `daily_learning.py`: интегрировать вызов в конец
  EOD-цикла (вместо отдельной задачи) для гарантии запуска после
  retrain’а.
- 🔲 TG-digest: компактная сводка top-3 changed metrics (>1 σ от 7 d
  rolling mean).
