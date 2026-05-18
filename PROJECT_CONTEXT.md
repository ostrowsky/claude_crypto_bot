# Claude Crypto Bot — Project Context

## Назначение

Telegram-бот для сканирования Binance Futures/Spot в режиме реального времени и генерации ранних сигналов BUY на монеты, которые к концу дня окажутся в топе по росту (top gainers).

**Главная метрика успеха (North Star):** максимально ранний сигнал BUY на монеты из топ-20 по суточному росту. Операционализирована как `watchlist_top_early_capture_pct` — доля топ-20 в watchlist'е, на которые бот выдал BUY с `capture_ratio ≥ 0.35` (до 35% от дневного движения).

**Проект:** `D:\Projects\claude_crypto_bot\`  
**Второй бот (НЕ ТРОГАТЬ):** `D:\Projects\gpt_crypto_bot\` — работает независимо на другом Telegram токене.

---

## P0 — Непрерывное обучение алгоритмов

**Торговые алгоритмы бота должны непрерывно обучаться на необходимых данных для улучшения метрик назначения (North Star). Если данных для этого обучения не хватает — их сбор является частью работы, а не отдельной задачей.**

Непрерывное совершенствование алгоритмов — **задача первого приоритета** проекта. Любые последующие изменения (config-параметры, рефакторинг, новые фичи) **должны валидироваться на сонаправленность с этим приоритетом до мёрджа**.

| Вопрос перед изменением | Вердикт |
|-------------------------|---------|
| Улучшает ли это способность бота учиться на данных? | ✅ предпочтительно |
| Оставляет ли loop обучения без изменений? | 🟡 допустимо |
| Уменьшает / замедляет / фрагментирует обучение? | 🚨 нужно обоснование или рефакторинг |

Практические следствия:

1. **Изменение, заблокированное отсутствием логирования, — это в первую очередь задача о добавлении логирования.** Не спекулируем по неполному state'у — инструментируем бота.
2. **Обход pipeline (ручные правки `config.py` без approved hypothesis, переобучение модели вне pipeline, ad-hoc tuning фильтров) ослабляет loop.** Делать только с документированным обоснованием.
3. **Разрыв между training-метриками и deployment-метриками — самый важный диагностический сигнал.** Не сводим в одну цифру, не прячем.
4. **Память обязательна.** Что пробовали, что approved, что rolled back — пишется в `decisions.jsonl` и `already_tried.jsonl`. Пропуск записи — регрессия.

Полное состояние auto-improvement loop (что реализовано, что нет, метрики North Star за всю историю) — в [`docs/specs/features/auto-improvement-loop-spec.md`](docs/specs/features/auto-improvement-loop-spec.md). **Это живой документ.** Каждый PR, затрагивающий компонент loop'а, должен обновить строку статуса и таблицу North Star progress.

---

## Архитектура

### Слои сигналов (pipeline)

```
Binance API → Indicators → Strategy (7 entry modes)
    → ML gating (CatBoost)
    → Ranker (candidate score)
    → Contextual Bandit (enter/skip gate)
    → Guards (trend quality, impulse speed, ranker veto)
    → Entry signal → Telegram
```

### 7 режимов входа (entry modes)
- `trend` — 15m тренд
- `strong_trend` — 1h сильный тренд
- `retest` — 1h ретест уровня
- `alignment` — 1h мультитаймфреймное выравнивание
- `impulse` — 15m импульс
- `impulse_speed` — 1h импульс скорости
- `breakout` — пробой

### Ключевые файлы

| Файл | Назначение |
|------|-----------|
| `files/bot.py` | Telegram UI, точка входа |
| `files/monitor.py` | Основной цикл мониторинга (live loop) |
| `files/strategy.py` | Логика сигналов, все 7 режимов входа |
| `files/config.py` | 300+ параметров конфигурации |
| `files/contextual_bandit.py` | LinUCB бандит (enter/skip + trail_k) |
| `files/offline_rl.py` | Offline обучение бандита |
| `files/daily_learning.py` | Оркестратор ежедневного обучения |
| `files/rl_headless_worker.py` | Фоновый RL worker (обучение ранкера) |
| `files/ml_candidate_ranker.py` | CatBoost ранкер кандидатов |
| `files/ml_signal_model.py` | CatBoost классификатор сигналов |
| `files/top_gainer_critic.py` | Оценка качества предсказания top gainers |
| `files/train_top_gainer.py` | Обучение top_gainer_model |
| `files/indicators.py` | Технические индикаторы |
| `files/report_rl_daily.py` | Ежедневный отчёт RL + bandit метрики |

### Runtime файлы

| Файл | Назначение |
|------|-----------|
| `.runtime/bot_bg.json` | PID бота |
| `.runtime/rl_worker_bg.json` | PID RL worker |
| `.runtime/tg_send_dedup.json` | Дедупликация Telegram отчётов |
| `.runtime/learning_progress.jsonl` | История метрик обучения (daily) |
| `files/top_gainer_dataset.jsonl` | Фичи ВСЕХ монет из watchlist × N снимков в день |
| `files/critic_dataset.jsonl` | Сигналы бота с результатами (source: signal-originated) |
| `files/bot_events.jsonl` | Все события бота (entry, blocked, exit) |

---

## ML / RL система

### Contextual Bandit (LinUCB)

- **Entry Bandit** (2 arms: SKIP=0, ENTER=1) — главный gate
- **Trail Bandit** (5 arms) — выбор trail_k и max_hold после решения ENTER
- Контекст: `slope_pct, adx, rsi, vol_x, ml_proba, daily_range, macd_hist, btc_vs_ema50, is_bull_day, mode, tf`
- Alpha = 2.0 для entry bandit (более агрессивное исследование)

### Награды (reward scheme, asymmetric)

| Ситуация | Награда |
|----------|---------|
| ENTER + top-20 gainer | +1.0 |
| SKIP + top-20 gainer | -0.8 (штраф за пропуск!) |
| SKIP + не top gainer | +0.10 |
| ENTER + не top gainer | -0.12 |

### Источники обучения (train_entry_bandit)

1. **Primary:** `top_gainer_dataset.jsonl` — ВСЕ ~105 монет watchlist × N снимков в день
2. **Secondary:** `critic_dataset.jsonl` — сигналы бота с реальными результатами

### Top Gainer Model (CatBoost)
- Тир-классификаторы: top5 / top10 / top20 / top50
- Переобучается ежедневно в рамках `daily_learning.py`
- Метрики: AUC, recall@0.3 (threshold)

### Метрики прогресса обучения — полная история

Источник: `.runtime/learning_progress.jsonl` (обновляется ежедневно в ~00:30 UTC).  
Каждая строка — результат ночного цикла за предыдущий торговый день.

| Дата | Recall@20 | UCB Sep | AUC top20 | Bandit updates | n_signal |
|------|-----------|---------|-----------|----------------|----------|
| 2026-04-07 | 99.0% | 0.047 | 0.610 | 130,336 | 5,889 |
| 2026-04-08 | 98.5% | 0.017 | 0.642 | 175,383 | 6,017 |
| 2026-04-09 | 100.0% | 0.046 | 0.637 | 223,088 | 6,337 |
| 2026-04-10 | 100.0% | 0.014 | 0.635 | 269,378 | 6,751 |
| 2026-04-11 | 100.0% | 0.091 | 0.635 | 316,635 | 7,052 |
| 2026-04-12 | 100.0% | 0.112 | 0.646 | 362,594 | 7,167 |
| 2026-04-13 | 100.0% | 0.118 | 0.654 | 409,189 | 7,579 |
| 2026-04-14 | 100.0% | 0.115 | 0.682 | 456,228 | 7,800 |
| 2026-04-15 | 100.0% | 0.130 | 0.886 | 503,661 | 8,000 ← cap hit |
| 2026-04-16 | 100.0% | 0.136 | **0.932** | 551,329 | 8,000 ← cap hit |
| 2026-04-17 | 100.0% | 0.144 | 0.978 | 601,326 | 10,040 (cap raised) |
| 2026-04-18 | 100.0% | **0.162** | 0.982 | 651,616 | 10,212 |
| 2026-04-19 | 100.0% | 0.153 | **0.991** | 702,286 | 10,439 |

**Ключевые события:**
- **Apr 10**: Recall@20 = 100% — достигнут и держится с тех пор.
- **Apr 15→16**: AUC top20 скачок 0.68→0.89 (модель резко улучшилась).
- **Apr 15**: `bandit_n_signal` упёрся в cap=8000. Фикс: `BANDIT_CRITIC_MAX_RECORDS=25_000`.
- **Apr 17**: UCB separation ×9 от старта (0.015→0.144), AUC top20=0.97.
- **Apr 19**: AUC top20=0.991 — почти идеальная классификация на обучающей выборке.

### Операционные метрики скаута (precision / recall по top-20 gainers)

Источник: `bot_events.jsonl` (входы) × `top_gainer_dataset.jsonl` (метки top20).  
Скрипт: `files/_diag_precision_recall.py`.  
**Важно:** до апреля знаменатель recall = top-20 внутри watchlist (~20/день); в апреле = глобальный top-20 ∩ watchlist (0–5/день) — методологии несопоставимы напрямую.

| Период | Входов/день | Precision | Recall | Примечание |
|--------|-------------|-----------|--------|------------|
| Март 2026 (avg) | 28–60 | 27–70% | 35–85% | top20 из watchlist-относительного рейтинга |
| 6–13 апр | 1–45 | 0–10% | 0–100% | кризис over-фильтрации + восстановление |
| 14–20 апр | 14–68 | 5–15% | 33–87% | после фиксов |
| **Последние 7д** | — | **6.7%** | **54.3%** | TP=19 из 35 возможностей |
| **Последние 14д** | — | **6.4%** | **41.7%** | TP=25 из 60 |
| **Последние 30д** | — | **20.7%** | **51.0%** | TP=174 из 341 |

**Per-mode precision (последние 14 дней, апрель 2026):**

| Режим | TF | Входов | TP | Precision |
|-------|----|--------|-----|-----------|
| impulse_speed | 15m | 99 | 21 | **21.2%** |
| impulse | 15m | 4 | 1 | 25.0% |
| strong_trend | 15m | 6 | 1 | 16.7% |
| retest | 15m | 32 | 2 | 6.2% |
| trend | 15m | 33 | 2 | 6.1% |
| alignment | 15m | 114 | 4 | 3.5% |
| impulse_speed | 1h | 62 | 1 | **1.6%** ← вето добавлено 2026-04-20 |
| alignment | 1h | 19 | 0 | 0% |
| прочие 1h | — | ~15 | 0 | 0% |

**Актуальные данные (2026-04-24, last 14d):**

| Режим | TF | Входов | TP | Precision |
|-------|----|--------|-----|-----------|
| impulse_speed | 15m | 185 | 45 | **24.3%** |
| impulse | 15m | 10 | 3 | 30.0% |
| impulse | 1h | 5 | 1 | 20.0% |
| impulse_speed | 1h | 90 | 4 | 4.4% |
| retest | 15m | 30 | 2 | 6.7% |
| trend | 15m | 49 | 2 | 4.1% |
| alignment | 15m | 179 | 6 | 3.4% |
| alignment | 1h | 20 | 0 | 0% |
| trend | 1h | 56 | 0 | 0% |
| breakout | 15m | 20 | 0 | 0% |

**Диагноз (2026-04-24):** режимный сдвиг — последние 14 дней тихий рынок (daily_range 3-4%). В 60-дневном окне все режимы работают (precision 14-49%). Главный дискриминатор TP vs FP — `daily_range` (TP avg = 12.5% vs FP avg = 6.5% для impulse_speed/15m).

**Исправление 2026-04-24:** добавлен гейт `_mode_daily_range_guard_reason`:
| Режим/TF | Условие блока | Prec 60d до→после | Прирост |
|-----------|---------------|-------------------|---------|
| alignment/15m | daily_range < 4.0% | 23.0% → 28.9% | +5.9pp |
| trend/15m | daily_range < 4.0% | 29.9% → 47.0% | +17.1pp |
| alignment/1h | daily_range < 5.0% | 30.9% → 38.9% | +8.0pp |
| trend/1h | slope < 0.50% | 14.1% → 18.8% | +4.6pp |
| impulse_speed/1h | daily_range < 7.0% | 16.5% → 19.8% | +3.3pp |

Параметры в config.py: `MODE_RANGE_QUALITY_GUARD_ENABLED`, `ALIGNMENT_15M_RANGE_MIN`, `TREND_15M_RANGE_MIN`, `ALIGNMENT_1H_RANGE_MIN`, `TREND_1H_SLOPE_MIN`, `IMPULSE_SPEED_1H_RANGE_MIN`. Reason code: `mode_range_quality`.

**Выводы по скауту:**
- `impulse_speed 1h` — главный источник мусора (62 входа, 1 TP, 1.6%). Исправлено 2026-04-20.
- `alignment` — много входов, низкая точность (3%). G1+G4 добавлены 2026-04-20.
- Macro-ограничение: в апреле топ-гейнеры уходят за пределы watchlist → потолок recall ограничен рынком.
- `ml_proba` НЕ различает TP от FP (TP proba ≈ FP proba = 0.467 для alignment/15m). Не использовать как основной фильтр.

**Диагностика alignment fast exits (баров ≤ 3):**

| Группа | n | win% | avg P&L | Sharpe |
|--------|---|------|---------|--------|
| fast (≤3 бара) | 127 | 31.5% | -0.348% | -3.56 |
| slow (>3 баров) | 154 | 52.6% | +0.576% | +3.18 |
| **дельта** | | | **+0.924%** | |

Перелом на баре 4. Fast exit share: 45.2% история, **53.7% последние 14 дней** — ухудшается.  
Grace period backtest (2026-04-20): не помогает — проблема в качестве входа, не в стопе.  
Корневая причина части fast exits: **Correlation Guard prune** без min-age (исправлено 2026-04-20).

### Pareto sweep Scout-гейтов (`files/analyze_blocked_gates.py`)

Реальные `take`-входы дают avg_r5 = **-0.016%**, но ряд `blocked`-гейтов имеет положительный avg_r5 — значит они блокируют прибыльные сигналы:

| gate | n | avg_r5 | win% | вердикт |
|------|---|--------|------|---------|
| `impulse_guard` | 550 | **+0.641%** | 40.0 | сильное over-blocking |
| `entry_score` | 1374 | +0.123% | 50.3 | over-blocking (наибольший объём) |
| `ranker_hard_veto` | 787 | +0.128% | 46.3 | всё ещё over-blocking |
| `clone_signal_guard` | 475 | +0.128% | 51.4 | over-blocking |
| `trend_quality` | 433 | +0.127% | 46.7 | лёгкое over-blocking |
| `ml_proba_zone` | 1340 | +0.009% | 44.1 | нейтрально |
| `open_cluster_cap` | 136 | -0.360% | 35.3 | **работает правильно** |
| `mtf` | 227 | -0.208% | 37.0 | **работает правильно** |

Отчёт `report_rl_daily.py` теперь содержит колонку AUC и алерты:
- просадка AUC ≥10% от пика последних 5 дней;
- `bandit_n_signal` стагнирует 3 дня подряд.

---

## Требования к обучению модели — anti-fast-reversal

**Цель (2026-04-25):** модель должна **избегать BUY-сигнала, если есть признаки скорого SELL-сигнала** (быстрый разворот ≤3 баров после входа).

**Мотивация:** 53.7% всех `alignment` входов в последние 14 дней закрываются в течение 3 баров со средним P&L = -0.348%. Каждый такой сигнал — это публикация в Telegram, которую пользователь не успевает обработать осмысленно (см. кейс C98USDT 2026-04-25 15:17 → 16:04, 3 бара, -2.22%).

**Конкретные требования:**
1. **Новая метка `label_fast_reversal`** в обучающем датасете (`critic_dataset.jsonl` + `top_gainer_dataset.jsonl`):
   - `1`, если за следующие 3 бара после входа цена опустилась ниже стоп-уровня (`entry × (1 - ATR_pct × trail_k)`)
   - `0` иначе
2. **Новый таргет для CatBoost-классификатора** или multitask-голова к существующему `ml_signal_model`:
   - выход `proba_fast_reversal ∈ [0,1]`
   - порог-блокировки `FAST_REVERSAL_PROBA_MAX` в `config.py` (default 0.55)
3. **Новый guard в `monitor.py`:** `_fast_reversal_guard_reason(...)` — блок, если `proba_fast_reversal > MAX`. Reason code: `fast_reversal_risk`.
4. **Обновление reward-схемы entry-бандита:**
   - ENTER + fast_reversal (≤3 бара exit с P&L < -0.5%) → reward = **-0.6** (между ENTER+top20 и ENTER+miss)
   - SKIP + fast_reversal → reward = **+0.30** (поощрение пропуска)
5. **Контекстная фича для бандита:** добавить `proba_fast_reversal` в context vector LinUCB.

**Ожидаемый эффект:**
- Снижение fast-exit share с 53.7% → ~30%
- Рост precision (меньше -2% сигналов в Telegram)
- Сохранение recall@top20 (TP-сигналы редко быстро разворачиваются)

**Признаки fast reversal (для feature engineering):**
- RSI > 75 (перекупленность на момент входа)
- price_edge от EMA20 > 2.5% (далеко ушла, риск отката)
- vol_x > 5.0 + ADX < 22 (объёмный спайк без устойчивого тренда)
- Близость к N-баровому high (G4-фича уже добавлена для alignment)
- daily_range < 4% AND ADX > 28 (поздняя стадия движения)

**Реализация:** в очереди после успешного релиза `_mode_daily_range_guard` (2026-04-25). Шаги:
1. Добавить разметку `label_fast_reversal` в `train_top_gainer.py` и резолвер критика.
2. Тренировать отдельную голову или совместную модель.
3. Бэктест на 60-дневном окне (по аналогии с `_backtest_mode_filters.py`) — порог + влияние на recall.
4. Подключить guard и reward-update только после положительного бэктеста.

---

## Telegram UI (главное меню) — спецификация

**Принцип (2026-04-25):** действия, которые пользователь выполняет ЧАСТО, должны быть в **первом ряду** главного меню. Текущая боль: чтобы посмотреть `Позиции [N/M]` нужно несколько нажатий → нужно вынести на первый экран.

**Целевая структура главного меню (3-ряда reply-keyboard):**

```
[ ▶ Анализ + Мониторинг (N монет) ]
[ 🔍 Только анализ ]   [ 📊 Позиции [open/max] ]
[ 📋 Список монет [N] ] [ ⚙ Настройки ]
```

`Позиции [open/max]` — счётчик в самой кнопке (например `[3/10]`), обновляется при каждом входе/выходе.

**Производительность кнопок:**
- Симптом: «кнопки реагируют медленно».
- Корень: `bot.py` обрабатывает callback в одном event-loop с тяжёлыми операциями (загрузка watchlist, чтение JSONL, ML-вызовы).
- Фикс: каждая кнопка должна сразу отдавать `answer_callback_query` (ack ≤300ms), а тяжёлая работа — в `asyncio.create_task` без блокировки.

---

## Расписание задач (Windows Scheduled Tasks)

| Задача | Время | Действие |
|--------|-------|----------|
| `CryptoBot_DailyLearning_EOD` | 02:30 local (00:30 UTC) | Полный цикл обучения: snapshot → resolve → train bandit → retrain model → report |
| `CryptoBot_IntradaySnapshot` | 08:30, 14:30, 20:30 local | Сбор фич для top_gainer_dataset (intraday snapshot) |

---

## Управление процессами

### Запуск / перезапуск
```
restart_bot.bat          — полный перезапуск (бот + RL worker)
start_bot_bg.ps1         — запуск только бота
start_rl_worker_bg.ps1   — запуск только RL worker
```

### Остановка
```
stop_bot_bg.ps1          — остановка бота (по PID из .runtime/bot_bg.json)
stop_rl_headless.bat     — остановка RL worker (по PID из .runtime/rl_worker_bg.json)
```

### Python
- Бот использует **встроенный Python:** `pyembed\python.exe`
- Рабочая директория для бота: `files\`
- Логи: `bot_stdout.log`, `bot_stderr.log` (в корне проекта)

---

## Известные проблемы и фиксы

### Критические баги (исправлены)

| Дата | Проблема | Фикс |
|------|----------|------|
| 2026-04-13 | `is_bull_day_now` использовалась до определения в monitor.py (~3125 vs ~3296) → silently crash каждого цикла, 0 сигналов overnight | Добавлено определение перед вызовом `_impulse_speed_entry_guard` |
| 2026-04-06 | 0 входов несмотря на 12 top gainers (+5-59%) — все заблокированы | См. раздел "Фильтры" ниже |

### Переобученные/перетянутые фильтры (2026-04-06)

Все фильтры были слишком жёсткими и блокировали 100% top gainers:

```python
ML_GENERAL_HARD_BLOCK_MAX = 1.01        # было 0.65 — блокировал высокую уверенность!
ML_GENERAL_HARD_BLOCK_MIN = 0.35        # было 0.55
ML_GENERAL_HARD_BLOCK_BULL_DAY_MIN = 0.28
ML_CANDIDATE_RANKER_HARD_VETO_15M_FINAL_MAX = -2.50   # было -0.75
TREND_15M_QUALITY_RSI_MAX = 72.0        # было 68.0
TREND_15M_QUALITY_RSI_MAX_BULL_DAY = 76.0
TREND_15M_QUALITY_PRICE_EDGE_MAX_PCT = 3.20            # было 2.40
TREND_15M_QUALITY_PRICE_EDGE_MAX_BULL_DAY_PCT = 4.00
```

**Правило:** Если фильтр блокирует >80% eventual top gainers — он сломан. Проверять по `bot_events.jsonl`.

### Фильтры с bull-day relaxation (2026-04-13)

```python
# trend_quality_guard — добавлено:
TREND_15M_QUALITY_DAILY_RANGE_MAX = 10.0           # обычный день
TREND_15M_QUALITY_DAILY_RANGE_MAX_BULL_DAY = 14.0  # bull day (TAO 12% был заблокирован)

# ranker hard veto 1h — добавлено TG condition:
ML_CANDIDATE_RANKER_HARD_VETO_1H_TOP_GAINER_MAX = 0.25  # вето только если И final плохой И TG prob низкий
```

### Trail-stop minimum buffer (2026-04-26)

Whipsaw-выходы на `impulse_speed` / `strong_trend` / `impulse`, когда
бандит выбирал `tight`/`very_tight` arm на волатильных монетах
(ALGOUSDT 04-22 и 04-26: ATR-trail hit на −0.4…−0.6 % за 30–50 мин,
далее цена шла вверх).

Фикс: пол на буфер trail-stop в долях от цены, применяется и при
init_trail, и при per-bar обновлении:

```
buffer = max(trail_k * ATR, min_pct(mode) * price)
trail  = price - buffer
```

```python
TRAIL_MIN_BUFFER_PCT_ENABLED = True
TRAIL_MIN_BUFFER_PCT_IMPULSE_SPEED = 0.015  # 1.5 %
TRAIL_MIN_BUFFER_PCT_STRONG_TREND  = 0.015
TRAIL_MIN_BUFFER_PCT_IMPULSE       = 0.012
# trend / alignment / retest / breakout / default = 0.0
```

Откат: `TRAIL_MIN_BUFFER_PCT_ENABLED = False` + рестарт.
Хелперы в `monitor.py` ~ L1705: `_trail_min_buffer_pct`,
`_compute_trail_buffer`. Полная спецификация:
`docs/specs/features/trail-min-buffer-spec.md`.

### Spec-first workflow (2026-04-26)

Проект перешёл на spec-first подход (вдохновлено
[ostrowsky/spec-first-bootstrap](https://github.com/ostrowsky/spec-first-bootstrap)):

- `AGENTS.md` — гайд процесса для AI-агентов.
- `docs/specs/templates/feature-spec.md` — шаблон для каждой фичи/фикса.
- `docs/specs/features/*.md` — по одной спецификации на изменение.
- `docs/specs/README.md` — индекс.

Любое нетривиальное изменение начинается с копии шаблона в
`docs/specs/features/<slug>-spec.md`, где фиксируются проблема,
scope, success metric, rollback и план верификации. В commit body
ссылка `Spec: docs/specs/features/<slug>-spec.md`.

---

## Дедупликация Telegram отчётов

Файл: `.runtime/tg_send_dedup.json`

| Отчёт | Cooldown | Ключ |
|-------|----------|------|
| RL train complete | 30 мин | `train_session` |
| Top gainer critic midday | 60 мин | `top_gainer_midday_YYYY-MM-DD` |
| Top gainer critic final | 60 мин | `top_gainer_final_YYYY-MM-DD` |

Причина: при нескольких запущенных экземплярах worker'а каждый отправлял свой отчёт независимо.

---

## Текущие блокировки (2026-04-13, исследуется)

| Монета | Причина | Статус |
|--------|---------|--------|
| COTI, RENDER | `BANDIT SKIP ucbs=[1.37, 1.18]` | Корректно — высокий RSI после уже случившегося движения |
| TAO | `trend_quality: daily_range 12.02% > 10.00%` | Исправлено (bull-day порог 14%) |
| XAI | `ranker hard veto: final -1.91 <= -1.50` | Исправлено (добавлен TG condition) |

---

## Конфигурационные флаги (важные)

```python
BANDIT_ENABLED = True                         # LinUCB gate on/off
ML_GENERAL_HARD_BLOCK_MAX = 1.01             # 1.01 = нет верхнего порога (снять верхний кап)
TOP_GAINER_CRITIC_ENABLED = True
RL_TRAIN_TELEGRAM_REPORTS_ENABLED = True
TOP_GAINER_CRITIC_TELEGRAM_REPORTS_ENABLED = True
TREND_15M_QUALITY_GUARD_ENABLED = True
ML_CANDIDATE_RANKER_HARD_VETO_ENABLED = True
```

---

## Watchlist

~105 монет Binance USDT perpetual futures. Файл: `files/watchlist.json`.

---

## Заметки для разработки

- **Не трогать** `D:\Projects\gpt_crypto_bot\` — отдельный бот
- Рабочая директория кода — `files\`, все импорты относительно неё
- Bot token хранится в `.runtime\bot_bg_runner.cmd` (runtime-generated)
- При тестировании скриптов из Python запускать из `files\`, не из корня
- Unicode в логах — cp1251 на Windows, использовать ASCII-only в `.bat`
