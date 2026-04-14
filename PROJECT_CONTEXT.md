# Claude Crypto Bot — Project Context

## Назначение

Telegram-бот для сканирования Binance Futures/Spot в режиме реального времени и генерации ранних сигналов BUY на монеты, которые к концу дня окажутся в топе по росту (top gainers).

**Главная метрика успеха:** максимально ранний сигнал BUY на монеты из топ-20 по суточному росту (как можно раньше в течение дня, до основного движения).

**Проект:** `D:\Projects\claude_crypto_bot\`  
**Второй бот (НЕ ТРОГАТЬ):** `D:\Projects\gpt_crypto_bot\` — работает независимо на другом Telegram токене.

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

### Метрики прогресса обучения (по состоянию на 2026-04-13)

| Дата | Recall@20 | UCB Sep | Updates |
|------|-----------|---------|---------|
| Apr 07 | 99.0% | +0.047 | 86,751 |
| Apr 13 | 100.0% | **+0.112** | 362,594 |

UCB separation растёт (+138% за неделю) — бандит учится различать winners.

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
