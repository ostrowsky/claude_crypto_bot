# Токен Telegram вне логов; переанализ не блокирует главный цикл

- **Slug:** `log-hygiene-scan-thread`
- **Status:** SHIPPED 2026-09-28 (исправления, флаги ON)
- **Truth-harness:** TH-07, TH-12, TH-13
- **Код:** `files/bot.py` (`_RedactSecrets`, `_install_log_hygiene`), `files/strategy.py`
  (`_run_analysis`), `files/config.py` (`HTTPX_LOG_LEVEL`, `ANALYSIS_IN_THREAD`)
- **Тесты:** `files/test_log_hygiene_and_scan_thread.py` (6)
- **Rollback:** `ANALYSIS_IN_THREAD = False` — прежний синхронный анализ;
  `HTTPX_LOG_LEVEL = "INFO"` — прежнее логирование запросов (фильтр токена остаётся).
  Поведение сигналов не меняется: maximum period и shadow не применимы.

## Токен в логах

httpx пишет каждый запрос к Bot API на уровне INFO, а в URL запроса — токен бота;
он оказывался в `bot_stderr.log` и во всех копиях логов в `history/`. Теперь httpx
логирует от WARNING, и на все обработчики корневого логгера поставлен фильтр,
заменяющий токен на `<TELEGRAM_TOKEN>` в сообщении и трассировке. Токен из
существующих логов вычищен; перевыпустить его в @BotFather — действие оператора.

## Подвисания главного цикла

Авто-переанализ (раз в 30 мин) вызывал `analyze_coin` ~200 раз подряд в главном
asyncio-цикле: предупреждения «event loop lag» до 23 с в 16:56, 17:27, 17:57,
18:28 (28.09), в эти секунды опрос монет стоял. `analyze_coin` — чистый расчёт без
общего изменяемого состояния; теперь каждый вызов идёт в `asyncio.to_thread`,
результаты те же (тест сравнивает оба пути).
