# Live-демо сценарий — точные команды + что говорить

> Если попросят "покажи как это работает" — следовать строго по
> этому сценарию. Если попросят чуть углубиться — открыть код.

## Подготовка ДО демо (за 5 мин до Zoom)

```powershell
# Открыть в Windows Terminal 3 вкладки:

# Вкладка 1: команды pipeline
cd D:\Projects\claude_crypto_bot

# Вкладка 2: чтение артефактов
cd D:\Projects\claude_crypto_bot\.runtime\pipeline\health

# Вкладка 3: код pipeline
cd D:\Projects\claude_crypto_bot\files
```

Открыть в редакторе/preview:
- `.runtime/pipeline/health/health-2026-05-11.md` — главный артефакт
- `files/pipeline_blind_critic.py` — на случай вопроса про L5
- `.runtime/pipeline/decisions/decisions.jsonl` — память
- `.runtime/pipeline/do_not_touch.json` — защищённые компоненты

---

## Демо-блок 1 · "Покажу что pipeline видит сейчас" (1 мин)

```powershell
pyembed\python.exe files\pipeline_run.py --daily
```

**Что говорить пока идёт (~6 секунд):**
> "Сейчас запустится весь daily-pipeline — L1 собирает метрики,
> L4 читает shadow-события, L5 пытается дать independent verdict,
> L7 проверяет outcome старых решений."

**После завершения:**
```powershell
# Показать тот же отчёт, что был сгенерирован
type .runtime\pipeline\health\health-2026-05-11.md | more
```

**Указать на:**
- Раздел "North-Star: early_capture_rate" — 6.7% (critical 🚨)
- Раздел "Training-to-Live Gap" — +66.7% (главный сигнал)
- Раздел "Red Flags" — 6 штук с конкретными hypotheses

> "Видите: тренировочный recall@20 = 100% (зелёный), а реальный
> early capture = 6.7% (красный). Без явного gap'а это два
> разных отчёта в Telegram, которые противоречат друг другу,
> и никто не сводит."

---

## Демо-блок 2 · "Как pipeline решает что менять" (1 мин)

```powershell
# Гипотезы уже сгенерированы L2. Показать одну с реальным verdict
type .runtime\pipeline\hypotheses\h-2026-05-11-entry_score_floor_relax.json | more
```

**Указать на:**
- `status: rejected`
- `validation_report.result.verdict: reject`
- `validation_report.result.reason: best floor adds +96.2% FP rate (limit +5%)`
- `n_blocked_events_in_window: 840`

**Что говорить:**
> "L2 предложил: понизить порог фильтра. L3-валидатор сам прогнал
> 840 заблокированных событий за 60 дней и обнаружил: если применить
> это, 96% разблокированных будут шумом. Verdict: **reject**, автоматически.
> Без L3 это попало бы в прод."

---

## Демо-блок 3 · "Pipeline проверяет сам себя" (1 мин)

```powershell
pyembed\python.exe files\pipeline_stress_test.py
```

**Что говорить пока идёт (~5 секунд):**
> "Сейчас pipeline сам подсунет себе 3 заведомо плохие гипотезы
> и проверит, что отклонит их. Без этого теста — мы не знаем,
> работает ли защита от регрессий."

**После завершения:**
> "3/3 pass. В первый прогон было 2/3 — тест поймал реальный баг
> в L6, где флаг `--approve` обходил safety-checks. Исправил, теперь
> зелёные. Это пример того, **почему pipeline должен ловить свои же баги**."

---

## Демо-блок 4 (опционально) · "Как L5 не может обмануть" (1 мин)

```powershell
code files\pipeline_blind_critic.py  # или type
```

**Показать функцию `critique()`** (около строки 100).

**Что говорить:**
> "Смотрите: критик принимает только `decision_id`. Внутри он читает
> ТОЛЬКО `_extract` из health-reports — числа до и после.
> Он физически не загружает hypothesis-файл с reasoning.
> Это не дисциплина — это архитектурное ограничение."

---

## Демо-блок 5 (опционально) · "Память pipeline" (30 сек)

```powershell
type .runtime\pipeline\decisions\decisions.jsonl
type .runtime\pipeline\do_not_touch.json
```

**Что говорить:**
> "decisions.jsonl — лог всех решений. Append-only, никогда не редактируется.
> already_tried.jsonl читает L2, чтобы не предлагать то же дважды.
> do_not_touch.json — список компонентов с доказанной работой, которые
> pipeline не имеет права трогать. Защита от 'я опять сломаю работающее'."

---

## Если демо урезали до 3 минут — только это:

```powershell
pyembed\python.exe files\pipeline_run.py --daily
# 6 секунд

type .runtime\pipeline\health\health-2026-05-11.md
# Показать gap +66.7%

pyembed\python.exe files\pipeline_stress_test.py
# 5 секунд, 3/3 pass
```

## Что говорить если показ зависнет

> "Извините, дайте секунду пока перезапущу — pipeline-логи я обычно
> вижу в реальном времени, дайте откатиться к скриншоту из утреннего
> прогона."

→ Открыть скриншот health-report.md в редакторе.

## Что говорить если спросят "а где модель/бот сам?"

> "Бот это отдельный продукт — он работает 24/7. Я сегодня показываю
> не его, а **harness вокруг него**. Бот выдаёт сигналы и данные,
> а pipeline честно оценивает: делает ли бот то, для чего был построен."

## Что говорить про "крипту" если занудятся

> "Крипто-домен здесь только источник данных и боль. Pipeline и его
> архитектура не специфичны для крипты — это паттерн, применимый
> к любой системе, где training metric ≠ user-facing outcome.
> Я хотел бы перенести этот подход на ваши задачи в BOS.PRO."
