# Daily learning pipeline (EOD orchestrator)

- **Slug:** `daily-learning-pipeline`
- **Status:** shipped (retroactive)
- **Created:** 2026-04-26
- **Owner:** core
- **Related:** `files/daily_learning.py`, `files/rl_headless_worker.py`,
  `files/report_rl_daily.py`, `CLAUDE.md` §5

---

## 1. Problem

Несколько ML-компонентов (signal, ranker, top-gainer, bandit) обучаются на
разных датасетах с разной частотой. Нужен единый orchestrator, чтобы:
(а) ни одно EOD-обучение не пропускалось, (б) метрики отчётности
агрегировались в один Telegram-отчёт.

## 2. Success metric

- 100 % EOD-задач (snapshot → resolve → train → report) выполняются за окно.
- Daily-отчёт публикуется не позже 03:00 local.

## 3. Scope

### In scope
- Scheduled tasks (Windows): EOD + intraday snapshots.
- Pipeline шагов в `daily_learning.py`.
- Headless RL worker (фоновое обучение бандита).

### Out of scope
- Сам ML training (см. соответствующие специнтерны).

## 4. Behaviour / design

### Scheduled tasks

| Task | Time (local) | Action |
|------|--------------|--------|
| `CryptoBot_DailyLearning_EOD` | 02:30 | Полный цикл |
| `CryptoBot_IntradaySnapshot` | 08:30 / 14:30 / 20:30 | Snapshot фичей в `top_gainer_dataset` |

### Полный EOD-цикл (`daily_learning.py`)

1. **Snapshot resolve** — для каждой записи в `top_gainer_dataset.jsonl`
   c `resolved=False` и daily-bar закрыт → проставить top-N labels.
2. **Bandit retrain** — `offline_rl.train_entry_bandit` поверх
   `top_gainer_dataset` + `critic_dataset` (cap 25 000).
3. **Top-gainer model retrain** — `train_top_gainer.py` →
   `top_gainer_model.json`.
4. **Ranker retrain** — `ml_candidate_ranker.json` (+ shadow A/B).
5. **Signal model retrain** — `ml_signal_model.json`.
6. **Report** — `report_rl_daily.py` отправляет отчёт в Telegram.

Каждый шаг идемпотентен: если упал в середине, повторный запуск
дочиновляет начатое.

### Headless RL worker

`rl_headless_worker.py` крутится фоном, раз в N минут вызывает
`offline_rl.train_entry_bandit` поверх свежих записей. PID в
`.runtime/rl_worker_bg.json`.

### Progress log

`.runtime/learning_progress.jsonl` — append-only история метрик
(recall@20, UCB sep, AUC, n_updates).

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `RL_TRAIN_TELEGRAM_REPORTS_ENABLED` | True | toggle TG отчёт от RL train |
| `BANDIT_CRITIC_MAX_RECORDS` | 25_000 | Cap critic-датасета |

Откат: остановить scheduler-задачу + `stop_rl_headless.bat`.

## 6. Risks

- **Snapshot miss** при сбое scheduler → recall просядет на 1 день.
- **Token conflict:** только один main-bot одновременно (Telegram dedup —
  см. CLAUDE.md §12).

## 7. Verification

- `files/report_rl_daily.py` — daily metrics report.
- `.runtime/learning_progress.jsonl` — history check.
- См. `CLAUDE.md` §4 «Learning progress».

## 8. Follow-ups

- Алерт в TG, если EOD-цикл не завершился к 03:30.
- Перенести scheduler с Windows-task в systemd-like supervisor (если
  будет миграция на Linux).
