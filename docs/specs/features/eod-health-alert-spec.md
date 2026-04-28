# EOD health alert

- **Slug:** `eod-health-alert`
- **Status:** draft
- **Created:** 2026-04-26
- **Owner:** core
- **Related:** `files/daily_learning.py`, `.runtime/learning_progress.jsonl`,
  `docs/reports/2026-04-28-learning-and-scout.md` §1

---

## 1. Problem

EOD-цикл (snapshot → resolve → train → report) может упасть тихо.
2026-04-25: `n_collected=0` в `learning_progress.jsonl` — пропущенный
intraday snapshot. Замечено только при ручном анализе на следующий день.

При повторных пропусках recall@top20 просядет, а команда узнает
с задержкой.

## 2. Success metric

**Primary:** любая аномалия EOD-цикла (`n_collected=0`, `model_status
!= "ok"`, `bandit_total_updates` не растёт) → Telegram-сообщение в
течение 1 ч после запланированного времени (03:30 local).

**Secondary:** false-positive алертов ≤ 1 / 30 d.

## 3. Scope

### In scope
- Чтение `learning_progress.jsonl` после EOD-цикла.
- Проверка простых invariants (см. §4).
- Telegram-сообщение в admin-чат.

### Out of scope
- Auto-recovery (повторный запуск задачи). Только notification.
- Health checks других подсистем (live monitor / RL worker) —
  отдельные специфичные алерты при необходимости.

## 4. Behaviour / design

### Triggers

После завершения `daily_learning.py` или по cron 03:45 local
(15 мин после планового завершения), скрипт `health_check_eod.py`:

1. Прочитать **последнюю запись** `learning_progress.jsonl`.
2. Проверить:

```python
def is_healthy(rec, prev):
    if not rec: return False, "no record"
    if rec["n_collected"] == 0:
        return False, "n_collected = 0 (snapshot miss)"
    if rec["model_status"] != "ok":
        return False, f"model_status = {rec['model_status']}"
    if prev and rec["bandit_total_updates"] <= prev["bandit_total_updates"]:
        return False, "bandit updates stalled"
    if rec["model_auc_top20"] is not None and rec["model_auc_top20"] < 0.85:
        return False, f"AUC drop: {rec['model_auc_top20']:.3f} < 0.85"
    return True, "ok"
```

3. Если unhealthy — отправить в TG:
   ```
   ⚠️ EOD health check FAILED [date]
   reason: <reason>
   last record: <ts, summary>
   ```

4. Если healthy — silent pass (никакого спама).

### Run

- Scheduled task `CryptoBot_EOD_HealthCheck` cron 03:45 local
  (через 15 минут после `CryptoBot_DailyLearning_EOD`).
- Single-run, не daemon.

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `EOD_HEALTH_ALERT_ENABLED` | True | Master switch |
| `EOD_HEALTH_ALERT_AUC_FLOOR` | 0.85 | AUC threshold для alarm |

**Rollback:** flag=False или удалить scheduled task.

## 6. Risks

- **Spam при temp issue.** Mitigation: dedup по дате (один alert на
  уникальный `<date, reason>`), интегрировать с
  `.runtime/tg_send_dedup.json`.
- **False positive если model_auc_top20 = None** (рандомный edge
  case в `daily_learning`). Mitigation: проверка `is not None`
  перед сравнением.
- **Cron не запустился.** Mitigation: уже out of scope (это
  meta-проблема — нужен внешний watchdog).

## 7. Verification

- [ ] **Unit тест:** `health_check_eod.py` на synthetic record’ах
      (healthy, n=0, stalled, AUC drop).
- [ ] **Manual trigger** на 2026-04-25 record (n_collected=0)
      → alert fired ✓.
- [ ] **Manual trigger** на 2026-04-28 record (healthy)
      → silent pass ✓.
- [ ] **7 d live**: ровно один alarm за пропущенный день, без
      false positives.

## 8. Follow-ups

- Расширить на live monitor: heartbeat файл `.runtime/bot_bg.json`
  старше 5 мин → alert.
- Расширить на RL worker: `.runtime/rl_worker_status.json`.
- Web-dashboard вместо TG (если алертов станет много).
