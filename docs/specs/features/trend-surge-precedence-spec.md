# Trend-surge precedence — H3

- **Slug:** `trend-surge-precedence`
- **Status:** draft → implementing (behind flag, default OFF)
- **Created:** 2026-05-02
- **Owner:** core
- **Related:** `2026-05-01-mode-audit.md` §H3 (mode-audit found
  `trend_surge` is dead-code in main pipeline).

---

## 1. Problem

`check_trend_surge_conditions` написан и валидирован
(`strategy.py:1481`), но в основном entry-pipeline он вызывается
**после** `entry_ok`:

```python
elif entry_ok:    # mode = trend|strong_trend|impulse_speed
elif surge_ok:    # ← запускается только если entry_ok=False
```

Это значит: каждый раз, когда coin удовлетворяет **обоим** условиям
(а surge почти всегда подразумевает entry_ok), surge никогда не
выигрывает precedence. Detector «начало устойчивого тренда»
фактически dead-code в основном цикле — работает только в
catchup-сканере / backfill / data_collector.

Surge имеет более раннюю и точную сигнатуру тренда (slope
ускоряется vs 3 бара назад, MACD растёт 2 бара подряд) — то,
что бот должен сигналить **в первую очередь**.

## 2. Success metric

**Primary:** доля entries с label `trend_surge` среди всех entries
≥ 5 % (сейчас < 1 % — surge срабатывает только когда `entry_ok`
fails).

**Secondary (anti-metrics):**
- recall@top20 не падает > 2 п.п.
- avg_pnl на reclassified entries (бывший `trend` → `trend_surge`)
  ≥ avg_pnl на оригинальных trend entries.
- TG msg_rate D2 не растёт.

**Acceptance после 7 d shadow:**
- ≥ 5 reclassifications зарегистрированы.
- Capture_ratio на surge-classified ≥ 0.20 (vs 0.16 baseline).

## 3. Scope

### In scope
- Config-flag `TREND_SURGE_PRECEDENCE_ENABLED` (default `False`).
- Когда `True`: `surge_ok` идёт **перед** `entry_ok` в pipeline.
- Новый sig_mode `trend_surge` (отдельный лейбл, не `impulse_speed`).
- Trail/max_hold для нового режима: используем `ATR_TRAIL_K_STRONG`
  и `MAX_HOLD_BARS_15M/1H` (как у impulse_speed/strong_trend).

### Out of scope
- Cooldown между surge-сигналами (уже есть `SURGE_COOLDOWN_BARS=20`).
- Изменения внутри `check_trend_surge_conditions` (логика
  не меняется).
- Полная rearchitecture mode-precedence (это H1, отдельная спека).

## 4. Behaviour / design

### Текущее (without flag)
```
elif brk_ok:    sig_mode = "breakout"
elif ret_ok:    sig_mode = "retest"
elif entry_ok:  sig_mode = trend|strong_trend|impulse_speed
elif surge_ok:  sig_mode = "impulse_speed"   ← только если entry_ok=False
elif imp_ok:    sig_mode = "impulse"
elif aln_ok:    sig_mode = "alignment"
```

### С флагом (TREND_SURGE_PRECEDENCE_ENABLED=True)
```
elif brk_ok:    sig_mode = "breakout"
elif ret_ok:    sig_mode = "retest"
elif surge_ok:  sig_mode = "trend_surge"      ← поднят
elif entry_ok:  sig_mode = trend|strong_trend|impulse_speed
elif imp_ok:    sig_mode = "impulse"
elif aln_ok:    sig_mode = "alignment"
```

Дополнительно: при `sig_mode == "trend_surge"` логировать
event-key `surge_won_over_entry_ok=True` (если entry_ok тоже =True),
чтобы постфактум считать частоту reclassifications.

### Trail / max_hold

```python
elif surge_ok and config.TREND_SURGE_PRECEDENCE_ENABLED:
    sig_mode = "trend_surge"
    trail_k = getattr(config, "ATR_TRAIL_K_TREND_SURGE",
                      getattr(config, "ATR_TRAIL_K_STRONG", 2.5))
    max_hold = (getattr(config, "MAX_HOLD_BARS_15M", 48)
                if tf == "15m" else getattr(config, "MAX_HOLD_BARS", 16))
```

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `TREND_SURGE_PRECEDENCE_ENABLED` | False | Master switch (default OFF — постепенный rollout) |
| `ATR_TRAIL_K_TREND_SURGE` | 2.5 (= STRONG) | Trail для surge-режима |

**Rollback:** flip flag → False. Никакой schema-migration.

## 6. Risks

- **Подмена метки** — раньше эти entries шли как `trend` /
  `impulse_speed`, теперь часть пойдёт как `trend_surge`. Bandit
  context-vector видит `mode` как onehot — для нового лейбла
  потребуется retrain (но bandit обучается ежедневно, само
  адаптируется).
- **Surge-cooldown** (`SURGE_COOLDOWN_BARS=20`) сейчас применяется
  только в impulse_scanner. Если он не применяется в monitor.py,
  один coin может выдать surge-сигнал каждый бар. Mitigation:
  cross-check с monitor.py code путём grep на `SURGE_COOLDOWN`
  (TODO before flag flip).
- **TG label change** — пользователь увидит `trend_surge` вместо
  `📈 Тренд` или `⚡ Быстрое движение`. Mitigation: добавить
  emoji+label в `_mode_labels` в `monitor.py`.

## 7. Verification

- [x] **Static check** — surge_ok сейчас в самом конце if-elif.
- [ ] **Smoke test** — flip flag в dev, открыть 1 entry на coin
      с известно-surge сигнатурой, увидеть `trend_surge` в логах.
- [ ] **7 d shadow** — flag=True, мониторить:
  - кол-во reclassifications;
  - precision на surge-classified;
  - capture_ratio surge vs baseline trend.
- [ ] **Rollback trigger:** recall@top20 падает > 2 п.п. за 3 дня
  → flip back, postmortem.

## 8. Follow-ups

- H1 (full mode-precedence rewrite) — H3 это локальная правка,
  H1 будет глобальным refactor’ом.
- Добавить `surge_ok_won_over_entry_ok` поле в entry event для
  post-hoc анализа.
- После 7 d с включённым flag — re-run mode-audit, обновить
  conflict map.
