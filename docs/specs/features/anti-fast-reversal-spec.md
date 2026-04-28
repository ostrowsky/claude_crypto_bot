# Anti-fast-reversal label, model, guard, reward

- **Slug:** `anti-fast-reversal`
- **Status:** draft (pieces planned per `CLAUDE.md` §4a)
- **Created:** 2026-04-26
- **Owner:** core
- **Related:** `CLAUDE.md` §4a, `files/critic_dataset.jsonl`,
  `files/top_gainer_dataset.jsonl`, `files/contextual_bandit.py`,
  `files/monitor.py`

---

## 1. Problem

Сейчас 53.7 % `alignment` сетапов закрываются в течение 3 баров с
avg P&L = −0.348 %. Это **нечитаемо** для подписчика TG-канала: BUY
прилетел и через 30–45 минут позиция уже минусит. Пользователь
вынужден игнорировать такие сигналы, обесценивая весь канал.

## 2. Success metric

**Primary:** доля сигналов с `pnl < 0 в первые 3 бара` падает с 53.7 %
до **≤ 30 %** на `alignment` и до ≤ 25 % overall.

**Secondary:** `recall@top20 ≥` текущего уровня (100 %), allow тоже не
просесть. Если recall упадёт > 5 п.п. — порог гард-блока ослабить.

## 3. Scope

### In scope
1. **Label** `label_fast_reversal ∈ {0,1}` в `critic_dataset.jsonl` и
   `top_gainer_dataset.jsonl`.
2. **Model output** `proba_fast_reversal ∈ [0,1]` (отдельная голова или
   новый CatBoost-классификатор).
3. **Guard** `_fast_reversal_guard_reason()` в `monitor.py`. Reason
   code: `fast_reversal_risk`.
4. **Bandit context** — добавить `proba_fast_reversal` в LinUCB context.
5. **Reward** обновление (см. таблицу).

### Out of scope
- Изменение `MAX_HOLD_BARS` или fast-loss-exit логики.
- Trail-stop изменения (см. `trail-min-buffer-spec`).

## 4. Behaviour / design

### Label rule

Для каждой записи (entry candidate) считаем:
```
trail_buffer = ATR_pct × trail_k
label_fast_reversal = 1, если за следующие 3 бара после entry
   low(t..t+3) ≤ entry × (1 − trail_buffer)
иначе 0
```

(В терминах buffer’а трейла — попадание в стоп в первые 3 бара.)

### Guard

```python
if proba_fast_reversal > FAST_REVERSAL_PROBA_MAX:
    return BlockReason("fast_reversal_risk", proba=proba_fast_reversal)
```

Default: `FAST_REVERSAL_PROBA_MAX = 0.55` (settable per-mode при необходимости).

### Reward

| Ситуация | Reward |
|---|---|
| ENTER + fast reversal (≤3 баров, P&L < −0.5 %) | **−0.6** |
| SKIP + fast reversal | **+0.30** |

(добавляются к существующей таблице из `contextual-bandit-spec`).

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `FAST_REVERSAL_GUARD_ENABLED` | False (initial) | Master switch |
| `FAST_REVERSAL_PROBA_MAX` | 0.55 | Порог блокировки |

Rollout: сначала `False` (только label/model/feature/reward, без блока).
После 7 дней live-данных — bandit-context-only. После 14 дней с
подтверждением `recall@top20` — включить `_GUARD_ENABLED`.

## 6. Risks

- **Over-blocking TG-кандидатов:** все «быстрые» движения up могут
  иметь короткий фейк-pullback в первые 3 бара. Mitigation: per-mode
  threshold, мягкий порог `0.55`, поэтапный rollout.
- **Calibration drift модели** при добавлении новой головы.

## 7. Verification

Implementation order (строго):

- [ ] **Label** в обоих dataset’ах.
- [ ] **Train** модель + AUC отчёт.
- [ ] **60-day backtest** — посчитать recall@top20 при разных порогах
      `FAST_REVERSAL_PROBA_MAX`, выбрать Pareto-точку.
- [ ] **Wire bandit context** (без блока).
- [ ] Через 7 дней — включить guard на `False → True`.

**Гард не включается, пока 60-day backtest не подтверждает
`recall@top20` ≥ текущего.**

## 8. Follow-ups

- Per-mode пороги (`FAST_REVERSAL_PROBA_MAX_ALIGNMENT`, и т.д.).
- Связка с `trail-min-buffer`: если floor увеличил buffer, метрика
  fast-reversal должна сама ослабиться → пересмотреть label через
  realised buffer.
