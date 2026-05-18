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

- [x] **Label** — code added 2026-05-12 (c233c12) BUT spec definition
      (trail-stop hit, needs `entry_context.atr_pct`/`trail_k`) is
      **structurally impossible on real data**: 0/17437 production
      records have `entry_context`, and the capture code is branch-only.
      Replaced with an outcome-based proxy in `train_fast_reversal.py`:
      `label = 1 if labels.ret_3 <= -0.3%` (mirrors the validated
      `_backtest_fast_reversal_by_mode.py` v1 definition). Computable on
      all history. 3 + 11 unit tests passing.
- [x] **Train** модель + AUC отчёт. ✅ 2026-05-18
      * Fixed `extract_features` (read non-existent `seq_features` →
        rewired to scalar `f` dict). AUC 0.53 → **0.63 train / 0.55 val**.
- [x] **60-day backtest** ✅ 2026-05-18 — `_backtest_fast_reversal_threshold.py`,
      3217 takes, 33.9% fast-reversal baseline:
      | T | win-recall | FR-rate | verdict |
      |---|-----------|---------|---------|
      | 0.55 (spec default) | 93.2% | 33.9→32.8% | **REJECT** (recall < 95%) |
      | 0.60 (best safe) | 95.9% | 33.9→33.0% | OK but only −0.9pp |
- [ ] **Wire bandit context** (без блока) — recommended next (soft signal).
- [ ] ~~Через 7 дней — включить guard~~ **BLOCKED by verdict below.**

**VERDICT (2026-05-18):** model (val AUC ~0.55) is too weak for a hard
guard. Every recall-safe threshold (≥0.60) yields a negligible
fast-reversal reduction (≤0.9pp) while still costing ~4% of winners at
the spec default. `FAST_REVERSAL_GUARD_ENABLED` stays **False**.
`proba_fast_reversal` should be consumed as a **soft LinUCB context
feature** (step 4), letting the policy weigh it without a hard block.
The acceptance gate did its job: it prevented shipping a NS-degrading
guard. Revisit only if a stronger model (richer features / non-linear)
lifts val AUC materially.

## 8. Follow-ups

- Per-mode пороги (`FAST_REVERSAL_PROBA_MAX_ALIGNMENT`, и т.д.).
- Связка с `trail-min-buffer`: если floor увеличил buffer, метрика
  fast-reversal должна сама ослабиться → пересмотреть label через
  realised buffer.
