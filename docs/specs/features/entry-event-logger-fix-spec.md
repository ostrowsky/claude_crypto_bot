# Entry-event logger fix · ranker fields

- **Slug:** `entry-event-logger-fix`
- **Status:** draft → implementing
- **Created:** 2026-04-30
- **Owner:** core
- **Related:** `_validate_1a_tgprob_megatrigger.py`,
  `_validate_4a_precision_prune.py`,
  `docs/reports/2026-04-30-roadmap-validation.md` §1A/4A

---

## 1. Problem

Entry-события в `bot_events.jsonl` логируют только базовые
индикаторы (`adx, rsi, vol_x, ml_proba, slope_pct, daily_range,
macd_hist, ema20, …`). **Ranker-output’ы отсутствуют** — они
вычисляются и сохраняются в `positions.json` / `pos.*`, но не
дублируются в event payload.

Эффект: половина north-star roadmap’а **невалидируема post-hoc**:

- 1A `top_gainer_prob` mega-trigger — нет данных для precision
  sweep по proba.
- 4A precision-prune — без `ranker_ev` / `ranker_quality_proba`
  возможен только sweep по `ml_proba` (marginal результат).
- Любой будущий tuning гардов на основе ranker-output’ов
  слеп post-hoc.

## 2. Success metric

После 7 d работы с patched logger:
- `_validate_1a_tgprob_megatrigger.py` retry → ≥ 80 % entries
  имеют `ranker_top_gainer_prob` non-null.
- `_validate_4a_precision_prune.py` retry → можно sweep’ать
  filters с ranker-полями.

## 3. Scope

### In scope
- Добавить 4 поля в entry-event payload в `monitor.py`:
  `ranker_top_gainer_prob`, `ranker_ev`, `ranker_quality_proba`,
  `signal_mode`.
- Никаких schema-changes в reader’ах (поля nullable: если
  ranker не вызывался — `None`).

### Out of scope
- Расширение схемы `blocked` events (отдельная мини-спека при
  необходимости).
- Архивный backfill старых событий (невозможно без re-run модели).

## 4. Behaviour / design

### Где логируется entry event

В `monitor.py` есть site, где после успешного entry формируется
JSON payload и пишется в `bot_events.jsonl`. Все ranker-поля
уже доступны в локальном scope (они только что были посчитаны
для `pos.ranker_*` и `pos.signal_mode`).

### Patch

```python
# В monitor.py, в entry-event payload dict
{
    # ... existing fields ...
    "ml_proba": ml_proba,
    # NEW:
    "ranker_top_gainer_prob": getattr(pos, "ranker_top_gainer_prob", None),
    "ranker_ev":              getattr(pos, "ranker_ev", None),
    "ranker_quality_proba":   getattr(pos, "ranker_quality_proba", None),
    "signal_mode":            getattr(pos, "signal_mode", None),
    # ... остальные fields ...
}
```

`getattr` со значением по умолчанию `None` — защита, если
ranker почему-то не вызвался.

## 5. Config flags & rollback

Изменение покрывает только формат логов. **Откат не нужен** —
старые reader-скрипты игнорируют unknown поля; новые fields
nullable.

## 6. Risks

- **JSONL line size ↑** на ~80 байт. На текущем темпе (≈30 entry/d)
  это +2.4 KB/d — pренебрежимо.
- **`getattr` на None** — все downstream скрипты используют
  `e.get("ranker_*")` с дефолтом, ничего не сломается.

## 7. Verification

- [x] Спека написана.
- [ ] Patch monitor.py.
- [ ] Restart bot.
- [ ] Через 1 ч: первое entry событие → `grep '"ranker_ev"'` в
      bot_events.jsonl возвращает ≥ 1 запись.
- [ ] Через 7 d: re-run `_validate_1a_*.py` и `_validate_4a_*.py`
      с реальными данными.

## 8. Follow-ups

- Расширить `blocked` events с теми же ranker-полями
  (для post-hoc анализа гардов).
- Добавить `entry_score` / `score_floor` в payload — для P5
  early-stage scoring audit.
