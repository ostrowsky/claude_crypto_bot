# Disable `breakout/15m` entry mode

- **Slug:** `breakout-15m-disable`
- **Status:** draft
- **Created:** 2026-04-26
- **Owner:** core
- **Related:** `files/strategy.py` (`breakout_entry`), `_validate_p7_breakout_15m.py`,
  `docs/reports/2026-04-28-improvements-validated.md` §P7

---

## 1. Problem

`breakout/15m` режим систематически не приносит edge — даже на coin’ах,
которые в тот же день оказываются в EOD top-20.

Backtest 30 d (`_validate_p7_breakout_15m.py`):

```
breakout/15m entries:                45
  on top-20 winners:                  5  (11.1%)
  on non-winners:                    40

Paired entry → exit:                 44, avg_pnl=-0.33%, win=23%, FR_v1=22.7%
  on top-20 (5):                     avg_pnl=+0.03%   ← даже здесь zero
  on non-winners (39):               avg_pnl=-0.37%
```

При FR_v1 = 22.7 % и нулевой capture даже на правильно угаданных
top-20, режим **отнимает время на исполнение и публикует слабые
сигналы** в Telegram.

## 2. Success metric

**Primary:** общий portfolio avg_pnl за 30 d не ухудшается (= больше
сетапов с отрицательным EV не публикуется).

**Secondary:** число FR_v1 событий в `bot_events.jsonl` уменьшается
пропорционально числу `breakout/15m` entries (≈ −10 событий/30 d).

**Acceptance:** после 7 d с `BREAKOUT_15M_ENABLED=False` —
recall@top20 не упал > 2 п.п. (не теряем уникальные top-20 winners).

## 3. Scope

### In scope
- Новый config-флаг `BREAKOUT_15M_ENABLED: bool = False`.
- Условие в `strategy.py::breakout_entry`: ранний return при
  `not BREAKOUT_15M_ENABLED`.
- Лог-line при первом блоке за сессию.

### Out of scope
- Удаление кода `breakout_entry` (оставляем для возможного rollback).
- Удаление breakout режима полностью (могут быть future-режимы
  `breakout/1h` etc).
- Изменение config’ов других режимов.

## 4. Behaviour / design

### Code change

В `files/strategy.py`, в начале `breakout_entry` (15m):

```python
def breakout_entry(...):
    if not getattr(config, "BREAKOUT_15M_ENABLED", True):
        return None  # mode disabled
    # … existing logic
```

### Config

```python
# files/config.py
BREAKOUT_15M_ENABLED: bool = False  # disabled 2026-04-26 per backtest
```

### Lifecycle

- Ранее открытые `breakout/15m` позиции дорабатывают по обычному
  trail-логику; они НЕ закрываются принудительно.
- После рестарта бот не выдаёт новых `breakout/15m` candidates.

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `BREAKOUT_15M_ENABLED` | `False` | Master switch |

**Rollback:** flip `BREAKOUT_15M_ENABLED=True` и restart. Никакой
миграции данных.

## 6. Risks

- **Потеря 5 top-20 entries / 30 d** (≈ 11 % всех breakout entries).
  Но `avg_pnl=+0.03 %` на этих 5 — capture-ratio близка к нулю;
  потеря marginal.
- **Bandit decision-boundary смещается:** меньше data-points для
  arm’а на этом режиме. Mitigation: trail-bandit получает signal
  только при entered-сетапах; если их 0 — context для этого режима
  замораживается, но не повреждается.
- **Future regret:** market-режим может измениться, и breakout/15m
  может снова заработать. Mitigation: rollback = 1 строка config.

## 7. Verification

- [ ] **Backtest:** уже сделан (`_validate_p7_breakout_15m.py`):
      avg_pnl=+0.03 % на top-20, −0.37 % на non-winners.
- [ ] **Live 7 d** с флагом False:
  - сравнить overall avg_pnl за 7 d с предыдущими 7 d (до отключения);
  - проверить, что recall@top20 не упал > 2 п.п.
- [ ] **Re-run backtest через 30 d** для подтверждения отсутствия
      regression в paired-trade pool.

## 8. Follow-ups

- Если live 7 d показывает, что overall avg_pnl **улучшилось** —
  закрыть spec как `shipped`, обновить `signal-pipeline-spec.md`
  (исключить breakout/15m из «7 entry modes» → 6).
- Если ухудшилось — rollback и audit `breakout_entry` логики
  (возможно, тонкая настройка, а не disable).
