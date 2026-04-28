# Portfolio rotation (ML-gated weak-leg eviction)

- **Slug:** `portfolio-rotation`
- **Status:** shipped (2026-04-17)
- **Created:** 2026-04-26 (retroactive)
- **Owner:** core
- **Related:** `files/rotation.py`, `files/test_rotation.py`,
  `files/backtest_portfolio_rotation_grid.py`, `CLAUDE.md` §8

---

## 1. Problem

Портфель упирается в `MAX_OPEN=10` со слабыми ногами (EV=−0.5), блокируя
22+ trending кандидатов. Naive `MAX_OPEN+1` даёт лишь avg_r5=+0.04 % —
переполнение убивает edge.

## 2. Success metric

- При full-portfolio + сильный новый кандидат → авто-эвикция самой слабой
  ноги.
- Грид-бектест: avg_r5 ≥ +0.20 %, Sharpe ≥ +2.5 на 30-дневном окне.

Pareto-победитель на 2026-04-17: `ml_proba ≥ 0.62`, без score-фильтра, n=211,
**avg_r5=+0.241 %, Sharpe=+2.75**.

## 3. Scope

### In scope
- `should_rotate(...) → RotationDecision`.
- `find_weakest_leg(positions) → sym|None`.
- `evict_position(...)` — soft-exit через trail.

### Out of scope
- Hard-sell на бирже (бот = signal-only).

## 4. Behaviour / design

### Условия входа в ротацию

`should_rotate` возвращает `(allowed, weak_sym, reason)`:

1. Портфель = `MAX_OPEN`.
2. Новый кандидат: `ml_proba ≥ ROTATION_ML_PROBA_MIN (0.62)`.
3. Есть `weak`: `ev ≤ ROTATION_WEAK_EV_MAX (-0.40)` И
   `bars_held ≥ ROTATION_WEAK_BARS_MIN (3)`.
4. На weak P&L < `ROTATION_PROFIT_PROTECT_PCT (0.5 %)`
   (нельзя выкидывать прибыльную позицию).
5. `ROTATION_MAX_PER_POLL=1` — максимум 1 эвикция за бар.

Каждое не-entry-решение логируется с `reason` для post-mortem.

### Eviction mechanism

`evict_position(pos, last_price)` ставит:

```python
pos.trail_stop = last_price * 1.001
```

Следующий ATR-poll закрывает позицию естественно — никакого forced
market-sell. Это даёт plenty of slippage room и не ломает Telegram-UX.

### Cooldown

`COOLDOWN_BARS=19` (был 24) — связано с ротацией, чтобы не было
немедленного re-entry в выкинутую монету.

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `ROTATION_ENABLED` | True | Master switch |
| `ROTATION_ML_PROBA_MIN` | 0.62 | Минимум на нового кандидата |
| `ROTATION_WEAK_EV_MAX` | −0.40 | Условие на «слабую ногу» |
| `ROTATION_WEAK_BARS_MIN` | 3 | Только если weak держится ≥ 3 баров |
| `ROTATION_PROFIT_PROTECT_PCT` | 0.5 | % P&L, выше которого weak не эвиктится |
| `ROTATION_MAX_PER_POLL` | 1 | Защита от cascade-эвикций |
| `COOLDOWN_BARS` | 19 | Анти re-entry |

Откат: `ROTATION_ENABLED=False` — поведение возвращается к pre-rotation.

## 6. Risks

- **Эвикция top-gainer’а:** покрыто `ROTATION_PROFIT_PROTECT_PCT` и
  `ROTATION_ML_PROBA_MIN` (новый кандидат должен сильно превосходить).
- **Cascade в shock-day:** `ROTATION_MAX_PER_POLL=1`.

## 7. Verification

- `files/test_rotation.py` — 21 unit-тест.
- `files/backtest_portfolio_rotation.py` — сценарный sweep.
- `files/backtest_portfolio_rotation_grid.py` — Pareto-грид (score ×
  ml_proba). Тот же паттерн используется для любого нового гейта.

## 8. Follow-ups

- Volatility-aware sizing → меньшие позиции при широком trail-floor.
- Расширить `find_weakest_leg` фичами beyond EV (capture_ratio_pred?).
