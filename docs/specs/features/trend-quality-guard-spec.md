# Trend quality guard (15m)

- **Slug:** `trend-quality-guard`
- **Status:** shipped (retroactive)
- **Created:** 2026-04-26
- **Owner:** core
- **Related:** `files/trend_scout_rules.py`, `CLAUDE.md` §7

---

## 1. Problem

`trend` режим на 15m часто триггерится на «уставшем» восходящем движении
(RSI > 72, цена слишком далеко от EMA20, дневной диапазон уже > 10 %).
Такие сетапы дают резкий fade. Нужен guard, который режет «late entry».

## 2. Success metric

- Среди blocked-bucket этого guard `avg_r5 ≤ take avg_r5` (т.е. блокирует
  реально слабое, не TG-кандидатов).

На 2026-04 status: mild over-blocking (avg_r5=+0.127 %, n=433,
take=−0.016 %). Кандидат на смягчение через Pareto.

## 3. Scope

### In scope
- 4 проверки: RSI cap, price-edge от EMA20, daily_range cap.
- Bull-day relaxation отдельным набором порогов.

### Out of scope
- Аналог для других режимов / TF.

## 4. Behaviour / design

Гард `trend_quality` в `trend_scout_rules.py`. Блокирует, если:

```
RSI(15m) > TREND_15M_QUALITY_RSI_MAX (72.0; bull-day 76.0)
ИЛИ
price_edge_from_ema20 > TREND_15M_QUALITY_PRICE_EDGE_MAX_PCT (3.20; bull-day 4.00)
ИЛИ
daily_range > TREND_15M_QUALITY_DAILY_RANGE_MAX (10.0; bull-day 14.0)
```

`is_bull_day` определяется в `monitor.py` (BTC vs EMA50, market breadth).

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `TREND_15M_QUALITY_GUARD_ENABLED` | True | Master switch |
| `TREND_15M_QUALITY_RSI_MAX` | 72.0 | RSI cap normal day |
| `TREND_15M_QUALITY_RSI_MAX_BULL_DAY` | 76.0 | Bull-day RSI cap |
| `TREND_15M_QUALITY_PRICE_EDGE_MAX_PCT` | 3.20 | Edge cap normal |
| `TREND_15M_QUALITY_PRICE_EDGE_MAX_BULL_DAY_PCT` | 4.00 | Edge cap bull |
| `TREND_15M_QUALITY_DAILY_RANGE_MAX` | 10.0 | Range cap normal |
| `TREND_15M_QUALITY_DAILY_RANGE_MAX_BULL_DAY` | 14.0 | Range cap bull |

Откат: `TREND_15M_QUALITY_GUARD_ENABLED=False`.

## 6. Risks

- **Over-blocking TG’ов:** уроки 2026-04-06 (RSI 68 был слишком жёстким
  → 100 % TG блок) и 2026-04-13 (TAO 12 % daily-range вылетел из
  старого 10 %-кэпа). Любое ужатие — только после Pareto sweep.

## 7. Verification

- `files/analyze_blocked_gates.py` — текущая выборка.

## 8. Follow-ups

- Возможный sweep по `RSI_MAX` (72→74) и `PRICE_EDGE` (3.2→3.5) — есть
  запас по +avg_r5 в blocked-bucket.
