# Trend/1h chop-filter

- **Slug:** `trend-1h-chop-filter`
- **Status:** draft → implementing
- **Created:** 2026-05-01
- **Owner:** core
- **Related:** `_validate_trend_chop_filter.py`, `files/trend_scout_rules.py`,
  пользовательский кейс STRKUSDT 2026-05-01 (ADX 20.2, slope 0.70 %).

---

## 1. Problem

`trend/1h` входит в боковики и микро-наклоны. Live-кейс
(STRKUSDT 2026-05-01): ADX 20.2, slope EMA20 +0.70 %, 4 дня
консолидации в диапазоне 0.0380–0.0400, catch-up drift −1.01 %.

Backtest 30 d (`_validate_trend_chop_filter.py`) подтверждает:
- Baseline: n=86 entries, **precision 1.2 %**, avg_pnl −0.17 %,
  FR_v1 22.6 %.
- 80 non-winners из 86 — каждый сжигает TG-канал шумом.

## 2. Success metric

**Primary:** на `trend/1h` precision ≥ 12 % при recall@top20 ≥ 95 %.

**Secondary (anti-metrics):**
- Total `trend/1h` entries падают ≥ 70 % (от 86 → ~6 / 30 d).
- avg_pnl ≥ 0.5 %.
- FR_v1 не вырастает > 5 п.п.

**Acceptance после 7 d live:**
- Precision actual ≥ 12 %.
- recall@top20 не падает > 2 п.п. на overall.
- TG-канал msg_rate D2 на trend/1h: ~5/d → ~0.5/d.

## 3. Scope

### In scope
- Новый Scout-rule в `trend_scout_rules.py`:
  `trend_1h_chop_filter` — блокирует `trend/1h` candidates,
  если `(ADX < 25) OR (slope_pct < 1.2) OR (vol_x < 1.3)`.
- Конфиг-флаги для каждого порога.
- Reason-code `trend_1h_chop` для post-hoc анализа.

### Out of scope
- `trend/15m` — backtest не показал чистый Pareto-win (recall
  падает до 29 %).
- `strong_trend/1h`, `alignment/1h` — другие гарды per spec.
- Изменение base-strategy в `strategy.py` (rule живёт в Scout-слое).

## 4. Behaviour / design

### Filter condition

```python
def _trend_1h_chop_predicate(ctx) -> bool:
    """
    Returns True if entry should be BLOCKED (chop detected).
    Applies only to mode == "trend" AND tf == "1h".
    """
    if ctx.mode != "trend" or ctx.tf != "1h":
        return False  # not applicable, don't block
    if not getattr(config, "TREND_1H_CHOP_FILTER_ENABLED", True):
        return False  # filter disabled
    adx_min = float(getattr(config, "TREND_1H_CHOP_ADX_MIN", 25.0))
    slope_min = float(getattr(config, "TREND_1H_CHOP_SLOPE_MIN", 1.2))
    vol_min = float(getattr(config, "TREND_1H_CHOP_VOL_MIN", 1.3))
    if ctx.adx < adx_min:    return True
    if ctx.slope_pct < slope_min: return True
    if ctx.vol_x < vol_min:  return True
    return False
```

Reason text:
```
"trend/1h chop: ADX {adx:.1f}<{adx_min} OR slope {slp:+.2f}<{slope_min} OR vol {vx:.2f}<{vol_min}"
```

### Bull-day relaxation

В bull-day импульсы шире, требования можно слегка смягчить
(аналогично `TREND_15M_QUALITY_*_BULL_DAY` подходу):

```python
TREND_1H_CHOP_ADX_MIN_BULL_DAY  = 22.0
TREND_1H_CHOP_SLOPE_MIN_BULL_DAY = 1.0
TREND_1H_CHOP_VOL_MIN_BULL_DAY   = 1.2
```

(Backtest этого не валидировал; бэктест дал 1.5 % bull-day overall —
малая выборка. Bull-day rule оставить как opt-in, default = standard
thresholds.)

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `TREND_1H_CHOP_FILTER_ENABLED` | True | Master switch |
| `TREND_1H_CHOP_ADX_MIN` | 25.0 | Минимум ADX |
| `TREND_1H_CHOP_SLOPE_MIN` | 1.2 | Минимум slope_pct |
| `TREND_1H_CHOP_VOL_MIN` | 1.3 | Минимум vol_x |
| `TREND_1H_CHOP_ADX_MIN_BULL_DAY` | 22.0 | Bull-day floor (если выбрано) |
| `TREND_1H_CHOP_USE_BULL_DAY_RELAX` | False | Активировать bull-day relax |

**Rollback:** `TREND_1H_CHOP_FILTER_ENABLED = False` → бот возвращается
к baseline. Никакой data-migration.

## 6. Risks

- **Recall на bull-day**: на коротком окне bull-day гипотеза не
  валидирована. Mitigation: `USE_BULL_DAY_RELAX=False` по умолчанию,
  включить после первой bull-day волны.
- **Pure-trend setups при низком vol_x**: иногда волатильность
  падает в начале нового тренда (классический Wyckoff `mark-up` start).
  Mitigation: vol_x cap не критичный (1.3 — мягкий), при необходимости
  поднимать только ADX.
- **Bandit confusion** — bandit видит только тех, кто прошёл гарды;
  его decision-boundary смещается. Не критично, обновится через
  next retrain.

## 7. Verification

- [x] **Backtest** на 30 d (`_validate_trend_chop_filter.py`):
  - baseline trend/1h: precision 1.2 %, avg_pnl −0.17 %, FR 22.6 %
  - filter (ADX≥25, slope≥1.2, vol≥1.3): precision 16.7 % (+15.5 pp),
    recall 100 %, avg_pnl +1.58 %.
- [ ] **Live 7 d** после flip:
  - precision ≥ 12 %
  - recall@top20 не упал > 2 п.п.
  - msg_rate D2 на trend/1h ≤ 1/d
- [ ] **Re-run validate script** через 7 d на новых данных.

### Rollback trigger
- recall@top20 упал > 2 п.п.
- 7 d без trend/1h entries (over-blocking) → рассмотреть смягчение
  до ADX≥22 / slope≥1.0.

## 8. Follow-ups

- Аналогичный анализ для `trend/15m` с другими порогами
  (recall 29 % при моих текущих — не подходит).
- Добавить `is_bull_day` в context Scout-rule’а для adaptive threshold.
- Объединить с P5 (ML blind-spot) — может оказаться, что filter
  блокирует часть blind-spot syms; cross-check.
