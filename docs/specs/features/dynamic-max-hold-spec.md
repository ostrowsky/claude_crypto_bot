# Dynamic `max_hold_bars` based on momentum

- **Slug:** `dynamic-max-hold`
- **Status:** draft
- **Created:** 2026-04-30
- **Owner:** core
- **Related:** `_validate_2a_dynamic_max_hold.py`,
  `docs/reports/2026-04-30-roadmap-validation.md` §2A,
  `docs/reports/2026-04-29-north-star-roadmap.md` (Path 2)

---

## 1. Problem

Бот выходит из позиции на time-trigger (`bars_held >= max_hold_bars`)
**даже когда тренд продолжается**. Это даёт капитальную просадку
по `EarlyCapture@top20`: за 30 d 13 time-exits на top-20 winners
оставили **+1445 % суммарного движения на столе** (в среднем
~111 % на trade). Топ-кейсы:

```
MANAUSDT 04-01 impulse_speed bars=23  pnl=−1.91%   eod=+366%   left=+368%
MDTUSDT  04-01 retest        bars=10  pnl=−0.23%   eod=+274%   left=+274%
ADAUSDT  04-03 impulse_speed bars=43  pnl=+1.36%   eod=+288%   left=+287%
ORDIUSDT 04-16 impulse_speed bars=24  pnl=+28.12%  eod=+173%   left=+145%
```

Базовый `MAX_HOLD_BARS=24` (1h) или 24 (15m) выбирался под защиту
от затяжных боковиков, но **в bull-day и продолжающемся импульсе**
он становится слабым звеном — мы фиксируемся именно тогда, когда
монета только разгоняется.

## 2. Success metric

**Primary:** capture_ratio (E2) на top-20 winners растёт с **0.16
→ 0.24** (+50 %) на 30-d backtest.

**Secondary (anti-metrics):**
- `recall@top20` не падает > 2 п.п.
- `whipsaw_rate` (Q2) не растёт > 2 σ от 7-d baseline.
- `max_drawdown_per_signal` не растёт > 1 σ.

**Acceptance threshold для promote:** capture_ratio mean ≥ 0.20
после 7 d shadow + 7 d live.

## 3. Scope

### In scope
- Дополнительная проверка в `monitor.py` exit-checker: перед
  закрытием по `bars_held >= max_hold_bars` оценить **momentum
  condition**. Если momentum still bullish → продлить hold на
  ещё N баров.
- Конфиг-флаги для thresholds.
- Лог-line при extension.

### Out of scope
- Изменение `ATR-trail` логики (защита от reversal остаётся).
- Изменение `EMA20-weakness` exit (отдельная спека `pullback-tolerance` 2B).
- Изменение `max_hold_bars` defaults — продление поверх существующего значения.

## 4. Behaviour / design

### Momentum condition

При каждом баре, когда `pos.bars_held >= pos.max_hold_bars`:

```python
def can_extend_hold(pos, bar_data):
    if not config.DYNAMIC_MAX_HOLD_ENABLED:
        return False
    # Trend integrity:
    if bar_data.adx < pos.entry_adx:           # ADX cooling
        return False
    if bar_data.close < bar_data.ema20:         # Below trend support
        return False
    # PnL must be in profit (don't bet on losers)
    pnl_pct = (bar_data.close - pos.entry_price) / pos.entry_price * 100
    if pnl_pct < config.DYNAMIC_MAX_HOLD_MIN_PNL_PCT:  # default 0.5 %
        return False
    # Cap on extensions:
    if getattr(pos, "max_hold_extensions", 0) >= config.DYNAMIC_MAX_HOLD_MAX_EXTENSIONS:
        return False
    return True
```

Если `can_extend_hold(pos, bar) is True`:
- `pos.max_hold_bars += config.DYNAMIC_MAX_HOLD_EXTEND_BY` (default 6 баров)
- `pos.max_hold_extensions = (pos.max_hold_extensions or 0) + 1`
- Лог: `[EXT] {sym} momentum-extend max_hold {old}→{new} (adx={...}, pnl={...}%)`

### Per-mode tuning

Применяется только к режимам, где validated impact:
- `impulse_speed` (15m + 1h) — основной win
- `strong_trend` (1h)
- `trend` (1h)
- `retest` (1h) — has `MDTUSDT` case in samples

Для `breakout`, `alignment`, `impulse` — оставить статический
max_hold (риск reversal слишком высок per Q1 FR rate).

### Cap

`DYNAMIC_MAX_HOLD_MAX_EXTENSIONS = 3` (max 3×6 = 18 extra bars).

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `DYNAMIC_MAX_HOLD_ENABLED` | False (до promote) | Master switch |
| `DYNAMIC_MAX_HOLD_MIN_PNL_PCT` | 0.5 | Минимум pnl для extension |
| `DYNAMIC_MAX_HOLD_EXTEND_BY` | 6 | Сколько баров добавлять |
| `DYNAMIC_MAX_HOLD_MAX_EXTENSIONS` | 3 | Cap на число extension’ов |
| `DYNAMIC_MAX_HOLD_MODES` | `["impulse_speed", "strong_trend", "trend", "retest"]` | Whitelist режимов |

**Rollback:** `DYNAMIC_MAX_HOLD_ENABLED = False` → бот возвращается к
статичному max_hold. Открытые позиции с уже-расширенным max_hold
дорабатывают по новому значению (no force-close).

## 6. Risks

- **Reversal в bull-day.** Momentum condition требует ADX growing
  AND price > EMA20 — но между барами condition может flip’нуться.
  Mitigation: ATR-trail продолжает работать; реверсал ловится trail’ом.
- **Bandit confusion.** Trail-bandit получает реальный
  `max_hold_bars` post-extension. Контекст бандита не включает
  «я продлил» — он увидит unusual long-hold. Mitigation:
  при первом extension писать context-shift в bandit memory
  как `max_hold_extended` (TBD; пока — `pass`).
- **Per-mode misclassification.** `impulse_speed/15m` бывает с
  очень коротким natural lifetime — cap `MAX_EXTENSIONS=3` защищает.

## 7. Verification

### Backtest (offline, по 30 d)
- [x] Уже сделано: `_validate_2a_dynamic_max_hold.py` →
  29 trades on top-20 winners would benefit, total +3000 % left.
- [ ] **Counterfactual sim:** для каждого time-exit + EMA-exit
  на top-20 winner, симулировать «продлеваем до EOD» с trail’ом.
  Compute realized pnl_extended vs pnl_actual. Compare aggregate
  capture_ratio vs baseline 0.16.

### Shadow A/B (7 d live)
- [ ] Деплой с `DYNAMIC_MAX_HOLD_ENABLED = False` И shadow-flag
  `DYNAMIC_MAX_HOLD_SHADOW = True`. Bot не меняет behaviour, но
  записывает в `bot_events.jsonl` events с reason=`dynamic_max_hold_shadow`
  для случаев, где extension сработал бы.

### Promote criteria
- [ ] Counterfactual sim: capture_ratio_extended − capture_ratio_actual
  ≥ +0.05 (50 % expected).
- [ ] Shadow 7 d: ≥ 5 events `would_extend` без increase в drawdown
  на симулированных post-extension barах.
- [ ] After promote (live 7 d): capture_ratio E2 ≥ 0.20, recall@20
  не падает > 2 п.п., Q2 whipsaw не растёт.

### Rollback trigger
- max_drawdown_per_signal вырос > 1 σ → flip flag, postmortem.
- recall@20 упал > 2 п.п.

## 8. Follow-ups

- Объединить с 2B (`pullback-tolerance-spec.md`): если PnL > 0
  и `top_gainer_prob` всё ещё > 0.5, не выходить по EMA-weakness.
- Добавить `realised_buffer_pct` и `max_hold_extended_count` в
  bandit context vector.
- Адаптивный `DYNAMIC_MAX_HOLD_EXTEND_BY` per mode (impulse_speed
  6 bars, strong_trend 4, retest 8).
