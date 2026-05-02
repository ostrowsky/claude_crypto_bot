# H5 · Trailing-only after break-even

- **Slug:** `h5-trailing-only-break-even`
- **Status:** draft → implementing (behind flag, default OFF)
- **Created:** 2026-05-02
- **Owner:** core
- **Related:** `_validate_h5_trailing_only.py`,
  `2026-05-02-ex1-baseline.md` (EX1 baseline),
  `2026-05-01-mode-audit.md` §H5.

---

## 1. Problem

Worst exit-class по EX1: **`ema20_weakness`** (median EX1 −0.010).
Бот закрывается по pattern «2 closes below EMA20» / «slope flip» /
«ADX weakening» **на лёгком retracement** в bull-trend, тогда как
coin продолжает движение.

Худшие кейсы (из EX1 baseline):
- APEUSDT 04-30 (top-20): exit pnl +0.51 %, potential +472 % → left
  на столе **+471 %**
- TRUUSDT 04-21: pnl −8.89 % при potential +465 %
- DOGSUSDT 04-19: pnl −5.77 % при +224 %
- (для negative-pnl случаев H5 не применяется — это защитные exits)

H5 backtest 30 d: 4 кейса с soft-EMA exit AND pnl ≥ +0.5 %, из них
1 на top-20 winner — **+471 % left on table** одной сделкой.

## 2. Success metric

**Primary:** EX1 median на top-20 winners растёт с +0.001 → ≥ +0.05
после 30 d с включённым flag.

**Secondary (anti-metrics):**
- Q2 whipsaw_rate не растёт > 2 σ.
- max_drawdown_per_signal не растёт > 1 σ.
- recall@top20 не падает > 2 п.п.

**Acceptance перед flip → True:**
- 7 d shadow: ≥ 3 «would-suppress» событий зарегистрированы в
  логах с reason `h5_shadow_suppress`.
- Их EX1 (counterfactually) ≥ +0.05.

## 3. Scope

### In scope
- Helper `_h5_suppress_soft_exit(reason, pnl_pct, mode)` в monitor.py.
- Условие: `pnl_pct >= H5_BREAK_EVEN_PCT (0.5)` AND reason matches
  soft EMA-pattern.
- Flag `H5_TRAILING_ONLY_AFTER_BREAK_EVEN_ENABLED` (default `False`).
- Shadow-mode `H5_TRAILING_ONLY_SHADOW` (default `True`):
  логировать «would_suppress» события без affecting behaviour.

### Suppressed reasons
- `2 закрытия подряд ниже EMA20 ...`
- `цена ниже EMA20 ... + подтверждённая слабость`
- `EMA20 разворачивается вниз (slope ...)`
- `ADX ослабевает ...`

### NOT suppressed (kept always)
- ATR-trail (handled separately, не через `check_exit_conditions`).
- Time max_hold.
- RSI overbought.
- All `⚠️ WEAK:` reasons (имеют свой own min-bars guard).
- Continuation-micro exits.
- MACD warnings (handled separately).

### Out of scope
- Изменение порога 0.5 % per mode (фиксированный default).
- Tighter ATR-trail после break-even (отдельная спека H7
  `continuous-trail-k`).
- Partial profit-lock (H8, отдельная спека).

## 4. Behaviour / design

### Helper

```python
def _h5_should_suppress(reason: Optional[str], pnl_pct: float) -> bool:
    """
    H5: trailing-only after break-even.
    Returns True if soft EMA-pattern exit should be suppressed because
    position is profitable enough to ride out a normal pullback.

    Spec: docs/specs/features/h5-trailing-only-break-even-spec.md
    """
    if not reason:
        return False
    pnl_min = float(getattr(config, "H5_BREAK_EVEN_PCT", 0.5))
    if pnl_pct < pnl_min:
        return False
    r = reason.lower()
    # Don't suppress WEAK signals (own guard) or RSI/MACD/time exits
    if r.startswith("⚠️") or "weak" in r:
        return False
    if "rsi" in r or "macd" in r:
        return False
    if "лимит" in r or "max_hold" in r:
        return False
    soft_markers = (
        "2 закрытия подряд ниже ema20",
        "ниже ema20",
        "ema20 разворачивается",
        "adx ослабевает",
    )
    return any(m in r for m in soft_markers)
```

### Wire

В `monitor.py` ~ L5317 (после `check_exit_conditions`):

```python
if reason:
    is_weak = reason.startswith("⚠️ WEAK:")
    if is_weak and pos.bars_elapsed < min_weak_bars:
        ...
    elif is_weak and _apply_trend_hold_weak_exit_override(...):
        ...
    # NEW: H5 trailing-only after break-even
    elif _h5_should_suppress(reason, current_pnl):
        h5_enabled = bool(getattr(config, "H5_TRAILING_ONLY_AFTER_BREAK_EVEN_ENABLED", False))
        h5_shadow = bool(getattr(config, "H5_TRAILING_ONLY_SHADOW", True))
        if h5_enabled:
            log.info("H5 SUPPRESS %s [%s] pnl=%.2f%%: %s",
                     sym, tf, current_pnl, reason)
            reason = None
        elif h5_shadow:
            log.info("H5 SHADOW would-suppress %s [%s] pnl=%.2f%%: %s",
                     sym, tf, current_pnl, reason)
            # behaviour unchanged
```

### Trail-stop в момент suppression

Когда H5 подавляет soft-exit, мы оставляем позицию открытой. ATR-trail
продолжает действовать без изменений. Whipsaw-buffer уже floor’ом
(`trail-min-buffer` v2.5.x) защищает downside.

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `H5_TRAILING_ONLY_AFTER_BREAK_EVEN_ENABLED` | False | Master switch |
| `H5_TRAILING_ONLY_SHADOW` | True | Logging-only mode (для acceptance) |
| `H5_BREAK_EVEN_PCT` | 0.5 | Минимальный pnl для suppression |

**Rollback:** `H5_TRAILING_ONLY_AFTER_BREAK_EVEN_ENABLED = False`
(шум в логах от shadow остаётся, но behaviour восстановлен).

## 6. Risks

- **Размер выборки маленький.** За 30 d только 4 H5-eligible exits.
  Statistical confidence низкий. Mitigation: 30+ d shadow перед flip.
- **Reversal protection delay.** ATR-trail может cрабатывать на
  1-3 бара позже чем ema20-pattern. На fast reversal коинов — это
  потерянная margin. Mitigation: trail-min-buffer floor уже есть.
- **Bandit confusion.** Pos durations растут при suppression. Bandit
  trail-arm context видит unusual long-hold. Mitigation: уже есть
  follow-up в `trail-min-buffer-spec` §8 (`max_hold_extended` flag
  для bandit).
- **TG UX.** Сделки длятся дольше — пользователь видит «🟢 BUY» и
  потом долго ждёт SELL. Mitigation: только профитные позиции
  держатся дольше; psychologically ОК.

## 7. Verification

- [x] Backtest 30 d (`_validate_h5_trailing_only.py`):
  - H5-eligible: 4 / 982 paired exits
  - top-20 H5-eligible: 1 (APEUSDT 04-30)
  - +471 % left on table одним кейсом
- [x] Soft-at-loss counter: 136 случаев (avg pnl −1.25 %) НЕ
  попадают под H5 — protective exits сохраняются.
- [ ] **Shadow 7 d** (default): подсчитать `H5 SHADOW would-suppress`
  events; их EX1 (counterfactually).
- [ ] **Acceptance**: ≥ 3 shadow events с потенциальным
  EX1 ≥ +0.05.
- [ ] **Flip → True** + 7 d live: EX1 median растёт ≥ +0.03.

## 8. Follow-ups

- **Shadow аналог через 30 d**: повторный запуск backtest с
  фактическими post-suppression данными.
- **Multi-tier threshold**: pnl ≥ +1.5 % → aggressive suppress
  всех soft, включая RSI; pnl ≥ +0.3 % → текущий H5; pnl < 0 →
  оставить всё.
- **H7 continuous trail_k**: после H5 suppression — tighter trail
  (k = 1.5 вместо 2.5) чтобы поймать реальный reversal быстрее.
- **EX1 re-baseline после 7 d** для сравнения.
