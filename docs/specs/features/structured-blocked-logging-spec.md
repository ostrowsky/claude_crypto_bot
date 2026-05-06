# Structured blocked-candidate logging (P0.1)

- **Slug:** `structured-blocked-logging`
- **Status:** draft → implementing
- **Created:** 2026-05-07
- **Owner:** core
- **Related:** `2026-05-07` TON case audit (manual Python diag for «why 39
  blocks»), `2026-05-06` STRK case (24h zero events without explanation),
  `claude_crypto_bot_analysis_for_claude.md` §11 / P0.1.

---

## 1. Problem

Когда пользователь спрашивает «почему нет сигнала на X» — расследование
сейчас требует **ручной Python-скрипт** для каждого symbol:

- Чтение `bot_events.jsonl` (10+ MB)
- Группировка по `event=blocked`, `sym=X`
- Парсинг текстовых `reason` строк
- Cross-reference с features из других мест

Текущий `log_blocked` пишет только: sym, tf, price, reason (text),
optional rsi/adx/vol_x/daily_range, signal_type. **Этого недостаточно**
чтобы:

1. Группировать по `reason_code` (текстовый reason содержит varied numbers).
2. Понять что блокировало бы coin при **другом** значении filter.
3. Cross-reference с ML/ranker scores.
4. Аггрегировать «would-be-signal» сетапы.

## 2. Success metric

**Primary:** на любую жалобу «почему нет сигнала» — один grep по
`reason_code` + features в `bot_events.jsonl`, ответ за < 1 минуты.

**Secondary:**
- Daily aggregation по `reason_code` показывает топ-5 over-blocking gates.
- Histogram features at-block-time для cross-mode analysis.
- 0 production behaviour change.

## 3. Scope

### In scope
- Расширить `botlog.log_blocked()` дополнительными optional kwargs
  для structured context.
- Добавить `reason_code` (structured) рядом с текстовым `reason`.
- В каждом call site в `monitor.py` передавать доступный context.
- Новый backtest script `_backtest_blocked_breakdown.py`:
  - груп по `reason_code` × top syms × top tf.
  - distribution features per gate.
  - «would-be-signal» candidates (если threshold relaxed).

### Out of scope
- Изменение поведения trading pipeline (только observability).
- Логирование candidate stages WATCH/HOLD (отдельная спека P1.2).
- Schema migration старых событий — новые поля nullable, старые reader’ы
  игнорируют.

## 4. Behaviour / design

### Extended log_blocked schema

```python
def log_blocked(
    sym: str, tf: str, price: float, reason: str,
    *,
    # text reason classification (structured short-form)
    reason_code: Optional[str] = None,
    gate: Optional[str] = None,            # e.g. "ml_zone", "trend_chop"
    signal_type: str = "buy",
    # current bar features
    rsi: Optional[float] = None,
    adx: Optional[float] = None,
    vol_x: Optional[float] = None,
    daily_range: Optional[float] = None,
    slope_pct: Optional[float] = None,
    macd_hist: Optional[float] = None,
    ema20: Optional[float] = None,
    ema50: Optional[float] = None,
    ema200: Optional[float] = None,
    price_edge_ema20_pct: Optional[float] = None,
    atr_pct: Optional[float] = None,
    # ML/ranker scores at block time
    ml_proba: Optional[float] = None,
    ranker_top_gainer_prob: Optional[float] = None,
    ranker_ev: Optional[float] = None,
    ranker_quality_proba: Optional[float] = None,
    ranker_final_score: Optional[float] = None,
    candidate_score: Optional[float] = None,
    score_floor: Optional[float] = None,
    # market context
    is_bull_day: Optional[bool] = None,
    btc_vs_ema50: Optional[float] = None,
    market_regime: Optional[str] = None,
    # gate-specific extras (free-form key/value)
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ...
```

Все поля nullable → старые callers продолжают работать.

### Reason code taxonomy (initial)

```
ml_zone              ML proba outside profitable zone
ranker_hard_veto     Ranker final_score below veto threshold
ranker_soft_veto     Ranker EV below soft veto
trend_chop           trend/1h chop filter (slope/adx/vol)
trend_quality        15m trend quality guard (RSI/edge/range)
entry_score          Score below floor
impulse_guard        impulse_speed sub-conditions (multiple)
mode_range_quality   daily_range outside mode-specific bounds
clone_guard          Similar setup limit
open_cluster_cap     Cluster open positions cap
correlation_guard    High correlation with existing pos
mtf                  Multi-timeframe disagreement
late_continuation    Continuation entry too late
late_impulse_rotation Late rotation candidate
cooldown             Symbol-level cooldown
portfolio            Portfolio full
time_block           Time-of-day block
clone_signal_guard   Same-cluster recent setup
strategy_cap         Strategy-level cap
enhanced_block       Phase-2 enhanced filter
ml_filter            Generic ML filter (legacy)
```

### Helper for monitor.py

```python
def _build_block_context(*, feat, i, candidate_score, score_floor,
                         ranker_proba, ranker_info, is_bull_day_now,
                         pos_obj=None) -> dict:
    """Assemble standard fields available at any block site."""
    return {
        "rsi": _safe_float(feat["rsi"][i]),
        "adx": _safe_float(feat["adx"][i]),
        "vol_x": _safe_float(feat["vol_x"][i]),
        "daily_range": _safe_float(feat.get("daily_range_pct", [None]*999)[i]),
        "slope_pct": _safe_float(feat["slope"][i]),
        "macd_hist": _safe_float(feat.get("macd_hist", [None]*999)[i]),
        "ema20": _safe_float(feat.get("ema_fast", [None]*999)[i]),
        "ema50": _safe_float(feat.get("ema_slow", [None]*999)[i]),
        "ema200": _safe_float(feat.get("ema200", [None]*999)[i]),
        "ml_proba": _safe_float(ranker_proba),
        "ranker_top_gainer_prob": (ranker_info or {}).get("top_gainer_prob"),
        "ranker_ev": (ranker_info or {}).get("ev"),
        "ranker_quality_proba": (ranker_info or {}).get("quality_proba"),
        "ranker_final_score": (ranker_info or {}).get("final_score"),
        "candidate_score": candidate_score,
        "score_floor": score_floor,
        "is_bull_day": is_bull_day_now,
        "btc_vs_ema50": float(getattr(config, "_btc_vs_ema50", 0.0)),
        "market_regime": str(getattr(config, "_market_regime", "neutral")),
    }
```

### Daily aggregation

`files/_backtest_blocked_breakdown.py` (new):
- Groups by `reason_code` × tf × symbol.
- Reports top-20 over-blocking gates per day.
- Histograms of `ml_proba`, `ranker_ev`, `slope_pct` per gate.
- Optional `--would-be-signal` mode: per gate, what % of blocked
  candidates would pass if threshold relaxed by N%.

## 5. Config flags & rollback

Без флагов. Backward-compat по default — новые kwargs опциональны.

**Rollback:** удалить вызовы дополнительных kwargs (но они nullable —
ничего не сломается).

## 6. Risks

- **JSONL line size grows** ~3-5× (от ~150 байт до ~500-800).
  - 30 entries/day × 800B = 24 KB/day. Не критично (events.jsonl уже 17MB).
- **Schema drift** для downstream readers. Mitigation: все новые поля
  optional, default null.
- **Dependence на feat array shape**: some block sites вызываются до
  full feat compute. Mitigation: helper `_build_block_context` обрабатывает
  None gracefully.

## 7. Verification

- [ ] Spec written.
- [ ] `botlog.log_blocked` extended with new kwargs.
- [ ] `_build_block_context()` helper in monitor.py.
- [ ] Top-8 high-frequency block sites updated to pass context:
  - ml_proba_zone (75% of TON blocks)
  - trend_1h_chop
  - entry_score
  - ranker_hard_veto
  - trend_quality
  - impulse_guard
  - mode_range_quality
  - clone_signal_guard / open_cluster_cap
- [ ] `_backtest_blocked_breakdown.py` written.
- [ ] Smoke: TON-style retro audit with new fields.
- [ ] Schema test (no field type errors on null).
- [ ] After 7 d: real-world test on live block stream.

## 8. Follow-ups

- Aggregation: добавить `reason_code` breakdown в weekly TG digest.
- Historic backfill: replay last 30d через extended logger? (skip — too
  expensive; new data goes forward).
- Cross-reference: при каждом missed top-20 из skill — automatically
  pull last 24h blocks for that sym с full context.
