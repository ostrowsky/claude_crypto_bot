# PEAK RISK shadow detector (P1.2)

- **Slug:** `peak-risk-shadow`
- **Status:** shipped 2026-05-07 v2.20.0 (shadow-only — logs events, no exits triggered)
- **Created:** 2026-05-07
- **Owner:** core
- **Related:** `2026-05-07 ICP/TON cases` (single SELL was always too late
  or too early; need lifecycle alerts), analysis doc §7.

---

## 1. Problem

Бот сейчас имеет ровно **два состояния** для открытой позиции: HOLD
и SELL. Когда SELL срабатывает — поздно (~70 % от EOD high уже
сожжено) или рано (RSI div WEAK на +0.24 % pnl).

Live cases:
- TON Trade #1: SELL по RSI 89 на +25 % pnl, цена пошла +45 % → left 20 %
- TON Trade #2: WEAK div SELL на +0.24 %, цена пошла +16 % → left 16 %
- ICPUSDT 2026-05-05: WEAK div SELL на +2.5 %, цена пошла +6.8 % → left 4.3 %

Pattern: бот вышел на **локальном retracement**, не на peak-risk.

PEAK RISK ≠ predict the top. PEAK RISK = **detect overextension zone**
where probability of significant pullback rises sharply. Outputs
**alert/score**, не SELL.

## 2. Success metric

**Phase 1 (shadow, this commit):**
- Detector работает, события пишутся в `bot_events.jsonl` как
  `peak_risk_shadow`.
- 7d shadow gathers ≥ 20 events для acceptance review.
- Каждый event имеет components breakdown (RSI, edge, MACD decel, etc).

**Phase 2 (TG alert + tighter trail, separate spec):**
- На каждом peak_risk shadow event с score ≥ N: TG message «⚠️ PEAK RISK X».
- Tighter trail (k *= 0.7) при score ≥ HIGH_THRESHOLD.

**Acceptance criteria for Phase 2 promotion:**
- Среди shadow events с score ≥ 70: ≥ 60 % предшествуют peak в течение
  N баров (validation against ZigZag-detected peaks).
- False-positive rate (peak_risk fires but trend continues > 5 %)
  ≤ 30 %.

## 3. Scope

### In scope
- Helper `_peak_risk_score(pos, feat, i, pnl_pct)` returns 0-100.
- Components (each 0-25 max):
  - RSI overextension: `(RSI - 75) * 2` clamped 0-25
  - Price edge vs EMA20: `(edge_pct - 10) * 1` clamped 0-25
  - MACD deceleration: 25 if hist[i] < hist[i-1] AND both > 0 else 0
  - Profit threshold: 25 if pnl_pct >= 5 % else linear from 0
- Threshold for shadow log: score >= 50.
- Per-position dedup: log once per score-bucket transition (50/70/90).
- Aggregator script.

### Out of scope (Phase 2)
- TG alerts.
- Trail tightening on peak_risk.
- PROFIT LOCK partial-position alert.
- Volume climax detection (needs higher-frequency data).
- Upper wick expansion (geometric heuristic).

## 4. Behaviour / design

### Score formula

```python
def _peak_risk_score(pos, feat, i, pnl_pct):
    if pnl_pct < 0:
        return 0  # not in profit, no peak risk

    rsi = feat["rsi"][i]
    rsi_score = max(0, min(25, (rsi - 75.0) * 2.0)) if np.isfinite(rsi) else 0

    cp = feat["close"][i]; ef = feat["ema_fast"][i]
    edge_pct = (cp / ef - 1.0) * 100 if (cp and ef and ef > 0) else 0
    edge_score = max(0, min(25, edge_pct - 5.0))  # 5..30% edge → 0..25 score

    mh = feat["macd_hist"]
    macd_score = 0
    if i >= 1 and np.isfinite(mh[i]) and np.isfinite(mh[i-1]):
        if mh[i] > 0 and mh[i] < mh[i-1]:
            macd_score = 25  # decelerating from positive

    if pnl_pct >= 5.0:
        profit_score = 25
    elif pnl_pct >= 1.0:
        profit_score = 25 * (pnl_pct - 1.0) / 4.0  # 1..5% → 0..25
    else:
        profit_score = 0

    return rsi_score + edge_score + macd_score + profit_score
```

### Shadow logging

In `monitor.py`, in the open-position polling section (after exit check):

```python
if pos and getattr(config, "PEAK_RISK_SHADOW_ENABLED", True):
    pr_score = _peak_risk_score(pos, feat, i, current_pnl)
    if pr_score >= getattr(config, "PEAK_RISK_SHADOW_THRESHOLD", 50):
        # Per-bucket dedup to avoid spam
        bucket = (pr_score // 20) * 20
        last_bucket = getattr(pos, "_peak_risk_last_bucket", -1)
        if bucket > last_bucket:
            pos._peak_risk_last_bucket = bucket
            botlog.log_peak_risk_shadow(
                sym, tf, close_now, pos.entry_price,
                pnl_pct=current_pnl, score=pr_score,
                rsi=feat["rsi"][i], price_edge_pct=edge_pct,
                # ... all components
            )
```

### Aggregator

`files/_backtest_peak_risk.py`:
- Count shadow events per day, by score bucket.
- For each event: did price actually peak within N bars? (validates).
- Per-mode breakdown.
- Export Phase 2 acceptance numbers.

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `PEAK_RISK_SHADOW_ENABLED` | True | Master switch (shadow only) |
| `PEAK_RISK_SHADOW_THRESHOLD` | 50 | Min score to log event |
| `PEAK_RISK_RSI_FLOOR` | 75 | RSI level where score starts |
| `PEAK_RISK_EDGE_FLOOR_PCT` | 5.0 | Price-edge level where score starts |

**Rollback:** `PEAK_RISK_SHADOW_ENABLED = False`. Pure observability,
no behaviour change.

## 6. Risks

- **Score formula arbitrary** — components weighted equally for v1.
  Mitigation: 7d shadow shows distribution; tune in Phase 2 spec.
- **JSONL volume** — one event per bucket transition per position.
  Estimate: ~5-15 events/day × 4-6 components/event ≈ negligible.
- **Per-position state** — `_peak_risk_last_bucket` attr added to Position
  (in-memory only). Reset on bot restart, OK because all open positions
  re-evaluate from current bar.

## 7. Verification

- [x] Spec written.
- [x] `_peak_risk_score()` helper implemented.
- [x] Shadow log call in monitoring poll.
- [x] `_backtest_peak_risk.py` aggregator written.
- [ ] Smoke: artificially inject high-RSI position, verify event logged.
- [ ] 7d live: count events, validate against ZigZag peaks.

## 8. Follow-ups (Phase 2)

- TG alerts: «⚠️ PEAK RISK BTCUSDT score=72 (RSI 88, edge +18%, MACD↓)»
- PROFIT LOCK: when score ≥ 70 AND pnl ≥ 5 %, send «🔒 protect 50%»
- Trail tightening: trail_k *= 0.7 when score ≥ 70 (separate spec)
- Numeric calibration via 7d shadow distribution
