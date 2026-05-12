# `label_sustained_uptrend` — clean ground-truth replacement for `label_top20`

- **Slug:** `sustained-uptrend-label`
- **Status:** shipped 2026-05-07 v2.20.0 (label backfill + `EarlyCapture@sustained` metric)
- **Created:** 2026-05-07
- **Owner:** core
- **Related:** `2026-05-01-mode-audit.md` §H4, `signal-evaluator-integration-spec.md`
  Phase B, `metrics-framework-spec.md`.

---

## 1. Problem

`label_top20` ставит `1` на любой coin закрывшийся в top-20 EOD. Это
включает:
- pump+dump (взлёт +30 %, обвал к нулю — формально topN, но **untradable**)
- vertical pumps (+50 % за 5 мин — единственная свеча зелёная)
- сделки против большой просадки (closed +5 %, но был draw -15 %)

ML обучается на этих кейсах как на «успехах» — это и есть источник
blind-spot’ов и late entries.

## 2. Success metric

**Primary:** `EarlyCapture@sustained` (parallel north-star) показывает
**другую** картину чем `EarlyCapture@top20` — расхождение ≥ 0.05 на 30d
выборке (доказывает что labels не дублируют друг друга).

**Secondary:**
- ≥ 70 % `label_top20=1` rows получают `label_sustained_uptrend=0` →
  фильтр работает (отсеивает pump+dump из «winners»).
- After P0.2 retrain on this label: blind-spot block rate (proba<0.10
  AND label_sustained=1) drops ≥ 30 %.

## 3. Scope

### In scope
- Бэкфилл `label_sustained_uptrend` в `top_gainer_dataset.jsonl` (или
  параллельный файл `top_gainer_dataset_v2.jsonl` для shadow-сравнения).
- Используется существующий `files/zigzag_labeler.py` (уже extracted).
- Параллельная метрика `EarlyCapture@sustained` в
  `_compute_early_capture.py`.
- Использует cached klines из `history/` (заполненные daily backfill task’ом).

### Out of scope
- Замена `label_top20` в всех downstream readers — параллельный label,
  пользователь решает что использовать.
- Использование label в bandit reward (отдельная спека).
- Использование в ML training (P0.2 Phase 2 — отдельный trigger).

## 4. Behaviour / design

### Label formula

```python
def label_sustained_uptrend(intraday_bars,
                            swing_pct=4.0,          # min swing
                            max_drawdown_pct=2.0,   # max counter-move
                            min_duration_bars=4,
                            min_close_gain_pct=5.0  # full-day return
                           ) -> int:
    """1 iff:
      - any ZigZag uptrend covers ≥ X% of the day's bar range
      - max intratrend drawdown <= max_drawdown_pct
      - close_eod >= entry_low * (1 + min_close_gain_pct/100)
    """
    trends = detect_uptrends(intraday_bars,
                             swing_pct=swing_pct,
                             max_drawdown_pct=max_drawdown_pct,
                             min_duration_bars=min_duration_bars)
    if not trends: return 0
    # Take the largest gain trend
    best = max(trends, key=lambda t: t.gain_pct)
    if best.gain_pct < min_close_gain_pct:
        return 0
    # EOD close should still be above entry — defends vs late dump
    # (we can check if last bar's close > best.start_price * 1.02)
    last_close = intraday_bars[-1]["close"]
    if last_close < best.start_price * 1.02:  # less than +2% by EOD = dumped
        return 0
    return 1
```

### Backfill script

`files/_backfill_sustained_uptrend.py`:
1. Stream `top_gainer_dataset.jsonl` once.
2. Group rows by `(date, sym)`.
3. For each unique pair: load klines from `history/<sym>_15m.csv`
   (already cached daily by `CryptoBot_KlinesBackfill_Daily`).
4. Filter klines to that UTC date.
5. Call `label_sustained_uptrend()`.
6. Apply same value to all snapshots of (date, sym).
7. Write to `top_gainer_dataset_v2.jsonl` (atomic — separate file).
8. Print summary: counts of (`label_top20=1, label_sustained=1`),
   (`top20=1, sustained=0`), etc.

### EarlyCapture@sustained

Add to `_compute_early_capture.py` parallel metric:
- Same formula as current EC: coverage × capture × time_lead.
- But ground truth = `label_sustained_uptrend=1` instead of
  `label_top20=1`.
- Output both numbers in the daily aggregator.

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `SUSTAINED_LABEL_SWING_PCT` | 4.0 | ZigZag min swing |
| `SUSTAINED_LABEL_DRAWDOWN_PCT` | 2.0 | Max counter-move |
| `SUSTAINED_LABEL_MIN_DURATION_BARS` | 4 | Min trend length |
| `SUSTAINED_LABEL_MIN_CLOSE_GAIN_PCT` | 5.0 | Day's overall gain floor |

**Rollback:** `top_gainer_dataset_v2.jsonl` independent file. Delete
to revert. ML/metric scripts continue using original.

## 6. Risks

- **Klines coverage** — если `history/<sym>_15m.csv` не покрывает
  нужный день, label = 0 (graceful). Daily backfill task supplies most.
- **Threshold sensitivity** — 5 % EOD gain может быть слишком жёстким
  для quiet markets. Mitigation: spec defaults parameterised, tune
  after 30d data.
- **Label is binary** — теряем градацию (4 % vs 50 % gain — оба = 1).
  Future: add `sustained_uptrend_gain_pct` numeric field.

## 7. Verification

- [x] Spec written.
- [x] `files/zigzag_labeler.py` extracted (Phase B v2.10.0 done).
- [x] `_backfill_sustained_uptrend.py` script written and run on 30d data.
- [x] `top_gainer_dataset_v2.jsonl` produced; summary printed showing
      label distribution.
- [x] `_compute_early_capture.py` extended with `--sustained-label-file`.
- [ ] Compare `EarlyCapture@top20` vs `EarlyCapture@sustained` on 30d.

## 8. Follow-ups

- After 7d: decide if `label_sustained_uptrend` replaces `label_top20`
  as ML training target (P0.2 Phase 2 retrigger).
- Add label_sustained to bandit reward (separate spec).
- Numeric `sustained_uptrend_gain_pct` for granular signals.
