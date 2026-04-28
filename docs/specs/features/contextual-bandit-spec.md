# Contextual bandit (LinUCB) — entry + trail

- **Slug:** `contextual-bandit`
- **Status:** shipped (retroactive)
- **Created:** 2026-04-26
- **Owner:** core
- **Related:** `files/contextual_bandit.py`, `files/offline_rl.py`,
  `files/rl_headless_worker.py`, `CLAUDE.md` §4

---

## 1. Problem

Жёсткие if-правила перетянуты или недотянуты — нужен механизм, который
учится на исторических исходах (top-gainer / no top-gainer / fast reversal)
и сам выбирает: ENTER или SKIP, и какой trail-buffer применить.

## 2. Success metric

- `recall@top20 ≥ 95 %` стабильно.
- UCB-separation (Δ между ENTER/SKIP arm scores) растёт со временем.
- AUC top20 модели-кандидата ≥ 0.85 (текущее: 0.93 на 2026-04-17).

## 3. Scope

### In scope
- LinUCB (entry: 2 arms, trail: 5 arms).
- Контекст-вектор и вывод reward.
- Источники данных и offline-обучение.

### Out of scope
- ML signal classifier (см. `ml-signal-model-spec` — TODO).
- Top-gainer model (см. `top-gainer-model-spec`).

## 4. Behaviour / design

### Entry bandit

- Arms: `SKIP=0`, `ENTER=1`.
- Алгоритм: LinUCB, alpha=2.0 (агрессивная exploration).
- Контекст: `slope_pct, adx, rsi, vol_x, ml_proba, daily_range, macd_hist,
  btc_vs_ema50, is_bull_day, mode, tf` (one-hot для категорий).

### Trail bandit

- 5 arms: `very_tight (×0.70)`, `tight (×0.85)`, `default (×1.00)`,
  `wide (×1.20)`, `very_wide (×1.40)`. См. `TRAIL_ARMS` в
  `contextual_bandit.py`.
- Множитель применяется к base `trail_k` режима.
- На выходе бандит назначает `pos.trail_k` и `pos.max_hold_bars`.

### Reward (асимметричный)

| Ситуация | Reward |
|---|---|
| ENTER + top-20 gainer | +1.0 |
| SKIP + top-20 gainer | −0.8 (miss penalty) |
| SKIP + not top gainer | +0.10 |
| ENTER + not top gainer | −0.12 |
| ENTER + fast reversal (≤3 баров, P&L < −0.5 %) | −0.6 (planned, см. `anti-fast-reversal-spec`) |
| SKIP + fast reversal | +0.30 (planned) |

### Источники обучения

1. **Primary:** `top_gainer_dataset.jsonl` — все ~105 watchlist coins ×
   N daily snapshots.
2. **Secondary:** `critic_dataset.jsonl` — реальные сигналы бота с outcome.

Cap: `BANDIT_CRITIC_MAX_RECORDS = 25_000` (был 8000, упирался в потолок —
зафиксировано 2026-04-17).

### Pipeline

- Live: `monitor.py` зовёт `select_entry_profile` → выбор arm + trail.
- Offline: `rl_headless_worker.py` фоном раз в N минут зовёт
  `offline_rl.train_entry_bandit` поверх двух источников.
- EOD: `daily_learning.py` — полный цикл (snapshot → resolve → train → report).

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `BANDIT_ENABLED` | True | Главный switch. False = только гарды. |
| `BANDIT_ALPHA` | 2.0 | Exploration coefficient |
| `BANDIT_CRITIC_MAX_RECORDS` | 25_000 | Потолок critic-датасета для обучения |

**Rollback:** `BANDIT_ENABLED=False` → бот переходит на pre-bandit логику.

## 6. Risks

- **Bandit confusion при фиксированных floor’ах** (например, trail-min-buffer):
  бандит видит requested `trail_k`, не realised buffer. Mitigation в
  follow-up `trail-min-buffer-spec` §8.
- **Cold start** для нового режима — alpha=2.0 спасает за счёт exploration.

## 7. Verification

- `files/report_rl_daily.py` — daily metrics report (recall, UCB sep, updates).
- Учётные точки в `CLAUDE.md` §4 «Learning progress».

## 8. Follow-ups

- Добавить `proba_fast_reversal` в context vector (когда fast-reversal label
  готов).
- Добавить `realised_buffer_pct` для trail-bandit (см. `trail-min-buffer-spec`).
