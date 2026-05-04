# Signal-efficiency-evaluator integration

- **Slug:** `signal-evaluator-integration`
- **Status:** **Phase A shipped** (skill + wrapper); Phase B-D draft
- **Created:** 2026-05-04
- **Owner:** core
- **Related:** `2026-05-01-mode-audit.md` §H4 (sustained-uptrend label),
  `ex1-realized-potential-spec.md`, `metrics-framework-spec.md`.

---

## 1. Problem

Skill `signal-efficiency-evaluator` (Claude Skill, `skills/`)
делает то, чего не было в текущем metric-стеке:
- **Hindsight ZigZag-labeler** для «sustainable uptrends» — это
  именно тот ground truth, который H4 из mode-audit спекулировал
  как замену `label_top20`.
- **Per-trade verdicts** (verdict ∈ {optimal, late_entry,
  premature_exit, …}) с конкретными timestamps, prices,
  recommendations.
- **Alpha vs buy-and-hold** — единственная честная метрика
  «стоит ли вообще запускать бота».

Skill **не дублирует** `rl_critic.py` / `rl_optimizer.py` /
`ml_signal_model.py`. Он produces **judgment**, не signals.

## 2. Success metrics

**Phase A acceptance:**
- Skill запускается на 7 d данных без ошибок.
- Smoke-test уже пройден (3 d, 4 coin: 13 trends, 4 caught,
  alpha +4.64 %).
- `evaluation_output/report.{json,md}` пишутся валидно.

**Phase B-D (после rollout):**
- ZigZag-labeler оживляет H4: `label_sustained_uptrend` в
  `top_gainer_dataset.jsonl`.
- New north-star variant `EarlyCapture@sustained` рядом с
  существующим `EarlyCapture@top20`.
- Per-trade verdicts становятся доп. reward-каналом для bandit.

## 3. Scope

### Phase A — Install + adapter (shipped)
- Skill installed at `skills/signal-efficiency-evaluator/`.
- Wrapper `files/_run_signal_evaluator.py`:
  - Translates our `bot_events.jsonl` schema (entry/exit/sym)
    → skill schema (BUY/SELL/symbol).
  - Pre-fetches klines via existing aiohttp `fetch_klines` →
    `history/<sym>_<tf>.csv` (skill needs `requests`, мы используем aiohttp).
  - Passes through schema-map and symbol-filter.
  - Defaults tuned for 15m crypto: swing 4 %, drawdown 2 %, min 4 bars.
- UTF-8 fixes в самом skill-script (Windows cp1251 default).

### Phase B — ZigZag labeler as bot module
- Извлечь `label_uptrends_zigzag()` из `evaluate_signals.py` в
  `files/zigzag_labeler.py` как stand-alone module.
- Backfill `label_sustained_uptrend` в `top_gainer_dataset.jsonl`
  (с per-symbol intraday klines).
- Параллельная метрика в `_compute_early_capture.py`:
  `EarlyCapture@sustained` — ground truth «coin был в реальном
  устойчивом uptrend» вместо «попал в top-20 по EOD».

### Phase C — Per-trade verdicts as bandit reward channel
- При каждом completed trade добавлять reward-component:
  - verdict `optimal` → +1.0 supplementary reward
  - verdict `late_entry_late_exit` → +0.2
  - verdict `losing_trade` или `optimal_entry_premature_exit`
    → −0.5
- Записывать в `.runtime/eval_feedback.jsonl` (отдельный от
  bandit-state `rl_memory.jsonl`).
- `offline_rl.train_entry_bandit` читает оба файла.

### Phase D — EX1 potential refinement
- Заменить `max(eod_return, tg_return_4h, tg_since_open)` proxy
  на `matched_trend.gain_pct` из ZigZag-labeler.
- Это даёт **истинный** intraday-high (не proxy через snapshots).
- Ожидание: текущий EX1 median +0.001 → reveal истинную capture
  без proxy-bias.

### Out of scope
- Live entry/exit decisions от skill (skill =
  hindsight evaluation, не realtime).
- Замена `ml_signal_model` / `ml_candidate_ranker` (skill не
  делает predictions).
- Auto-tuning thresholds (отдельная спека Phase E).

## 4. Behaviour / design

### Phase A wiring (shipped)

```
Manual / scheduled run:
  pyembed/python.exe files/_run_signal_evaluator.py \
    --window-days 7 --timeframe 15m

Wrapper does:
  1. Read files/bot_events.jsonl, filter window
  2. Translate schema → .runtime/_skill_events_translated.jsonl
  3. Identify symbols seen, pre-fetch klines via aiohttp
     → history/<sym>_<tf>.csv
  4. Invoke skills/.../evaluate_signals.py with our schema
  5. Outputs: evaluation_output/{report.json, report.md}
```

### Phase B sketch

```python
# files/zigzag_labeler.py
def label_uptrends_zigzag(bars, swing_threshold_pct=4.0,
                          max_intratrend_drawdown_pct=2.0,
                          min_trend_duration_bars=4):
    """Extracted from skill. Returns list of UpTrend records."""
    ...

# Backfill script:
# files/_backfill_sustained_uptrend.py
#   For each (date, sym) in top_gainer_dataset, fetch intraday klines,
#   run labeler, set label_sustained_uptrend = 1 if any trend covers
#   the day with gain >= 4%.
```

### Phase C reward structure

```python
# В offline_rl.train_entry_bandit, additional reward layer:
existing_reward = ...   # current top20-based formula
verdict_reward  = {     # from .runtime/eval_feedback.jsonl
    "optimal":                       +1.0,
    "late_entry_optimal_exit":       +0.5,
    "optimal_entry_late_exit":       +0.5,
    "late_entry_late_exit":          +0.2,
    "optimal_entry_premature_exit":  -0.3,
    "losing_trade":                  -0.5,
}.get(verdict, 0.0)
final_reward = 0.7 * existing_reward + 0.3 * verdict_reward
```

Веса 0.7/0.3 — старт, далее tune через A/B.

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `SIGNAL_EVALUATOR_ENABLED` | True | Phase A wrapper доступен |
| `SUSTAINED_UPTREND_LABEL_ENABLED` | False | Phase B: вычислять label_sustained |
| `BANDIT_VERDICT_REWARD_WEIGHT` | 0.0 | Phase C: вес verdict-rewardа (0 = выкл) |
| `EX1_USE_ZIGZAG_POTENTIAL` | False | Phase D: ZigZag вместо proxy |

**Rollback per phase:** flag = False / 0.0. Skill артефакты на диске
не мешают.

## 6. Risks

- **Threshold sensitivity.** ZigZag thresholds (4 % swing / 2 %
  drawdown / 4 bars) откалиброваны на 15m crypto-majors, но наши
  watchlist-альты могут иметь другую шумность. Sweep потребуется
  при Phase B rollout.
- **Hindsight bias** — verdict оптимальности ВСЕГДА после факта.
  Использовать ТОЛЬКО для evaluation/reward, **никогда** для
  online entry-решений.
- **Reward-double-count** — Phase C: skill грейдит trade ПОСЛЕ
  exit, но bandit и так получает reward на closing event.
  Mitigation: explicit weight (default 0.0), отдельный JSONL.
- **Compute cost** — Phase B backfill всех ~105 watchlist coin ×
  N daily snapshots требует ~K klines fetches. Один раз → cached.
- **Schema drift** — наш `bot_events.jsonl` имеет разные variants
  (`event:"entry"` vs legacy `signal_type`). Wrapper toggles на
  current schema; future schema changes → wrapper update.

## 7. Verification

### Phase A (shipped)
- [x] Skill files installed at `skills/signal-efficiency-evaluator/`.
- [x] UTF-8 patches applied to skill (cp1251 issue on Windows).
- [x] Wrapper translates schema correctly.
- [x] Smoke-test on 3 d × 4 coin: skill exit code 0, report.md
  generated.
- [x] Output sane: `13 trends found, miss rate 69 %, capture
  ratio 6.3 %, alpha +4.64 %` — численно сходится с нашими raw
  метриками.

### Phase B (TBD)
- [ ] Extract `label_uptrends_zigzag()` to `files/zigzag_labeler.py`.
- [ ] Backfill 30 d of `label_sustained_uptrend`.
- [ ] Compare `recall@top20` vs `recall@sustained` overall.
- [ ] Decide: replace north-star or keep both.

### Phase C (TBD)
- [ ] Add eval-feedback record format documented в
  `references/feedback_format.md`.
- [ ] Bandit reward A/B: train two bandits with different weights
  (0.0 vs 0.3).
- [ ] After 7 d: compare UCB sep on each.

### Phase D (TBD)
- [ ] Modify `_backtest_ex1_realized_potential.py` to use ZigZag
  potential when `EX1_USE_ZIGZAG_POTENTIAL=True`.
- [ ] Compare new EX1 median vs current proxy-based.

## 8. Follow-ups

- Scheduled task `CryptoBot_SignalEvaluator_Weekly` cron Sundays
  03:30 local — auto-run skill on last 7 d, push report to TG.
- Threshold sweep по timeframe (15m: 4/2/4 → 1h: 7/3/4 → 1d: 12/5/2).
- Per-mode evaluation: фильтр по `signal_mode` в wrapper, отдельный
  report per mode.
- Cross-correlation skill verdicts ↔ existing critic_dataset labels
  (sanity check).
