# Metrics framework — measurement & optimisation roadmap

- **Slug:** `metrics-framework`
- **Status:** draft
- **Created:** 2026-04-26
- **Owner:** core
- **Related:** все sub-system specs; `docs/reports/2026-04-28-*`.

---

## 1. Problem

Текущие метрики (`recall@top20=100 %`, `AUC=0.989`, `UCB sep=0.154`,
`avg_pnl`) **не измеряют главную цель**: «earliest BUY на coin до основного
движа». Recall меряет *видимость* в bandit-датасете, AUC — модельную
классификацию, avg_pnl — постфактум-доходность. Между ними нет числа,
которое отвечает на вопрос: **«насколько рано и качественно мы
обслуживаем подписчика TG-канала».**

## 2. Цели декомпозиции

Главная цель распадается на 4 sub-objectives, каждый со своими метриками:

| # | Слой | Вопрос | Anti-pattern если игнорировать |
|---|------|--------|-------------------------------|
| **C** | Coverage | Видим ли мы top-20 победителей? | silent miss, blind-spot |
| **E** | Earliness | Реагируем ли ДО основного движа? | покупка на хаях |
| **Q** | Quality | Сигнал actionable для подписчика? | fast reversal, whipsaw |
| **D** | Discrimination | Шумим ли на не-победителях? | spam в TG, false BUY |

---

## 3. North-star metric

### `EarlyCapture@top20` (weekly)

```
EarlyCapture@top20 = mean over EOD top-20 winners of:
    coverage_flag   ×   capture_ratio   ×   time_lead_score

where:
  coverage_flag    = 1 if bot emitted entry, else 0
  capture_ratio    = (eod_high − entry_price) / (eod_high − day_open)  ∈ [0, 1]
  time_lead_score  = clamp((entry_time_UTC_hours_from_open) / 24, 0, 1)
                     reversed: earlier = higher
                     formula: 1 − (entry_hour_after_open / 24)
```

Range: 0–1 per winner; среднее — единое число для всей недели.

**Текущее baseline (estimate, ещё не измерено):**
- coverage ≈ 0.80
- capture_ratio: TBD (нужен P8 backtest)
- time_lead: для 47 entered winners 26 % после 12 UTC → среднее ~0.6
- **Estimate**: 0.8 × 0.5 × 0.6 ≈ **0.24**

**Цель Q3 2026:** 0.40 (+67 %). Достижимо через P5 (coverage), P3-redesign
(earliness), P2 (quality).

---

## 4. Метрики по слоям

### Layer C — Coverage

#### C1. `coverage_funnel_top20`
**Формула:** `entries_on_top20_winners / total_top20_winner_days`,
с поправкой «бот держит позицию» (= считается как entered).
**Источник:** `top_gainer_dataset.jsonl` (label_top20=1) + `bot_events.jsonl`.
**Cadence:** weekly.
**Скрипт:** `_backtest_top20_coverage_funnel.py` (готов).
**Текущее:** 74.6 % сырое → ~80 % с поправкой.
**Цель:** ≥ 90 %.
**Lever:** P5 (ML blind-spots), P3 (pre-move screener).

#### C2. `silent_miss_rate`
**Формула:** `top20_winners_with_zero_events / total_top20_winner_days`.
**Источник:** те же.
**Cadence:** daily.
**Текущее:** 7.9 % сырое → ~3 % с поправкой на holding.
**Цель:** ≤ 1 %.
**Lever:** P5, audit early-stage scoring.

#### C3. `model_recall@top20` (existing)
**Формула:** см. `report_rl_daily.py`.
**Текущее:** 100 % с 2026-04-10.
**Lever:** обучение модели (`daily_learning.py`).
**Caveat:** меряет видимость в bandit-обучении, не выданные сигналы.
Не путать с C1.

---

### Layer E — Earliness

#### E1. `time_to_signal` (TTS)
**Формула:** часов между «main move start» и `entry_event` для top-20 winners.
- `main move start` = первый 1h close, превышающий `prev_close + 1.5 × ATR_24h`.
**Источник:** klines (нужен fetch) + `bot_events.jsonl`.
**Cadence:** weekly.
**Скрипт:** `_backtest_time_to_signal.py` (TBD).
**Текущее:** не измерено.
**Цель:** median ≤ 30 мин (т.е. на следующем 1h close после старта).
**Lever:** P3 (pre-move screener), P4 (impulse_guard relaxation),
P5 (ML blind-spots).

#### E2. `realized_capture_ratio`
**Формула:** `(eod_high − entry_price) / (eod_high − day_open)` ∈ [0, 1].
**Источник:** `bot_events.jsonl` (entry_price) + klines (eod_high, day_open).
**Cadence:** daily.
**Скрипт:** `_backtest_capture_ratio.py` (TBD).
**Текущее:** не измерено.
**Цель:** mean ≥ 0.5 (ловим минимум половину дневного движа после entry).
**Lever:** все P-fixes; индикатор «насколько рано».

#### E3. `pre_move_alert_lead_time` (если P3 имплементирован)
**Формула:** часов между intraday-heads-up trigger и actual entry.
**Источник:** snapshots `top_gainer_dataset.jsonl` + bot events.
**Скрипт:** `_validate_p3_premove_screener.py` (готов как simulation).
**Текущее:** simulation: median +4 ч.
**Цель:** ≥ 2 ч median sustained на live data.

---

### Layer Q — Quality

#### Q1. `fast_reversal_rate` (per mode)
**Формула:** `paired_trades_with_bars≤3_AND_pnl≤-0.3% / total_paired_trades`.
**Источник:** `bot_events.jsonl` paired (entry, exit).
**Cadence:** weekly.
**Скрипт:** `_backtest_fast_reversal_by_mode.py` (готов).
**Текущее:** overall 15 %; alignment/1h 31 %, impulse/1h 20 %.
**Цель:** overall ≤ 8 %, ни один режим не выше 15 %.
**Lever:** P2 (anti-fast-reversal label/guard), P7 (kill breakout/15m).

#### Q2. `whipsaw_rate` (per mode)
**Формула:** `paired_trades_exiting_via_trail_in_≤5_bars_with_pnl∈[−1.5%,0%]
            / total_paired_trades`.
**Источник:** те же.
**Cadence:** weekly.
**Текущее:** не сегментировано отдельно от FR; нужен дополнительный grep.
**Цель:** ≤ 5 % overall.
**Lever:** trail-min-buffer (shipped), bandit re-training.

#### Q3. `fast_reversal_drag`
**Формула:** `sum(pnl) over FR_v1 trades / sum(pnl) over all losing trades`.
**Источник:** те же.
**Cadence:** weekly.
**Текущее:** **28 %** (FR забирает 28 % всей просадки).
**Цель:** ≤ 12 %.
**Lever:** P2.

#### Q4. `actionable_signal_rate`
**Формула:** `1 − fast_reversal_rate` (per Q1).
**Цель:** ≥ 92 %.

---

### Layer D — Discrimination

#### D1. `signal_precision_top20`
**Формула:** `entries_on_top20_winners / total_entries_in_period`.
**Источник:** `bot_events.jsonl` + `top_gainer_dataset.jsonl`.
**Cadence:** weekly.
**Скрипт:** `_backtest_signal_precision.py` (TBD; легко из funnel logic).
**Текущее:** не измерено отдельно. Из 30 d данных:
~628 entries vs ~140 top-20 winner-days → precision ≈ 22 %
(но один winner может получить multiple entries — нужно дедуп).
**Цель:** ≥ 35 %.
**Lever:** ужесточение гардов (целевой sweep), bandit.

#### D2. `tg_message_rate`
**Формула:** entries в TG-канале в день.
**Источник:** `bot_events.jsonl` event=entry.
**Cadence:** daily.
**Текущее:** 628 / 30 ≈ 21 entry/day. Высоко.
**Цель:** 8–12 entry/day (читаемый канал).
**Lever:** D1 fixes.

#### D3. `bandit_ucb_separation` (existing)
**Формула:** см. `report_rl_daily.py`.
**Текущее:** 0.154.
**Цель:** > 0.20 sustained.
**Lever:** обучение, expanded context vector.

---

## 5. Сводная dashboard-схема

```
NORTH-STAR        EarlyCapture@top20        target 0.40 (now ~0.24)
       │
       ├─ Coverage   C1 funnel       80 % → 90 %
       │             C2 silent_miss   3 % → 1 %
       │             C3 recall@20   100 % (existing, not main lever)
       │
       ├─ Earliness  E1 TTS         (?) → ≤ 30 m
       │             E2 capture     (?) → ≥ 0.50
       │             E3 lead_time   (sim 4h, validate live)
       │
       ├─ Quality    Q1 FR_rate     15 % → 8 %
       │             Q2 whipsaw     (?) → 5 %
       │             Q3 FR_drag     28 % → 12 %
       │
       └─ Discrim.   D1 precision   22 % → 35 %
                     D2 msg_rate    21/d → 10/d
                     D3 UCB sep     0.154 → 0.20
```

---

## 6. Implementation plan

### Phase 1 — Instrumentation (что измеряем)

| Метрика | Скрипт | Status |
|---------|--------|--------|
| C1 funnel | `_backtest_top20_coverage_funnel.py` | ✅ ready |
| C2 silent | derive from C1 | ✅ ready |
| Q1 FR rate | `_backtest_fast_reversal_by_mode.py` | ✅ ready |
| Q3 FR drag | derive from Q1 | ✅ ready |
| E3 lead_time sim | `_validate_p3_premove_screener.py` | ✅ ready |
| **E1 TTS** | `_backtest_time_to_signal.py` | ❌ TBD |
| **E2 capture** | `_backtest_capture_ratio.py` | ❌ TBD |
| **D1 precision** | `_backtest_signal_precision.py` | ❌ TBD |
| **Q2 whipsaw** | extend FR script с trail-only фильтром | ❌ TBD |
| **NS** north-star | `_compute_early_capture.py` aggregator | ❌ TBD |

### Phase 2 — Daily metrics report

Новый скрипт `files/report_metrics_daily.py`:
1. Запускается в конце `daily_learning.py`.
2. Зовёт все 13 backtest-скриптов на «вчерашние сутки».
3. Append’ит JSON-row в `.runtime/metrics_daily.jsonl`.
4. Отправляет компактный summary в TG (только если изменения > X σ).

### Phase 3 — Optimisation loops (по слоям)

Каждое улучшение из `2026-04-28-improvements-validated.md` имеет
явные target metrics:

| Improvement | Влияет на | Acceptance |
|-------------|-----------|-----------|
| P5 ML blindspot | C1, C2, C3 | C1 ≥ 88 %, C2 ≤ 1.5 %, AUC ≥ 0.97 |
| P2 anti-FR | Q1, Q3, Q4 | Q1 ≤ 8 %, Q3 ≤ 12 %, recall не падает |
| P7 disable breakout/15m | Q1, D2 | Q1 mode-level → 0; D2 −2/d |
| P3-redesign screener | C1, E1, E3 | E3 sustained ≥ 2 h, FP/day ≤ 30 |
| P4 impulse_guard 15m | C1 | C1 +2 п.п. без проседания Q1 |
| P6 EOD health alert | (operational) | 0 missed snapshots / month |
| trail-min-buffer | Q2 | Q2 на impulse_speed/strong_trend ≤ 5 % |

---

## 7. Optimisation methodology

Стандартный rollout-цикл для каждой инициативы:

```
1. Define target metric & acceptance threshold (см. таблицу выше)
2. Implement change behind config flag (ROLLOUT_FLAG = True/False)
3. Backtest на 30 d historical: target metric улучшается, anti-metrics
   не деградируют > 2 σ
4. Shadow A/B на live data (если применимо): 7 d, два бота с разными
   configs, dump в shadow_report.json
5. Promote: flip flag, watch metric daily
6. After 7 d live: compare vs pre-rollout baseline; rollback if any
   acceptance threshold нарушен.
```

### Anti-metrics (не должны деградировать)

При оптимизации одной метрики, отслеживаем 3 anti-metrics:

- **`max_drawdown_per_signal`** — не растёт > 0.5 σ.
- **`recall@top20`** — не падает > 2 п.п.
- **`tg_message_rate`** — не растёт > 30 % (защита от спама).

---

## 8. Verification (для самой framework-спеки)

- [ ] Скрипты Phase 1 (5 TBD) написаны и запускаются.
- [ ] `report_metrics_daily.py` собирает все 13 метрик в один JSON-row.
- [ ] Baseline-snapshot 7 d (`docs/reports/<date>-metrics-baseline.md`).
- [ ] Каждая будущая спека (P5, P2, P7, …) ссылается на target metrics
      из этой framework.
- [ ] Через 30 d — first review of north-star: достигнут ли 0.30+?

---

## 9. Follow-ups

- Ввести **per-mode dashboard** — каждый режим имеет собственный
  набор C/E/Q/D. Сейчас агрегация overall сглаживает разрывы.
- Расширить anti-metrics на **risk-adjusted** (Sharpe-style):
  `mean_pnl / std_pnl × √n`.
- Добавить **calibration plot** для top_gainer_prob — predicted
  vs realized rate. Сейчас AUC не показывает calibration.
- Долгосрочно: **profit-per-message** для TG-канала
  (мера UX-полезности).
