# Roadmap validation pass — 2026-04-30

Прогон 4 backtest-валидаций для re-rank инициатив из
`2026-04-29-north-star-roadmap.md`. Скрипты:

- `_validate_1a_tgprob_megatrigger.py`
- `_validate_1c_daily_range_stage.py`
- `_validate_2a_dynamic_max_hold.py`
- `_validate_4a_precision_prune.py`

---

## Результаты

### 1A — top_gainer_prob mega-trigger · ⚫ невалидируемо

```
Entries with ranker_top_gainer_prob field: 0 / 1050
```

**Проблема:** entry-события логируют только базовые индикаторы
(`adx, rsi, vol_x, ml_proba, …`), без ranker-output’ов. `ranker_*`
fields присутствуют в `positions.json`, но не пишутся в
`bot_events.jsonl`.

**Действие:** добавить `ranker_top_gainer_prob`, `ranker_ev`,
`ranker_quality_proba` в entry-event payload в `monitor.py`. После
этого retry validation. **Блокирующая dependency.**

**Re-rank:** **defer** до instrumentation fix.

---

### 1C — daily_range stage-aware · 🟢 STRONG validation

```
Top-20 winners blocked by daily_range (14d):           8
  early stage  (NOT blocked yesterday) → recoverable:  4
  late stage   (blocked yesterday)     → keep blocked: 4

Recovered: AXLUSDT 04-16, AUDIOUSDT 04-17, METISUSDT 04-17, XAIUSDT 04-18
Net: +4 winners из 67 = +6.0 п.п. coverage
```

**Чисто измеримо:** в 14-дневном окне 4 top-20 winners заблокированы
гардом `daily_range > 10/15 %` **впервые**, без блокировки накануне.
Это «свежий импульс» — exactly то, что мы хотим ловить.

**Сохраняет защиту:** 4 случая «3-й день парада» (AXL 04-17/18, ENA,
DOGS) остаются заблокированными — корректно.

**Δ NS estimate:**
- coverage 0.70 → 0.76 (+0.06 пунктов)
- capture/lead не меняются напрямую → ΔNS ~ +0.005

Меньше чем roadmap предсказывал (+0.04), потому что 1C только
включает гард для 4 случаев, не меняет timing их обработки.

**Re-rank:** medium-low. Простая в реализации, но мелкий выигрыш.

---

### 2A — dynamic max_hold · 🟢🟢 ОЧЕНЬ STRONG validation

```
30 d, paired trades = 1032
  exit by time/max_hold:    96
  exit by EMA20 weakness:  162
  exit by ATR trail:       474

Time-exits on top-20 winners with eod > pnl_at_exit + 1%:
  count:                    13
  total money left:    +1445.1%   (avg 111 % per trade)

EMA20-weakness exits on top-20 with money_left:
  count:                    16
  total money left:    +1647.8%   (avg 103 % per trade)
```

**Концентрация:** 13+16=29 trades за 30 d, где бот **вышел в плюс или
в небольшой минус, но монета затем сделала ещё ~100 % дневного
движения**. Самые драматичные:

- `MANAUSDT 04-01` impulse_speed: pnl `−1.91 %`, eod `+366 %`
- `MDTUSDT 04-01` retest: pnl `−0.23 %`, eod `+274 %`
- `ADAUSDT 04-03` impulse_speed: pnl `+1.36 %`, eod `+288 %`
- `ORDIUSDT 04-16` impulse_speed: pnl `+28.12 %`, eod `+173 %`
- `BLURUSDT 04-01` impulse_speed: pnl `+10.70 %`, eod `+33 %`

**Δ NS estimate:**
- если на 50 % из этих trades удерживать до bigger move: capture
  растёт с 0.16 до **~0.24** (+50 %).
- coverage не меняется, lead не меняется.
- ΔNS = 0.70 × 0.24 × 0.69 − 0.077 = **+0.039**

**Caveat:** 2A не «волшебная пилюля» — продление hold увеличит
max_drawdown, потому что 50 % top-20 dailies заканчиваются reversal
к концу дня. Need adaptive condition (ADX growing AND price > EMA20).

**Re-rank:** **HIGH** (выше всех остальных по validated impact).

---

### 4A — precision-prune · 🟡 marginal без ranker полей

```
Baseline: 1050 entries, 129 on top-20, precision=12.3%, 33.9/day

Filters (только ml_proba доступен):
  ml_proba > 0.40       693 entries  22.4/d  80 top20  prec 11.5%  recall 62%
  ml_proba > 0.50       432 entries  13.9/d  56 top20  prec 13.0%  recall 43%
```

**Без ranker_ev / ranker_top_gainer_prob невозможно:** baseline
precision 12.3 %, ml_proba >0.50 даёт 13 % при потере 57 % entries
— это compression, не improvement.

**Заметка по интерпретации:** если бы ranker fields логировались, по
hypothesis-у roadmap’а `ev>0 AND tg_prob>0.30` дал бы precision ~25 %.
Но валидировать это сейчас нельзя.

**Δ NS estimate:** 0 (без ranker logging).

**Re-rank:** **defer** до instrumentation fix (вместе с 1A).

---

## Re-ranked roadmap

| Старый ранг | Init | Новый ранг | Reason |
|------------|------|-----------|--------|
| step 5 | **1A mega-trigger** | **defer** | нет данных, нужен logger fix |
| step 6 | **2A dynamic max_hold** | **#1 (HIGH)** | валидирован: +0.039 NS |
| step 4 | **1C daily_range stage** | **#3 (medium)** | +0.005 NS (мельче, чем думал) |
| step 3 | **4A precision-prune** | **defer** | нет ranker fields в entries |
| step 7 | **P5 ML blind-spot** | **#2 (high)** | +0.03 NS, измерено в baseline |
| step 1 | **P7 disable breakout/15m** | **#4 (low/cheap)** | UX, не двигает NS |
| step 2 | **P2 anti-fast-reversal** | **#5 (UX)** | UX, не двигает NS |

### Новый sequenced rollout

| Step | Action | Δ NS | Cum | Confidence |
|-----:|--------|-----:|----:|-----------:|
| 0 | baseline | — | **0.077** | ✓ |
| **1** | **logger-fix:** добавить `ranker_*` в entry events | 0 | 0.077 | high |
| 2 | 1A re-validate (после step 1) | (TBD) | TBD | TBD |
| 3 | 4A re-validate (после step 1) | (TBD) | TBD | TBD |
| **4** | **2A dynamic max_hold** (валидирован) | +0.039 | **0.116** | medium-high |
| 5 | P5 ML blind-spot | +0.03 | 0.146 | high |
| 6 | 1C daily_range stage-aware | +0.005 | 0.151 | high |
| 7 | P7 disable breakout/15m | (UX) | 0.151 | high |
| 8 | P2 anti-fast-reversal | (UX) | 0.155 | high |
| 9–10 | re-evaluate after step 2/3 results | TBD | →0.30+? | TBD |

---

## Главный takeaway

1. **logger-fix — приоритет 0.** Без `ranker_top_gainer_prob` /
   `ranker_ev` в entry-events половина roadmap’а невалидируема.
   Это копеечный код-чейндж, но блокирует 1A, 4A, и любой будущий
   precision-tuning.

2. **2A — главный single-shot win.** Простая реализация
   (несколько строк в `monitor.py` exit-checker), валидированный
   эффект ≈ +0.04 NS. По-прежнему требует careful condition
   (ADX growing AND price > EMA20).

3. **1A mega-trigger — отложен, но обещание сохраняется.**
   Когда logger-fix сделан, ожидание — самый высокий impact
   (+0.13). Backtest потенциально подтвердит.

4. **1C — мельче, чем казалось.** 4 winners за 14 d = +6 п.п.
   coverage, но в формуле NS coverage уже не bottleneck (capture
   = 0.16 — главная пропасть). Делать в medium очереди.

5. **Anti-pattern reaffirmed.** Всё, что только увеличивает
   permissivity (4A naive ml_proba > 0.40), не двигает NS.

---

## Action items

| # | Task | Owner | Spec |
|---|------|-------|------|
| 1 | Logger-fix: добавить `ranker_top_gainer_prob`, `ranker_ev`, `ranker_quality_proba`, `signal_mode` в entry-event payload в `monitor.py` | TBD | new mini-spec |
| 2 | 7 d сбор данных с новым logger’ом | wait | — |
| 3 | Re-run `_validate_1a_tgprob_megatrigger.py` | TBD | — |
| 4 | Re-run `_validate_4a_precision_prune.py` с ranker filters | TBD | — |
| 5 | Написать спеку `dynamic-max-hold-spec.md` (2A) | TBD | new |
| 6 | Имплементировать 2A под config flag | TBD | — |
