# Improvement proposals — validation pass — 2026-04-28

Каждое предложение из `2026-04-28-improvements.md` прогнано через
бэктест. Скрипты: `files/_validate_p*.py`.

## Summary table — переоценка приоритетов

| # | Was | Now | Reason |
|---|-----|-----|--------|
| **P1** silent-miss audit | high | **low** | Из 5 silent — 3 это «in cooldown / holding», только 2 реальных miss |
| **P2** anti-fast-reversal | high | **high** | Подтверждено: 28 % drag, per-mode FR подтверждена |
| **P3** pre-move screener | medium | **needs-redesign** | Lead time +4ч ✅, но 75 FP/day ❌ |
| **P4** impulse_guard breakdown | medium | **medium** | Видны конкретные кандидаты на смягчение |
| **P5** ML blind-spots | medium | **HIGH (raised)** | 24 % top-20 winners заблокированы ML-zone, 19 extreme |
| **P6** EOD health alert | low | **low (cheap)** | Нечего валидировать, ставить и забыть |
| **P7** kill breakout/15m | low | **medium (raised)** | Подтверждено: даже на top-20 winners 0 % capture |
| **P8** capture-ratio reporting | low | **low** | Instrumentation, не нужна валидация |
| **P9** trail-floor verify | (deferred) | (deferred) | Ждёт 7+ дней данных |

---

## P1 — Silent-miss audit · понижено в приоритете

**Гипотеза:** 5 top-20 winners за 14 d не получили ни одного события.
Recover их = +8 п.п. coverage.

**Валидация:** прогнал `_validate_p1_silent_miss.py`.

| Sym/Date | Реальная причина |
|---|---|
| ENAUSDT 04-18 | накануне `entry` 04-17 08:08 — **держим позицию**, новых событий не нужно |
| BLURUSDT 04-23 | накануне `entry` 04-22 03:52 — **держим позицию** |
| METISUSDT 04-18 | 04-17 entry 02:03 → exit 02:32 — короткая сделка; cooldown_bars=19 ≈ 19h, мог захватить начало 04-18 |
| **AUDIOUSDT 04-18** | **REAL silent miss.** 04-17 был блок daily_range>15%, 04-19 ml_proba 0.021 |
| **FLUXUSDT 04-25** | **REAL silent miss.** 04-24 exit рано (EMA20 weakness), cooldown к 04-25 уже истёк |

**Реальная coverage:** не 74.6 %, а **≥80 %** после поправки на «держим позицию».
Истинных silent-промахов 2 из 14d, что ≈ 1 в неделю. Их обоих покрывает
**P5 (ML blind-spots)** — AUDIO/FLUX оба фигурируют в P5-листе.

**Вывод:** P1 как отдельная инициатива не нужна. Покрывается P5.

---

## P2 — Anti-fast-reversal · подтверждено high impact

**Гипотеза:** 28 % суммарной просадки = fast-reversal; per-mode threshold
лучше глобального.

**Валидация (от предыдущего раунда, `_backtest_fast_reversal_by_mode.py`):**

```
Mode/TF              n   FR_v1  avg_pnl_FR
alignment/1h        35   31.4%  -0.92%
impulse/1h          10   20.0%  -3.45%   ← худшая в долларах
impulse/15m         17   29.4%  -2.34%
retest/1h           24   25.0%  -1.47%
breakout/15m        44   22.7%  -0.56%
trend/1h            84   21.4%  -1.45%
strong_trend/15m     9    0.0%    —      ← guard выкл.
```

Dispersion FR_v1 от 0 % до 31 % — **per-mode threshold обязателен**.

**Дополнительная backtest-готовность:** label `label_fast_reversal` ещё
не вычислен в datasete. Полноценный backtest c proba_fast_reversal
возможен ТОЛЬКО после implementation step 1 спеки. Соответствует §7
спеки `anti-fast-reversal-spec.md`.

**Вывод:** оставить high. Action: начать с label.

---

## P3 — Pre-move screener · needs redesign

**Гипотеза:** soft heads-up при `daily_range > 3%` ловит top-20 winners
за 4 ч до strict-entry, low FP.

**Валидация (`_validate_p3_premove_screener.py`):**

```
Threshold sweep:
  thr     winner-hit %   FP/day   lead-time mdn(h)
  0.02         100.0%     75.8         +4.0
  0.03         100.0%     75.8         +4.0
  0.05         100.0%     75.6         +4.0
  0.10         100.0%     75.1         +4.0
```

- ✅ Recall 100 % top-20 — отлично.
- ✅ Median lead-time **+4 ч** перед entry — реальный early-warning.
- ❌ **FP = 75 non-winners в день** при любом пороге 2-10 %. Threshold
  не помогает, потому что движ ≥3 % за день — обычное дело для волатильного
  альткоина.

**Что нужно:** дополнительный filter — `top_gainer_prob` от модели,
которая у нас AUC=0.989 на label_top20. На каждом intraday-snapshot
запускать модель → если `proba_top20 > 0.5 AND tg_return_since_open > 0.03`
→ heads-up.

**Backtest этого требует offline-inference на исторических snapshot’ах**
(дополнительная работа, ~1–2 ч). Пока не сделано — P3 в `needs-redesign`.

**Вывод:** идея жизнеспособна, но в текущем варианте слишком шумная.
Нужно отдельной спекой: `pre-move-screener-spec.md` с явным шагом
«inference на каждом snapshot».

---

## P4 — Impulse_guard sub-conditions · medium, видны кандидаты

**Гипотеза:** разбить `impulse_guard` n=731 на под-условия даст
конкретные пороги для смягчения.

**Валидация (`_validate_p4_impulse_guard_subconds.py`, 14 d):**

```
sub-condition           n    верность
daily_range_1h_15      865   работает (movе уже большой)
daily_range_1h_10      734   спорно (bull-day кейсы)
weak_15m_adx_<20       578   кандидат на смягчение
weak_1h_adx_<18       ~500   кандидат, auto-tuner уже снизил 22→14
RSI_>76               ~150   кандидат, был уже relaxed 68→76
mode_range_low_<7%      46   кандидат на нижний bound
open_cluster_cap       400   работает
```

**Топ кандидат на ручное смягчение:** `weak_15m_adx_<20`
(578 блоков за 14 d). Auto-tuner работает с **1h ADX**, но **15m ADX**
(другая константа) не двигает. Trace: проверить, какие top-20 winners
были блокированы 15m ADX < 20. Если ≥ 5 — снизить до 18.

**Вывод:** P4 medium. Самостоятельная спека:
`impulse-guard-15m-adx-relax-spec.md` после конкретной 15m-выборки.

---

## P5 — ML signal-model blind-spots · HIGH, поднято с medium

**Гипотеза:** ML model даёт <0.10 на некоторых top-20 winners → blind-spot.

**Валидация (`_validate_p5_ml_blindspots.py`, 30 d):**

```
Top-20 winners blocked by ML-zone:   34 / ~140 winner-days  (~24%)
  extreme blind-spot (proba < 0.10):  19   ← critical
  moderate (0.10-0.28):               15
  near-threshold (>= 0.28):            0

Top recurring blind-spot symbols:
  TRUUSDT      3 winner-days
  BLURUSDT     3 winner-days
  MDTUSDT      2  ORDIUSDT  2  AUDIOUSDT  2
  QIUSDT       2  COMPUSDT  2  APEUSDT    2
```

**Сила сигнала:** 24 % top-20 winners проходят через ML-block. AUC=0.989
обманчив — модель делает большинство решений правильно, но систематически
промахивается на определённом подтипе сетапа (TRU/BLUR/ORDI/AUDIO).

**Действия (явный план):**
1. Найти features этих 19 extreme cases на момент блокировки.
2. Force-include в training set с весом ×3.
3. Retrain → shadow A/B (уже инфраструктура есть в `ml_candidate_ranker_shadow_report.json`).
4. Метрика: `recall@top20` на blind-spot subset должен вырасти с 0 % до >50 %, общий AUC не должен упасть ниже 0.97.

**Вывод:** **самый высокий impact из всех P**. Поднять в top-priority.
Спека: `ml-signal-blindspot-recovery-spec.md`.

---

## P6 — EOD health alert · noop

Backtest не нужен. Cheap. 1 строка конфига + tg-send. Ставить.

---

## P7 — Disable breakout/15m · подтверждено, поднято до medium

**Валидация (`_validate_p7_breakout_15m.py`, 30 d):**

```
breakout/15m entries:                45
  on top-20 winners:                  5  (11.1%)
  on non-winners:                    40

Paired entry→exit:                   44, avg_pnl=-0.33%, win=23%, FR_v1=22.7%
  on top-20 (5):                     avg_pnl=+0.03%   ← даже здесь zero
  on non-winners (39):               avg_pnl=-0.37%
```

Даже когда мы ловим top-20 winner через breakout/15m — **profit ≈ 0**.
Режим не извлекает edge.

**Вывод:** flip `BREAKOUT_15M_ENABLED=False`, observe 7 d.
Спека: `breakout-15m-disable-spec.md` (короткая, с rollback-критерием).

---

## P8 — Capture-ratio reporting · keep low

Instrumentation. Не меняет поведение. Опционально, после P2/P5/P7.

---

## P9 — Trail-min-buffer 7-day verify · идёт по плану

Ждём 7 дней данных, потом re-run `_backtest_trail_arm_pnl.py`.
Action item уже в `trail-min-buffer-spec.md` §7.

---

## Финальный rerank

Top-3 после валидации (изменилось!):

1. **P5 ML blind-spot recovery** ← поднято с medium до HIGH.
   24 % winners системно теряются в ML-zone. Конкретные коины известны.
2. **P2 anti-fast-reversal** — подтверждён: 28 % drag, per-mode threshold.
3. **P7 disable breakout/15m** — подтверждён: даже winners дают +0.03 %.

Опущено:
- P1 silent-miss → low (3 из 5 «промахов» это holding).
- P3 pre-move screener → needs-redesign (FP 75/day слишком много).

Side effects:
- 4 рабочих под-условия `impulse_guard` распарсены — P4 готов к
  таргетной спеке по `15m_ADX < 20`.
