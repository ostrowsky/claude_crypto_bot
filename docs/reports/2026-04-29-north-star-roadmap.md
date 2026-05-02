# North-star roadmap — пути увеличения EarlyCapture@top20

`EarlyCapture@top20 = coverage × capture × time_lead`

Текущее: `0.697 × 0.161 × 0.685 ≈ 0.077`. Цель: `≥ 0.40`.

---

## 1. Математика рычагов

| Lever | Current | Target | Sensitivity (∂NS) |
|-------|--------:|-------:|-------------------|
| coverage | 0.70 | 0.90 | low–medium |
| **capture** | **0.16** | **0.50** | **HIGHEST** (×3.1 фактор) |
| time_lead | 0.69 | 0.85 | low–medium |

**Ключевой инсайт:** capture и time_lead **физически связаны**. Чем
раньше входишь, тем (а) меньше «уже сделанной» части движа, (б) ниже
entry_hour_UTC. Один lever — earliness — двигает обе оси
одновременно.

```
если TTS падает с +4.94h до +0.5h:
  capture растёт ~ (1 − 0.5/total_move_h) × current_capture
  time_lead тоже растёт (entry_hour ↓)
```

Поэтому ниже разделяю четыре независимых пути по убыванию impact:

```
Path 1: EARLINESS         ← главный (двигает 2 lever’а сразу)
Path 2: HOLD_LONGER       ← capture-only
Path 3: COVERAGE          ← coverage-only (lowest leverage)
Path 4: PRECISION/UX      ← не двигает NS напрямую, но снижает спам
```

---

## 2. Path 1 — Earliness (двигает capture + lead)

### 1A. `top_gainer_prob` как **mega-trigger**

**Идея:** model AUC top20 = 0.989. Если `proba_top20 ≥ 0.70` на
любом 1h baring confirmation → **ENTER немедленно**, минуя стандартную
гард-каскад (кроме hard-risk: portfolio cap, cooldown).

**Логика:** model уверена; стандартные гарды лишь отсекали бы
«поздних», но бот и так опаздывает на 5h. Прямой путь — сразу
доверять модели.

**Backtest validation:**
- За 30 d посчитать, сколько (date, sym) пар имели хотя бы один
  snapshot с `proba_top20 ≥ 0.70`.
- Из них — сколько в реальности были top-20 winners (precision).
- Если precision ≥ 60 % при ≥ 0.50 проницаемости → mega-trigger
  жизнеспособен.

**Ожидаемый Δ NS:** time_lead +0.10, capture +0.20 → ΔNS ~ +0.13.
**Risk:** пропустить hard-блок (portfolio overload). Mitigation:
обходим только soft-гарды.

**Spec:** `tg-prob-mega-trigger-spec.md` (TBD).

### 1B. P3-redesign: pre-move screener с `top_gainer_prob`-фильтром

**Идея:** soft-heads-up в TG, когда `daily_range > 3 % AND
proba_top20 > 0.5`. Не entry-сигнал, но юзер видит на 4 h раньше.

**Backtest validation (уже частично сделан в
`_validate_p3_premove_screener.py`):**
- Median lead +4 h ✓
- recall 100 % ✓
- FP 75/d ❌ → нужен `proba_top20`-фильтр на каждом snapshot.

**Ожидаемый Δ NS:** косвенный (повышает осведомлённость подписчика,
не прямые entries бота). Не вкладывается в формулу EarlyCapture
напрямую, но даёт UX-ценность.

**Spec:** уже есть `pre-move-screener-spec.md` нужно создать (P3
needs-redesign).

### 1C. Дифференцированный `daily_range` гард

**Проблема:** гард `daily_range > 10/15 %` блокирует на дне 2-3
многодневного rally — но мы бы хотели зайти в **первый день**, когда
range был только 5 %.

**Идея:** разделить 2 случая:
- `daily_range > 10 %` AND `prev_day_close - 7d_low_close < 5 %`
  (свежий импульс) → **разрешать**
- `daily_range > 15 %` AND `prev_day_range > 10 %` (3-й день парада)
  → **блокировать**

**Backtest validation:** в `_validate_p1_silent_miss.py` мы видели
AXLUSDT 04-16/17/18 блокирован 3 дня. Если рассматривать только
первый день (04-16) — entry был бы валиден.

**Ожидаемый Δ NS:** coverage +0.05, capture +0.05 (раньше входим в
новые impulses) → ΔNS ~ +0.04.

**Spec:** `daily-range-stage-aware-spec.md` (TBD).

### 1D. ML floor relaxation на bull-day

**Идея:** `ML_GENERAL_HARD_BLOCK_MIN = 0.28` сейчас. Опустить до
0.20 на bull-day с shadow A/B.

**Backtest validation:** из `_validate_p5_ml_blindspots.py` мы видели,
что 19 из 34 winner-days имели `proba < 0.10` — даже 0.20 их не
спасёт. Но 15 случаев в зоне `0.10–0.28`, которые отпали бы.

**Ожидаемый Δ NS:** coverage +0.03 → ΔNS ~ +0.01.

**Risk:** spam ↑. Mitigation: только если `is_bull_day=True`.

**Spec:** мелкая правка config; объединить с P5.

---

## 3. Path 2 — Hold longer (capture-only)

### 2A. Dynamic `max_hold_bars` based on momentum

**Идея:** `MAX_HOLD_BARS=24` сейчас фиксирован. Если pos.pnl > 1 %
**и** ADX продолжает расти → продлевать holds на каждый бар, где
ADX (now) ≥ ADX (entry).

**Backtest validation:** из `_backtest_capture_ratio.py` median
capture = 0.002 — 50 % entries закрываются практически на entry-цене
(ATR-trail + max_hold). Если на 50 % из них продлить hold —
average capture может вырасти на +0.10.

**Ожидаемый Δ NS:** capture +0.10–0.15 → ΔNS ~ +0.06.

**Risk:** при reversal задерживаем exit, max_drawdown ↑. Mitigation:
условие на ADX growing AND price > EMA20.

**Spec:** `dynamic-max-hold-spec.md` (TBD).

### 2B. Re-entry после micro-pullback

**Проблема:** `EMA20 weakness` exit (см. FLUXUSDT 04-24) срабатывает
на нормальном retracement в bull-day. После `cooldown_bars=19` мы
уже отстали.

**Идея:** если `pos.pnl > 0` на момент EMA20-weakness И
`top_gainer_prob` всё ещё > 0.5 → **не выходить, переключиться в
trailing-only mode**.

**Backtest validation:** найти trades с reason=`EMA20 weakness`,
которые имели бы > 1 % PnL если бы продержались до EOD.

**Ожидаемый Δ NS:** capture +0.05 → ΔNS ~ +0.03.

**Spec:** `pullback-tolerance-spec.md` (TBD).

### 2C. Cooldown bypass на признанных top-N coins

**Идея:** если бот уже захватил coin, который потом стал top-20
gainer’ом, и cooldown ещё активен — **позволить re-entry** при
повторном signal-сетапе.

**Backtest validation:** найти cases где после exit (cooldown active)
тот же coin показал ещё +5 % движа в тот же день.

**Ожидаемый Δ NS:** capture +0.03 → ΔNS ~ +0.01.

**Spec:** `cooldown-bypass-top-gainer-spec.md` (TBD).

---

## 4. Path 3 — Coverage (lowest leverage)

### 3A. P5 — ML blind-spot recovery (уже draft)

Спека есть. 24 % winners блокируются ML-zone. После oversampling
weight=3.0 → ожидание 10 %.

**Δ NS:** coverage +0.05 → ΔNS ~ +0.02–0.03.

### 3B. Audit early-stage scoring floor

**Идея:** `score_floor_at_entry = 56` — иногда отсекает legit ранние
сетапы. Анализ: какой % top-20 winners в их early-stage имел score
51–56?

**Spec:** объединить с P5.

---

## 5. Path 4 — Precision/UX (не двигает NS, важно для подписчика)

### 4A. D1/D2 pruning — повысить precision

**Проблема:** 41 entry/day, 9 % precision = 91 % шум.

**Backtest validation:** найти feature-комбинации, разделяющие
top-20 winners от non-winners. Кандидаты: `ranker_ev > 0`, `ml_proba
> 0.40`, `top_gainer_prob > 0.30`.

При жёстком `ranker_ev > 0`: estimate −60 % entries, +30 % precision.

**Spec:** `precision-prune-spec.md` (TBD).

### 4B. P7 — disable breakout/15m (уже draft)

Спека есть. Q1 mode-level → 0, D2 −2/d.

### 4C. P2 — anti-fast-reversal (уже draft)

Q1 −7 п.п., Q3 −16 п.п.

---

## 6. Sequenced rollout — путь к 0.40

Обоснование порядка:
1. **Сначала precision** (P7 + P2 + 4A). Освобождает «бюджет шума» для
   path 1, без чего любые liberalisations дадут спам в TG.
2. **Затем earliness** (1A + 1C). Главный driver. После прунинга
   precision можно безопасно понижать пороги.
3. **Затем hold-longer** (2A + 2B). Усиливает уже-полученное.
4. **Coverage** (3A) — incremental.

| Step | Initiative | Expected NS | Cumulative | Confidence |
|-----:|-----------|------------:|-----------:|-----------|
| 0 | baseline | — | **0.077** | ✓ measured |
| 1 | P7 disable breakout/15m | — (UX) | 0.077 | high |
| 2 | P2 anti-fast-reversal | (UX, indirect) | 0.080 | high |
| 3 | 4A precision-prune (ranker_ev>0) | +0.01 | 0.090 | medium |
| 4 | 1C daily_range stage-aware | +0.04 | 0.130 | medium |
| 5 | 1A `top_gainer_prob` mega-trigger | +0.13 | **0.260** | medium |
| 6 | 2A dynamic max_hold | +0.06 | 0.320 | medium |
| 7 | P5 ML blind-spot | +0.03 | 0.350 | high |
| 8 | 2B pullback tolerance | +0.03 | 0.380 | low |
| 9 | 1D ML floor bull-day | +0.01 | **0.390** | medium |
|10 | 1B P3 screener TG-prob filter | +0.03 (UX) | 0.420 | medium |

**Время:** реализм — 8–12 недель при последовательной реализации
(каждый шаг = спека → backtest → shadow A/B → promote → 7 d watch).

---

## 7. Anti-pattern: «давайте просто понизим пороги»

Соблазн увеличить coverage за счёт массового ослабления гардов
(например, `ML_PROBA_MIN: 0.28 → 0.05`). Это даст +5 % coverage и
−40 % precision → TG-канал заваливается на 60 entries/day, 5 %
precision. **NS не вырастет** (capture останется ~0.16, потому что
гард не меняет timing entry).

**Главный принцип роадмапа:** earliness + selectivity, не permissivity.
Мы не хотим больше сигналов — мы хотим **более ранние сигналы на
более узкой выборке**.

---

## 8. Risks & rollback

Каждая инициатива выше — отдельная спека с rollback-флагом. Acceptance
threshold для каждой:

- **NS не падает > 0.005** относительно pre-rollout 7-day rolling mean.
- **recall@top20 ≥ 95 %**.
- **max_drawdown_per_signal не растёт > 1 σ**.
- **TG msg_rate не растёт > 30 %**.

Если любой anti-metric нарушен → flip flag, postmortem в
`docs/reports/`.

---

## 9. Action items для следующей итерации

| # | Action | Owner |
|---|--------|-------|
| 1 | Backtest 1A (top_gainer_prob mega-trigger): validate precision ≥ 60 % при proba ≥ 0.7 | TBD |
| 2 | Backtest 1C (daily_range stage-aware): доля winners с blocked daily_range при early-stage | TBD |
| 3 | Backtest 2A (dynamic max_hold): доля trades, где momentum-condition при exit-time ещё bullish | TBD |
| 4 | Backtest 4A (precision-prune): подсчитать precision @ ranker_ev>0 | TBD |
| 5 | Написать спеку `tg-prob-mega-trigger-spec.md` после backtest 1 | TBD |

После backtest’ов 1–4 → re-rank initiatives по реальному impact и
переписать step 5–10 plan если нужно.
