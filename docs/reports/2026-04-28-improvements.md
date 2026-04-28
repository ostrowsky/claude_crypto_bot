# Improvement proposals — verified — 2026-04-28

Все гипотезы проверены бэктестами; источники указаны в каждой секции.
Цель проекта: **earliest BUY на coin, попадающий в top-20 daily gainers**.
Поэтому метрика приоритезации — `coverage × earliness × signal-to-noise`.

Скрипты в `files/_backtest_*`:
- `_backtest_top20_coverage_funnel.py` (14 d, 63 (date,sym) hits)
- `_backtest_fast_reversal_by_mode.py` (30 d, 1 045 paired trades)
- `_backtest_entry_lateness.py` (14 d, 47 entries on top-20 winners)

---

## Верифицированные находки

### F1 · Coverage funnel — теряем 25 % top-20 победителей

| Bucket | n | % | Что значит |
|---|---|---|---|
| `entered`        | 47 | **74.6 %** | бот выдал сигнал |
| `blocked_only`   | 10 | **15.9 %** | бот видел, но заблокировал |
| `no_event`       |  5 |  **7.9 %** | silent miss — даже candidate не сработал |
| `other_event`    |  1 |  1.6 % | побочные события без entry |

**Хуже всего 2026-04-18:** 6 top-20 winners, **только 2 entered, 3 no-event** (METIS, AUDIO, ENA), 1 blocked. 33 % покрытия за день.

Силент-промахи (5 за 14 дней):
`METISUSDT 04-18`, `AUDIOUSDT 04-18`, `ENAUSDT 04-18`, `BLURUSDT 04-23`, `FLUXUSDT 04-25`.

### F2 · Главные блокеры top-20 winners

Самый частый block-reason среди заблокированных winners — **`daily_range > 10/15 %`**:

```
AXLUSDT 04-16  daily_range 54.69%  blocked 125 раз
AXLUSDT 04-17  daily_range 54.00%  blocked  33 раза
AXLUSDT 04-18  daily_range 50.45%  blocked 213 раз  ← 3 дня подряд блокировки!
DOGSUSDT 04-17 daily_range 38.29%  blocked  87 раз
```

**Парадокс гарда:** при daily_range = 50 % движ ужé прошёл — блок корректен.
Но это значит, бот **раньше** не сработал на этом коине, когда daily_range
был 5–15 %. Проблема не в этом гарде, а в том, что **раньше-стадия скрининг
не зацепился** (15m trend / 1h alignment не дали кандидат).

Второй частый блокер — **`ML proba 0.000–0.094 outside zone`** (3 раза, AUDIOUSDT 04-17/04-19, и т.д.):
ML signal-model даёт **нулевую вероятность** для coin’а, который окажется в top-20.
Blind spot модели — не обучилась распознавать определённые подтипы импульса.

### F3 · Fast-reversal — 28 % всех потерь, не равномерно по режимам

Из 1 045 paired trades за 30 d, **15 % выходят с loss ≤ −0.3 % за ≤3 баров**,
и они тянут на **28 % суммарной просадки** (−184 % из −659 %).

Top offenders по доле fast-reversal (FR_v1: ≤3 баров и pnl ≤ −0.3 %):

| mode/tf | n | FR_v1 | avg_pnl_FR |
|---|---:|---:|---:|
| `alignment/1h`     |  35 | **31.4 %** | −0.92 % |
| `impulse/15m`      |  17 | **29.4 %** | −2.34 % |
| `impulse/1h`       |  10 | **20.0 %** | −3.45 % |
| `retest/1h`        |  24 | **25.0 %** | −1.47 % |
| `breakout/15m`     |  44 | **22.7 %** | −0.56 % |
| `trend/1h`         |  84 | **21.4 %** | −1.45 % |
| `alignment/15m`    | 226 |  20.8 %    | −0.73 % |
| `impulse_speed/15m`| 307 |   6.2 %    | −1.73 % |
| `strong_trend/15m` |   9 |   0.0 %    |    —    |

CLAUDE.md §4a называет `alignment` 53.7 % — это совпадает с **v3** (любой
pnl ≤3 баров): `alignment/15m` v3 = 44 %, `alignment/1h` v3 = 69 %. Узкое
определение fast-reversal-LOSS даёт меньшие, но всё равно тяжёлые числа.

### F4 · Lateness — entries распределены здорово по UTC

26 % entries после 12 UTC, основная масса на 00–11 UTC (Asia+early EU window).
Bot не хронически опаздывает по часам — **проблема в самом факте отсутствия
сигнала**, а не в его поздней генерации.

---

## Предложения, ранжированные по impact для главной цели

### P1 (high impact, low risk) — Spec: новый

**Расследовать silent-miss дни.** Для 5 silent-miss’ев + дня 04-18:
выгрузить все события (включая candidate / near_miss / scout-блоки) и
найти, на какой стадии воронки они потерялись:
- ML signal-zone? → расширить training set по этим коинам.
- Score floor? → понизить bull-day floor.
- Mode не triggered? → добавить screener-mode (см. P5).

Плановый артефакт: `docs/specs/features/silent-miss-recovery-spec.md`
+ конкретные конфиг-правки.

**Backtest validation:** на 14 d data покажет N из 5 потерянных
winners перешли бы в `candidate` или `entered` после правок.

### P2 (high impact, moderate risk) — Spec уже есть (draft)

**Завершить anti-fast-reversal.** F3 показывает **28 % суммарной
просадки** = fast reversal. Топ-3 страдальца: `impulse/1h` (avg
fast-loss −3.45 %!), `impulse/15m` (−2.34 %), `trend/1h` (−1.45 %).

Действовать по пунктам спеки `docs/specs/features/anti-fast-reversal-spec.md`:
1. Добавить label_fast_reversal в оба dataset’а.
2. Натренировать proba_fast_reversal head.
3. **60 d backtest** перед включением guard.
4. **Per-mode threshold:** alignment/1h, impulse/* — ниже cutoff
   (агрессивнее блочим), strong_trend/15m — guard выкл. (уже 0 %).

### P3 (medium impact, low risk) — Spec: новый

**Pre-move screener mode.** F2 показывает: к моменту daily_range = 50 %
бот блокирует, но РАНЬШЕ ничего не было (silent на ранней стадии).

Идея: добавить отдельный лёгкий канал в TG с «soft heads-up»:
condition `top_gainer_prob > 0.5 AND daily_range > 3 % AND no entry yet
in last 2 h`. Это не entry-сигнал, а **уведомление о развитии**.
Юзер сам решает.

Плюсы: ловит coin до strict-mode trigger; снижает «silent miss» как
proxy.
Риски: spam в TG, ложные heads-up. Mitigation: cooldown 4 ч / sym;
отдельный канал/раздел.

**Backtest validation:** на 14 d, для каждого top-20 winner смотрим,
был ли момент `top_gainer_prob > 0.5 AND daily_range > 3 %` ДО
основного движа. Если ≥ 80 % winners получили бы heads-up — фича
обоснована.

### P4 (medium impact, low risk) — Spec уже есть в action items

**Per-feature Pareto на `impulse_guard` (n=731, +0.355 % miss)**.
Гарду одного целого reason_code мало — он, скорее всего, состоит
из 3-4 подусловий (ADX, vol_x, slope, daily_range), и режут именно
1-2. Разбить — найти которое.

**Backtest validation:** уже есть инфра в `analyze_blocked_gates.py`,
дополнить sub-feature breakdown.

### P5 (medium impact, moderate risk) — Spec: новый

**ML signal-model retrain с расширенным negative-mining.** F2
показывает блин-спот: AUDIOUSDT даёт `ML proba 0.000` и попадает
в top-20. Модель не видит сетап.

Действия:
1. Найти все top-20 winner’ы с `ml_proba < 0.10` за 30 d (хотя бы
   AUDIO, могут быть ещё).
2. Force-include в training set с увеличенным весом.
3. Retrain → shadow A/B → компарируем recall@top20 + AUC.

**Backtest validation:** AUC top20 vs текущая (0.989). Ожидание:
+0.005 не критично, но recall на blind-spot винcoins должен вырасти
с 0 до >0.5.

### P6 (low impact, low risk) — Spec уже драфт в daily-learning

**EOD health alert.** F (не в этом отчёте, но из 2026-04-28 report):
2026-04-25 `n_collected=0`, snapshot пропущен. Cheap фикс:
TG-алерт, если `n_collected=0` или EOD не закончил к 03:30 local.

### P7 (medium impact, low confidence) — Spec: новый

**`breakout/15m` режим — кандидат на отключение.** F3: оба arm
убыточны (`very_tight −0.30 %`, `default −0.36 %`), `FR_v1 = 22.7 %`,
`trail_exit% = 91/77 %`. Похоже, режим систематически ловит
ложные пробои.

Backtest: за 30 d у нас 44 paired trades, ноль положительной
группы. Прежде чем выключать — Pareto sweep по фичам режима:
возможно, узкий sub-set (например, только bull-day breakouts) даёт
edge. Если нет — flip `BREAKOUT_15M_ENABLED=False`.

### P8 (low impact, low risk) — Spec: новый

**Capture-ratio reporting.** Сейчас ranker предсказывает
`capture_ratio_pred`, но реальная capture не считается. Для
каждого entered top-20 winner’а посчитать `(eod_high − entry_price) /
(eod_high − prev_close)`. Daily-aggregated → repository в
`docs/reports/`.

Не меняет поведение бота, но даёт прямую метрику для главной
цели проекта (early entry = high capture). Используется для
обоснования всех остальных правок.

### P9 (low impact, modest value) — Spec уже есть

**Trail-min-buffer 7-day verification** — ждём накопления данных
(уже планово, см. `trail-min-buffer-spec.md` §7).

---

## Что я НЕ предлагаю и почему

- **Расширение watchlist.** `CLAUDE.md` §14: immutable, мы не имеем
  доступа к торговле других coin’ов. Adding to watchlist бесполезно.
- **Position sizing в TG.** Бот = signal-only; рекомендация размера
  смещает проект в advisor-категорию. Можно — но это product
  decision, не оптимизация.
- **Cross-exchange аппетит / spot-фьючерсы арбитраж.** Out of scope
  для signal-bot.
- **Замена CatBoost на LGBM/XGBoost.** AUC 0.989 — модель не
  bottleneck, дальнейший рост через **новые фичи**, а не алгоритм.

---

## Итог

3 действия дают наибольший прирост покрытия top-20:

1. **P1 silent-miss audit** → потенциально +5 / 63 (8 п.п. coverage).
2. **P2 anti-fast-reversal** → −10..15 % drag, **большой UX-выигрыш**
   (TG-канал перестаёт публиковать «BUY» с моментальным фейдом).
3. **P3 pre-move screener** → ранняя коммуникация, прямой удар по
   главной метрике «early BUY».

Остальные — incremental.
