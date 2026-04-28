# Learning progress & scout review — 2026-04-28

Источники:
- `.runtime/learning_progress.jsonl` (22 daily записи, 2026-04-07 … 2026-04-28)
- `files/analyze_blocked_gates.py` (Pareto sweep по reason_code)
- `files/_backtest_trail_arm_pnl.py` (paired entry→exit за 30 d)
- `files/trend_scout_changelog.jsonl` (13 авто-правок гейтов 04-15 … 04-22)
- `.runtime/trend_scout_report.json` (последний цикл)

---

## 1. Bandit / model — итог

| Дата | recall@20 | UCB sep | AUC top20 | total_updates | n_signal |
|------|----------:|--------:|----------:|--------------:|---------:|
| 2026-04-07 | 99.0 % | 0.047 |   —    |    86 751 |  5 888 |
| 2026-04-13 | 100 %  | 0.112 | 0.654 |   362 594 |  7 167 |
| 2026-04-16 | 100 %  | 0.130 | **0.886** ↑ |   503 661 |  8 000 ⚠ |
| 2026-04-17 | 100 %  | 0.136 | 0.932 |   551 329 |  8 000 ⚠ |
| 2026-04-18 | 100 %  | 0.144 | 0.978 |   601 326 | 10 040 ✓ |
| 2026-04-22 | 100 %  | 0.138 | 0.990 |   805 408 | 11 072 |
| 2026-04-25 | 100 %  | 0.141 | 0.988 |   963 913 | 11 945 |
| **2026-04-28** | **100 %** | **0.154** | **0.989** | **1 126 220** | **12 605** |

### Что значит

- **Recall@top20 = 100 %** с 2026-04-10 непрерывно. Бандит «видит» всех
  фактических TG-победителей среди ~105 watchlist coin’ов.
- **AUC top20 = 0.989** — модель top-gainer’а уверенно отделяет
  победителей от не-победителей. Скачок 0.68 → 0.89 на 04-15→04-16
  — момент снятия cap’а `bandit_n_signal=8000` (раньше упирался,
  обучение по signal-датасету застопорилось).
- **UCB-separation 0.15** — стабильно, сильно выше apr-baseline 0.04.
  ENTER vs SKIP раздвигается; exploration alpha=2.0 ещё держит
  любопытство.
- **+1.04 M updates за 3 недели** (86 751 → 1 126 220, ×13). Темп
  обучения здоровый, ежедневно ~50 k новых записей.

### Аномалия

- **2026-04-25**: `n_collected=0` — пропущен intraday snapshot
  (вероятно, scheduled task упал или был «спящий» компьютер).
  Один день, без эффекта на recall (модель оперирует EOD-датасетом).
  Watch-point: алерт в TG, если `n_collected=0` снова — добавлен
  follow-up в `daily-learning-pipeline-spec.md` §8.

### Вывод по обучению

Зрелая фаза. Модель TG-prob калибрована (AUC ~0.99), бандит
сошёлся к стабильному decision-boundary, recall не теряется.
Дальнейший рост качества — за счёт **новых фичей** (planned:
`proba_fast_reversal`, `realised_buffer_pct`), а не объёма данных.

---

## 2. Скаут — Pareto sweep гейтов (last 14 d)

```
action     reason_code                     n  avg_r5%  win5%  Sh*sqN  miss_vs_take
take       take                          2652   +0.002  44.6   +0.08    base
blocked    ml_proba_zone                 1844   −0.103  40.7   −1.74   −0.106  ← рабочий
blocked    entry_score                   1684   +0.072  48.0   +3.10   +0.070  ← over-block
candidate  rule_signal                   1063   −0.115  39.8   −3.03    —
candidate  near_miss                      820   +0.013  44.1   +0.43    —
blocked    ranker_hard_veto               806   +0.120  45.5   +1.91   +0.117  ← over-block
blocked    portfolio                      741   +0.048  43.7   +0.89   +0.046
blocked    impulse_guard                  731   +0.357  40.1   +1.85   +0.355  ← OVER-BLOCK HARD
blocked    trend_quality                  669   +0.075  44.5   +1.75   +0.073  ← over-block
blocked    clone_signal_guard             543   +0.091  50.3   +1.59   +0.088  ← over-block
blocked    cooldown                       519   +0.020  41.2   +0.15   +0.018
blocked    mtf                            234   −0.248  36.3   −1.89   −0.250  ← рабочий
blocked    open_cluster_cap               216   −0.362  31.5   −2.90   −0.365  ← рабочий
blocked    mode_range_quality              93   −0.312  34.4   −2.12   −0.314  ← рабочий
blocked    late_impulse_rotation           36   +0.566  55.6   +2.61   +0.563  ← over-block (мелкая n)
```

### Чтение
- `take.avg_r5 = +0.002 %` — фактический edge близок к нулю на 5-баровом
  горизонте. Большая часть положительного edge’а живёт **в blocked-bucket’ах**
  → значит, мы режем своими руками то, что должно проходить.
- **Рабочие гарды (avg_r5 < take, отрицательный):** `ml_proba_zone`,
  `mtf`, `open_cluster_cap`, `mode_range_quality`. Сохранять как есть.
- **Over-blocking (Sharpe×√n ≥ +1.5 и avg_r5 > take):**
  - `impulse_guard` (n=731, +0.355 %, +1.85 σ) — самый болезненный.
  - `entry_score` (n=1684, +0.07 %, +3.10 σ) — самый массовый.
  - `ranker_hard_veto` (n=806, +0.12 %, +1.91 σ) — спекой
    `ml-candidate-ranker-spec.md` уже зафиксировано как кандидат
    на смягчение.
  - `trend_quality` (n=669, +0.07 %, +1.75 σ) — порог RSI или
    price-edge стоит поднять (см. `trend-quality-guard-spec.md` §8).
  - `clone_signal_guard` (n=543, +0.09 %, +1.59 σ) — Trend Scout
    уже смягчает (см. §3 ниже), 6 → 14 за 7 дней.
  - `late_impulse_rotation` (n=36, малая выборка — игнор пока
    не накопится).

### Что делать
- Запустить per-feature Pareto на `impulse_guard` и `entry_score`
  → найти подмножество условий, реально дающих edge.
- Каждое смягчение — отдельная спека (по правилу AGENTS.md);
  не ужимать одновременно, чтобы не повторять Apr-06 («3 фильтра
  блокировали 100 % top-gainers»).

---

## 3. Авто-тюнинг (Trend Scout)

13 авто-применений за 04-15 … 04-22. Распределение:

| Параметр | Изменения | Net direction |
|----------|-----------|----------------|
| `CLONE_SIGNAL_GUARD_MAX_SIMILAR` | 8 шагов: 6 → 14 | relaxing |
| `IMPULSE_SPEED_1H_ADX_MIN` | 5 шагов: 22 → 14 | relaxing |

### Корреляция с Pareto

- `CLONE_SIGNAL_GUARD` ослаблен 6→14 — но в Pareto всё ещё
  `over-block` (n=543, +0.09 %). **Ослабление работает в нужную сторону**,
  возможно нужен ещё шаг (14→16) — но осторожно: clone-guard защищает от
  «дубликатов» (тот же сетап в коррелированных коинах), верхней границы
  ослабления нет в спеке.
- `IMPULSE_SPEED_1H_ADX_MIN` опущен 22→14 — но `impulse_guard` всё ещё
  `+0.355 %` over-block. ADX-floor — лишь часть условия; другие
  компоненты (vol_x, slope) — стоит проверить отдельно.

### Текущий цикл

```
ts: 2026-04-28T15:43:37Z
candidates_total: 6
candidates_trending: 3
entered: 1
blocked_trending: 2
proposals: []
```

3 trending кандидата, 1 вход, 2 заблокированы. Скаут не предложил новых
правок — текущая конфигурация считается стабильной для этого цикла.

---

## 4. Trail-arm — сводка (paired trades 30 d)

После фикса `trail-min-buffer` (2026-04-26) ATR-only whipsaw’ы должны
ослабнуть. Surface для всех paired trades:

```
=== Trail-hit losses (pnl<0 AND trail-stop) ===
arm           n  avg_loss   avg_bars  impulse/strong
very_tight   72   −1.08 %    5.9         13
tight        67   −1.70 %    5.7         67   ← все потери на impulse/strong
default     128   −1.03 %    6.0         27
wide          4   −1.20 %    7.8          3
very_wide    42   −1.76 %    8.0         32
```

- 67 потерь на `tight` arm — **все 67** на impulse/strong → подтверждает
  гипотезу из `trail-min-buffer-spec`. Floor 1.5 % ATR-trail должен этот
  кластер срезать.
- `very_tight` (n=72, −1.08 %) — раскидан по разным режимам, ATR floor
  работает только на impulse/strong/_speed; другие режимы остаются как
  были (default = 0).

### Per-mode highlights (avg P&L)

| mode/tf | best arm | avg P&L | n |
|---------|----------|---------|---|
| impulse_speed/15m | very_tight | +0.59 % | 25 |
| impulse_speed/1h  | default    | +0.59 % |  9 |
| impulse_speed/1h  | tight      | +0.36 % | 60 |
| trend/15m         | default    | +0.37 % | 42 |
| retest/15m        | default    | +0.16 % | 58 |
| breakout/15m      | (все ≤ 0)  | −0.30 % | 22 |
| impulse/15m       | default    | +0.53 % |  9 |

Замечание: `breakout/15m` (n=44) — обе arm’ы убыточны (very_tight −0.30 %,
default −0.36 %), trail-exit% = 91 % / 77 %. Похоже на структурную
проблему режима (а не trail’а). Кандидат на отдельную спеку
`breakout-15m-review`.

---

## 5. Открытые позиции (snapshot 2026-04-16 06:00 UTC bar 98)

9 open: `CELO trend`, `BAT/ENA/SUI/ADA/INJ/ORDI impulse_speed`,
`ETC alignment`, `LINK trend`. ATR-floor виден в работе:

- `ADAUSDT impulse_speed`: trail = entry × (1 − 0.017) ≈ floor
  1.5 % активен.
- `INJUSDT impulse_speed/15m`: trail = price − 1.57 % → floor активен.
- `ORDIUSDT impulse_speed/15m`: daily +50 %, ATR-buffer (~10 %) сам
  доминирует — floor неактивен, как и должно.

---

## 6. Action items (приоритет)

| # | Задача | Owner | Спека |
|---|--------|-------|-------|
| 1 | Per-feature Pareto на `impulse_guard` (+0.36 % miss, n=731) | tbd | new |
| 2 | Per-feature Pareto на `entry_score` (n=1684, самый массовый) | tbd | new |
| 3 | 7-day live verification trail-min-buffer (≥50 paired trades на target modes) | claude | trail-min-buffer §7 |
| 4 | Реализовать `anti-fast-reversal` label (этап 1 из 5) | tbd | anti-fast-reversal |
| 5 | Алерт TG если `n_collected=0` в EOD | tbd | daily-learning §8 |
| 6 | Ревью режима `breakout/15m` (обе arm убыточны) | tbd | new |

---

## 7. Резюме

**Обучение:** созрело. Метрики стабильны, AUC ~0.99, recall@20 100 %,
ежедневно +50 k обновлений. Bottleneck — не данные, а недостающие
фичи (fast-reversal, realised buffer).

**Скаут:** 4 рабочих гарда (`ml_proba_zone`, `mtf`, `open_cluster_cap`,
`mode_range_quality`), 6 over-blocking. Авто-тюнинг ослабляет 2 из 6
(`CLONE_SIGNAL_GUARD`, `IMPULSE_SPEED_1H_ADX_MIN`), но `impulse_guard`
и `entry_score` требуют ручного per-feature Pareto.

**Trail-floor:** в проде с 04-26, floor виден на live-позициях, эффект
на 30-d backtest пока не measurable (не накопилось). Re-run через
неделю.
