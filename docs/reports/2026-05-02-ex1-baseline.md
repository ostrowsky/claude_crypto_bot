# EX1 baseline + H3 deployment — 2026-05-02

**Версия:** v2.7.0 · 2026-05-02 02:27 (UTC+3)

Два изменения в одном заходе:
- **EX1 metric** (instrumentation): `realized_to_potential_capture`
- **H3 trend-surge precedence** (behind flag, default OFF)

---

## 1. EX1 baseline (last 30 d)

```
Total paired trades: 981
  on top-20 winners: 113
  with potential data: 974
```

### Overall
| Group | n | median EX1 | mean | p25 | p75 | EX1≥0.5 |
|-------|---:|----------:|-----:|----:|----:|--------:|
| top-20 only   | 111 | **+0.001** | +0.004 | −0.005 | +0.005 | **0 %** |
| non-winners   | 857 | −0.000 | −0.000 | −0.003 | +0.002 | 0 % |

### Per mode/tf (top-20 only)
| Mode/TF | n | median EX1 |
|---------|--:|-----------:|
| impulse_speed/15m | 69 | +0.001 |
| alignment/15m | 9 | +0.001 |
| impulse_speed/1h | 8 | +0.002 |
| trend/15m | 7 | +0.001 |
| retest/15m | 5 | +0.000 |
| impulse/15m | 6 | +0.002 |
| strong_trend/15m | 3 | −0.002 |

### Per exit-class (top-20 only)
| Reason | n | median EX1 |
|--------|--:|-----------:|
| `atr_trail` | 50 | −0.000 |
| `rsi` (overbought) | 41 | **+0.005** ← лучший |
| `ema20_weakness` | 17 | **−0.010** ← худший |
| `other` | 3 | −0.005 |

### Worst 10 cases (left-on-table)

```
date       sym       mode/tf            pnl       potential   EX1
04-21      QIUSDT    impulse_speed/15m  −10.95 %  +172.4 %   −0.06    "2 closes below EMA20"
04-17      METISUSDT impulse_speed/15m   −1.78 %   +60.4 %   −0.03    "ATR-trail"
04-28      APEUSDT   impulse_speed/15m   −5.24 %  +184.1 %   −0.03    "2 closes below EMA20"
04-18      QIUSDT    impulse_speed/1h   −12.50 %  +448.7 %   −0.03    "ATR-trail"
04-19      DOGSUSDT  impulse_speed/15m   −5.77 %  +224.4 %   −0.03    "2 closes below EMA20"
04-21      TRUUSDT   impulse_speed/15m   −8.89 %  +465.1 %   −0.02    "2 closes below EMA20"
04-21      ORDIUSDT  impulse/15m         −3.91 %  +289.5 %   −0.01    "ATR-trail"
```

## 2. Что говорят данные

### Главное

**EX1 ≈ 0 на ВСЕХ режимах и группах.** Бот ловит top-20 winner’ы
правильно (coverage 0.70 уже видели), но **не извлекает прибыль**:
median realized = 0.001 × potential.

В абсолютных числах: на 113 paired trades в top-20 за 30 d средний
trade = +0.07 % при typical EOD-движе coin’а ≈ +30-200 %. Захват
**< 1 ‰**.

### Главные утечки

1. **`ema20_weakness` exit class — median EX1 −0.010.** Это
   единственная категория с отрицательной медианой. Trades
   систематически заканчиваются с loss, а coin продолжает движение.
   Топ-7 worst cases все имеют exit reason «2 closes below EMA20»
   на coins, которые потом сделали +170 .. +465 %.

2. **`atr_trail` median = −0.000.** Trail-stop тоже ничего не
   ловит — выходим в безубыток или мини-loss. Pre-fix данных
   (до v2.5.1 trail-min-buffer 26.04). Через 7 d можно посмотреть
   изменилось ли.

3. **`rsi`-overbought exit единственный с положительным median.**
   +0.005 — но это всё равно «один полпроцента из 100 % движения».
   Ловим только частичку.

### Вывод по EX1

Метрика **корректна и работает.** Подтверждает гипотезу аудита:
exit-side — главный источник утечки north-star. С реальными
intraday klines цифры будут ещё хуже (наш `potential` —
нижняя оценка через snapshot-фичи).

## 3. H3 trend-surge precedence (deployed flag-off)

Спецификация: `docs/specs/features/trend-surge-precedence-spec.md`.

```python
# files/config.py
TREND_SURGE_PRECEDENCE_ENABLED: bool = False  # default OFF
ATR_TRAIL_K_TREND_SURGE: float = 2.5
```

```python
# files/monitor.py — новый pipeline (когда flag=True):
elif surge_ok and config.TREND_SURGE_PRECEDENCE_ENABLED:
    sig_mode = "trend_surge"
    trail_k = config.ATR_TRAIL_K_TREND_SURGE
    if entry_ok:
        log.info("SURGE WON over entry_ok %s ...")
elif entry_ok:
    ...
elif surge_ok:    # legacy fallback (когда flag=False)
    sig_mode = "impulse_speed"
```

TG-метка добавлена: `🌱 Старт тренда (slope-ускорение)`.

### Что ожидать от flag flip

- entries с `sig_mode = trend_surge` появятся в логах.
- При обоих true: surge_ok AND entry_ok — лог-line `SURGE WON over
  entry_ok` для post-hoc счётчика reclassifications.
- Bandit увидит новый `mode` категорию, обновится через next retrain.

### Acceptance перед flip → True (через 7 d)

- ≥ 5 surge entries наблюдено (без flag, через legacy ветку).
- Их EX1 / capture / pnl документированы.
- Решение: flip → True или нет.

---

## 4. Итог по KPI

Phase 1 instrumentation (Apr 28-29) дал baseline. Сейчас добавлен
exit-side. **Полный snapshot:**

| ID | Метрика | Текущее | Цель |
|----|---------|--------:|-----:|
| NS | EarlyCapture@top20 | 0.071 | 0.40 |
| C1 | coverage_funnel_top20 | 70 % (raw) | 90 % |
| E1 | TTS median | +3.28 h | ≤+0.5 h |
| E2 | capture_ratio mean | 0.067 | 0.50 |
| **EX1** | **realized/potential median** | **0.001** | ≥0.30 (interim, без klines) |
| Q2 | whipsaw_rate | 13.5 % | ≤5 % |
| D1 | signal_precision | 10.2 % | ≥35 % |
| D2 | tg_message_rate | 35/d | 8-12/d |

**Изменения с прошлого baseline (Apr 29 → May 2):**
- E1 TTS улучшилось +4.94h → +3.28h (−1.7h). Но это, скорее всего,
  effect trend/1h chop-filter (срезали наиболее ленивые entries).
- D2 msg_rate упало 41 → 35/d (chop-filter работает).
- D1 precision выросло 9 % → 10.2 %.
- NS почти не изменился (0.074 → 0.071) — добавились новые
  winner-days с тем же низким capture.

## 5. Next steps

1. **7 d watch** на v2.7.0 с **flag=False**:
   - сколько surge_ok-only entries (без overlap с entry_ok) в день?
   - какой их EX1 vs trend/impulse_speed entries?
2. После 7 d → решение по flip flag.
3. **Параллельно начать H5** (trailing-only after break-even).
   Это **прямо** атакует главный leak — `ema20_weakness` exits
   на прибыльных позициях.

---

## Приложение · Что НЕ изменилось в этом релизе

- Никаких изменений в gate-каскаде.
- Никаких изменений в bandit/ML.
- TG-сообщения для подписчиков идентичны (новый mode label
  активируется только когда flag=True).
