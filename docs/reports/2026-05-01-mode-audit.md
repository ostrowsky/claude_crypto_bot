# Комплексный аудит entry-режимов, метрик и гипотез

**Дата:** 2026-05-01
**Версия бота на момент аудита:** v2.6.0 (trend/1h chop-filter shipped)
**Аудитор role:** crypto-trader + signals-app architect
**Метод:** прямое чтение `strategy.py` + `monitor.py` + cross-reference с
30 d backtest-данными.

---

## TL;DR (3 главных вывода)

1. **Между режимами 4 архитектурных конфликта**, главный — `trend`
   и `impulse_speed` запускаются одной функцией `check_entry_conditions`,
   а различаются только trail-k. `impulse_speed` — это **не отдельный
   detector**, а маркер расширения trail. Семантически режимов
   фактически **6**, не 7 (CLAUDE.md лжёт).

2. **Метрики не соответствуют главной цели.** Текущий KPI-стек
   (recall@20=100 %, AUC=0.989) меряет видимость в bandit-датасете,
   не качество сигналов в TG-канале. North-star `EarlyCapture@top20 = 0.074`
   уже введён, но critical missing piece — **отсутствует exit-side
   metric** («прибыль зафиксирована / возможная»). Бот меряет вход,
   игнорирует выход.

3. **«Максимально ранний BUY на устойчивый тренд» структурно
   несовместим с текущим pipeline.** `IMPULSE` ловит за 1-3 бара
   (хорошо), `TREND_SURGE` ловит ускорение (тоже рано), но они
   **отключены по умолчанию** или маскируются `entry_ok` который
   срабатывает раньше с другим набором фильтров. Разрыв между
   «детектор» и «логика выбора» создаёт race-condition: бот
   срабатывает на trend-режим до того, как surge/impulse детекторы
   получают шанс.

---

## ЧАСТЬ 1 · Аудит режимов и противоречий

### 1.1 Реестр режимов (по факту, не по доке)

| # | Режим | Detector | Где назначается | trail_k | max_hold | Фактическая роль |
|---|-------|----------|-----------------|--------:|---------:|------------------|
| 1 | `trend` | `check_entry_conditions` | `get_effective_entry_mode`→ `trend` | 2.0 | 16/48 | дефолт по `entry_ok` |
| 2 | `strong_trend` | `check_entry_conditions` + `get_entry_mode` reclassify | `get_effective_entry_mode`→ `strong_trend` | 2.5 | 16/48 | расширенный trail |
| 3 | `impulse_speed` | `check_entry_conditions` + `price_speed≥1.5` + альтернативный путь от `_one_hour_impulse_speed_entry_guard` | `get_effective_entry_mode` или surge fallback | 2.5 | 16/48 | расширенный trail для быстрых движений |
| 4 | `breakout` | `check_breakout_conditions` (отдельная функция) | `brk_ok` ветка | 1.5 | 6 | пробой флэта |
| 5 | `retest` | `check_retest_conditions` (отдельная функция) | `ret_ok` ветка | 1.8 | 10 | откат к EMA20 |
| 6 | `impulse` | `check_impulse_conditions` (отдельная) | `imp_ok` ветка | 2.0 | 48/16 | r1+r3 спайк |
| 7 | `alignment` | `check_alignment_conditions` (отдельная) | `aln_ok` ветка | 2.0 | 16/48 | плавный устойчивый тренд |
| (8) | `impulse_cross` | `check_ema_cross_conditions` (легаси) | EMA20 пробой снизу | 2.5 | 16/48 | устаревший, ~7 трейдов / 30d |
| (9) | `trend_surge` | `check_trend_surge_conditions` | **только в catch-up scan, не в основном pipeline** | — | — | "детектор начала тренда", фактически dead-code для онлайн |

> **Конфликт #0 (документация):** CLAUDE.md заявляет «7 entry modes»
> с `breakout` 15m. Фактически активны 7-8 в реальных событиях
> (`impulse_cross` всё ещё пишется), а `trend_surge` написан, но
> в основной поток входа не подключён (вызывается только в catch-up
> и backtest). Нужно либо оживить, либо удалить.

### 1.2 Конфликты между режимами

#### Конфликт A · `trend` vs `impulse_speed` — это **не разные режимы**

Оба проходят `check_entry_conditions` и различаются ТОЛЬКО внутри
`get_entry_mode(feat, i)`:
```python
if ADX≥28 and vol_x≥2.0 and ema_sep_ok and ema50_rising:
    return "strong_trend"
if price_speed_3bar ≥ 1.5%:
    return "impulse_speed"
return "trend"
```

Все три имеют **одинаковые** входные пороги (slope≥X, RSI∈[lo,hi], MACD>0,
range≤max). `impulse_speed` отличается только тем, что **trail шире
после входа**. То есть «режим» — это лейбл для выбора trail-k, а не
detector с уникальной логикой обнаружения.

**Следствие:** в TG канал прилетает «📈 Тренд» / «⚡ Быстрое движение»
с разными ATR×, но фундаментально это один и тот же сигнал. Подписчик
не получает доп. инфы.

**Дополнительно:** наш только что внедрённый `trend-1h-chop-filter`
блокирует только `mode == "trend" AND tf == "1h"`. Если `get_entry_mode`
переклассифицирует на `impulse_speed` (price_speed ≥ 1.5 %), фильтр
**не срабатывает**. То есть STRK-подобный кейс при price_speed ≥ 1.5 %
снова пройдёт. Нужен audit: сколько blocked `trend/1h chop` событий
переклассифицировались бы в `impulse_speed` и обошли бы гард.

#### Конфликт B · `IMPULSE` vs `TREND_SURGE` vs `ALIGNMENT` — три «начало тренда»

| | IMPULSE | TREND_SURGE | ALIGNMENT |
|--|---------|-------------|-----------|
| Идея | резкий спайк за 1-3 бара | ускорение slope vs 3-бара назад | плавный устойчивый рост |
| `r1` | ≥ 1.5 % | — | — |
| `r3` | ≥ 2.0 % | — | — |
| MACD | hist > 0 | hist > 0 + растёт 2 бара | hist > 0 **5 баров подряд** |
| slope | > 0 | резкое ускорение | ≥ 0.05 % |
| ADX | — | — | ≥ 15 |
| daily_range | ≤ effective_max | **не проверяется** | ≤ ALIGNMENT_RANGE_MAX |

Все три **могут срабатывать одновременно** — но порядок `if-elif`
в `monitor.py` ~ L4480 такой:

```
if catchup:        ← повторное проигрывание
elif brk_ok:       ← BREAKOUT
elif ret_ok:       ← RETEST
elif entry_ok:     ← TREND / STRONG_TREND / IMPULSE_SPEED
elif surge_ok:     ← TREND_SURGE        ← после entry_ok
elif imp_ok:       ← IMPULSE             ← после surge_ok
elif aln_ok:       ← ALIGNMENT           ← последний
```

**Проблема приоритетов:** `IMPULSE` и `TREND_SURGE` — это самые
ранние детекторы, **НО они в самом конце цепочки**. Если `entry_ok`
выдаёт «trend» с медленным ADX/slope, мы получаем сигнал на trend-режиме
(низкое качество), а не на impulse-режиме (раннее обнаружение).

`ALIGNMENT` единственный без ADX-требования — наоборот, ловит
коины с лагающим ADX. Но он последний → если `entry_ok` уже сработал
(а у trend-режима порог ADX тоже ниже на bull-day), `alignment` не
получит шанса.

#### Конфликт C · `BREAKOUT/15m` vs `IMPULSE` — пересекающиеся условия

`BREAKOUT` требует close > max(high флэта) + vol≥2.0 + MACD растёт + range≤4 %.
`IMPULSE` требует r1≥1.5 % + r3≥2.0 % + body≥X.

На пробое флэта типично: r1≈2-3 %, r3≈3-5 %, vol≈2-3, MACD растёт.
Оба сработают. Текущая precedence: `brk_ok` идёт ПЕРВЫМ → BREAKOUT
выигрывает → trail_k=1.5 (узкий) + max_hold=6 (короткий).

Backtest показал: **BREAKOUT/15m** = 41 % whipsaw rate, avg_pnl −0.33 %,
**0.03 % capture даже на top-20 winners**. То есть BREAKOUT
систематически ворует кандидатов у IMPULSE (который имеет trail_k=2.0
и max_hold=48, лучше держит).

**Это и есть rationale спеки `breakout-15m-disable-spec.md`** —
disable освободит этих кандидатов для IMPULSE / TREND_SURGE,
которые ловят то же движение с лучшим trail-сопровождением.

#### Конфликт D · `RETEST` vs `TREND` на «отскоке»

`RETEST` требует:
- low касался EMA20 за последние 5 баров
- close > EMA20 в текущем баре
- close > prev close
- slope ≥ 0.1 %
- ADX ≥ 20
- RSI < 65

`TREND` (через `check_entry_conditions`):
- close > EMA20 > EMA50
- slope ≥ regime.slope_min (0.3-0.6)
- ADX ≥ regime.adx_min (20-25)
- RSI ≤ regime.rsi_hi (~72)

На свежем отскоке с EMA20 (после касания) **оба условия выполняются
одновременно**. Precedence: RETEST идёт раньше TREND → RETEST берёт.
Но trail у RETEST уже (1.8 vs 2.0), max_hold короче (10 vs 16/48).

**Эффект:** на прибыльном тренде с пуллбэком к EMA20 нас закрывает
по trail RETEST, тогда как TREND-режим продержался бы дольше и
поймал больше движа. Это — структурный leak `EarlyCapture` через
неправильную классификацию.

### 1.3 Сводная карта противоречий

```
                    ┌───────────────┐
catch-up ──────────►│  catchup_mode │
                    └─────┬─────────┘
       Конфликт A: trail-only differentiation
                    ┌─────▼──────────┬──────────────┬─────────────┐
                    │   entry_ok     │              │             │
                    │ → trend /      │              │             │
                    │   strong_trend │              │             │
                    │ → impulse_speed│ ← одинаковый │             │
                    └─────┬──────────┘   detector   │             │
       Конфликт C: brk_ok первый, ворует у IMPULSE  │             │
                    ┌─────▼──┐ ┌─────▼─┐ ┌──────────▼─┐ ┌─────────▼┐
                    │  brk   │ │  ret  │ │   surge    │ │   aln    │
                    │  out   │ │  est  │ │ (dead-code)│ │ (последн)│
                    └────────┘ └───┬───┘ └────────────┘ └──────────┘
       Конфликт D: ret_ok ворует у trend, узкий trail
       Конфликт B: surge/impulse в конце - ловят тоже  ┌─────────┐
                                  что и entry_ok      │ impulse │
                                                       │  cross  │
                                                       │ (легаси)│
                                                       └─────────┘
```

### 1.4 Структурные баги

- **`get_entry_mode` не учитывает 1h_impulse_speed_guard.** На 1h
  отдельный guard `_one_hour_impulse_speed_entry_guard` блокирует
  по daily_range>10/15 %, но reclassification на `impulse_speed`
  внутри `entry_ok` идёт по другому правилу (price_speed ≥ 1.5 %).
  Получается, что часть entries называется `impulse_speed`, но
  должна была быть отброшена 1h-guard’ом.

- **`trend_surge` отключён в основном pipeline.** Detector работает
  только в catchup-сканере, что обнуляет ROI его написания.

- **Bandit видит `mode` как onehot-категорию.** Если `trend` и
  `impulse_speed` — это одна и та же логика с разным trail, бандит
  может находить ложные паттерны: «mode=trend хуже работает», тогда
  как разница только в trail-k.

---

## ЧАСТЬ 2 · Аудит метрик

### 2.1 Что меряется сейчас (после Phase-1 instrumentation)

| Layer | Метрики | Соответствие цели |
|-------|---------|--------------------|
| Coverage | C1 funnel, C2 silent-miss, C3 recall@20 | ✅ хорошо |
| Earliness | E1 TTS, E2 capture_ratio, E3 lead_time | ⚠️ measure entry timing, **не measure trend duration captured** |
| Quality | Q1 FR rate, Q2 whipsaw, Q3 FR drag | ⚠️ measure быстрые потери, **не measure что сделка закрылась рано на тренде** |
| Discrim | D1 precision, D2 msg_rate, D3 UCB sep | ✅ хорошо |
| North-star | EarlyCapture@top20 = 0.074 | ⚠️ хорош на entry-side, **слаб на exit-side** |

### 2.2 Главный gap — нет ни одной exit-side метрики

Текущий `EarlyCapture` = `coverage × capture_ratio × time_lead`. Все три
лeg’а — про точку входа. **Метрика молчит** про:

- Закрылись ли мы **до** разворота тренда (хорошо).
- Закрылись ли мы **на середине** растущего тренда (плохо — left
  money on table).
- Закрылись ли мы **после** разворота с просадкой (плохо — donated
  обратно).

Из найденных backtests (`_validate_2a_dynamic_max_hold.py`,
прошлый раунд): **на top-20 winners 29 trades закрылись по
time-trigger / EMA-weakness, оставив суммарно ~3000 % движения
на столе**. Это — невидимая утечка.

### 2.3 Что нужно добавить — Exit-side metrics

#### **EX1 · `realized_to_potential_capture`**

```
realized = (exit_price − entry_price) / entry_price
potential = (eod_high − entry_price) / entry_price
ratio = realized / potential   ∈ (−∞, 1]
```

**Текущая оценка (грубо, по доступным данным):** для top-20 entries
median ≈ 0.3 (ловим треть от max-движения). Цель: ≥ 0.6.

#### **EX2 · `early_exit_loss_pct`**

```
для exit с reason ∈ {time_max_hold, EMA20_weakness, MACD_warn}:
  если на момент exit price > entry AND price < eod_high * 0.95:
    loss = (eod_high − exit_price) / eod_high * 100
```

Доля «оставленной маржи» на ранних выходах. Цель: ≤ 10 %.

#### **EX3 · `whipsaw_after_recovery_rate`**

```
доля trail-stop exits, после которых цена вернулась выше entry в
течение N (5) баров.
```

Меряет «лишние» trail-выходы. Сейчас Q2 whipsaw = 12.5 %, но Q2
не различает «правильный whipsaw» (price ушла дальше вниз) и
«плохой» (price вернулась).

#### **EX4 · `exit_timing_alignment`**

```
для top-20 winners:
  bars_от_entry_до_exit / bars_от_entry_до_eod_high
```

1.0 = вышли на пике. > 1.0 = после пика. < 1.0 = до пика.
Идеал: median ≈ 0.85-1.0.

### 2.4 Главный gap — нет proxy «sustained trend»

Текущая labelling (label_top20) отвечает на «была ли coin в top-20
по EOD return». Но **бот должен сигналить начало УСТОЙЧИВОГО тренда**,
а не просто «coin будет в top-20».

Top-20 по EOD return может включать coins:
- которые в +30 % за 1 час, потом разворот
- которые медленно растут весь день +25 %
- которые делают 2 свечи памп и dump до nearly zero

Для нашей цели важен **второй тип**. Нужен новый label.

#### **Label предложение: `label_sustained_uptrend`**

```python
def label_sustained_uptrend(klines_intraday) -> int:
    """
    Returns 1 iff:
      - max drawdown after entry_bar relative to entry < 5%
        (тренд не сломался)
      - duration of uptrend (bars from entry to peak) >= 12 (1h × 12 = 12h)
      - eod_close >= entry × 1.05 (минимум +5% в конце)
      - rectangle_uptrend_quality ≥ 0.7   (75% time price > VWAP_session)
    """
```

С таким лейблом: метрика **`recall@sustained` точнее меряет цель**
проекта чем `recall@top20`. Top-20 winner может оказаться pump+dump
(низкий sustained_uptrend score), и сигнал на него — не достижение.

---

## ЧАСТЬ 3 · Гипотезы для роста

### Гипотезы по «максимально ранний BUY на устойчивый тренд»

#### **H1 · Перестроить precedence режимов: detector first, trail_k second**

**Идея:** разделить «обнаружение события» и «выбор сопровождения».

```
# Сейчас (плохо):
brk → ret → entry → surge → impulse → aln → cross
# Где entry = trend|strong_trend|impulse_speed (одна детекция, разные trail)

# Предлагается:
1. detect_event() → {impulse, breakout, surge, alignment, retest, trend}
2. classify_trail() → trail_k, max_hold (отдельно от detector)
3. log signal_mode = detector.name (для семантики в TG)
```

`impulse_speed` исчезает как отдельный режим, его trail-расширение
становится **функцией от price_speed**, применяемой к ЛЮБОМУ режиму.

**Validation:** сравнить за 30 d сегодняшнюю классификацию vs
новую. Должно: (a) не ломать precision (все trail’ы те же); (b)
показать, что N entries с label=trend на самом деле impulse/surge
по новой классификации; они получат лучший trail.

**Δ NS estimate:** +0.02 (через capture).

#### **H2 · Multi-stage entry: candidate → confirmed → fired**

**Идея:** перестать выдавать BUY одной строкой при первом срабатывании.
Вместо этого:

```
Stage 1 «candidate» (≥ 1 признак, top_gainer_prob ≥ 0.5):
  scout silent-watch, без TG
Stage 2 «pre-confirmation» (≥ 2 признака):
  TG soft-alert «👀 Наблюдаю COIN»
Stage 3 «fired» (текущие условия):
  TG🟢 «🟢 BUY COIN»
```

Подписчик получает 4-часовое раннее предупреждение и решает сам
открывать ли превентивно. Бот формально не входит до stage 3.

**Условие:** автоматический watch-list изменяется только если
`top_gainer_prob` ≥ 0.5 на бар. На watchlist 105 coin × 0.05 ≈
5 кандидатов в любой момент → не спам.

**Validation:** для top-20 winners из последних 14 d посчитать,
сколько имели `proba_top20 ≥ 0.5` за ≥ 2 ч до bot’s entry.
По данным `_validate_p3_premove_screener.py`: median lead +4 ч.

**Δ NS estimate:** +0.10 (lead +0.05, и indirectly capture +0.05
через user-action).

#### **H3 · Активировать `trend_surge` в основном pipeline**

**Идея:** detector написан и тестировался, но не подключён.
Пробросить в `if-elif` ветки в `monitor.py` ~ L4501 ПЕРЕД entry_ok:

```python
# precedence:
brk → ret → SURGE → IMPULSE → entry_ok → ALIGNMENT → cross
```

Surge ловит **ускорение slope** — формальное определение «начало
устойчивого тренда» (slope EMA20 резко вырос vs 3 бара назад,
MACD растёт 2 бара подряд).

**Validation:** offline replay 30 d с новым precedence; сравнить
TTS и capture_ratio. Surge-classified entries должны иметь TTS
ниже (раньше) чем те же coin’ы с classification = trend.

**Δ NS estimate:** +0.03 (TTS, capture).

#### **H4 · Заменить top_gainer label на sustained_uptrend label**

**Идея:** см. §2.4. ML модель учится на label_top20 → она учится
ловить и pump-dump, и slow uptrend. Подписчику нужен только slow
uptrend.

**Validation:** обучить parallel модель на label_sustained_uptrend,
сравнить:
- AUC sustained_label > 0.85
- recall@sustained ≥ 95 % на 14 d holdout
- precision на TG-канале (entries that ARE sustained_uptrends) ≥ 30 %
  (текущее D1 = 9 % на label_top20 — но половина из них это pump-dumps,
  поэтому реальное user-utility ниже).

**Δ NS estimate:** не меняет NS напрямую, но меняет смысл NS.
Lossless redefinition: новый north-star = `EarlyCapture@sustained`.

### Гипотезы по «exit в момент максимизирующий прибыль»

#### **H5 · Trailing-only после break-even**

**Идея:** убрать time-based и EMA-weakness exits **после** того
как pos.pnl ≥ 0.5 %. Оставить только:
- ATR-trail (уже есть)
- RSI > 80 reversal candidates (агрессивный)
- 2 closes below EMA50 (катастрофический)

Сейчас при pos.pnl=+1 % бот может выйти по time_max_hold или
EMA20-weakness, оставив 5-20 % движа. На медленных трендах
(alignment) EMA20 пересекает регулярно как часть нормального движения.

**Validation:** counterfactual sim — для всех paired trades
с (entry, exit, eod_high), посчитать: какая была бы реализованная
pnl, если бы выход определялся только trail’ом.

**Δ NS estimate:** +0.05 через capture.

**Risk:** на разворотах trail может пробить позже → drawdown.
Mitigation: tight ATR-trail (k=1.0) после profit_protect_pct.

#### **H6 · Reversal detector на основе orderbook + funding**

**Идея:** для exit использовать НЕ price+EMA индикаторы, а внешний
sentiment-сигнал:
- funding rate > 0.05 % / 8h → перегрев (longs over-positioned)
- top-of-book imbalance < -0.3 → продавцы доминируют
- RSI divergence на 1h
- (опционально) on-chain whale outflow

Эти сигналы **опережают** price reversal на 1-3 бара.

**Validation:** для top-20 winners на 30 d — для каждого entry
посчитать time_until_first_funding_spike vs eod_high_time. Если
funding spike лидирует high на ≥ 1 бар → exit-trigger жизнеспособен.

**Δ NS estimate:** +0.04 через capture.

**Risk:** требует подключения orderbook/funding feed (сейчас нет).
Сложность реализации высокая.

#### **H7 · Adaptive trail_k tied to ATR-vs-daily_range ratio**

**Идея:** на «волатильном дне» (`daily_range > 8 %`) trail_k должен
быть шире, на «спокойном» — уже. Сейчас bandit это адаптивит, но
через 5 arm’ов которые имеют дискретные мультипликаторы (0.7, 0.85,
1.0, 1.2, 1.4). Continuous формула:

```
trail_k_effective = base_trail_k × clip(daily_range / 5.0, 0.7, 1.6)
```

**Validation:** существующий backtest `_backtest_trail_arm_pnl.py`
расширить — сравнить continuous формулу с bandit-выбором arm’а.

**Δ NS estimate:** +0.02 через capture.

#### **H8 · Profit-locked partial exits (50/50 split)**

**Идея:** сигнал «частичный TP» при pnl ≥ 5 %: «закрой 50 %
позиции, остальное держи на trail». В TG: `🟢 LOCK 50% PROFIT
COIN at +5%`. Уменьшает risk-in-trade, психологически легче
для подписчика держать остаток.

**Validation:** counterfactual — если бы 50 % закрывали на
+5 %, какая итоговая pnl среднего trade? На паре «pump+dump»
50 % profit гарантирован, а другая 50 % дорабатывает trail.

**Δ NS estimate:** напрямую не двигает (NS на entry-side).
Но **это улучшение exit-stack’а**, видно в EX1 / EX4.

---

## ЧАСТЬ 4 · Sequenced rollout (приоритезированный)

```
Week 1-2: Re-architecture (H1, H3)
  → mode-precedence rewrite + trend_surge активация
  → metric: TTS median падает на ≥ 1 ч

Week 3-4: Exit-side fixes (H5, H7)
  → trailing-only after break-even
  → continuous trail_k formula
  → metric: EX1 (realized/potential) median растёт с ~0.3 до ~0.5

Week 5-6: Multi-stage entry (H2)
  → 3-stage signal pipeline
  → metric: lead_time E1 median +2 h

Week 7-8: Sustained-uptrend label (H4)
  → новая модель + retraining
  → metric: D1 precision на label_sustained ≥ 30 %

Week 9-10: Exit-side measurement (EX1-4)
  → instrumentation для всех 4 exit-метрик
  → metric: EX1 + EX4 в daily_metrics.jsonl

Week 11-12: H6 experimental
  → reversal detector via funding/OB (опционально, требует data feed)
```

---

## ЧАСТЬ 5 · Acceptance после полного rollout

| Metric | Сейчас | Target | Lever |
|--------|-------:|-------:|-------|
| North-star EarlyCapture@top20 | 0.074 | 0.40 | все H1-H8 |
| TTS median (entry) | +4.94 h | ≤ +0.5 h | H1, H2, H3 |
| Capture_ratio E2 | 0.16 | ≥ 0.50 | H5, H7 |
| Realized/Potential EX1 | (TBD) | ≥ 0.60 | H5, H7, H8 |
| Whipsaw Q2 | 12.5 % | ≤ 5 % | trail-min-buffer (shipped) + H7 |
| Signal precision D1 | 9 % | ≥ 35 % | H2, H4, P5 |
| TG msg_rate D2 | 41/d | 8-12/d | H2 (multi-stage), 4A precision-prune |

---

## Приложение · Что НЕ делать

- **Не добавлять 8-й режим.** Архитектурно сейчас 6-7 пересекающихся —
  добавление ещё одного только умножит конфликты.
- **Не уменьшать ADX-floor на trend** ниже 20. Бэктест чёткий: ADX < 20
  на trend = chop, мы его только что блокировали.
- **Не меняем watchlist.** CLAUDE.md §14: immutable.
- **Не покупать orderbook-feed,** пока не валидирована H6 на public
  funding-rate-only данных.
