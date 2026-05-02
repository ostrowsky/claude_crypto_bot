# EX1 · realized-to-potential capture metric

- **Slug:** `ex1-realized-potential`
- **Status:** draft → implementing (instrumentation only, no behaviour change)
- **Created:** 2026-05-02
- **Owner:** core
- **Related:** `2026-05-01-mode-audit.md` §2.2 (нет ни одной exit-side
  метрики), `metrics-framework-spec.md` (Phase 1 — все 13 на entry-side).

---

## 1. Problem

Все 13 метрик framework-спеки измеряют точку входа. Backtest 2A
показал: 29 trades на top-20 winners закрылись по time/EMA-weakness
с **~3000 % суммарного оставленного движа**. Эту утечку никакая
из существующих метрик не показывает.

Нужна метрика, отвечающая: **«сколько от потенциального движа мы
зафиксировали».**

## 2. Success metric (для самой метрики)

Метрика измеряемая, не оценочная — по сути это observability.
Acceptance: метрика воспроизводимо считается на 30 d данных и
попадает в `report_metrics_daily.py`.

После 7 d сбора — baseline для всех будущих exit-side improvements
(H5 trailing-only, H7 continuous trail_k, H8 partial profit-lock).

## 3. Scope

### In scope
- Скрипт `_backtest_ex1_realized_potential.py`.
- Интеграция в daily aggregator `report_metrics_daily.py`.
- Документация ограничений (mocked eod_high из `eod_return_pct`,
  без intraday klines).

### Out of scope
- Изменения в exit-логике (H5-H8 — отдельные специ).
- Получение реального intraday-high через klines fetch (TBD,
  отдельная спека `klines-fetch-spec`).

## 4. Behaviour / design

### Формула

```
для каждого paired (entry, exit) с (date, sym) ∈ top-20:
  realized_pct  = (exit_price - entry_price) / entry_price * 100
  potential_pct = max(
      eod_return_pct,          # full-day move (proxy of close-vs-prev-close)
      tg_return_4h,            # snapshot 4h move
      tg_return_since_open     # snapshot since-open move
  )                            # max — best available proxy of intraday high
  ex1 = realized_pct / potential_pct  if potential_pct > 0 else nan
  clamp(ex1, -0.5, 1.5)
```

Распределение: median, p25, p75, mean, % с ex1 ≥ 0.5.

**Ограничение:** `potential` — нижняя оценка реального intraday-high
(не имеем klines). Реальный EX1 будет ≤ предложенного. Цель
≥ 0.6 — конкретное число пересмотрим, когда klines-fetch добавим.

### Per-mode breakdown

```
Mode/TF              n   median EX1   mean EX1   ex1>=0.5 %
trend/15m
trend/1h
impulse_speed/15m
... etc
```

Чтобы видеть, какой режим / TF лучше «дожимает» сделки до high.

### Per-exit-reason breakdown

```
Exit reason             n   median EX1   share of total losses
ATR-trail               ...
time_max_hold           ...   ← ожидаемо лучше (ловим high)
EMA20_weakness          ...   ← ожидаемо хуже (рано вышли)
RSI_overbought          ...
...
```

Покажет, какие exit-причины систематически бросают деньги.

## 5. Config flags & rollback

Никаких. Это чистая измерительная задача, не меняет behaviour.

## 6. Risks

- **eod_return_pct в датасете может быть в decimal или %.**
  В предыдущих скриптах использовали heuristic `if abs(eod) > 5
  then % else decimal`. Сохранить тот же подход.
- **`potential` иногда недо-оценивает реальный high** на pump+dump
  coins (pump на +50 %, dump до 0). EX1 в таких случаях может
  выглядеть лучше реального — это будет ясно после klines-fetch.

## 7. Verification

- [ ] Скрипт запускается без ошибок на 30 d.
- [ ] METRIC_JSON на выходе — для daily-aggregator.
- [ ] Daily aggregator подхватывает new метрику.
- [ ] Baseline-snapshot записан в `2026-05-02-ex1-baseline.md`.

## 8. Follow-ups

- `klines-fetch-spec` для получения настоящих intraday-high.
- EX2 / EX3 / EX4 — остальные 3 exit-side метрики из mode-audit
  (отдельные специ).
- Cross-correlation EX1 с bandit trail-arm choice — должна быть
  сильная связь.
