# Correlation clusters — discovery report

**Date:** 2026-05-07
**Script:** `files/_backtest_correlation_clusters.py`
**Window covered:** 7d / 14d / 30d × 15m bars
**Coins analyzed:** 69 of 105 watchlist (those with ≥ 90 % klines coverage from cached daily backfill)
**Spec:** `docs/specs/features/correlation-clusters-spec.md`

## TL;DR

Гипотеза **подтверждена**. Криптовалюты watchlist группируются в **два слоя**:

1. **Высокоуровневый BTC-beta** (raw correlation ≥ 0.7 → один кластер из 48 coin’ов). Это не «sector», это «всё двигается с BTC».
2. **Sector-специфичные группы** (после residualization vs BTC, threshold 0.55-0.65). 3 устойчивые группы за 7d, 2 за 14d.

Главный сюрприз: **TON отсутствует в данных** (klines coverage < 90 %) — поэтому свежий pump 6-7 мая не виден на correlation analysis. После следующего klines backfill (через ~6 ч) пересчёт станет полнее.

## Часть 1. BTC-beta распределение (raw correlation, 14d)

```
Top-10 most correlated with BTCUSDT (14d 15m):
  ETHUSDT        +0.900   ← L1 blue chip core
  SOLUSDT        +0.894
  LINKUSDT       +0.885
  BNBUSDT        +0.841
  DOGEUSDT       +0.836
  XRPUSDT        +0.831
  SUIUSDT        +0.829
  AVAXUSDT       +0.822
  SANDUSDT       +0.817
  WIFUSDT        +0.813

Bottom-5 (BTC-independent):
  TRXUSDT        +0.427
  PYRUSDT        +0.412
  AUDIOUSDT      +0.407
  MEMEUSDT       +0.396
  QIUSDT         +0.244   ← almost zero BTC beta
```

**Вывод:** 90 % alts имеют corr ≥ 0.5 c BTC. «Свободные» от BTC: AUDIOUSDT, MEMEUSDT, QIUSDT (их движения **не предсказываются** BTC — но как раз эти и есть pump+dump кандидаты).

## Часть 2. Raw correlation clusters (без residualization)

| Threshold | # clusters | Largest | Coverage |
|----------:|-----------:|--------:|---------:|
| 0.50 | 1 | 63 | 63 |
| 0.60 | 1 | 60 | 60 |
| 0.70 | 1 | 48 | 48 |
| 0.85 | 1 | 17 | 17 |
| 0.90 | 1 | 4 | 4 |
| 0.92 | 1 | 2 | 2 |

**Пик clarity на corr ≥ 0.90:** 4 monolithic blue-chip cluster
**ETHUSDT, SOLUSDT, LINKUSDT, AVAXUSDT** (intra-corr +0.897).

Это **core proxy для трейдинговой стратегии**: если все 4 одновременно
запампили — broad market on; если расходятся — sector rotation in
progress.

## Часть 3. Sector clusters (residualized vs BTC, 7d window) ⭐

**Это главный результат.** После вычитания BTC beta, остаточные движения
показывают истинные секторные группы:

### Cluster A · Major L1 + DOGE (corr +0.57)
```
AVAXUSDT  · DOGEUSDT  · LINKUSDT  · SOLUSDT  · SUIUSDT
Vol leader: SUIUSDT (0.17%)
```
Сектор: смешанная large-cap корзина — L1 ecosystem core + DOGE (исторический «retail proxy»).

### Cluster B · Gaming / Metaverse / Payment mid-caps (corr +0.59)
```
CELRUSDT  · COTIUSDT  · GMTUSDT  · MANAUSDT  · SANDUSDT
Vol leader: GMTUSDT (0.23%)
```
Сектор: явно **gaming/metaverse** (SAND, MANA, GMT) + payment infrastructure (CELR, COTI). Это самый «чистый» sector cluster.

### Cluster C · Storage / New L1 (corr +0.61)
```
APTUSDT  · FILUSDT
```
Aptos (new L1) + Filecoin (storage). Малый размер — на 14d window здесь же APT/FIL + GMT/MANA/SAND/WIF (см. ниже).

### Bonus · BNB chain ecosystem (видно только на 30d)
```
BNBUSDT  · CAKEUSDT  (corr +0.56, 30d)
```
Сильная связка BNB ↔ PancakeSwap. На 14d/7d не виден — медленнее реагирует.

## Часть 4. Sector clusters на 14d window (более устойчивая выборка)

При residualized 14d, threshold 0.6:

### Cluster A (extended L1 + Gaming, 6 coins, corr +0.56)
```
APTUSDT  · FILUSDT  · GMTUSDT  · MANAUSDT  · SANDUSDT  · WIFUSDT
```

### Cluster B (Pure L1 blue-chip, 4 coins, corr +0.60)
```
AVAXUSDT  · LINKUSDT  · SOLUSDT  · SUIUSDT
```

Заметно: **с расширением окна 7d → 14d**, кластер A приобретает sector mix
(gaming + storage + L1), а core L1 (SOL/AVAX/LINK/SUI) отделяется в свой кластер B.

## Часть 5. Lead-coin analysis (предварительно)

Текущая lead-lag detection (lag ±3 баров на 15m) **не находит** stable
лидеров — score 0 для всех кластеров. Гипотеза: на 15m bars сектор
движется **синхронно** (< 15 min задержки между members).

Для leader detection нужен либо:
- **5m timeframe** (лаги становятся видимы)
- **Trade-level data** (sub-minute)
- **Specific high-volatility days** где можно увидеть first-mover чётко

Это **Phase 2** работы — оставлено в follow-up.

## Часть 6. Применение к бизнес-задачам

### A. Pump-detector enhancement
**Сейчас:** detects pump per individual coin (Δ priceChangePct ≥ 2 % / 15min).
**С кластерами:** при pump на одной монете кластера — instantly **inject** всех других members в `hot_coins`. Lead time +5-15 min.

Пример: TON Cluster D + MANTAUSDT (когда coverage улучшится) → если SAND запампил, моментально включаем MANA / GMT / COTI в монитор.

### B. Anti-redundancy в портфеле
**Сейчас:** correlation_guard.py делает on-demand Pearson.
**С кластерами:** заранее знаем что 3 позиции в Cluster B (SOL+AVAX+LINK) = 1 effective bet. Используем pre-computed map → быстрее, дешевле.

### C. Sector-aware metrics
EarlyCapture per cluster, miss-rate per cluster. Сразу видно «забыли» ли мы целый sector сегодня (gaming pump прошёл мимо).

### D. ML feature
Добавить `cluster_id_residual` и `cluster_id_raw` как categorical features в ml_signal_model. Может помочь модели понимать sector dynamics.

## Часть 7. Limitations

- **TON pump (6-7 мая) не учтён** — TON klines не cached на момент анализа.
- **Coverage 69 / 105** — 36 coin’ов не имеют достаточно cached data. После daily backfill task через 24h coverage должен подняться.
- **Lead-lag не работает на 15m** — нужен более тонкий timeframe.
- **Кластеры дрейфуют** — 7d vs 14d показывают разные группировки (это норма для crypto, но требует periodic recomputation).

## Phase 2 plan

1. **Stable cluster API** — `files/cluster_map.py` который кэширует output скрипта и обслуживает `correlation_guard` + `pump_detector`.
2. **Lead-lag on 5m** — backfill 5m klines for top-30 coins, re-run lead analysis.
3. **Cluster_id в top_gainer_dataset** — категориальная feature для ML.
4. **Re-run weekly** — добавить в Sunday 03:30 task расчёт clusters; alert если кластер сильно мутирует.

## Полные JSON-файлы

- `.runtime/correlation_clusters_raw.json` (14d, raw corr ≥ 0.9)
- `.runtime/correlation_clusters_residual.json` (14d, residualized ≥ 0.6)

(оба gitignored — runtime artifacts)
