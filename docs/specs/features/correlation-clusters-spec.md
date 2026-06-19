# Correlation clusters discovery

- **Slug:** `correlation-clusters`
- **Status:** shipped 2026-05-07 v2.21.0 (analysis script + first report)
- **Created:** 2026-05-07
- **Owner:** core
- **Related:** `files/correlation_guard.py` (existing reactive guard),
  `pump-detector` (potential lead-coin propagation).

---

## 1. Problem

Криптовалюты движутся **синхронными группами** (L1 alts, memes, DeFi,
L2s, AI-tokens). Bot имеет `correlation_guard` который защищает портфель
от over-concentration **постфактум** (если 2 cluster-members уже в
позициях, третий блокируется). Но он:

- **Не строит группы заранее** — только pairwise correlation в момент
  evaluation.
- **Не показывает группы пользователю** — нет визуализации, нет анализа
  «кто лидер кластера».
- **Не используется pump-detector’ом** — когда BTC-beta member pump’ит,
  бот не активирует сканирование остальных members.

Гипотеза: identification stable groups (corr ≥ 0.7 на 14d log-returns)
→ возможность (a) lead-coin → cluster инжекция, (b) anti-redundancy
(не входить в 3 одинаковых сделки), (c) sector-aware risk management.

## 2. Success metrics

**Phase 1 (this commit):**
- Скрипт строит cluster map на cached 15m klines.
- Identified ≥ 3 distinct clusters with intra-cluster avg corr ≥ 0.7.
- Lead-lag analysis identifies probable «leader» per cluster.
- Report committed under `docs/reports/`.

**Phase 2 (future):**
- Pump-detector использует cluster membership: lead-coin pump →
  inject other members в hot_coins.
- Cluster-aware report metrics: capture per cluster, miss rate per cluster.

## 3. Scope

### In scope
- `files/_backtest_correlation_clusters.py` — analysis script:
  - Load cached `history/<sym>_15m.csv` for watchlist coins.
  - Compute pairwise Pearson on log-returns for the window.
  - Build adjacency graph for threshold pairs.
  - Union-Find clustering across multiple thresholds.
  - Per-cluster: avg intra-correlation, mean lead-lag bars.
  - Output: human-readable table + JSON of cluster assignment.
- Daily report `docs/reports/2026-05-07-correlation-clusters.md`.

### Out of scope (Phase 2)
- Wire clusters into `pump-detector` (separate spec).
- Wire clusters into `correlation_guard` as pre-computed map.
- Time-varying cluster recomputation in production.

## 4. Behaviour / design

### Algorithm

```python
1. Load each sym's 15m klines for last N days.
2. Resample to common timestamp grid; drop syms with <90% coverage.
3. Compute log-returns: r[t] = log(close[t] / close[t-1]).
4. Pairwise Pearson correlation matrix C[i,j].
5. For each threshold t in [0.5, 0.6, 0.7, 0.8]:
   - Edges = {(i,j) : C[i,j] >= t}
   - Connected components via Union-Find = clusters_at_threshold[t].
6. Per cluster:
   - Avg intra-correlation = mean(C[i,j] for i,j in cluster)
   - Lead-lag analysis: for each pair, compute Pearson at lag ∈ {-3,-2,-1,0,+1,+2,+3} bars
     → coin with most "first" wins is the leader.
   - Volatility profile (mean abs return).
```

### Lead-lag identification

Для каждой пары (A, B) в кластере, вычисляем:
```
corr(A[t], B[t+k]) for k ∈ [-3, +3]
best_k = argmax(corr)
```
- best_k > 0 → A leads B by k bars
- best_k < 0 → B leads A by -k bars
- best_k = 0 → simultaneous

«Leader» кластера = coin с самым высоким net lead score across всех пар.

## 5. Config flags & rollback

Нет (analysis script, не меняет behaviour бота).

## 6. Risks

- **Window choice** — 14d может пропустить sector rotation. Mitigation:
  output multiple windows (7d, 14d, 30d).
- **Klines coverage** — некоторые coins имеют меньше cached data.
  Mitigation: skip syms with < 90 % coverage of the window.
- **Spurious clusters** — высокий corr может быть из-за market-wide
  movement (BTC beta). Mitigation: report BTC corr separately.

## 7. Verification

- [x] Spec written.
- [x] Script implemented.
- [x] Script runs on cached klines without errors.
- [x] Output report identifies ≥ 3 stable clusters.
- [x] Lead-coin analysis included.

## 8. Follow-ups

- Wire to pump-detector: when lead-coin Δ > threshold, inject cluster
  followers regardless of their own ticker priceChangePct.
- Add `cluster_id` to `top_gainer_dataset` for ML feature.
- Time-series of cluster membership stability (do clusters drift?).
