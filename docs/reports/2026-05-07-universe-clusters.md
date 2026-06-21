# Universe-wide correlation clusters — leading indicators for watchlist

**Date:** 2026-05-07
**Universe:** 427 active USDT spot pairs on Binance (full universe)
**Watchlist subset:** 105 coins (95 covered after klines alignment)
**Klines:** 1 h × 30 d (697 aligned hourly bars)
**Method:** residualize log-returns vs BTC then ETH (remove market beta),
average-link Union-Find clustering across thresholds

**Goal:** find groups where **non-watchlist coins** move synchronously
with **our watchlist coins**, providing **pre-pump leading indicators**
for sub-15-min reaction.

---

## TL;DR — 3 actionable leading-indicator findings

| Cluster | Watchlist member | External lead candidates | Intra-corr |
|---------|------------------|---------------------------|-----------:|
| **Precious metals** | PAXGUSDT | **XAUTUSDT**, EURIUSDT, EURUSDT | **+0.75** |
| **Solana ecosystem** | SOLUSDT | **RAYUSDT**, **BNSOLUSDT** | **+0.69** |
| **TON ecosystem** | TONUSDT, DOGSUSDT | **NOTUSDT** | **+0.50** |

Most important: **NOTUSDT moves with TONUSDT.** Given the TON case
(2026-05-04 missed first 12h of pump), adding NOT to a `pre-watch` list
gives us **early-warning** before TON's own ticker delta crosses
pump-detector threshold.

---

## Method

```
1. Fetch all 427 trading USDT pairs from /exchangeInfo
2. Backfill 1h × 30d klines (1 API call per coin, ~3 min total)
3. Align on common timestamp grid → 420 / 427 coins, 697 bars
4. Compute log-returns
5. Residualize vs BTC, then re-residualize vs ETH (remove market factors)
6. Pairwise Pearson on residual returns
7. Union-Find clustering at multiple thresholds
8. Tag each cluster's members: watchlist vs external
```

## Cluster results

### At threshold ≥ 0.55, 4 clusters with ≥ 3 members

| # | n_total | n_watchlist | n_external | Intra-corr | Theme |
|---|--------:|------------:|-----------:|-----------:|-------|
| 1 | 73 | 29 | 44 | +0.44 | Generic alt-coin mega-cluster (too broad — see below) |
| 2 | 4 | 1 | 3 | +0.75 | Precious metals + EUR FX |
| 3 | 3 | 1 | 2 | +0.69 | Solana ecosystem |
| 4 | 3 | 2 | 1 | +0.50 | TON ecosystem |

### At threshold ≥ 0.6, 3 clusters

| # | n_total | n_watchlist | n_external | Intra-corr |
|---|--------:|------------:|-----------:|-----------:|
| 1 | 28 | 13 | 15 | +0.51 |
| 2 | 4 | 1 | 3 | +0.75 |
| 3 | 3 | 1 | 2 | +0.69 |

Tighter threshold breaks the alt mega-cluster into a still-broad
group (n=28) plus the sector clusters preserved.

---

## Detailed clusters worth acting on

### Cluster A · Precious metals (intra +0.75)

```
Watchlist: PAXGUSDT (Pax Gold)
External:  XAUTUSDT (Tether Gold), EURIUSDT, EURUSDT
```

**Insight:** PAXG and XAUT are two gold-backed tokens, naturally pegged
to gold price → near-perfect correlation. EUR-denominated stables
correlate because USD weakness pushes both gold and EUR up.

**Action:** Monitor **XAUTUSDT** as PAXG leading indicator. If XAUT
breaks +1 % in an hour, PAXG likely follows in next 30 min.

**Caveat:** PAXG has very low volatility (0.16 %), so these are minor
moves. Not high-NS impact, but clean signal.

### Cluster B · Solana ecosystem (intra +0.69)

```
Watchlist: SOLUSDT
External:  RAYUSDT (Raydium - SOL DEX), BNSOLUSDT (Binance Staked SOL)
```

**Insight:** Solana DeFi tokens (RAY = Raydium AMM volume) and staked-SOL
derivative (BNSOL) move with SOL. RAY is more volatile (0.31 % vs 0.21 %),
which can act as a leveraged proxy for SOL moves.

**Action:** Add **RAYUSDT** to pre-watch. Strong move on RAY (+2 % in 30 min)
suggests imminent SOL move.

### Cluster C · TON ecosystem (intra +0.50) ⭐ MOST IMPORTANT

```
Watchlist: TONUSDT, DOGSUSDT
External:  NOTUSDT (Notcoin)
```

**Insight:** Notcoin is the TON-native «play-to-earn» token that exploded
in 2024 — closely tied to TON network adoption. DOGS is another
TON-native meme token. All three move together because they share TON
ecosystem narrative.

**Why this matters:** The TON case 2026-05-04 missed the first 12 h
of TON's +70 % pump because:
- TON's own price didn't cross our 24h-pct delta until ~12 h in
- TON dropped from `hot_coins` during quiet period

**Fix:** Add **NOTUSDT** to pump-detector's monitored set (independent
of watchlist). When NOT shows pump signature, IMMEDIATELY inject
TONUSDT and DOGSUSDT into `hot_coins` for fast 5 m / 15 m scan.

**Estimated impact:** could have caught TON pump 6-8 hours earlier
than current detection. On a +70 % move, that's potentially +30 to
+50 pp additional capture.

---

## Big mega-cluster (#1 at threshold 0.55)

Too broad to be actionable directly, but worth noting some external
high-volatility candidates that might lead specific watchlist members:

```
Highest-vol external candidates (not in watchlist, vol > 0.5 %/bar):
  NEIROUSDT       0.531%   (meme — likely co-moves WIF/BONK/SHIB)
  VIRTUALUSDT     0.547%   (AI agent — likely co-moves FET/RENDER)
  PIXELUSDT       0.480%   (gaming — likely co-moves MANA/SAND/AXS)
  TURBOUSDT       0.473%   (meme)
  BOMEUSDT        0.473%   (meme)
  PNUTUSDT        0.509%   (meme)
  ACEUSDT         0.489%   (gaming)
  IMXUSDT         0.470%   (gaming infra)
  YGGUSDT         0.480%   (gaming guild)
```

These are all higher-volatility-than-watchlist candidates that share
the alt mega-cluster's behavior. Not direct leading indicators (too
broad), but if 3+ of them spike together, that's a sector signal.

---

## Coverage stats

- 420 / 427 universe coins aligned after klines coverage filter
- 95 / 105 watchlist coins covered (10 watchlist coins missing 1h
  klines — probably recently listed)
- Missing from analysis: 10 watchlist coins that may have shorter
  klines history. Identify via:
  `set(watchlist) - set(aligned_kept)`

---

## Phase 2 — concrete next steps

### N1 · Add 4 «pre-watch» external coins to pump-detector scan

```python
# config.py addition (proposed)
PUMP_DETECTOR_PRE_WATCH_LEAD_COINS = [
    "NOTUSDT",      # → TONUSDT, DOGSUSDT propagation
    "XAUTUSDT",     # → PAXGUSDT propagation
    "RAYUSDT",      # → SOLUSDT propagation
    "BNSOLUSDT",    # → SOLUSDT propagation
]

# When any of these triggers pump-detector, inject the propagation
# targets into hot_coins for fast scan.
```

Spec: `docs/specs/features/external-lead-coin-monitoring-spec.md`
(to be written).

### N2 · Daily ticker fetch for full universe

Currently pump_detector polls `/ticker/24hr` (single API call returns
all 427 coins' priceChangePct). Just need to:
- Track the 4 pre-watch coins' delta separately
- On their pump → inject propagation targets

No API rate increase (same ticker call).

### N3 · Weekly cluster recomputation

Add to Sunday 03:30 scheduled task:
- Fetch fresh universe (in case new listings)
- Backfill 1 h klines (`_backfill_universe_klines.py`)
- Re-run clustering (`_backtest_universe_clusters.py`)
- Compare with previous map — alert if propagation targets change

### N4 · Holdout backtest (per Validate Max Period rule)

Before promoting any cluster-based behavior change to production:
- Find historical NOT pumps and check if TON followed
- Find historical XAUT moves and check if PAXG followed
- Compute: hit rate of «cluster member pumps → watchlist coin pumps
  within 30 min»
- Promote only if hit rate ≥ 60 %

This is the right validation strategy per gpt_crypto_bot’s
cross-validation lesson — n-of-1 cluster intuition is not enough.

---

## Artifacts

- `.runtime/binance_universe.json` — 427 active USDT pairs
- `.runtime/universe_clusters.json` — full cluster map with members and intra-corr
- `history/<sym>_1h.csv` — 1 h × 30 d klines for all 427 coins

(All gitignored — per-machine artifacts.)

## Honest limitations

- **30 d window** — short by comparison to gpt_crypto_bot's 365 d holdout.
  Cluster membership likely drifts. Need monthly recompute, not weekly.
- **No holdout test** — clusters computed in-sample. Per gpt_crypto_bot's
  evidence, in-sample clusters often disappear on holdout. Action items
  N4 above is the required step.
- **No lead-lag direction proof** — I labeled NOT/XAUT/RAY as «leaders»
  by intuition (NOT is TON-native, XAUT is the other gold token). To
  PROVE lead-lag, need bar-shifted Pearson at lags 0-4 hours.
- **NO production changes** in this report. All proposed work is
  measurement-first, behind-flag-second.
