# Cluster lead-lag dependencies — does one cluster's move drive another?

**Date:** 2026-05-07
**Hypothesis under test (user):** movements of one cluster depend on
another — e.g. the BTC cluster's move triggers other clusters. Find all
such dependencies.
**Method:** max-period klines, residualized + raw Union-Find clustering,
lagged cross-correlation with train/holdout stability gate, event study.
**Status:** research-only. NO production change.

---

## TL;DR

1. **At the cluster (sector) level on 1h bars — NO lead-lag.** Crypto is
   effectively **one cluster**: even at raw corr ≥ 0.7, 76 coins (incl.
   BTC) sit in a single group. The only separate groups are dead/isolated
   (LUNA/LUNC/USTC Terra, GAS/NEO/ONT/QTUM old-L1). Cross-cluster
   net-asymmetry is ~0 — sectors move **contemporaneously**, none leads.

2. **1h is too coarse.** BTC→alt propagation in crypto lives in minutes;
   a 1h bar already absorbs it. Re-ran at **15m**.

3. **At 15m, BTC DOES weakly lead alts — but mostly intra-bar.**
   - Contemporaneous corr BTC↔alt-basket = **+0.82** (move together).
   - Forward net-asymmetry BTC→basket: **+0.029 @ 30m, +0.035 @ 45m**,
     positive in BOTH train and holdout. Real but tiny.
   - Event study: a big BTC bar (top-decile, |ret|≥0.32%) moves the alt
     basket **+0.56% in the SAME 15m bar**, then only **+0.07% over the
     next 60m** (pos 52%). Propagation is ~90% within the bar.

4. **The actionable finding is per-coin:** a subset of **lower-liquidity
   laggard alts reliably follow BTC by one 15m bar**, stable across
   train/holdout. These are the only real "BTC-cluster-leads-X"
   dependencies in the data.

---

## Cluster-level (1h, max period 365d req → 4283 aligned bars, Dec 24→Jun 20)

### Residualized clustering (sectors) @ 0.5
3 clusters: one mega (32, all watchlist) + GAS/NEO/ONT/QTUM + LUNA/LUNC/USTC.
**Directed lead dependencies (train+holdout stable): NONE.**

### Raw clustering (BTC forms majors cluster)
| Threshold | Clusters ≥3 | Structure |
|-----------|-------------|-----------|
| 0.6 | 2 | C1 = 143 coins (incl. BTC) · C2 = LUNA/LUNC/USTC |
| 0.7 | 1 | C1 = 76 coins (incl. BTC) — everything else isolated |

BTC-cluster vs LUNA cluster: bestlag 5h, fwd +0.016 / rev +0.017 →
**symmetric, no lead.** Contemporaneous corr 0.57–0.66.

**Conclusion:** on 1h bars there is no exploitable cluster→cluster lead.
The market is one beta-driven blob; the few separable clusters are
illiquid relics with no predictive coupling.

---

## Coin-level (15m, 90d, 95 coins aligned, Mar 22→Jun 20)

### A) BTC → alt-basket lagged cross-correlation

| Horizon | fwd | rev | net | train | holdout |
|---------|----:|----:|----:|------:|--------:|
| contemp (k=0) | **+0.819** | — | — | — | — |
| +30m | +0.025 | −0.004 | **+0.029** | +0.017 | +0.030 |
| +45m | +0.016 | −0.019 | **+0.035** | +0.025 | +0.010 |
| +60m | +0.014 | −0.015 | +0.028 | +0.038 | −0.002 |

ETH→basket: same shape, even weaker (net ≤ +0.013). BTC is the stronger
leader of the two, peaking at 30–45m.

### B) Event study — alt-basket forward return after a big BTC bar

Big BTC bar = top-decile |ret| ≥ 0.32% (n≈430 up, 435 down).

| | next 15m | next 30m | next 60m |
|---|---:|---:|---:|
| BTC **UP** big → basket | −0.006% | +0.025% | **+0.070%** (pos 52%) |
| BTC **DN** big → basket | −0.010% | −0.036% | −0.045% (pos 49%) |
| same-bar (ref, UP) | **+0.563%** | | |

The 0.56% same-bar co-move vs 0.07% next-hour drift confirms: **alts
move WITH BTC inside the 15m bar**; only a small same-direction tail
leaks into the next hour.

### C) Per-coin BTC-lead (stable train+holdout, net > 0.02) — the real dependencies

| Coin | best lag | fwd corr | net asym |
|------|---------:|---------:|---------:|
| BNTUSDT | 15m | +0.087 | **+0.117** |
| SNXUSDT | 15m | +0.016 | +0.067 |
| ZILUSDT | 15m | +0.039 | +0.067 |
| QIUSDT | 15m | +0.042 | +0.065 |
| KSMUSDT | 15m | +0.023 | +0.065 |
| AUDIOUSDT | 15m | +0.034 | +0.060 |
| LQTYUSDT | 15m | +0.042 | +0.056 |
| METISUSDT | 15m | +0.023 | +0.056 |
| GLMUSDT | 15m | +0.040 | +0.053 |
| UMA, GMT, TRX, FIL, MANA, SHIB, POL, WIF, MEME … | 30–60m | — | +0.040–0.051 |

25 coins show stable positive BTC-lead. The strongest are
**lower-liquidity laggards** (BNT, SNX, ZIL, QI, KSM, AUDIO, LQTY, METIS,
GLM) — they react to a BTC move one 15m bar later. Higher-cap names
(SHIB, POL, TRX, FIL, WIF) lag 30–60m with smaller asymmetry.

---

## Answer to the hypothesis

**"Movement of the BTC cluster triggers other clusters" — PARTIALLY TRUE,
but not at cluster granularity and not large enough for a standalone rule:**

- ❌ **Cluster→cluster (1h):** false. One blob, no lead.
- 🟡 **BTC→alt-basket (15m):** real but tiny (net 0.03, +0.07%/60m drift).
- ✅ **BTC→specific laggard coins (15m):** real & holdout-stable for ~25
  coins, strongest for illiquid laggards (BNT/SNX/ZIL/QI/KSM/AUDIO/METIS/GLM).

This matches gpt_crypto_bot's 365d conclusion: cluster/BTC confirmation is
a **context/timing feature, not a BUY trigger**. The economically-honest
read: most BTC→alt propagation is complete within 15 minutes, so by the
time our scan reacts the move is already in.

---

## Inter-sector lead-lag — ALL clusters, not just BTC (15m, 90d)

Data-driven Union-Find collapses to one mega-cluster at every threshold
(single-linkage chaining), so sectors were defined by narrative
(L1_major / L2 / defi / meme / gaming / ai / ton_eco / sol_eco / oldgen)
and all ordered pairs tested on raw 15m sector-mean returns.
Script: `_backtest_sector_lead_lag.py`.

### Contemporaneous coupling (k=0)
Every sector pair sits at **+0.70 … +0.93** — one market. **One exception:
`ton_eco` is decoupled** (+0.48…+0.57 with everything) — it trades on its
own narrative (TON/NOT/DOGS), the only sector not glued to beta.

### Stable directional dependencies (train+holdout)
Only 3 survive the holdout gate — **all point INTO `oldgen`**:

| Leader | → Follower | lag | net asym | train | holdout |
|--------|-----------|----:|---------:|------:|--------:|
| ai | → oldgen | 15m | +0.022 | +0.008 | +0.027 |
| sol_eco | → oldgen | 30m | +0.018 | +0.014 | +0.014 |
| L1_major | → oldgen | 15m | +0.015 | +0.004 | +0.017 |

### Sector leadership score (mean net-asymmetry vs all others)
| Rank | Sector | Score | Role |
|------|--------|------:|------|
| 1 | **sol_eco** | +0.0095 | leads |
| 2 | **meme** | +0.0080 | leads |
| 3 | **L1_major** | +0.0057 | leads |
| 4 | ai | +0.0027 | leads |
| 5 | defi | +0.0016 | ~neutral |
| 6 | ton_eco | −0.0040 | decoupled |
| 7 | L2_scaling | −0.0052 | lags |
| 8 | gaming | −0.0065 | lags |
| 9 | **oldgen** | **−0.0117** | universal laggard |

### Inter-cluster dependency structure (the answer)
1. **Fast sectors that move first:** `sol_eco`, `meme`, `L1_major`.
2. **Universal laggard:** `oldgen` (ZIL/ZRX/BAT/GLM/BNT/AUDIO/CHZ/JASMY/
   CELO/QNT) — follows everyone by **15–30 min**, the strongest and most
   consistent dependency in the data. Coherent with the BTC-coin finding
   (the same illiquid names topped the BTC-laggard list).
3. **`gaming` and `L2` also lag** the majors slightly.
4. **`ton_eco` is the odd one out** — weakly coupled to the market, its
   own driver. (My earlier 30d TON→NOT cluster was a *within-sector* link,
   not a cross-cluster lead; it does NOT lead the broad market.)

**So inter-cluster dependencies DO exist** — fast sectors
(sol_eco/meme/L1) → `oldgen` laggards at 15–30 min — but the magnitudes
(net 0.015–0.022) are economically marginal, same caveat as the BTC test.

## Proposed use (shadow-only, gated)

### L1 · BTC-impulse timing feature for the laggard set
For the 9 strongest laggards (BNT, SNX, ZIL, QI, KSM, AUDIO, LQTY, METIS,
GLM): when BTC prints a big bar (|ret|≥ top-decile ≈ 0.32%/15m), add a
shadow feature `btc_impulse_lead` to the entry bandit for these coins on
the **next** 15m bar. Direction = sign(BTC bar).

Gate: this is a 0.05–0.12 correlation edge — must show it improves
EARLY top-mover capture (NS), not per-trade pnl (fees would eat 0.07%).
Validate via bandit shadow replay before any wiring.

### L2 · Do NOT build cluster→cluster gating
The 1h cluster lead-lag is null. Any "wait for BTC cluster" gate would
add latency with no predictive payoff. Explicitly rejected.

---

## Honest limitations

- **Magnitudes are economically tiny.** Net-asymmetry 0.03–0.12 on
  hourly/15m returns is a weak edge; round-trip fees (~0.07%) exceed the
  60m drift. Only defensible as an EARLINESS/timing nudge (per project
  objective: earliness, not pnl), never as a profit rule.
- **15m window only 90d** (vs 365d for 1h). Laggard list may drift; the
  train/holdout split (45d/45d) is the stability guard but is short.
- **Per-coin nets, while stable-signed, are small** — BNT's +0.117 is the
  only one that's clearly above noise; the rest cluster near 0.05.
- **No execution model.** A 15m-lag signal needs sub-15m reaction to be
  usable; our scan cadence may already miss it.
- **NO production change** in this report.

## Artifacts

- `files/_backfill_universe_365d.py` — 365d×1h backfill (427 coins)
- `files/_backfill_watchlist_15m.py` — 90d×15m backfill (watchlist+peers)
- `files/_backtest_cluster_lead_lag.py` — cluster lead-lag (1h)
- `files/_backtest_btc_lead_15m.py` — BTC→alt lead-lag + event study (15m)
- `files/_backtest_sector_lead_lag.py` — all-sector inter-cluster lead-lag (15m)
- `history/<sym>_1h_365d.csv`, `history/<sym>_15m_90d.csv` (gitignored)
