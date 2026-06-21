# H-DECOUPLE validation — do decoupled coins catch idiosyncratic top-movers?

**Date:** 2026-05-07
**Hypothesis:** the bot is tuned to beta (moves-with-market). Coins that
DETACH from the market (low trailing correlation) and run on their own
narrative are the top-gainers it under-captures. If low trailing-corr
predicts a forward big move BEYOND volatility, the signal is real and
real-time gateable.
**Method:** 365d × 1h klines, train/holdout split, vol-controlled.
Script: `files/_backtest_decoupling.py`. **Status:** research-only.

---

## TL;DR — VALIDATED on universe, weaker but directionally confirmed on watchlist

Decoupling predicts forward idiosyncratic big-moves (≥10% / 24h), the
effect is **monotonic and holds out of sample**, and survives a
volatility control — but ONLY among already-volatile coins. The precise
actionable signal is **"high-vol AND decoupled"**, not low-corr alone.

This is the project's NS tail (catching top-20 gainers), NOT per-trade
PnL — mean forward return is negative in every bucket; the signal lives
in the big-move *probability*, exactly what an early-alert bot wants.

---

## Universe (397 coins, Dec 24→Jun 20, 135k samples)

Big-move% (≥10%/24h) by trailing-corr quintile (Q1=decoupled):

| Bucket | TRAIN big% | HOLDOUT big% | mean vol |
|--------|-----------:|-------------:|---------:|
| Q1 DECOUPLED | 2.95 | **4.16** | 0.0137–0.0154 |
| Q2 | 2.03 | 2.58 | |
| Q3 | 1.99 | 1.96 | |
| Q4 | 1.92 | 1.35 | |
| Q5 COUPLED | 1.78 | 0.98 | 0.0085 |

Monotonic decline Q1→Q5 in holdout — decoupled coins are **4.2×** more
likely to make a big move than coupled ones.

### Vol-controlled (decoupling beyond volatility?)
Low-corr vs high-corr big-move rate WITHIN each vol tercile:

| Vol tercile | TRAIN lift (pp) | HOLDOUT lift (pp) |
|-------------|----------------:|------------------:|
| low | −0.84 | −0.01 |
| mid | +0.23 | +1.04 |
| **high** | **+1.85** | **+3.14** |

A **high-vol + decoupled** coin makes a big move **5.64%** of the time in
holdout vs **0.98%** for coupled — a real edge on top of volatility.
Low-vol decoupled = just noise (null/negative lift), correctly rejected.

---

## Watchlist-only (95 coins, full 365d, 68k samples) — the tradeable pool

The watchlist is dominated by high-cap majors that rarely decouple
(Q5 corr 0.90–0.99), so variation is compressed and the effect is weaker:

| Bucket | TRAIN big% | HOLDOUT big% |
|--------|-----------:|-------------:|
| Q1 DECOUPLED | 2.94 | 2.29 |
| Q5 COUPLED | 2.77 | 1.15 |

Vol-controlled high-vol tercile: HOLDOUT lift **+2.40pp** (decoupled 3.86%
vs coupled 1.46%) — same direction, ~2.6×. BUT train high-vol lift was
**−0.59pp** (noisy) → the watchlist signal is NOT bulletproof; it is a
soft prior, not a hard gate.

---

## Concrete recall on REAL watchlist top-gainers (no lookahead)

Translating the edge into "bot could pre-flag N of M". Flag = vol_pctile
≥0.66 AND trailing_corr ≤0.60, computed at day-start from trailing data
only. Outcome = that day's actual top movers.
Script: `files/_backtest_decoupling_recall.py`.

**Loose target (top-20/95 per day OR ≥10%, base rate ~21%):** flag lift
only ×1.1–1.18 — near useless. The loose definition is mostly "top
quintile", not rockets.

**Strict rockets (top-5/day OR daily ≥12%, base rate ~5.5%):**

| Split | real rockets | flag recall | flag precision | base rate | LIFT |
|-------|-------------:|------------:|---------------:|----------:|-----:|
| TRAIN | 957 | 11.5% (110/957) | 12.9% | 5.8% | **×2.25** |
| HOLDOUT | 948 | 22.5% (213/948) | 10.2% | 5.5% | **×1.87** |

**Plain-language answer:** a flagged coin-day is ~**2× more likely** to be
a real rocket than a random watchlist coin-day. The flag pre-identifies
**~11–22% of all real rockets** before the move. But absolute precision is
low (~10–13%) — **~8 misses per hit** — so it is a *priority signal*, not
a detector you can trust on its own.

## Verdict & proposed use

**VALIDATED as a soft signal.** Concrete, gateable, holdout-stable on the
universe; weaker but directionally confirmed on the watchlist.

### Proposed: `decoupling_score` shadow feature (NOT a hard gate)
- `trailing_corr_7d` = coin's 168h correlation to the market basket
  (known at decision time).
- `decoupling_score = (vol_percentile high) AND (trailing_corr low)`.
- Add to the **entry bandit context** and the **candidate ranker** as a
  feature — let the learner weight it. A high-vol decoupled watchlist
  coin gets an early-alert priority bump.
- Reward on the NS (early top-20 capture), NOT per-trade PnL.

### Why a feature, not a gate
Train high-vol watchlist lift was negative — a hard "only enter if
decoupled" rule would over-block in some regimes (the §7 over-block
lesson). Feed it to the bandit so it can be regime-conditioned.

### Gate before any wiring
Bandit shadow replay must show improved early top-20 capture (NS) without
recall loss, per the §4a / RM-22 pattern.

---

## Why this beats the lead-lag idea

| Hypothesis | Result |
|------------|--------|
| Cluster→cluster lead-lag (1h) | NULL — one market |
| BTC→alt lead-lag (15m) | real but ~90% intra-bar, net 0.03, fee-killed |
| Sector→oldgen lead-lag (15m) | tiny (net 0.015–0.022), illiquid laggards |
| **Decoupling → big-move (24h)** | **+3.14pp vol-ctrl lift, holdout-stable, NS-relevant** |

Decoupling is the one signal here with a holdout-stable, economically
meaningful effect on the metric the bot actually targets (top-mover
capture), because it predicts the idiosyncratic *tail*, not the mean.

## Honest limitations

- Watchlist effect is weaker than universe (majors rarely decouple) and
  train high-vol tercile was noisy → soft prior only.
- Mean forward return negative everywhere — this is a rare-tail predictor;
  most high-vol decoupled coins still don't pop. 5.64% hit rate means
  ~18 misses per hit. Useful for *ranking/priority*, useless as certainty.
- `trailing_corr_7d` needs to be computed live (168h rolling) — cheap, but
  a new live feature to maintain.
- 24h forward horizon; intraday earliness within the day not tested here.

## Artifacts
- `files/_backtest_decoupling.py` — this backtest
- `history/<sym>_1h_365d.csv` (gitignored)
