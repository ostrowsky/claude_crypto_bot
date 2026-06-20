# Cross-bot validation: gpt_crypto_bot cluster audit vs our hypotheses

**Date:** 2026-05-07
**Author:** core
**Status:** validation review (read-only — gpt_crypto_bot is sibling
project, not editable per CLAUDE.md §1)

## 0. What I'm validating

Two reports from `D:\Projects\gpt_crypto_bot\docs\reports\`:

1. `price-pattern-clusters-2026-06-19.md` — 365 d holdout audit of price
   pattern clusters with proper train/test split.
2. `cluster-derived-bot-hypotheses-2026-06-19.md` — 6 hypothesis backlog
   derived from that audit.

Against **our own** recent work:
- `docs/reports/2026-05-07-correlation-clusters.md` (14 d in-sample)
- `docs/reports/2026-05-07-ns-hypotheses-roadmap.md` (9 NS hypotheses)

## 1. Strength of their evidence

Methodologically their audit is **harder evidence than mine**:

| Dimension | Ours (14 d) | Theirs (365 d holdout) |
|-----------|-------------|------------------------|
| Period | 14 d single-window | 8 mo train + 4 mo holdout |
| Residualization | BTC only | BTC + ETH + cross-sectional mean |
| Coverage | 69 / 105 (klines limited) | 97 / 103 (1h klines) |
| Out-of-sample test | none — in-sample only | proper holdout split |
| Validation backtest | not done | 3 momentum-horizon backtests with 11 k trades |
| Cluster method | Union-Find on threshold | Average-link hierarchical |

Their **train residual intra-corr = +0.23**, **holdout residual intra
+0.17**, with **inter +0.005 negative**. Spread persists out-of-sample
→ **clusters are real, not in-sample artefacts**.

Our 14 d analysis at corr ≥ 0.6 found similar sector groups (gaming /
L1 / BNB ecosystem) — qualitatively consistent. **Same phenomenon,
their measurement is more rigorous.**

## 2. The big finding that overrules my H-CV1

Their **3 momentum-horizon backtests on 365 d holdout**:

| Horizon | Variant | Trades | Total return | Avg return | Win % |
|---------|---------|-------:|-------------:|-----------:|------:|
| 3h→6h | baseline | 6 341 | **−2 325 %** | −0.37 % | 38.8 |
| 3h→6h | **cluster confirm** | 3 534 | **−1 350 %** | −0.38 % | 39.0 |
| 6h→12h | baseline | 3 530 | −1 240 % | −0.35 % | 40.5 |
| 6h→12h | **cluster confirm** | 2 154 | −432 % | −0.20 % | 41.0 |
| 12h→24h | baseline | 1 872 | −1 089 % | −0.58 % | 38.8 |
| 12h→24h | **cluster confirm** | 1 116 | −516 % | −0.46 % | 38.8 |

«cluster confirm» = enter only if N peers in same cluster are also moving.
**It loses money on every horizon.** It cuts trades by ~40 %, improves
loss/trade slightly on longer horizons, but never beats break-even.

### What this means for **my H-CV1** (cluster pump propagation)

My proposal: when one cluster member pumps, inject **all** members into
`hot_coins` for immediate entry consideration.

**This is essentially the cluster-confirm hypothesis they tested and
rejected at 365 d × 11 k trades.** My estimate of "+5-10 pp coverage"
was based on intuition + their cluster B (gaming) example. **Their
evidence shows the directional shift is negative on the holdout.**

I should **demote H-CV1** from P1 (week 1) to either:
- **Rejected outright** (most conservative; their evidence is strong), OR
- **Re-scoped to exposure control only** (their P1/P2 direction), which
  was already in our `correlation_guard.py` as reactive logic but not
  cluster-precomputed.

**Honest call:** their evidence supersedes mine. **H-CV1 in current form
is not promotion-ready.**

## 3. Critique of their conclusions (in fairness)

Their evidence is strong but has gaps worth flagging:

### Weakness 1: baseline strategy was naive momentum

```
"baseline top momentum" = top-momentum signal × hold N hours
```

This is **already a losing strategy** (−0.37 % avg / trade). Cluster
filter doesn't fix bad signal. They didn't test on a **good** base
strategy (e.g. their actual production bot's entry logic).

**Implication:** their cluster-confirm rejection is valid **for momentum
filtering** but doesn't directly invalidate cluster propagation as a
**signal-source boost** to a separate working strategy. Our bot's
entries use 7 modes with ranker + bandit — different enough that this
benchmark may not apply directly.

**But:** absent counter-evidence, I should default to their finding,
not hope mine is special.

### Weakness 2: 1h timeframe ≠ our use case

They tested 3h, 6h, 12h momentum hold periods on 1h bars. Our bot's
typical hold is **8-48 bars on 15m / 1h**, much shorter. The cluster
mechanism (lead-coin → follower lag) is fastest at **sub-hour** scales
(5-30 min), which their backtest can't see.

**But:** their 12h-momentum cluster-confirm did show *smaller* loss
than baseline (−0.20 % vs −0.35 %), so the signal is **directionally
weak** but not zero. Maybe 5m timeframe (which neither of us has yet)
would show different results.

### Weakness 3: cluster as *binary filter* is the harshest test

Their methodology: "cluster confirm" = require N peer pumps. This is
**hard filter**. A **soft override** (cluster activity as one of many
features to ML scorer) might survive their reject.

**Tying to our context:** our H-CV3 (cluster ML override) IS softer —
it only fires when ML *already* says no. That's a different test than
"add cluster as gate". They didn't test override-style usage.

## 4. Hypothesis-by-hypothesis mapping

Their backlog vs our hypotheses:

| Their priority | Our equivalent | Status after their evidence |
|----------------|----------------|------------------------------|
| **P1** cluster exposure cap (1 per group) | similar to our `correlation_guard` (already prod, reactive) | **adopt their pre-computed approach** — better than our on-demand Pearson |
| **P2** same-cluster replacement | rotation.py concept | strong complement to our rotation logic; consider port |
| **P3** cluster breadth as exit-tail selector | **complements our H-CP2 (PEAK RISK)** | **strong synergy** — peer-positive-rate gates would refine our PEAK RISK trail-tighten |
| **P4** cluster-wide wake-up confirmation | very close to **our H-CV1** in softer form | **adopt this softer framing** — diagnostic-only WATCH first, not auto-inject |
| **P5** group regime risk-off throttle | similar to **our H-TL2** (BTC gate) but cluster-level | consider adding peer-breadth alongside BTC slope |
| **P6** rolling cluster confidence | weekly recompute we already planned | already in scope |

### Their P3 deserves a closer look — it's our **best missing idea**

Their P3 says: for already-open winners, when exit triggers, check if
**cluster peers are still moving positive**. If yes, this might be a
sector impulse you're cutting too early.

**Concrete formula they propose:**
- Peer average return over 3h / 6h / 12h
- Peer positive rate
- Whether symbol is leader or laggard

This is **exactly** what our PEAK RISK + ZigZag-gated WEAK
(H-CP2 / H-CP3) is missing — peer context. A coin's WEAK div is much
more meaningful **alone vs whole-cluster-still-positive**. If 5 of 6
gaming peers are still rising and we're WEAK-exiting MANA, that's
likely premature.

**Action: add peer-breadth feature to our PEAK RISK score and ZigZag
gate.**

## 5. Updated decision: revise our roadmap

### Demote / kill

- **H-CV1 (cluster pump propagation)** — their 365 d evidence directly
  invalidates the entry-boost framing. Demote to:
  - **WATCH-only logging** (not auto-inject) — matches their P4
  - Wait 30 d of OUR live data before reconsidering live entry impact

### Strengthen / add

- **H-CV1' (revised, lower priority)** = log-only cluster wake-up
  events; never auto-inject. Replay-tested later if data supports it.

- **NEW: H-CP-PEER** (their P3 ported)
  - In PEAK RISK score, add peer-positive-rate component.
  - In ZigZag WEAK suppress, additionally require peer-still-positive.
  - Shadow-only first.

- **NEW: H-RIS-PEER** (their P1+P2 ported)
  - Replace reactive `correlation_guard.py` Pearson with pre-computed
    cluster map (from our cluster-recompute weekly task).
  - Cap 1 position per learned group when ≥ 3 members.
  - Same-group replacement preferred over expansion when portfolio full.

### Keep as-is

- H-CP1 (vol-scaled H5) — independent of cluster work; data ready.
- H-CP2 (PEAK RISK trail tighten) — still primary; now with peer
  breadth as added input (per H-CP-PEER).
- H-CP3 (ZigZag-gated WEAK) — keep; add peer gate.
- H-CV2 (ML blind-spot retrain) — independent of cluster work.
- H-TL2 (BTC-correlated gate+boost) — directionally consistent with
  their P5; maybe also add cluster-level direction.
- H-GT1 (sustained label as ML target) — independent; structurally
  most important.

### Defer

- H-CV3 (cluster ML override) — needs more careful framing as **soft
  override only when other guards also positive**, not blanket bypass.
- H-TL1 (5 m secondary scan) — orthogonal; still high cost.

## 6. Updated sprint 1 plan

Revised list (4 items, week 1):

| # | Hypothesis | Source | Effort | Risk |
|---|-----------|--------|--------|------|
| 1 | **H-CP1** vol-scaled H5 | ours | 0.5 d | low |
| 2 | **H-CP3 + peer breadth** ZigZag-gated WEAK with peer-still-positive | ours + their P3 | 1.5 d | low |
| 3 | **H-RIS-PEER** pre-computed cluster cap + same-group replacement | their P1+P2 | 2 d | low-med |
| 4 | **H-TL2** BTC gate+boost (with cluster-level direction) | ours + their P5 hint | 1 d | low-med |

H-CV1 in original form is **out of sprint 1**. The cluster work pivots
from "use clusters to enter MORE" to "use clusters to enter SMARTER and
hold longer".

Estimated impact updated:
- coverage: less than my original estimate (no auto-cluster-inject)
- conditional capture: **higher** than my original (peer breadth in
  exits is a strong signal)
- net NS: similar to original ~+8-15 pp by end of week 1, but **with
  much higher confidence** because hypotheses now have cross-validation
  from the gpt_crypto_bot audit.

## 7. Honest meta-note

I missed the cluster-confirm-as-entry rejection because:
- I only did 14 d in-sample, no holdout.
- I didn't even **run a backtest** — relied on intuition + cluster map
  evidence.

Going forward, **any cluster-related entry mechanism MUST be
holdout-backtested before promotion**, mirroring their methodology.

## 8. Action items

- [ ] Update `docs/reports/2026-05-07-ns-hypotheses-roadmap.md` with
      revised sprint 1 (this commit).
- [ ] Move H-CV1 to "deferred / log-only WATCH" status in any future
      doc.
- [ ] Open spec drafts for **H-CP-PEER** and **H-RIS-PEER** (ported
      from gpt_crypto_bot hypotheses, adapted to our code).
- [ ] When implementing H-CP1 (still day-1 task), include peer-breadth
      hooks for later H-CP3 work.
- [ ] Keep gpt_crypto_bot reports as primary external reference for
      anything cluster-related.

## 9. What is **NOT** done by this validation

- I cannot edit gpt_crypto_bot. This is purely about updating OUR
  roadmap based on what they've shown.
- I cannot reproduce their 365 d holdout from our cache (we have at
  most 30 d klines). To do equivalent work in our project would need
  365 d klines backfill (1.5 GB+ at 15m, doable but heavy).
- I am NOT auto-importing any of their code. Their methods are
  inspiration; the implementation in our codebase needs to honor our
  config + flag + spec workflow.
