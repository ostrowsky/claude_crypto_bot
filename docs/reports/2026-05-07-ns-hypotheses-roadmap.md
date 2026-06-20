# North-Star hypotheses roadmap

**Date:** 2026-05-07
**Author:** core
**Purpose:** consolidate hypotheses for raising the North Star
`watchlist_top_early_capture_pct` based on observed data from sessions
2026-04-28 → 2026-05-07.

## 0. Updated North Star definition

Per `CLAUDE.md` §1 (canonical):

```
watchlist_top_early_capture_pct =
  #(bot bought top-20 with capture_ratio ≥ 0.35) / #(top-20 in watchlist)
```

This is **stricter** than mean capture ratio. Threshold 0.35 means we
must catch ≥ 35 % of the day's move per trade.

**Current state on top-20 entries (14d):**

```
Mean capture_ratio:     0.12
Median capture_ratio:   0.05
Trades with cap ≥ 0.35: 7 / 47 = 15 %
```

→ Reaching even 30 % strict-capture-rate gives a doubling on this leg.

**Formal decomposition:**

```
NS  =  coverage  ×  P(cap ≥ 0.35 | entered)
```

Two independent levers; both can be moved in parallel.

## 1. Data inputs powering these hypotheses

| Source | What | Used by |
|--------|------|---------|
| TON case 2026-05-04..06 | 3 entries (#1 +25 %, #2 +0.24 % WEAK, #3 −1.72 % trail), 211 ML-zone blocks (75 %), 39 trend-chop blocks | H-CP1, H-CP3, H-CV2, H-CV3 |
| ICP case 2026-05-05 | WEAK div exit at +2.5 % continued to +6.8 % | H-CP3 |
| STRK case 2026-05-06 | silent miss — coin dropped from hot_coins, pump started in scan gap | H-CV1, H-TL1 |
| sustained-label backfill | 78.1 % pump-and-dump rate in label_top20 | H-GT1 |
| correlation clusters report | sector groups: gaming/L1/storage; QI/MEME/AUDIO have low BTC-corr (= our ML blind-spots) | H-CV1, H-CV3, H-TL2 |
| _validate_p5_ml_blindspots | 48 top-20 winner-days ML-blocked, 28.9 % extreme (proba < 0.10) | H-CV2 |
| EarlyCapture@top20 vs @sustained | 0.055 vs 0.031 (1.8× worse on sustained universe) | H-GT1 |
| EX1 baseline | median realized/potential = 0.001; worst exit class `ema20_weakness` (median −0.010) | H-CP1, H-CP3 |
| PEAK RISK shadow design | bot has no «overextension zone» concept; only SELL/HOLD binary | H-CP2 |

## 2. Hypotheses

Each hypothesis carries a short ID `H-<axis><n>`:

- `H-CV*` — Coverage (raise # entered top-20 winners)
- `H-CP*` — Conditional capture (raise P(cap ≥ 0.35 | entered))
- `H-TL*` — Time-lead (enter earlier in the move)
- `H-GT*` — Ground-truth quality (improves all levers indirectly)

---

### H-CV1 · Cluster pump propagation

**Observation:**
The correlation-clusters report identified Cluster B (gaming / metaverse /
payment): **CELR, COTI, GMT, MANA, SAND** with intra-correlation +0.59.
When SAND pumps, MANA / GMT / COTI typically follow within 15-30 min.
Bot's `pump_detector` operates **per-coin** on its own 24h-pct delta —
followers lag.

**Hypothesis:**
Triggering cluster-wide injection on a single member's pump cuts
follower scan-lag by 5-15 min and adds coverage of sector pumps that
current per-coin pump-detector misses.

**Solution:**
In `_pump_detector_loop()`, when sym X triggers pump injection:
1. Look up `cluster_id` of X from `.runtime/correlation_clusters_residual.json`.
2. Inject ALL members of that cluster into `hot_coins` (with `signal_now=True`).
3. Mark cluster cooldown (don't re-inject same cluster within
   `CLUSTER_PUMP_COOLDOWN_MIN` = 20 min) to avoid storm.

**Data ready:** ✅ Cluster map exists. Pump detector active.

**Implementation cost:** 1 day (logic + config flag + smoke test).

**Risks:**
- *False sector pumps* — one coin pumps for idiosyncratic reason, members
  don't follow → wasted scan slot.
  Mitigation: cluster_id is **residual-correlation** based (after BTC beta
  out), so spurious BTC-only co-movement is excluded.

**Validation:**
Backtest on last 30d: for each pump-detected event, count how many
cluster members ALSO pumped within 30 min. If ≥ 40 %, hypothesis valid.

**Acceptance:**
After 7d live with flag ON:
- ≥ 5 cluster-wide injections logged
- Of injected followers, ≥ 30 % became entries (not just blocked)
- NS coverage delta ≥ +3 pp vs prior 7d baseline

**Estimated impact:** **+5-10 pp coverage** on sector-pump days
(estimated ~30 % of days fit this pattern).

**Dependencies:** correlation cluster map (cron-recomputed weekly).

---

### H-CV2 · ML signal-model blind-spot retrain

**Observation:**
- 48 top-20 winner-days blocked by `ml_proba_zone` over last 30d.
- 28.9 % of all ML-zone blocks have proba < 0.10 (extreme blind-spot).
- TON: 211 of 280 (75 %) blocks were ml_proba 0.001-0.25.
- ORDIUSDT, APEUSDT, AXLUSDT: 4 separate winner-days each missed.
- Recurring blind-spot syms: TRU, BLUR, MDT, ORDI, AUDIO, QI, COMP, APE.

These syms cluster low on BTC-correlation (per cluster report:
QI corr 0.24, MEME 0.40, AUDIO 0.41) → they are **independent pumpers**
that the BTC-beta-tuned model doesn't see coming.

**Hypothesis:**
Retraining `ml_signal_model` with sample_weight = 3.0 on
`(y_top20 = 1 AND prev_proba < 0.10)` rows corrects the systematic miss
without sacrificing overall AUC.

**Solution:**
```bash
pyembed/python.exe files/ml_signal_model.py \
  --blindspot-weight 3.0 \
  --blindspot-proba-threshold 0.10 \
  --prev-model files/ml_signal_model.json \
  --model-out files/ml_signal_model_v2.json
```
Plumbing already in v2.19.0. Run shadow A/B for 7d via parallel scoring
in `_ml_general_score` (write v2 score to events but use v1 in entry
gate). After 7d compare:
- AUC on top-20 task
- block-rate on blind-spot subset
- false-positive rate (proba > 0.5 on non-winners)

**Data ready:** ⏳ pending — needs structured-block aggregation from
v2.17.0 to have ≥ N blind-spot examples flagged in critic_dataset.
ETA 2-3 days.

**Implementation cost:** 1 day code, 7d shadow watch.

**Risks:**
- *Overfit to blind-spot syms* — model loses generality.
  Mitigation: weight ≤ 5, monitor AUC overall.
- *AUC drop* — accept if blind-spot recall up ≥ 30 pp.

**Validation:**
- Spec §7 acceptance: blind-spot block rate 28.9 % → ≤ 10 %.
- AUC stays ≥ 0.97.
- False-positive rate not up > 20 % rel.

**Estimated impact:** **+10-15 pp coverage** on isolated pumps
(these are also our highest-volatility coins, so capture potential is
extra high).

**Dependencies:** P0.1 structured-block data accumulation (~48 h).

---

### H-CV3 · Cluster-aware ML proba override

**Observation:**
ML model gives proba 0.001 on TON when TON's cluster lead is actively
pumping. The model never saw the cluster context. Cluster activity is
**external validation** of pump probability — independent of the
in-model features.

**Hypothesis:**
When `ml_proba` would otherwise block (< zone min), but the coin's
cluster lead-coin is in active pump state, override and let downstream
gates decide. This restores coverage on isolated pumps without
retraining the model.

**Solution:**
In `_ml_general_score` block-path (monitor.py ~ L3735):
```python
if ml_proba < _min:
    cluster_id = get_cluster_id(sym)  # from .runtime cache
    if cluster_id and is_cluster_pumping(cluster_id, lookback_min=15):
        log.info("ML_ZONE OVERRIDE %s: ml_proba=%.3f but cluster=%d pumping",
                 sym, ml_proba, cluster_id)
        # Don't block; downstream guards still apply
    else:
        reason = "ML proba ..."
        log_blocked(reason_code="ml_zone", ...)
        return
```

`is_cluster_pumping()` reads `_pump_detector_loop` state OR direct
ticker/24hr poll on lead-coin.

**Data ready:** ✅ Cluster map + pump-detector state.

**Implementation cost:** 1 day.

**Risks:**
- *Bypass legitimate ML block* — sometimes model is right.
  Mitigation: override condition is **strong** (active cluster pump =
  rare event); also we don't auto-enter, just unblock for downstream.
- *Pump detector latency* — may not flag pump for first 5 min.
  Mitigation: use `priceChangePercent` snapshot, not ring-buffer delta.

**Validation:**
Counterfactual on 30d: for each `ml_zone` block where coin became top-20,
check if cluster lead pumped in the prior 15 min. If ≥ 40 % of cases,
hypothesis valid.

**Acceptance:**
- Override fires ≤ 5/day (precision protection).
- Of overrides, ≥ 30 % become entries.
- NS coverage delta ≥ +2 pp.

**Estimated impact:** **+5-8 pp coverage** at low precision-risk.

**Dependencies:** cluster map, pump-detector state introspection.

---

### H-CP1 · Volatility-scaled H5 break-even threshold

**Observation:**
TON Trade #2 specifically:
- mode=impulse_speed/15m, pnl=+0.24 % at WEAK RSI div exit
- H5 per-mode threshold (v2.20.0) for impulse_speed = 0.3 % → did NOT fire
- daily-range that day was ~15 % (extreme volatility)
- Coin continued to +16 % (gave up 14 pp of capture)

The 0.3 % threshold is calibrated for **normal** volatility. On a
15 %-daily-range day, +0.24 % pnl ≈ noise of one bar's wick. Suppress
threshold should scale **inversely** with day volatility.

**Hypothesis:**
Effective H5 threshold = `base_H5 × clip(5 / max(DR_pct, 1), 0.5, 1.5)`
catches the missed TON-class without false-positive H5 suppress on
normal days.

**Solution:**
In `_h5_break_even_pct(mode, tf)` add daily_range scaling:
```python
def _h5_break_even_pct(mode, tf, daily_range=None):
    base = ...  # existing per-mode lookup
    if daily_range is None or not config.H5_VOL_SCALING_ENABLED:
        return base
    ratio = clip(5.0 / max(daily_range, 1.0), 0.5, 1.5)
    return base * ratio
```

Pass `daily_range` from the exit-check call site.

**Data ready:** ✅ EX1 baseline + per-mode H5 history. Daily-range is
in every event.

**Implementation cost:** 0.5 day (code + backtest).

**Risks:**
- *Premature H5 suppress on noise* — at min cap 0.5, threshold becomes
  0.15 % for impulse_speed. Mitigation: hard floor at 0.1 % via clip.
- *Misses on low-vol days* — on DR=3 %, ratio=1.5 → H5 = 0.45 % for
  impulse_speed. Trade-off acceptable: slow days have less giveup.

**Validation:**
Backtest 30d on H5-eligible exits:
1. Re-classify with vol-scaled threshold.
2. For each NEW suppress, simulate "would have continued" via
   `realized_pnl_at_eod_high / entry`.
3. Aggregate delta capture vs current per-mode H5.

**Acceptance criteria:**
- Median EX1 on H5-modes ↑ ≥ +0.05.
- Whipsaw rate Q2 not up > 2 pp.
- 5 historical TON#2-class cases all now correctly suppress.

**Estimated impact:** **+15-25 pp on conditional capture in volatile days**
(estimated ~25 % of days fit).

**Dependencies:** none.

---

### H-CP2 · PEAK RISK score → trail tighten (not exit)

**Observation:**
TON Trade #1: exited at RSI 89 (single-feature exit) at +25 % pnl. Coin
continued to +45 %. We left 20 pp on the table. RSI alone is too crude
— it fires AT the top AND in continuation moves.

PEAK RISK shadow (P1.2 v2.20.0) emits a composite 0-100 score combining
RSI + price_edge_vs_EMA20 + MACD_decel + profit_level. Higher score =
more probable peak zone.

**Hypothesis:**
When PEAK RISK score ≥ 70: tighten trail by `trail_k *= 0.6` and SUPPRESS
soft RSI-alone exits. Lets the position ride until ATR-trail catches a
real reversal, instead of pattern-based pre-emptive exit.

**Solution:**
Phase 2 of `peak-risk-shadow-spec.md`:
```python
# In exit-decision path, after _peak_risk_score()
if pr_score >= 70:
    # Tighten trail in-place
    pos.trail_k = pos.trail_k * config.PEAK_RISK_TRAIL_TIGHTEN_FACTOR
    log.info("PEAK_RISK_TIGHTEN %s score=%.0f trail_k=%.2f→%.2f",
             sym, pr_score, pos.trail_k / factor, pos.trail_k)
    # Suppress RSI-alone exits
    if reason and "rsi" in reason.lower() and "weak" not in reason.lower():
        log.info("PEAK_RISK_SUPPRESS_RSI %s pnl=%.2f score=%.0f", sym, pnl, pr_score)
        reason = None
```

Trail-tighten is **monotonic** (doesn't loosen on score drop). One-shot
per position.

**Data ready:** ⏳ 7d of PEAK RISK shadow events from v2.20.0.

**Implementation cost:** 1 day code, 7d watch.

**Risks:**
- *Wider drawdown* if ATR is large at peak.
  Mitigation: combo with min_buffer floor (already in prod).
- *Trail too tight* — gets stopped on first dip.
  Mitigation: factor 0.6 (not 0.5); also re-test on backtest.

**Validation:**
Counterfactual on existing exits with simulated PEAK RISK score:
- For each WEAK / RSI exit where score would have been ≥ 70:
  simulate trail-only continuation.
- Aggregate capture delta.

**Acceptance criteria:**
- ≥ 20 PEAK RISK score-70+ events in 7d shadow.
- Counterfactual median EX1 on those events ↑ ≥ +0.10.
- Q2 whipsaw not up > 3 pp.

**Estimated impact:** **+20-30 pp on conditional capture for winners**
(this is the H4-direct fix).

**Dependencies:** P1.2 PEAK RISK shadow data accumulation (7d).

---

### H-CP3 · ZigZag-gated WEAK exit suppression

**Observation:**
ICP 2026-05-05: WEAK RSI-div exit at +2.5 % pnl. ZigZag (had we computed
it real-time) would have shown an active uptrend still going — no
counter-move > 1.5 % from last swing high. Pattern-based WEAK was wrong;
structural ZigZag was right.

**Hypothesis:**
Before WEAK suppression, check if real-time ZigZag shows the position
still inside a fresh uptrend. If yes, suppress the WEAK (even with H5
inactive). This is **complementary** to H-CP2 (PEAK RISK is about TOP;
ZigZag is about CONTINUATION).

**Solution:**
```python
if reason and reason.startswith("⚠️ WEAK"):
    if pos.signal_mode in ("alignment", "trend", "strong_trend"):
        bars = build_realtime_bars(sym, lookback=20)  # last 20 closed 15m bars
        trends = zigzag_labeler.detect_uptrends(bars, swing_pct=2.0,
                                                 max_drawdown_pct=1.5,
                                                 min_duration_bars=3)
        if trends and trends[-1].end_idx >= len(bars) - 2:
            # Last trend is current → still rising
            log.info("ZIGZAG_SUPPRESS_WEAK %s last_trend gain=%.1f%%",
                     sym, trends[-1].gain_pct)
            reason = None
```

**Data ready:** ✅ `zigzag_labeler.py` exists (P1.1 work).

**Implementation cost:** 1 day (real-time bar builder + smoke test).

**Risks:**
- *Bar-realtime computation cost* — at scale (40 active positions),
  may add 100-200 ms per poll. Mitigation: cache last result per
  position per 5 min.
- *ZigZag thresholds wrong* — swing 2 % may miss small reversals.
  Mitigation: thresholds calibrated to mode (trend = tighter).

**Validation:**
Backtest 30d on WEAK exits in trend modes:
- For each, simulate "would have continued" via subsequent 30 bars.
- If ≥ 50 % continued past +5 %, suppression is value.

**Acceptance criteria:**
- WEAK-exit rate ↓ ≥ 30 % on trend/alignment/strong_trend.
- Median pnl on those held positions ↑ ≥ +2 pp.
- Q2 whipsaw not up > 3 pp.

**Estimated impact:** **+10-15 pp on conditional capture** for slow-trend
modes (those most affected by premature WEAK exits).

**Dependencies:** none structural; zigzag_labeler ready.

---

### H-TL1 · 5-minute secondary scan for hot_coins

**Observation:**
TON pump start was ~07:00 UTC. Bot's 15m bar closed at 07:14 — by then
price was already +3 %. First entry possible only at 07:14. On 5m bars
the signature appeared at 07:05 → 9 min earlier.

**Hypothesis:**
For coins in `hot_coins` subset (~30-40 active), running a parallel 5m
scan and triggering entry only when **both** 5m signal AND 1h trend
agree, cuts time-to-signal by 5-15 min on fast pumps without raising
false-positive rate.

**Solution:**
1. Backfill 5m klines for watchlist (separate script).
2. In `monitoring_loop`, for hot_coins, additional async task: fetch
   last 5m klines, compute indicators, run a stripped-down entry check.
3. If 5m says BUY AND existing 1h logic also says BUY → entry.
   Otherwise skip.

**Data ready:** ❌ Needs 5m klines backfill (~30d × 105 coins × 8640
bars/30d = ~3 M kline records, several MB cache).

**Implementation cost:** 5 days (klines fetch script, scan logic
duplication, careful concurrency).

**Risks:**
- *2× noise* — 5m has more whipsaws.
  Mitigation: 1h-confirmation gate.
- *API rate-limit* — extra REST calls.
  Mitigation: 5m only for hot_coins (subset), not watchlist.

**Validation:**
Replay 30d of pumps with hypothetical 5m scan. Measure TTS delta.

**Acceptance:**
- TTS median ↓ ≥ 5 min on fast pumps.
- False-positive rate not up > 15 % rel.

**Estimated impact:** **+10-15 pp time-lead score** on fast pumps
(~20 % of days).

**Dependencies:** 5m klines backfill (~1 day to bootstrap), continuous
maintenance via daily task.

---

### H-TL2 · BTC-correlated gating + boost

**Observation:**
90 % alts have corr ≥ 0.5 with BTC. Core L1 (ETH/SOL/LINK/AVAX) at
+0.90. Bot ignores BTC direction. On strong BTC pumps, all core L1s
move synchronously — we should priority-scan them. On BTC dumps, alts
typically follow — we should defensively gate.

**Hypothesis:**
- **Gate:** For coins with BTC-corr ≥ 0.7, do NOT enter if
  `btc_slope_1h < -0.5 %` (avoid swimming against the tide).
- **Boost:** When `btc_slope_1h > +0.7 %`, inject all high-corr coins
  into hot_coins as "BTC propagation" (analogous to H-CV1 but BTC-driven).

**Solution:**
Two patches:
1. In entry pipeline (after preview_mode set), check BTC slope from
   monitor's existing `_btc_vs_ema50` or fetch fresh; if condition, block.
2. In `_pump_detector_loop`, when BTC pumps detected, inject high-corr
   subset.

**Data ready:** ✅ Correlation matrix from H-CV1 cache.

**Implementation cost:** 1 day.

**Risks:**
- *Missed decoupled pumps* — QI/MEME (low BTC-corr) bypass this — that
  is correct, they shouldn't be gated/boosted by BTC.
- *Over-gating on dip days* — many coins blocked.
  Mitigation: gate only applies to slope < −0.5 % (strong negative).

**Validation:**
Backtest 30d:
- For each BTC slope > +0.7 % window, count subsequent 1h alt entries
  and their cap ≥ 0.35 rate.
- For each slope < −0.5 % window, count blocked alts that would have
  lost vs won — confirms gate is protective.

**Acceptance:**
- On BTC-up days: high-corr coin coverage ↑ ≥ +5 pp.
- On BTC-down days: high-corr loser rate ↓ ≥ +3 pp.

**Estimated impact:** **+3-5 pp time-lead + +2-3 pp coverage** on
BTC-driven days (~40 % of days).

**Dependencies:** correlation cluster cache.

---

### H-GT1 · Promote label_sustained_uptrend as primary ML target

**Observation:**
The single most consequential finding from the entire session:
**78.1 % of label_top20=1 pairs are pump-and-dumps** (per backfill
script crosstab on 30d).

The ML model is trained on label_top20. It's learning to predict pumps
+ dumps. That's why it does WORSE on sustained universe
(EarlyCapture@top20 0.055 vs @sustained 0.031).

**Hypothesis:**
Retraining `ml_signal_model` on `label_sustained_uptrend` (target =
clean ZigZag-validated trend) yields a model that **actually**
predicts what we want: trades with realized capture ≥ 0.35.

**Solution:**
1. Update `ml_signal_model.train_and_evaluate` to optionally use
   `label_sustained_uptrend` from `top_gainer_dataset_v2.jsonl`
   instead of label_top20 as positive class.
2. Train shadow model `ml_signal_model_sustained.json`.
3. 30 d A/B in production: write both probas to events, use legacy
   in entry gate, compare.
4. If sustained-model has higher P(cap ≥ 0.35 | entered) AND comparable
   coverage → swap.

**Data ready:** ✅ `top_gainer_dataset_v2.jsonl` exists.

**Implementation cost:** 3 days code + 30 d A/B.

**Risks:**
- *Lower volume of training positives* — n_sustained=237 vs n_top20=128
  (more), but only 28 in both → tighter feature signal. Could lose
  some discrimination on noisy cases.
  Mitigation: also include n_top20 for blind-spot weighting (combo
  with H-CV2).
- *Model less aggressive overall* — fewer ENTERs.
  Mitigation: this might actually be correct — those skipped were the
  pump-and-dumps we don't want to ride.

**Validation:**
Two parallel models for 30 days. Metrics tracked per day:
- AUC on both labels
- Production P(cap ≥ 0.35 | entered)
- TG msg_rate (precision proxy)

**Acceptance criteria:**
After 30 d:
- P(cap ≥ 0.35 | entered) on sustained-model ↑ ≥ +5 pp vs legacy.
- AUC on sustained label ≥ 0.85.
- recall@sustained ≥ 90 % overall.

**Estimated impact:** **+10-20 pp on NS through multiple channels**
(model now picks RIGHT setups → better entries → better captures →
better lead times).

**Dependencies:** sustained label backfill complete (✅ already done).

---

## 3. Prioritized rollout

| Tier | Sprint | Hypothesis | Net est | Effort | Risk | Data ready? |
|------|--------|-----------|--------:|--------|------|:-----------:|
| **P1** | week 1 | **H-CP1** vol-scaled H5 | +15-25 pp cap | 0.5d | low | ✅ |
| **P1** | week 1 | **H-CV1** cluster pump propagation | +5-10 pp cov | 1d | low | ✅ |
| **P1** | week 1 | **H-CP3** ZigZag-gated WEAK | +10-15 pp cap | 1d | low | ✅ |
| **P1** | week 1 | **H-TL2** BTC-correlated gate+boost | +5-8 pp | 1d | low-med | ✅ |
| **P2** | week 2 | **H-CV3** cluster ML override | +5-8 pp cov | 1d | medium | ✅ |
| **P2** | week 2 | **H-CV2** ML blind-spot retrain | +10-15 pp cov | 1d + 7d shadow | medium | ⏳ 48 h |
| **P2** | week 2 | **H-CP2** PEAK RISK trail tighten | +20-30 pp cap | 1d + 7d shadow | medium | ⏳ 7 d |
| **P3** | month 1 | **H-GT1** sustained-label ML target | +10-20 pp | 3d + 30d A/B | high | ✅ |
| **P3** | month 1+ | **H-TL1** 5m secondary scan | +10-15 pp tl | 5d | high | ❌ klines backfill |

### Cumulative net estimate

Conservative (compound with 70 % efficiency multiplier per hypothesis):

```
Coverage:   1.00 × 1.08 × 1.06 × 1.07 × 1.05 × 1.12 × 1.10  ≈  1.62  (×1.6 on cov)
Capture:    1.00 × 1.20 × 1.12 × 1.25 × 1.15                  ≈  1.93  (×1.9 on cap)
Lead time:  1.00 × 1.05 × 1.10                                ≈  1.16  (×1.16 on tl)
```

Combined NS estimate: `1.6 × 1.9 × 1.16 ≈ ×3.5` on current 0.055 →
`~0.19` ceiling. That's still below the canonical strict-capture-0.35
floor of perfect bot. Realistic target: **NS = 0.12-0.15 in 6-8 weeks**,
which represents 2-3× improvement.

## 4. Cross-cutting risks

- **All hypotheses except H-GT1 build on existing labels and models.**
  H-GT1 is the structural ground-truth fix; without it, all per-trade
  improvements have a ceiling determined by `label_top20`'s 78 %
  pump-and-dump rate. H-GT1 raises that ceiling.

- **Cluster recompute frequency.** Clusters drift weekly. Need a
  scheduled task to refresh the cluster map (Sunday 04:00 local
  alongside skill weekly).

- **Live regression risk** — every threshold change has been
  back-tested but the TON regression on `trend_chop_filter` (v2.6.0 →
  rollback in v2.16.0) shows backtest != live. Mandate:
  - All P1 changes go behind config flag, default OFF
  - 7 d shadow on flag=False (logging only)
  - Then 7 d live with daily aggregated check

## 5. What we are NOT doing (and why)

- **H-CV4 lower entry_score floor on bull-day** — marginal impact;
  P0.1 structured-block data should validate first.
- **H-CP4 generic adaptive trail** — overlaps with H-CP1, H-CP2.
- **H-GT2 multi-target ML head** — too large refactor; H-GT1 covers it.
- **Detector-first big-bang refactor** — risky; case for incremental
  patches still strong.

## 6. First action

**Start with H-CP1 (vol-scaled H5).** Cheapest, fastest, directly
addresses the single most observable failure (TON Trade #2-class). Data
ready. Risk near zero (flag-gated).

Sprint 1 plan day-by-day:
- Day 1: H-CP1 spec → code → backtest → ship behind flag.
- Day 2: H-CV1 spec → code → smoke test → ship behind flag.
- Day 3: H-CP3 spec → code → backtest → ship behind flag.
- Day 4: H-TL2 spec → code → backtest → ship behind flag.
- Day 5: All P1 flags flipped ON.
- Day 6-7: 48 h watch + daily NS report.

End of week 1 expected NS lift: **+8 to +15 pp** (coverage and capture
combined).
