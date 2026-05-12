# Auto-Improvement Loop — full spec, living document

- **Slug:** `auto-improvement-loop`
- **Status:** in-progress — see component matrix below
- **Created:** 2026-05-12
- **Last updated:** 2026-05-12 (after Sprint 1 partial: RM-14 + RM-15 landed)
- **Owner:** Vasiliy Ostrovsky + Claude
- **North Star:** `watchlist_top_early_capture_pct` (see [PROJECT_CONTEXT.md](../../../PROJECT_CONTEXT.md))
- **Project priority:** P0 (founding principle, see [CLAUDE.md §0](../../../CLAUDE.md))

> **This document is the single source of truth for what's wired up and what
> isn't.** Every PR that touches a loop component must update:
>
> 1. The component's status row in §3.
> 2. The North Star progress table in §4 (one new row per measurable change).
> 3. The `Last updated` header above.
> 4. The roadmap in §5 (mark items done, add follow-ups discovered while shipping).

---

## 1. Mission

A closed loop that, without daily human babysitting:

1. **Measures** the bot's behaviour against the North Star every day.
2. **Detects** systematic problems (red flags + per-mode incidents).
3. **Generates** candidate config changes (rules + Claude).
4. **Validates** them on historical data before exposing to the operator.
5. **Surfaces** validated proposals in one unified Telegram message.
6. **Applies** approved changes (today: operator presses a button; future: runtime override).
7. **Measures** the actual effect with bootstrap CI and market-drift normalisation.
8. **Rolls back** when measured effect contradicts expectation.
9. **Remembers** outcomes so the next iteration neither repeats failed work nor loses successful patterns.

The loop's success is judged exclusively on its effect on the North Star —
not on how many hypotheses it generates, not on how clever any single
prediction is.

---

## 2. Architecture — 7 layers + memory + delivery

```
                       ┌─────────────────────────┐
                       │  bot_events.jsonl        │  ← raw signal of truth
                       │  critic_dataset.jsonl    │
                       │  top_gainer_critic_*.json│
                       │  evaluation_output/      │
                       └────────┬─────────────────┘
                                │
   ┌────────────────────────────▼─────────────────────────────────┐
   │  L1 — daily metrics + red flags                              │
   │  bot_health_report.py                                        │
   └────────────────────────────┬─────────────────────────────────┘
                                │
   ┌────────────────────────────▼─────────────────────────────────┐
   │  L2 — hypothesis generator (rules + Claude + incidents boost)│
   │  pipeline_hypothesis.py · pipeline_claude_client.py          │
   └────────────────────────────┬─────────────────────────────────┘
                                │
   ┌────────────────────────────▼─────────────────────────────────┐
   │  L3 — backtest validator (Pareto sweep, per-rule)            │
   │  pipeline_validator.py                                       │
   └────────────────────────────┬─────────────────────────────────┘
                                │
   ┌────────────────────────────▼─────────────────────────────────┐
   │  L4 — counterfactual / shadow                                │
   │  pipeline_shadow.py  (sim_handlers + bot-side shadow events) │
   └────────────────────────────┬─────────────────────────────────┘
                                │
   ┌────────────────────────────▼─────────────────────────────────┐
   │  L5 — blind regression critic (rule + Claude in parallel)    │
   │  pipeline_blind_critic.py                                    │
   └────────────────────────────┬─────────────────────────────────┘
                                │
   ┌────────────────────────────▼─────────────────────────────────┐
   │  L6 — operator approval (safety_checks + Claude advisor)     │
   │  pipeline_approve.py · pipeline_baseline.py                  │
   └────────────────────────────┬─────────────────────────────────┘
                                │  ⤴ baseline pinned at apply
                                │
   ┌────────────────────────────▼─────────────────────────────────┐
   │  L7 — outcome attribution + rollback recommendations         │
   │  pipeline_monitor.py · pipeline_attribution.py               │
   └────────────────────────────┬─────────────────────────────────┘
                                │
   ┌────────────────────────────▼─────────────────────────────────┐
   │  Memory   decisions.jsonl · already_tried.jsonl ·            │
   │           do_not_touch.json · baselines/  · attribution.jsonl│
   └──────────────────────────────────────────────────────────────┘
                                │
   ┌────────────────────────────▼─────────────────────────────────┐
   │  Delivery — one unified Telegram message per day             │
   │  pipeline_notify.py                                          │
   └──────────────────────────────────────────────────────────────┘
```

---

## 3. Component status matrix

Legend: ✅ done · 🟡 partial · ❌ not implemented · ⏸ deferred

### L1 — Metrics collection

| ID | Component | Status | Tests | Notes |
|----|-----------|--------|-------|-------|
| L1-a | `bot_health_report.py` daily snapshot | ✅ | `test_bot_health_critic_phase.py` (6) | reads `top_gainer_critic_*_final` preferred over `_midday` |
| L1-b | Training health (recall@20, UCB sep, AUC) | ✅ | indirect | from `learning_progress.jsonl` |
| L1-c | Deployment health (early_capture, FPR, …) | ✅ | indirect | from critic final |
| L1-d | Per-mode signal evaluator | ✅ | — | `_weekly_signal_eval_with_tg.py` writes `evaluation_output/per_mode/<mode>/report.json` |
| L1-e | Red flag detection (persistent ≥4d) | ✅ | — | `bot_health_report.detect_red_flags()` |
| L1-f | Training-to-live gap explicit metric | ✅ | — | most important diagnostic per §0 |

### L2 — Hypothesis generator

| ID | Component | Status | Tests | Notes |
|----|-----------|--------|-------|-------|
| L2-a | Rule-based: `entry_score_floor_relax` | ✅ | — | `rule_entry_score_floor` |
| L2-b | Rule-based: `tighten_proba_<mode>` | ✅ | — | `rule_losing_mode_disable` (misnamed; produces tighten) |
| L2-c | Rule-based: `relax_gate_<gate>` | ✅ | — | `rule_overblock_gate` |
| L2-d | Claude augmentation (≤3 extra hypotheses) | ✅ | — | gated by `pipeline_claude_client.is_enabled()` |
| L2-e | already_tried 30d cooldown | ✅ | — | `filter_already_tried()` |
| L2-f | do_not_touch enforcement | ✅ | — | `filter_locked_keys()` |
| L2-g | Incident → severity bump (premature/losers/missed) | ✅ | indirect | `_attach_incident_evidence()` |
| L2-h | New rule types from incidents (`loosen_trail_k_*`, `lower_entry_floor_*`) | ❌ | — | blocked by L3/L4 validator coverage |
| L2-i | Outcome-aware hypothesis priority (learn from hits/misses) | ❌ | — | future |

### L3 — Backtest validator

| ID | Component | Status | Tests | Notes |
|----|-----------|--------|-------|-------|
| L3-a | `validate_entry_score_floor` (per-coin replay) | ✅ | — | the only fully wired validator |
| L3-b | `validate_gate_threshold` (generic) | 🟡 | — | emits diagnostic only, no verdict |
| L3-c | Universal Pareto sweep over any gate | ❌ | — | **blocked on L1-x: blocked-event state logging** |
| L3-d | `disable_mode_*` validator | ⏸ | — | covered by L4 sim instead |

### L4 — Counterfactual / shadow

| ID | Component | Status | Tests | Notes |
|----|-----------|--------|-------|-------|
| L4-a | `sim_disable_mode` (critic_dataset counterfactual) | ✅ | `test_pipeline_shadow.py` (7) | uses `labels.ret_5` |
| L4-b | `sim_widen_watchlist_match_tolerance` (upper-bound) | ✅ | `test_pipeline_shadow.py` (6) | marked with `_caveat` |
| L4-c | `sim_loosen_trail_k_*` | ❌ | — | needs bar-level price data per trade |
| L4-d | Auto-run sim after L3 for pending hypotheses | ✅ | — | wired in `pipeline_run.py --daily` |
| L4-e | Recency split (full + last-half-window) | ✅ | — | helps detect "fixes already worked" |
| L4-f | Volume context (% of all signals) | ✅ | `test_pipeline_notify.py` | `_count_total_takes_in_window` |
| L4-g | Bot-side real shadow events (`event: shadow`) | 🟡 | — | exists for `H5_TRAILING_ONLY`, `TREND_SURGE_PRECEDENCE`, `PEAK_RISK_EXIT`; no universal per-config-key shadow |

### L5 — Blind regression critic

| ID | Component | Status | Tests | Notes |
|----|-----------|--------|-------|-------|
| L5-a | Rule-based blind verdict | ✅ | — | `critique()` in `pipeline_blind_critic.py` |
| L5-b | Sensitive-metric regression check | ✅ | — | 2pp drop on early_capture / bought / recall@20 |
| L5-c | Claude parallel blind verdict | ✅ | — | `claude_critique()` — both see only numbers |
| L5-d | Agreement check → downgrade to `needs_review` | ✅ | — | the highest-value feature in L5 |

### L6 — Approval

| ID | Component | Status | Tests | Notes |
|----|-----------|--------|-------|-------|
| L6-a | Interactive y/N prompt | ✅ | — | |
| L6-b | `safety_checks()` (locked_keys / no_validation / L3=reject / terminal) | ✅ | — | |
| L6-c | Claude advisor (full context, never gating) | ✅ | — | logged to `approve_advisory.jsonl` |
| L6-d | Baseline pin (14d pre + 30d ref) | ✅ | — | `pipeline_baseline.py` |
| L6-e | Apply instructions printed for operator | ✅ | — | manual edit + restart |
| L6-f | Auto-apply approved changes | ❌ | — | deferred; needs runtime override system to avoid daemon writing to `config.py` |
| L6-g | `--rollback` recording | ✅ | — | writes decision + already_tried |

### L7 — Attribution + rollback

| ID | Component | Status | Tests | Notes |
|----|-----------|--------|-------|-------|
| L7-a | Naive before/after (legacy) | ✅ | — | `pipeline_monitor.py` |
| L7-b | Normalised delta (raw − market_drift) | ✅ | — | `pipeline_attribution.py` |
| L7-c | Bootstrap CI95 over difference of medians | ✅ | — | 1000 resamples, no scipy dependency |
| L7-d | Sensitive-metric regression check | ✅ | — | orthogonal to expected-hit |
| L7-e | Pipeline meta-metric `hit_rate` | ✅ | — | weighted hits / (hits + misses + regressions) |
| L7-f | Rollback recommendations log | ✅ | — | `monitor/rollback_recommendations.jsonl` |
| L7-g | Rollback rec rendered in Telegram | ✅ | `test_pipeline_notify.py` | de-duped against already-rolled-back |
| L7-h | Auto-execute rollback | ❌ | — | linked to L6-f |
| L7-i | **Drift detection on ranker_proba (RM-14)** | ✅ | `test_pipeline_drift.py` (16) | KS two-sample, stdlib-only, wired into `pipeline_run.py --daily` |
| L7-j | **Multi-objective constraints (RM-15)** | ✅ | `test_pipeline_attribution_multiobj.py` (14) | Sharpe/maxdd checks in attribution; violations promote verdict to `regression` |

### Memory

| ID | Component | Status | Tests | Notes |
|----|-----------|--------|-------|-------|
| M-a | `decisions.jsonl` (append-only) | ✅ | — | source of truth |
| M-b | `already_tried.jsonl` (30d cooldown) | ✅ | — | feeds L2 blocklist |
| M-c | `do_not_touch.json` (locked keys + gates) | ✅ | — | feeds L2 and L6 |
| M-d | Baselines/`baseline_pre_<id>.json` | ✅ | — | one per approved decision |
| M-e | Attribution log + ROI dashboard | 🟡 | — | jsonl exists; aggregated dashboard not built |

### Delivery

| ID | Component | Status | Tests | Notes |
|----|-----------|--------|-------|-------|
| D-a | Unified daily Telegram | ✅ | `test_pipeline_notify.py` (102) | one message replaces 4 legacy streams |
| D-b | Incidents block (premature/missed/losing) | ✅ | — | folds in signal-evaluator data |
| D-c | Rollback recommendations block | ✅ | — | |
| D-d | Hypothesis review block (with `incident_evidence`) | ✅ | — | |
| D-e | Bottom-line ✅/⚠️/❌ verdict per hypothesis | ✅ | — | deterministic, not LLM |
| D-f | Recency drift label | ✅ | — | "эффект ослаб" / "усиливается" |
| D-g | Dedup per day (`tg_send_dedup.json`) | ✅ | — | reuses existing infra |
| D-h | Legacy Telegram streams disabled by default | ✅ | `test_pipeline_notify.py` | 4 config flags, all default False |

### Bot infrastructure

| ID | Component | Status | Tests | Notes |
|----|-----------|--------|-------|-------|
| B-a | UI snapshot cache (frozen dataclass + lock) | ✅ | `test_bot_ui_cache.py` (12) | <1ms hot reads |
| B-b | Background snapshot keeper | ✅ | — | refreshes every 5s |
| B-c | Event-loop lag detector | ✅ | — | warns at >500ms over sleep budget |
| B-d | Per-step menu timings logged | ✅ | — | WARNING if total > 1s |
| B-e | UI watchdog (existing) | ✅ | — | force-exit on stuck polling |
| B-f | **Blocked-event full state logging** | ❌ | — | **P0 — see §5 roadmap** |
| B-g | **Structured `reason_code` enum** | ❌ | — | **P0 — see §5 roadmap** |

### ML / Learning

| ID | Component | Status | Tests | Notes |
|----|-----------|--------|-------|-------|
| ML-a | CatBoost top_gainer_model retrain (nightly) | ✅ | — | `daily_learning.py` |
| ML-b | Contextual bandit (entry: 2 arms) | ✅ | — | existing |
| ML-c | Contextual bandit (trail: 5 arms) | ✅ | — | existing |
| ML-d | Bandit context includes `mode, tf, is_bull_day, …` | ✅ | — | |
| ML-e | **Bandit context includes active config-state hash** | ❌ | — | learning ignores recent config changes |
| ML-f | **`label_fast_reversal` in dataset** | ❌ | — | **P0 — see [anti-fast-reversal-spec.md](anti-fast-reversal-spec.md)** |
| ML-g | **`proba_fast_reversal` model output** | ❌ | — | depends on ML-f |
| ML-h | **`_fast_reversal_guard_reason()` in monitor.py** | ❌ | — | depends on ML-g |
| ML-i | **Anti-fast-reversal bandit reward shaping** | ❌ | — | depends on ML-f |

### External infrastructure (Claude API, scheduler, etc.)

| ID | Component | Status | Tests | Notes |
|----|-----------|--------|-------|-------|
| X-a | Claude API client (stdlib HTTP, audit log) | ✅ | — | `pipeline_claude_client.py`, BOM-safe key reader |
| X-b | Stress test (3 synthetic regressions) | ✅ | — | namespace-isolated |
| X-c | Stress test catalog expansion | ❌ | — | only 3 cases; need ~8 |
| X-d | Task Scheduler daily wiring | ✅ | — | `pipeline_run_daily.bat` |
| X-e | Time-zone unit tests | ❌ | — | mixed UTC/Belgrade in critic / events / attribution |

---

## 4. North Star progress

The single most important table in this document. **Every approved decision
must add at least one row** with the post-window measurement. Stale or
missing rows = the loop isn't really closing.

| Date | early_capture_pct | gap (train→live) | hit_rate (cumulative) | Change applied | Source decision_id |
|------|-------------------|------------------|------------------------|----------------|---------------------|
| 2026-05-11 | 6.7% (midday) → 40.0% (final) | +66.7% (midday view) | n/a | — | — |
| 2026-05-12 | tbd (early-day, no critic yet) | tbd | n/a | unified Telegram + UI cache + critic phase fix | — |
| 2026-05-12 | n/a (baseline operational) | n/a | n/a | RM-14 drift detection wired (catches ranker_proba distribution shift early) | — |
| 2026-05-12 | n/a (baseline operational) | n/a | n/a | RM-15 Sharpe + maxdd constraints in attribution (a hypothesis that improves NS but degrades risk profile now correctly classified `regression`) | — |

Conventions for filling the table:
- One row per measurable change. Re-measurements every 7/14/30 days get
  their own rows.
- `early_capture_pct` from `final` phase only (per CLAUDE.md §0 honest-measurement principle).
- `hit_rate` is cumulative pipeline meta-metric since project start. Target: ≥ 0.60.
- `Change applied` is a one-line human description.
- `Source decision_id` points to `decisions.jsonl` for traceability.

---

## 5. Roadmap — prioritised

Each item directly contributes to North Star (`early_capture_pct`) or unblocks
something that does. The "Why for North Star" column makes the chain explicit.

### Sprint 0 — P0 unblock (loop cannot close without these)

| ID | Item | Effort | Why for North Star |
|----|------|--------|---------------------|
| RM-1 | Extend `bot_events.jsonl` blocked events with full decision state (`ranker_final_score`, `candidate_score`, `score_floor`, `ml_proba`, `gate_threshold`) | 4-6h | Unblocks universal Pareto-sweep validators (L3-c) so future `relax_gate_*` hypotheses can ACTUALLY measure effect — without this, ~40% of L2 proposals end at `pending_manual_validation` |
| RM-2 | Structured `reason_code` enum in blocked events | 2h | Lets L4 sim filter blocked events generically by gate name — same unblocking as RM-1 |
| RM-3 | Anti-fast-reversal full chain (label → proba → guard → bandit reward) — see [`anti-fast-reversal-spec.md`](anti-fast-reversal-spec.md) | 2-3 days | 53.7% of `alignment` entries reverse within 3 bars. These ARE the false-positive buys that drag North Star down. Direct attack on the biggest signal-quality leak |

### Sprint 1 — Principled statistics + risk discipline (≈ 1 week)

These add **causal rigour and risk guardrails** the current loop lacks. Most
production trading systems have these; we don't yet.

| ID | Item | Effort | Why for North Star |
|----|------|--------|---------------------|
| ~~RM-14~~ | ~~**Drift detection** — KS-test / ADWIN on `ranker_proba` distribution daily; alert + optional force-retrain when drift > threshold~~ ✅ **DONE 2026-05-12** — `pipeline_drift.py` + 16 tests; wired into orchestrator | 1 day | Currently we retrain blindly every night. With drift detection we catch market-regime shift (bull → chop) BEFORE recall@20 collapses. Prevents NS degradation, not improvement, but the baseline matters |
| ~~RM-15~~ | ~~**Multi-objective approve constraints** — hypothesis must not regress Sharpe or drawdown beyond thresholds, in addition to expected delta~~ ✅ **DONE 2026-05-12** — Sharpe (rel drop > 10%) + max-drawdown (abs growth > 10pp) checks added to `pipeline_attribution._portfolio_objectives`; violations promote verdict to `regression`. 14 tests. | 1 day | Today we optimise `early_capture` in isolation. A hypothesis that improves NS by 5pp while doubling drawdown is currently approved. With this gate it's blocked |
| RM-16 | **Synthetic control attribution** — for each treated period, build synthetic control from similar non-treated periods (matched on volatility regime + BTC trend) | 3-4 days | Current `market_drift = ref_post − ref_pre` is crude; bootstrap CIs straddle zero on mid-size effects. Synthetic control gives tighter CIs on same data → more decisions correctly classified hit/miss → faster correct iteration |
| RM-17 | **Bayesian optimization** for numerical thresholds (Optuna) — replace point-estimate rule heuristics with Gaussian Process surrogate over 20 trials | 1 week | For continuous params (`ENTRY_SCORE_FLOOR`, `ML_PROBA_MIN_*`), BO explores the Pareto frontier in 20 backtests where rule-based gives one number. Higher chance of finding the actual optimum |

### Sprint 2 — Close auto-execution gap (≈ 1 week)

| ID | Item | Effort | Why for North Star |
|----|------|--------|---------------------|
| RM-4 | **Runtime config-override mechanism** — bot reads `decisions.jsonl` at startup and applies approved overrides on top of `config.py` defaults; no daemon writing to `config.py` | 1 day | Today every apply requires 4 manual steps. Auto-execute is unsafe without this layer. Speeds operator cycle from hours to seconds and unblocks RM-5/RM-18 |
| RM-5 | **Auto-rollback** gated by `AUTO_ROLLBACK_ENABLED=False` default, uses RM-4 | 4h | When attribution says "miss with CI excluding zero on wrong side", system rolls back without operator action. Saves NS from dragging during ignored regression |
| RM-18 | **Canary rollout** (5% → 25% → 100% by trades-per-day) with mSPRT sequential test gating each promotion | 1 week | Today each apply is full bet on 14 days. Canary cuts initial risk 20× and surfaces a bad change in ~3 days instead of 14 |

### Sprint 3 — Learning system upgrades (≈ 1 week)

| ID | Item | Effort | Why for North Star |
|----|------|--------|---------------------|
| RM-7 | Include **config-state hash** in bandit context vector | 1h | Bandit currently doesn't know which gates are enabled. When config changes, the policy is silently stale. With hash in context, bandit re-learns per config epoch |
| RM-8 | **Outcome-aware hypothesis priority** — mini-classifier on `decisions.jsonl` predicting `P(hit \| hypothesis_features)` | 1 day | L2 today ranks by severity only. With this, hypotheses similar to past hits surface first → fewer wasted approve cycles |
| RM-9 | Capture **operator-vs-advisor disagreement** reasons → calibration dataset for L6 Claude advisor system prompt | half day | When operator overrides Claude's reject, the disagreement signal is lost. With this, Claude advisor gradually learns operator's risk tolerance |

### Sprint 4 — Operator visibility & ROI (≈ 1 week)

| ID | Item | Effort | Why for North Star |
|----|------|--------|---------------------|
| RM-6 | **ROI dashboard** (`pipeline_roi.py`) — quarterly trend of `hit_rate`, `regression_rate`, per-target attribution | half day | Today operator sees one Telegram message per day, no longitudinal view. Dashboard makes "are we improving over time?" answerable in 5 sec |
| RM-10 | **Drill-down** — per-`trade_id` step-by-step decision path explaining why each premature exit / loss happened | half day | When Telegram says "GLMUSDT premature exit", operator can't act on it. Drill-down points to the specific decision step (trail_k value? EMA cross timing? orderflow signal?) so the next hypothesis is targeted |
| RM-11 | **Portfolio-level rolling alpha vs B&H** as primary attribution metric | 1 day | Per-trade NS is necessary but not sufficient. Portfolio alpha captures interaction effects between concurrent positions and rotation decisions |

### Sprint 5 — Regime + causal frontier (≈ 2-3 weeks, optional)

These are the highest-effort items; only attempt after Sprints 0-4 are stable.

| ID | Item | Effort | Why for North Star |
|----|------|--------|---------------------|
| RM-19 | **Regime-conditional policies** — HMM with changepoint detection; parameters conditional on regime (bull / chop / bear) | 2-3 weeks | Parameters optimal in bull market fail in chop. Regime-conditional config eliminates the worst class of false positives we currently miss |
| RM-20 | **Causal graph** for hypothesis interactions — Structural Causal Model + mediation analysis | 2 weeks | Apply A + Apply B ≠ sum of individual effects (gates interact). Causal graph predicts interaction; without it we sometimes "improve" NS by combination of changes that individually would be approved but together regress |

### Sprint 6 — Robustness (background, ongoing)

| ID | Item | Effort | Why for North Star |
|----|------|--------|---------------------|
| RM-12 | Stress test catalog expansion (Claude timeout, race conditions, malformed payloads, cold-start, concurrent decision retry) | 1 day | Pipeline outage = no NS improvement. Each new failure-mode tested = one less days/weeks of silent stagnation |
| RM-13 | Time-zone unit tests + UTC normalisation at write time | 3h | Off-by-one-day reports already exist in critic phase mixing; risk of silently using wrong day's data for attribution |

### Already covered by other specs (out of this roadmap)

- L4-g (bot-side per-feature shadow) — partial coverage exists; expanding is feature-by-feature, tracked in each feature's own spec.

---

## 5b. Roadmap → North Star chain (verification)

The user's challenge: "удостоверься, что roadmap ведет к достижению метрик
назначения бота". Here's the explicit causal chain.

```
NS = early_capture_pct
      ↑
      ├─ depends on signal QUALITY (bot's BUYs hit top-20 early)
      │     ↑
      │     ├─ depends on model accuracy (already ML-a..d) ✅
      │     ├─ depends on filter calibration (gates not blocking winners) ← RM-1, RM-2, RM-17
      │     ├─ depends on fast-reversal suppression (no whipsaw entries)  ← RM-3
      │     └─ depends on regime awareness (bull vs chop)                 ← RM-19
      │
      ├─ depends on signal QUANTITY (enough BUYs to catch top-20)
      │     ↑
      │     ├─ depends on gate looseness                  ← RM-1, RM-2 (validate relax)
      │     ├─ depends on watchlist coverage              ← RM-4 (apply approved widen)
      │     └─ depends on canary safety on volume changes ← RM-18
      │
      ├─ depends on PIPELINE EFFECTIVENESS at finding & applying improvements
      │     ↑
      │     ├─ depends on validation rigour      ← RM-16 (synthetic control)
      │     ├─ depends on learning from outcomes ← RM-8 (outcome-aware priority)
      │     ├─ depends on tight cycle time       ← RM-4, RM-5 (auto-execute)
      │     ├─ depends on multi-objective sanity ← RM-15 (no Sharpe regressions)
      │     └─ depends on drift detection        ← RM-14 (catch staleness early)
      │
      └─ depends on OPERATOR VISIBILITY (acts on right things)
            ↑
            ├─ ROI dashboard            ← RM-6
            ├─ Drill-down incident root ← RM-10
            └─ Portfolio-level alpha    ← RM-11
```

**Every RM-N exists to remove one specific friction in this chain.** No roadmap
item is "nice to have" — each has a target node in the diagram that it
unblocks or strengthens.

---

## 5c. Sprint priorities (what to do next)

If resources are constrained, follow this order:

1. **Sprint 0 (P0)** — RM-1, RM-2, RM-3. These unblock entire classes of future work.
2. **Sprint 1** (today's most achievable upgrade) — RM-14 + RM-15 first (≈ 2 days, no dependencies, immediate NS protection); then RM-16 + RM-17 (≈ 2 weeks, jumps to research-quality).
3. **Sprint 2** — RM-4 (foundation), then RM-5, RM-18 (depend on RM-4).
4. **Sprints 3-6** — schedule by appetite.

The reason RM-14 + RM-15 come first in Sprint 1: they are **defensive**
(prevent NS degradation) and have zero dependencies, so they can ship today.
RM-16 + RM-17 are offensive (improve NS) but heavier — start them once the
defensive floor is in.

---

## 6. Update protocol

When you ship a change that touches any component in §3:

1. **Edit the status row.** Change `❌` → `🟡` → `✅` as appropriate. Update `Tests` column with the relevant test module count.
2. **If the change is measurable on the North Star, add a row to §4** with the date, the metric value (from `final` phase critic), and reference to the source `decision_id`.
3. **Bump `Last updated` at the top.**
4. **If the change closes a roadmap item, strike it through in §5** and add any follow-ups discovered while shipping.
5. **Reference this spec slug (`auto-improvement-loop`) in the PR description** so reviewers know to verify the spec is in sync.

**Rule of thumb:** if a reviewer can't tell from this spec alone what's working vs not, the spec is wrong, not the reviewer.

---

## 7. Non-goals

These are deliberately out of scope and should not be added without explicit discussion:

- **A general-purpose MLOps platform.** This spec is for THIS bot, not a framework.
- **Real-time RL from each tick.** The loop is daily/weekly, not millisecond. RL inside `monitor.py` is separate (existing contextual bandit).
- **Web UI / dashboard.** Telegram is the operator surface. Logs are diagnosis surface.
- **Multi-bot generalisation.** `gpt_crypto_bot` is explicitly out of scope (CLAUDE.md §1).

---

## 8. Anti-patterns observed during construction (learned the hard way)

These caused real bugs in earlier iterations. Don't repeat:

1. **Two reports for the same metric (midday vs final), both sent to Telegram.** Confuses the operator and gets blamed on the bot.
2. **Health report using midday phase because final wasn't ready yet at 03:00.** Always pick final if available; midday is a fallback.
3. **Disk I/O on the UI render path.** Cache + background refresh.
4. **Generating hypotheses without validators.** They all become `pending_manual_validation` and clutter the review block.
5. **Counterfactual sim on blocked events where `ret_5 = 0` because trade didn't happen.** Looks like "no effect" but is actually "no signal" — must caveat.
6. **Approving a change without pinning the baseline.** Without it, post-attribution is naive before/after.
7. **Letting `decisions.jsonl` and `already_tried.jsonl` drift.** Both must be appended in the same code path.
