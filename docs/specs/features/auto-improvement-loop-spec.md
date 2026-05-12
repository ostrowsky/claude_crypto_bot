# Auto-Improvement Loop — full spec, living document

- **Slug:** `auto-improvement-loop`
- **Status:** in-progress — see component matrix below
- **Created:** 2026-05-12
- **Last updated:** 2026-05-12
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

Conventions for filling the table:
- One row per measurable change. Re-measurements every 7/14/30 days get
  their own rows.
- `early_capture_pct` from `final` phase only (per CLAUDE.md §0 honest-measurement principle).
- `hit_rate` is cumulative pipeline meta-metric since project start. Target: ≥ 0.60.
- `Change applied` is a one-line human description.
- `Source decision_id` points to `decisions.jsonl` for traceability.

---

## 5. Roadmap — prioritised

### P0 — Unblock universal validation (loop cannot close without these)

| ID | Item | Effort | Unblocks |
|----|------|--------|----------|
| RM-1 | Extend `bot_events.jsonl` blocked events with full decision state (`ranker_final_score`, `candidate_score`, `score_floor`, `ml_proba`, `gate_threshold`) | 4-6h | L3-c, L4-c, L2-h |
| RM-2 | Structured `reason_code` enum in blocked events | 2h | L3-c, L2-h |
| RM-3 | Anti-fast-reversal full chain (label → proba → guard → bandit reward) — see [`anti-fast-reversal-spec.md`](anti-fast-reversal-spec.md) | 2-3 days | ML-f → ML-i |

### P1 — Close the auto-execution gap

| ID | Item | Effort |
|----|------|--------|
| RM-4 | Runtime config-override mechanism (`decisions.jsonl` → applied overrides at bot startup, no `config.py` edit needed) | 1 day |
| RM-5 | Auto-rollback gated by `AUTO_ROLLBACK_ENABLED` flag (default off), uses RM-4 | 4h |
| RM-6 | ROI dashboard (`pipeline_roi.py`) — quarterly trend of hit_rate, regression_rate, per-target attributions | half day |

### P1 — Learning system upgrades

| ID | Item | Effort |
|----|------|--------|
| RM-7 | Include config-state hash in bandit context | 1h |
| RM-8 | Outcome-aware hypothesis priority (mini-classifier on `decisions.jsonl` history) | 1 day |
| RM-9 | Capture operator-vs-advisor disagreement reasons → feedback loop for Claude advisor prompt | half day |

### P2 — Operator visibility

| ID | Item | Effort |
|----|------|--------|
| RM-10 | Drill-down: "why did this trade go wrong" (per trade_id step-by-step) | half day |
| RM-11 | Portfolio-level rolling alpha vs B&H metric | 1 day |

### P3 — Robustness

| ID | Item | Effort |
|----|------|--------|
| RM-12 | Stress test catalog expansion (Claude timeout, race conditions, malformed payloads, cold-start path) | 1 day |
| RM-13 | Time-zone unit tests + UTC normalisation at write time | 3h |

### P4 — Already covered by other specs

- L4-g (bot-side per-config-key shadow) — partial coverage exists; expanding is feature-by-feature, not infrastructure work. Treat as part of each feature's own spec.

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
