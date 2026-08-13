# Continuous-improvement agent — architecture design

- **Slug:** `continuous-improvement-agent`
- **Status:** design (no implementation)
- **Created:** 2026-08-13
- **Owner:** Vasiliy Ostrovsky + Claude
- **Objective:** a closed loop in which an LLM agent reads the bot's own data,
  proposes changes to raise `NS_EarlyCapture@top20`, hands them to an
  independent validator, and decides what happens next — without ever being able
  to fool itself or the operator.

---

## 0. The honest starting point

This component already exists and is dead. Any design that ignores that will
rebuild the corpse.

`pipeline_hypothesis.py → pipeline_validator.py → pipeline_shadow.py →
pipeline_blind_critic.py → pipeline_approve.py → pipeline_monitor.py →
pipeline_attribution.py` are all present, wired, and scheduled. And:

- all 16 pending L2 hypotheses referenced `config_key`s **that do not exist** in
  `config.py` (`MAX_LATENESS_PCT_IMPULSE_SPEED`, `ML_PROBA_MIN_IMPULSE_SPEED`,
  `ENTRY_SCORE_MIN_5M`, …), and two proposed position sizing, which this bot
  does not have;
- none had a registered validator, so L3 could not check them either;
- **no decision has come through the pipeline since 2026-06-17.** Every change
  since — the bandit leak fix, the curtail fallback, the soft gate, the
  monitored-set change — came from manual analysis.

Four root causes, and each becomes a structural constraint below:

| Root cause | Constraint it forces |
|---|---|
| The generator wrote free text; nothing checked the proposal was even *expressible* | **Typed hypothesis contract validated against a machine-derived capability registry** (§3.2, §4.1) |
| Dedup was exact-key, so semantically identical retries were never caught | **Semantic dedup against a negative-results register** (§3.6) |
| A hypothesis with no registered validator still entered the queue | **No validator ⇒ no queue entry**; validators are a registry, not an afterthought (§4.2) |
| Nothing measured whether the generator was any good | **Meta-evaluation of the agent itself, on a temporal holdout** (§7) |

The founding principle (CLAUDE.md §0) is unchanged: continuous learning is P0.
This design is what makes it survive contact with an LLM.

---

## 1. Design principles

1. **The agent proposes; deterministic code disposes.** No LLM output ever
   reaches `config.py`, a dataset, or a live gate without passing a
   non-LLM validator and a recorded decision.
2. **Integrity boundary around the scorer.** The component that measures a
   hypothesis is a separate process the agent has no write access to. An agent
   that can edit its own grader is not being evaluated.
3. **The metric is immutable to the agent.** Changing a metric, its denominator
   or its label provenance is a human-approved path (§3.5). Otherwise the loop
   optimises the ruler.
4. **Every number carries its provenance.** Data snapshot hash, split, base
   rate, lift, uptime adjustment — all machine-checked by the Truth Harness
   (TH-01…TH-12) at every boundary, not by the agent's good intentions.
5. **"Underpowered" is a first-class verdict.** "Рано судить" must be a state
   the machine can be in, or the loop will manufacture conclusions (§0a rule 10).
6. **Falsifiability is required at authoring time.** A hypothesis that cannot
   state what result would kill it is not admitted.
7. **Budgets are hard.** Tokens, validation compute, operator attention, and
   live blast radius are all capped; exceeding a cap stops the loop rather than
   degrading quietly.

---

## 2. Architecture — six planes

```
┌─ CONTROL PLANE ──────────────────────────────────────────────────────────┐
│  Truth Harness (TH-01..12) · policy engine (do_not_touch, budgets)        │
│  lifecycle hooks · circuit breaker · audit log                            │
└──────────────┬───────────────────────────────────────────────────────────┘
               │ every arrow below crosses a hook
┌─ EVIDENCE PLANE ─────────────┐        ┌─ AGENT PLANE (LLM) ──────────────┐
│ event_store.sqlite (offsets) │  MCP   │ Analyst   → incidents            │
│ feature/label stores         │◄──────►│ Author    → hypotheses           │
│ metric registry (canonical)  │ (read) │ Adversary → kill-or-pass (judge) │
│ capability registry (AST)    │        │ Referee   → blind verdict (judge)│
│ RAG index: specs, decisions, │        │ Allocator → what to spend on     │
│   reports, negative register │        │ Historian → memory writes        │
└──────────────┬───────────────┘        └──────────────┬───────────────────┘
               │                                        │ propose() only
               │                          ┌─────────────▼───────────────────┐
               │                          │ VALIDATION PLANE (no LLM)       │
               │                          │ validator registry · replay     │
               └─────────────────────────►│ temporal split · anchors        │
                                          │ leakage checks · power analysis │
                                          └─────────────┬───────────────────┘
                                                        │ validation report
                                          ┌─────────────▼───────────────────┐
                                          │ PROMOTION PLANE                 │
                                          │ shadow → forward cohort → live  │
                                          │ flag + rollback + attribution   │
                                          └─────────────┬───────────────────┘
                                                        │ outcome
                                          ┌─────────────▼───────────────────┐
                                          │ MEMORY PLANE                    │
                                          │ decisions · already_tried ·     │
                                          │ negative register · agent trace │
                                          └─────────────────────────────────┘
```

The loop is a cycle, but **information is asymmetric on purpose**: the Referee
never sees the hypothesis author's predicted effect, and the Validation plane
never sees the agent's rationale. Both are anti-confirmation-bias measures.

---

## 3. Evidence plane

### 3.1 Structured evidence — the event store

`files/event_store.py` (shipped): the JSONL journal stays authoritative and a
SQLite mirror is synced by byte offset. Two properties this design leans on:

- **Queryable diagnosis.** The Analyst asks SQL questions ("which gate blocked
  top-20 winners in the last 14 uptime-adjusted days") instead of re-parsing
  98 MB per question.
- **Byte offsets are content addresses.** For an append-only journal,
  `(source_file, byte_offset, row_count)` uniquely identifies a prefix of
  history. Every validation pins that triple as its **data snapshot hash**, so a
  result is reproducible and two results are comparable only when their
  snapshots are (TH-04). This falls out of the store for free and is the
  cleanest provenance mechanism available here.

### 3.2 Capability registry — the fix for invented config keys

Derived mechanically from source, never hand-maintained:

- every `config.py` constant: name, type, current value, permitted range,
  the module that reads it, whether it is in `do_not_touch.json`;
- every registered gate (`trend_scout_rules.BlockRule`), entry mode, exit
  policy, reward term, bandit context feature;
- every canonical metric and its computing script.

The Author agent may only propose changes whose `target` resolves here. The 16
dead hypotheses become **unrepresentable**, not merely rejected.

### 3.3 RAG over project knowledge

Hybrid retrieval (BM25 + embeddings) over: `docs/specs/**`, `docs/reports/**`,
`CLAUDE.md`, `PROJECT_CONTEXT.md`, `decisions.jsonl`, `already_tried.jsonl`, the
negative-results register, and backtest docstring verdicts.

Two non-obvious requirements:

- **Chunk on semantic units** (one spec section, one decision record, one
  verdict block), because a retrieved half-verdict is worse than none.
- **Recency and expiry are ranking features.** TH-10 already expires evidence;
  retrieval must prefer live evidence and mark stale evidence as stale in the
  context window, or the agent will confidently cite a refuted 77-day-old claim.

### 3.4 Feature/label stores

`top_gainer_dataset.jsonl`, `critic_dataset.jsonl`, `ml_dataset.jsonl`, klines.
Read-only to the agent. The **label provenance** of each is a first-class field
(`rolling_24h_same_snapshot` today) so any hypothesis resting on a leaky label
is flagged before compute is spent — this is exactly how "recall@20 = 100%"
survived for months.

### 3.5 Metric registry — immutable to the agent

One canonical metric per business question (the existing
`metrics-canonical-spec`), each with: definition, denominator name, label
provenance, uptime policy, owner, version. The agent can *read* it and *propose*
a change through a separate human path, but the loop's scoring always uses the
registered version. Without this, the cheapest way to raise a metric is to
redefine it — and this repo has already had one silent denominator change.

### 3.6 Negative-results register

Every refuted hypothesis with the numbers that killed it, plus the nine already
in the repo and the four inherited from the sibling bot (static threshold exits,
early RSI-WEAK profiles, impulse-expansion tails, pure-rank labels). Dedup
against it is **semantic** (embedding similarity + mechanism-class match), not
exact-key: the historical dedup compared config keys and therefore never fired.

---

## 4. Agent plane — roles and why they are separate

One model, several roles, each with its own skill, tool subset, context budget
and success metric. Separation is not aesthetic: it is how information asymmetry
is enforced.

### 4.1 Analyst → incidents

Runs the **Failure Casebook** pattern on a schedule: rank concrete historical
cases by opportunity cost (blocked winners, late entries, giveback), emit
*structured incidents* with case ids and source rows — never prose.
Explicitly forbidden from proposing solutions; that separation stops the
diagnosis from being written backwards from a favoured fix.

**Success metric:** fraction of incidents that later appear in a supported
hypothesis.

### 4.2 Author → hypotheses

Consumes incidents + RAG + capability registry. Emits the typed contract:

```
hypothesis {
  id, created_at, generation, incident_refs[]
  intent      { metric: <registry id>, direction, predicted_delta, confidence }
  mechanism   { kind: gate_threshold | reward_shape | feature_add | exit_policy
                      | routing | data_collection,
                target: <capability registry id>, change: {from, to} }
  population  { which rows this affects, expected n/day, base rate }
  rationale   { evidence_refs[]: (query_hash | report_id | case_id), causal_story }
  falsifier   { the result that kills this }          # required
  validator   { registered_validator_id, params }      # must exist
  risk        { blast_radius, rollback, do_not_touch_ok }
  th_rules[]  { which harness invariants this must satisfy }
}
```

Rejected at authoring time, before any compute: unresolvable target, missing
validator, missing falsifier, `do_not_touch` violation, semantic duplicate of a
refuted hypothesis, or a predicted effect on a metric the mechanism cannot
plausibly touch.

**Self-consistency:** k independent generations at temperature; only hypotheses
that survive de-duplication *and* the Adversary proceed. Divergence across
samples is itself a signal — a hypothesis only one sample produced is ranked
lower.

### 4.3 Adversary → kill before compute (LLM-as-judge #1)

Cheap, runs before validation. Attacks the hypothesis with a fixed rubric drawn
from §0a: is the target leaky, is the population one the bot actually samples
(rule 6), is the claimed effect a base-rate illusion (rule 1), is this the
tenth retry of a refuted idea, is the causal story tautological (the
"lateness = near-peak = low upside" failure)?

**Success metric — and this is measurable:** of the hypotheses it killed, how
many would have validated positive (false-kill rate), estimated by periodically
letting a random 10% through regardless. An adversary with a 0% false-kill rate
is too permissive; one above ~15% is destroying value.

### 4.4 Referee → blind verdict (LLM-as-judge #2)

Sees the **validation report only**: numbers, splits, anchors, power — with the
hypothesis text, the predicted effect and the author's rationale stripped. Judges
whether the evidence supports a change, and writes the recommendation the
operator reads.

Blindness is the point: the existing `pipeline_blind_critic.py` already encodes
this idea, and it is the single cheapest defence against a persuasive rationale
carrying a weak result.

### 4.5 Allocator → spend the scarce resource

Validation compute and operator attention are the binding constraints, not
ideas. The Allocator runs **Thompson sampling over hypothesis families**
(gate relaxation, reward shaping, exit policy, feature addition, data
collection), with per-family priors updated from `decisions.jsonl` outcomes.

There is a pleasing symmetry here: the bot uses a bandit to decide which
candidates to enter; the improvement loop uses a bandit to decide which
hypotheses to test. Both must publish base rate and lift.

### 4.6 Historian → memory writes

The only agent with write tools, and they write **only** to memory artifacts
(decisions, already_tried, negative register, agent trace) — never to config,
never to datasets. Every write is append-only and hashed.

---

## 5. Validation plane — the integrity boundary

**No LLM anywhere in this plane.** It runs as a separate process with its own
MCP endpoint; the agent submits a hypothesis id and polls for a report.

### 5.1 Validator registry

Each `mechanism.kind` maps to registered validators with fixed contracts:

| kind | validator | anchors it must report |
|---|---|---|
| `gate_threshold` | Pareto sweep on the bot's **own** entries | current policy, relax-all, block-all |
| `reward_shape` | offline bandit refit + temporal holdout | old label, random policy |
| `feature_add` | ablation with and without, same split | leaky-feature ablation |
| `exit_policy` | forward-path replay from real entries | base exit, hold-to-EOD |
| `routing` | counterfactual mode reassignment | as-is |
| `data_collection` | coverage/power estimate only | — |

### 5.2 Mandatory report contract

```
validation_report {
  hypothesis_id, validator_id, started_at, elapsed
  data_snapshot { sources[], byte_offsets[], row_counts[], window, hash }
  split         { kind: temporal, train_window, holdout_window, days_full,
                  days_excluded_down }
  population    { n, base_rate, is_bot_own_entries: bool }
  result        { primary, value, ci, lift_vs_base, action_rate,
                  estimated_NS_delta }
  anchors       { … }                       # per §5.1
  leakage       { same_snapshot_label, answer_encoding_features[], verdict }
  power         { min_detectable_effect, achieved_n, verdict }
  verdict       supported | refuted | underpowered | invalid
  reproduce     { seed, cmd, code_commit }
}
```

`invalid` means the validator itself could not run honestly (stale data, failed
freshness SLO, snapshot mismatch). A failed harness is never a pass.

### 5.3 Preconditions

Before any validation runs: artifact freshness SLO green (shipped,
`TH05_ARTIFACT_FRESHNESS`), data snapshot resolvable, code commit clean. A loop
that trains on a stalled input produces confident nonsense — this repo lost
58 days of labels to exactly that.

---

## 6. Promotion plane

Three gates, none skippable, each with a stated rollback:

1. **Shadow.** Behaviour logged, not applied. Required whenever the shadow can
   answer the question (§0a rule 7).
2. **Independent forward cohort.** Rows strictly after the decision timestamp,
   maturity rule (both T+5 and T+10 present), minimum sample **and** minimum
   days, then numeric thresholds. Borrowed from the sibling bot's
   `forward-shadow-promotion-gates`; the cost of not having it is documented —
   the 8% trail widen backtested positive over 35 days and lost 54.9% cumulative
   in 5 live days.
3. **Live behind a flag,** default = current behaviour, rollback = flip the flag,
   decision recorded in `decisions.jsonl` (which auto-applies as a runtime
   override — so a rollback needs a superseding record, not a config edit).

**Attribution** (L7, exists) measures the realised effect with bootstrap CI and
market-drift normalisation, and writes the outcome back to the Allocator's
priors and the negative register. This is the only arrow that makes the loop
*learn* rather than merely *act*.

---

## 7. Meta-evaluation — how we know the agent is worth its tokens

Without this the loop is theatre. Three measurements, all on temporal holdouts:

1. **Agent backtest.** Freeze the entire corpus as of date T (trivial with byte
   offsets), run the loop, and compare its proposals against what actually
   happened after T. Did it find the changes that worked? Did it avoid the nine
   refuted ones? Scored as recall of known-good and rejection rate of
   known-bad — **with base rates published**, since a generator that proposes
   everything trivially "finds" every win.
2. **Judge calibration.** A labelled set of past hypotheses with known outcomes;
   both judges are scored on it, and the score is reported alongside their
   verdicts. An uncalibrated judge is a random number generator with prose.
3. **Loop yield.** Decisions per month that reached live and survived
   attribution, versus the manual baseline. If the loop is below the human
   baseline it is a research project, not a component, and should be labelled as
   such in the morning report.

---

## 8. Cross-cutting mechanisms

**MCP as the tool boundary.** Two servers: `evidence` (read-only: query events,
get metric, search knowledge, resolve capability, read decisions) and
`validation` (submit, poll, fetch report). Least privilege is enforced by the
protocol, not by prompt instructions — the agent has no tool that writes config
or datasets, so no jailbreak or confused-deputy path leads there. Every call is
logged with arguments and result hash for the audit trail.

**Hooks** at fixed lifecycle points:

| Hook | Enforces |
|---|---|
| `pre_tool_use` | budget, do_not_touch, argument schema |
| `post_generation` | hypothesis contract + harness checks |
| `pre_validation` | freshness SLO, snapshot resolvable |
| `post_validation` | report contract completeness, ratio context present |
| `pre_promotion` | forward-cohort gate, rollback stated |
| `post_attribution` | memory write, prior update |
| `pre_commit` | existing `truth_harness change --staged` |

**Skills** as versioned procedures in `.claude/skills/` and `skills/`: truth
harness audit (exists), failure casebook, hypothesis authoring, promotion gate,
MD compliance (exists). Skills are how a role's method is reviewed and changed
deliberately, rather than drifting inside a prompt.

**Circuit breaker.** The loop halts and notifies when: two consecutive
promotions regress in attribution, the harness reports a blocking finding, a
freshness SLO lapses, or a budget is exhausted. Halting is a normal outcome, not
an error.

---

## 9. Failure modes this design is built against

Each maps to something that actually happened here.

| Failure | Historical instance | Structural defence |
|---|---|---|
| Proposals reference nothing real | 16 invented `config_key`s | Capability registry; unresolvable target is unrepresentable |
| Refuted ideas re-tested | exact-key dedup never fired | Semantic dedup vs negative register |
| Leaky label taken as skill | "recall@20 = 100%", AUC 0.99 | Label provenance field + mandatory leakage section |
| Base-rate illusion | ENTER on 73% of everything | Ratio context enforced by TH-01 in the report contract |
| In-sample reported as achievement | bandit post-fit recall | Temporal split enforced by the validator, not the agent |
| Metric redefined instead of improved | April denominator change | Metric registry immutable to the agent |
| Backtest→live gap | 8% trail widen, −54.9% in 5 days | Forward-cohort gate |
| Silent input stall | 58-day backfill lock | Freshness SLO as a validation precondition |
| Agent grades itself | — | Validation plane is a separate process, no write tools |
| Confident prose over weak numbers | — | Blind Referee sees numbers only |
| Cost blowup | — | Hard budgets + circuit breaker |

---

## 10. Rollout phases, each with an exit gate

| Phase | Builds | Exit gate |
|---|---|---|
| 0 | Capability registry, metric registry, contracts, negative register | Registry reproduces every existing config key and gate; the 16 dead hypotheses are rejected mechanically |
| 1 | Analyst + Author in **propose-only** mode; MCP evidence server | ≥10 hypotheses, 100% contract-valid, ≥3 judged worth testing by the operator |
| 2 | Validation plane + validator registry | Re-validates three historical decisions and reproduces their published numbers within CI |
| 3 | Adversary + Allocator | Adversary false-kill rate measurable and <15% on the calibration set |
| 4 | Blind Referee + shadow promotion | One hypothesis reaches shadow through the loop end-to-end with no manual step |
| 5 | Forward-cohort gate + operator-approved live promotion | One live promotion survives attribution; agent backtest published |

Phases 0–2 contain no autonomous action at all. That is deliberate: the previous
version of this component failed at phase 0 and nobody noticed for two months.

---

## 11. Explicitly out of scope

- **No LLM in the live trading path.** Latency, non-determinism and
  auditability all forbid it; the bandit stays the online decision-maker.
- **No agent-initiated threshold tuning** without a validation report.
- **No metric or denominator changes by the agent.**
- **No watchlist changes** (immutable, CLAUDE.md §14).
- **No new long-running worker** until phase 4, and then only with a launcher, a
  status check and integration into `restart_full_stack.bat`.

---

## 12. Open questions for the operator

1. **Autonomy ceiling.** Should phase 5 ever auto-promote to live behind a flag,
   or is operator approval permanent? The design supports both; the difference
   is one policy setting and a much larger blast-radius budget.
2. **Budget.** What daily token and validation-compute budget is acceptable? The
   Allocator needs a number to allocate against.
3. **Cadence.** Nightly (with the EOD cycle) or weekly? Weekly gives attribution
   time to mature; nightly finds stalls sooner.
4. **North Star provenance first?** The metric is still `provisional` because
   ground truth comes from the same rolling-24h snapshot as the features. A loop
   optimising a provisional metric will faithfully optimise its bias. Building
   immutable later-EOD labels first may be the correct phase 0.
