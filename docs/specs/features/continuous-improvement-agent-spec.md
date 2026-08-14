# Continuous-improvement agent — architecture design (v2)

- **Slug:** `continuous-improvement-agent`
- **Status:** design (no implementation)
- **Created:** 2026-08-13 · **Revised:** 2026-08-14 after external review
- **Owner:** Vasiliy Ostrovsky + Claude
- **Objective:** a closed loop in which an LLM agent reads the bot's own data,
  proposes changes that raise early capture of top movers, hands them to an
  independent validator, and decides what happens next — without being able to
  fool itself, the operator, or the runtime.

## Revision note — v2.1, and an uncomfortable pattern

**A third review found that six of the defects I had demanded the competing
design fix were absent from this one.** Walking skeleton before platform, a
power-expansion branch, a defined `policy_epoch`, an operational contract with
one owner and an acknowledgement SLO, migration of existing negative results,
and outcome-based acceptance — I required all six of them elsewhere and shipped
none of them here. That asymmetry is itself a finding: a reviewer is far better
at seeing a missing gate than at noticing its absence in their own design.
Corrected in §5.6, §3.7, §6.4, §7, §9 and §13.

## Revision note — what v1 got wrong

Two reviews (one from a GPT-authored competing design, one a direct critique of
v1) found defects that v1's own claims contradicted. They are corrected here and
listed openly, because a design document that quietly absorbs its errors is the
same failure mode as a report that quietly absorbs a bad week.

| v1 claim | Why it was wrong | v2 |
|---|---|---|
| "The agent has no tool that writes config" **and** the Historian writes `decisions.jsonl` | Confirmed live on 2026-08-14: `_config_runtime_overrides.py` applies `diff.to` at every `import config`, `pipeline_approve._maybe_auto_restart` spawns a detached restart, and `PIPELINE_AUTO_APPLY` **defaults to on**. Two gating constants are overridden right now, and the newest approved record in the file was written by an LLM | §4.6, §6.1 — four separate stores. The real switch is `AUTO_APPLY_OVERRIDES_ENABLED` (config.py); `PIPELINE_AUTO_APPLY=0` only suppresses the auto-restart and was named wrongly in an earlier draft |
| `(source_file, byte_offset, row_count)` is a content address | An offset identifies a position, not content. This repo demonstrably rewrites these files — 1.09 GB of orphaned `.tmp` from failed whole-file rewrites was cleared on 2026-08-13 | §3.1 — hash chain over prefixes; offsets keep only their real job, incremental reading |
| "No arrow when \|Δ\| < MDE" | MDE is a design-time power parameter, not a decision rule on observed data | §13.3 — CI versus a pre-registered practical-significance threshold |
| MDE figures (±22 pp, ±17 pp) | Computed assuming independent coin-days. Crypto moves inside a day are strongly correlated, so effective n is materially lower and **those figures are optimistic** | §13.2 — day-clustered bootstrap, numbers restated as upper bounds on power |
| `Coverage@move5` and `Precision@alert` are "two sides of the same event" | They used different anchors and horizons (intraday to midpoint vs +5% within 24h) | §13.2 — one `MoveEvent` definition serves both |
| Thompson sampling over hypothesis families | Delayed outcomes, few experiments and shifting policy epochs make it fit the validator, not the market | §4.5 — expected impact × information gain ÷ cost, with a fixed exploration quota |
| Self-consistency over k generations | Conflicts with a minimal budget, and agreement among samples of one model is not evidence | §4.2 — dropped |
| "MCP enforces least privilege" | MCP is a protocol. Enforcement is server-side ACL and capability tokens | §8 |
| Loop utility measured as decisions/month | Rewards churn | §7 — avoided harm, research precision, share of correct `NO_CHANGE` |

Also adopted from the competing design: purge/embargo, a sealed holdout the
generator never sees, a multiple-testing ledger, placebo/negative-control runs,
immutable versioned contracts, retry invariance, retrieved content treated as
untrusted data, and `NO_CHANGE` as a normal successful outcome.

---

## 0. The honest starting point

This component already exists and is dead. Any design that ignores that rebuilds
the corpse.

`pipeline_hypothesis.py → … → pipeline_attribution.py` are present, wired and
scheduled. And: all 16 pending hypotheses referenced `config_key`s that **do not
exist**; none had a registered validator; **no decision has come through the
pipeline since 2026-06-17.** Every change since — the bandit leak fix, the
curtail fallback, the soft gate, the monitored-set change — came from manual
analysis.

The measurement layer is also not currently trustworthy. `truth_harness full`,
`as_of 2026-08-14T09:12Z` @ `419c8b7`: **7 blocking findings and 1 warning** —
leaky North-Star and top-gainer targets (TH-03 ×2), a row-index split that can
put one UTC day on both sides (TH-04), no canonical portfolio alpha and no
canonical ZigZag EX1 (TH-11 ×2), gate evidence 77 days past a 30-day budget
(TH-10), a stale ranker artifact (TH-05, transient), and 47 legacy backtests
without a durable verdict (TH-08, warning).

The count carries a timestamp because it moves: freshness findings clear on
their own, and a harness run inside a git worktree sees different runtime
artifacts than one at the runtime root.

**Therefore the first phase is repairing measurement and provenance, not
launching agents.** The loop starts in `RESEARCH_ONLY` — but the exit condition
is **zero blocking findings in the product-scoped profiles**
(`discovery/alert`, `exit`), not a globally green harness. TH-11 demands
canonical portfolio alpha for a product with no position sizing, so "green
everywhere" is not a condition this system can ever satisfy, and writing it as
the gate would build a permanent freeze.

Four root causes of the previous death, each now a structural constraint:

| Root cause | Constraint |
|---|---|
| Free text; nothing checked the proposal was expressible | Typed contract against a capability registry derived from source (§3.2, §4.2) |
| Exact-key dedup never fired | Semantic similarity **warning**, not an automatic ban (§3.6) |
| A hypothesis with no validator still entered the queue | No validator ⇒ no registration (§5.1) |
| Nothing measured whether the generator was any good | Meta-evaluation on a frozen-world holdout (§7) |

---

## 1. Design principles

1. **The agent proposes; deterministic code disposes.** No LLM output reaches
   `config.py`, a dataset, a runtime override or a live gate without a non-LLM
   validation report and a signed approval.
2. **The execution channel is not writable by any agent.** This is the v1 defect
   worth stating as a principle rather than a footnote (§6.1).
3. **Integrity boundary around the scorer.** The validator is a separate process
   with its own identity, a read-only frozen snapshot, and a sealed holdout the
   proposing agent never sees.
4. **The objective is immutable to the agent.** Metric, denominator, label
   provenance and guardrails are a human-approved contract (§3.5).
5. **Provenance is cryptographic, not positional.** Hashes of data prefix,
   schema, universe, code and config — not byte offsets (§3.1).
6. **`NO_CHANGE`, `UNDERPOWERED`, `INVALID` and `REJECTED` are successful
   outcomes.** A cycle that correctly changes nothing has done its job.
7. **Falsifiability at authoring time**, and the contract is immutable once
   registered; revision means a new version.
8. **Budgets are hard.** Tokens, validation compute, operator attention and live
   blast radius are capped; exceeding a cap halts the loop.
9. **Retrieved content is data, never instruction.**

---

## 2. Architecture — planes and the flow between them

```
┌─ CONTROL PLANE ────────────────────────────────────────────────────────────┐
│ Truth Harness (TH-01..12 + agent invariants) · policy engine · budgets      │
│ durable orchestrator (state machine, leases, retries, dead-letter)          │
│ liveness watchdog · circuit breaker · audit log                             │
└───────────────┬────────────────────────────────────────────────────────────┘
                │ every transition below crosses a hook
┌─ EVIDENCE ────────────────┐         ┌─ AGENT PLANE (LLM) ──────────────────┐
│ event store (hash-chained)│  MCP    │ Analyst    → incidents               │
│ point-in-time feature/    │ read-   │ Author     → hypothesis contract     │
│   label store             │ only    │ Adversary  → kill-before-compute     │
│ ObjectiveContract (v)     │◄───────►│ Referee    → blind verdict (advisory)│
│ capability registry (AST) │         │ Planner    → what to spend on        │
│ RAG (time-aware, untrusted│         │ Historian  → research ledger only    │
│   content)                │         └──────────────┬───────────────────────┘
└───────────────┬───────────┘                        │ PromotionRequest
                │                     ┌──────────────▼───────────────────────┐
                │                     │ VALIDATION (no LLM, own identity)    │
                │                     │ frozen snapshot · sealed holdout     │
                └────────────────────►│ purge/embargo · bounded Strategy DSL │
                                      │ placebo runs · multiple-testing ledger│
                                      └──────────────┬───────────────────────┘
                                                     │ signed ResultBundle
                                      ┌──────────────▼───────────────────────┐
                                      │ STATISTICAL AUDITOR (no LLM)         │
                                      │ independent recompute · CI · guards  │
                                      └──────────────┬───────────────────────┘
                                                     │
                                      ┌──────────────▼───────────────────────┐
                                      │ PROMOTION GOVERNOR (deterministic)   │
                                      │ + SignedApproval (operator)          │
                                      │ shadow → canary → flagged live       │
                                      └──────────────┬───────────────────────┘
                                                     │ outcome
                                      ┌──────────────▼───────────────────────┐
                                      │ MEMORY: research ledger · negatives · │
                                      │ experiment registry · agent trace     │
                                      └──────────────────────────────────────┘
```

Information is asymmetric on purpose: the Referee never sees the persuasive
rationale, the validator never sees it either, and the Author never sees the
sealed holdout.

---

## 3. Evidence plane

### 3.1 Provenance — hashes, not offsets

`files/event_store.py` (shipped) syncs the JSONL journal into SQLite from the
last byte offset. **Byte offsets are for incremental reading only.** They do not
prove immutability: a file can be rewritten, rotated or restored with different
bytes at the same offsets — and this repo does rewrite its datasets, which is
exactly how 1.09 GB of orphaned `.tmp` files accumulated.

A snapshot manifest therefore carries:

| field | why |
|---|---|
| `prefix_hash` (BLAKE3 of bytes 0..offset), maintained as a **rolling hash chain** per append batch | proves the prefix never changed; cheap because the journal is append-only |
| `schema_hash` | field meanings changed silently once already (`trend_chop` vs `trend/1h chop:`) |
| `universe_hash` | the watchlist defines the denominator |
| `code_commit`, `config_hash` | the same data under different gates is a different experiment |
| `label_maturity_cutoff` | rows whose T+10 label is not yet real must not enter |

Two results are comparable only when their manifests match on data, schema,
universe and objective version.

### 3.2 Capability registry

Derived mechanically from source: every `config.py` constant with type, range,
reader module and `do_not_touch` status; every `BlockRule`, entry mode, exit
policy, reward term, bandit feature; every canonical metric and its script. A
hypothesis whose target does not resolve here is **unrepresentable**, which is
what makes the 16 dead hypotheses impossible rather than merely rejected.

### 3.3 Point-in-time feature/label store

Every row carries `available_at` (when the bot could have known it) and
`label_mature_at` (when the outcome became real). Retrieval and validation
filter on both. Without this, a "temporal" split still leaks through labels that
matured after the cut.

### 3.4 RAG, time-aware and untrusted

Hybrid BM25 + embeddings over specs, reports, decisions, negative register.
Filtered by `policy_epoch`, `available_at`, `label_mature_at`, `universe_hash`,
action layer and evidence-expiry status; stale evidence is labelled stale in
context (TH-10 currently flags 77-day-old gate evidence, and an agent will cite
it confidently otherwise).

Retrieved content is **data, not instruction**. Numeric tables and candles never
enter the vector index; a deterministic SQL layer aggregates them and the agent
receives values with source id, hash, cutoff and coverage.

### 3.4b `policy_epoch` — the identity of decision behaviour

Used as a retrieval and comparability filter throughout this document, and left
undefined in v2. It is the **semantic identity of production decision
behaviour**, not the identity of every commit.

A new epoch begins when a change can alter candidate eligibility, scoring,
routing, gate order or action selection; entry/exit/re-entry behaviour;
decision-time feature or label semantics used by a live model; active-universe
eligibility or capacity; **or an effective runtime override that changes any of
those** — which is why the override channel in §6.1 is an epoch-level concern,
not a footnote.

Documentation, tests, logging-only fields, performance refactors and repairs
with demonstrated decision-trace equivalence do **not** open a new epoch; their
code and config hashes still enter the manifest.

An epoch transition does not delete prior evidence. Each earlier result becomes
`directly_comparable`, `transportable_with_bridge` (requires a registered
overlap analysis on the same candidate population) or `historical_only`.
**Market regime is recorded separately** — a regime change does not rewrite
policy identity, and vice versa.

### 3.5 ObjectiveContract — versioned, human-only

The single place the goal is defined; the agent can read it and propose a change
through a separate human path, never edit it.

```
ObjectiveContract v1
  mission_kpi        early_capture over canonical mature eligible top movers
  target             >= 0.25 (floor), 0.40 (goal)
  provenance         later-EOD immutable labels          # phase 0 dependency
  guardrails (non-inferiority margins, all must hold)
      alert_precision        not worse than baseline by > 3 pp
      alerts_per_day         not more than baseline × 1.25
      fast_reversal_rate     not worse by > 2 pp
      silent_miss_rate       not worse
      per-gate harm          no gate's winners-lost increases
  data requirements  full numerator/denominator, coverage, downtime,
                     label maturity, universe snapshot
  incomplete days    UNKNOWN — never success, never miss
```

**On the economic gate.** The review asks for canonical portfolio alpha after
fees and slippage as a promotion gate, and the harness does report TH-11 FAIL
for its absence. The concern is right — early capture must not be bought by
alerting on more junk — but the specific metric does not fit this product: this
bot has **no position sizing**; it emits alerts. So the guardrails above are the
product-appropriate expression of the same requirement, and whether to build a
simulated-portfolio economic gate at all is left as an explicit operator
question (§14), not silently skipped.

### 3.6 Negative-results register

Every refuted hypothesis with the numbers that killed it, including the nine in
this repo and four inherited from the sibling bot. Similarity search produces a
**warning that demands a statement of what new evidence justifies the retry** —
not an automatic ban. A regime-conditional variant of a refuted idea is a
legitimately new hypothesis, and v1's automatic dedup would have blocked it.

---

## 4. Agent plane

### 4.1 Analyst → incidents
Ranks concrete historical cases by opportunity cost; emits structured incidents
with case ids. Forbidden from proposing remedies, so the diagnosis is not
written backwards from a favoured fix.

### 4.2 Author → hypothesis contract
Consumes incidents, RAG and the capability registry; emits an **immutable,
versioned** contract: intent (metric from the ObjectiveContract, direction,
minimum practically significant effect), mechanism (target resolved in the
registry, change), affected layer, allowed decision-time features, candidate
strategy **within a bounded Strategy DSL** (never arbitrary code), competing
explanation, falsifier, frozen baseline, guardrails, population and power
requirement, split plan with purge/embargo, robustness/regime slices, cost
assumptions, shadow/canary plan, rollback.

Once registered it cannot be edited; revision creates a child version with a
parent link. Self-consistency sampling is dropped — it costs budget and
agreement among samples of one model is not evidence.

### 4.3 Adversary → kill before compute
Attacks with a fixed rubric from §0a: leaky target, population the bot never
samples, base-rate illusion, tautological causal story, retry without new
evidence.

**Measured by false-kill rate — but not the way v2 proposed it.** Estimating it
by letting a random 10% of killed hypotheses through assumed a stream of
experiments. At ≤12 decision-grade versions a year that is roughly **one control
per year**, from which no rate can be estimated. Corrected:

- **Primary: a fixed historical calibration corpus** — past hypotheses whose
  outcomes are known, including the nine refuted here and the four inherited.
  Scored offline, so it costs no live capacity and has an n worth quoting.
- **Secondary: a rare live pass-through**, once or twice a year, as a check that
  the corpus has not drifted from the live population — never the estimator.

A 0% false-kill rate on the corpus still means the adversary is too permissive.

### 4.4 Referee → blind, advisory
Sees the signed result bundle, the **pre-registered** metric, acceptance
criteria, guardrails and scope — but not the persuasive rationale, not the
author's predicted effect, and with candidate/baseline randomly presented as
A/B. Verdict is `PASS | FAIL | INCONCLUSIVE` with evidence ids.

**Advisory only.** No judge verdict can overturn a deterministic failure from
the auditor or the harness; the Governor decides. Judges are calibrated against
an expert-labelled set and their calibration score is printed beside every
verdict.

Known limitation, stated rather than hidden: Author, Adversary and Referee share
a model class, so their errors correlate. Mitigation is measurement — judge
agreement on the calibration set is itself reported — and, when budget allows, a
different model for the Referee.

### 4.5 Planner → allocation
Ranks by **expected impact × information gain ÷ cost**, with a fixed exploration
quota. Thompson sampling is deferred: with few experiments, delayed outcomes and
shifting policy epochs it would learn which families the validator likes.

### 4.6 Historian → research ledger only
Writes to `ResearchExperimentLedger` (append-only) and the negative register.
**It has no write path to `decisions.jsonl`, runtime overrides or config.**

---

## 5. Validation plane — the integrity boundary

No LLM. Separate process, own service identity, network disabled, read-only
frozen snapshot.

### 5.1 Registry and admission
Each `mechanism.kind` maps to registered validators. No validator ⇒ no
registration. Hypotheses execute only within the bounded Strategy DSL; arbitrary
Python from a hypothesis is never run.

### 5.2 Method requirements
Maximum available period · chronological walk-forward · **purge and embargo**
around split boundaries sized to label maturity · **sealed final holdout** the
Author cannot read · day/regime **block bootstrap** for CIs · pre-declared regime
slices · sensitivity to costs where costs apply · **placebo / negative-control
(A/A) runs** proving the validator reports no effect when there is none ·
**multiple-testing ledger** with alpha spending across all registered
experiments.

### 5.2b Power expansion — what happens when the answer is "not enough data"

At ~20 winners a week, purge/embargo, holdout reservation, regime slices and a
multiple-testing ledger together make `UNDERPOWERED` the most likely outcome of
any given cycle. Declaring that a valid verdict is honest; leaving it as the
loop's steady state is not. When a primary population is infeasible, the
orchestrator registers exactly **one** pre-declared expansion action before any
further hypothesis version may consume validation budget:

1. extend the calendar window while preserving `policy_epoch` comparability;
2. pool exchangeable symbols with a day-clustered partial-pooling model instead
   of pretending symbol-days are independent;
3. replace a sparse binary response with a continuous one — remaining return,
   captured fraction — keeping the canonical binary objective as a guardrail;
4. lower the event threshold, but only after registering and measuring its
   transfer relationship to the canonical objective;
5. widen the real candidate population, or pick a mechanism with a larger
   eligible one;
6. repair missing observation/outcome logging where the limit is data loss
   rather than market rarity;
7. terminate as `ACCEPTED_UNKNOWN` when none of the above preserves the causal
   question.

The choice is made **from the pre-result power report**, never after inspecting
a favourable slice. Changing the outcome, threshold, population or pooling model
creates a new hypothesis version and cannot retroactively rescue the old result.

### 5.2c Holdout that does not eat the newest data

A permanently sealed holdout is wrong at this event rate: it consumes exactly
the most recent and most relevant evidence, forever. Instead:

- **one-use forward cohorts** are the final evidence for a hypothesis version;
  once revealed they become historical prior, and retuning requires a new
  version and a new future cohort;
- historical evaluation uses **rolling-origin walk-forward with cross-fitting**,
  so no window is permanently withheld;
- purge/embargo is applied **only where feature/label overlap requires it**,
  sized to label maturity, rather than as a blanket cost.

### 5.3 Signed result bundle
Manifest (§3.1), baseline and candidate metrics, paired deltas with CIs,
denominator, coverage, regime stability, guardrail outcomes, artifacts, full
error status, seeds, and a verdict of
`supported | refuted | underpowered | invalid`.

### 5.4 Independent recompute — from raw, not from artifacts

Recomputing "from the artifacts" is not independence: if the validator dropped
rows or built the eligible population wrongly, its artifacts inherit the error
and the recompute cheerfully confirms it.

The auditor therefore:

- reads the **orchestrator-frozen raw snapshot**, not the validator's outputs;
- builds eligibility and the denominator with its **own** implementation, from
  the registered metric contract — a second implementation is the point;
- reconstructs the candidate decision trace from the registered policy, then
  recomputes the primary metric, guardrails, interval and manifest hashes;
- **must not import the validator's aggregation code**, or the two share a bug;
- treats the validator's own trace and metrics as *comparison evidence*, never
  as input.

A mismatch, a missing raw input, or an inability to reproduce yields `invalid`,
records both payloads, and cannot be overridden by any judge, the operator
canary or the governor.

### 5.5 Preconditions
Freshness SLO green (`TH05_ARTIFACT_FRESHNESS`, shipped), manifest resolvable,
clean commit, labels mature. A loop that trains on a stalled input produces
confident nonsense — this repo lost 58 days of labels to exactly that.

---

## 6. Promotion plane

### 6.1 Four stores, one execution channel

The v1 defect, fixed structurally:

| Store | Writer | Readable by agent | Executable |
|---|---|---|---|
| `ResearchExperimentLedger` | Historian (LLM) | yes | **no** |
| `PromotionRequest` | Governor, from a signed bundle | yes | no |
| `SignedApproval` | Governor + operator key | yes | yes — the only one |
| `RuntimeOverrideStore` | release process, from SignedApproval only | **no** | yes |

Production reads only signed release records. Today `decisions.jsonl` is both
memory and execution channel; splitting it is a prerequisite, not a later
refinement.

### 6.2 Stages

`SHADOW` (logged, not applied) → `CANARY` → `LIVE behind a flag`.

**Canary — and why a symbol subset is not a valid experiment here.** A random
symbol split looks natural but violates independence: the bot shares position
slots, rotation, cluster and correlation caps, cooldowns and one alert budget
across all symbols. A treatment symbol taking a slot changes what the control
symbols could have done, so the "control" is not a control.

The staged answer:

1. **Shadow twin** — candidate and baseline both evaluate every symbol with
   fully independent state (positions, cooldowns, caps, budget). No interference
   because nothing is shared. This is the main evidence stage.
2. **Time-switchback canary** — if a live stage is still wanted, randomise over
   pre-declared time blocks rather than symbols, so the shared resources belong
   to one policy at a time.
3. **Operator-only channel** — candidate messages go to a separate topic or a
   tagged digest, never duplicating the production alert. For a single-operator
   product this may be the final alert-quality gate, with acceptance criteria
   pre-registered before the output is seen, and the operator's decision
   recorded against those criteria so post-hoc rationalisation is visible.

Stop conditions at every stage: guardrail breach, alert-rate excursion, data
integrity violation, or any harness blocking finding → the **candidate flag**
goes off and the frozen baseline is restored. The monitor never stops the
baseline because a candidate failed.

**Operator approval is a permanent rule for behaviour-affecting changes**, not a
toggle. Bounded auto-promotion, if ever enabled, is limited to a pre-approved
envelope: risk-reducing rollbacks and disabling a degrading feature.

### 6.2b Rollback triggers — pre-registered, not declarative

Every promotion manifest fixes these **before** the candidate goes live, and the
monitor's only action is `candidate_flag = OFF` followed by restoration of the
frozen baseline. It never stops the baseline because a candidate failed.

| Class | Trigger | Latency |
|---|---|---|
| Data integrity | candle/feature age past the timeframe limit, consecutive scan failures, active-universe coverage below floor, snapshot/policy provenance mismatch | immediate |
| Message safety | any duplicate production alert on the same dedup key; unique alerts/day above the hard cap; candidate message rate above baseline × 1.25; Telegram delivery failures above threshold | immediate |
| Quality | guardrail non-inferiority breached on the pre-registered sequential rule, over a pre-declared window and a minimum mature sample | only **after** the maturity rule — never on an early adverse streak |

A quality rollback that fires before its maturity rule is a false alarm by
construction, which is why the three classes have different latencies.

### 6.3 Attribution
Realised effect with day-clustered bootstrap CI and market-drift normalisation,
written back to the Planner's priors and the negative register.

---

## 7. Meta-evaluation

1. **Agent backtest with a frozen world.** Freeze at date T not only the data
   but the MD corpus, prompts, skills, capability registry, tool schemas and
   model snapshot; run the loop; compare proposals against what actually
   happened after T. Freezing data alone leaks — v1's version did.
2. **Judge calibration** on a labelled set, reported beside every verdict.
3. **Loop utility**, measured as: avoided harm (correct rejections of changes
   that later proved harmful), research precision (share of promoted changes
   surviving attribution), attributed improvement, and **share of correct
   `NO_CHANGE`**. Not decisions per month, which rewards churn.
4. **Mandatory baseline comparison.** The agent is scored against a
   deterministic opportunity-priority baseline — rank incidents by measured
   opportunity cost, propose the top one — on proposal validity, supported-
   hypothesis precision, harmful validations avoided, cost and latency. An agent
   that does not beat that baseline stays a summariser and **may not select the
   primary hypothesis**. Where the operator's historical proposals are not
   recoverable in comparable form, that is recorded as a limitation and the
   deterministic baseline stands alone; the comparison is not quietly skipped.
5. **Outcome SLO, not just structure.** Within **30 days of Phase 1 completing,
   at least one real admitted hypothesis must reach a terminal validator
   result.** Otherwise the implementation is a liveness failure *even if every
   structural test passes*, and the mandated response is to simplify the loop or
   return to manual research — not to extend the deadline.

---

## 8. Interfaces, hooks, skills

**MCP is the interface; the boundary is server-side ACL plus capability
tokens.** Read tools for evidence; `experiments.register`, `backtest.submit`,
`backtest.status`, `backtest.get_signed_result` for the loop. Snapshot freezing
belongs to the orchestrator, not the agent. `harness.verify` is rate-limited and
every call is ledgered, or an agent will verify until it passes.

Hooks: `before_hypothesis_register` (objective alignment, provenance, similarity
warning) · `before_backtest_submit` (frozen snapshot, leakage scan, purge sizing,
power requirement) · `after_backtest_complete` (independent recompute, artifact
hashing) · `before_judge` (harness, A/B masking) · `before_shadow` (rollback,
guardrails, sample requirement) · `before_canary` (full harness, judge quorum,
operator approval) · `after_promotion` (drift monitor, automatic rollback) ·
`pre_commit` (existing).

Skills: `crypto-bot-truth-harness` and `md-compliance` (exist),
`objective-metric-auditor`, `causal-hypothesis-author`,
`experiment-spec-designer`, `time-series-backtest-reviewer`,
`promotion-evidence-reviewer`.

---

## 9. Liveness — so the new loop cannot die quietly

The strongest evidence for this section is this repo's own history: a bot dead
8 days unnoticed, a scheduler silent 11 days on battery, a backfill lock stale
58 days, a pipeline with no decision for two months. A loop without liveness
guarantees will fail identically and be discovered by accident.

Durable state machine with persisted transitions · idempotent steps keyed by
hypothesis version · leases with expiry so a crashed worker's work is reclaimed
(the stale-lock lesson, generalised) · bounded retries with a dead-letter queue ·
reconciliation on start.

**Operational contract — concrete, because a placeholder is not a contract.**

| Item | Value |
|---|---|
| Accountable owner | `repository maintainer` — one person. Finding categories route a queue; they do not imply staffed teams |
| Finding states | `OPEN → ACKNOWLEDGED → REPAIRING → VERIFIED`, or `ACCEPTED_DEBT(review_at=…)`, or `SUPERSEDED` |
| SLO | on **acknowledgement at the next weekly triage**, not on repair time. Repair dates are estimates recorded at triage |
| `ACCEPTED_DEBT` | an honest triage outcome, **not a waiver** — the dependent claim stays blocked until repair or a separately approved, expiring waiver |
| Watchdog | alarm when **no state transition in 10 days**, or no evidence pack at the weekly slot |
| Capacity | ≤1 primary validation admission/week; ≤3 simultaneous `FORWARD_WAITING` versions, each with a fixed wake-up condition; **≤12 decision-grade forward versions/year** until measured throughput says otherwise |
| Forward maturity | 2–4 weeks per hypothesis version — this, not the report cadence, is the programme's clock |

**The weekly report must never imply weekly self-improvement.** At ~12 forward
versions a year, throughput is the binding constraint, and it is raised by
removing operational stalls and choosing powered populations (§5.2b) — never by
weakening maturity rules.

---

## 10. Agent invariants added to the Truth Harness

The agent cannot modify the ObjectiveContract · the Author never sees the sealed
holdout · validation contains no LLM and runs no arbitrary code · every
hypothesis and every run is registered, including failures · the evaluation
window cannot be chosen after seeing a result · a retry cannot silently change
seed, population or parameters · a judge cannot overturn a deterministic fail ·
missing / stale / partial always yields `UNKNOWN` · promotion without rollback
and forward evidence is impossible · retrieved content is untrusted data ·
`NO_CHANGE` is a successful terminal state.

---

## 11. Statistics: what this data can actually support

### 11.1 The North Star cannot answer a weekly question

Measured here (60 days, 43 full days, watchlist-filtered `label_top20`):

```
per-winner score:  n = 125   mean = 0.0612   sd = 0.1430
                   63% of winners score exactly 0        CV = 2.34
```

Minimum detectable difference between two weeks, 80% power, α = 0.05:

| unit | n / week | MDE | vs the metric's own level |
|---|---:|---:|---|
| top-20 winners (**current NS**) | ~20 | 0.127 | **2.1×** |
| move events ≥ +5% | ~83 | 0.062 | 1.0× |
| two pooled weeks | ~160 | 0.045 | 0.7× |

Three independent disqualifiers: power (a product of three factors, one a
mostly-zero Bernoulli, is the worst shape for a small sample); unreadable
direction (coverage up and capture down cancel); and provenance that moves with
the confound (labels come from the same rolling-24h snapshot as the features,
and which snapshots exist depends on uptime).

**These MDE figures assume independent coin-days and are therefore optimistic.**
Moves within a day share market beta, so the effective sample is smaller than n
suggests. All CIs are computed with a **day-clustered block bootstrap**, and the
table above is an upper bound on achievable power, not an estimate of it.

### 11.2 One `MoveEvent`, two metrics

v1 defined coverage and precision on different anchors and horizons, so they
were not two sides of one event. Corrected — a single event definition:

```
MoveEvent  v2
  universe_snapshot   watchlist hash
  day                 UTC day, strictly            # not Europe/Budapest
  qualifies_if        max(high) over the UTC day >= open × (1 + THRESHOLD)
  threshold           +5%          # proxy for weekly steering
  opportunity_window  [UTC open, early_deadline]
  early_deadline      first crossing of open × (1 + EARLY_FRACTION)
  early_fraction      +2.5%        # FIXED, not derived from the move's size
  dedup               one event per (symbol, UTC day); alerts inside the window
                      collapse to the earliest
  label_mature_at     UTC day close + settlement
```

**v1 defined this wrongly and the error inverted the window.** The midpoint was
"half the move's amplitude", so a +6% move had its midpoint at +3% — *before*
the +5% anchor that was supposed to open the window. Coverage would then have
been measured against a deadline that preceded the event's own start. The
deadline is now a **fixed** +2.5% crossing, independent of how large the move
turned out to be, which also removes hindsight from the deadline itself.

The day is UTC. `Europe/Budapest` introduces 23- and 25-hour days at DST
boundaries and disagrees with the exchange day — a silent denominator defect in
exactly the metric built to be trustworthy.

| metric | definition on `MoveEvent v2` | n / week |
|---|---|---:|
| `Coverage@move` | of qualifying MoveEvents, the share with a first eligible alert at or before `early_deadline` | ~83 |
| `Precision@alert` | of unique symbol-day alerts emitted before `early_deadline` or the day's cutoff, the share whose symbol-day later qualifies | ~143 |

Both now use **the same pair of boundaries** on the same event version, which is
what makes them a genuine precision/recall pair rather than two similar-sounding
rates.

Recall and precision on the same event, so trading one for the other is visible.
Ground truth comes from **exchange klines**, not `top_gainer_dataset`, so the
weekly metric does not inherit the provisional label's bias — and that work is
also step one of retiring `provisional` on the North Star.

`move5` is an explicitly labelled **proxy for weekly steering**. It never
replaces the top-20 mission metric, which is reported alongside it, unpowered
and marked so.

**The surrogate must be validated, not assumed.** The chain that would justify
steering on it has three links and only the first is demonstrated:

```
better forward_top10_min3 label   → (shown: holdout lift 0.65× → 4.07×)
  → better coverage/earliness on mature EOD top-20   → NOT shown
    → better alerts actually sent                    → NOT shown
```

The weekly report publishes the measured relationship between `move5` and
canonical top-20 outcomes. If that relationship is weak or unstable, `move5`
stays diagnostic and **cannot produce a positive overall progress verdict**.

### 11.3 The inference rule

- Every number carries `n`, base rate, day-clustered CI, and the power the
  design had.
- **Decisions use the observed CI against a pre-registered practical-significance
  threshold and non-inferiority margin** — not MDE, which is a design parameter
  and cannot serve as a threshold on observed data (v1 error).
- Trend across weeks uses a **confidence sequence / CUSUM with a yearly
  false-alarm budget**, not 52 independent tests, which manufacture ~2.6 false
  trends a year at α = 0.05.
- Windows are uptime-matched, or there is no comparison.
- Where the interval spans the threshold: `UNDERPOWERED`, printed as such.

### 11.4 The weekly report

1. **Did anything move?** The `MoveEvent` pair with CIs and the honest verdict —
   usually "within noise", which is information.
2. **What did we do?** Deterministic counters: gate harm, silent misses, alert
   and entry rates, freshness lapses, harness findings, and every promotion with
   its attribution state.
3. **What next?** ≤3 ranked proposals, each with falsifier and validator.

Quarterly adds the North Star with its CI and states whether it is still
provisional.

---

## 12. Operating point

Weekly cadence, minimal budget. The report is **deterministic**; the LLM only
interprets a finished evidence pack — a prepared table, not a database — and
emits at most a reading, ≤3 hypotheses and a recommended next step. The weekly
report therefore exists and is trustworthy **before** any agent is built, which
is also the correct order given §0.

---

## 13. Rollout

| Phase | Builds | Exit gate |
|---|---|---|
| **-1** | **Walking skeleton — named, not abstract.** `files/improvement_fixture_validator.py` (`FixtureDeltaValidatorAdapter`): stdlib only, no network, no production state, **must not import `replay_backtest.py`, `monitor.py` or any trading module**, computes one fixed baseline/candidate delta over ≤64 rows of the checked-in fixture `files/testdata/control_plane_smoke_fixture.json`, and carries a test-only corruption mode. Plus a minimal snapshot/contract/attempt record, the §5.4 verifier, and one terminal state in the ledger. LLM, RAG, promotion and the registries are stubbed | `pyembed\python.exe files
un_control_plane_smoke.py` carries a fresh attempt from `OBSERVED` to a verified terminal result **in under ten seconds**; it is repeatable and restart-safe; it cannot reach any release store; and the corrupted bundle lands in `INVALID_RESULT` instead of at the governor. **No registry or label work starts until this passes** — it proves the pipe conducts an experiment, and it is explicitly not market evidence |
| **0a** | Repair measurement: immutable later-EOD labels, day-grouped splits, provenance fields, product-scoped harness profiles | zero blocking findings **in the `discovery/alert` and `exit` profiles**; `execution/portfolio` carries a signed waiver. A globally green harness is not a reachable gate — TH-11 demands portfolio alpha this product does not have, so requiring it builds a permanent freeze |
| **0b** | ObjectiveContract, capability registry, contracts, four-store split, **and migration of existing negative results** — the casebook, rejected hypotheses, decision records and the 47 verdict-less backtests, each tagged `CONFIRMED_NEGATIVE` / `LEGACY_UNVERIFIED` / `DUPLICATE` / `MIGRATION_ERROR`. Migration never invents a missing denominator, and an unverified item raises a similarity warning rather than a rejection. **The first real cycle is blocked until this inventory is complete**; the Phase -1 skeleton is not | The 16 dead hypotheses are rejected mechanically; no LLM path reaches a runtime override |
| 1 | Deterministic weekly report on `MoveEvent` | Two consecutive weeks published with CI, verdicts and no unsupported arrow |
| 2 | Validation service: snapshot manifests, purge/embargo, sealed holdout, placebo, ledger | Reproduces three historical decisions within CI; A/A returns no effect |
| 3 | Analyst + Author, propose-only, via MCP | ≥10 contract-valid hypotheses; ≥3 judged worth testing by the operator |
| 4 | Adversary + Planner + Referee + auditor | Adversary false-kill rate < 15% on the calibration set |
| 5 | Shadow twin, then time-switchback canary, operator-approved | One change completes shadow → canary → live and survives attribution; agent backtest published |

Phases 0a–2 contain no autonomous action and no LLM decisions. The previous
version of this component failed at phase 0 and nobody noticed for two months.

---

## 14. Open questions for the operator

1. **Economic gate.** This bot has no position sizing, so canonical portfolio
   alpha is not a natural gate; the guardrails in §3.5 are the product-appropriate
   substitute. But TH-11 will keep reporting FAIL until *some* profitability
   evidence exists. Build a simulated 10-slot portfolio purely as a guardrail, or
   formally record that this product is judged on detection and accept the
   finding as a known, documented exception?
2. **Autonomy ceiling.** Operator approval is permanent in this design. The only
   candidate for bounded automation is risk-reducing rollback. Agreed?
3. **Move threshold.** `+5%` buys ~83 events/week against ~20 for top-20, at the
   cost of measuring something broader than the mission. `+8%` gives ~33/week —
   too weak to steer on. Accept +5% as the labelled proxy?
