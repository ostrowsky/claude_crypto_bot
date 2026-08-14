# Control-plane walking skeleton (Phase −1)

- **Slug:** `control-plane-walking-skeleton`
- **Status:** spec → implementation
- **Created:** 2026-08-14
- **Parent:** [`continuous-improvement-agent`](continuous-improvement-agent-spec.md) Phase −1
- **Truth-harness invariants:** TH-03 (holdout discipline is *not* claimed here),
  TH-05 (unknown stays unknown), TH-12 (traceable to evidence)
- **Rollback:** delete the six new files; nothing existing changes behaviour

## Problem

The improvement loop that exists in this repository has produced **no decision
through the pipeline since 2026-06-17**, while accumulating seven layers, a
memory, a notifier and a scheduler. Its sibling project shows the same disease
at larger scale: 147 specs and a research package imported by no production
module.

Both failed at the same place — not at "promoting a bad change", but at
**never completing one experiment end to end**. So the first thing built here is
not a registry, a label store or an agent. It is the smallest vertical slice
that carries one hypothesis from `OBSERVED` to a *verified* terminal state, and
refuses a corrupted one.

This slice is deliberately **not market evidence** and cannot support any
trading conclusion. Its only claim is: the pipe conducts an experiment.

## Scope

In:

- an immutable snapshot manifest with content hashes, created by the
  orchestrator;
- an immutable, versioned hypothesis contract;
- an append-only attempt ledger, idempotent by attempt id;
- a deterministic fixture validator with a test-only corruption mode;
- an independent result verifier that recomputes from **raw**;
- a governor stub that consumes only a verified result;
- one runner command.

Out: LLM, RAG, registries, promotion, shadow, canary, live data, any trading
module.

## Architecture conformance

The slice exists to make these architectural claims executable rather than
aspirational. Each maps to a test.

| Invariant (from the parent spec) | How this slice enforces it |
|---|---|
| The orchestrator owns snapshots; the agent cannot freeze evidence | Only the runner creates a manifest; the validator receives one and cannot mint one |
| The validator is isolated from production | `improvement_fixture_validator` is stdlib-only and **must not import** `replay_backtest`, `monitor`, `strategy`, `config` or any trading module |
| Independent recompute is from raw, not artifacts | `control_plane_verifier` re-reads the raw fixture, builds eligibility and the denominator with its own implementation, and **must not import the validator** |
| A corrupted result never reaches the governor | Corruption mode flips the validator's claimed metric; the verifier must return `INVALID_RESULT` and the governor must refuse it |
| Every attempt reaches a terminal or waiting state | The ledger's last event for the attempt carries `status=TERMINAL` and an `outcome_reason` |
| Ratios carry base rate and lift (TH-01) | The result bundle is rejected by contract if `base_rate` or `lift` is missing |
| Research memory is not an execution channel | Nothing in the slice writes to `decisions.jsonl`, `config.py` or any runtime override |

## Contracts

```
SnapshotManifest       snapshot_id · raw_path · raw_sha256 · row_count ·
                       schema_hash · created_at_utc · created_by
HypothesisContract     hypothesis_id · version · snapshot_id · policy_baseline ·
                       policy_candidate · primary_metric · guardrail ·
                       minimum_practical_effect · falsifier · frozen=True
ResultBundle           attempt_id · hypothesis_id · snapshot_id ·
                       baseline{n, entered, hits, precision, base_rate} ·
                       candidate{…} · delta · lift · guardrail_ok ·
                       validator_build · claimed_verdict
VerifiedResult         result + verifier_recompute + agreement=True
LedgerEvent            attempt_id · seq · stage · status · outcome_reason ·
                       payload_hash · at_utc
```

`stage ∈ {OBSERVED, PREPARED, REGISTERED, VALIDATING, VERIFYING, DECIDED,
CLOSED}`, `status ∈ {ACTIVE, WAITING, TERMINAL}`, and `outcome_reason` is a
**closed** vocabulary — extending it is a spec change, not a code change, so the
ledger stays queryable.

## The fixture experiment

64 rows shaped like decision candidates: `{row_id, day, symbol, score,
forward_move_pct}`. Baseline policy admits `score >= 0.50`, candidate admits
`score >= 0.60`. Primary metric is **precision** — the share of admitted rows
whose `forward_move_pct >= 3.0` — reported with its base rate and lift, because
a precision without them is exactly the failure §0a rule 1 exists to prevent.
Guardrail: the candidate's admission rate must not exceed baseline × 1.25.

The fixture is constructed so the candidate wins on precision while passing the
guardrail. That is a property of a hand-made file, not a market finding.

## Verification

`pyembed\python.exe files\run_control_plane_smoke.py`

Exit gates, all asserted by `test_control_plane_skeleton.py`:

1. a fresh attempt reaches `VERIFIED`/`TERMINAL` in **under 10 seconds**;
2. re-running with the same attempt id is idempotent — the ledger does not grow;
3. `--corrupt` yields `INVALID_RESULT`, terminal, and the governor is never
   reached;
4. the validator module imports no trading module (AST check);
5. the verifier does not import the validator (AST check);
6. a result missing `base_rate` or `lift` is rejected by contract;
7. no file outside `.runtime/control_plane/` is written.

## Findings from the architecture review

Reviewing the slice against its own conformance table found two gaps that the
happy path did not.

**An unhandled exception left the attempt `ACTIVE` forever.** If freezing the
snapshot or the validator raised, the ledger held only in-progress events and no
terminal one — breaking the single invariant this slice exists to demonstrate.
Fixed: a crash is now a terminal outcome with a named reason
(`snapshot_invalid` / `contract_rejected`), and a test drives it by pointing the
runner at a missing fixture.

**Known limitation, deliberately left in.** `ResultBundle.with_claimed_precision`
constructs a bundle with a false headline. It exists so the corruption gate can
be exercised, but it means the frozen contract ships with a method for forging
itself. Acceptable in a skeleton whose only consumer is its own test; **not
acceptable once a real validator exists** — at that point corruption must be
injected from the test harness, not offered by the contract. Recorded here so
the next author does not discover it by surprise.

## Explicitly not claimed

No holdout, no purge/embargo, no power analysis, no market data, no statistical
validity of any kind. Phase 0 adds those. If this slice is ever cited as
evidence about the bot's performance, that citation is wrong.
