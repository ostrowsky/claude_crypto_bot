"""Deterministic fixture validator for the control-plane walking skeleton.

This adapter exists to prove that the control plane can carry one experiment
from a frozen snapshot to a signed result. It is **not a market validator**: it
reads a 64-row hand-made fixture and computes one fixed baseline/candidate
delta. Nothing it produces may be cited as evidence about the bot.

Isolation is the point and is enforced by tests:

* stdlib only — no numpy, no CatBoost, no aiohttp;
* imports no trading module (`monitor`, `strategy`, `config`, `replay_backtest`,
  …), so it cannot read or move production state;
* receives a `SnapshotManifest` and cannot mint one — whoever chooses the data
  chooses the answer, so freezing belongs to the orchestrator.

Spec: docs/specs/features/control-plane-walking-skeleton-spec.md
"""
from __future__ import annotations

from control_plane_contracts import (ArmResult, ContractError,
                                     HypothesisContract, ResultBundle,
                                     SnapshotManifest)

BUILD = "fixture-delta-v1"


class FixtureDeltaValidatorAdapter:
    """One policy threshold against another, on frozen rows."""

    build = BUILD

    def run(self, manifest: SnapshotManifest, hypothesis: HypothesisContract,
            *, attempt_id: str) -> ResultBundle:
        if hypothesis.snapshot_id != manifest.snapshot_id:
            raise ContractError("hypothesis was registered against a different snapshot")

        rows = manifest.load_rows()          # re-verifies the content hash
        total = len(rows)
        hit_threshold = hypothesis.hit_threshold_pct
        positives = sum(1 for r in rows if float(r["forward_move_pct"]) >= hit_threshold)
        base_rate = positives / total if total else 0.0

        def evaluate(threshold: float) -> ArmResult:
            admitted = [r for r in rows if float(r["score"]) >= threshold]
            hits = sum(1 for r in admitted
                       if float(r["forward_move_pct"]) >= hit_threshold)
            precision = hits / len(admitted) if admitted else 0.0
            # Ratio context travels with the ratio, always (TH-01). The bundle
            # contract refuses to be constructed without it.
            return ArmResult(
                n=total,
                entered=len(admitted),
                hits=hits,
                precision=round(precision, 6),
                base_rate=round(base_rate, 6),
                lift=round(precision / base_rate, 6) if base_rate else 0.0,
                admission_rate=round(len(admitted) / total, 6) if total else 0.0,
            )

        baseline = evaluate(hypothesis.policy_baseline)
        candidate = evaluate(hypothesis.policy_candidate)
        delta = round(candidate.precision - baseline.precision, 6)

        ratio = (candidate.entered / baseline.entered) if baseline.entered else float("inf")
        guardrail_ok = ratio <= hypothesis.guardrail_max_admission_ratio

        # The validator states a verdict; it does not get to be believed. The
        # verifier recomputes from raw and the governor consumes only the
        # verified result.
        if not guardrail_ok:
            verdict = "refuted"
        elif delta >= hypothesis.minimum_practical_effect:
            verdict = "supported"
        else:
            verdict = "refuted"

        return ResultBundle(
            attempt_id=attempt_id,
            hypothesis_id=hypothesis.hypothesis_id,
            snapshot_id=manifest.snapshot_id,
            baseline=baseline,
            candidate=candidate,
            delta=delta,
            guardrail_ok=guardrail_ok,
            validator_build=self.build,
            claimed_verdict=verdict,
        )
