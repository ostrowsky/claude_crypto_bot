"""Independent recompute — from raw, never from the validator's artifacts.

Recomputing "from the result bundle" is not verification: if the validator
dropped rows or built the eligible population wrongly, its artifacts inherit the
error and a recompute over them cheerfully confirms it. So this module:

* re-reads the **raw** snapshot and re-verifies its content hash;
* builds eligibility and the denominator with its **own** implementation of the
  registered contract — a second implementation is the entire point;
* treats the validator's numbers as a claim to be checked, never as input;
* **does not import the validator**, or the two would share a bug.

A disagreement is `INVALID_RESULT`. It is never a rounding note, and no judge,
operator canary or governor may override it.

Spec: docs/specs/features/control-plane-walking-skeleton-spec.md
"""
from __future__ import annotations

from control_plane_contracts import (HypothesisContract, ResultBundle,
                                     SnapshotManifest, VerifiedResult)

TOLERANCE = 1e-9


def _recompute(rows: list[dict], hypothesis: HypothesisContract) -> dict:
    """Deliberately a different shape from the validator's implementation:
    one pass, counters only, no intermediate admitted lists."""
    hit_threshold = hypothesis.hit_threshold_pct
    total = 0
    positives = 0
    b_entered = b_hits = 0
    c_entered = c_hits = 0
    for row in rows:
        total += 1
        score = float(row["score"])
        is_hit = float(row["forward_move_pct"]) >= hit_threshold
        if is_hit:
            positives += 1
        if score >= hypothesis.policy_baseline:
            b_entered += 1
            b_hits += 1 if is_hit else 0
        if score >= hypothesis.policy_candidate:
            c_entered += 1
            c_hits += 1 if is_hit else 0

    base_rate = positives / total if total else 0.0
    b_prec = b_hits / b_entered if b_entered else 0.0
    c_prec = c_hits / c_entered if c_entered else 0.0
    return {
        "rows": total,
        "base_rate": base_rate,
        "baseline_precision": b_prec,
        "candidate_precision": c_prec,
        "baseline_entered": b_entered,
        "candidate_entered": c_entered,
        "delta": c_prec - b_prec,
        "lift": (c_prec / base_rate) if base_rate else 0.0,
        "admission_ratio": (c_entered / b_entered) if b_entered else float("inf"),
    }


def verify(manifest: SnapshotManifest, hypothesis: HypothesisContract,
           bundle: ResultBundle) -> VerifiedResult:
    if bundle.snapshot_id != manifest.snapshot_id:
        return VerifiedResult(False, {}, "result was produced against another snapshot")
    try:
        rows = manifest.load_rows()          # raises if the raw bytes changed
    except Exception as exc:
        return VerifiedResult(False, {}, f"raw snapshot unusable: {exc}")

    mine = _recompute(rows, hypothesis)

    checks = (
        ("candidate precision", mine["candidate_precision"], bundle.candidate.precision),
        ("baseline precision", mine["baseline_precision"], bundle.baseline.precision),
        ("base rate", mine["base_rate"], bundle.candidate.base_rate),
        ("delta", mine["delta"], bundle.delta),
        ("candidate entered", float(mine["candidate_entered"]), float(bundle.candidate.entered)),
        ("baseline entered", float(mine["baseline_entered"]), float(bundle.baseline.entered)),
    )
    for name, recomputed, claimed in checks:
        if abs(recomputed - claimed) > 1e-6:
            return VerifiedResult(
                False, mine,
                f"{name}: recomputed {recomputed:.6f} vs claimed {claimed:.6f}")

    guardrail = mine["admission_ratio"] <= hypothesis.guardrail_max_admission_ratio
    if guardrail != bundle.guardrail_ok:
        return VerifiedResult(False, mine, "guardrail outcome disagrees")

    expected = ("supported"
                if guardrail and mine["delta"] >= hypothesis.minimum_practical_effect
                else "refuted")
    if expected != bundle.claimed_verdict:
        return VerifiedResult(
            False, mine,
            f"verdict: recomputed {expected!r} vs claimed {bundle.claimed_verdict!r}")

    return VerifiedResult(True, mine)
