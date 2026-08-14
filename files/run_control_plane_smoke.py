"""Walking skeleton: one attempt, OBSERVED -> verified TERMINAL, in seconds.

    pyembed\\python.exe files\\run_control_plane_smoke.py
    pyembed\\python.exe files\\run_control_plane_smoke.py --corrupt

The improvement loop this belongs to has never completed an experiment: seven
layers exist, and no decision has come through them since 2026-06-17. So the
first thing built is not a registry or an agent — it is proof that the pipe
conducts an experiment and rejects a corrupted one.

The orchestrator (this file) owns snapshot freezing, state transitions and the
ledger. The validator receives frozen evidence; the verifier recomputes from
raw; the governor sees only a verified result. Nothing here can write to
`config.py`, `decisions.jsonl` or any runtime override.

This slice is not market evidence and cannot support a trading conclusion.

Spec: docs/specs/features/control-plane-walking-skeleton-spec.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from control_plane_contracts import (AttemptLedger, HypothesisContract,  # noqa: E402
                                     LedgerEvent, SnapshotManifest, sha256_of)
from control_plane_verifier import verify  # noqa: E402
from improvement_fixture_validator import FixtureDeltaValidatorAdapter  # noqa: E402

FIXTURE = HERE / "testdata" / "control_plane_smoke_fixture.json"
DEFAULT_ROOT = HERE.parent / ".runtime" / "control_plane"


def _governor(verified, bundle) -> str:
    """Consumes only a verified result. Deterministic; no judge may override."""
    if not verified.agreement:
        raise AssertionError("governor reached with an unverified result")
    return bundle.claimed_verdict


def run(*, attempt_id: str = "smoke", root: Path | None = None,
        corrupt: bool = False) -> dict:
    root = Path(root) if root else DEFAULT_ROOT
    ledger = AttemptLedger(root)

    # Restart safety: an attempt that already reached a terminal state is not
    # replayed. Idempotence is what makes the ledger a history rather than a log.
    closed = ledger.is_closed(attempt_id)
    if closed:
        return {"attempt_id": attempt_id, "stage": closed["stage"],
                "status": closed["status"], "outcome_reason": closed["outcome_reason"],
                "verified": closed["outcome_reason"] != "invalid_result",
                "governor_reached": closed["outcome_reason"] in ("supported", "refuted"),
                "replayed": True}

    def record(stage: str, status: str, reason: str, payload) -> None:
        ledger.append(LedgerEvent(attempt_id=attempt_id, seq=ledger.next_seq(),
                                  stage=stage, status=status,
                                  outcome_reason=reason,
                                  payload_hash=sha256_of(payload)))

    record("OBSERVED", "ACTIVE", "in_progress", {"fixture": str(FIXTURE)})

    # An unhandled exception would leave the attempt ACTIVE forever, which
    # breaks the one invariant this slice exists to demonstrate: every attempt
    # reaches a terminal or waiting state. A crash is a terminal outcome with a
    # named reason, not an absence of one.
    try:
        # Only the orchestrator freezes evidence.
        manifest = SnapshotManifest.freeze(FIXTURE, created_by="orchestrator")
        record("PREPARED", "ACTIVE", "in_progress", manifest.snapshot_id)

        hypothesis = HypothesisContract.smoke(snapshot_id=manifest.snapshot_id)
        record("REGISTERED", "ACTIVE", "in_progress", hypothesis.hypothesis_id)

        record("VALIDATING", "ACTIVE", "in_progress", FixtureDeltaValidatorAdapter.build)
        bundle = FixtureDeltaValidatorAdapter().run(manifest, hypothesis,
                                                    attempt_id=attempt_id)
        if corrupt:
            # Test-only: keep everything, lie about the headline number.
            bundle = bundle.with_claimed_precision(candidate_precision=0.99)

        record("VERIFYING", "ACTIVE", "in_progress", bundle.validator_build)
        verified = verify(manifest, hypothesis, bundle)
    except Exception as exc:
        reason = ("snapshot_invalid" if isinstance(exc, (OSError, KeyError))
                  else "contract_rejected")
        record("CLOSED", "TERMINAL", reason,
               {"error": f"{type(exc).__name__}: {exc}"})
        return {"attempt_id": attempt_id, "stage": "CLOSED", "status": "TERMINAL",
                "outcome_reason": reason, "verified": False,
                "governor_reached": False,
                "disagreement": f"{type(exc).__name__}: {exc}",
                "recompute": {}, "replayed": False}

    governor_reached = False
    if not verified.agreement:
        record("CLOSED", "TERMINAL", "invalid_result",
               {"disagreement": verified.disagreement})
        outcome = "invalid_result"
    else:
        verdict = _governor(verified, bundle)
        governor_reached = True
        record("DECIDED", "ACTIVE", "in_progress", verdict)
        record("CLOSED", "TERMINAL", verdict, verified.recompute)
        outcome = verdict

    return {"attempt_id": attempt_id, "stage": "CLOSED", "status": "TERMINAL",
            "outcome_reason": outcome, "verified": verified.agreement,
            "governor_reached": governor_reached,
            "disagreement": verified.disagreement,
            "recompute": verified.recompute, "replayed": False}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="control-plane walking skeleton")
    ap.add_argument("--attempt-id", default="smoke")
    ap.add_argument("--corrupt", action="store_true",
                    help="inject a false claim; the verifier must reject it")
    ap.add_argument("--root", type=Path, default=None)
    args = ap.parse_args(argv)

    out = run(attempt_id=args.attempt_id, root=args.root, corrupt=args.corrupt)
    print("=" * 66)
    print(f"attempt {out['attempt_id']}  ->  {out['stage']}/{out['status']}"
          f"  reason={out['outcome_reason']}")
    print("=" * 66)
    if out.get("replayed"):
        print("  already terminal — replay is a no-op (restart-safe)")
    elif out["verified"]:
        r = out["recompute"]
        print(f"  verified independently from raw: {r['rows']} rows")
        print(f"  baseline precision {r['baseline_precision']:.3f} "
              f"({r['baseline_entered']} admitted)")
        print(f"  candidate precision {r['candidate_precision']:.3f} "
              f"({r['candidate_entered']} admitted)")
        print(f"  delta {r['delta']:+.3f}   base rate {r['base_rate']:.3f}   "
              f"lift {r['lift']:.2f}x")
        print(f"  governor reached: {out['governor_reached']}")
    else:
        print(f"  REJECTED by the verifier: {out['disagreement']}")
        print(f"  governor reached: {out['governor_reached']} (must be False)")
    print()
    print("  This slice proves the protocol. It is not market evidence.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
