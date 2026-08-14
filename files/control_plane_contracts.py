"""Immutable contracts for the control-plane walking skeleton.

Everything here is frozen on purpose. The improvement loop this replaces died in
part because a hypothesis could be edited after its result was seen and because
"memory" and "execution" shared a file. Freezing the objects is the cheapest
structural answer available at this stage.

Stdlib only, and nothing here imports a trading module.

Spec: docs/specs/features/control-plane-walking-skeleton-spec.md
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Closed vocabularies. Extending one is a spec change, not a code change —
# otherwise the ledger stops being queryable and the state explosion the parent
# spec removed simply reappears one level down.
STAGES = ("OBSERVED", "PREPARED", "REGISTERED", "VALIDATING", "VERIFYING",
          "DECIDED", "CLOSED")
STATUSES = ("ACTIVE", "WAITING", "TERMINAL")
OUTCOME_REASONS = (
    "in_progress", "snapshot_invalid", "contract_rejected", "needs_validator",
    "waiting_for_data", "power_expansion", "metric_redesign",
    "supported", "refuted", "underpowered", "accepted_unknown",
    "invalid_result", "budget_exhausted", "forward_rejected",
    "operator_rejected", "rolled_back",
)


class ContractError(ValueError):
    """A contract was violated. Never downgraded to a warning."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ── snapshot ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SnapshotManifest:
    """Content-addressed evidence. Created by the orchestrator, never by an
    agent or a validator — whoever can choose the data can choose the answer."""
    snapshot_id: str
    raw_path: str
    raw_sha256: str
    row_count: int
    schema_hash: str
    created_at_utc: str
    created_by: str

    @classmethod
    def freeze(cls, raw_path: Path, *, created_by: str) -> "SnapshotManifest":
        raw = Path(raw_path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        rows = json.loads(raw.decode("utf-8"))["rows"]
        if not rows:
            raise ContractError("refusing to freeze an empty snapshot")
        schema = sha256_of(sorted(rows[0].keys()))
        return cls(
            snapshot_id=f"snap-{digest[:12]}",
            raw_path=str(raw_path),
            raw_sha256=digest,
            row_count=len(rows),
            schema_hash=schema,
            created_at_utc=_utc(),
            created_by=created_by,
        )

    def load_rows(self) -> list[dict]:
        """Read the raw bytes and verify they still hash to the manifest."""
        raw = Path(self.raw_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.raw_sha256:
            raise ContractError("snapshot content changed after freezing")
        return json.loads(raw.decode("utf-8"))["rows"]


# ── hypothesis ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HypothesisContract:
    hypothesis_id: str
    version: int
    snapshot_id: str
    policy_baseline: float
    policy_candidate: float
    primary_metric: str
    guardrail_max_admission_ratio: float
    minimum_practical_effect: float
    falsifier: str
    hit_threshold_pct: float

    @classmethod
    def smoke(cls, *, snapshot_id: str) -> "HypothesisContract":
        return cls(
            hypothesis_id="hyp-smoke-precision",
            version=1,
            snapshot_id=snapshot_id,
            policy_baseline=0.50,
            policy_candidate=0.60,
            primary_metric="precision",
            guardrail_max_admission_ratio=1.25,
            minimum_practical_effect=0.05,
            falsifier=("candidate precision does not exceed baseline by the "
                       "minimum practical effect, or admission rate exceeds "
                       "baseline x 1.25"),
            hit_threshold_pct=3.0,
        )


# ── result ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ArmResult:
    n: int
    entered: int
    hits: int
    precision: float
    base_rate: float
    lift: float
    admission_rate: float


@dataclass(frozen=True)
class ResultBundle:
    attempt_id: str
    hypothesis_id: str
    snapshot_id: str
    baseline: ArmResult
    candidate: ArmResult
    delta: float
    guardrail_ok: bool
    validator_build: str
    claimed_verdict: str

    def with_claimed_precision(self, *, candidate_precision: float) -> "ResultBundle":
        """Test-only corruption: keep everything, lie about the headline."""
        lying = ArmResult(**{**asdict(self.candidate),
                             "precision": candidate_precision})
        return ResultBundle(**{**asdict(self), "candidate": lying,
                               "baseline": self.baseline,
                               "validator_build": self.validator_build + "+corrupt"})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ResultBundle":
        for arm in ("baseline", "candidate"):
            block = data.get(arm) or {}
            # TH-01: a ratio without its base rate and lift is not evidence, so
            # a bundle that omits them cannot be constructed at all.
            for required in ("base_rate", "lift"):
                if required not in block:
                    raise ContractError(
                        f"{arm}.{required} missing — a precision without its "
                        f"base rate and lift is not evidence")
        return cls(
            attempt_id=data["attempt_id"],
            hypothesis_id=data["hypothesis_id"],
            snapshot_id=data["snapshot_id"],
            baseline=ArmResult(**data["baseline"]),
            candidate=ArmResult(**data["candidate"]),
            delta=data["delta"],
            guardrail_ok=data["guardrail_ok"],
            validator_build=data["validator_build"],
            claimed_verdict=data["claimed_verdict"],
        )


@dataclass(frozen=True)
class VerifiedResult:
    agreement: bool
    recompute: dict
    disagreement: str = ""


# ── ledger ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LedgerEvent:
    attempt_id: str
    seq: int
    stage: str
    status: str
    outcome_reason: str
    payload_hash: str
    at_utc: str = field(default_factory=_utc)

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ContractError(f"unknown stage {self.stage!r}")
        if self.status not in STATUSES:
            raise ContractError(f"unknown status {self.status!r}")
        if self.outcome_reason not in OUTCOME_REASONS:
            raise ContractError(
                f"unknown outcome_reason {self.outcome_reason!r} — the "
                f"vocabulary is closed; extending it is a spec change")


class AttemptLedger:
    """Append-only, idempotent by attempt id.

    Idempotence is what makes a restart safe: replaying an attempt that already
    reached a terminal state must not append a second history for it.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "attempts.jsonl"

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in
                self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def is_closed(self, attempt_id: str) -> dict | None:
        for event in reversed(self.events()):
            if event["attempt_id"] == attempt_id and event["status"] == "TERMINAL":
                return event
        return None

    def append(self, event: LedgerEvent) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

    def next_seq(self) -> int:
        return len(self.events())
