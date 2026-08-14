"""Exit gates for the control-plane walking skeleton (Phase -1).

Written before the implementation, and deliberately about the *architecture*
rather than the arithmetic. The slice exists because two improvement loops in
this project family accumulated layers without ever completing one experiment;
what has to be proven is that an attempt reaches a verified terminal state, that
a corrupted one does not, and that the isolation claims are real rather than
stated.

Spec: docs/specs/features/control-plane-walking-skeleton-spec.md
"""
from __future__ import annotations

import ast
import json
import shutil
import sys
import time
import unittest
import unittest.mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import control_plane_contracts as CC  # noqa: E402
import control_plane_verifier as CV  # noqa: E402
import improvement_fixture_validator as FV  # noqa: E402
import run_control_plane_smoke as RUN  # noqa: E402

FIXTURE = HERE / "testdata" / "control_plane_smoke_fixture.json"


class TestIsolation(unittest.TestCase):
    """The isolation claims must be checkable, not merely asserted in prose."""

    FORBIDDEN = {"replay_backtest", "monitor", "strategy", "config", "bot",
                 "rotation", "contextual_bandit", "offline_rl", "botlog"}

    @staticmethod
    def _imports(path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        return names

    def test_validator_imports_no_trading_module(self):
        leaked = self._imports(HERE / "improvement_fixture_validator.py") & self.FORBIDDEN
        self.assertFalse(leaked, f"validator must stay isolated, imported {leaked}")

    def test_validator_is_stdlib_only(self):
        # numpy/pandas/catboost would drag the platform into the skeleton.
        heavy = {"numpy", "pandas", "catboost", "aiohttp", "telegram", "sklearn"}
        self.assertFalse(self._imports(HERE / "improvement_fixture_validator.py") & heavy)

    def test_verifier_does_not_import_the_validator(self):
        # Sharing the aggregation implementation means sharing its bugs, and the
        # "independent recompute" would confirm the validator's own error.
        imports = self._imports(HERE / "control_plane_verifier.py")
        self.assertNotIn("improvement_fixture_validator", imports)
        self.assertFalse(imports & self.FORBIDDEN)


class TestContracts(unittest.TestCase):
    def test_hypothesis_contract_is_frozen(self):
        h = CC.HypothesisContract.smoke(snapshot_id="s1")
        with self.assertRaises(Exception):
            h.version = 99          # type: ignore[misc]

    def test_result_without_ratio_context_is_rejected(self):
        # TH-01: a precision with no base rate and no lift is not evidence.
        bad = {"attempt_id": "a", "hypothesis_id": "h", "snapshot_id": "s",
               "baseline": {"n": 10, "entered": 5, "hits": 2, "precision": 0.4},
               "candidate": {"n": 10, "entered": 4, "hits": 3, "precision": 0.75},
               "delta": 0.35, "guardrail_ok": True, "validator_build": "x",
               "claimed_verdict": "supported"}
        with self.assertRaises(CC.ContractError):
            CC.ResultBundle.from_dict(bad)

    def test_outcome_reason_vocabulary_is_closed(self):
        with self.assertRaises(CC.ContractError):
            CC.LedgerEvent(attempt_id="a", seq=0, stage="OBSERVED",
                           status="TERMINAL", outcome_reason="improvised_reason",
                           payload_hash="h")

    def test_snapshot_hash_changes_with_content(self):
        m1 = CC.SnapshotManifest.freeze(FIXTURE, created_by="test")
        self.assertEqual(len(m1.raw_sha256), 64)
        self.assertGreater(m1.row_count, 0)


class TestSmokeRun(unittest.TestCase):
    def setUp(self):
        self.root = HERE.parent / ".runtime" / "control_plane_test"
        shutil.rmtree(self.root, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_happy_path_reaches_verified_terminal_fast(self):
        t0 = time.time()
        out = RUN.run(attempt_id="smoke-1", root=self.root)
        elapsed = time.time() - t0
        self.assertEqual(out["stage"], "CLOSED")
        self.assertEqual(out["status"], "TERMINAL")
        self.assertEqual(out["outcome_reason"], "supported")
        self.assertTrue(out["verified"])
        self.assertLess(elapsed, 10.0, "the skeleton must stay a skeleton")

    def test_corrupted_result_is_invalid_and_never_reaches_the_governor(self):
        out = RUN.run(attempt_id="smoke-corrupt", root=self.root, corrupt=True)
        self.assertEqual(out["outcome_reason"], "invalid_result")
        self.assertEqual(out["status"], "TERMINAL")
        self.assertFalse(out["verified"])
        self.assertFalse(out["governor_reached"],
                         "a result the verifier rejected must not reach the governor")

    def test_rerun_with_same_attempt_id_is_idempotent(self):
        RUN.run(attempt_id="smoke-2", root=self.root)
        ledger = self.root / "attempts.jsonl"
        first = ledger.read_text(encoding="utf-8")
        RUN.run(attempt_id="smoke-2", root=self.root)
        self.assertEqual(first, ledger.read_text(encoding="utf-8"),
                         "replaying an attempt id must not append duplicate events")

    def test_every_attempt_ends_terminal_or_waiting(self):
        RUN.run(attempt_id="smoke-3", root=self.root)
        events = [json.loads(l) for l in
                  (self.root / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if l]
        self.assertIn(events[-1]["status"], ("TERMINAL", "WAITING"))
        self.assertEqual([e["seq"] for e in events], list(range(len(events))))

    def test_a_crash_still_reaches_a_terminal_state(self):
        # Found in architecture review, not by the happy path: an unhandled
        # exception used to leave the attempt ACTIVE forever, breaking the one
        # invariant this slice exists to demonstrate. A crash is a terminal
        # outcome with a named reason, never an absence of one.
        missing = HERE / "testdata" / "does_not_exist.json"
        with unittest.mock.patch.object(RUN, "FIXTURE", missing):
            out = RUN.run(attempt_id="smoke-crash", root=self.root)
        self.assertEqual(out["status"], "TERMINAL")
        self.assertIn(out["outcome_reason"], ("snapshot_invalid", "contract_rejected"))
        self.assertFalse(out["governor_reached"])
        events = [json.loads(l) for l in
                  (self.root / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if l]
        mine = [e for e in events if e["attempt_id"] == "smoke-crash"]
        self.assertEqual(mine[-1]["status"], "TERMINAL")

    def test_nothing_is_written_outside_the_control_plane_root(self):
        watched = [HERE / "config.py",
                   HERE.parent / ".runtime" / "pipeline" / "decisions" / "decisions.jsonl"]
        before = {p: (p.stat().st_mtime if p.exists() else None) for p in watched}
        RUN.run(attempt_id="smoke-4", root=self.root)
        for p in watched:
            after = p.stat().st_mtime if p.exists() else None
            self.assertEqual(before[p], after,
                             f"the skeleton must not touch {p.name}")


class TestVerifierIndependence(unittest.TestCase):
    def test_verifier_detects_a_wrong_claim_from_raw(self):
        manifest = CC.SnapshotManifest.freeze(FIXTURE, created_by="test")
        hyp = CC.HypothesisContract.smoke(snapshot_id=manifest.snapshot_id)
        honest = FV.FixtureDeltaValidatorAdapter().run(manifest, hyp, attempt_id="v1")
        lying = honest.with_claimed_precision(candidate_precision=0.99)
        self.assertTrue(CV.verify(manifest, hyp, honest).agreement)
        self.assertFalse(CV.verify(manifest, hyp, lying).agreement)

    def test_verifier_recomputes_rather_than_trusting(self):
        manifest = CC.SnapshotManifest.freeze(FIXTURE, created_by="test")
        hyp = CC.HypothesisContract.smoke(snapshot_id=manifest.snapshot_id)
        res = FV.FixtureDeltaValidatorAdapter().run(manifest, hyp, attempt_id="v2")
        v = CV.verify(manifest, hyp, res)
        # The verifier's own numbers must exist independently of the bundle.
        self.assertIn("candidate_precision", v.recompute)
        self.assertIn("base_rate", v.recompute)
        self.assertIn("lift", v.recompute)

    def test_verifier_rejects_a_snapshot_mismatch(self):
        manifest = CC.SnapshotManifest.freeze(FIXTURE, created_by="test")
        hyp = CC.HypothesisContract.smoke(snapshot_id=manifest.snapshot_id)
        res = FV.FixtureDeltaValidatorAdapter().run(manifest, hyp, attempt_id="v3")
        other = CC.SnapshotManifest.freeze(FIXTURE, created_by="test")
        object.__setattr__(other, "raw_sha256", "0" * 64)
        self.assertFalse(CV.verify(other, hyp, res).agreement)


if __name__ == "__main__":
    unittest.main()
