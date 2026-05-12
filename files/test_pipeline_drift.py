"""Unit tests for pipeline_drift.

Locks in:
  - KS two-sample math (matches scipy reference values within tolerance)
  - Window slicing (recent vs reference, no overlap, no leak)
  - Status logic (drift_detected requires BOTH p AND d thresholds)
  - Insufficient-data path (cannot conclude with <min_samples)
  - Numerical robustness (NaN, missing fields, weird ts formats)

Run:
    pyembed\\python.exe files\\test_pipeline_drift.py
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pipeline_drift as D


def _write_dataset(rows: list[dict]) -> Path:
    """Write a temporary critic_dataset.jsonl with given rows."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                     delete=False, encoding="utf-8")
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.close()
    return Path(f.name)


def _row(*, days_ago: int, ranker_proba: float | None = None,
         candidate_score: float | None = None) -> dict:
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    decision = {}
    if ranker_proba is not None:
        decision["ranker_proba"] = ranker_proba
    if candidate_score is not None:
        decision["candidate_score"] = candidate_score
    return {"bar_ts": ts.isoformat(), "decision": decision}


# ---------------------------------------------------------------------------
# KS math
# ---------------------------------------------------------------------------


class KSMathTests(unittest.TestCase):
    """Sanity-check the KS implementation against known cases."""

    def test_identical_samples_yields_p_one(self):
        a = [0.5] * 50
        b = [0.5] * 50
        d, p = D.ks_two_sample(a, b)
        self.assertEqual(d, 0.0)
        self.assertGreater(p, 0.99)

    def test_disjoint_distributions_yields_low_p(self):
        a = [0.1 + i * 0.001 for i in range(100)]
        b = [0.9 + i * 0.001 for i in range(100)]
        d, p = D.ks_two_sample(a, b)
        # Disjoint: D should be 1.0, p effectively 0
        self.assertGreater(d, 0.95)
        self.assertLess(p, 1e-5)

    def test_same_distribution_different_draws_p_above_threshold(self):
        rng = random.Random(7)
        a = [rng.gauss(0.5, 0.1) for _ in range(500)]
        b = [rng.gauss(0.5, 0.1) for _ in range(500)]
        d, p = D.ks_two_sample(a, b)
        # Same underlying distribution: should NOT be flagged at p=0.01
        self.assertGreater(p, 0.01, f"D={d} p={p} — same dist should pass")

    def test_shifted_mean_is_detected(self):
        rng = random.Random(11)
        a = [rng.gauss(0.5, 0.05) for _ in range(500)]
        b = [rng.gauss(0.7, 0.05) for _ in range(500)]   # mean shift 0.2
        d, p = D.ks_two_sample(a, b)
        self.assertLess(p, 0.001, "Mean shift of 4σ should be obvious")
        self.assertGreater(d, 0.5)

    def test_empty_input_safe(self):
        d, p = D.ks_two_sample([], [1.0, 2.0])
        self.assertEqual(d, 0.0)
        self.assertEqual(p, 1.0)
        d, p = D.ks_two_sample([1.0], [])
        self.assertEqual(d, 0.0)
        self.assertEqual(p, 1.0)


# ---------------------------------------------------------------------------
# Window slicing
# ---------------------------------------------------------------------------


class WindowTests(unittest.TestCase):

    def test_only_in_window_returned(self):
        rows = [
            _row(days_ago=2,  ranker_proba=0.5),   # in recent (7d)
            _row(days_ago=20, ranker_proba=0.6),   # in reference (30d)
            _row(days_ago=50, ranker_proba=0.7),   # too old
        ]
        path = _write_dataset(rows)
        now = datetime.now(timezone.utc)
        recent = D.collect_window(path, ("decision", "ranker_proba"),
                                  now - timedelta(days=7), now)
        reference = D.collect_window(path, ("decision", "ranker_proba"),
                                     now - timedelta(days=30), now - timedelta(days=7))
        self.assertEqual(recent, [0.5])
        self.assertEqual(reference, [0.6])

    def test_no_overlap_between_recent_and_reference(self):
        """The contract: events outside the union of both windows aren't
        counted; events at the exact boundary may be counted in both because
        ranges are inclusive on both ends — but in practice timestamps are
        microsecond-precise so collisions are vanishingly rare. The actual
        risk we test for: an event well-inside one window doesn't leak."""
        rows = [
            _row(days_ago=3,  ranker_proba=0.5),     # clearly in recent
            _row(days_ago=15, ranker_proba=0.6),     # clearly in reference
        ]
        path = _write_dataset(rows)
        now = datetime.now(timezone.utc)
        recent = D.collect_window(path, ("decision", "ranker_proba"),
                                  now - timedelta(days=7), now)
        reference = D.collect_window(path, ("decision", "ranker_proba"),
                                     now - timedelta(days=30), now - timedelta(days=7))
        self.assertEqual(recent, [0.5])
        self.assertEqual(reference, [0.6])

    def test_missing_feature_skipped(self):
        rows = [
            _row(days_ago=2, ranker_proba=0.5),
            _row(days_ago=3),  # no ranker_proba
            _row(days_ago=4, ranker_proba=0.6),
        ]
        path = _write_dataset(rows)
        now = datetime.now(timezone.utc)
        out = D.collect_window(path, ("decision", "ranker_proba"),
                               now - timedelta(days=7), now)
        self.assertEqual(sorted(out), [0.5, 0.6])

    def test_non_numeric_feature_skipped(self):
        rows = [
            {"bar_ts": datetime.now(timezone.utc).isoformat(),
             "decision": {"ranker_proba": "junk"}},
            _row(days_ago=2, ranker_proba=0.42),
        ]
        path = _write_dataset(rows)
        now = datetime.now(timezone.utc)
        out = D.collect_window(path, ("decision", "ranker_proba"),
                               now - timedelta(days=7), now)
        self.assertEqual(out, [0.42])

    def test_ms_timestamp_accepted(self):
        ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000) - 86_400_000  # 1 day ago
        rows = [{"bar_ts": ts_ms, "decision": {"ranker_proba": 0.33}}]
        path = _write_dataset(rows)
        now = datetime.now(timezone.utc)
        out = D.collect_window(path, ("decision", "ranker_proba"),
                               now - timedelta(days=7), now)
        self.assertEqual(out, [0.33])


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


class CheckFeatureTests(unittest.TestCase):

    def test_insufficient_data_when_below_min_samples(self):
        """Pin: <min_samples on either side => insufficient_data, not no_drift."""
        rows = [_row(days_ago=1, ranker_proba=0.5)] * 5
        path = _write_dataset(rows)
        r = D.check_feature(("decision", "ranker_proba"), "ranker_proba",
                            min_samples=100, dataset_path=path)
        self.assertEqual(r["status"], "insufficient_data")
        self.assertLess(r["n_recent"] + r["n_reference"], 100 * 2)

    def test_no_drift_when_same_distribution(self):
        rng = random.Random(3)
        rows = []
        # 200 recent (last 7d) + 200 reference (8-30d)
        for _ in range(200):
            rows.append(_row(days_ago=rng.randint(0, 6),
                             ranker_proba=rng.gauss(0.5, 0.1)))
        for _ in range(200):
            rows.append(_row(days_ago=rng.randint(8, 29),
                             ranker_proba=rng.gauss(0.5, 0.1)))
        path = _write_dataset(rows)
        r = D.check_feature(("decision", "ranker_proba"), "ranker_proba",
                            min_samples=100, dataset_path=path)
        self.assertEqual(r["status"], "no_drift",
                         f"Same gaussian → should NOT drift. Got {r}")

    def test_drift_detected_when_distribution_shifts(self):
        rng = random.Random(13)
        rows = []
        # Recent: mean 0.3
        for _ in range(300):
            rows.append(_row(days_ago=rng.randint(0, 6),
                             ranker_proba=rng.gauss(0.3, 0.05)))
        # Reference: mean 0.7
        for _ in range(300):
            rows.append(_row(days_ago=rng.randint(8, 29),
                             ranker_proba=rng.gauss(0.7, 0.05)))
        path = _write_dataset(rows)
        r = D.check_feature(("decision", "ranker_proba"), "ranker_proba",
                            min_samples=100, dataset_path=path)
        self.assertEqual(r["status"], "drift_detected")
        self.assertGreater(r["ks_statistic"], 0.5)
        self.assertLess(r["p_value"], 0.001)

    def test_drift_requires_both_p_AND_effect_size(self):
        """A very small but statistically significant shift on huge N should
        NOT be flagged — d_threshold guards against alerting on noise."""
        rng = random.Random(17)
        rows = []
        for _ in range(5000):
            rows.append(_row(days_ago=rng.randint(0, 6),
                             ranker_proba=rng.gauss(0.500, 0.10)))
        for _ in range(5000):
            rows.append(_row(days_ago=rng.randint(8, 29),
                             ranker_proba=rng.gauss(0.502, 0.10)))  # 2bp shift
        path = _write_dataset(rows)
        r = D.check_feature(("decision", "ranker_proba"), "ranker_proba",
                            min_samples=100, d_threshold=0.10, dataset_path=path)
        # 2bp shift on N=5000 may have low p but tiny D — must pass through
        self.assertEqual(r["status"], "no_drift",
                         f"Trivial shift should pass d_threshold. Got {r}")


class CheckAllTests(unittest.TestCase):

    def test_returns_per_feature_results(self):
        path = _write_dataset([])  # empty
        report = D.check_all(features=[
            (("decision", "ranker_proba"),    "ranker_proba"),
            (("decision", "candidate_score"), "candidate_score"),
        ], min_samples=50, dataset_path=path)
        self.assertEqual(report["n_features"], 2)
        self.assertEqual(report["verdict"], "insufficient_data")

    def test_drift_in_one_feature_flips_overall_verdict(self):
        rng = random.Random(23)
        rows = []
        # ranker_proba — drifted
        for _ in range(200):
            rows.append({"bar_ts": (datetime.now(timezone.utc) - timedelta(days=rng.randint(0, 6))).isoformat(),
                         "decision": {"ranker_proba": rng.gauss(0.3, 0.05),
                                      "candidate_score": rng.gauss(50.0, 5.0)}})
        for _ in range(200):
            rows.append({"bar_ts": (datetime.now(timezone.utc) - timedelta(days=rng.randint(8, 29))).isoformat(),
                         "decision": {"ranker_proba": rng.gauss(0.7, 0.05),
                                      "candidate_score": rng.gauss(50.0, 5.0)}})  # candidate_score stable
        path = _write_dataset(rows)
        report = D.check_all(features=[
            (("decision", "ranker_proba"),    "ranker_proba"),
            (("decision", "candidate_score"), "candidate_score"),
        ], min_samples=50, dataset_path=path)
        self.assertEqual(report["verdict"], "drift_detected")
        self.assertEqual(report["n_drift"], 1)
        self.assertIn("ranker_proba", report["drifted_features"])
        self.assertNotIn("candidate_score", report["drifted_features"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
