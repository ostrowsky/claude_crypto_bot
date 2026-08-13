"""Truth contracts for the strategic learning-progress analyzer."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyze_learning_progress as A


class LearningProgressTruthTests(unittest.TestCase):
    def test_provisional_and_in_sample_history_cannot_claim_progress(self):
        learning = [{
            "ts": "2026-08-13T00:00:00Z",
            "bandit_recall_top20": 1.0,
            "bandit_evaluation_scope": "in_sample_post_fit",
            "model_auc_top20": 0.99,
            "model_evaluation_scope": "time_sorted_row_holdout_same_snapshot_label",
            "model_label_timing": "same_snapshot_current_24h_leaderboard",
        }]
        metrics = [{
            "ts": "2026-08-13T08:00:00Z",
            "_compute_early_capture.py": {"early_capture": 0.07},
        }]
        health = {
            "north_star": {
                "metric": "EarlyCapture@top20", "value": 0.07,
                "status": "provisional", "baseline_7d": None,
            },
        }
        with mock.patch.object(A, "load_learning_progress", return_value=learning), \
             mock.patch.object(A, "load_metrics_daily", return_value=metrics), \
             mock.patch.object(A, "load_latest_health", return_value=health), \
             mock.patch.object(A, "load_decisions", return_value=[]), \
             mock.patch.object(A, "load_attribution_meta", return_value=None), \
             mock.patch.object(A, "parse_spec", return_value={"available": False}):
            got = A.analyze(Path("."), None)

        self.assertEqual(got["north_star"]["trend"][0], "UNKNOWN")
        self.assertEqual(got["ml_history"]["recall_at_20"][0], "UNKNOWN")
        self.assertEqual(got["ml_history"]["auc_top20"][0], "UNKNOWN")
        rendered = A.render(got)
        self.assertIn("UNKNOWN", rendered)
        self.assertIn("immutable later-EOD", rendered)
        self.assertIn("activity, not objective progress", rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
