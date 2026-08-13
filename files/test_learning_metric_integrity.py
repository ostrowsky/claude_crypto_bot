"""Focused contracts for learning-metric truthfulness.

Run:
    python -X utf8 files/test_learning_metric_integrity.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import offline_rl


class RatioContextTests(unittest.TestCase):
    def test_recall_is_interpretable_only_beside_action_and_base_rates(self):
        got = offline_rl._binary_policy_ratio_context(
            total_rows=100,
            total_enter=80,
            total_positive=20,
            true_positive_enter=20,
        )

        self.assertEqual(got["recall"], 1.0)
        self.assertEqual(got["action_rate"], 0.8)
        self.assertEqual(got["base_rate"], 0.2)
        self.assertEqual(got["precision"], 0.25)
        self.assertEqual(got["recall_lift"], 1.25)
        self.assertEqual(got["precision_lift"], 1.25)

    def test_zero_denominators_do_not_invent_lift(self):
        got = offline_rl._binary_policy_ratio_context(
            total_rows=0,
            total_enter=0,
            total_positive=0,
            true_positive_enter=0,
        )

        self.assertEqual(got["recall"], 0.0)
        self.assertEqual(got["action_rate"], 0.0)
        self.assertEqual(got["recall_lift"], 0.0)
        self.assertEqual(got["precision_lift"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
