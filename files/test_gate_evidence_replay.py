"""Exit gates for the do_not_touch re-verification (TH-10).

The property that matters: a lock is only refreshed by evidence from the policy
running now, and never by the absence of evidence.

Spec: docs/specs/features/gate-evidence-replay-spec.md
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _replay_gate_evidence as R  # noqa: E402


class TestDayExtraction(unittest.TestCase):
    def test_reads_iso_ts_signal(self):
        self.assertEqual(R._day_of({"ts_signal": "2026-08-17T07:00:00Z"}),
                         "2026-08-17")

    def test_falls_back_to_bar_ts_millis(self):
        # 2026-08-17T07:00:00Z
        self.assertEqual(R._day_of({"bar_ts": 1786950000000}), "2026-08-17")

    def test_unknown_shape_returns_none_not_a_wrong_day(self):
        # The first version silently returned None for EVERY row, which made the
        # epoch window read n=0 — indistinguishable from a quiet bot.
        self.assertIsNone(R._day_of({"when": "yesterday"}))


class TestVerdict(unittest.TestCase):
    def test_thin_bucket_is_not_evidence(self):
        v = R.verdict_for([0.5] * (R.MIN_N - 1), [0.0] * 100)
        self.assertFalse(v["available"])

    def test_over_blocking_needs_both_positive_miss_and_sharpe(self):
        # A positive average against a bad baseline is not a reason to open a
        # gate: in the live epoch the take baseline is -0.883%, so buckets that
        # merely lost less show a positive miss with a negative Sharpe.
        losing_but_better = [-0.4 + (0.02 if i % 2 else -0.02) for i in range(60)]
        v = R.verdict_for(losing_but_better, [-0.9] * 100)
        self.assertGreater(v["miss_vs_take"], 0)
        self.assertLess(v["sharpe_sqrt_n"], 0)
        self.assertFalse(v["over_blocking"])

    def test_a_genuinely_profitable_bucket_is_flagged(self):
        winning = [0.6 + (0.05 if i % 2 else -0.05) for i in range(60)]
        v = R.verdict_for(winning, [-0.1] * 100)
        self.assertTrue(v["over_blocking"])

    def test_baseline_is_reported_beside_the_ratio(self):
        v = R.verdict_for([0.1] * 40, [-0.5] * 100)
        self.assertIn("take_baseline_avg_r5", v)   # TH-01


class TestWriteRefusesOnIncompleteEvidence(unittest.TestCase):
    def test_write_is_guarded_in_source(self):
        src = (HERE / "_replay_gate_evidence.py").read_text(encoding="utf-8")
        self.assertIn("refusing --write", src)
        # The guard must consider BOTH unverified and over-blocking gates.
        self.assertIn("if unverifiable or over:", src)

    def test_max_period_only_does_not_count_as_confirmed(self):
        src = (HERE / "_replay_gate_evidence.py").read_text(encoding="utf-8")
        self.assertIn('elif tag == "epoch":', src)
        self.assertIn("max-period only, not current policy", src)


if __name__ == "__main__":
    unittest.main()
