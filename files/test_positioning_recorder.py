"""Guards for the positioning recorder and its clustering analysis.

This data exists because Binance serves open interest, taker flow and long/short
positioning for 30 DAYS ONLY. Everything not written down is gone for good, so
the recorder's job is to be boring and correct for months. The tests below pin
the ways it could quietly stop being either.

The clustering guards matter more than they look. Clustering ALWAYS returns
clusters; on 87 rows it would return beautiful ones, and they would be noise
wearing the shape of a result. The refusal to run below a sample threshold is
the feature, not the limitation.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import positioning_recorder as PR  # noqa: E402
import _cluster_positioning as CL  # noqa: E402

REC_SRC = (HERE / "positioning_recorder.py").read_text(encoding="utf-8")
CLU_SRC = (HERE / "_cluster_positioning.py").read_text(encoding="utf-8")
CMD_SRC = (HERE.parent / "run_positioning_recorder.cmd").read_text(encoding="utf-8")


class TestFlowClassification(unittest.TestCase):
    """Open interest alone is direction-blind: every contract has a long and a
    short. Price over the same window is what says which side the money took."""

    def test_money_in_while_price_rises_is_longs_opening(self):
        self.assertEqual(PR.classify(5.0, 3.0), "longs_opening")

    def test_money_in_while_price_falls_is_shorts_opening(self):
        self.assertEqual(PR.classify(5.0, -3.0), "shorts_opening")

    def test_money_out_while_price_rises_is_short_covering(self):
        # The distinction that matters most: a rise carried by closures has no
        # new demand behind it and stops when the closers are done.
        self.assertEqual(PR.classify(-5.0, 3.0), "short_covering")

    def test_money_out_while_price_falls_is_longs_closing(self):
        self.assertEqual(PR.classify(-5.0, -3.0), "longs_closing")

    def test_small_moves_are_called_flat_rather_than_forced(self):
        self.assertEqual(PR.classify(0.1, 0.1), "flat")

    def test_missing_inputs_do_not_become_a_class(self):
        self.assertEqual(PR.classify(None, 3.0), "unknown")
        self.assertEqual(PR.classify(5.0, None), "unknown")


class TestTheRecorderCannotLoseData(unittest.TestCase):
    def test_the_store_is_append_only_on_snapshot(self):
        self.assertIn('io.open(STORE, "a", encoding="utf-8")', REC_SRC)

    def test_snapshot_runs_before_resolve_in_the_scheduled_job(self):
        # resolve rewrites the whole store; running it first would operate on a
        # file the snapshot is about to append to.
        self.assertLess(CMD_SRC.index("snapshot"), CMD_SRC.index("resolve"))

    def test_the_job_log_is_appended_not_truncated(self):
        # The log is the evidence the task actually ran -- the absence of which
        # cost 11 silent days in CLAUDE.md section 5.
        self.assertIn('>> "%LOG%"', CMD_SRC)
        self.assertNotIn('> "%LOG%" 2>&1', CMD_SRC.replace('>> "%LOG%" 2>&1', ''))

    def test_failures_are_recorded_rather_than_swallowed(self):
        self.assertIn("SNAPSHOT FAILED", CMD_SRC)
        self.assertIn("RESOLVE FAILED", CMD_SRC)


class TestReportAlwaysShowsTheBaseline(unittest.TestCase):
    """A class that 'wins 40% of the time' means nothing until the pool's own
    rate sits beside it -- on a day the whole market rises, everything wins."""

    def test_the_pool_row_is_printed(self):
        self.assertIn("ALL COINS (pool)", REC_SRC)

    def test_small_groups_are_refused_rather_than_scored(self):
        self.assertIn("below --min-n", REC_SRC)

    def test_it_warns_while_the_history_is_short(self):
        self.assertIn("This is an anecdote, not evidence", REC_SRC)


class TestClusteringRefusesUntilItCan(unittest.TestCase):
    def test_it_refuses_below_the_sample_threshold(self):
        self.assertIn("NOT ENOUGH DATA", CLU_SRC)
        self.assertIn("refusing to cluster", CLU_SRC)

    def test_it_counts_days_not_rows(self):
        # ~87 coins in one snapshot is closer to ONE observation than to 87:
        # the whole market moves together.
        self.assertIn("min_days", CLU_SRC)
        self.assertIn("not independent", CLU_SRC)

    def test_the_split_is_by_time(self):
        self.assertIn('cut = days[int(len(days) * 0.7)]', CLU_SRC)

    def test_the_verdict_comes_from_a_shuffled_null(self):
        # Not silhouette, not inertia: those measure how round the clusters are,
        # which is not the question being asked. The word appears in the module
        # docstring saying exactly that, so the check is that nothing COMPUTES
        # it -- prose explaining its absence is the point, not a violation.
        self.assertIn("rng.shuffle(yy)", CLU_SRC)
        code = CLU_SRC.split('"""', 2)[-1]
        for banned in ("silhouette_score", "def silhouette", "inertia_"):
            self.assertNotIn(banned, code)

    def test_the_verdict_words_come_from_z_and_nothing_else(self):
        self.assertIn('"  real separation" if z > 3', CLU_SRC)
        self.assertIn("indistinguishable from random", CLU_SRC)

    def test_the_multiple_comparison_risk_is_stated(self):
        self.assertIn("multiple-comparison", CLU_SRC)


class TestSpreadMeasuresOutcomeNotShape(unittest.TestCase):
    def test_identical_outcomes_give_zero_spread(self):
        lab = np.array([0, 0, 1, 1])
        y = np.array([2.0, 2.0, 2.0, 2.0])
        self.assertAlmostEqual(CL.spread(lab, y), 0.0, places=9)

    def test_separated_outcomes_give_positive_spread(self):
        lab = np.array([0, 0, 1, 1])
        y = np.array([1.0, 1.0, 5.0, 5.0])
        self.assertAlmostEqual(CL.spread(lab, y), 2.0, places=9)

    def test_a_tiny_extreme_cluster_cannot_dominate(self):
        # Weighting by cluster size stops a 1-row cluster with a wild return
        # from carrying the whole number.
        big = np.array([0] * 99 + [1])
        y = np.concatenate([np.zeros(99), np.array([100.0])])
        self.assertLess(CL.spread(big, y), 2.5)


class TestKMeansIsSane(unittest.TestCase):
    def test_it_separates_two_obvious_blobs(self):
        rng = np.random.RandomState(0)
        X = np.vstack([rng.normal(-5, 0.3, size=(50, 3)),
                       rng.normal(+5, 0.3, size=(50, 3))])
        lab, C = CL.kmeans(X, 2, seed=0)
        self.assertEqual(len(set(lab[:50])), 1)
        self.assertEqual(len(set(lab[50:])), 1)
        self.assertNotEqual(lab[0], lab[50])

    def test_it_returns_one_centre_per_k(self):
        rng = np.random.RandomState(1)
        X = rng.normal(size=(60, 4))
        lab, C = CL.kmeans(X, 3, seed=0)
        self.assertEqual(C.shape[0], 3)
        self.assertEqual(len(lab), 60)


if __name__ == "__main__":
    unittest.main(verbosity=2)
