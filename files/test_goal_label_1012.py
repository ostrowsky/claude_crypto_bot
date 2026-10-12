"""Goal-aligned ML label research (goal-aligned-ml-label-1012-spec.md).

Written from the spec before the script: timing, labels, walk-forward embargo,
matched admission, the reachable-day count and the pre-registered verdict.
"""
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import _backtest_goal_label as G

ROOT = Path(__file__).resolve().parent.parent
T0 = datetime(2026, 10, 11, 13, 0, tzinfo=timezone.utc)


def idx_of(n, start=T0):
    return {start + timedelta(minutes=15 * k): k for k in range(n)}


class TestTiming(unittest.TestCase):
    def test_15m_decides_on_its_own_bar(self):
        self.assertEqual(G.decision_index(idx_of(10), T0, "15m"), 0)

    def test_1h_decides_on_the_last_15m_bar_of_the_hour(self):
        self.assertEqual(G.decision_index(idx_of(10), T0, "1h"), 3)

    def test_missing_bar_is_none(self):
        self.assertIsNone(G.decision_index(idx_of(2), T0, "1h"))

    def test_decision_time_is_the_bar_close(self):
        self.assertEqual(G.decision_time(T0, "15m"), T0 + timedelta(minutes=15))
        self.assertEqual(G.decision_time(T0, "1h"), T0 + timedelta(hours=1))


class TestForwardPeak(unittest.TestCase):
    def test_excludes_the_decision_bar(self):
        high = np.array([200.0, 101, 102, 103])
        close = np.array([100.0, 100, 100, 100])
        self.assertAlmostEqual(G.peak_after(high, close, 0, 3), 3.0)

    def test_truncated_window_is_none(self):
        self.assertIsNone(G.peak_after(np.ones(3), np.ones(3), 0, 3))

    def test_peak5_horizon_follows_tf(self):
        self.assertEqual(G.PEAK5_BARS["15m"], 5)
        self.assertEqual(G.PEAK5_BARS["1h"], 20)


class TestVolUnit(unittest.TestCase):
    def test_median_of_prior_days_only(self):
        rng = {"2026-10-%02d" % d: float(d) for d in range(1, 12)}
        # the day itself (11) never counts; prior 14 days available = 1..10
        self.assertEqual(G.prior_range_median(rng, "2026-10-11"), 5.5)

    def test_too_little_history_is_none(self):
        self.assertIsNone(G.prior_range_median({"2026-10-10": 3.0}, "2026-10-11"))


class TestLabels(unittest.TestCase):
    def test_each_arm(self):
        lab = G.labels_for(peak5=2.1, peak24=6.0, vol_unit=2.5, early_winner=False)
        self.assertEqual(lab, {"C_peak5": 1, "D24_5": 1, "D24_10": 0, "DVOL_2": 1, "GOAL": 0})
        lab = G.labels_for(peak5=0.5, peak24=11.0, vol_unit=6.0, early_winner=True)
        self.assertEqual(lab, {"C_peak5": 0, "D24_5": 1, "D24_10": 1, "DVOL_2": 0, "GOAL": 1})

    def test_unknown_vol_unit_is_negative(self):
        self.assertEqual(G.labels_for(peak5=0, peak24=50, vol_unit=None, early_winner=False)["DVOL_2"], 0)


class TestWalkForward(unittest.TestCase):
    def test_embargo_and_month(self):
        t = [datetime(2026, 4, 29, tzinfo=timezone.utc), datetime(2026, 4, 30, 12, tzinfo=timezone.utc),
             datetime(2026, 5, 3, tzinfo=timezone.utc), datetime(2026, 6, 1, tzinfo=timezone.utc)]
        folds = dict((m, (tr, te)) for m, tr, te in G.folds(t, ["2026-05"]))
        tr, te = folds["2026-05"]
        self.assertEqual(list(tr), [0])          # 04-30 12:00 is inside the 1-day embargo
        self.assertEqual(list(te), [2])


class TestAdmission(unittest.TestCase):
    def test_top_share(self):
        m = G.admit_top(np.array([0.1, 0.9, 0.5, 0.3, 0.7]), 0.6)
        self.assertEqual(list(m), [False, True, True, False, True])


class TestReachable(unittest.TestCase):
    def test_counts_admitted_rows_before_the_crossing(self):
        op = datetime(2026, 10, 11, tzinfo=timezone.utc)
        W = [("2026-10-11", "ICPUSDT", op, op + timedelta(hours=15)),
             ("2026-10-11", "XUSDT", op, op + timedelta(hours=10))]
        rows = [("ICPUSDT", op + timedelta(hours=13, minutes=15)),   # before the crossing
                ("XUSDT", op + timedelta(hours=11))]                 # after it
        self.assertEqual(G.reachable(rows, W), [1, 0])


class TestVerdict(unittest.TestCase):
    def test_pre_registered_rule(self):
        self.assertEqual(G.verdict(1.0, 0.2, -0.05, 0.6), "supported")
        self.assertEqual(G.verdict(1.0, -0.1, -0.05, 0.6), "inconclusive")
        self.assertEqual(G.verdict(1.0, 0.2, -0.05, 0.4), "inconclusive")
        self.assertEqual(G.verdict(0.0, -0.5, 0.1, 1.0), "refuted")
        self.assertEqual(G.verdict(2.0, 1.0, -0.2, 1.0), "refuted")

    def test_pre_registration(self):
        self.assertEqual(G.ARMS, ("C_peak5", "D24_5", "D24_10", "DVOL_2", "GOAL"))
        self.assertEqual((G.A_PRIMARY, G.A_SECONDARY), (0.60, 0.40))
        self.assertEqual(G.EMBARGO, timedelta(days=1))
        self.assertEqual(G.CI_LEVEL, 0.9875)
        self.assertEqual((G.GOAL_PP, G.NONINF_PP), (0.5, -0.10))


if __name__ == "__main__":
    unittest.main(verbosity=2)
