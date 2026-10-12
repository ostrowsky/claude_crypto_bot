"""ML-label follow-ups (ml-label-followups-1012-spec.md), written from the spec first."""
import random
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

import _backtest_ml_label_followups as F

T0 = datetime(2026, 10, 11, 13, 0, tzinfo=timezone.utc)
M15 = timedelta(minutes=15)


def row(sym, k, exit_k, pnl, tf="15m"):
    return {"sym": sym, "tf": tf, "dec": T0 + k * M15, "exit": T0 + exit_k * M15, "pnl": pnl}


class TestDedup(unittest.TestCase):
    def test_one_position_per_coin_plus_cooldown(self):
        rows = [row("A", 0, 4, 1.0),       # opens; busy until 4 + 19 bars = 23
                row("A", 2, 6, 2.0),       # inside the open trade
                row("A", 22, 30, 3.0),     # inside the cooldown
                row("A", 24, 30, 4.0),     # free again
                row("B", 2, 6, 5.0)]       # another coin is independent
        self.assertEqual(F.dedup_trades(rows, cooldown=19), [0, 3, 4])

    def test_incomplete_trade_never_opens(self):
        rows = [row("A", 0, 4, float("nan")), row("A", 1, 4, 1.0)]
        self.assertEqual(F.dedup_trades(rows, cooldown=19), [1])

    def test_cooldown_uses_the_rows_own_tf(self):
        rows = [row("A", 0, 4, 1.0, tf="1h"), row("A", 30, 40, 1.0, tf="1h")]
        # exit at +4 bars of 15m = 14:00; 19 one-hour bars of cooldown -> busy until 09:00 next day
        self.assertEqual(F.dedup_trades(rows, cooldown=19), [0])


class TestChanged(unittest.TestCase):
    def test_added_and_removed(self):
        a = np.array([True, True, False, False])
        c = np.array([True, False, True, False])
        add, rem = F.split_changed(a, c)
        self.assertEqual((list(add), list(rem)), ([1], [2]))


class TestClusterDelta(unittest.TestCase):
    def test_point_is_the_mean_difference(self):
        rnd = random.Random(1)
        pt, lo, hi = F.cluster_delta(["d1", "d2"], [1.0, 3.0], ["d1", "d2"], [0.0, 2.0], rnd, boot=200)
        self.assertAlmostEqual(pt, 1.0)
        self.assertLessEqual(lo, pt)
        self.assertGreaterEqual(hi, pt)


class TestOutcomes(unittest.TestCase):
    def test_test1_outcome(self):
        self.assertEqual(F.outcome_shadow(-0.05, 0.3), "shadow-worthy")
        self.assertEqual(F.outcome_shadow(-0.5, -0.01), "not worth a shadow")
        self.assertEqual(F.outcome_shadow(-0.3, 0.2), "undetermined")

    def test_pre_registration(self):
        self.assertEqual(F.CI, 0.95)
        self.assertEqual(F.MIN_MLDS_TRAIN, 500)
        self.assertEqual(F.ARMS2, ("SRC_MLDS", "SRC_CRITIC"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
