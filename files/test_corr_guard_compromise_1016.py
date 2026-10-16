"""correlation_guard compromises (corr-guard-compromise-1016-spec.md), written from the spec first."""
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import _backtest_corr_guard_compromise as X

ROOT = Path(__file__).resolve().parent.parent
D = datetime(2026, 10, 15, tzinfo=timezone.utc)


class TestAdmission(unittest.TestCase):
    def test_cluster_size_parse(self):
        r = "corr_guard: rho=0.737 to WLDUSDT, cluster size 2/2 (TIAUSDT, WLDUSDT)"
        self.assertEqual(X.cluster_size(r), 2)
        self.assertIsNone(X.cluster_size("something else"))

    def test_cap3(self):
        self.assertTrue(X.admit("CAP3", {"n": 2, "price": 1.0}, 1.0))
        self.assertFalse(X.admit("CAP3", {"n": 3, "price": 1.0}, 1.0))
        self.assertFalse(X.admit("CAP3", {"n": None, "price": 1.0}, 1.0))

    def test_precross(self):
        self.assertTrue(X.admit("PRECROSS", {"n": 5, "price": 1.024}, 1.0))
        self.assertFalse(X.admit("PRECROSS", {"n": 5, "price": 1.025}, 1.0))
        self.assertFalse(X.admit("PRECROSS", {"n": 5, "price": 1.0}, None))

    def test_off(self):
        self.assertTrue(X.admit("OFF", {"n": 9, "price": 2.0}, 1.0))

    def test_day_open_is_the_first_bar_of_the_utc_day(self):
        bars = [(D - timedelta(minutes=15), 0.9, 0, 0, 0, 0), (D, 1.0, 0, 0, 0, 0),
                (D + timedelta(minutes=15), 1.1, 0, 0, 0, 0)]
        idx = {b[0]: i for i, b in enumerate(bars)}
        self.assertEqual(X.day_open(bars, idx, D + timedelta(hours=13)), 1.0)
        self.assertIsNone(X.day_open(bars, idx, D + timedelta(days=1, hours=1)))


class TestTail(unittest.TestCase):
    def test_tail_weights_added_by_p(self):
        cur = {"d1": [1.0, -3.0], "d2": [-2.0]}
        add = {"d1": [-4.0], "d3": [2.0]}
        dd, worst = X.tail(cur, add, p=0.5)
        # daily: d1 = -2 - 2 = -4, d2 = -2, d3 = +1 -> cumulative -4, -6, -5 -> maxdd -6, worst -4
        self.assertEqual((dd, worst), (-6.0, -4.0))

    def test_verdict(self):
        ok = dict(dd_cur=-100, dd_new=-105, worst_cur=-90, worst_new=-95)
        self.assertEqual(X.verdict(1.0, 0.2, -0.05, **ok), "supported")
        self.assertEqual(X.verdict(1.0, 0.2, -0.05, dd_cur=-100, dd_new=-115, worst_cur=-90, worst_new=-95),
                         "refuted")
        self.assertEqual(X.verdict(1.0, 0.2, -0.05, dd_cur=-100, dd_new=-105, worst_cur=-90, worst_new=-100),
                         "refuted")
        self.assertEqual(X.verdict(0.0, -0.5, 0.0, **ok), "refuted")
        self.assertEqual(X.verdict(1.0, 0.2, -0.2, **ok), "refuted")
        self.assertEqual(X.verdict(1.0, -0.1, -0.05, **ok), "inconclusive")

    def test_pre_registration(self):
        self.assertEqual(X.ARMS, ("CAP3", "PRECROSS", "OFF"))
        self.assertEqual((X.CI, X.TAIL_TOL, X.CROSS), (0.975, 0.10, 0.025))

    def test_documented(self):
        self.assertIn("corr-guard-compromise-1016",
                      (ROOT / "docs/specs/features/corr-guard-compromise-1016-spec.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
