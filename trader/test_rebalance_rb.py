"""RB -- periodic rebalancing vs holding the same initial weights (rebalancing-rb-1029-spec.md §4).
Written BEFORE rebalance.py / experiment_rb.py.

Interface the tests fix:
  rebalance.simulate(px: {sym: array}, weights: {sym | "USDT": w}, period: int | None, fm, start=1e6, min_order=10)
      -> {"equity": array (one mark per day, at that day's price, before trading), "final": float,
          "trade_days": [day index], "fees": float, "turnover": float, "cash": float, "qty": {sym: float}}
      period None = buy-and-hold of the initial weights
  experiment_rb.candidates() (6), admit(res), open_test(marker, info), main(["test"]) == 2
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np


def fm0():
    import fill_model as FM
    return FM.FillModel(taker_fee=0.0, maker_fee=0.0, half_spread=0.0, impact_k=0.0)


def fm1():
    import fill_model as FM
    return FM.model("taker", 1)


class TestEngine(unittest.TestCase):
    def test_bh_equals_weighted_price_relatives(self):
        import rebalance as RB
        px = {"BTC": np.array([100.0, 110.0, 90.0, 120.0]), "ETH": np.array([10.0, 9.0, 12.0, 11.0])}
        r = RB.simulate(px, {"BTC": 0.5, "ETH": 0.5}, None, fm0())
        want = 1e6 * (0.5 * px["BTC"] / 100 + 0.5 * px["ETH"] / 10)
        self.assertTrue(np.allclose(r["equity"][1:], want[1:]))
        self.assertEqual(r["trade_days"], [0])

    def test_constant_prices_no_change(self):
        import rebalance as RB
        px = {"BTC": np.full(10, 100.0), "ETH": np.full(10, 10.0)}
        r = RB.simulate(px, {"BTC": 0.5, "ETH": 0.5}, 1, fm0())
        self.assertTrue(np.allclose(r["equity"], 1e6))
        self.assertAlmostEqual(r["turnover"], 1e6)                      # only the initial purchase

    def test_schedule(self):
        import rebalance as RB
        px = {"BTC": 100.0 * np.exp(np.linspace(0, 0.3, 22)), "ETH": np.full(22, 10.0)}
        r = RB.simulate(px, {"BTC": 0.5, "ETH": 0.5}, 7, fm0())
        self.assertEqual(r["trade_days"], [0, 7, 14, 21])

    def test_rebalanced_value_and_costs(self):
        import rebalance as RB
        px = {"BTC": np.array([100.0, 200.0, 200.0])}
        r = RB.simulate(px, {"BTC": 0.5, "USDT": 0.5}, 1, fm0())
        # day 1: 0.5M of BTC became 1.0M; equity 1.5M -> sell 0.25M of BTC
        self.assertAlmostEqual(r["equity"][1], 1.5e6)
        self.assertAlmostEqual(r["turnover"], 0.5e6 + 0.25e6)
        self.assertAlmostEqual(r["equity"][2], 1.5e6)
        c = RB.simulate(px, {"BTC": 0.5, "USDT": 0.5}, 1, fm1())
        self.assertGreater(c["fees"], 0.0)
        side = 0.001 + 0.0008                                             # taker + half-spread per side
        self.assertAlmostEqual(c["fees"] / c["turnover"], side, delta=0.0002)

    def test_conservation_and_usdt_leg(self):
        import rebalance as RB
        rng = np.random.default_rng(1)
        px = {"BTC": 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 60))), "ETH": 10 * np.exp(np.cumsum(rng.normal(0, 0.04, 60)))}
        r = RB.simulate(px, {"BTC": 0.4, "ETH": 0.3, "USDT": 0.3}, 7, fm1())
        last = r["cash"] + sum(q * px[s][-1] for s, q in r["qty"].items())
        self.assertAlmostEqual(r["final"], last, places=4)
        self.assertGreater(r["cash"], 0.0)                               # the USDT leg is held as cash


class TestDecisions(unittest.TestCase):
    def test_candidates_admit_marker(self):
        import experiment_rb as EX
        self.assertEqual(len(EX.candidates()), 6)
        good = {"paired": {"lo95": 0.001}, "paired_by_year": {"2023": 0.01, "2024": -0.01, "2025": 0.02}}
        self.assertTrue(EX.admit(good)[0])
        self.assertFalse(EX.admit(dict(good, paired={"lo95": -0.001}))[0])
        self.assertFalse(EX.admit(dict(good, paired_by_year={"2023": -0.01, "2024": -0.01, "2025": 0.02}))[0])
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "RB_TEST_OPENED.json"
            EX.open_test(m, {})
            with self.assertRaises(RuntimeError):
                EX.open_test(m, {})
        self.assertEqual(EX.main(["test"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
