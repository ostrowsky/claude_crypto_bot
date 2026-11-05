"""G1 / G2 stage 0 (funding-premium-g1g2-1105-spec.md §6). Written BEFORE funding_premium.py. Synthetic bars, offline.

Interface the tests fix:
  funding_premium.g1_trade(bars, t_fund, rate, cost_side, prev_rate=None) -> {"ret_pct", "price_pct", "rate_pct",
       "cost_pct", "entry_close_ms", "causal"} | None
       side: short at rate > 0, long at rate < 0; entry: close of the bar closing at t_fund - 15 min (t_fund floored to 15 min);
       exit: close of the bar closing at t_fund + 15 min
  funding_premium.causal_ok(rate, prev_rate, thr) -> abs(rate) >= thr, abs(prev_rate) >= thr, same sign
  funding_premium.g2_trades(perp, spot, mult, thr_of_bar, funding, cost_of_bar, excluded_until_ms=0)
       -> [{"ret_pct", "price_pct", "funding_pct", "cost_pct", "entry_close_ms", "reason", "prem_entry", "prem_exit", "bars"}]
       thr_of_bar(t_ms) -> threshold or None; cost_of_bar(t_ms) -> cost fraction per round trip of the pair
"""
import unittest

import numpy as np

BAR = 15 * 60 * 1000
T0 = 1_700_000_000_000 - (1_700_000_000_000 % BAR)


def bars(closes, t0=T0):
    return {"t": np.array([t0 + i * BAR for i in range(len(closes))], dtype=np.int64), "c": np.array(closes, dtype=float)}


class TestG1(unittest.TestCase):
    def setUp(self):
        self.t_fund = T0 + 10 * BAR                      # a funding event at the open of bar 10

    def test_short_when_rate_positive_and_decomposition(self):
        import funding_premium as F
        c = [100.0] * 20
        c[8] = 100.0                                     # bar 8 closes at t_fund - 15 min: entry
        c[9] = 101.0
        c[10] = 99.0                                     # bar 10 closes at t_fund + 15 min: exit
        r = F.g1_trade(bars(c), self.t_fund, 0.002, 0.001)
        self.assertAlmostEqual(r["price_pct"], 1.0)      # short gains 1%
        self.assertAlmostEqual(r["rate_pct"], 0.2)
        self.assertAlmostEqual(r["cost_pct"], 0.2)
        self.assertAlmostEqual(r["ret_pct"], 1.0 + 0.2 - 0.2)
        self.assertEqual(r["entry_close_ms"], int(T0 + 9 * BAR))

    def test_long_when_rate_negative(self):
        import funding_premium as F
        c = [100.0] * 20
        c[10] = 99.0
        r = F.g1_trade(bars(c), self.t_fund, -0.002, 0.0)
        self.assertAlmostEqual(r["price_pct"], -1.0)     # long loses 1%
        self.assertAlmostEqual(r["rate_pct"], 0.2)       # receives |rate|
        self.assertAlmostEqual(r["ret_pct"], -0.8)

    def test_floor_and_missing_bars(self):
        import funding_premium as F
        c = [100.0] * 20
        self.assertIsNotNone(F.g1_trade(bars(c), self.t_fund + 5, 0.002, 0.0))        # 5 ms after the boundary
        self.assertIsNone(F.g1_trade(bars(c[:10]), self.t_fund, 0.002, 0.0))          # no exit bar
        gap = bars(c)
        gap = {k: np.delete(v, 8) for k, v in gap.items()}                            # no entry bar
        self.assertIsNone(F.g1_trade(gap, self.t_fund, 0.002, 0.0))

    def test_causal_row(self):
        import funding_premium as F
        self.assertTrue(F.causal_ok(0.002, 0.0015, 0.001))
        self.assertFalse(F.causal_ok(0.002, 0.0005, 0.001))           # the previous rate is below the threshold
        self.assertFalse(F.causal_ok(0.002, -0.002, 0.001))           # the sign flipped
        self.assertFalse(F.causal_ok(0.002, None, 0.001))
        self.assertFalse(F.causal_ok(0.0005, 0.002, 0.001))


class TestG2(unittest.TestCase):
    def run_g2(self, perp, spot, mult=1.0, funding=None, cost=0.0, excluded=0, thr=0.004):
        import funding_premium as F
        return F.g2_trades(bars(perp), bars(spot), mult, lambda i: thr, funding, lambda i: cost, excluded)

    def test_entry_after_signal_and_exit_by_threshold(self):
        spot = [100.0] * 20
        perp = [100.0] * 20
        perp[5] = 101.0                                  # premium 1% at bar 5: the signal
        perp[6] = 100.8                                  # entry at bar 6's close
        perp[7] = 100.5
        perp[8] = 100.1                                  # premium 0.1% < 0.4/3: exit signal at bar 8
        perp[9] = 100.0                                  # exit at bar 9's close
        t = self.run_g2(perp, spot)
        self.assertEqual(len(t), 1)
        self.assertEqual(t[0]["entry_close_ms"], int(T0 + 7 * BAR))
        self.assertEqual(t[0]["reason"], "converged")
        self.assertEqual(t[0]["bars"], 3)
        self.assertAlmostEqual(t[0]["price_pct"], (100.8 - 100.0) / 100.8 * 100, places=6)

    def test_time_exit(self):
        spot = [100.0] * 60
        perp = [101.0] * 60                              # the premium never closes
        t = self.run_g2(perp, spot)
        self.assertEqual(t[0]["reason"], "time")
        self.assertEqual(t[0]["bars"], 32)

    def test_multiplier(self):
        spot = [0.001] * 20
        perp = [1.0] * 20                                # 1000 contract: no premium at all
        self.assertEqual(self.run_g2(perp, spot, mult=0.001), [])
        perp2 = [1.0] * 20
        perp2[5] = 1.01
        self.assertEqual(len(self.run_g2(perp2, spot, mult=0.001)), 1)

    def test_funding_inside_the_hold_is_received(self):
        spot = [100.0] * 20
        perp = [100.0] * 20
        perp[5] = 101.0
        perp[6] = 101.0
        entry_close = T0 + 7 * BAR
        f = {"t": np.array([entry_close - 1, entry_close + 1], dtype=np.int64), "rate": np.array([0.05, 0.001])}
        t = self.run_g2(perp, spot, funding=f)
        self.assertAlmostEqual(t[0]["funding_pct"], 0.1)

    def test_one_position_per_coin_and_costs(self):
        spot = [100.0] * 80
        perp = [101.0] * 80                              # the signal stays on
        t = self.run_g2(perp, spot, cost=0.003)
        self.assertEqual(t[0]["entry_close_ms"] + 32 * BAR, t[0]["entry_close_ms"] + t[0]["bars"] * BAR)
        self.assertGreater(t[1]["entry_close_ms"], t[0]["entry_close_ms"] + 31 * BAR)   # no overlap
        self.assertAlmostEqual(t[0]["cost_pct"], 0.3)
        self.assertAlmostEqual(t[0]["ret_pct"], t[0]["price_pct"] + t[0]["funding_pct"] - 0.3)

    def test_listing_days_excluded_and_no_threshold(self):
        import funding_premium as F
        spot = [100.0] * 20
        perp = [100.0] * 20
        perp[5] = 101.0
        self.assertEqual(self.run_g2(perp, spot, excluded=T0 + 10 * BAR), [])
        self.assertEqual(F.g2_trades(bars(perp), bars(spot), 1.0, lambda i: None, None, lambda i: 0.0, 0), [])

    def test_bars_are_matched_by_time(self):
        import funding_premium as F
        perp = bars([100.0] * 25)
        spot = bars([100.0] * 25, t0=T0 - 5 * BAR)       # the spot series starts 5 bars earlier
        spot["c"][10] = 99.0                             # spot bar 10 is perp bar 5: premium +1%
        t = F.g2_trades(perp, spot, 1.0, lambda i: 0.004, None, lambda i: 0.0, 0)
        self.assertEqual(len(t), 1)
        self.assertEqual(t[0]["entry_close_ms"], int(T0 + 7 * BAR))


if __name__ == "__main__":
    unittest.main(verbosity=2)
