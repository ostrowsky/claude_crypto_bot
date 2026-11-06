"""N5 implied against realized volatility (n5-1106-spec.md §4). Written BEFORE n5_vrp.py. Synthetic, offline.

Interface the tests fix:
  n5_vrp.first_minute(text, day_ms) -> float or None          BVOL file: value at [00:00:00, 00:01:00)
  n5_vrp.rv30(t, c, start_ms) -> float or None                percent, annualised on 365 days
  n5_vrp.newey_west(x, lag) -> {"mean", "se", "n"}
  n5_vrp.leland_points(sigma, cost_side=0.0005) -> float
  n5_vrp.expiry_days(symbol, day) -> int                      BTC-230929-40000-P on 2023-08-30 -> 30
  n5_vrp.spread_rows(text) -> [(day, hour, symbol, delta, buy_iv, sell_iv)]
  n5_vrp.day_spread(rows) -> median width in vol points, or None
"""
import math
import unittest

import numpy as np

BAR = 900_000
DAY = 86_400_000


class TestIV(unittest.TestCase):
    H = "calc_time,symbol,base_asset,quote_asset,index_value\n"

    def test_first_minute(self):
        import n5_vrp as N
        txt = self.H + f"{DAY},BTCBVOLUSDT,BTCBVOL,USDT,55.5\n{DAY + 1000},BTCBVOLUSDT,BTCBVOL,USDT,99.0\n"
        self.assertEqual(N.first_minute(txt, DAY), 55.5)

    def test_later_record_is_a_missing_day(self):
        import n5_vrp as N
        self.assertIsNone(N.first_minute(self.H + f"{DAY + 60_000},B,B,U,55.5\n", DAY))
        self.assertEqual(N.first_minute(self.H + f"{DAY + 59_999},B,B,U,55.5\n", DAY), 55.5)

    def test_microseconds(self):
        import n5_vrp as N
        d = 1739836800000
        self.assertEqual(N.first_minute(self.H + f"{d * 1000 + 5},B,B,U,41.0\n", d), 41.0)

    def test_empty(self):
        import n5_vrp as N
        self.assertIsNone(N.first_minute(self.H, DAY))


class TestRV(unittest.TestCase):
    def _path(self, n, r=0.001, t0=0):
        return t0 + np.arange(n + 1, dtype=np.int64) * BAR, 100 * np.exp(np.arange(n + 1) * r)

    def test_constant_returns(self):
        import n5_vrp as N
        t, c = self._path(2880)
        want = 100 * math.sqrt(365 / 30 * 2880 * 0.001 ** 2)
        self.assertAlmostEqual(N.rv30(t, c, 0), want, places=6)

    def test_nothing_before_start_or_after_30_days(self):
        import n5_vrp as N
        t, c = self._path(2880, t0=0)
        base = N.rv30(t, c, 0)
        t2 = np.r_[-2 * BAR, -BAR, t, t[-1] + BAR]
        c2 = np.r_[1.0, 500.0, c, c[-1] * 10]
        self.assertAlmostEqual(N.rv30(t2, c2, 0), base, places=9)

    def test_needs_95_percent_of_intervals(self):
        import n5_vrp as N
        t, c = self._path(2880)
        k = np.ones(len(t), dtype=bool)
        k[1000:1200] = False                                           # 201 intervals lost of 2880: 93%
        self.assertIsNone(N.rv30(t[k], c[k], 0))
        k = np.ones(len(t), dtype=bool)
        k[1000:1100] = False                                           # 101 lost: 96.5%
        self.assertIsNotNone(N.rv30(t[k], c[k], 0))

    def test_jump_across_a_gap_is_not_counted(self):
        import n5_vrp as N
        t, c = self._path(2880, r=0.0)
        c = c.copy()
        c[1100:] *= 2.0                                                # the move happens inside the gap
        k = np.ones(len(t), dtype=bool)
        k[1000:1100] = False
        self.assertAlmostEqual(N.rv30(t[k], c[k], 0), 0.0)


class TestNeweyWest(unittest.TestCase):
    def test_lag_zero_is_the_plain_se(self):
        import n5_vrp as N
        x = np.random.default_rng(1).normal(size=500)
        r = N.newey_west(x, 0)
        self.assertAlmostEqual(r["se"], float(np.sqrt(((x - x.mean()) ** 2).mean() / len(x))))
        self.assertAlmostEqual(r["mean"], float(x.mean()))

    def test_independent_noise_stays_close(self):
        import n5_vrp as N
        x = np.random.default_rng(2).normal(size=4000)
        self.assertAlmostEqual(N.newey_west(x, 30)["se"] / N.newey_west(x, 0)["se"], 1.0, delta=0.15)

    def test_overlap_widens(self):
        import n5_vrp as N
        e = np.random.default_rng(3).normal(size=4000)
        x = np.convolve(e, np.ones(30) / 30, mode="valid")
        self.assertGreater(N.newey_west(x, 30)["se"], 3 * N.newey_west(x, 0)["se"])


class TestCosts(unittest.TestCase):
    def test_leland_formula(self):
        import n5_vrp as N
        sigma = 0.5
        le = math.sqrt(2 / math.pi) * 0.001 / (sigma * math.sqrt(1 / 365))
        self.assertAlmostEqual(N.leland_points(sigma), 100 * sigma * (math.sqrt(1 + le) - 1))

    def test_no_cost_no_addition(self):
        import n5_vrp as N
        self.assertEqual(N.leland_points(0.5, cost_side=0.0), 0.0)

    def test_expiry(self):
        import n5_vrp as N
        self.assertEqual(N.expiry_days("BTC-230929-40000-P", "2023-08-30"), 30)
        self.assertEqual(N.expiry_days("ETH-231027-2000-C", "2023-10-20"), 7)

    HEAD = "date,hour,symbol,underlying,type,strike,open,high,low,close,volume_contracts,volume_usdt,best_bid_price,best_ask_price,best_bid_qty,best_ask_qty,best_buy_iv,best_sell_iv,mark_price,mark_iv,delta,gamma,vega,theta,openinterest_contracts,openinterest_usdt\n"

    def _row(self, day, hour, sym, buy, sell, delta):
        return f"{day},{hour},{sym},BTCUSDT,P,x,0,0,0,0,0,0,1,2,1,1,{buy},{sell},1,0.5,{delta},0,0,0,0,0\n"

    def test_filter(self):
        import n5_vrp as N
        d = "2023-08-30"
        good = "BTC-230929-40000-P"
        rows = N.spread_rows(self.HEAD
                             + self._row(d, "00", good, "0.50", "0.56", "-0.50")
                             + self._row(d, "12", good, "0.50", "0.54", "0.45")
                             + self._row(d, "06", good, "0.50", "0.99", "-0.50")          # hour
                             + self._row(d, "00", good, '""', "0.56", "-0.50")            # empty IV
                             + self._row(d, "00", good, "0E-8", "0.56", "-0.50")          # zero IV
                             + self._row(d, "00", good, "0.50", "0.99", "-0.70")          # delta
                             + self._row(d, "00", "BTC-230908-40000-P", "0.50", "0.99", "-0.50"))     # 9 days
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(N.day_spread(rows), 5.0)                # median of 6 and 4 vol points

    def test_no_rows(self):
        import n5_vrp as N
        self.assertIsNone(N.day_spread([]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
