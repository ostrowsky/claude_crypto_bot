"""G3a COIN-M carry (coinm-carry-g3a-1105-spec.md §5). Written BEFORE coinm.py. Synthetic, offline.

Interface the tests fix:
  coinm.Fees(spot, perp) cost per side as fractions
  coinm.simulate_pair(marks, funding, fees, capital=1.0) -> {"series": [(t_ms, equity_usd)], "funding_usd", "costs_usd",
       "min_margin_ratio", "rebalances", "n0"}
       marks: [(t_ms, perp_price, spot_price)] one a day; funding: [(t_ms, rate, perp_price)]
  coinm.parse_cm_klines(text) / parse_cm_funding(text)  (archive_bh parsers on the COIN-M file layout)
"""
import unittest

DAY = 86_400_000


def marks(prices, spot=None):
    spot = spot if spot is not None else prices
    return [(i * DAY, float(f), float(s)) for i, (f, s) in enumerate(zip(prices, spot))]


class TestNeutral(unittest.TestCase):
    def test_usd_value_does_not_depend_on_the_coin_price(self):
        import coinm as C
        for path in ([100, 150, 75, 50, 200], [100, 50, 25, 400, 100]):
            r = C.simulate_pair(marks(path), [], C.Fees(0.0, 0.0))
            for _, e in r["series"]:
                self.assertAlmostEqual(e, 1.0, places=9)
            self.assertEqual(r["rebalances"], 0)

    def test_notional_equals_the_capital(self):
        import coinm as C
        r = C.simulate_pair(marks([100, 100]), [], C.Fees(0.0, 0.0), capital=10.0)
        self.assertAlmostEqual(r["n0"], 10.0)

    def test_basis_change_is_marked(self):
        import coinm as C
        r = C.simulate_pair(marks([100, 101, 100], spot=[100, 100, 100]), [], C.Fees(0.0, 0.0))
        e = [v for _, v in r["series"]]
        self.assertAlmostEqual(e[1], 1 + 1.0 * (1 / 101 - 1 / 100) * 100)     # short lost as the perp rose over spot
        self.assertAlmostEqual(e[2], 1.0)                                     # back to flat


class TestFunding(unittest.TestCase):
    def test_short_receives_at_positive_rate_pays_at_negative(self):
        import coinm as C
        m = marks([100, 100, 100])
        up = C.simulate_pair(m, [(DAY + 1, 0.001, 100.0)], C.Fees(0.0, 0.0))
        dn = C.simulate_pair(m, [(DAY + 1, -0.001, 100.0)], C.Fees(0.0, 0.0))
        self.assertAlmostEqual(up["series"][-1][1], 1.001)
        self.assertAlmostEqual(dn["series"][-1][1], 0.999)
        self.assertAlmostEqual(up["funding_usd"], 0.001)

    def test_event_outside_the_window_is_ignored(self):
        import coinm as C
        m = marks([100, 100])
        r = C.simulate_pair(m, [(0, 0.01, 100.0), (2 * DAY, 0.01, 100.0)], C.Fees(0.0, 0.0))
        self.assertAlmostEqual(r["series"][-1][1], 1.0)                       # at the entry mark and after the end


class TestRehedge(unittest.TestCase):
    def test_only_above_the_band(self):
        import coinm as C
        m = marks([100, 100, 100, 100])
        small = C.simulate_pair(m, [(DAY + 1, 0.004, 100.0)], C.Fees(0.0, 0.0))
        self.assertEqual(small["rebalances"], 0)
        big = C.simulate_pair(m, [(DAY + 1, 0.006, 100.0)], C.Fees(0.0, 0.0))
        self.assertEqual(big["rebalances"], 1)

    def test_cost_only_on_the_change(self):
        import coinm as C
        m = marks([100, 100, 100, 100])                       # capital 1, flat 100, perp cost 1%, funding 1%
        r = C.simulate_pair(m, [(DAY + 1, 0.01, 100.0)], C.Fees(0.0, 0.01))
        self.assertEqual(r["rebalances"], 1)
        self.assertAlmostEqual(r["n0"], 0.99)                 # the entry cost is paid out of the coin
        entry = 0.01
        rehedge = (0.9999 - 0.99) * 0.01                      # N goes 0.99 -> 0.9999, only the change pays
        exit_ = 0.9999 * 0.01
        self.assertAlmostEqual(r["costs_usd"], entry + rehedge + exit_, places=9)


class TestCosts(unittest.TestCase):
    def test_entry_and_exit_pay_both_legs(self):
        import coinm as C
        r = C.simulate_pair(marks([100, 100]), [], C.Fees(0.002, 0.001))
        e0 = 1.0 * (1 - 0.002) * (1 - 0.001)
        self.assertAlmostEqual(r["series"][0][1], e0)
        self.assertAlmostEqual(r["series"][-1][1], e0 * (1 - 0.002) * (1 - 0.001))
        self.assertAlmostEqual(r["costs_usd"], 1.0 - r["series"][-1][1])

    def test_money_is_conserved_at_a_flat_price(self):
        import coinm as C
        r = C.simulate_pair(marks([100, 100, 100, 100]), [(DAY + 1, 0.01, 100.0), (2 * DAY + 1, -0.002, 100.0)], C.Fees(0.001, 0.001))
        self.assertAlmostEqual(r["series"][-1][1], 1.0 + r["funding_usd"] - r["costs_usd"], places=9)


class TestMargin(unittest.TestCase):
    def test_margin_ratio_never_thin_at_lev_one(self):
        import coinm as C
        r = C.simulate_pair(marks([100, 1000, 10000]), [], C.Fees(0.0, 0.0))
        self.assertGreaterEqual(r["min_margin_ratio"], 0.99)


class TestParsers(unittest.TestCase):
    def test_cm_files(self):
        import coinm as C
        k = C.parse_cm_klines("open_time,open,high,low,close,volume,close_time,base_volume,count,taker_buy_volume,taker_buy_base_volume,ignore\n"
                              "1700000000000,1,2,0.5,1.5,10,1700000899999,3,4,5,6,0\n")
        self.assertEqual(k["t"].tolist(), [1700000000000])
        self.assertAlmostEqual(float(k["c"][0]), 1.5)
        f = C.parse_cm_funding("calc_time,funding_interval_hours,last_funding_rate\n1700000000000,8,0.0001\n")
        self.assertEqual(f["t"].tolist(), [1700000000000])
        self.assertAlmostEqual(float(f["rate"][0]), 0.0001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
