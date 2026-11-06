"""N1-D carry with the coin as collateral (n1d-1106-spec.md §2). Written BEFORE n1d_carry.py. Synthetic, offline.

Interface the tests fix:
  n1d_carry.Fees(spot, perp, exchange)      cost per side as fractions; exchange = the auto-exchange cost
  n1d_carry.simulate_pair(marks, funding, fees, capital=1.0, buffer=0.02, haircut=0.95, mmr=0.01) ->
      {"series": [(t, equity)], "funding_usd", "costs_usd", "exchanges": [{"t", "usd", "share"}], "min_margin_ratio", "n_coin", "n_short"}
      marks: [(t, perp, spot, fx, high)] one a day; funding: [(t, rate, perp_px, fx)]
"""
import unittest

DAY = 86_400_000


def marks(perp, spot=None, fx=None, high=None):
    spot = spot if spot is not None else perp
    fx = fx if fx is not None else [1.0] * len(perp)
    high = high if high is not None else perp
    return [(i * DAY, float(p), float(s), float(f), float(h)) for i, (p, s, f, h) in enumerate(zip(perp, spot, fx, high))]


def fees(spot=0.0, perp=0.0, exchange=0.0):
    import n1d_carry as N
    return N.Fees(spot, perp, exchange)


class TestNeutral(unittest.TestCase):
    def test_value_does_not_depend_on_the_price(self):
        import n1d_carry as N
        for path in ([100, 150, 50, 100], [100, 50, 150, 75], [100, 150, 150, 150]):
            r = N.simulate_pair(marks(path), [], fees())
            for _, e in r["series"]:
                self.assertAlmostEqual(e, 1.0, places=9)
            self.assertEqual(r["exchanges"], [])

    def test_hedge_by_quantity(self):
        import n1d_carry as N
        r = N.simulate_pair(marks([100, 200]), [], fees(spot=0.001))
        self.assertAlmostEqual(r["n_coin"], r["n_short"])
        self.assertAlmostEqual(r["n_coin"], 0.98 * 0.999 / 100)

    def test_basis_is_marked(self):
        import n1d_carry as N
        r = N.simulate_pair(marks([100, 101, 100], spot=[100, 100, 100]), [], fees())
        e = [v for _, v in r["series"]]
        self.assertAlmostEqual(e[1], 1 - 0.0098)                       # the short lost 1 per coin, 0.0098 coins
        self.assertAlmostEqual(e[2], 1.0)


class TestHaircut(unittest.TestCase):
    def test_only_the_margin_room_changes(self):
        import n1d_carry as N
        m = marks([100, 300, 100], high=[100, 400, 100])
        a = N.simulate_pair(m, [(DAY + 1, 0.001, 100.0, 1.0)], fees(0.001, 0.001), haircut=0.95)
        b = N.simulate_pair(m, [(DAY + 1, 0.001, 100.0, 1.0)], fees(0.001, 0.001), haircut=0.50)
        self.assertEqual(a["series"], b["series"])
        self.assertGreater(a["min_margin_ratio"], b["min_margin_ratio"])

    def test_margin_ratio_formula(self):
        import n1d_carry as N
        r = N.simulate_pair(marks([100, 100]), [], fees(), haircut=0.95, mmr=0.01)
        q = 0.98 / 100
        self.assertAlmostEqual(r["min_margin_ratio"], (0.95 * q * 100 + 0.02) / (0.01 * q * 100))

    def test_margin_uses_the_day_high(self):
        import n1d_carry as N
        low = N.simulate_pair(marks([100, 100, 100]), [], fees())["min_margin_ratio"]
        spike = N.simulate_pair(marks([100, 100, 100], high=[100, 1200, 100]), [], fees())["min_margin_ratio"]
        self.assertLess(spike, low / 10)


class TestFunding(unittest.TestCase):
    def test_short_receives_a_positive_rate_in_usdt(self):
        import n1d_carry as N
        r = N.simulate_pair(marks([100, 100, 100]), [(DAY + 1, 0.001, 100.0, 1.0)], fees())
        self.assertAlmostEqual(r["funding_usd"], 0.98 * 0.001)
        self.assertAlmostEqual(r["series"][-1][1], 1 + 0.98 * 0.001)

    def test_pays_a_negative_rate(self):
        import n1d_carry as N
        r = N.simulate_pair(marks([100, 100, 100]), [(DAY + 1, -0.001, 100.0, 1.0)], fees())
        self.assertLess(r["funding_usd"], 0)

    def test_not_reinvested(self):
        import n1d_carry as N
        r = N.simulate_pair(marks([100, 100, 100, 100]), [(DAY + 1, 0.001, 100.0, 1.0), (2 * DAY + 1, 0.001, 100.0, 1.0)], fees())
        self.assertAlmostEqual(r["funding_usd"], 2 * 0.98 * 0.001)
        self.assertAlmostEqual(r["n_coin"], 0.98 / 100)

    def test_events_outside_the_marks_are_ignored(self):
        import n1d_carry as N
        r = N.simulate_pair(marks([100, 100]), [(0, 0.01, 100.0, 1.0), (DAY + 5, 0.01, 100.0, 1.0)], fees())
        self.assertAlmostEqual(r["funding_usd"], 0.0)


class TestExchange(unittest.TestCase):
    def test_only_when_usdt_is_negative(self):
        import n1d_carry as N
        r = N.simulate_pair(marks([100, 100, 100]), [(DAY + 1, -0.001, 100.0, 1.0)], fees())
        self.assertEqual(r["exchanges"], [])                           # 2% buffer covers it

    def test_pays_the_given_costs_and_restores_zero(self):
        import n1d_carry as N
        # buffer 2%; funding -3% of 0.98 = -2.94%: the deficit is 0.94%
        ev = [(DAY + 1, -0.03, 100.0, 1.0)]
        r = N.simulate_pair(marks([100, 100, 100]), ev, fees(exchange=0.002))
        self.assertEqual(len(r["exchanges"]), 1)
        x = r["exchanges"][0]
        deficit = 0.98 * 0.03 - 0.02
        self.assertAlmostEqual(x["usd"], deficit / (1 - 0.002))
        self.assertAlmostEqual(x["share"], x["usd"] / 1.0)
        self.assertAlmostEqual(r["n_coin"], 0.98 / 100 - x["usd"] / 100)
        self.assertAlmostEqual(r["n_short"], 0.98 / 100)
        self.assertAlmostEqual(r["costs_usd"], x["usd"] * 0.002)

    def test_exchange_leaves_a_long_gap_marked(self):
        import n1d_carry as N
        ev = [(DAY + 1, -0.03, 100.0, 1.0)]
        flat = N.simulate_pair(marks([100, 100, 100]), ev, fees())
        up = N.simulate_pair(marks([100, 100, 200]), ev, fees())
        self.assertLess(up["series"][-1][1], flat["series"][-1][1])    # less coin than short: a price rise hurts


class TestMoney(unittest.TestCase):
    def test_flat_price_identity_with_every_cost(self):
        import n1d_carry as N
        ev = [(DAY + 1, 0.001, 100.0, 1.0), (2 * DAY + 1, -0.03, 100.0, 1.0)]
        r = N.simulate_pair(marks([100, 100, 100, 100]), ev, fees(0.002, 0.001, 0.003))
        self.assertTrue(r["exchanges"])
        self.assertAlmostEqual(r["series"][-1][1], 1 + r["funding_usd"] - r["costs_usd"], places=10)

    def test_with_a_basis_change(self):
        import n1d_carry as N
        r = N.simulate_pair(marks([100, 103], spot=[100, 101]), [(DAY - 1, 0.001, 101.0, 1.0)], fees(0.002, 0.001))
        q = r["n_coin"]
        basis = q * (101 - 100) + q * (100 - 103)
        self.assertAlmostEqual(r["series"][-1][1], 1 + r["funding_usd"] + basis - r["costs_usd"], places=10)

    def test_entry_and_exit_costs(self):
        import n1d_carry as N
        r = N.simulate_pair(marks([100, 100]), [], fees(0.002, 0.001))
        q = 0.98 * (1 - 0.002) / 100
        want = 0.98 * 0.002 + q * 100 * 0.001 + q * 100 * 0.002 + q * 100 * 0.001
        self.assertAlmostEqual(r["costs_usd"], want)


class TestUSDC(unittest.TestCase):
    def test_a_usdc_rise_is_a_gain_on_the_short_when_the_coin_holds_in_usdt(self):
        import n1d_carry as N
        # the coin stays at 100 USDT, USDC goes to 1.01 USDT: the perp in USDC falls to 100 / 1.01, the short gains in USDC
        r = N.simulate_pair(marks([100.0, 100.0 / 1.01], spot=[100.0, 100.0], fx=[1.0, 1.01]), [], fees())
        q = r["n_coin"]
        self.assertAlmostEqual(r["series"][-1][1], 1.0 + q * (100.0 - 100.0 / 1.01) * 1.01, places=10)

    def test_funding_is_converted(self):
        import n1d_carry as N
        r = N.simulate_pair(marks([100, 100, 100]), [(DAY + 1, 0.001, 100.0, 0.99)], fees())
        self.assertAlmostEqual(r["funding_usd"], 0.98 * 0.001 * 0.99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
