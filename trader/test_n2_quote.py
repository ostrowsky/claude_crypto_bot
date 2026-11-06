"""N2 quoting on a USDC perp hedged on the USDT perp (n2-1106-spec.md §2). Written BEFORE n2_quote.py. Synthetic, offline.

Interface the tests fix:
  n2_quote.parse_agg(text) -> {"t", "p", "q", "m"}          sorted by time, duplicates of agg id removed, ms
  n2_quote.schedule(t_u, p_u, band=0.5e-4, refresh=1000, min_gap=200) -> (d, F)
  n2_quote.Cfg(theta, l_q, l_h, h, tick, q=1000.0, g=20000.0, maker=0.0, taker_usdt=0.0005, taker_usdc=0.0004, slip=0.0, strict=True)
  n2_quote.quotes(F, cfg) -> (bid, ask)
  n2_quote.simulate(usdc, usdt, cfg, fx=None, funding_c=None, funding_u=None, sched=None) ->
      {"fills": [{"t", "s", "x", "notional", "H", "F_quote", "fx"}], "total", "spread", "fees", "slip", "funding", "fx", "unwind",
       "days": {day: {"pnl", "notional", "n"}}, "w_usdc", "w_usdt"}
"""
import unittest

import numpy as np

DAY = 86_400_000


def tape(rows):
    """rows: (t, price, qty, buyer_is_maker)."""
    a = np.array(rows, dtype=float)
    return {"t": a[:, 0].astype(np.int64), "p": a[:, 1], "q": a[:, 2], "m": a[:, 3].astype(bool)}


def cfg(**k):
    import n2_quote as N
    base = dict(theta=1e-4, l_q=100, l_h=100, h=5e-4, tick=0.01, maker=0.0, taker_usdt=0.0, taker_usdc=0.0, slip=0.0)
    base.update(k)
    return N.Cfg(**base)


UT = tape([(0, 100.0, 1, 0), (10_000, 100.0, 1, 0), (20_000, 100.0, 1, 0), (30_000, 100.0, 1, 0)])


class TestParser(unittest.TestCase):
    def test_units_duplicates_order(self):
        import n2_quote as N
        a = N.parse_agg("agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
                        "2,101.5,3,1,1,1700000001000,false\n"
                        "1,101.0,2,1,1,1700000000000000,true\n"
                        "2,101.5,3,1,1,1700000001000,false\n")
        self.assertEqual(a["t"].tolist(), [1700000000000, 1700000001000])
        self.assertEqual(a["p"].tolist(), [101.0, 101.5])
        self.assertEqual(a["q"].tolist(), [2.0, 3.0])
        self.assertEqual(a["m"].tolist(), [True, False])

    def test_no_header(self):
        import n2_quote as N
        a = N.parse_agg("1,101.0,2,1,1,1700000000000,True\n")
        self.assertEqual(a["t"].tolist(), [1700000000000])
        self.assertEqual(a["m"].tolist(), [True])


class TestSchedule(unittest.TestCase):
    def test_refresh_each_second_on_a_flat_tape(self):
        import n2_quote as N
        t = np.arange(0, 5000, 100, dtype=np.int64)
        d, F = N.schedule(t, np.full(len(t), 100.0))
        self.assertEqual(d.tolist(), [0, 1000, 2000, 3000, 4000])

    def test_band_triggers_after_the_minimum_gap(self):
        import n2_quote as N
        t = np.array([0, 50, 150, 250, 400], dtype=np.int64)
        p = np.array([100.0, 101.0, 102.0, 103.0, 103.0])
        d, F = N.schedule(t, p)
        self.assertEqual(d.tolist()[:2], [0, 250])                     # 50 and 150 are inside the 200 ms limit
        self.assertEqual(F.tolist()[:2], [100.0, 103.0])

    def test_rate_limit_holds_on_a_wild_tape(self):
        import n2_quote as N
        t = np.arange(0, 2000, 10, dtype=np.int64)
        p = 100 + np.arange(len(t)) * 0.5
        d, _ = N.schedule(t, p)
        self.assertGreaterEqual(int(np.diff(d).min()), 200)
        self.assertLessEqual(int(((d >= 0) & (d < 1000)).sum()), 5)

    def test_small_move_waits_for_the_second(self):
        import n2_quote as N
        t = np.array([0, 300, 900, 1000], dtype=np.int64)
        d, _ = N.schedule(t, np.array([100.0, 100.0001, 100.0002, 100.0002]))
        self.assertEqual(d.tolist(), [0, 1000])


class TestQuotes(unittest.TestCase):
    def test_tick_rounds_against_us(self):
        import n2_quote as N
        c = cfg(theta=1e-4, h=5e-4, tick=0.1)
        b, a = N.quotes(np.array([100.03]), c)
        self.assertAlmostEqual(float(b[0]), 99.9)                       # 99.96998 down
        self.assertAlmostEqual(float(a[0]), 100.1)                      # 100.09 up

    def test_exact_tick_is_not_moved(self):
        import n2_quote as N
        b, a = N.quotes(np.array([100.0]), cfg(theta=0.0, h=1e-3, tick=0.1))
        self.assertAlmostEqual(float(b[0]), 99.9)
        self.assertAlmostEqual(float(a[0]), 100.1)


class TestFillRule(unittest.TestCase):
    def test_strictly_through_with_the_right_flag(self):
        import n2_quote as N
        # bid = floor(100 * (1 - 6e-4)) = 99.94
        cases = [(99.93, 1, True),   # seller taker below the bid
                 (99.94, 1, False),  # at the bid: not through
                 (99.93, 0, False),  # buyer taker: not ours
                 ]
        for px, m, want in cases:
            u = tape([(1000, px, 100, m)])
            r = N.simulate(u, UT, cfg())
            self.assertEqual(len(r["fills"]) == 1, want, (px, m))

    def test_ask_mirror(self):
        import n2_quote as N
        for px, m, want in [(100.07, 0, True), (100.06, 0, False), (100.07, 1, False)]:
            r = N.simulate(tape([(1000, px, 100, m)]), UT, cfg())
            self.assertEqual(len(r["fills"]) == 1, want, (px, m))

    def test_fill_price_is_the_limit_and_size_is_capped(self):
        import n2_quote as N
        r = N.simulate(tape([(1000, 99.0, 100, 1)]), UT, cfg())
        f = r["fills"][0]
        self.assertAlmostEqual(f["x"], 99.94)
        self.assertAlmostEqual(f["notional"], 1000.0)
        r = N.simulate(tape([(1000, 99.0, 0.5, 1)]), UT, cfg())
        self.assertAlmostEqual(r["fills"][0]["notional"], 99.0 * 0.5)

    def test_equal_price_sensitivity(self):
        import n2_quote as N
        r = N.simulate(tape([(1000, 99.94, 100, 1)]), UT, cfg(strict=False))
        self.assertEqual(len(r["fills"]), 1)


class TestLatency(unittest.TestCase):
    def test_quote_uses_nothing_later_than_t_minus_lq(self):
        import n2_quote as N
        # the USDT perp falls to 90 at 1000; at 1050 the live quote still comes from 100 with l_q = 100
        ut = tape([(0, 100.0, 1, 0), (1000, 90.0, 1, 0), (5000, 90.0, 1, 0)])
        r = N.simulate(tape([(1050, 99.5, 100, 1)]), ut, cfg())
        self.assertEqual(len(r["fills"]), 1)
        self.assertEqual(r["fills"][0]["F_quote"], 100.0)
        # at 1200 the decision of 1000 is live: F = 90, the bid is near 89.9, and 99.5 is no fill
        r = N.simulate(tape([(1200, 99.5, 100, 1)]), ut, cfg())
        self.assertEqual(len(r["fills"]), 0)

    def test_hedge_is_the_first_trade_not_before_t_plus_lh(self):
        import n2_quote as N
        ut = tape([(0, 100.0, 1, 0), (1000, 100.0, 1, 0), (1099, 50.0, 1, 0), (1100, 101.0, 1, 0), (1200, 102.0, 1, 0)])
        r = N.simulate(tape([(1000, 99.0, 100, 1)]), ut, cfg(l_q=10))
        self.assertEqual(r["fills"][0]["H"], 101.0)

    def test_side_is_gone_until_replaced(self):
        import n2_quote as N
        u = tape([(1000, 99.0, 100, 1), (1050, 99.0, 100, 1), (1099, 99.0, 100, 1)])
        r = N.simulate(u, UT, cfg())                                   # l_q = 100: back at 1100
        self.assertEqual([f["t"] for f in r["fills"]], [1000])
        u = tape([(1000, 99.0, 100, 1), (1100, 99.0, 100, 1)])
        r = N.simulate(u, UT, cfg())
        self.assertEqual([f["t"] for f in r["fills"]], [1000, 1100])

    def test_replacement_honours_the_rate_limit(self):
        import n2_quote as N
        # schedule effective at 1100 (decision 1000 + 100); a fill at 1101 with l_q = 100 would re-place at 1201 anyway,
        # and with l_q = 10 the 200 ms from the last change decide: back at 1210 (decision 1000 effective 1010)
        ut = tape([(0, 100.0, 1, 0), (1000, 100.0, 1, 0), (5000, 100.0, 1, 0)])
        u = tape([(1011, 99.0, 100, 1), (1100, 99.0, 100, 1), (1209, 99.0, 100, 1), (1210, 99.0, 100, 1)])
        r = N.simulate(u, ut, cfg(l_q=10))
        self.assertEqual([f["t"] for f in r["fills"]], [1011, 1210])


class TestInventory(unittest.TestCase):
    def _sells(self, n, t0=1000, step=2000):
        return [(t0 + i * step, 99.0, 100, 1) for i in range(n)]

    def test_at_the_limit_the_increasing_side_is_off(self):
        import n2_quote as N
        ut = tape([(0, 100.0, 1, 0), (60_000, 100.0, 1, 0)])
        r = N.simulate(tape(self._sells(5)), ut, cfg(g=2000.0))
        self.assertEqual(len(r["fills"]), 2)
        self.assertAlmostEqual(sum(f["notional"] * f["s"] for f in r["fills"]), 2000.0)

    def test_last_fill_is_cut_to_the_room_left(self):
        import n2_quote as N
        ut = tape([(0, 100.0, 1, 0), (60_000, 100.0, 1, 0)])
        r = N.simulate(tape(self._sells(5)), ut, cfg(g=1500.0))
        self.assertEqual([round(f["notional"]) for f in r["fills"]], [1000, 500])

    def test_reducing_side_stays_and_reopens_the_other(self):
        import n2_quote as N
        ut = tape([(0, 100.0, 1, 0), (60_000, 100.0, 1, 0)])
        rows = self._sells(3) + [(8000, 101.0, 100, 0), (9000, 99.0, 100, 1)]
        r = N.simulate(tape(sorted(rows)), ut, cfg(g=1000.0))
        self.assertEqual([f["s"] for f in r["fills"]], [1, -1, 1])


class TestMoney(unittest.TestCase):
    def _run(self, **k):
        import n2_quote as N
        ut = tape([(i * 500, 100.0 + 0.3 * np.sin(i / 7), 1, 0) for i in range(600)])
        rows = []
        for i in range(1, 290):
            rows.append((i * 1000 + 3, 99.0 + 0.1 * np.sin(i), 100, 1) if i % 2 else (i * 1000 + 3, 101.0 + 0.1 * np.sin(i), 100, 0))
        fx = {"t": np.arange(0, 300) * 60_000, "c": 1 + 0.001 * np.sin(np.arange(300) / 3)}
        fc = {"t": np.array([100_000, 200_000]), "rate": np.array([0.001, -0.0005])}
        fu = {"t": np.array([100_000, 200_000]), "rate": np.array([0.0007, 0.0003])}
        return N.simulate(tape(rows), ut, cfg(**k), fx=fx, funding_c=fc, funding_u=fu)

    def test_identity_with_every_cost_on(self):
        r = self._run(maker=0.0001, taker_usdt=0.0005, taker_usdc=0.0004, slip=0.0002)
        self.assertGreater(len(r["fills"]), 10)
        parts = r["spread"] - r["fees"] - r["slip"] + r["funding"] + r["fx"] + r["unwind"]
        self.assertAlmostEqual(r["total"], parts, places=8)

    def test_daily_pnl_adds_up_to_the_total(self):
        r = self._run(maker=0.0001, taker_usdt=0.0005, taker_usdc=0.0004, slip=0.0002)
        self.assertAlmostEqual(sum(d["pnl"] for d in r["days"].values()), r["total"], places=8)

    def test_a_clean_round_trip_earns_the_spread(self):
        import n2_quote as N
        ut = tape([(0, 100.0, 1, 0), (5000, 100.0, 1, 0), (10_000, 100.0, 1, 0)])
        u = tape([(1000, 99.0, 100, 1), (5000, 101.0, 100, 0)])
        r = N.simulate(u, ut, cfg())
        q = 1000 / 99.94
        q2 = 1000 / 100.06
        self.assertEqual(len(r["fills"]), 2)
        self.assertAlmostEqual(r["spread"], q * (100 - 99.94) + q2 * (100.06 - 100))
        self.assertAlmostEqual(r["fees"], 0.0)

    def test_hedge_fee_is_charged_on_the_hedge_leg(self):
        import n2_quote as N
        ut = tape([(0, 100.0, 1, 0), (5000, 100.0, 1, 0), (10_000, 100.0, 1, 0)])
        r = N.simulate(tape([(1000, 99.0, 100, 1), (5000, 101.0, 100, 0)]), ut, cfg(taker_usdt=0.001))
        q = 1000 / 99.94 + 1000 / 100.06
        self.assertAlmostEqual(r["fees"], q * 100 * 0.001)


class TestFxAndFunding(unittest.TestCase):
    def test_fx_has_no_lookahead(self):
        import n2_quote as N
        fx = {"t": np.array([0, 60_000]), "c": np.array([1.0, 2.0])}
        self.assertEqual(N.fx_at(fx, 60_000), 1.0)                     # the first minute has closed, the second has not
        self.assertEqual(N.fx_at(fx, 119_999), 1.0)
        self.assertEqual(N.fx_at(fx, 120_000), 2.0)

    def test_usdc_cash_moves_with_the_rate(self):
        import n2_quote as N
        ut = tape([(0, 100.0, 1, 0), (70_000, 100.0, 1, 0)])
        fx = {"t": np.array([0, 60_000]), "c": np.array([1.0, 1.01])}
        r = N.simulate(tape([(60_000, 99.0, 100, 1), (120_000, 102.0, 100, 0)]), ut, cfg(), fx=fx)
        self.assertNotAlmostEqual(r["fx"], 0.0)
        self.assertAlmostEqual(r["total"], r["spread"] + r["fx"] + r["unwind"], places=8)

    def test_funding_sign_by_leg(self):
        import n2_quote as N
        ut = tape([(0, 100.0, 1, 0), (2000, 100.0, 1, 0), (9000, 100.0, 1, 0)])
        u = tape([(1000, 99.0, 100, 1), (8000, 99.5, 1e-9, 1)])
        fc = {"t": np.array([5000]), "rate": np.array([0.001])}
        fu = {"t": np.array([5000]), "rate": np.array([0.0])}
        r = N.simulate(u, ut, cfg(), funding_c=fc, funding_u=fu)
        self.assertLess(r["funding"], 0)                               # long USDC pays a positive rate
        fc0 = {"t": np.array([5000]), "rate": np.array([0.0])}
        fu1 = {"t": np.array([5000]), "rate": np.array([0.001])}
        r = N.simulate(u, ut, cfg(), funding_c=fc0, funding_u=fu1)
        self.assertGreater(r["funding"], 0)                            # short USDT receives it


if __name__ == "__main__":
    unittest.main(verbosity=2)
