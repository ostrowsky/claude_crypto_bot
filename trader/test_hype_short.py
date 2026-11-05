"""Hype-short family H-A / H-B (hype-short-family-1105-spec.md §6). Written BEFORE hype_short.py. Synthetic daily panels.

Interface the tests fix:
  hype_short.Panel(days, syms, close[D,S], qvol[D,S], first[S], last[S], fund[D,S])     days: ints (day numbers)
  hype_short.universe(panel, i) -> list of column indexes   (age >= 30, top-100 turnover over [i-30, i), >= 50 names)
  hype_short.leaders(panel, i, uni, L, k, blocked) -> columns: top past return close[i-1-L] -> close[i-1]
  hype_short.run_ha(panel, L, costs, seed=None, k=10, start=0, end=None) -> {"series": [(i, R%)], "equity_min", ...}
  hype_short.run_hb(panel, btc, costs, seed=None, start=0, end=None) -> {"series", "entries", ...}
  hype_short.funding_pnl(side, notional, rate) -> short receives rate * notional, long pays it
  hype_short.beta_past(cohort_rets, btc_rets) -> slope on past days only, clipped to [0.5, 2], 1.0 below 30 days
  costs: callable(col, i) -> cost per side as a fraction (taker + slippage)
"""
import unittest

import numpy as np

S_N = 60


def panel(D=120, S=S_N, seed=1, first=None, last=None, drift=None):
    import hype_short as H
    rnd = np.random.default_rng(seed)
    r = rnd.normal(0, 0.02, size=(D, S))
    if drift is not None:
        r += drift
    close = 100 * np.exp(np.cumsum(r, axis=0))
    qvol = np.tile(np.linspace(2e6, 1e6, S), (D, 1))
    f = np.zeros(S, dtype=int) if first is None else np.asarray(first)
    l = np.full(S, D - 1) if last is None else np.asarray(last)
    for j in range(S):
        close[:f[j], j] = np.nan
        close[l[j] + 1:, j] = np.nan
    return H.Panel(list(range(D)), [f"S{j}USDT" for j in range(S)], close, qvol, f, l, np.zeros((D, S)))


def zero_cost(col, i):
    return 0.0


class TestUniverseAndSignal(unittest.TestCase):
    def test_universe_age_and_size(self):
        import hype_short as H
        first = np.zeros(S_N, dtype=int)
        first[:5] = 50                                             # young perps
        p = panel(first=first)
        u = H.universe(p, 60)
        self.assertTrue(all(c >= 5 for c in u))                    # age < 30 excluded
        self.assertEqual(len(u), S_N - 5)
        small = panel(S=40)
        self.assertEqual(H.universe(small, 60), [])                # fewer than 50 names: no positions

    def test_leaders_use_closes_up_to_yesterday(self):
        import hype_short as H
        p = panel()
        i = 70
        p.close[i, 3] = p.close[i - 1, 3] * 5                      # today's jump must not count
        p.close[i - 1, 7] = p.close[i - 2, 7] * 3                  # yesterday's jump does
        u = H.universe(p, i)
        top = H.leaders(p, i, u, 1, 10, blocked=set())
        self.assertIn(7, top)
        self.assertNotIn(3, top[:1])
        self.assertEqual(len(top), 10)
        top2 = H.leaders(p, i, u, 1, 10, blocked={7})
        self.assertNotIn(7, top2)
        self.assertEqual(len(top2), 10)


class TestBook(unittest.TestCase):
    def test_funding_sign(self):
        import hype_short as H
        self.assertAlmostEqual(H.funding_pnl("short", 1000.0, 0.001), 1.0)
        self.assertAlmostEqual(H.funding_pnl("short", 1000.0, -0.001), -1.0)
        self.assertAlmostEqual(H.funding_pnl("long", 1000.0, 0.001), -1.0)

    def test_flat_market_zero_cost_is_flat(self):
        import hype_short as H
        p = panel()
        p.close[:] = 100.0
        r = H.run_ha(p, 7, zero_cost)
        self.assertTrue(all(abs(v) < 1e-9 for _, v in r["series"]))

    def test_costs_charged_and_money_conserved(self):
        import hype_short as H
        p = panel()
        r0 = H.run_ha(p, 7, zero_cost)
        r1 = H.run_ha(p, 7, lambda c, i: 0.001)
        self.assertLess(sum(v for _, v in r1["series"]), sum(v for _, v in r0["series"]))
        self.assertTrue(r1["conservation_ok"])
        self.assertIs(type(r1["conservation_ok"]), bool)        # goes to JSON, not numpy.bool_
        self.assertGreater(r1["equity_min"], 0)

    def test_short_leg_profits_when_leaders_fall(self):
        import hype_short as H
        p = panel()
        p.close[:] = 100.0
        p.close[40:, 50:] = 150.0                                   # ten names jump +50% on day 40 (not cols 0-9:
                                                                    # those win the ties of the flat days before)
        for i in range(42, 61):                                     # ... and slide back to 100 by day 60
            p.close[i, 50:] = 150.0 - 50.0 * (i - 41) / 19
        p.close[61:, 50:] = 100.0
        r = H.run_ha(p, 7, zero_cost)
        self.assertGreater(sum(v for _, v in r["series"]), 0)       # short the leaders, long the flat rest

    def test_name_cap(self):
        import hype_short as H
        self.assertEqual(H.NAME_CAP, 1 / 20)

    def test_control_is_seeded(self):
        import hype_short as H
        p = panel()
        a = H.run_ha(p, 7, zero_cost, seed=5)["series"]
        b = H.run_ha(p, 7, zero_cost, seed=5)["series"]
        self.assertEqual(a, b)


class TestHB(unittest.TestCase):
    def test_entry_exit_and_cap(self):
        import hype_short as H
        D, S = 200, 40
        first = np.array([0] + [10 + 3 * j for j in range(S - 1)])
        p = panel(D=D, S=S, first=first)
        r = H.run_hb(p, 0, zero_cost)
        e = {x["col"]: x for x in r["entries"]}
        self.assertEqual(e[1]["entry_i"], first[1] + 2)
        self.assertEqual(e[1]["exit_i"], first[1] + 60)
        self.assertLessEqual(max(r["short_share_after_entry"]), 0.5 + 1e-9)
        self.assertNotIn(0, e)                                      # BTC (col 0) is the hedge, never shorted

    def test_delisting_exit(self):
        import hype_short as H
        D, S = 200, 5
        first = np.array([0, 20, 20, 20, 20])
        last = np.array([D - 1, 40, D - 1, D - 1, D - 1])
        p = panel(D=D, S=S, first=first, last=last)
        r = H.run_hb(p, 0, zero_cost)
        e = {x["col"]: x for x in r["entries"]}
        self.assertEqual(e[1]["exit_i"], 40)

    def test_beta_past(self):
        import hype_short as H
        rnd = np.random.default_rng(0)
        b = rnd.normal(0, 0.02, 100)
        self.assertAlmostEqual(H.beta_past(list(1.5 * b), list(b)), 1.5, places=6)
        self.assertEqual(H.beta_past(list(5 * b), list(b)), 2.0)
        self.assertEqual(H.beta_past(list(b[:20]), list(b[:20])), 1.0)


class TestFundingByLeg(unittest.TestCase):
    """The book's own funding formula, not the helper (fourth-review-1105-spec.md §3)."""

    def flat(self, D=120, S=S_N):
        p = panel(D=D, S=S)
        p.close[:] = 100.0
        return p

    def test_book_long_pays_short_receives(self):
        import hype_short as H
        p = self.flat()
        p.fund[5, :2] = 0.001
        b = H._Book(p, zero_cost)
        b.pos[("x", 0)] = [0, 2.0, 100.0]
        b.pos[("x", 1)] = [1, -3.0, 100.0]
        b.mark(5)
        self.assertAlmostEqual(b.fund_long, -2.0 * 100 * 0.001)
        self.assertAlmostEqual(b.fund_short, 3.0 * 100 * 0.001)
        self.assertAlmostEqual(b.fund, b.fund_long + b.fund_short)

    def test_negative_rate_flips_both(self):
        import hype_short as H
        p = self.flat()
        p.fund[5, :2] = -0.001
        b = H._Book(p, zero_cost)
        b.pos[("x", 0)] = [0, 2.0, 100.0]
        b.pos[("x", 1)] = [1, -3.0, 100.0]
        b.mark(5)
        self.assertGreater(b.fund_long, 0)
        self.assertLess(b.fund_short, 0)

    def test_run_ha_short_names_receive(self):
        import hype_short as H
        p = self.flat()
        p.fund[:, :10] = 0.001                      # ties make cols 0-9 the leaders: net short there
        r = H.run_ha(p, 7, zero_cost)
        self.assertGreater(r["fund_short"], 0)
        self.assertEqual(r["fund_long"], 0.0)
        self.assertGreater(r["final_equity"], 1.0)

    def test_run_ha_long_leg_pays(self):
        import hype_short as H
        p = self.flat()
        p.fund[:, 10:] = 0.001                      # only names outside the short leg: held long
        r = H.run_ha(p, 7, zero_cost)
        self.assertLess(r["fund_long"], 0)
        self.assertEqual(r["fund_short"], 0.0)
        self.assertLess(r["final_equity"], 1.0)

    def test_run_hb_short_receives_and_hedge_pays(self):
        import hype_short as H
        D, S = 200, 5
        p = panel(D=D, S=S, first=np.array([0, 20, 20, 20, 20]))
        p.close[:] = np.where(np.isfinite(p.close), 100.0, np.nan)
        p.fund[:, 1:] = 0.001
        r = H.run_hb(p, 0, zero_cost)
        self.assertGreater(r["fund_short"], 0)
        self.assertEqual(r["fund_long"], 0.0)
        p.fund[:] = 0.0
        p.fund[:, 0] = 0.001
        r = H.run_hb(p, 0, zero_cost)
        self.assertLess(r["fund_long"], 0)
        self.assertEqual(r["fund_short"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
