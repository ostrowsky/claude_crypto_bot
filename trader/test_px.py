"""PX -- market-neutral pairs trading (pairs-trading-px-1104-spec.md §6). Written BEFORE px.py / experiment_px.py.
Synthetic arrays, offline.

Interface the tests fix:
  px.spread_z(la, lb, F) -> z of the last day (window = last F log-ratio values) | None
  px.ret_corr(la, lb, F) -> correlation of the last F daily log returns | None
  px.select(cands, free, busy) -> chosen [{a, b, z}]: by |z| desc, one coin in one pair, at most `free`
  px.legs(a, b, z) -> (long_spot, short_perp): the cheap leg is bought, the rich leg is shorted
  px.exit_reason(z, held_days, delisted) -> None | "converged" | "stop" | "time" | "delisted"
  px.open_pos(long_sym, short_sym, s_px, p_px, N, fees) / px.close_pos(pos, s_px, p_px, fees) -> pnl parts
  px.simulate(market, cand, fees, lo, hi, plan=None, seed=None) -> {series, trades, liquidations, ...}
      plan = [(entry_day, exit_day)] -> random control: random pair of that day's pool, random direction
  experiment_px.candidates() (4), admit(res)
"""
import math
import unittest
from datetime import datetime, timezone

import numpy as np

DAY = 86_400_000


def nofee():
    import carry as C
    return C.Fees(spot_taker=0.0, perp_taker=0.0, slip=0.0)


def synth_market(n_days=160, jump_day=110, jump=0.25, revert_after=3):
    """A and B move together; A jumps +25% on jump_day and reverts revert_after days later. C, D independent."""
    import carry as C
    import xsmom as XS
    rnd = np.random.default_rng(3)
    d0 = XS.day_num("2024-01-01")
    base = np.cumsum(rnd.normal(0, 0.01, n_days))
    la = base.copy()                                   # no idiosyncratic noise: the spread is 0 until the jump
    lb = base.copy()
    la[jump_day:jump_day + revert_after] += math.log(1 + jump)
    series = {"AUSDT": la, "BUSDT": lb, "CUSDT": np.cumsum(rnd.normal(0, 0.02, n_days)),
              "DUSDT": np.cumsum(rnd.normal(0, 0.02, n_days))}
    spot, perp, fund = {}, {}, {}
    for s, l in series.items():
        px_ = 100 * np.exp(l)
        d = {"day": np.arange(d0, d0 + n_days, dtype=np.int64), "close": px_, "first_close": px_,
             "qv": np.full(n_days, 1e9), "n_bars": np.full(n_days, 96), "complete": np.ones(n_days, bool),
             "has_first": np.ones(n_days, bool)}
        spot[s] = d
        perp[s] = {**d, "high": px_ * 1.001}
        t = (np.arange(d0, d0 + n_days, dtype=np.int64) * DAY)[:, None] + np.array([0, 8, 16]) * 3600 * 1000
        fund[s] = {"t": t.ravel(), "rate": np.full(t.size, 0.0001)}
    return C.Market(spot=XS.Panel.from_daily(spot), perp=C.PerpPanel.from_daily(perp), funding=fund), d0


class TestHelpers(unittest.TestCase):
    def test_spread_z_and_corr(self):
        import px
        la = np.log(np.array([1.0, 1.1, 1.2, 1.3, 1.5]))
        lb = np.log(np.array([1.0, 1.0, 1.1, 1.2, 1.2]))
        s = la - lb
        w = s[-4:]
        self.assertAlmostEqual(px.spread_z(la, lb, 4), (s[-1] - w.mean()) / w.std(ddof=1), places=12)
        self.assertIsNone(px.spread_z(la[:3], lb[:3], 4))
        la2 = la.copy()
        la2[-2] = np.nan
        self.assertIsNone(px.spread_z(la2, lb, 4))
        r = np.diff(la)[-3:], np.diff(lb)[-3:]
        self.assertAlmostEqual(px.ret_corr(la, lb, 3), float(np.corrcoef(*r)[0, 1]), places=12)
        self.assertIsNone(px.ret_corr(la, lb, 10))

    def test_select(self):
        import px
        c = [{"a": "A", "b": "B", "z": 2.1}, {"a": "A", "b": "C", "z": -3.0}, {"a": "D", "b": "E", "z": 2.5},
             {"a": "B", "b": "F", "z": 2.2}]
        got = px.select(c, free=2, busy=set())
        self.assertEqual([(x["a"], x["b"]) for x in got], [("A", "C"), ("D", "E")])
        got = px.select(c, free=5, busy={"C"})
        self.assertEqual([(x["a"], x["b"]) for x in got], [("D", "E"), ("B", "F")])     # A-B loses B to B-F

    def test_legs_and_exits(self):
        import px
        self.assertEqual(px.legs("A", "B", 2.3), ("B", "A"))      # spread high: A rich -> short A, buy B
        self.assertEqual(px.legs("A", "B", -2.3), ("A", "B"))
        self.assertEqual(px.exit_reason(0.4, 3, False), "converged")
        self.assertEqual(px.exit_reason(-4.1, 3, False), "stop")
        self.assertEqual(px.exit_reason(1.5, 20, False), "time")
        self.assertIsNone(px.exit_reason(1.5, 19, False))
        self.assertEqual(px.exit_reason(1.5, 3, True), "delisted")

    def test_pos_pnl_costs_and_funding(self):
        import carry as C
        import px
        fees = C.Fees(spot_taker=0.001, perp_taker=0.0005, slip=0.0008)
        p = px.open_pos("BUSDT", "AUSDT", 10.0, 20.0, 1000.0, fees)
        self.assertAlmostEqual(p["q_s"], 1000.0 / (10.0 * 1.0008) * (1 - 0.001))
        self.assertAlmostEqual(p["perp_entry"], 20.0 * (1 - 0.0008))
        self.assertAlmostEqual(p["q_p"], 1000.0 / p["perp_entry"])
        self.assertAlmostEqual(p["fees_in"], 1000.0 * 0.0005)
        p["funding"] = 3.0
        r = px.close_pos(p, 11.0, 19.0, fees)
        spot = p["q_s"] * 11.0 * (1 - 0.0008) * (1 - 0.001) - 1000.0
        perp_exit = 19.0 * (1 + 0.0008)
        perp = p["q_p"] * (p["perp_entry"] - perp_exit) - p["fees_in"] - p["q_p"] * perp_exit * 0.0005
        self.assertAlmostEqual(r["pnl"], spot + perp + 3.0, places=9)
        z = px.close_pos(px.open_pos("BUSDT", "AUSDT", 10.0, 20.0, 1000.0, nofee()), 10.0, 20.0, nofee())
        self.assertAlmostEqual(z["pnl"], 0.0, places=9)


class TestSimulate(unittest.TestCase):
    def test_one_converging_trade(self):
        import carry as C
        import experiment_px as X
        import px
        import xsmom as XS
        m, d0 = synth_market()
        cand = X.candidates()["PX_F30_Z20"]
        r = px.simulate(m, cand, nofee(), XS.day_str(d0 + 60), XS.day_str(d0 + 150))
        ab = [t for t in r["trades"] if {t["long"], t["short"]} == {"AUSDT", "BUSDT"}]
        self.assertGreaterEqual(len(ab), 1)
        t = ab[0]
        self.assertEqual((t["long"], t["short"]), ("BUSDT", "AUSDT"))
        self.assertEqual(t["entry_day"], XS.day_str(d0 + 111))     # decided on closes to d - 1
        self.assertEqual(t["reason"], "converged")
        self.assertGreater(t["pnl"], 0)
        self.assertAlmostEqual(t["gross_price_pct"] + t["funding_pct"], t["ret_pct"], places=9)   # no fees here
        self.assertTrue(r["conservation_ok"])
        self.assertEqual(r["liquidations"], 0)

    def test_random_control_reproducible(self):
        import experiment_px as X
        import px
        import xsmom as XS
        m, d0 = synth_market()
        cand = X.candidates()["PX_F30_Z20"]
        lo, hi = XS.day_str(d0 + 60), XS.day_str(d0 + 150)
        plan = [(XS.day_str(d0 + 70), XS.day_str(d0 + 75)), (XS.day_str(d0 + 111), XS.day_str(d0 + 114))]
        a = px.simulate(m, cand, nofee(), lo, hi, plan=plan, seed=4)
        b = px.simulate(m, cand, nofee(), lo, hi, plan=plan, seed=4)
        self.assertEqual(a["series"], b["series"])
        self.assertEqual([(t["entry_day"], t["exit_day"]) for t in a["trades"]], plan)


class TestAdmission(unittest.TestCase):
    def test_candidates_and_admit(self):
        import experiment_px as X
        c = X.candidates()
        self.assertEqual(sorted(c), ["PX_F30_Z20", "PX_F30_Z25", "PX_F60_Z20", "PX_F60_Z25"])
        self.assertEqual(c["PX_F60_Z25"], {"F": 60, "z_in": 2.5})
        good = {"agg": {"lo95": 0.001, "mean_pct": 0.02}, "random_p95": 0.01, "n_trades": 100, "liquidations": 0,
                "r_by_year": {"2023": 0.01, "2024": -0.01, "2025": 0.02}}
        self.assertTrue(X.admit(good)[0])
        for k, v in (("agg", {"lo95": -0.001, "mean_pct": 0.02}), ("random_p95", 0.03), ("n_trades", 99),
                     ("liquidations", 1), ("r_by_year", {"2023": -0.01, "2024": -0.01, "2025": 0.02})):
            self.assertFalse(X.admit(dict(good, **{k: v}))[0], k)


if __name__ == "__main__":
    unittest.main(verbosity=2)
