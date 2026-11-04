"""FC-W -- passive carry on the N most liquid pairs (fc-wide-carry-fcw-1104-spec.md §5). Written BEFORE fcw.py /
experiment_fcw.py. Offline.

Interface the tests fix:
  fcw.chooser(N)(ctx) -> list       first N of ctx.universe; a held pair is kept while it is in the first 2N
  fcw.candidates() -> {"FCW_N4": CarryConfig(K=4, ...), "FCW_N6": ..., "FCW_N10": ...}
  fcw.map_perp(perp_sym, spot_syms) -> (spot_sym, mult) | None
  fcw.ratio_ok(spot_close, perp_close, mult, tol) -> bool    median(perp / mult / spot) within 1 +- tol
  experiment_fcw.admit(res)
"""
import unittest

import numpy as np


def ctx(universe, held=()):
    import carry as C
    import xsmom as XS
    return XS.Ctx("2024-01-10", 0, frozenset(held), list(universe), {}, C.CarryConfig(W=3, K=4, theta_in=0.0), None)


UNI = [f"S{k:02d}USDT" for k in range(20)]


class TestChooser(unittest.TestCase):
    def test_first_n(self):
        import fcw
        self.assertEqual(fcw.chooser(4)(ctx(UNI)), UNI[:4])

    def test_hysteresis(self):
        import fcw
        # S06 held: inside the first 2N = 8 -> kept; S09 held: outside -> dropped
        got = fcw.chooser(4)(ctx(UNI, held={"S06USDT", "S09USDT"}))
        self.assertEqual(len(got), 4)
        self.assertIn("S06USDT", got)
        self.assertNotIn("S09USDT", got)
        self.assertEqual([s for s in got if s != "S06USDT"], UNI[:3])
        # a held pair that left the universe is dropped
        got = fcw.chooser(4)(ctx(UNI[1:], held={"S00USDT"}))
        self.assertEqual(got, UNI[1:5])

    def test_candidates(self):
        import fcw
        c = fcw.candidates()
        self.assertEqual(sorted(c), ["FCW_N10", "FCW_N4", "FCW_N6"])
        self.assertEqual(c["FCW_N6"].K, 6)
        self.assertEqual(c["FCW_N6"].lev, 1.0)
        self.assertFalse(c["FCW_N6"].passive)


class TestMapping(unittest.TestCase):
    def test_map_perp(self):
        import fcw
        spot = {"BTCUSDT", "SHIBUSDT", "PEPEUSDT"}
        self.assertEqual(fcw.map_perp("BTCUSDT", spot), ("BTCUSDT", 1))
        self.assertEqual(fcw.map_perp("1000SHIBUSDT", spot), ("SHIBUSDT", 1000))
        self.assertIsNone(fcw.map_perp("1000XECUSDT", spot))
        self.assertIsNone(fcw.map_perp("BTCBUSD", spot))
        self.assertIsNone(fcw.map_perp("BTCUSDT_230331", spot))

    def test_ratio_ok(self):
        import fcw
        s = np.array([1.0, 1.1, 1.2, np.nan])
        self.assertTrue(fcw.ratio_ok(s, s * 1000 * 1.01, 1000, 0.05))
        self.assertFalse(fcw.ratio_ok(s, s * 1.2, 1, 0.05))
        self.assertFalse(fcw.ratio_ok(np.array([np.nan]), np.array([1.0]), 1, 0.05))


class TestAdmission(unittest.TestCase):
    def test_admit(self):
        import experiment_fcw as X
        good = {"agg": {"lo95": 0.001, "mean_pct": 0.02}, "liquidations": 0, "paired_passive": {"lo95": 0.001},
                "random_p95": 0.01, "r_by_year": {"2023": 0.01, "2024": -0.01, "2025": 0.02}}
        self.assertTrue(X.admit(good)[0])
        for k, v in (("agg", {"lo95": -0.001, "mean_pct": 0.02}), ("liquidations", 1), ("paired_passive", {"lo95": -0.001}),
                     ("random_p95", 0.03), ("r_by_year", {"2023": -0.01, "2024": -0.01, "2025": 0.02})):
            self.assertFalse(X.admit(dict(good, **{k: v}))[0], k)


if __name__ == "__main__":
    unittest.main(verbosity=2)
