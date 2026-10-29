"""RV -- BTC <-> ETH spot rotation (btc-eth-rotation-rv-1028-spec.md §5). Written BEFORE rotation_rv.py / experiment_rv.py."""
import unittest

import numpy as np

import xsmom as XS
from test_timing_tm import panel_from, walk, zero_fm


class TestChoice(unittest.TestCase):
    def test_no_lookahead(self):
        import rotation_rv as RV
        n = 200
        c = {"BTCUSDT": walk(n, 1), "ETHUSDT": walk(n, 2)}
        p1 = panel_from(c)
        p2 = panel_from({s: v[:150] + [x * 4 for x in v[150:]] for s, v in c.items()})
        for name, cand in RV.candidates().items():
            a = RV.holdings(p1, cand, 60, 151)
            b = RV.holdings(p2, cand, 60, 151)
            self.assertTrue(np.array_equal(a, b), name)

    def test_mom_and_rev(self):
        import rotation_rv as RV
        n = 60
        btc = [100.0] * n
        eth = [100.0 * 1.01 ** i for i in range(n)]          # ETH stronger
        p = panel_from({"BTCUSDT": btc, "ETHUSDT": eth})
        mom = RV.holdings(p, RV.candidates()["RV_mom_14"], 30, 40)
        rev = RV.holdings(p, RV.candidates()["RV_rev_14"], 30, 40)
        self.assertTrue(np.all(mom == 1))                     # 1 = ETH
        self.assertTrue(np.all(rev == 0))


class TestMoney(unittest.TestCase):
    def test_constant_eth_is_price_ratio(self):
        import rotation_rv as RV
        n = 80
        c = {"BTCUSDT": walk(n, 3), "ETHUSDT": walk(n, 4)}
        p = panel_from(c)
        r0, r1 = 10, 70
        res = RV.simulate(p, np.ones(r1 - r0, dtype=int), zero_fm(), XS.day_str(p.d0 + r0), XS.day_str(p.d0 + r1),
                          start_equity=1000.0)
        self.assertAlmostEqual(res["final_equity"], 1000.0 * c["ETHUSDT"][r1 - 1] / c["ETHUSDT"][r0 - 1], places=6)

    def test_switch_moves_everything(self):
        import rotation_rv as RV
        n = 80
        c = {"BTCUSDT": walk(n, 5), "ETHUSDT": walk(n, 6)}
        p = panel_from(c)
        r0, r1 = 10, 70
        h = np.zeros(r1 - r0, dtype=int)
        h[20:] = 1                                           # BTC until row 29, ETH from row 30
        res = RV.simulate(p, h, zero_fm(), XS.day_str(p.d0 + r0), XS.day_str(p.d0 + r1), start_equity=1000.0)
        b, e = c["BTCUSDT"], c["ETHUSDT"]
        exp = 1000.0 * b[r0 + 20 - 1] / b[r0 - 1] * e[r1 - 1] / e[r0 + 20 - 1]
        self.assertAlmostEqual(res["final_equity"], exp, places=6)
        self.assertTrue(res["conservation_ok"])
        self.assertEqual(res["n_switches"], 1)

    def test_conservation_with_fees(self):
        import fill_model as FM
        import rotation_rv as RV
        n = 300
        p = panel_from({"BTCUSDT": walk(n, 7), "ETHUSDT": walk(n, 8)})
        res = RV.run(p, RV.candidates()["RV_mom_14"], FM.model("taker", 1), XS.day_str(p.d0 + 70), XS.day_str(p.d0 + 290))
        self.assertTrue(res["conservation_ok"])


class TestControls(unittest.TestCase):
    def test_shift(self):
        import rotation_rv as RV
        h = (np.cumsum(np.random.default_rng(1).standard_normal(400)) > 0).astype(int)
        sh = RV.shift(h, seed=4)
        self.assertEqual(int(sh.sum()), int(h.sum()))
        circ = lambda x: int(np.sum(x[1:] != x[:-1]) + (x[0] != x[-1]))  # noqa: E731
        self.assertEqual(circ(sh), circ(h))
        self.assertFalse(np.array_equal(sh, h))


class TestAdmission(unittest.TestCase):
    def test_grid_and_admit(self):
        import experiment_rv as EX
        self.assertEqual(sorted(EX.candidates()), sorted(f"RV_{k}_{n}" for k in ("mom", "rev") for n in (14, 28, 56)))
        g = {"mean_pct": 0.1, "random_means": [0.0] * 50, "paired_5050": {"mean_pct": 0.01},
             "paired_5050_by_year": {"2023": 0.01, "2024": -0.01, "2025": 0.02}, "n_switches": 12}
        self.assertTrue(EX.admit(g)[0])
        for k, v in (("random_means", [0.2] * 50), ("paired_5050", {"mean_pct": -0.01}), ("n_switches", 11),
                     ("paired_5050_by_year", {"2023": -0.01, "2024": -0.01, "2025": 0.02})):
            self.assertFalse(EX.admit(dict(g, **{k: v}))[0], k)
        self.assertEqual(EX.main(["test"]), 2)


class TestOneLook(unittest.TestCase):
    """Spec §4 test rule (k = 1): R lo > 0, paired 50/50 lo > 0, mean R > p95 of random rotations; marker once."""

    def test_decide(self):
        import experiment_rv as EX
        good = ({"mean_pct": 0.2, "lo95": 0.01}, {"mean_pct": 0.05, "lo95": 0.001}, [0.1] * 50)
        self.assertTrue(EX.decide(*good)["accepted"])
        self.assertFalse(EX.decide({"mean_pct": 0.2, "lo95": -0.01}, good[1], good[2])["accepted"])
        self.assertFalse(EX.decide(good[0], {"mean_pct": 0.05, "lo95": -0.001}, good[2])["accepted"])
        self.assertFalse(EX.decide(good[0], good[1], [0.3] * 50)["accepted"])

    def test_marker_once_and_operator(self):
        import tempfile
        from pathlib import Path
        import experiment_rv as EX
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "RV_TEST_OPENED.json"
            EX.open_test(m, {})
            with self.assertRaises(RuntimeError):
                EX.open_test(m, {})
        self.assertEqual(EX.main(["test"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
