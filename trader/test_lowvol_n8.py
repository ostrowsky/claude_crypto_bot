"""N8 -- low-volatility alt basket (lowvol-alts-n8-1028-spec.md §6). Written BEFORE lowvol.py / experiment_n8.py."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_xsmom import arr15, ds, panel


def walk(n, seed, vol):
    rnd = np.random.default_rng(seed)
    return list(100 * np.exp(np.cumsum(vol * rnd.standard_normal(n))))


class TestVolScores(unittest.TestCase):
    def test_no_lookahead_and_window(self):
        import lowvol as LV
        n = 80
        a = walk(n, 1, 0.02)
        p1 = panel({"AUSDT": arr15(a)})
        b = a[:50] + [x * 5 for x in a[50:]]
        p2 = panel({"AUSDT": arr15(b)})
        s1 = LV.vol_scores(p1, ds(50), ["AUSDT"], 28)
        s2 = LV.vol_scores(p2, ds(50), ["AUSDT"], 28)
        self.assertAlmostEqual(s1["AUSDT"], s2["AUSDT"], places=12)
        lr = np.diff(np.log(np.asarray(a[21:50])))           # 28 returns ending with day 49
        self.assertAlmostEqual(s1["AUSDT"], float(np.std(lr, ddof=1)), places=10)

    def test_needs_28_full_returns(self):
        import lowvol as LV
        p = panel({"AUSDT": arr15(walk(40, 2, 0.02))})
        self.assertEqual(LV.vol_scores(p, ds(28), ["AUSDT"], 28), {})          # 27 returns only
        self.assertIn("AUSDT", LV.vol_scores(p, ds(29), ["AUSDT"], 28))

    def test_pick_k_lowest_ties_by_name(self):
        import lowvol as LV
        self.assertEqual(LV.pick({"B": 0.01, "A": 0.01, "C": 0.03, "D": 0.02}, 3), ["A", "B", "D"])


class TestRotation(unittest.TestCase):
    def test_holds_the_calm_pairs(self):
        import fill_model as FM
        import lowvol as LV
        import xsmom as XS
        n = 140
        cols = {}
        for i in range(6):
            cols[f"C{i}USDT"] = arr15(walk(n, 10 + i, 0.005), vol=1000.0)   # calm
            cols[f"N{i}USDT"] = arr15(walk(n, 30 + i, 0.08), vol=1000.0)    # noisy
        p = panel(cols)
        cfg = XS.Config(L=1, H=7, K=6, liq_min_days=5, min_age_days=0)
        r = XS.rotate(p, cfg, FM.model("taker", 1), ds(60), ds(120), chooser=LV.lowvol_chooser(p, 6, 28),
                      start_equity=1_000_000.0)
        held = {t["sym"] for t in r["trades"]}
        self.assertTrue(held)
        self.assertTrue(all(s.startswith("C") for s in held), held)
        self.assertTrue(r["conservation_ok"])


class TestDecisionAndMarker(unittest.TestCase):
    def test_decide(self):
        import experiment_n8 as EX
        ok = EX.decide({"mean_pct": 0.05, "lo95": 0.001}, {"lo95": 0.001}, [0.0] * 50)
        self.assertTrue(ok["accepted"])
        self.assertFalse(EX.decide({"mean_pct": 0.05, "lo95": -0.001}, {"lo95": 0.001}, [0.0] * 50)["accepted"])
        self.assertFalse(EX.decide({"mean_pct": 0.05, "lo95": 0.001}, {"lo95": -0.001}, [0.0] * 50)["accepted"])
        self.assertFalse(EX.decide({"mean_pct": 0.05, "lo95": 0.001}, {"lo95": 0.001}, [0.06] * 50)["accepted"])

    def test_marker_once(self):
        import experiment_n8 as EX
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "M.json"
            EX.open_test(m, {"x": 1})
            with self.assertRaises(RuntimeError):
                EX.open_test(m, {"x": 1})

    def test_needs_operator(self):
        import experiment_n8 as EX
        self.assertEqual(EX.main(["test"]), 2)

    def test_frozen_rule(self):
        import experiment_n8 as EX
        c = EX.config()
        self.assertEqual((c.H, c.K, c.min_age_days, c.n_liq), (7, 10, 60, 100))
        self.assertEqual(EX.VOL_DAYS, 28)


if __name__ == "__main__":
    unittest.main(verbosity=2)
