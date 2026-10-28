"""MD -- multi-day spot trading (multiday-spot-md-1028-spec.md §4). Written BEFORE multiday.py / experiment_md.py."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_xsmom import arr15, ds, panel


def zero_fm():
    import fill_model as FM
    return FM.FillModel(taker_fee=0.0, maker_fee=0.0, half_spread=0.0, impact_k=0.0)


def walk(n, seed, vol=0.02, drift=0.0):
    rnd = np.random.default_rng(seed)
    return list(100 * np.exp(np.cumsum(drift + vol * rnd.standard_normal(n))))


def ucfg():
    import xsmom as XS
    return XS.Config(L=1, H=1, K=10, liq_min_days=5, min_age_days=0)


class TestEvents(unittest.TestCase):
    def test_brk(self):
        import multiday as MD
        c = [100.0] * 30
        c[25] = 105.0                                     # day 25 closes above the prior 20-day max
        p = panel({"AUSDT": arr15(c)})
        self.assertIn("AUSDT", MD.events(p, ds(26), "BRK", ["AUSDT"]))
        self.assertAlmostEqual(MD.events(p, ds(26), "BRK", ["AUSDT"])["AUSDT"], 0.05, places=9)
        self.assertEqual(MD.events(p, ds(27), "BRK", ["AUSDT"]), {})   # day 26 = 100 < 105

    def test_dip(self):
        import multiday as MD
        c = [100.0 * 1.01 ** i for i in range(60)]        # steady uptrend
        c[57] = c[54] * 0.90                              # -10% over 3 days (vs close[d-4]), above SMA50?
        p = panel({"AUSDT": arr15(c)})
        ev = MD.events(p, ds(58), "DIP", ["AUSDT"])
        sma = np.mean(c[8:58])
        if c[57] > sma:
            self.assertAlmostEqual(ev["AUSDT"], 0.10, places=9)
        else:
            self.assertEqual(ev, {})
        c2 = list(c)
        c2[57] = c[54] * 0.95                             # only -5%: no dip
        self.assertEqual(MD.events(panel({"AUSDT": arr15(c2)}), ds(58), "DIP", ["AUSDT"]), {})

    def test_vol(self):
        import multiday as MD
        c = [100.0 + 0.1 * i for i in range(30)]
        vol = {25: 5.0}                                    # day 25: 5x the usual volume, an up day
        p = panel({"AUSDT": arr15(c, day_vol=vol)})
        ev = MD.events(p, ds(26), "VOL", ["AUSDT"])
        self.assertIn("AUSDT", ev)
        self.assertGreater(ev["AUSDT"], 4.9)

    def test_no_lookahead(self):
        import multiday as MD
        n = 90
        base = {f"S{i}USDT": walk(n, i) for i in range(5)}
        p1 = panel({s: arr15(v) for s, v in base.items()})
        alt = {s: v[:70] + [x * 3 for x in v[70:]] for s, v in base.items()}
        p2 = panel({s: arr15(v, day_vol={k: 9.0 for k in range(70, n)}) for s, v in alt.items()})
        for kind in ("BRK", "DIP", "VOL"):
            self.assertEqual(MD.events(p1, ds(70), kind, list(base)), MD.events(p2, ds(70), kind, list(base)), kind)


class TestBook(unittest.TestCase):
    def setUp(self):
        n = 120
        self.cols = {f"S{i}USDT": walk(n, 50 + i, 0.03, 0.002) for i in range(15)}
        self.p = panel({s: arr15(v, vol=1000.0) for s, v in self.cols.items()})

    def test_hold_exactly_h_days_k_slots_no_reentry(self):
        import multiday as MD
        r = MD.simulate(self.p, {"kind": "BRK", "H": 5, "K": 4}, zero_fm(), ds(40), ds(110), ucfg(),
                        start_equity=1000.0)
        self.assertGreater(len(r["trades"]), 3)
        for t in r["trades"]:
            if t["reason"] == "hold":
                self.assertEqual(MD.day_gap(t["entry_day"], t["exit_day"]), 5)
        self.assertLessEqual(r["max_open"], 4)
        # never two open positions in one coin
        spans = {}
        for t in r["trades"]:
            spans.setdefault(t["sym"], []).append((t["entry_day"], t["exit_day"]))
        for s, sp in spans.items():
            sp.sort()
            for (a0, b0), (a1, b1) in zip(sp, sp[1:]):
                self.assertGreaterEqual(a1, b0, s)
        self.assertTrue(r["conservation_ok"])

    def test_slot_is_equity_over_k(self):
        import multiday as MD
        r = MD.simulate(self.p, {"kind": "BRK", "H": 5, "K": 4}, zero_fm(), ds(40), ds(110), ucfg(), start_equity=1000.0)
        first = sorted(r["trades"], key=lambda t: t["entry_day"])[0]
        self.assertAlmostEqual(first["cost"], 250.0, places=6)

    def test_random_control_matches_counts(self):
        import multiday as MD
        r = MD.simulate(self.p, {"kind": "BRK", "H": 5, "K": 4}, zero_fm(), ds(40), ds(110), ucfg(), start_equity=1000.0)
        rr = MD.simulate(self.p, {"kind": "BRK", "H": 5, "K": 4}, zero_fm(), ds(40), ds(110), ucfg(),
                         start_equity=1000.0, chooser=MD.random_chooser(r["moves"], seed=3))
        self.assertEqual({d: n for d, n in r["moves"].items() if n}, {d: n for d, n in rr["moves"].items() if n})
        self.assertTrue(rr["conservation_ok"])


class TestExperiment(unittest.TestCase):
    def test_grids(self):
        import experiment_md as EX
        u = EX.candidates_u()
        self.assertEqual(len(u), 9)
        self.assertEqual({c["kind"] for c in u.values()}, {"BRK", "DIP", "VOL"})
        self.assertEqual(sorted({c["H"] for c in u.values()}), [2, 5, 10])
        b = EX.candidates_b()
        self.assertEqual(len(b), 9)
        self.assertEqual(sorted({p.exits.max_bars for p in b.values()}), [288, 480, 960])
        self.assertEqual(sorted({p.exits.floor_override for p in b.values()}), [0.08, 0.12, 0.20])

    def test_b_windows(self):
        import experiment_md as EX
        self.assertEqual(EX.b_windows("2026-10-27"), ("2026-06-21", "2026-07-01", "2026-10-17"))

    def test_admit_u(self):
        import experiment_md as EX
        g = {"mean_pct": 0.05, "random_means": [0.0] * 50, "paired_ew": {"mean_pct": 0.01},
             "paired_ew_by_year": {"2023": 0.01, "2024": -0.01, "2025": 0.02}, "n_trades": 200}
        self.assertTrue(EX.admit_u(g)[0])
        for k, v in (("random_means", [0.06] * 50), ("paired_ew", {"mean_pct": -0.01}), ("n_trades", 199),
                     ("paired_ew_by_year", {"2023": -0.01, "2024": -0.01, "2025": 0.02})):
            self.assertFalse(EX.admit_u(dict(g, **{k: v}))[0], k)

    def test_admit_b(self):
        import experiment_md as EX
        g = {"changed": 100, "paired": {"lo95": 0.01, "mean_pct": 0.1}, "agg": {"mean_pct": 0.01}}
        self.assertTrue(EX.admit_b(g))
        self.assertFalse(EX.admit_b(dict(g, changed=99)))
        self.assertFalse(EX.admit_b(dict(g, paired={"lo95": -0.01, "mean_pct": 0.1})))
        self.assertFalse(EX.admit_b(dict(g, agg={"mean_pct": -0.001})))

    def test_marker_and_operator(self):
        import experiment_md as EX
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "M.json"
            EX.open_test(m, {})
            with self.assertRaises(RuntimeError):
                EX.open_test(m, {})
        self.assertEqual(EX.main(["test"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
