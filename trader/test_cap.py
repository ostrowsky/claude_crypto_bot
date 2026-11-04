"""CAP -- market-wide capitulation basket (capitulation-basket-cap-1104-spec.md §5). Written BEFORE cap.py /
experiment_cap.py. Synthetic arrays, offline.

Interface the tests fix:
  cap.breadth(grid, pairs, uni, W, Y, min_pairs=50) -> share per grid bar (NaN if fewer than min_pairs pairs)
      grid: int64 15m open times; pairs: {sym: (t15, c15)}; uni: {day_number: set(syms)}
  cap.events(share, X, hold_bars) -> [grid index]           non-overlapping: next >= i + 1 + hold_bars
  cap.basket(grid, i, hold_bars, pairs, syms, fm) -> {ret_pct, n_pairs, entry_close_ms, exit_close_ms}
  cap.mean_ci(rets, alpha, seed) -> {mean_pct, lo95, hi95, n}
  experiment_cap.candidates() (8), admit(res)
"""
import math
import unittest
from datetime import datetime, timezone

import numpy as np

BAR = 15 * 60 * 1000
DAY = 86_400_000


def ms(s):
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)


T0 = ms("2020-03-01 00:00")


def zero_fm():
    import fill_model as FM
    return FM.FillModel(taker_fee=0.0, maker_fee=0.0, half_spread=0.0, impact_k=0.0)


def market(n_pairs=60, n_bars=96 * 4, crash_bar=None, crash_share=0.0, drop=0.10):
    grid = T0 + np.arange(n_bars, dtype=np.int64) * BAR
    pairs = {}
    for k in range(n_pairs):
        c = np.full(n_bars, 100.0 + k)
        if crash_bar is not None and k < int(round(crash_share * n_pairs)):
            c[crash_bar:] = c[0] * (1 - drop)
        pairs[f"P{k:02d}USDT"] = (grid.copy(), c)
    uni = {int(T0 // DAY) + d: set(pairs) for d in range(n_bars // 96 + 1)}
    return grid, pairs, uni


class TestBreadth(unittest.TestCase):
    def test_share_and_min_pairs(self):
        import cap
        grid, pairs, uni = market(crash_bar=200, crash_share=0.5, drop=0.06)
        s = cap.breadth(grid, pairs, uni, W=4, Y=0.05)
        self.assertAlmostEqual(s[200], 0.5)                       # 30 of 60 down 6% over 1 h
        self.assertAlmostEqual(s[203], 0.5)
        self.assertAlmostEqual(s[204], 0.0)                       # the 1 h window has passed the drop
        self.assertTrue(np.isnan(s[2]))                           # no bar 1 h back
        g2, p2, u2 = market(n_pairs=40, crash_bar=200, crash_share=1.0)
        self.assertTrue(np.isnan(cap.breadth(g2, p2, u2, W=4, Y=0.05)[200]))     # 40 < 50 pairs

    def test_universe_of_the_day_and_no_future(self):
        import cap
        grid, pairs, uni = market(crash_bar=200, crash_share=0.5, drop=0.06)
        d = int(grid[200] // DAY)
        uni2 = dict(uni)
        uni2[d] = set(sorted(pairs)[30:])                         # crashed pairs not in that day's universe
        self.assertTrue(np.isnan(cap.breadth(grid, pairs, uni2, W=4, Y=0.05)[200]))   # 30 < 50 pairs left
        p3 = {k: (t, c.copy()) for k, (t, c) in pairs.items()}
        for k in p3:
            p3[k][1][250:] *= 0.5                                 # a later move does not change bar 200
        self.assertAlmostEqual(cap.breadth(grid, p3, uni, W=4, Y=0.05)[200], 0.5)


class TestEventsAndBasket(unittest.TestCase):
    def test_events_non_overlapping(self):
        import cap
        s = np.zeros(1000)
        s[[100, 150, 300, 301]] = 0.6
        self.assertEqual(cap.events(s, 0.5, hold_bars=96), [100, 300])           # 150 < 100 + 1 + 96
        self.assertEqual(cap.events(s, 0.7, hold_bars=96), [])
        s[np.isnan(s)] = 0
        s2 = s.copy()
        s2[500] = np.nan
        self.assertEqual(cap.events(s2, 0.5, hold_bars=96), [100, 300])

    def test_basket(self):
        import cap
        n = 96 * 3
        grid = T0 + np.arange(n, dtype=np.int64) * BAR
        a = np.linspace(100, 110, n)
        b = np.linspace(50, 45, n)
        pairs = {"AUSDT": (grid.copy(), a), "BUSDT": (grid.copy(), b), "DUSDT": (grid[:120].copy(), np.full(120, 10.0))}
        r = cap.basket(grid, 10, 96, pairs, ["AUSDT", "BUSDT"], zero_fm())
        exp = ((a[107] / a[11] - 1) + (b[107] / b[11] - 1)) / 2 * 100
        self.assertAlmostEqual(r["ret_pct"], exp, places=9)
        self.assertEqual(r["n_pairs"], 2)
        self.assertEqual(r["entry_close_ms"], int(grid[11]) + BAR)
        d = cap.basket(grid, 100, 96, pairs, ["DUSDT"], zero_fm())                 # delisted inside the hold
        self.assertAlmostEqual(d["ret_pct"], -50.0)
        self.assertIsNone(cap.basket(grid, 10, 96, pairs, ["XUSDT"], zero_fm()))   # no pair -> no event

    def test_mean_ci(self):
        import cap
        rnd = np.random.default_rng(5)
        v = list(rnd.normal(1.0, 2.0, 40))
        a = cap.mean_ci(v, alpha=0.05)
        self.assertAlmostEqual(a["mean_pct"], round(float(np.mean(v)), 4))
        self.assertLess(a["lo95"], a["mean_pct"])
        b = cap.mean_ci(v, alpha=0.05 / 8)
        self.assertLessEqual(b["lo95"], a["lo95"])
        self.assertEqual(cap.mean_ci(v, alpha=0.05), a)
        self.assertEqual(a["n"], 40)


class TestAdmission(unittest.TestCase):
    def test_candidates_and_admit(self):
        import experiment_cap as X
        c = X.candidates()
        self.assertEqual(len(c), 8)
        self.assertEqual(c["CAP_W1_X25_H24"], {"W": 4, "Y": 0.05, "X": 0.25, "H": 96})
        self.assertEqual(c["CAP_W4_X50_H72"], {"W": 16, "Y": 0.10, "X": 0.50, "H": 288})
        good = {"agg": {"lo95": 0.1, "mean_pct": 2.0}, "rand_p95": 1.0, "n_events": 10,
                "by_period": {"2018-2019": 1.0, "2020-2021": -0.5, "2022": 0.3}}
        self.assertTrue(X.admit(good)[0])
        for k, v in (("agg", {"lo95": -0.1, "mean_pct": 2.0}), ("rand_p95", 2.5), ("n_events", 9),
                     ("by_period", {"2018-2019": -1.0, "2020-2021": -0.5, "2022": 0.3})):
            self.assertFalse(X.admit(dict(good, **{k: v}))[0], k)


class Test2026Look(unittest.TestCase):
    """Operator 2026-11-04: one look at 2026 for the frozen CAP_W4_X50_H24 (spec journal)."""

    def test_verdict_and_single_look(self):
        import tempfile
        from pathlib import Path
        import experiment_cap as X
        self.assertEqual(X.RULE_2026, "CAP_W4_X50_H24")
        self.assertEqual(X.verdict_2026({"n": 4, "mean_pct": 9.0, "lo95": 1.0}, 0.5), "TOO_FEW")
        self.assertEqual(X.verdict_2026({"n": 6, "mean_pct": 2.0, "lo95": 0.1}, 0.5), "CONFIRMED_2026")
        self.assertEqual(X.verdict_2026({"n": 6, "mean_pct": 2.0, "lo95": -0.1}, 0.5), "NOT_SHOWN_2026")
        self.assertEqual(X.verdict_2026({"n": 6, "mean_pct": 0.4, "lo95": 0.1}, 0.5), "NOT_SHOWN_2026")
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "CAP_2026_OPENED"
            X.open_2026(m)
            with self.assertRaises(SystemExit):
                X.open_2026(m)

if __name__ == "__main__":
    unittest.main(verbosity=2)
