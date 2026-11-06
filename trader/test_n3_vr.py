"""N3 stage 1: variance ratio (n3-1106-spec.md §2). Written BEFORE n3_vr.py. Synthetic, offline.

Interface the tests fix:
  n3_vr.returns(t, c, v) -> (r, ok)         r has len(t) - 1 values, ok False where the interval is not a clean 15m step
  n3_vr.vr(r, ok, q) -> {"vr", "z_star", "n", "windows"}
  n3_vr.symbol_month(t, c, v, horizons) -> {q: {...}} or None below MIN_RETURNS
  n3_vr.pool_months(rows) -> [(month, mean of VR - 1)]        rows = {month: {sym: vr_minus_1}}
  n3_vr.go(pool_ci, years, horizons) -> {"go", "pair"}        pool_ci = {q: {"mean", "hi"}}, years = {q: {year: mean}}
  n3_vr.terciles(sigma) -> {sym: 0|1|2}
  n3_vr.sigma_before(t, c, v, month_start_ms) -> float
  n3_vr.bounce_bias(c, sigma, q) -> float
"""
import math
import unittest

import numpy as np

BAR = 15 * 60 * 1000


def walk(n, seed, sd=0.005):
    rng = np.random.default_rng(seed)
    t = np.arange(n + 1, dtype=np.int64) * BAR
    c = 100 * np.exp(np.r_[0, np.cumsum(rng.normal(0, sd, n))])
    return t, c, np.ones(n + 1)


class TestVR(unittest.TestCase):
    def test_random_walk_is_unbiased_at_month_length(self):
        import n3_vr as N
        for q in (4, 16, 96):
            vals = []
            for s in range(60):
                t, c, v = walk(2880, s)
                r, ok = N.returns(t, c, v)
                vals.append(N.vr(r, ok, q)["vr"])
            self.assertAlmostEqual(float(np.mean(vals)), 1.0, delta=0.04 if q == 96 else 0.015)

    def test_ar1_negative_gives_vr2_about_08(self):
        import n3_vr as N
        rng = np.random.default_rng(11)
        e = rng.normal(0, 0.005, 200_000)
        r = np.zeros_like(e)
        for i in range(1, len(e)):
            r[i] = -0.2 * r[i - 1] + e[i]
        self.assertAlmostEqual(N.vr(r, np.ones(len(r), dtype=bool), 2)["vr"], 0.8, delta=0.01)

    def test_matches_the_closed_form_without_gaps(self):
        import n3_vr as N
        r = np.array([1.0, -2.0, 3.0, 0.5, -1.0, 2.0, -0.5, 1.5])
        mu = r.mean()
        n, q = len(r), 2
        sa = ((r - mu) ** 2).sum() / (n - 1)
        s = np.array([r[i] + r[i + 1] for i in range(n - 1)])
        m = q * (n - q + 1) * (1 - q / n)
        self.assertAlmostEqual(N.vr(r, np.ones(n, dtype=bool), q)["vr"], ((s - q * mu) ** 2).sum() / m / sa)

    def test_robust_z_is_not_inflated_by_heteroskedasticity(self):
        import n3_vr as N
        rej = 0
        sims = 300
        for s in range(sims):
            rng = np.random.default_rng(1000 + s)
            sig = np.where(np.arange(2880) % 960 < 96, 0.03, 0.003)              # volatility bursts
            r = rng.normal(0, 1, 2880) * sig
            rej += abs(N.vr(r, np.ones(2880, dtype=bool), 4)["z_star"]) > 1.96
        self.assertLess(rej / sims, 0.11)
        self.assertGreater(rej / sims, 0.01)


class TestGaps(unittest.TestCase):
    def test_missing_bar_removes_its_interval(self):
        import n3_vr as N
        t = np.array([0, 1, 2, 4, 5], dtype=np.int64) * BAR                      # bar 3 is absent
        c = np.array([100.0, 101, 102, 104, 105])
        r, ok = N.returns(t, c, np.ones(5))
        self.assertEqual(ok.tolist(), [True, True, False, True])

    def test_zero_volume_counts_as_missing(self):
        import n3_vr as N
        t = np.arange(4, dtype=np.int64) * BAR
        r, ok = N.returns(t, np.array([100.0, 101, 102, 103]), np.array([1.0, 0.0, 1.0, 1.0]))
        self.assertEqual(ok.tolist(), [False, False, True])

    def test_window_with_a_gap_is_not_counted(self):
        import n3_vr as N
        r = np.array([0.01, -0.01, 0.02, 0.0, 0.01, -0.02])
        ok = np.array([True, True, False, True, True, True])
        self.assertEqual(N.vr(r, ok, 2)["windows"], 3)                           # (0,1), (3,4), (4,5)
        self.assertEqual(N.vr(r, ok, 2)["n"], 5)

    def test_below_the_minimum_is_none(self):
        import n3_vr as N
        t, c, v = walk(N.MIN_RETURNS - 1, 1)
        self.assertIsNone(N.symbol_month(t, c, v, (4,)))
        t, c, v = walk(N.MIN_RETURNS, 1)
        self.assertIsNotNone(N.symbol_month(t, c, v, (4,)))


class TestPool(unittest.TestCase):
    def test_equal_weight_inside_the_month(self):
        import n3_vr as N
        rows = {"2024-01": {"A": -0.1, "B": 0.1, "C": -0.3}, "2024-02": {"A": 0.2}}
        out = dict(N.pool_months(rows))
        self.assertAlmostEqual(out["2024-01"], -0.1)
        self.assertAlmostEqual(out["2024-02"], 0.2)


class TestGo(unittest.TestCase):
    def _years(self, sign):
        return {str(y): sign * 0.01 for y in range(2020, 2026)}

    def test_two_adjacent_pass(self):
        import n3_vr as N
        ci = {4: {"mean": -0.03, "hi": -0.01}, 8: {"mean": -0.03, "hi": -0.01}, 16: {"mean": 0.01, "hi": 0.03}, 32: {"mean": 0.0, "hi": 0.02}, 96: {"mean": 0.0, "hi": 0.02}}
        y = {q: self._years(-1) for q in ci}
        r = N.go(ci, y, (4, 8, 16, 32, 96))
        self.assertTrue(r["go"])
        self.assertEqual(r["pair"], (4, 8))

    def test_not_adjacent_fails(self):
        import n3_vr as N
        ci = {4: {"mean": -0.03, "hi": -0.01}, 8: {"mean": 0.01, "hi": 0.02}, 16: {"mean": -0.03, "hi": -0.01}, 32: {"mean": 0.01, "hi": 0.02}, 96: {"mean": -0.03, "hi": -0.01}}
        y = {q: self._years(-1) for q in ci}
        self.assertFalse(N.go(ci, y, (4, 8, 16, 32, 96))["go"])

    def test_three_negative_years_fail(self):
        import n3_vr as N
        ci = {q: {"mean": -0.03, "hi": -0.01} for q in (4, 8)}
        y = {q: {"2020": -1, "2021": -1, "2022": -1, "2023": 1, "2024": 1, "2025": 1} for q in (4, 8)}
        self.assertFalse(N.go(ci, y, (4, 8))["go"])

    def test_four_negative_years_pass(self):
        import n3_vr as N
        ci = {q: {"mean": -0.03, "hi": -0.01} for q in (4, 8)}
        y = {q: {"2020": -1, "2021": -1, "2022": -1, "2023": -1, "2024": 1, "2025": 1} for q in (4, 8)}
        self.assertTrue(N.go(ci, y, (4, 8))["go"])


class TestTercile(unittest.TestCase):
    def test_sigma_uses_nothing_from_the_month(self):
        import n3_vr as N
        t, c, v = walk(4000, 3)
        start = int(t[3000])
        a = N.sigma_before(t, c, v, start)
        c2 = c.copy()
        c2[3000:] *= np.exp(np.arange(len(c2) - 3000) * 0.05)                    # a violent month
        self.assertEqual(a, N.sigma_before(t, c2, v, start))

    def test_three_groups_by_sigma(self):
        import n3_vr as N
        g = N.terciles({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0, "f": 6.0})
        self.assertEqual((g["a"], g["b"], g["c"], g["d"], g["e"], g["f"]), (0, 0, 1, 1, 2, 2))


class TestBounce(unittest.TestCase):
    def test_bias_formula(self):
        import n3_vr as N
        self.assertAlmostEqual(N.bounce_bias(0.0001, 0.005, 4), -(2 * 1e-8 / 2.5e-5) * 0.75)
        self.assertAlmostEqual(N.bounce_bias(0.0, 0.005, 4), 0.0)
        self.assertTrue(math.isfinite(N.bounce_bias(0.0001, 0.005, 96)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
