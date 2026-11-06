"""Helpers shared by N1-N5 (n-family-experiment-designs-1105-spec.md §1, n3-1106-spec.md §2). Written BEFORE n_common.py. Offline.

Interface the tests fix:
  n_common.ci(x, alpha=0.05, block=20, boot=20000, seed=29) -> {"mean", "lo", "hi", "ci_level", "ci_method", "block", "boot", "n"}
  n_common.cluster_ci(values, clusters, alpha=0.05, block=1, boot=20000, seed=29) -> same; resamples whole clusters in time order
  n_common.required_n(sigma, delta, alpha, power=0.80, m=1.0, rho=0.0) -> (n, n_eff)
  n_common.Volumes(by_symbol, first_day) where by_symbol = {sym: {day: quote volume}}, first_day = {sym: first day ever traded}
  n_common.Volumes.ranks(day, exclude=()) -> {sym: rank}     design §1.5: [d-30, d-1], >= 25 days of 30, age >= 30 days
  n_common.Volumes.top(day, k) -> [syms]
  n_common.quote_path(sym) -> percent-encoded symbol for archive URLs
"""
import unittest

import numpy as np


class TestCI(unittest.TestCase):
    def test_fields_and_mean(self):
        import n_common as N
        r = N.ci(list(np.arange(100.0)), alpha=0.05, block=5, boot=2000)
        self.assertAlmostEqual(r["mean"], 49.5)
        self.assertLess(r["lo"], 49.5)
        self.assertGreater(r["hi"], 49.5)
        self.assertEqual(r["ci_level"], 0.95)
        self.assertEqual(r["ci_method"], "block_bootstrap_percentile")
        self.assertEqual((r["block"], r["boot"], r["n"]), (5, 2000, 100))

    def test_default_boot_is_the_design_value(self):
        import n_common as N
        self.assertEqual(N.ci(list(np.ones(50)), alpha=0.05)["boot"], 20000)

    def test_normal_fallback_when_few_draws_beyond_the_bound(self):
        import n_common as N
        r = N.ci(list(np.random.default_rng(1).normal(size=200)), alpha=0.0125, block=5, boot=1000)
        self.assertEqual(r["ci_method"], "normal_block_se")
        r = N.ci(list(np.random.default_rng(1).normal(size=200)), alpha=0.0125, block=5, boot=20000)
        self.assertEqual(r["ci_method"], "block_bootstrap_percentile")

    def test_too_few(self):
        import n_common as N
        r = N.ci([1.0, 2.0, 3.0], alpha=0.05)
        self.assertEqual(r["ci_method"], "too_few")
        self.assertIsNone(r["lo"])

    def test_blocks_widen_the_interval_under_autocorrelation(self):
        import n_common as N
        rng = np.random.default_rng(3)
        e = rng.normal(size=2000)
        x = np.zeros(2000)
        for i in range(1, 2000):
            x[i] = 0.9 * x[i - 1] + e[i]
        a, b = N.ci(list(x), block=1, boot=4000), N.ci(list(x), block=50, boot=4000)
        self.assertGreater(b["hi"] - b["lo"], 2 * (a["hi"] - a["lo"]))

    def test_reproducible(self):
        import n_common as N
        x = list(np.random.default_rng(5).normal(size=80))
        self.assertEqual(N.ci(x, boot=2000), N.ci(x, boot=2000))


class TestClusterCI(unittest.TestCase):
    def test_mean_is_the_event_mean(self):
        import n_common as N
        r = N.cluster_ci([1.0, 3.0, 5.0] * 10, ["a", "a", "b"] * 10, boot=500)
        self.assertAlmostEqual(r["mean"], 3.0)

    def test_clustered_events_widen_the_interval(self):
        import n_common as N
        rng = np.random.default_rng(7)
        day = rng.normal(size=100)
        vals, cl = [], []
        for i, d in enumerate(day):
            for _ in range(20):
                vals.append(float(d + 0.1 * rng.normal()))
                cl.append(i)
        a = N.cluster_ci(vals, cl, boot=2000)
        b = N.cluster_ci(vals, list(range(len(vals))), boot=2000)
        self.assertGreater(a["hi"] - a["lo"], 2 * (b["hi"] - b["lo"]))


class TestRatioCI(unittest.TestCase):
    def test_ratio_of_sums(self):
        import n_common as N
        r = N.ratio_ci([1.0, 3.0] * 10, [1.0, 1.0] * 10, block=1, boot=2000, scale=1e4)
        self.assertAlmostEqual(r["mean"], 2e4)
        self.assertEqual(r["ci_method"], "block_bootstrap_percentile")

    def test_a_heavy_day_counts_by_its_denominator(self):
        import n_common as N
        r = N.ratio_ci([10.0] + [0.0] * 19, [10.0] + [1.0] * 19, block=1, boot=500)
        self.assertAlmostEqual(r["mean"], 10 / 29)

    def test_too_few(self):
        import n_common as N
        self.assertEqual(N.ratio_ci([1.0], [1.0])["ci_method"], "too_few")


class TestPower(unittest.TestCase):
    def test_formula(self):
        import n_common as N
        n, ne = N.required_n(1.0, 0.1, 0.05)
        self.assertAlmostEqual(n, ((1.959964 + 0.841621) / 0.1) ** 2, places=2)
        self.assertAlmostEqual(ne, n)
        n, ne = N.required_n(1.0, 0.1, 0.05, m=5, rho=0.5)
        self.assertAlmostEqual(ne, n * 3.0)

    def test_sign_of_delta_does_not_matter(self):
        import n_common as N
        self.assertEqual(N.required_n(1.0, -0.1, 0.05), N.required_n(1.0, 0.1, 0.05))


def _days(first, n):
    from datetime import date, timedelta
    d = date.fromisoformat(first)
    return [(d + timedelta(days=i)).isoformat() for i in range(n)]


class TestUniverse(unittest.TestCase):
    def test_no_data_of_the_day_or_later(self):
        import n_common as N
        days = _days("2024-01-01", 60)
        a = {d: 10.0 for d in days}
        b = {d: 5.0 for d in days}
        b["2024-02-15"] = 1e9                               # the decision day
        b["2024-02-20"] = 1e9                               # later
        v = N.Volumes({"A": a, "B": b}, {"A": "2024-01-01", "B": "2024-01-01"})
        self.assertEqual(v.top("2024-02-15", 1), ["A"])
        b["2024-02-14"] = 1e9                               # the day before does count
        v = N.Volumes({"A": a, "B": b}, {"A": "2024-01-01", "B": "2024-01-01"})
        self.assertEqual(v.top("2024-02-15", 1), ["B"])

    def test_needs_25_days_of_30(self):
        import n_common as N
        days = _days("2024-01-01", 60)
        full = {d: 10.0 for d in days}
        thin = {d: 99.0 for d in days if d < "2024-01-20" or d >= "2024-02-11"}     # 24 of the 30 days before 02-15 ... counted below
        window = _days("2024-01-16", 30)
        self.assertEqual(sum(1 for d in window if d in thin), 8)
        v = N.Volumes({"A": full, "T": thin}, {"A": "2024-01-01", "T": "2024-01-01"})
        self.assertEqual(v.top("2024-02-15", 2), ["A"])

    def test_age_at_least_30_days(self):
        import n_common as N
        days = _days("2024-01-01", 60)
        new = {d: 99.0 for d in days}
        v = N.Volumes({"A": {d: 1.0 for d in days}, "N": new}, {"A": "2024-01-01", "N": "2024-01-20"})
        self.assertEqual(v.top("2024-02-15", 2), ["A"])                             # 26 days old
        v2 = N.Volumes({"A": {d: 1.0 for d in days}, "N": new}, {"A": "2024-01-01", "N": "2024-01-16"})
        self.assertEqual(v2.top("2024-02-15", 2), ["N", "A"])                       # exactly 30 days old

    def test_fewer_than_k_returns_all_admitted(self):
        import n_common as N
        days = _days("2024-01-01", 60)
        v = N.Volumes({"A": {d: 1.0 for d in days}}, {"A": "2024-01-01"})
        self.assertEqual(v.top("2024-02-15", 20), ["A"])

    def test_exclude_pegged(self):
        import n_common as N
        days = _days("2024-01-01", 60)
        v = N.Volumes({"A": {d: 1.0 for d in days}, "USDCUSDT": {d: 9.0 for d in days}}, {"A": "2024-01-01", "USDCUSDT": "2024-01-01"})
        self.assertEqual(v.top("2024-02-15", 2, exclude=("USDCUSDT",)), ["A"])

    def test_ranks_are_one_based(self):
        import n_common as N
        days = _days("2024-01-01", 60)
        v = N.Volumes({"A": {d: 1.0 for d in days}, "B": {d: 2.0 for d in days}}, {"A": "2024-01-01", "B": "2024-01-01"})
        self.assertEqual(v.ranks("2024-02-15"), {"B": 1, "A": 2})


class TestPaths(unittest.TestCase):
    def test_non_ascii_symbol_is_encoded(self):
        import n_common as N
        self.assertEqual(N.quote_path("BTCUSDT"), "BTCUSDT")
        self.assertTrue(N.quote_path("牛来USDT").isascii())


if __name__ == "__main__":
    unittest.main(verbosity=2)
