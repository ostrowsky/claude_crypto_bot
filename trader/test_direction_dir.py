"""DIR -- supervised direction forecast at 4h / 1d (direction-forecast-dir-1029-spec.md §7). Written BEFORE direction_dir.py /
experiment_dir.py. Synthetic arrays, offline.

Interface the tests fix:
  pair_features(c, qv) -> {name: array}           per-pair features at every bar index, from bars <= that index
  label(entry, exit_, delisted) -> r | None        exit missing: delisted -> -0.5, else None
  block_first_close(t15, c15, block_start_ms) -> close of the 15m bar that opens the block | None
  select(scores: {sym: p}) -> [sym]                floor(n / 10) highest, ties by name
  turnover_costs(selections: [[sym]], c) -> [{sym: cost}]   c/2 on entry, c/2 on exit
  auc(y, s) / auc_ci(y, s, days, alpha, seed, boot) -> (auc, lo, hi)
  folds(rows) -> purged walk-forward (4 folds, 2-day embargo) over rows with t0 / t1 datetimes
  experiment_dir: candidates(), admit(res), open_test(marker, info), main(["test"]) == 2
"""
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

BAR = 15 * 60 * 1000


class TestFeaturesAndLabel(unittest.TestCase):
    def test_no_lookahead(self):
        import direction_dir as D
        rng = np.random.default_rng(3)
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 80)))
        qv = rng.uniform(1, 2, 80)
        f1 = D.pair_features(c, qv)
        c2, qv2 = c.copy(), qv.copy()
        c2[51:] *= 3.0
        qv2[51:] *= 10
        f2 = D.pair_features(c2, qv2)
        self.assertEqual(set(f1), {"r1", "r2", "r3", "r6", "r12", "r24", "vol24", "z1", "vq", "pos24"})
        for k in f1:
            a, b = f1[k][:51], f2[k][:51]
            self.assertTrue(np.allclose(a, b, equal_nan=True), k)
        self.assertAlmostEqual(f1["r1"][10], np.log(c[10] / c[9]))
        self.assertAlmostEqual(f1["r6"][30], np.log(c[30] / c[24]))
        self.assertTrue(np.isnan(f1["r24"][10]))

    def test_label(self):
        import direction_dir as D
        self.assertAlmostEqual(D.label(100.0, 110.0, False), 0.10)
        self.assertAlmostEqual(D.label(100.0, None, True), -0.5)
        self.assertIsNone(D.label(100.0, None, False))

    def test_block_first_close(self):
        import direction_dir as D
        t0 = 1_700_000_000_000 // (4 * 3600 * 1000) * (4 * 3600 * 1000)
        t = np.arange(40, dtype=np.int64) * BAR + t0
        c = np.arange(40, dtype=float) + 1
        self.assertEqual(D.block_first_close(t, c, t0 + 16 * BAR), 17.0)          # first bar of block k+1
        self.assertIsNone(D.block_first_close(np.delete(t, 16), np.delete(c, 16), t0 + 16 * BAR))


class TestSelectionAndCosts(unittest.TestCase):
    def test_select(self):
        import direction_dir as D
        sc = {f"S{i:02d}": i / 100 for i in range(25)}
        self.assertEqual(D.select(sc), ["S24", "S23"])
        self.assertEqual(D.select({f"S{i}": 0.5 for i in range(9)}), [])
        self.assertEqual(D.select({f"S{i:02d}": 0.5 for i in range(10)}), ["S00"])

    def test_turnover_costs(self):
        import direction_dir as D
        cost = D.turnover_costs([["A"], ["A", "B"], ["A"], []], 0.36)
        self.assertAlmostEqual(cost[0]["A"], 0.18)                    # entry
        self.assertAlmostEqual(cost[1]["A"], 0.0)
        self.assertAlmostEqual(cost[1]["B"], 0.36)                    # in and out
        self.assertAlmostEqual(cost[2]["A"], 0.18)                    # exit
        self.assertEqual(cost[3], {})


class TestStats(unittest.TestCase):
    def test_auc_ci(self):
        import direction_dir as D
        rng = np.random.default_rng(5)
        n = 3000
        y = rng.integers(0, 2, n)
        days = np.repeat(np.arange(300), 10)
        a, lo, hi = D.auc_ci(y, y + rng.normal(0, 0.01, n), days, alpha=0.05, seed=29, boot=300)
        self.assertGreater(lo, 0.99)
        a, lo, hi = D.auc_ci(y, rng.normal(0, 1, n), days, alpha=0.05, seed=29, boot=300)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertAlmostEqual(D.auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])), 0.75)

    def test_folds_embargo(self):
        import direction_dir as D
        t = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = [{"t0": t + timedelta(days=i), "t1": t + timedelta(days=i + 1)} for i in range(200)]
        fs = D.folds(rows)
        self.assertEqual(len(fs), 4)
        for tr, te in fs:
            start = min(rows[i]["t0"] for i in te)
            self.assertTrue(all(rows[i]["t1"] < start - timedelta(days=2) for i in tr))


class TestAdmission(unittest.TestCase):
    def good(self, **kw):
        r = {"auc_lo": 0.51, "sel": {"mean_pct": 0.1, "lo95": 0.01}, "paired": {"mean_pct": 0.05, "lo95": 0.001},
             "sel_by_year": {"2023": -0.1, "2024": 0.1, "2025": 0.2}}
        r.update(kw)
        return r

    def test_admit_marker_operator(self):
        import experiment_dir as EX
        self.assertEqual(sorted(EX.candidates()), ["DIR_1d", "DIR_4h"])
        self.assertTrue(EX.admit(self.good())[0])
        for k, v in (("auc_lo", 0.5), ("sel", {"mean_pct": 0.1, "lo95": -0.01}), ("paired", {"mean_pct": 0.1, "lo95": 0.0}),
                     ("sel_by_year", {"2023": -0.1, "2024": -0.1, "2025": 0.2})):
            self.assertFalse(EX.admit(self.good(**{k: v}))[0], k)
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "DIR_TEST_OPENED.json"
            EX.open_test(m, {})
            with self.assertRaises(RuntimeError):
                EX.open_test(m, {})
        self.assertEqual(EX.main(["test"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
