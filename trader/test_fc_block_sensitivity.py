"""FC_PASSIVE CI vs block length (carry-funding-money-1021-spec.md, entry of 2026-11-05). Written BEFORE the code.

Interface the tests fix:
  experiment_fc.block_sensitivity(series, alpha, blocks=(5, 20, 60)) -> {block: {"lo", "hi", "mean", "ci_level",
       "ci_method", "lo_gt_0"}}
  experiment_fc.BLOCK_WINDOWS -> {"train": (from, to, alpha), "test": (...), "bh19": (...)}
"""
import unittest
from datetime import datetime, timedelta

import numpy as np


def series(n=400, seed=4, rho=0.9):
    """Positive mean with long regimes: AR(1) noise, so long blocks should widen the CI."""
    rnd = np.random.default_rng(seed)
    e, x = 0.0, []
    for _ in range(n):
        e = rho * e + rnd.normal(0, 0.01)
        x.append(0.004 + e)
    return [((datetime(2024, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d"), float(v)) for i, v in enumerate(x)]


class TestBlocks(unittest.TestCase):
    def test_keys_and_widening(self):
        import experiment_fc as EF
        r = EF.block_sensitivity(series(), 0.05)
        self.assertEqual(sorted(r), [5, 20, 60])
        for b in (5, 20, 60):
            self.assertEqual(r[b]["ci_level"], 0.95)
            self.assertLess(r[b]["lo"], r[b]["mean"])
            self.assertEqual(r[b]["lo_gt_0"], r[b]["lo"] > 0)
        w5, w60 = r[5]["hi"] - r[5]["lo"], r[60]["hi"] - r[60]["lo"]
        self.assertGreater(w60, w5)                      # persistent regimes: long blocks see more variance

    def test_windows_are_the_original_decisions(self):
        import experiment_fc as EF
        w = EF.BLOCK_WINDOWS
        self.assertEqual(w["train"], ("2023-03-01", "2026-01-01", 0.05))
        self.assertEqual(w["test"], ("2026-01-01", "2026-10-22", 0.05))
        self.assertEqual(w["bh19"], ("2020-02-01", "2023-01-01", 0.01))


if __name__ == "__main__":
    unittest.main(verbosity=2)
