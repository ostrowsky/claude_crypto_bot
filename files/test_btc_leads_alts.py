"""BTC-leads-alts research (btc-leads-alts-1003-spec.md): helper sanity and pre-registration."""
import unittest
from pathlib import Path

import numpy as np

import _backtest_btc_leads_alts as B

ROOT = Path(__file__).resolve().parent.parent


class TestHelpers(unittest.TestCase):
    def test_beta_recovers_a_known_slope(self):
        rng = np.random.default_rng(1)
        b = rng.normal(0, 1, 5000)
        a = 1.3 * b + rng.normal(0, 0.3, 5000)
        self.assertAlmostEqual(B.beta(a, b), 1.3, places=1)

    def test_aggregation_takes_the_last_close_of_each_block(self):
        c = np.arange(1, 13, dtype=float)
        self.assertEqual(list(B.agg(c, 4)), [4.0, 8.0, 12.0])

    def test_pre_registration(self):
        self.assertEqual(B.THETAS, (0.005, 0.010))
        self.assertEqual(B.FOCUS, ("SOLUSDT", "ETHUSDT"))
        self.assertEqual(B.FEE, 0.10)

    def test_documented(self):
        spec = (ROOT / "docs/specs/features/btc-leads-alts-1003-spec.md").read_text(encoding="utf-8")
        self.assertIn("REFUTED", spec)
        for doc in ("CLAUDE.md", "PROJECT_CONTEXT.md"):
            self.assertIn("btc-leads-alts-1003-spec.md", (ROOT / doc).read_text(encoding="utf-8"), doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
