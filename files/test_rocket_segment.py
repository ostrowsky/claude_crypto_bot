"""Guards for the rocket-segment research (definition, no look-ahead, time split).

Spec: docs/specs/features/rocket-segment-spec.md
"""
import ast
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = HERE.parent / "docs" / "specs" / "features" / "rocket-segment-spec.md"
SCRIPTS = ("_rocket_dataset.py", "_backtest_rocket_rules.py", "_backtest_rocket_taker.py", "_fetch_taker_15m.py")


class TestScripts(unittest.TestCase):
    def test_parse_and_no_machine_paths(self):
        for s in SCRIPTS:
            src = (HERE / s).read_text(encoding="utf-8")
            ast.parse(src)
            self.assertNotIn("D:/", src, s)

    def test_rocket_definition(self):
        src = (HERE / "_rocket_dataset.py").read_text(encoding="utf-8")
        self.assertIn('"rocket": int(dmax >= 0.10 and dclose >= 0.6 * dmax)', src)
        self.assertIn("CROSS = 0.025", src)
        self.assertIn("TRIGGERS = (0.025, 0.05, 0.075)", src)

    def test_labels_look_only_forward(self):
        src = (HERE / "_rocket_dataset.py").read_text(encoding="utf-8")
        self.assertIn("after = slice(i + 1, ds + 96)", src)
        # the alert is known at the crossing bar's close: features stop at i, the signal time is bar i + 1
        self.assertIn('"t": (t0 + (i + 1) * STEP)', src)

    def test_features_use_closed_higher_timeframe_bars(self):
        src = (HERE / "_rocket_dataset.py").read_text(encoding="utf-8")
        self.assertIn("k1h = (i - 3) // 4", src)

    def test_split_is_by_time(self):
        for s in ("_backtest_rocket_rules.py", "_backtest_rocket_taker.py"):
            src = (HERE / s).read_text(encoding="utf-8")
            self.assertIn('SPLIT = "2026-03-01"', src)
            self.assertNotIn("train_test_split", src)
            self.assertNotIn("shuffle", src)

    def test_rules_are_chosen_on_train_only(self):
        src = (HERE / "_backtest_rocket_rules.py").read_text(encoding="utf-8")
        self.assertIn("np.nanquantile(A[f][tr]", src)
        self.assertIn("cands.append((y[m & tr].mean()", src)


class TestSpec(unittest.TestCase):
    def setUp(self):
        if not SPEC.exists():
            self.skipTest("spec not in this checkout")
        self.text = SPEC.read_text(encoding="utf-8")

    def test_numbers(self):
        for tok in ("1 321", "7.7%", "13.3%", "0.675", "41.7% → 12.6%", "+9.6%", "−0.02% [−0.13, +0.10]", "R-10"):
            self.assertIn(tok, self.text)

    def test_the_goal_is_documented(self):
        for doc in ("CLAUDE.md", "PROJECT_CONTEXT.md"):
            t = (HERE.parent / doc).read_text(encoding="utf-8")
            self.assertIn("rocket-segment-spec.md", t, doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
