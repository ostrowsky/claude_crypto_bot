"""Guards for the P2/P3 validation of the 2026-09-25 audit package.

Spec: docs/specs/features/p2-validation-0925-spec.md
"""
import ast
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SPEC = HERE.parent / "docs" / "specs" / "features" / "p2-validation-0925-spec.md"


class TestScript(unittest.TestCase):
    def test_parse_and_no_machine_paths(self):
        src = (HERE / "_backtest_curtail_fallback_tq.py").read_text(encoding="utf-8")
        ast.parse(src)
        self.assertNotIn("D:/", src)

    def test_forecast_zero_is_kept_apart(self):
        # E-4 already lifts forecast 0.000; E-6 must not count it twice
        src = (HERE / "_backtest_curtail_fallback_tq.py").read_text(encoding="utf-8")
        self.assertIn('"forecast 0.000" in r', src)

    def test_curtail_history_uses_the_live_formula(self):
        src = (HERE / "_backtest_curtail_fallback_tq.py").read_text(encoding="utf-8")
        for k in ("IMPULSE_SPEED_CURTAIL_WINDOW_DAYS", "IMPULSE_SPEED_CURTAIL_PNL_THRESHOLD",
                  "IMPULSE_SPEED_CURTAIL_MIN_TRADES", "trade_exit_pnl"):
            self.assertIn(k, src)


class TestD3(unittest.TestCase):
    def test_one_definition_left(self):
        tree = ast.parse((HERE / "bot.py").read_text(encoding="utf-8"))
        defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_ensure_positions_monitored"]
        self.assertEqual(len(defs), 1)

    def test_no_duplicate_top_level_defs_in_bot(self):
        tree = ast.parse((HERE / "bot.py").read_text(encoding="utf-8"))
        names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        self.assertEqual(sorted(n for n in set(names) if names.count(n) > 1), [])


class TestSpec(unittest.TestCase):
    def setUp(self):
        if not SPEC.exists():
            self.skipTest("spec not in this checkout")
        self.text = SPEC.read_text(encoding="utf-8")

    def test_numbers_and_verdicts(self):
        for tok in ("101 из 106", "1 866 (33%)", "−1.148%", "E-6b", "2026-09-26", "ОПРОВЕРГНУТА", "инертен",
                    "−0.14% [−1.28, +1.25]"):
            self.assertIn(tok, self.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
