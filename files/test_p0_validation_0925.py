"""Guards for the P0 validation of the 2026-09-25 audit package.

Spec: docs/specs/features/p0-validation-0925-spec.md
"""
import ast
import re
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = HERE.parent / "docs" / "specs" / "features" / "p0-validation-0925-spec.md"
SCRIPTS = ("_backtest_exit_engine_calibration.py", "_backtest_fast_loss_ema_exit.py", "_backtest_tq_forecast0_goal.py")


def _load(script, name, **ns):
    """Pull one function out of a script without running the script (they run at import)."""
    tree = ast.parse((HERE / script).read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    g = {"np": np, **ns}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), script, "exec"), g)
    return g[name]


class TestScripts(unittest.TestCase):
    def test_parse_and_no_machine_paths(self):
        for s in SCRIPTS:
            src = (HERE / s).read_text(encoding="utf-8")
            ast.parse(src)
            self.assertNotIn("D:/", src, s)
            self.assertIn("Path(__file__).resolve().parent", src, s)


class TestCalibratedTrail(unittest.TestCase):
    """The P-1 engine: close-anchored ratchet, exit at the first CLOSE below the stop."""

    def setUp(self):
        self.trail = _load("_backtest_tq_forecast0_goal.py", "live_trail", FLOOR=0.0)

    def test_ratchets_on_close_and_exits_at_close(self):
        c = np.array([100, 101, 104, 108, 107.5, 105, 103.0])
        atr = np.full(len(c), 1.0)
        j, pnl = self.trail(c, atr, 0, 2.0)
        # stop after bar 3 = 108 - 2 = 106; bar 5 closes 105 < 106 -> exit there, at that close
        self.assertEqual(j, 5)
        self.assertAlmostEqual(pnl, 5.0)

    def test_stop_never_moves_down(self):
        c = np.array([100, 110, 109.0, 100.0])
        atr = np.array([1.0, 1.0, 5.0, 5.0])     # ATR widens after the high
        j, _ = self.trail(c, atr, 0, 2.0)
        self.assertEqual(j, 3)                   # stop stays at 108 from bar 1, not 109-10=99

    def test_floor_widens_the_stop(self):
        trail = _load("_backtest_tq_forecast0_goal.py", "live_trail", FLOOR=0.08)
        c = np.array([100, 101, 97.0, 95.0, 96.0])
        atr = np.full(len(c), 0.5)
        j, _ = trail(c, atr, 0, 2.0)
        self.assertEqual(j, len(c) - 1)          # 8% floor: nothing within the window breaches


class TestGateOrder(unittest.TestCase):
    """P-3's downstream pass rate counts the gates AFTER trend_quality; pin that order to monitor.py."""

    def test_post_tq_gates_come_after_trend_quality(self):
        src = (HERE / "monitor.py").read_text(encoding="utf-8")
        body = src[src.index("async def _poll_coin("):]
        tq = body.index('signal_type="trend_quality"')
        g = {}
        exec(re.search(r"POST_TQ = \{[^}]*\}", (HERE / "_backtest_tq_forecast0_goal.py").read_text(encoding="utf-8")).group(0), g)
        for gate in g["POST_TQ"]:
            pos = body.find('signal_type="%s"' % gate)
            self.assertGreater(pos, tq, gate)
        for gate in ("impulse_guard", "ml_proba_zone", "entry_score"):
            self.assertLess(body.index('signal_type="%s"' % gate), tq, gate)

    def test_forecast_is_the_last_trend_quality_check(self):
        src = (HERE / "monitor.py").read_text(encoding="utf-8")
        fn = src[src.index("def _trend_entry_quality_guard_reason("):]
        fn = fn[:fn.index("\ndef ")]
        self.assertGreater(fn.index("weak 15m trend"), max(fn.index("price edge"), fn.index("daily_range"), fn.index("RSI {")))


class TestSpec(unittest.TestCase):
    def setUp(self):
        if not SPEC.exists():
            self.skipTest("spec not in this checkout")
        self.text = SPEC.read_text(encoding="utf-8")

    def test_headline_numbers(self):
        for tok in ("+0.007%", "+0.550%", "57%", "+0.84%", "+0.07%", "+0.29%", "11.0%", "14.0%", "18.8%",
                    "+0.194", "[+0.015, +0.365]", "24.4%"):
            self.assertIn(tok, self.text)

    def test_verdicts(self):
        for tok in ("ОПРОВЕРГНУТА", "ПРОШЛА", "рано судить"):
            self.assertIn(tok, self.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
