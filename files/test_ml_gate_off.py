"""Guards for the ML-gate-off check. Spec: docs/specs/features/ml-gate-off-spec.md"""
import ast
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class TestMlGateOff(unittest.TestCase):
    def setUp(self):
        self.src = (HERE / "_backtest_disable_ml_gate.py").read_text(encoding="utf-8")

    def test_parse_no_machine_paths(self):
        ast.parse(self.src)
        self.assertNotIn("D:/", self.src)

    def test_both_ml_gates_and_the_gate_order(self):
        self.assertIn('ML_GATES = {"ml_proba_zone", "ml_filter"}', self.src)
        # only impulse_guard runs before the ML gate in _poll_coin
        mon = (HERE / "monitor.py").read_text(encoding="utf-8")
        body = mon[mon.index("async def _poll_coin("):]
        self.assertLess(body.index('signal_type="impulse_guard"'), body.index('signal_type="ml_proba_zone"'))
        self.assertLess(body.index('signal_type="ml_proba_zone"'), body.index('signal_type="trend_quality"'))

    def test_model_change_split(self):
        self.assertIn('MODEL_CHANGE = "2026-09-07T19:31"', self.src)

    def test_spec(self):
        spec = HERE.parent / "docs/specs/features/ml-gate-off-spec.md"
        if not spec.exists():
            self.skipTest("spec not in this checkout")
        t = spec.read_text(encoding="utf-8")
        for tok in ("11.7%", "−0.135 п.п. [−0.256, −0.026]", "+0.863%", "+0.159%", "не отключать"):
            self.assertIn(tok, t)


if __name__ == "__main__":
    unittest.main(verbosity=2)
