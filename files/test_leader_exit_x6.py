"""Guards for X-6. Spec: docs/specs/features/leader-exit-x6-spec.md"""
import ast
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class TestX6(unittest.TestCase):
    def setUp(self):
        self.src = (HERE / "_backtest_leader_exit_x6.py").read_text(encoding="utf-8")

    def test_parse_no_machine_paths(self):
        ast.parse(self.src)
        self.assertNotIn("D:/", self.src)

    def test_same_leader_rule_as_the_live_alert(self):
        self.assertIn("LEAD_RANK, LEAD_RET, LEAD_BARS = 3, 0.075, 8", self.src)
        import config
        self.assertEqual((config.LEADER_ALERT_RANK_MAX, config.LEADER_ALERT_MIN_RET, config.LEADER_ALERT_HOLD_BARS), (3, 0.075, 8))

    def test_only_leaders_confirmed_before_the_real_exit_change(self):
        self.assertIn("lead = np.where(LEAD[ie:ix, j])[0]", self.src)
        self.assertIn('return t["real"], False', self.src)

    def test_spec(self):
        spec = HERE.parent / "docs/specs/features/leader-exit-x6-spec.md"
        if not spec.exists():
            self.skipTest("spec not in this checkout")
        t = spec.read_text(encoding="utf-8")
        for tok in ("101 = 2.4%", "+0.024 п.п. [−0.009, +0.061]", "97.6%"):
            self.assertIn(tok, t)


if __name__ == "__main__":
    unittest.main(verbosity=2)
