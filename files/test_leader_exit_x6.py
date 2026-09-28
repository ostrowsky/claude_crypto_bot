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



class TestX7(unittest.TestCase):
    def setUp(self):
        self.src = (HERE / "_backtest_leader_exit_x7.py").read_text(encoding="utf-8")

    def test_parse_no_machine_paths(self):
        ast.parse(self.src)
        self.assertNotIn("D:/", self.src)

    def test_variants_fixed(self):
        self.assertIn("LEAD_A = lead_matrix(3, 0.075, 4)", self.src)
        self.assertIn("LEAD_B = lead_matrix(3, 0.05, 1)", self.src)
        self.assertIn('for name in ("X-7a", "X-7b", "X-7c"):', self.src)

    def test_spec_x7(self):
        spec = HERE.parent / "docs/specs/features/leader-exit-x6-spec.md"
        if not spec.exists():
            self.skipTest("spec not in this checkout")
        t = spec.read_text(encoding="utf-8")
        for tok in ("## X-7", "+1.18 [+0.75, +1.68]", "7 из 7", "+0.026 п.п. [−0.030, +0.087]"):
            self.assertIn(tok, t)



class TestX8X9(unittest.TestCase):
    def setUp(self):
        self.src = (HERE / "_backtest_leader_x8_x9.py").read_text(encoding="utf-8")

    def test_parse_no_machine_paths(self):
        ast.parse(self.src)
        self.assertNotIn("D:/", self.src)

    def test_variants_fixed(self):
        for tok in ("LEAD_9A = lead_matrix(3, 0.03, 1)", "LEAD_9B = lead_matrix(3, 0.025, 1)",
                    "LEAD_9C = lead_matrix(5, 0.03, 1)", "LEAD_B = lead_matrix(3, 0.05, 1)"):
            self.assertIn(tok, self.src)

    def test_x8_reentry_only_inside_the_cooldown_and_without_a_real_reentry(self):
        self.assertIn("win_end = min(N2 - 1, ix + COOL)", self.src)
        self.assertIn("if any(ix < e <= q for e in real_entries[j]):", self.src)

    def test_spec_x8_x9(self):
        spec = HERE.parent / "docs/specs/features/leader-exit-x6-spec.md"
        if not spec.exists():
            self.skipTest("spec not in this checkout")
        t = spec.read_text(encoding="utf-8")
        for tok in ("## X-8 / X-9", "+0.42 [+0.19, +0.68]", "138 = 27.8%", "−0.002 [−0.141, +0.125]"):
            self.assertIn(tok, t)


if __name__ == "__main__":
    unittest.main(verbosity=2)
