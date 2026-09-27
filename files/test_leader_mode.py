"""Guards for the leader-mode research. Spec: docs/specs/features/leader-mode-spec.md"""
import ast
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class TestLeaderMode(unittest.TestCase):
    def setUp(self):
        self.src = (HERE / "_backtest_leader_mode.py").read_text(encoding="utf-8")

    def test_parse_no_machine_paths(self):
        ast.parse(self.src)
        self.assertNotIn("D:/", self.src)

    def test_time_split_and_train_selection(self):
        self.assertIn('SPLIT = "2026-03-01"', self.src)
        self.assertIn('res.sort(key=lambda x: -(x[1]["mean"]', self.src)   # x[1] = train summary

    def test_no_look_ahead_in_entry(self):
        # rank from the bar's own close; the trade enters at that close and only later bars move the stop
        self.assertIn("row = RET[i]", self.src)
        self.assertIn("ep = C[i, j]", self.src)
        self.assertIn("for q in range(i + 1, last + 1):", self.src)

    def test_spec_numbers(self):
        spec = (HERE.parent / "docs/specs/features/leader-mode-spec.md")
        if not spec.exists():
            self.skipTest("spec not in this checkout")
        t = spec.read_text(encoding="utf-8")
        for tok in ("+0.04%", "+0.97%", "−3.87%", "38–50%", "97%"):
            self.assertIn(tok, t)



class TestLeaderModeV2(unittest.TestCase):
    def setUp(self):
        self.src = (HERE / "_backtest_leader_mode_v2.py").read_text(encoding="utf-8")

    def test_parse_no_machine_paths(self):
        ast.parse(self.src)
        self.assertNotIn("D:/", self.src)

    def test_variants_fixed_and_all_reported(self):
        # pre-registered: a fixed list, every variant printed on train AND test, no sort/selection
        self.assertIn("VARIANTS = [", self.src)
        self.assertIn('print("%-30s TRAIN %s"', self.src)
        self.assertNotIn(".sort(", self.src)

    def test_leadership_exit_waits_one_hour(self):
        self.assertIn("q - i >= 4 and RANK[q, j] > 10", self.src)

    def test_spec_v2(self):
        spec = HERE.parent / "docs/specs/features/leader-mode-spec.md"
        if not spec.exists():
            self.skipTest("spec not in this checkout")
        t = spec.read_text(encoding="utf-8")
        for tok in ("## v2", "70%", "−1.24% [−2.50, +0.08]", "−0.88% [−1.67, −0.08]"):
            self.assertIn(tok, t)


if __name__ == "__main__":
    unittest.main(verbosity=2)
