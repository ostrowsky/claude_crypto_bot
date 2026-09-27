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


if __name__ == "__main__":
    unittest.main(verbosity=2)
