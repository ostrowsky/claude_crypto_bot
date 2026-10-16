"""correlation_guard re-judged by the goal (corr-guard-goal-1016-spec.md)."""
import unittest
from pathlib import Path

import _backtest_corr_guard_goal as C

ROOT = Path(__file__).resolve().parent.parent


class TestCorrGuardGoal(unittest.TestCase):
    def test_maxdd(self):
        self.assertEqual(C.maxdd([5, -3, -4, 2, -1]), -7)
        self.assertEqual(C.maxdd([1, 2, 3]), 0)

    def test_pre_registration(self):
        self.assertEqual((C.SINCE, C.TAIL_SHARE, C.TAIL_PP), ("2026-07-07", 0.10, 0.10))

    def test_documented(self):
        spec = (ROOT / "docs/specs/features/corr-guard-goal-1016-spec.md").read_text(encoding="utf-8")
        self.assertIn("drawdown", spec)
        for doc in ("CLAUDE.md", "PROJECT_CONTEXT.md"):
            self.assertIn("corr-guard-goal-1016-spec.md", (ROOT / doc).read_text(encoding="utf-8"), doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
