"""Guards for the cooldown re-entry measurement (2026-09-25).

Spec: docs/specs/features/cooldown-reentry-spec.md
"""
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = (HERE / "_backtest_cooldown_reentry.py").read_text(encoding="utf-8")
SPEC = (HERE.parent / "docs" / "specs" / "features" / "trend-quality-caps-spec.md")


class TestMethod(unittest.TestCase):
    def test_the_window_is_the_live_cooldown(self):
        self.assertIn('int(getattr(cfg, "COOLDOWN_BARS", 19))', SRC)

    def test_only_the_first_firing_after_an_exit_is_the_candidate(self):
        # every later firing in the same window is removed from the control too,
        # or the control would contain the very rows being tested
        self.assertIn("if first is None:", SRC)
        self.assertIn("in_cd.add(id(rs[k]))", SRC)
        self.assertIn("if id(r) not in in_cd", SRC)

    def test_split_by_exit_class(self):
        for tok in ("IN PROFIT", "AT A LOSS", "WEAK exit", "RSI-overbought exit", "ATR-trail exit"):
            self.assertIn(tok, SRC)


class TestVerdict(unittest.TestCase):
    def test_numbers_travel_with_the_file(self):
        self.assertIn("VERDICT 2026-09-25: REFUTED. The cooldown stays.", SRC)
        for tok in ("1 395 (34%)", "[-0.07, +0.16]", "+1.42%"):
            self.assertIn(tok, SRC)


class TestPreRegistration(unittest.TestCase):
    def test_the_daily_range_retest_is_pre_registered(self):
        if not SPEC.exists():
            self.skipTest("spec not present in this checkout")
        text = SPEC.read_text(encoding="utf-8")
        self.assertIn("Pre-registered re-test", text)
        self.assertIn("rows logged after 2026-09-25", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
