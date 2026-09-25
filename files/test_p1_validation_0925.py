"""Guards for the P1 validation of the 2026-09-25 audit package.

Spec: docs/specs/features/p1-validation-0925-spec.md
"""
import ast
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SPEC = HERE.parent / "docs" / "specs" / "features" / "p1-validation-0925-spec.md"
SCRIPTS = ("_backtest_1h_coins_on_15m.py", "_backtest_ranker_and_score_floor.py", "_backtest_reenable_ema_cross.py")


class TestScripts(unittest.TestCase):
    def test_parse_and_no_machine_paths(self):
        for s in SCRIPTS + ("_audit_winner_funnel.py",):
            src = (HERE / s).read_text(encoding="utf-8")
            ast.parse(src)
            self.assertNotIn("D:/", src, s)

    def test_replays_do_not_count_ema_cross_as_live(self):
        # EMA_CROSS_ENABLED is False live; the replay checks it last, so its rows are cross-only bars
        for s in ("_backtest_1h_coins_on_15m.py", "_audit_winner_funnel.py"):
            src = (HERE / s).read_text(encoding="utf-8")
            self.assertIn('== "ema_cross"', src, s)
            self.assertIn("continue", src[src.index('== "ema_cross"'):][:200], s)

    def test_ema_cross_is_still_off(self):
        import config
        self.assertIs(config.EMA_CROSS_ENABLED, False)

    def test_cross_only_candidates_become_alignment(self):
        # the E-5 backtest trails them as alignment and applies the alignment range floor
        src = (HERE / "monitor.py").read_text(encoding="utf-8")
        start = src.index("        if any_signal:\n            early_15m_continuation")
        seg = src[start:src.index('preview_mode = "alignment"', start)]
        self.assertTrue(seg.rstrip().endswith("else:"))        # the fall-through branch
        self.assertNotIn("cross_ok", seg)                         # no mode of its own
        self.assertIn('floor_pct("alignment")', (HERE / "_backtest_reenable_ema_cross.py").read_text(encoding="utf-8"))

    def test_the_bonus_is_added_before_the_logged_score(self):
        # E-3 subtracts clip(final)*weight from the LOGGED candidate_score
        src = (HERE / "monitor.py").read_text(encoding="utf-8")
        self.assertIn("candidate_score += _ml_candidate_ranker_runtime_bonus(ranker_info)", src)


class TestSpec(unittest.TestCase):
    def setUp(self):
        if not SPEC.exists():
            self.skipTest("spec not in this checkout")
        self.text = SPEC.read_text(encoding="utf-8")

    def test_numbers(self):
        for tok in ("49.4%", "10.2%", "12.3%", "13.4%", "0.340", "0.693", "+0.97 п.п. [+0.57, +1.40]",
                    "15.8%", "−0.097", "332"):
            self.assertIn(tok, self.text)

    def test_verdicts(self):
        for tok in ("ОПРОВЕРГНУТА", "рано", "P3", "не доказана"):
            self.assertIn(tok, self.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
