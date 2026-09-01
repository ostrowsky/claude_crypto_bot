"""Guards for the alignment trail widening (2026-09-01).

A trail widening justified by a backtest has failed live in this repo before:
impulse_speed 1.5% -> 8% in June, rolled back five days later after avg/trade
went +0.02 -> -0.62. These tests pin the things that would make the current
change wrong in the same silent way — the wrong mode moved, the sweep's stop
ordering flattering itself, or the rollback value going missing from the file
someone will read at 3am.

Spec: docs/specs/features/alignment-trail-width-spec.md
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config  # noqa: E402

SWEEP = (HERE / "_backtest_trail_width_sweep.py").read_text(encoding="utf-8")
CFG = (HERE / "config.py").read_text(encoding="utf-8")


class TestOnlyAlignmentMoved(unittest.TestCase):
    """The sweep found alignment wants 8% and impulse_speed wants ~1%. Widening
    both would repeat June exactly."""

    def test_alignment_floor_is_eight_percent(self):
        self.assertAlmostEqual(config.TRAIL_MIN_BUFFER_PCT_ALIGNMENT, 0.08, places=6)

    def test_impulse_speed_was_not_touched(self):
        # 8% here lost live in June and loses in the sweep (-0.206 vs +0.212).
        self.assertAlmostEqual(config.TRAIL_MIN_BUFFER_PCT_IMPULSE_SPEED, 0.015,
                               places=6)

    def test_the_losing_modes_were_not_widened(self):
        # trend / breakout / strong_trend lose at EVERY width, so their exit rule
        # is not the thing that is wrong with them.
        self.assertAlmostEqual(config.TRAIL_MIN_BUFFER_PCT_TREND, 0.0, places=6)
        self.assertAlmostEqual(config.TRAIL_MIN_BUFFER_PCT_BREAKOUT, 0.0, places=6)

    def test_the_mechanism_is_still_enabled(self):
        # The floor is inert without this, so the change would be a silent no-op.
        self.assertTrue(config.TRAIL_MIN_BUFFER_PCT_ENABLED)


class TestTheEvidenceTravelsWithTheNumber(unittest.TestCase):
    def test_rollback_is_written_next_to_the_value(self):
        self.assertIn("Rollback = 0.0.", CFG)

    def test_the_sample_size_is_recorded(self):
        self.assertIn("1323 real alignment entries", CFG)

    def test_the_june_precedent_is_recorded(self):
        # Whoever reads this at 3am must see that this exact kind of change has
        # failed live here before.
        self.assertIn("ROLLED BACK after a live regression", CFG)


class TestTheSweepCannotFlatterItself(unittest.TestCase):
    def test_a_bar_cannot_forgive_its_own_drawdown(self):
        # The stop must be tested against the low BEFORE the peak absorbs that
        # bar's high; otherwise a single spiky bar both sets a new peak and
        # escapes the drawdown it printed, inflating every width's result.
        body = SWEEP.split("def replay(")[1].split("\ndef ")[0]
        stop_at = body.index("if lo <= stop:")
        peak_at = body.index("peak = max(peak, hi)")
        self.assertLess(stop_at, peak_at,
                        "the peak is updated before the stop is tested")

    def test_realized_and_available_are_both_reported(self):
        # Capture alone would pick the widest stop every time; realized alone
        # would pick the tightest. Neither may be summarised away (TH-11).
        self.assertIn('"realized"', SWEEP)
        self.assertIn('"available"', SWEEP)
        self.assertIn('"capture"', SWEEP)

    def test_a_no_stop_baseline_is_printed(self):
        self.assertIn("HOLD (none)", SWEEP)

    def test_thin_samples_are_flagged_not_hidden(self):
        self.assertIn("THIN", SWEEP)

    def test_the_population_limit_is_stated_in_the_output(self):
        self.assertIn("says nothing about the", SWEEP)


if __name__ == "__main__":
    unittest.main(verbosity=2)
