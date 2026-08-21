"""Guards for the section-7 gate test and the bandit it convicted.

CLAUDE.md section 7 states the rule: a filter that blocks the eventual winners is
broken. Applying it needs two things that are easy to get wrong, and both were
gotten wrong at least once in this project already:

  a baseline -- "rejects rose 24% of the time" is meaningless until the pool's own
  rate sits next to it, because in a rising market everything rises;

  deduplication -- a gate that re-fires every poll would otherwise weight itself
  up. 98 rejections of CRV in one day is one opinion about CRV, not 98.

The measurement that followed found the bandit rejecting candidates that
outperform the ones the bot buys (1.52% vs 1.41% median forward peak), which is
why BANDIT_ENABLED is now False.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config  # noqa: E402

SRC = (HERE / "_backtest_gate_overblocking.py").read_text(encoding="utf-8")
CFG = (HERE / "config.py").read_text(encoding="utf-8")


class TestTheBanditVetoIsOff(unittest.TestCase):
    def test_the_flag_is_false(self):
        self.assertFalse(config.BANDIT_ENABLED)

    def test_the_evidence_travels_with_the_flag(self):
        # A bare `False` in a config file is a decision nobody can audit later.
        for token in ("1.52%", "1.41%", "13 332", "Rollback = True"):
            self.assertIn(token, CFG,
                          "config lost the evidence for disabling the bandit: %s" % token)

    def test_only_the_entry_veto_is_disabled(self):
        # Trail bandit and offline training are separate machinery and were not
        # part of what the measurement convicted.
        self.assertIn("Trail bandit and offline training are untouched", CFG)


class TestTheMeasurementHasABaseline(unittest.TestCase):
    """Without the pool row every gate looks fine on a rising day."""

    def test_the_pool_row_is_computed_and_printed(self):
        self.assertIn("ALL CANDIDATES (base)", SRC)
        self.assertIn("base3 = 100.0 * sum", SRC)

    def test_a_gate_is_flagged_only_against_that_baseline(self):
        self.assertIn("s3 > base3 * 1.15", SRC)

    def test_entries_are_measured_alongside_rejects(self):
        # The sharpest form of the question: if a gate's rejects outrun what the
        # bot actually took, the gate is on the wrong side of its own pipeline.
        self.assertIn('ev if ev == "entry" else rc', SRC)


class TestDeduplicationIsLoadBearing(unittest.TestCase):
    def test_rows_are_collapsed_per_symbol_hour_gate(self):
        self.assertIn("seen.setdefault((sym, hour, tag)", SRC)

    def test_the_reason_is_written_down(self):
        self.assertIn("opinion about CRV, not 98", SRC)

    def test_the_hour_bucket_is_actually_truncated(self):
        self.assertIn("replace(minute=0, second=0, microsecond=0)", SRC)


class TestForwardMeasurementIsPeakNotClose(unittest.TestCase):
    def test_it_takes_the_peak_over_the_horizon(self):
        # The target is the day's largest MOVE; a run that was given back was
        # still a run the bot should have caught.
        self.assertIn("max(b[2] for b in fut)", SRC)

    def test_rows_without_enough_forward_bars_are_dropped(self):
        self.assertIn("if len(fut) < max(2, hours // 2)", SRC)


if __name__ == "__main__":
    unittest.main(verbosity=2)
