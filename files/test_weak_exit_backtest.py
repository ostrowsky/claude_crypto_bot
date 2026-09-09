"""Guards for the WEAK-exit counterfactual.

The result is a refusal to change anything, which is the kind of result nobody
re-checks. So the counting has to be pinned: a replay that quietly favours the
trail would have produced the opposite verdict and shipped a live exit change.

Spec: docs/specs/features/weak-exit-above-breakeven-spec.md
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _backtest_weak_exit_above_breakeven as W  # noqa: E402

SRC = (HERE / "_backtest_weak_exit_above_breakeven.py").read_text(encoding="utf-8")


class Cfg:
    TRAIL_MIN_BUFFER_PCT_ENABLED = True
    H5_BREAK_EVEN_PCT = 0.5
    # a 10% floor for the synthetic mode, so these tests set the trail width
    # explicitly instead of depending on whatever ATR the fake bars imply
    TRAIL_MIN_BUFFER_PCT_TEST = 0.10


def bar(i, close, high=None, low=None):
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
    return (ts, close, high if high is not None else close,
            low if low is not None else close, close, 100.0)


class TestTheStopIsTestedBeforeThePeakMoves(unittest.TestCase):
    """The ordering bug that silently inflates every trailing backtest: if the
    peak absorbs this bar's high first, the bar's own spike widens the stop that
    the same bar then fails to hit."""

    def test_a_bar_that_spikes_then_collapses_still_stops_out(self):
        # flat at 100, then one bar with a high of 130 and a low of 80.
        bars = [bar(i, 100.0) for i in range(20)]
        bars.append(bar(20, 85.0, high=130.0, low=80.0))
        bars += [bar(i, 85.0) for i in range(21, 30)]
        # width 10% off a peak of 100 puts the stop at 90; the low of 80 hits it.
        out = W.replay(bars, 19, 15, 100.0, trail_k=0.0, mode="test",
                       cfg=Cfg, max_bars=5)
        self.assertIsNotNone(out)
        # if the peak had moved to 130 first, the stop would be 117 -- and the
        # replay would report a LOSS from a much higher stop, not this one.
        self.assertAlmostEqual(out, -10.0, places=6)

    def test_the_source_keeps_that_order(self):
        i_stop = SRC.index("if lo <= stop:")
        i_peak = SRC.index("peak = max(peak, hi)")
        self.assertLess(i_stop, i_peak,
                        "the stop must be tested before the peak absorbs the bar")


class TestTheCounterfactualIsBounded(unittest.TestCase):
    def test_the_peak_is_seeded_from_the_run_since_entry(self):
        # The real trail would already have been tracking the high; seeding at
        # the exit bar alone would hand the counterfactual a free reset.
        bars = [bar(i, 100.0) for i in range(15)]
        bars[5] = bar(5, 100.0, high=140.0)          # the run's high, before exit
        bars += [bar(i, 100.0, low=100.0) for i in range(15, 25)]
        out = W.replay(bars, 14, 3, 100.0, trail_k=0.0, mode="test",
                       cfg=Cfg, max_bars=5)
        # peak 140, width 10% -> stop 126, already above price: stops immediately
        self.assertIsNotNone(out)
        self.assertGreater(out, 20.0)

    def test_an_unresolvable_window_returns_none_not_zero(self):
        bars = [bar(i, 100.0) for i in range(5)]     # too short for ATR14
        self.assertIsNone(W.replay(bars, 4, 0, 100.0, 2.0, None, Cfg, 96))

    def test_a_zero_width_trail_is_refused(self):
        # trail_k 0 and no mode floor would mean "stop at the peak" -- an exit
        # rule that books the high of the run and flatters the counterfactual.
        bars = [bar(i, 100.0) for i in range(30)]
        self.assertIsNone(W.replay(bars, 20, 10, 100.0, 0.0, None, Cfg, 5))


class TestThePopulationsAreHonest(unittest.TestCase):
    def test_a_control_group_exists(self):
        # WEAK below break-even: H5 would not touch it, so it measures how much
        # of any gain is just a rising market.
        self.assertIn("WEAK, below (control)", SRC)

    def test_the_break_even_floor_comes_from_config(self):
        self.assertIn('getattr(cfg, "H5_BREAK_EVEN_PCT", 0.5)', SRC)

    def test_results_are_split_by_month_and_regime(self):
        self.assertIn("BY MONTH", SRC)
        self.assertIn("BY REGIME", SRC)

    def test_the_small_cell_bar_is_stated_in_the_output(self):
        # The September cell that raised the question was n=28; the bar has to be
        # printed so a later reader cannot quote a n=16 cell as a finding.
        self.assertIn("is an anecdote", SRC)

    def test_the_verdict_is_recorded_in_the_file(self):
        # TH-08: a refuted hypothesis carries the numbers that killed it.
        self.assertIn("VERDICT 2026-09-09: REFUTED", SRC)
        for token in ("-0.207% per trade", "699", "btc_up -0.22%"):
            self.assertIn(token, SRC)


class TestWeakDetectionMatchesTheLiveRule(unittest.TestCase):
    def test_the_same_two_markers_the_bot_uses(self):
        # monitor._h5_should_suppress keys off the warning emoji or "weak";
        # classifying differently here would grade a different population.
        mon = (HERE / "monitor.py").read_text(encoding="utf-8")
        self.assertIn('"weak" in r', mon)
        self.assertIn('r.startswith(WARN) or "weak" in r.lower()', SRC)


if __name__ == "__main__":
    unittest.main(verbosity=2)
