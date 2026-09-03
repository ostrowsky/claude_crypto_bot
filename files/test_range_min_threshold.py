"""Guards for the daily-range floor measurement.

The measurement said 4% stays and 2% would admit half-quality candidates. These
tests protect the ways that conclusion could be an artefact of how it was counted
rather than of the market.

Spec: docs/specs/features/range-min-threshold-spec.md
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _backtest_range_min_threshold as R  # noqa: E402

SRC = (HERE / "_backtest_range_min_threshold.py").read_text(encoding="utf-8")


class TestBandsPartitionCleanly(unittest.TestCase):
    """Overlapping or gapped bands would double-count or silently drop
    candidates, and the 2-4% figure is a sum of two of them."""

    def test_bands_are_contiguous_with_no_gaps(self):
        for (a_lo, a_hi), (b_lo, b_hi) in zip(R.BANDS, R.BANDS[1:]):
            self.assertEqual(a_hi, b_lo,
                             "gap or overlap between %s and %s" % (a_hi, b_lo))

    def test_the_first_band_starts_at_zero(self):
        self.assertEqual(R.BANDS[0][0], 0.0)

    def test_membership_is_half_open_so_edges_land_once(self):
        # A daily_range of exactly 2.0 must fall in 2-3 and nowhere else.
        hits = [(lo, hi) for lo, hi in R.BANDS if lo <= 2.0 < hi]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0], (2.0, 3.0))

    def test_the_change_under_test_is_exactly_two_bands(self):
        # 4% -> 2% admits [2,3) and [3,4); the report sums precisely those.
        self.assertIn('by_band.get((2.0, 3.0))', SRC)
        self.assertIn('by_band.get((3.0, 4.0))', SRC)


class TestTheCountingCannotFlatterAGate(unittest.TestCase):
    def test_decisions_are_deduplicated_by_symbol_hour(self):
        # A guard that re-fires every poll would otherwise weight its own
        # opinion by how often the loop happened to run that hour.
        self.assertIn("seen.setdefault((sym, hour, ev, rc)", SRC)
        self.assertIn("deduplicated by symbol-hour", SRC)

    def test_both_baselines_are_reported(self):
        # POOL answers "better than coin-blind in the same hours"; ENTRIES TAKEN
        # answers "as good as what the bot already buys". One alone decides
        # nothing.
        self.assertIn("POOL (all cand)", SRC)
        self.assertIn("ENTRIES TAKEN", SRC)

    def test_volume_per_day_is_reported_beside_quality(self):
        # Admitting more candidates costs MAX_OPEN slots; quality alone would
        # hide half the trade.
        self.assertIn('"/day"', SRC)

    def test_thin_bands_are_suppressed_not_reported(self):
        self.assertIn("if len(v) < 20:", SRC)
        self.assertIn("too thin to judge", SRC)


class TestForwardMeasurement(unittest.TestCase):
    def test_it_measures_the_peak_not_the_close(self):
        # The target is the day's largest MOVE; a run given back was still a run.
        body = SRC.split("def peak(")[1].split("\ndef ")[0]
        self.assertIn("max(b[2] for b in fut)", body)

    def test_it_only_looks_at_bars_after_the_decision(self):
        body = SRC.split("def peak(")[1].split("\ndef ")[0]
        self.assertIn("b[0] > when", body)

    def test_it_refuses_a_half_resolved_window(self):
        # A candidate with two of eight bars available would otherwise report a
        # small peak and read as a bad candidate.
        body = SRC.split("def peak(")[1].split("\ndef ")[0]
        self.assertIn("len(fut) < max(2, hours // 2)", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
