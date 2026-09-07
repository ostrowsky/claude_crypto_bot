"""Guards for the peak-label measurement.

The result is extraordinary — the live label ranks big movers LAST, AUC 0.27-0.33
at every cut — and extraordinary results in this repo have three times this week
turned out to be artefacts of how they were counted. These tests protect the
counting.

Spec: docs/specs/features/peak-training-label-spec.md
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _backtest_gate_floor_remap as R  # noqa: E402

REMAP = (HERE / "_backtest_gate_floor_remap.py").read_text(encoding="utf-8")
LABEL = (HERE / "_backtest_peak_label.py").read_text(encoding="utf-8")


def bar(i, close, high=None, low=None):
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
    return (ts, close, high if high is not None else close,
            low if low is not None else close, close, 100.0)


class TestPeakCannotSeeItsOwnBar(unittest.TestCase):
    """The features describe bar t. If the peak included bar t's own high the
    label would contain the thing it is meant to predict."""

    def setUp(self):
        # bar 0 has a huge high; the next five are flat. A correct peak is ~0%.
        self.bars = [bar(0, 100.0, high=200.0)] + [bar(i, 100.0) for i in range(1, 12)]
        R._IDX[("TESTUSDT", "1h")] = (self.bars,
                                      {b[0]: i for i, b in enumerate(self.bars)})

    def tearDown(self):
        R._IDX.pop(("TESTUSDT", "1h"), None)

    def test_the_signal_bars_own_high_is_excluded(self):
        p = R.peak_ahead("TESTUSDT", "1h", self.bars[0][0], 5)
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p, 0.0, places=6)

    def test_a_later_high_is_included(self):
        self.bars[3] = bar(3, 100.0, high=110.0)
        R._IDX[("TESTUSDT", "1h")] = (self.bars,
                                      {b[0]: i for i, b in enumerate(self.bars)})
        p = R.peak_ahead("TESTUSDT", "1h", self.bars[0][0], 5)
        self.assertAlmostEqual(p, 10.0, places=6)

    def test_a_truncated_window_returns_none_rather_than_a_short_peak(self):
        # Two of five bars available would report a small peak and read as a bad
        # candidate, which is a silent bias against the most recent rows.
        last = self.bars[-2][0]
        self.assertIsNone(R.peak_ahead("TESTUSDT", "1h", last, 5))

    def test_an_unknown_timestamp_is_none(self):
        self.assertIsNone(R.peak_ahead("TESTUSDT", "1h",
                                       datetime(2020, 1, 1, tzinfo=timezone.utc), 5))


class TestTheComparisonIsFair(unittest.TestCase):
    def test_both_models_are_graded_against_the_same_truth(self):
        # Not each on its own label -- that would compare two different exams.
        self.assertIn("te_peaks >= 3.0", LABEL)
        self.assertIn("graded against the SAME truth", LABEL)

    def test_the_old_model_is_compared_at_equal_admit_rate(self):
        # Comparing at the same NUMBER would confound ordering with where the
        # distributions happen to sit; the quantile makes it about ordering.
        self.assertIn("thr_o = float(np.quantile(so, 1.0 - float(m15.mean())))", REMAP)

    def test_a_no_model_baseline_is_reported(self):
        self.assertIn("NO MODEL (base)", LABEL)
        self.assertIn("no gate", REMAP)

    def test_stability_runs_more_than_one_cut(self):
        # One split is one observation; the inversion has to survive several.
        self.assertIn("for frac in (0.50, 0.60, 0.70, 0.80)", REMAP)
        self.assertIn("for frac in (0.50, 0.60, 0.70, 0.80)", LABEL)


class TestTheSplitIsByTime(unittest.TestCase):
    def test_rows_are_sorted_before_cutting(self):
        self.assertIn('out.sort(key=lambda rp: rp[0]["_dt"])', REMAP)

    def test_the_scaler_is_fitted_on_train_only(self):
        # Fitting on everything leaks the holdout's distribution into training.
        self.assertIn("sc = M.StandardScaler().fit(Xtr)", REMAP)
        self.assertNotIn("StandardScaler().fit(np.vstack([Xtr, Xte]))", REMAP)


class TestScopeIsStated(unittest.TestCase):
    def test_the_downstream_gates_are_named_in_the_output(self):
        # Admit rates here are the ML floor alone; reading them as trade counts
        # would overstate what the change does.
        self.assertIn("trend_quality, trend_chop", REMAP)

    def test_the_population_limit_is_stated(self):
        self.assertIn("bot's own candidates", LABEL.replace("bot's OWN", "bot's own")
                      ) if "bot's" in LABEL else self.assertIn("population", LABEL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
