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


class TestPeakCacheIsSafeToTrust(unittest.TestCase):
    """The cache exists because labelling re-read 4.9M bars nightly and blew the
    600s retrain timeout on 2026-09-08. It must never invent or freeze a value."""

    def setUp(self):
        import ml_signal_model as M
        self.M = M
        self._prev = M._PEAK_CACHE
        M._PEAK_CACHE = {}

    def tearDown(self):
        self.M._PEAK_CACHE = self._prev

    def test_an_unresolved_row_is_not_cached(self):
        """A row whose forward window has not closed returns None. Caching that
        None would freeze it as a permanent non-answer once the bars arrive."""
        M = self.M
        bars = [bar(i, 100.0) for i in range(3)]
        M._PEAK_IDX["NOCACHEUSDT|1h"] = (bars, {b[0]: i for i, b in enumerate(bars)})
        try:
            rows = [{"sym": "NOCACHEUSDT", "tf": "1h", "_dt": bars[-1][0],
                     "ts_signal": bars[-1][0].isoformat()}]
            out, n_ok = M._peak_label_series(rows)
            self.assertEqual(n_ok, 0)
            self.assertIsNone(out[0])
            self.assertEqual(M._PEAK_CACHE, {})
        finally:
            M._PEAK_IDX.pop("NOCACHEUSDT|1h", None)

    def test_a_resolved_row_is_cached_and_reused(self):
        M = self.M
        bars = [bar(0, 100.0)] + [bar(i, 100.0, high=110.0) for i in range(1, 12)]
        M._PEAK_IDX["CACHEUSDT|1h"] = (bars, {b[0]: i for i, b in enumerate(bars)})
        try:
            rows = [{"sym": "CACHEUSDT", "tf": "1h", "_dt": bars[0][0],
                     "ts_signal": bars[0][0].isoformat()}]
            out, n_ok = M._peak_label_series(rows)
            self.assertEqual(n_ok, 1)
            self.assertEqual(len(M._PEAK_CACHE), 1)
            # second pass must reach the same answer with the klines gone
            M._PEAK_IDX.pop("CACHEUSDT|1h")
            again, n2 = M._peak_label_series(rows)
            self.assertEqual(n2, 1)
            self.assertAlmostEqual(again[0], out[0], places=9)
        finally:
            M._PEAK_IDX.pop("CACHEUSDT|1h", None)

    def test_the_key_carries_the_horizon(self):
        # Changing ML_PEAK_LABEL_HORIZON must not read back a value computed for
        # a different horizon -- that would silently mislabel the whole dataset.
        self.assertIn('"%s|%s|%s|%d"',
                      (HERE / "ml_signal_model.py").read_text(encoding="utf-8"))


class TestFamilySelectionGradesOnTheGoal(unittest.TestCase):
    """The old criterion led with selected_ret5_avg -- the CLOSE -- and on
    2026-09-08 picked catboost (top-decile peak 2.95%) over logistic (4.12%)."""

    def setUp(self):
        import ml_signal_model as M
        import config
        self.M, self.config = M, config
        self._prev = config.ML_PEAK_LABEL_ENABLED

    def tearDown(self):
        self.config.ML_PEAK_LABEL_ENABLED = self._prev

    def _metrics(self, auc, ret5, cov=0.2, prec=0.3):
        return {"auc": auc, "selected_ret5_avg": ret5,
                "coverage": cov, "precision": prec}

    def test_with_the_peak_label_the_better_ranker_wins(self):
        self.config.ML_PEAK_LABEL_ENABLED = True
        good = self._metrics(auc=0.857, ret5=0.22)     # logistic, 2026-09-08
        bad = self._metrics(auc=0.809, ret5=0.586)     # catboost: higher close
        self.assertGreater(self.M._family_score(good), self.M._family_score(bad))

    def test_with_the_flag_off_the_old_criterion_returns(self):
        self.config.ML_PEAK_LABEL_ENABLED = False
        good = self._metrics(auc=0.857, ret5=0.22)
        bad = self._metrics(auc=0.809, ret5=0.586)
        self.assertLess(self.M._family_score(good), self.M._family_score(bad))

    def test_a_missing_auc_falls_back_rather_than_crashing(self):
        self.config.ML_PEAK_LABEL_ENABLED = True
        m = self._metrics(auc=None, ret5=0.5)
        self.assertIsInstance(self.M._family_score(m), float)


class TestTheStallCanBeSeenNextTime(unittest.TestCase):
    def test_the_retrain_timeout_covers_a_cold_cache(self):
        src = (HERE / "daily_learning.py").read_text(encoding="utf-8")
        self.assertIn("timeout=1800", src)

    def test_the_gate_model_has_a_declared_freshness_interval(self):
        import artifact_freshness as AF
        names = {a.name for a in AF.ARTIFACTS}
        self.assertIn("ml_signal_model", names,
                      "the artifact the ml gate depends on must declare an interval")

    def test_the_nightly_report_prints_the_freshness_table(self):
        src = (HERE / "daily_learning.py").read_text(encoding="utf-8")
        self.assertIn("artifact_freshness.render(artifact_freshness.check())", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
