"""Exit gates for the trend-start detector.

The result this protects is a catch rate of 95.7% against a random baseline of
13.8-28.7%. Every way that number could be wrong runs through one of these:
a label that peeks forward past its own bar, a give-back check ordered so a bar
forgives its own drawdown, an alert credited to a trend it fired outside of, or
a catch rate reported without the random baseline that makes it meaningful.

Spec: docs/specs/features/trend-start-detector-spec.md
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _backtest_trend_start_detector as TD  # noqa: E402

SRC = (HERE / "_backtest_trend_start_detector.py").read_text(encoding="utf-8")


def bar(i, close, high=None, low=None, vol=100.0):
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
    return (ts, close, high if high is not None else close,
            low if low is not None else close, close, vol)


class TestLabelIsTheZigZagDefinition(unittest.TestCase):
    def test_reaching_the_target_cleanly_is_a_one(self):
        bars = [bar(0, 100)] + [bar(i, 100 + i) for i in range(1, 8)]
        self.assertEqual(TD.will_run(bars, 0, 5.0, 2.0, 0), 1)

    def test_a_give_back_before_the_target_is_a_zero(self):
        bars = [bar(0, 100), bar(1, 103, high=103), bar(2, 100, low=100.9)]
        self.assertEqual(TD.will_run(bars, 0, 5.0, 2.0, 0), 0)

    def test_give_back_is_measured_from_the_running_peak_not_from_entry(self):
        # Up to 104, then down to 102: that is -1.9% from the peak, still alive.
        bars = [bar(0, 100), bar(1, 104, high=104, low=104),
                bar(2, 102, high=102, low=102.1), bar(3, 106, high=106)]
        self.assertEqual(TD.will_run(bars, 0, 5.0, 2.0, 0), 1)

    def test_a_bar_cannot_forgive_its_own_drawdown(self):
        # One bar prints a new high AND a deep low. Updating the peak before
        # testing the give-back would let that bar excuse itself and inflate the
        # positive rate; the low must be judged against the PREVIOUS peak.
        bars = [bar(0, 100), bar(1, 100, high=104, low=97)]
        self.assertEqual(TD.will_run(bars, 0, 5.0, 2.0, 0), 0)

    def test_unresolved_within_a_finite_horizon_is_none(self):
        bars = [bar(0, 100)] + [bar(i, 100.5) for i in range(1, 5)]
        self.assertIsNone(TD.will_run(bars, 0, 5.0, 2.0, 3))

    def test_horizon_zero_runs_to_resolution(self):
        # Multi-week trends are targets too, so the unbounded form must keep
        # walking rather than give up at an arbitrary bar count.
        bars = [bar(0, 100)] + [bar(i, 100 + i * 0.1) for i in range(1, 200)]
        self.assertEqual(TD.will_run(bars, 0, 5.0, 2.0, 0), 1)
        self.assertIsNone(TD.will_run(bars, 0, 5.0, 2.0, 10))

    def test_running_out_of_data_unresolved_is_none_not_zero(self):
        bars = [bar(0, 100), bar(1, 101)]
        self.assertIsNone(TD.will_run(bars, 0, 5.0, 2.0, 0))


class TestFeaturesCannotSeeTheFuture(unittest.TestCase):
    def test_a_feature_row_depends_only_on_bars_up_to_it(self):
        base = [bar(i, 100 + (i % 5)) for i in range(200)]
        short = TD.feature_table(base[:150])
        long = TD.feature_table(base)
        # Row 149 must be identical whether or not bars 150..199 exist.
        for k in TD.FEATS:
            self.assertAlmostEqual(short[149][k], long[149][k], places=9,
                                   msg="feature %s leaked from later bars" % k)

    def test_every_declared_feature_is_actually_produced(self):
        rows = TD.feature_table([bar(i, 100 + i * 0.1) for i in range(200)])
        for k in TD.FEATS:
            self.assertIn(k, rows[-1], "declared but never computed: %s" % k)

    def test_the_label_starts_after_the_feature_bar(self):
        self.assertIn("for k in range(i + 1, last):", SRC)


class TestConsolidationFeature(unittest.TestCase):
    """Both charts the operator showed have two to three days of flat range
    before the move, and the earlier feature set could not see it: base_range
    measures how TIGHT a fixed window was, never how LONG the quiet lasted."""

    def test_a_long_flat_base_counts_more_bars_than_a_short_one(self):
        flat = TD.feature_table([bar(i, 100.0) for i in range(200)])
        choppy = TD.feature_table(
            [bar(i, 100.0 if i % 2 else 130.0) for i in range(200)])
        self.assertGreater(flat[-1]["bars_in_base"],
                           choppy[-1]["bars_in_base"])

    def test_it_is_bounded_so_one_dead_coin_cannot_dominate(self):
        rows = TD.feature_table([bar(i, 100.0) for i in range(400)])
        self.assertLessEqual(rows[-1]["bars_in_base"], 200)


class TestCatchAccounting(unittest.TestCase):
    def test_an_alert_counts_only_inside_the_trend_window(self):
        self.assertIn("if st <= ts <= en", SRC)

    def test_remaining_move_is_measured_from_the_alert_price(self):
        # 'ahead%' is the only tradeable quantity; measuring it from the trend
        # start instead would credit the model with the part it missed.
        self.assertIn('(peak / px - 1) * 100', SRC)

    def test_how_far_into_the_move_is_reported(self):
        # A hit at 80% of the way up passes any hit-rate metric and is worth
        # nothing, so the report is not allowed to omit this column.
        self.assertIn('"into"', SRC)
        self.assertIn("how far INTO the move", SRC)

    def test_false_alerts_are_reported_not_buried(self):
        self.assertIn("outside any trend", SRC)


class TestTheRandomBaselineExists(unittest.TestCase):
    """A long trend is easy to hit by accident: firing on 2% of bars at random
    lands inside a 100-bar trend about 87% of the time from its length alone.
    Without this baseline the catch rate measures trend duration."""

    def test_the_baseline_script_is_present_and_matches_the_budget(self):
        p = HERE / "_diag_catch_random_baseline.py"
        self.assertTrue(p.exists(), "the catch rate has no baseline to be read against")
        src = p.read_text(encoding="utf-8")
        self.assertIn("RATE = 0.02", src)
        self.assertIn("random.Random", src)


class TestSplitIsByTime(unittest.TestCase):
    def test_the_cut_is_a_day_boundary_and_train_is_strictly_earlier(self):
        self.assertIn('train = [r for r in rows if r["_day"] < cut]', SRC)
        self.assertIn('test = [r for r in rows if r["_day"] >= cut]', SRC)

    def test_the_null_is_refit_across_seeds(self):
        self.assertIn("shuffle=True", SRC)
        self.assertIn("for s in range(5)", SRC)


class TestPopulationIsUpstreamOfTheBot(unittest.TestCase):
    """Every negative result so far was measured on the bot's own entries and
    therefore on whatever the upstream gates admitted. This target is upstream
    of the bot entirely, so the population must not be the event log."""

    def test_it_does_not_read_the_event_log(self):
        self.assertNotIn("bot_events.jsonl", SRC)
        self.assertIn("UP.watchlist()", SRC)


if __name__ == "__main__":
    unittest.main()
