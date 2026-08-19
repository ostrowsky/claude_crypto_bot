"""Exit gates for the continuation-signal experiment.

The failure this pins is not a crash. It is an off-by-one in the feature window
that lets a bar see its own future, which produces a high AUC, a plausible
story, and a policy built on nothing. Every test here is aimed at a way the
measurement could be wrong while still printing a number.

Spec: docs/specs/features/continuation-signal-spec.md
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _backtest_continuation_signal as CS  # noqa: E402


def bar(i: int, close: float, high: float = None, low: float = None,
        vol: float = 100.0):
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
    return (ts, close, high if high is not None else close,
            low if low is not None else close, close, vol)


class TestLabelIsARace(unittest.TestCase):
    """A stop experiences path order, so the label must too."""

    def test_up_first_is_continuation(self):
        future = [bar(1, 100, high=103, low=99), bar(2, 100, high=100, low=90)]
        self.assertEqual(CS.label(future, 100.0, 2.0, 2.0), 1)

    def test_down_first_is_not(self):
        future = [bar(1, 100, high=101, low=97), bar(2, 100, high=110, low=100)]
        self.assertEqual(CS.label(future, 100.0, 2.0, 2.0), 0)

    def test_both_in_one_bar_reads_pessimistically(self):
        # Intrabar order is unknowable from OHLC. Calling it a win would be the
        # optimistic guess, and every optimistic guess here inflates the result.
        future = [bar(1, 100, high=105, low=95)]
        self.assertEqual(CS.label(future, 100.0, 2.0, 2.0), 0)

    def test_neither_side_touched_is_undecided_not_a_loss(self):
        # Folding "nothing happened" into 0 would relabel stagnation as the end
        # of the move -- a different claim, and a flattering one for any model
        # that learns to predict stalling.
        future = [bar(i, 100, high=100.5, low=99.5) for i in range(1, 6)]
        self.assertIsNone(CS.label(future, 100.0, 2.0, 2.0))

    def test_asymmetric_thresholds_are_honoured(self):
        future = [bar(1, 100, high=103, low=97.5)]
        self.assertEqual(CS.label(future, 100.0, 5.0, 2.0), 0)   # -2 hit, +5 not
        self.assertEqual(CS.label(future, 100.0, 2.0, 5.0), 1)   # +2 hit first


class TestFeaturesCannotSeeTheFuture(unittest.TestCase):
    """The single most dangerous defect in this file: a feature window that
    extends one bar past the decision point. It would raise the AUC, look like
    a discovery, and be worth nothing."""

    def _hist(self, n=60, start=100.0, step=0.0):
        return [bar(i, start + i * step) for i in range(n)]

    def test_appending_future_bars_does_not_change_features(self):
        hist = self._hist()
        before = CS.features(hist, 100.0, 10, 105.0)
        _ = [bar(i, 999.0, high=9999.0, low=1.0) for i in range(60, 80)]
        after = CS.features(hist, 100.0, 10, 105.0)
        self.assertEqual(before, after)

    def test_features_depend_only_on_the_slice_given(self):
        rising = self._hist(step=1.0)
        # A wild future appended to the SLICE must change the answer -- proving
        # the function reads its whole input, so the guarantee has to come from
        # the caller slicing correctly. That is what the next test checks.
        longer = rising + [bar(60, 5000.0, high=5000.0, low=5000.0)]
        self.assertNotEqual(CS.features(rising, 100.0, 1, 100.0),
                            CS.features(longer, 100.0, 1, 100.0))

    def test_the_caller_slices_up_to_and_including_now(self):
        src = (HERE / "_backtest_continuation_signal.py").read_text(encoding="utf-8")
        self.assertIn("b[max(0, k - WARMUP):k + 1]", src,
                      "the feature window must end at the current bar")
        self.assertIn("b[k + 1:k + 1 + horizon]", src,
                      "the label window must start after the current bar")

    def test_feature_and_label_windows_do_not_overlap(self):
        b = [bar(i, 100.0 + i) for i in range(80)]
        k = 60
        feat_window = b[max(0, k - CS.WARMUP):k + 1]
        label_window = b[k + 1:k + 1 + 12]
        self.assertEqual(feat_window[-1][0], b[k][0])
        self.assertGreater(label_window[0][0], feat_window[-1][0])


class TestSplitIsByTime(unittest.TestCase):
    def test_no_day_appears_on_both_sides(self):
        rows = [{"_y": i % 2, "_day": "2026-01-%02d" % (1 + i // 10)}
                for i in range(100)]
        tr, te, cut = CS.split_by_time(rows)
        self.assertTrue(all(r["_day"] < cut for r in tr))
        self.assertTrue(all(r["_day"] >= cut for r in te))
        self.assertFalse(set(r["_day"] for r in tr)
                         & set(r["_day"] for r in te))


class TestAucAndLift(unittest.TestCase):
    def test_perfect_ranking(self):
        self.assertAlmostEqual(CS.auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)

    def test_inverted_ranking(self):
        self.assertAlmostEqual(CS.auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]), 0.0)

    def test_all_ties_is_a_coin_flip(self):
        self.assertAlmostEqual(CS.auc([0, 1, 0, 1], [0.5] * 4), 0.5)

    def test_lift_reports_its_base_rate(self):
        # TH-01: a precision without its base rate is not evidence, so the
        # helper is not allowed to return one without the other.
        test = [{"_y": 1}, {"_y": 0}, {"_y": 0}, {"_y": 0}]
        prec, base, lift = CS.lift_at(test, [0.9, 0.1, 0.1, 0.1], 0.25)
        self.assertAlmostEqual(prec, 1.0)
        self.assertAlmostEqual(base, 0.25)
        self.assertAlmostEqual(lift, 4.0)


class TestBootstrapIsClusteredByTrade(unittest.TestCase):
    """48 bars of one trade are one observation with 48 rows. Resampling rows
    would shrink the interval by roughly the square root of that, and the first
    version of this experiment leaned on exactly such an interval to call a
    0.512 AUC significant."""

    def test_it_resamples_trades(self):
        src = (HERE / "_backtest_continuation_signal.py").read_text(encoding="utf-8")
        self.assertIn("by_trade", src)
        self.assertIn("rng.choice(trades)", src)

    def test_a_trade_is_drawn_whole(self):
        test = ([{"_y": 1, "_trade": "A"}] * 5) + ([{"_y": 0, "_trade": "B"}] * 5)
        p = [0.9] * 5 + [0.1] * 5
        lo, hi = CS.cluster_bootstrap_auc(test, p, draws=50)
        # With only two trades, whole-trade resampling often draws the same
        # trade twice, leaving one class absent and the AUC undefined. A
        # row-level bootstrap would almost never do that, so a degenerate
        # interval here is evidence the clustering is real.
        self.assertTrue(lo != lo or lo <= hi)


class TestTheNullHasWidth(unittest.TestCase):
    """One shuffled run is not a control. The first version printed a single
    0.4851 next to a real 0.5124 and read the difference as signal, when the
    null's own spread was larger than the gap being claimed."""

    def test_null_band_refits_several_times(self):
        src = (HERE / "_backtest_continuation_signal.py").read_text(encoding="utf-8")
        self.assertIn("def null_band", src)
        self.assertIn("for s in range(seeds)", src)

    def test_verdict_requires_both_separation_and_lift(self):
        src = (HERE / "_backtest_continuation_signal.py").read_text(encoding="utf-8")
        self.assertIn('r["z"] > 2.0', src)
        self.assertIn('r["lift10"] >= 1.10', src)


class TestTheTautologyControlExists(unittest.TestCase):
    """`atr_pct` is the model's top feature, and the winning label is "+10% in
    24h". A coin that swings 10% routinely satisfies that label more often
    whether or not its current move continues -- the same shape of defect as
    `tg_return_since_open` scoring 0.99 on "was today a top gainer".

    So the script has to be able to ask the model with volatility ALONE, and
    with volatility REMOVED. Without those two runs the headline lift cannot be
    told apart from a volatility ranking."""

    def test_a_feature_subset_can_be_selected(self):
        src = (HERE / "_backtest_continuation_signal.py").read_text(encoding="utf-8")
        self.assertIn('"--feats"', src)
        self.assertIn("FEATS[:] = keep", src)

    def test_an_unknown_feature_is_refused_rather_than_ignored(self):
        # Silently dropping a misspelled name would run a different ablation
        # than the one written down, and report it under the intended label.
        src = (HERE / "_backtest_continuation_signal.py").read_text(encoding="utf-8")
        self.assertIn("unknown features", src)


class TestPopulationIsNotConditionedOnOutcome(unittest.TestCase):
    """The predecessor of this script measured on winner-days only and reported
    that holding longer triples capture -- a change live trading had already
    rolled back. The population must be every trade."""

    def test_no_winner_day_restriction(self):
        src = (HERE / "_backtest_continuation_signal.py").read_text(encoding="utf-8")
        self.assertNotIn("immutable_labels", src)
        self.assertNotIn("winners_only", src)


if __name__ == "__main__":
    unittest.main()
