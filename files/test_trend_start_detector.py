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
NL = chr(10)


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
        # The dataset is now numpy columns, so the split is a mask on the day
        # column -- but it must still be a DAY boundary with train strictly
        # earlier, which is the property this test exists to hold.
        self.assertIn('Xtr, mtr = sub(meta["day"] < cut)', SRC)
        self.assertIn('Xte, test = sub(meta["day"] >= cut)', SRC)
        self.assertIn('cut = days[int(len(days) * 0.7)]', SRC)

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


class TestStartLabelIsActuallyWired(unittest.TestCase):
    """This class exists because it was not, and the failure was invisible.

    A patch inserted `start_bars` and reported success without checking; the
    anchor it matched on had drifted, so the function was defined and never
    called. Three runs then printed "label: this bar is within 6h of the start"
    above numbers produced by the FORWARD label -- identical AUC to four
    decimals across supposedly different labels was the only clue.
    """

    def test_start_bars_is_called_and_its_result_used(self):
        self.assertIn("starts = (start_bars(", SRC)
        self.assertIn("y = starts.get(i, 0)", SRC)

    def test_the_middle_of_a_trend_is_a_negative(self):
        # Without this the model keeps scoring confirmation and loses nothing.
        self.assertIn("1 if i <= a + window else 0", SRC)

    def test_the_header_cannot_claim_a_label_the_run_did_not_use(self):
        self.assertIn('if args.label == "start":', SRC)


class TestEarlinessConstraint(unittest.TestCase):
    """Measuring the PRICE of earliness needs the budget to be spent only on
    early bars. Re-ranking instead of suppressing would let a bar at RSI 80 keep
    its slot and hide the cost."""

    def test_high_rsi_bars_are_suppressed_not_reranked(self):
        self.assertIn('np.where(rsi_col >= args.max_rsi, 0.0, probs)', SRC)
        self.assertIn('rsi_col = Xte[:, FEATS.index("rsi")]', SRC)

    def test_it_reports_how_much_of_the_universe_it_removed(self):
        self.assertIn("test bars remain eligible", SRC)


class TestTimeframeIsAppliedEverywhereOrNowhere(unittest.TestCase):
    """The 15m run must move the WHOLE experiment onto the 15m grid.

    Features on 15m while trends are still detected on 1h would score a fine
    grid against a coarse label, and every catch would be credited or denied by
    a mismatch rather than by the model.
    """

    def test_features_labels_and_trends_all_read_the_same_loader(self):
        # CS.bars inside load_bars IS the 1h branch; what must not exist is a
        # second, unrouted call that ignores --tf.
        self.assertEqual(SRC.count("CS.bars(sym)"), 1,
                         "a code path still reads 1h bars without going "
                         "through load_bars")
        self.assertIn("def load_bars(sym: str, tf: str)", SRC)
        self.assertIn("b = load_bars(sym, tf)", SRC)

    def test_trend_detection_uses_the_timeframe(self):
        self.assertIn("def trends_tf(", SRC)
        self.assertNotIn("UP.trends_for(sym, args.run", SRC)

    def test_min_duration_scales_with_the_timeframe(self):
        # min_bars=4 means "four hours" on 1h. Left at 4 bars on 15m it would
        # admit one-hour trends and change the population being scored, so the
        # comparison would no longer be about resolution.
        self.assertIn('min_duration_bars=4 * (4 if tf == "15m" else 1)', SRC)

    def test_warmup_scales_with_the_feature_windows(self):
        # WARMUP exists so MA99 is meaningful; scaling the windows without it
        # would train on bars whose longest average is still filling up.
        self.assertIn("warm = WARMUP * sc", SRC)
        self.assertIn("range(warm,", SRC)


class TestWindowScalingIsHonest(unittest.TestCase):
    def test_the_1h_path_is_unchanged_by_default(self):
        # sc defaults to 1, so every committed 1h number stays reproducible.
        self.assertIn("return max(1, int(args.window_scale or 1))", SRC)

    def test_a_high_and_its_low_use_the_same_window(self):
        """Scaling one side of a range and not the other is silent corruption.

        The first patch did exactly that -- lo24 scaled, hi24 left at 23 bars --
        so base_range_24 would have divided a 24*sc-bar high by a 24-bar low and
        the feature would have been a ratio of two different windows.
        """
        import re as _re
        body = SRC.split("def feature_table")[1].split(NL + "def ")[0]
        self.assertEqual(_re.findall(r"i - [0-9]+\)", body), [],
                         "an unscaled literal lookback survives")

    def test_scaling_reaches_every_lookback_not_just_the_averages(self):
        for frag in ("_sma_series(closes, 25*sc)", "_ema_series(closes, 12*sc)",
                     "_rsi_series(closes, 14*sc)", "_atr_pct_series(bars, 14*sc)",
                     "_sma_series(vols, 20*sc)", "rsi[max(0, i - 6*sc)]",
                     "closes[i - 12*sc]", "max(-1, i - 200*sc)"):
            self.assertIn(frag, SRC, "unscaled lookback left behind: %s" % frag)

    def test_the_15m_cache_is_a_separate_file_from_the_committed_ones(self):
        # Folding 15m into CS.bars would silently change what every earlier
        # result was computed on.
        self.assertIn('"%s_15m_419d.csv" % sym', SRC)


if __name__ == "__main__":
    unittest.main()
