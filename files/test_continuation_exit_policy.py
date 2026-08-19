"""Exit gates for the continuation exit-policy replay.

A replay is a simulation, and a simulation that is subtly wrong produces a
number in exactly the same format as one that is right. These pin the four
places where being wrong would look like a result: the policy acting on
information it would not have, the comparison running against a different set
of trades, the exit rule not actually firing, and the duration control not
controlling for duration.

Spec: docs/specs/features/continuation-signal-spec.md
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _backtest_continuation_exit_policy as EP  # noqa: E402


def path_from(closes, probs=None, entry_price=None):
    entry_price = entry_price if entry_price is not None else closes[0]
    steps = []
    for i, c in enumerate(closes):
        s = {"feat": {}, "y": None, "close": c, "day": "2026-01-01",
             "pnl": (c / entry_price - 1) * 100}
        if probs is not None:
            s["p"] = probs[i]
        steps.append(s)
    return {"sym": "XUSDT", "entry": None, "entry_price": entry_price,
            "day": "2026-01-01", "steps": steps}


class TestThresholdPolicy(unittest.TestCase):
    def test_it_leaves_at_the_first_bar_below_threshold(self):
        p = path_from([100, 110, 120, 130], probs=[0.9, 0.9, 0.01, 0.9])
        pnl, held = EP.replay_threshold(p, 0.05)
        self.assertEqual(held, 2)
        self.assertAlmostEqual(pnl, 20.0)

    def test_bar_zero_can_never_be_the_exit(self):
        # The entry decision belongs to a different system. Letting this policy
        # reverse it in the same hour would measure an entry filter and report
        # it as an exit rule.
        p = path_from([100, 150], probs=[0.0, 0.0])
        pnl, held = EP.replay_threshold(p, 0.5)
        self.assertEqual(held, 1)
        self.assertAlmostEqual(pnl, 50.0)

    def test_never_triggering_ends_at_the_time_stop(self):
        # A policy that can decline to ever exit is not a policy, so the last
        # bar is a terminal condition rather than an open position.
        p = path_from([100, 101, 102], probs=[0.9, 0.9, 0.9])
        pnl, held = EP.replay_threshold(p, 0.05)
        self.assertEqual(held, 2)
        self.assertAlmostEqual(pnl, 2.0)

    def test_a_missing_score_does_not_silently_exit(self):
        # `.get("p", 1.0)` defaults to "keep holding". Defaulting to 0 would
        # make every unscored bar an exit and turn a scoring bug into a
        # flattering short-hold policy.
        p = path_from([100, 90, 80])
        pnl, held = EP.replay_threshold(p, 0.5)
        self.assertEqual(held, 2)


class TestFixedTrailReference(unittest.TestCase):
    def test_it_stops_after_a_give_back_from_the_peak(self):
        p = path_from([100, 120, 100])
        pnl, held = EP.replay_fixed_trail(p, 10.0)
        self.assertEqual(held, 2)
        self.assertAlmostEqual(pnl, 0.0)

    def test_a_rising_path_is_not_stopped(self):
        p = path_from([100, 110, 120])
        pnl, held = EP.replay_fixed_trail(p, 10.0)
        self.assertEqual(held, 2)
        self.assertAlmostEqual(pnl, 20.0)


class TestDurationControl(unittest.TestCase):
    """A threshold policy changes holding time, and holding longer has a P&L of
    its own. Without a control matched on duration, a duration effect would be
    reported as a model result."""

    def test_the_control_ignores_every_feature(self):
        p = path_from([100, 105, 130], probs=[0.0, 0.0, 0.0])
        pnl, held = EP.replay_random(p, 1, random.Random(0))
        self.assertEqual(held, 1)
        self.assertAlmostEqual(pnl, 5.0)

    def test_it_never_returns_bar_zero_either(self):
        p = path_from([100, 105])
        _, held = EP.replay_random(p, 0, random.Random(0))
        self.assertEqual(held, 1)

    def test_it_is_clamped_to_the_available_path(self):
        p = path_from([100, 105, 110])
        _, held = EP.replay_random(p, 99, random.Random(0))
        self.assertEqual(held, 2)

    def test_the_control_is_drawn_from_the_policys_own_holds(self):
        src = (HERE / "_backtest_continuation_exit_policy.py").read_text(encoding="utf-8")
        self.assertIn('pool = best["holds"]', src)
        self.assertIn("rng.choice(pool)", src)


class TestComparisonIsLikeForLike(unittest.TestCase):
    def test_summarise_compares_aligned_trades(self):
        res = [1.0, 2.0, 3.0]
        act = [0.0, 5.0, 1.0]
        d = EP.summarise("x", res, act)
        self.assertEqual(d["n"], 3)
        self.assertAlmostEqual(d["beats"], 2 / 3)
        self.assertAlmostEqual(d["median"], 2.0)

    def test_the_replay_trains_only_on_earlier_days(self):
        src = (HERE / "_backtest_continuation_exit_policy.py").read_text(encoding="utf-8")
        self.assertIn('if p["day"] >= cut:', src)
        self.assertIn('continue', src)
        self.assertIn('test = [p for p in paths if p["day"] >= cut]', src)

    def test_the_embargo_is_on_the_bar_not_only_the_trade(self):
        # A trade entered before the cut runs on for up to 48 bars, so its later
        # bars land on the test days. Filtering only by entry day leaves the
        # holdout adjacent to training rather than separate from it.
        src = (HERE / "_backtest_continuation_exit_policy.py").read_text(encoding="utf-8")
        self.assertIn('if s["day"] >= cut:', src)

    def test_the_baseline_is_the_same_trades_not_all_trades(self):
        # Comparing a policy on late trades against actual exits on ALL trades
        # would be the incomparable-windows error wearing a result's clothes.
        src = (HERE / "_backtest_continuation_exit_policy.py").read_text(encoding="utf-8")
        self.assertIn("actual = [a for _, a in have]", src)
        self.assertIn("trades = [p for p, _ in have]", src)

    def test_unmatched_trades_are_counted_not_dropped_silently(self):
        src = (HERE / "_backtest_continuation_exit_policy.py").read_text(encoding="utf-8")
        self.assertIn("actual exit known for %d of %d replayed trades", src)


class TestPathsKeepUndecidedBars(unittest.TestCase):
    """The ranking experiment drops bars whose label is undecided. A policy has
    to act on every bar it holds through, so dropping them here would let the
    simulation skip precisely the ambiguous moments."""

    def test_undecided_bars_are_retained(self):
        src = (HERE / "_backtest_continuation_exit_policy.py").read_text(encoding="utf-8")
        self.assertIn("keeps bars whose LABEL is undecided", src)
        self.assertNotIn('if y is None:\n                continue', src)


if __name__ == "__main__":
    unittest.main()
