"""Exit gates for the goal-3 exit-timing evidence.

The property that matters: the counterfactual cannot be run on a sample chosen
by the outcome. Replaying "hold longer" on winner-days alone produced "an 8%
trail triples capture", contradicting a live rollback of exactly that change.

Spec: docs/specs/features/exit-timing-spec.md
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _backtest_exit_timing as X


class TestPopulationIsSelectable(unittest.TestCase):
    def setUp(self):
        self.src = (HERE / "_backtest_exit_timing.py").read_text(encoding="utf-8")

    def test_all_trades_is_reachable(self):
        self.assertIn("--all-trades", self.src)
        self.assertIn("winners_only", self.src)

    def test_the_bias_is_named_where_the_switch_lives(self):
        # A flag whose danger is documented only in the spec is a flag someone
        # flips without reading the spec.
        self.assertIn("conditions the sample on the outcome", self.src)

    def test_the_header_states_which_population_ran(self):
        self.assertIn("ALL trades", self.src)
        self.assertIn("winner-days only", self.src)


class TestCaptureRefusesToBeUndefined(unittest.TestCase):
    """`realized / (realized + left)` flips sign when realized is negative, and
    prints "-5.8% capture" — not a smaller capture but an undefined one."""

    def test_negative_realized_gives_no_number(self):
        self.assertIn("if realized <= 0 or tot <= 0:",
                      (HERE / "_backtest_exit_timing.py").read_text(encoding="utf-8"))

    def test_source_picks_bars_that_cover_the_day(self):
        # bars(tf) or bars("1h") fell back only when the file was MISSING, so a
        # 15m cache holding 30 days returned non-empty for a June trade and the
        # row silently had no forward path: 81 of 423 had data.
        src = (HERE / "_backtest_exit_timing.py").read_text(encoding="utf-8")
        self.assertIn("def bars_covering(", src)


class TestReplay(unittest.TestCase):
    def _bars(self, prices):
        t0 = datetime(2026, 5, 1, tzinfo=timezone.utc)
        return [(t0 + timedelta(hours=i), p, p * 0.99, p)
                for i, p in enumerate(prices)]

    def test_a_stop_that_never_triggers_exits_at_the_close(self):
        X._BARS[("TSTUSDT", "1h")] = self._bars([100, 101, 102, 103])
        trade = {"sym": "TSTUSDT", "tf": "1h", "day": "2026-05-01",
                 "entry": 100.0, "exit": 100.0,
                 "exit_dt": datetime(2026, 5, 1, tzinfo=timezone.utc)}
        got = X.replay_trailing(trade, 50.0)
        self.assertAlmostEqual(got, 3.0, places=4)

    def test_a_tight_stop_triggers_and_caps_the_loss(self):
        X._BARS[("TSTUSDT", "1h")] = self._bars([100, 90, 80])
        trade = {"sym": "TSTUSDT", "tf": "1h", "day": "2026-05-01",
                 "entry": 100.0, "exit": 100.0,
                 "exit_dt": datetime(2026, 5, 1, tzinfo=timezone.utc)}
        got = X.replay_trailing(trade, 2.0)
        self.assertLess(got, 0.0)
        self.assertGreater(got, -10.0, "the stop must cap it, not ride to the low")

    def test_no_forward_path_returns_none_not_zero(self):
        X._BARS[("NOPEUSDT", "1h")] = []
        trade = {"sym": "NOPEUSDT", "tf": "1h", "day": "2026-05-01",
                 "entry": 100.0, "exit": 100.0,
                 "exit_dt": datetime(2026, 5, 1, tzinfo=timezone.utc)}
        self.assertIsNone(X.replay_trailing(trade, 5.0))


if __name__ == "__main__":
    unittest.main()
