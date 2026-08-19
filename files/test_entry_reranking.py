"""Exit gates for the fixed-budget entry reranking evidence (TH-01 / TH-06).

The property that matters: the comparison cannot flatter itself. The budget is
held fixed per day, the split is by time, and no ratio is reported without the
random control that says whether it means anything.

Spec: docs/specs/features/entry-reranking-spec.md
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _backtest_entry_reranking as R  # noqa: E402


def rows(spec):
    """`spec` = list of (day, score, label)."""
    return [{"day": d, "sym": f"S{i}", "x": {"s": sc}, "y": y}
            for i, (d, sc, y) in enumerate(spec)]


class TestBudgetIsPerDay(unittest.TestCase):
    def test_a_quiet_day_is_not_scored_against_a_busy_one(self):
        # One day with 10 alerts and one with 2: a global top-k would take the
        # whole busy day and ignore the quiet one.
        spec = [("d1", i, 0) for i in range(10)] + [("d2", 99, 1), ("d2", 1, 0)]
        p, w = R.precision_at_budget(rows(spec), lambda r: r["x"]["s"], 0.5)
        # d1 contributes 5 losers, d2 contributes its single winner
        self.assertEqual(w, 1)
        self.assertAlmostEqual(p, 100.0 / 6, places=4)

    def test_full_budget_keeps_everything(self):
        spec = [("d1", 3, 1), ("d1", 1, 0), ("d2", 2, 0)]
        p, w = R.precision_at_budget(rows(spec), lambda r: r["x"]["s"], 1.0)
        self.assertEqual(w, 1)
        self.assertAlmostEqual(p, 100.0 / 3, places=4)

    def test_a_day_always_keeps_at_least_one(self):
        # Rounding a 1-alert day to zero would silently drop the day.
        spec = [("d1", 5, 1)]
        _, w = R.precision_at_budget(rows(spec), lambda r: r["x"]["s"], 0.1)
        self.assertEqual(w, 1)


class TestPerfectAndReversedOrderings(unittest.TestCase):
    def _spec(self):
        # 4 alerts/day, 1 winner, winner has the highest score
        return [("d1", 9, 1), ("d1", 3, 0), ("d1", 2, 0), ("d1", 1, 0)]

    def test_a_perfect_signal_reaches_100_percent(self):
        p, _ = R.precision_at_budget(rows(self._spec()), lambda r: r["x"]["s"], 0.25)
        self.assertEqual(p, 100.0)

    def test_a_reversed_signal_reaches_zero(self):
        p, _ = R.precision_at_budget(rows(self._spec()),
                                     lambda r: -r["x"]["s"], 0.25)
        self.assertEqual(p, 0.0)
        # This is why "BELOW the random band" is a real category and not just a
        # weak signal: ranker_final_score and ranker_ev land there.


class TestRandomControlIsReportedWithEveryRatio(unittest.TestCase):
    def test_band_is_a_range_not_a_point(self):
        spec = [(f"d{d}", i, 1 if i == 0 else 0)
                for d in range(20) for i in range(6)]
        mean, lo, hi = R.random_band(rows(spec), 0.5)
        self.assertLess(lo, hi, "a control with no spread cannot judge anything")
        self.assertTrue(lo <= mean <= hi)

    def test_control_is_deterministic(self):
        spec = [(f"d{d}", i, 1 if i == 0 else 0)
                for d in range(10) for i in range(4)]
        self.assertEqual(R.random_band(rows(spec), 0.5),
                         R.random_band(rows(spec), 0.5))


class TestScriptDiscipline(unittest.TestCase):
    def setUp(self):
        self.src = (HERE / "_backtest_entry_reranking.py").read_text(encoding="utf-8")

    def test_split_is_by_time_not_random(self):
        self.assertIn("TRAIN_FRAC", self.src)
        self.assertIn('rows.sort(key=lambda r: r["day"])', self.src)

    def test_only_entry_time_fields_are_used(self):
        # Nothing may enter the ranking that the decision did not have.
        for banned in ("pnl", "exit_price", "eod_return", "ret_5"):
            self.assertNotIn(f'"{banned}"', self.src)

    def test_the_selection_caveat_is_printed_not_just_specced(self):
        self.assertIn("already PASSED every", self.src)


if __name__ == "__main__":
    unittest.main()
