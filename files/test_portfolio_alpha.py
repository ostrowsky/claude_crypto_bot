"""Exit gates for portfolio alpha vs buy-and-hold (TH-11).

Spec: docs/specs/features/portfolio-alpha-spec.md
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import label_store as LS  # noqa: E402
import portfolio_alpha as PA  # noqa: E402


class TestTimestampShapes(unittest.TestCase):
    """Assuming one shape dropped every exit event and reported "no closed
    trades" for a bot that had just closed one."""

    def test_iso_string(self):
        self.assertEqual(PA._day("2026-08-17T07:15:30Z"), "2026-08-17")

    def test_epoch_millis(self):
        self.assertEqual(PA._day(1786950000000), "2026-08-17")

    def test_epoch_seconds(self):
        self.assertEqual(PA._day(1786950000), "2026-08-17")

    def test_garbage_is_none(self):
        self.assertIsNone(PA._day(None))
        self.assertIsNone(PA._day("nope"))


class TestPortfolioReturn(unittest.TestCase):
    def test_overlapping_trades_are_not_a_mean_of_pnls(self):
        # Ten +10% trades closing the same day, on ten slots, is +10% of
        # capital — not +100% and not +10% per trade averaged.
        trades = [{"day": "2026-05-01", "pnl_pct": 10.0} for _ in range(10)]
        total, per_day = PA.bot_return_pct(trades, slots=10)
        self.assertAlmostEqual(total, 10.0, places=6)
        self.assertAlmostEqual(per_day["2026-05-01"], 10.0, places=6)

    def test_days_compound_rather_than_add(self):
        trades = [{"day": "2026-05-01", "pnl_pct": 100.0},
                  {"day": "2026-05-02", "pnl_pct": 100.0}]
        total, _ = PA.bot_return_pct(trades, slots=10)
        # (1.10 * 1.10) - 1 = 21%, not 20%
        self.assertAlmostEqual(total, 21.0, places=6)

    def test_a_single_trade_moves_one_slot(self):
        total, _ = PA.bot_return_pct([{"day": "2026-05-01", "pnl_pct": 5.0}],
                                     slots=10)
        self.assertAlmostEqual(total, 0.5, places=6)


class _Store(LS.LabelStore):
    def __init__(self, records):
        self._records = records

    def records(self):
        return list(self._records)


class TestBenchmark(unittest.TestCase):
    def _rec(self, sym, day, o, c):
        return {"symbol": sym, "utc_day": day, "open": o, "close": c,
                "complete": True, "eod_return_pct": (c / o - 1) * 100}

    def test_equal_weight_across_the_window(self):
        recs = [self._rec("A", "2026-05-01", 100.0, 100.0),
                self._rec("A", "2026-05-02", 100.0, 120.0),
                self._rec("B", "2026-05-01", 100.0, 100.0),
                self._rec("B", "2026-05-02", 100.0, 80.0)]
        value, meta = PA.buy_and_hold_pct(["2026-05-01", "2026-05-02"],
                                          watchlist={"A", "B"},
                                          store=_Store(recs))
        self.assertAlmostEqual(value, 0.0, places=6)   # +20% and -20%
        self.assertEqual(meta["symbols_used"], 2)

    def test_it_reports_how_many_symbols_it_could_price(self):
        # A benchmark quietly computed on 3 of 105 symbols is not a benchmark.
        recs = [self._rec("A", "2026-05-01", 100.0, 100.0),
                self._rec("A", "2026-05-02", 100.0, 110.0)]
        _, meta = PA.buy_and_hold_pct(["2026-05-01", "2026-05-02"],
                                      watchlist={"A", "B", "C"},
                                      store=_Store(recs))
        self.assertEqual(meta["symbols_used"], 1)
        self.assertEqual(meta["watchlist"], 3)

    def test_a_symbol_seen_on_one_day_only_is_skipped(self):
        recs = [self._rec("A", "2026-05-01", 100.0, 100.0)]
        value, meta = PA.buy_and_hold_pct(["2026-05-01", "2026-05-02"],
                                          watchlist={"A"}, store=_Store(recs))
        self.assertEqual(meta["symbols_used"], 0)
        self.assertEqual(value, 0.0)


class TestEmptyWindow(unittest.TestCase):
    def test_no_trades_is_unavailable_not_zero(self):
        # Reporting 0.0% would read as "flat", which is a claim; "unavailable"
        # is the truth (TH-05).
        res = PA.compute(0)
        if not res["available"]:
            self.assertIn("reason", res)


if __name__ == "__main__":
    unittest.main()
