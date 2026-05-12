"""Unit tests for the multi-objective constraint addition (RM-15) to
pipeline_attribution.

The contract: a hypothesis that improves its target metric but degrades
Sharpe by >10% relative OR grows max drawdown by >10pp absolute is NOT a
win — verdict must come back as `regression`.

Run:
    pyembed\\python.exe files\\test_pipeline_attribution_multiobj.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pipeline_attribution as A


class SharpeTests(unittest.TestCase):

    def test_positive_returns_positive_sharpe(self):
        s = A._sharpe([1.0, 2.0, 0.5, 1.5, 1.2])
        self.assertIsNotNone(s)
        self.assertGreater(s, 0)

    def test_zero_variance_returns_none(self):
        s = A._sharpe([1.0, 1.0, 1.0, 1.0])
        self.assertIsNone(s)

    def test_too_few_samples_returns_none(self):
        self.assertIsNone(A._sharpe([]))
        self.assertIsNone(A._sharpe([1.0]))

    def test_negative_mean_yields_negative_sharpe(self):
        s = A._sharpe([-1.0, -2.0, -0.5, -1.5])
        self.assertLess(s, 0)


class MaxDrawdownTests(unittest.TestCase):

    def test_monotonic_gains_zero_drawdown(self):
        self.assertEqual(A._max_drawdown_pct([1.0, 1.0, 1.0]), 0.0)

    def test_single_drop_recorded(self):
        # +10%: equity 1.10. Then -5%: equity 1.045. Drawdown = 0.055/1.10 = 0.05
        dd = A._max_drawdown_pct([10.0, -5.0])
        self.assertAlmostEqual(dd, 0.05, places=3)

    def test_drawdown_relative_to_peak_not_origin(self):
        dd = A._max_drawdown_pct([20.0, -20.0])
        self.assertAlmostEqual(dd, 0.20, places=3)

    def test_only_max_drawdown_returned(self):
        rets = [10.0, -3.0, 5.0, -8.0]
        dd = A._max_drawdown_pct(rets)
        self.assertAlmostEqual(dd, 0.0803, places=3)

    def test_empty_returns_none(self):
        self.assertIsNone(A._max_drawdown_pct([]))


def _ts_ms(now: datetime, days_ago: int) -> int:
    return int((now - timedelta(days=days_ago)).timestamp() * 1000)


def _write_critic(rows: list[dict]) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                     delete=False, encoding="utf-8")
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.close()
    return Path(f.name)


def _take_row(*, days_ago: int, pnl: float, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    return {
        "bar_ts":   _ts_ms(now, days_ago),
        "decision": {"action": "take"},
        "labels":   {"trade_exit_pnl": pnl, "ret_5": pnl},
    }


class PortfolioObjectivesTests(unittest.TestCase):
    """The function assumes decision_ts is in the past and looks
    `pre_window_days` BACK from decision_ts and `days_after` FORWARD. In tests
    we set decision_ts = now - 14d so both windows fit inside our synthetic
    history."""

    PRE_DAYS  = 14
    POST_DAYS = 14

    def _setup(self, rows: list[dict]):
        path = _write_critic(rows)
        return mock.patch.object(A, "CRITIC_DATASET", path)

    def _decision_ts(self):
        return datetime.now(timezone.utc) - timedelta(days=self.POST_DAYS)

    def _pre_row(self, days_before_decision: int, pnl: float) -> dict:
        # pre window starts (PRE_DAYS) days before decision_ts.
        # So an event N days before decision is "POST_DAYS + N" days ago from now.
        return _take_row(days_ago=self.POST_DAYS + days_before_decision, pnl=pnl)

    def _post_row(self, days_after_decision: int, pnl: float) -> dict:
        # post window starts AT decision_ts. An event N days after decision
        # is "POST_DAYS - N" days ago from now (still in the past, since
        # decision_ts itself is POST_DAYS days ago).
        return _take_row(days_ago=max(0, self.POST_DAYS - days_after_decision), pnl=pnl)

    def test_insufficient_trades_short_circuits(self):
        rows = [self._pre_row(days_before_decision=2, pnl=0.5)] * 3
        with self._setup(rows):
            r = A._portfolio_objectives(self._decision_ts(), days_after=self.POST_DAYS,
                                         baseline={"pre_window_days": self.PRE_DAYS})
        self.assertEqual(r["status"], "insufficient_trades")
        self.assertNotIn("violations", r)

    def test_no_violations_when_stable(self):
        rows = []
        # Pre: 15 stable trades around +1%
        for i in range(15):
            rows.append(self._pre_row(days_before_decision=2 + (i % 5),
                                       pnl=1.0 - 0.1 * (i % 3)))
        # Post: same shape
        for i in range(15):
            rows.append(self._post_row(days_after_decision=2 + (i % 5),
                                        pnl=1.0 - 0.1 * (i % 3)))
        with self._setup(rows):
            r = A._portfolio_objectives(self._decision_ts(), days_after=self.POST_DAYS,
                                         baseline={"pre_window_days": self.PRE_DAYS})
        self.assertEqual(r["status"], "ok", f"got {r}")
        self.assertEqual(r["violations"], [])

    def test_drawdown_growth_flagged(self):
        rows = []
        # Pre: 15 stable +1% trades (no drawdown)
        for i in range(15):
            rows.append(self._pre_row(days_before_decision=2 + (i % 5), pnl=1.0))
        # Post: 15 deep losses (huge drawdown)
        for i in range(15):
            rows.append(self._post_row(days_after_decision=2 + (i % 5), pnl=-10.0))
        with self._setup(rows):
            r = A._portfolio_objectives(self._decision_ts(), days_after=self.POST_DAYS,
                                         baseline={"pre_window_days": self.PRE_DAYS})
        self.assertEqual(r["status"], "violations", f"got {r}")
        constraints = {v["constraint"] for v in r["violations"]}
        self.assertIn("maxdd_abs_growth", constraints)

    def test_sharpe_drop_flagged(self):
        rows = []
        # Pre: nearly-constant +1% — high Sharpe
        for i in range(15):
            rows.append(self._pre_row(days_before_decision=2 + (i % 5),
                                       pnl=1.0 + 0.01 * (i % 3)))
        # Post: same mean but huge variance — much lower Sharpe
        for i in range(15):
            pnl = 1.0 + (5.0 if i % 2 == 0 else -5.0)
            rows.append(self._post_row(days_after_decision=2 + (i % 5), pnl=pnl))
        with self._setup(rows):
            r = A._portfolio_objectives(self._decision_ts(), days_after=self.POST_DAYS,
                                         baseline={"pre_window_days": self.PRE_DAYS})
        self.assertEqual(r["status"], "violations", f"got {r}")
        constraints = {v["constraint"] for v in r["violations"]}
        self.assertIn("sharpe_relative_drop", constraints)


class ConstantsTests(unittest.TestCase):

    def test_constants_exposed(self):
        self.assertGreater(A.MULTI_OBJ_SHARPE_REL_MAX_DROP, 0)
        self.assertGreater(A.MULTI_OBJ_MAXDD_ABS_MAX_GROWTH, 0)
        self.assertGreaterEqual(A.MULTI_OBJ_MIN_TRADES, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
