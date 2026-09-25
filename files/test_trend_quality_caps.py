"""Guards for the trend_quality caps measurement (2026-09-25).

The replica of the guard must keep the live check ORDER and read the live
thresholds; otherwise "lifting price_edge frees 17 of 381" would be an artefact.

Spec: docs/specs/features/trend-quality-caps-spec.md
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
import _backtest_trend_quality_caps as T  # noqa: E402

MON = (HERE / "monitor.py").read_text(encoding="utf-8")
SRC = (HERE / "_backtest_trend_quality_caps.py").read_text(encoding="utf-8")


def x(pe=1.0, dr=5.0, rsi=60.0, fc=None, vol=2.0, adx=30.0, slope=0.5, bull=False):
    return {"pe": pe, "dr": dr, "rsi": rsi, "fc": fc, "vol": vol, "adx": adx, "slope": slope, "bull": bull}


class TestReplicaMatchesTheLiveGuard(unittest.TestCase):
    def test_live_check_order(self):
        body = MON[MON.index("def _trend_entry_quality_guard_reason"):]
        body = body[:body.index("\ndef ")]
        order = [body.index(t) for t in ("price edge", "daily_range {", "RSI {rsi", "weak 15m trend")]
        self.assertEqual(order, sorted(order))

    def test_thresholds_come_from_config(self):
        self.assertEqual(T.CUR["pe"], config.TREND_15M_QUALITY_PRICE_EDGE_MAX_PCT)
        self.assertEqual(T.CUR["dr_b"], config.TREND_15M_QUALITY_DAILY_RANGE_MAX_BULL_DAY)
        self.assertEqual(T.CUR["rsi"], config.TREND_15M_QUALITY_RSI_MAX)

    def test_first_failing_check_wins(self):
        self.assertEqual(T.guard(x(pe=9, dr=40, rsi=90), T.CUR), "pe")
        self.assertEqual(T.guard(x(dr=40, rsi=90), T.CUR), "dr")
        self.assertEqual(T.guard(x(rsi=90), T.CUR), "rsi")

    def test_bull_day_uses_bull_thresholds(self):
        self.assertIsNone(T.guard(x(pe=3.5, bull=True), T.CUR))
        self.assertEqual(T.guard(x(pe=3.5, bull=False), T.CUR), "pe")

    def test_missing_forecast_takes_the_alt_path(self):
        self.assertIsNone(T.guard(x(fc=None, vol=2.0, adx=30, slope=0.5), T.CUR))
        self.assertEqual(T.guard(x(fc=None, vol=0.5), T.CUR), "weak")

    def test_relaxing_one_cap_still_applies_the_later_checks(self):
        self.assertEqual(T.guard(x(pe=9, dr=40), T.relaxed("pe", 1e9)), "dr")


class TestVerdictTravels(unittest.TestCase):
    def test_numbers(self):
        self.assertIn("VERDICT 2026-09-25: REFUTED as a lever", SRC)
        for tok in ("frees only 17", "[+0.01, +1.38]", "+0.02%", "one of 12", "[-0.05, +0.34]"):
            self.assertIn(tok, SRC)


if __name__ == "__main__":
    unittest.main(verbosity=2)
