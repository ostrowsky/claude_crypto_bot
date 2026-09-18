"""Guards for the zero-forecast measurement.

The verdict is "leave the gate alone", which nobody re-checks. The two things
most likely to have faked it are pinned here: the split between "no data" and
"bad forecast", and the kline source that silently dropped September.

Spec: docs/specs/features/trend-quality-zero-forecast-spec.md
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _backtest_trend_quality_zero_forecast as Z  # noqa: E402

SRC = (HERE / "_backtest_trend_quality_zero_forecast.py").read_text(encoding="utf-8")


def ev(reason=None, event="blocked", tf="15m", code=None):
    e = {"event": event, "tf": tf, "reason": reason}
    if code:
        e["reason_code"] = code
    return e


class TestTheSplitIsExact(unittest.TestCase):
    def test_exact_zero_is_no_data(self):
        r = "trend quality guard: weak 15m trend (forecast 0.000 < 0.250, vol 1.0)"
        self.assertEqual(Z.classify(ev(r)), "NO DATA")

    def test_a_computed_forecast_is_not_no_data(self):
        # -0.157 and 0.051 are real opinions; lumping them with the default
        # would blur the only distinction this file exists to draw.
        for f in ("-0.157", "0.051", "0.249"):
            r = "trend quality guard: weak 15m trend (forecast %s < 0.250)" % f
            self.assertEqual(Z.classify(ev(r)), "LOW FCST", f)

    def test_other_trend_quality_reasons_are_not_counted(self):
        # price edge / RSI / daily_range rejections never reach the forecast
        # check; counting them as NO DATA would inflate the population.
        r = "trend quality guard: price edge 5.08% > 4.00%"
        self.assertIsNone(Z.classify(ev(r, code="trend_quality")))

    def test_only_15m_is_in_scope(self):
        r = "trend quality guard: weak 15m trend (forecast 0.000 < 0.250)"
        self.assertIsNone(Z.classify(ev(r, tf="1h")))

    def test_a_later_gate_counts_as_passed(self):
        self.assertEqual(Z.classify(ev("x", code="mode_range_quality")), "PASSED")
        self.assertEqual(Z.classify(ev(None, event="entry")), "PASSED")

    def test_an_earlier_gate_is_not_passed(self):
        # ml_zone runs BEFORE trend_quality; those candidates were never judged.
        self.assertIsNone(Z.classify(ev("x", code="ml_proba_zone")))


class TestTheKlineSourceCoversTheWindow(unittest.TestCase):
    """The first run lost 4104 rows and all of September because the loader read
    a backfill that ends 2026-08-20. The merge must see both files."""

    def _write(self, fp, rows):
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write("ts,open,high,low,close,volume\n")
            for ts, px in rows:
                fh.write("%s,%s,%s,%s,%s,1\n" % (ts.isoformat(), px, px, px, px))

    def test_both_files_are_merged_and_the_newer_one_wins(self):
        t0 = datetime(2026, 8, 20, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self._write(d / "XUSDT_15m_419d.csv",
                        [(t0 + timedelta(minutes=15 * i), 1.0) for i in range(4)])
            self._write(d / "XUSDT_15m.csv",
                        [(t0 + timedelta(minutes=15 * i), 2.0) for i in range(2, 8)])
            prev = Z.TD.HISTORY
            Z.TD.HISTORY = d
            Z._IDX.pop("XUSDT", None)
            try:
                bars, idx = Z.index_15m("XUSDT")
            finally:
                Z.TD.HISTORY = prev
                Z._IDX.pop("XUSDT", None)
        self.assertEqual(len(bars), 8)                  # 0..7, no duplicates
        self.assertEqual(bars[0][4], 1.0)               # only in the backfill
        self.assertEqual(bars[3][4], 2.0)               # overlap: rolling wins
        self.assertEqual(bars[-1][0], t0 + timedelta(minutes=105))


class TestThePeakIsHonest(unittest.TestCase):
    def setUp(self):
        t0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
        self.t0 = t0
        bars = [(t0 + timedelta(minutes=15 * i), 100.0, 100.0, 100.0, 100.0, 1.0)
                for i in range(10)]
        bars[0] = (t0, 100.0, 150.0, 100.0, 100.0, 1.0)   # decision bar spikes
        bars[3] = (bars[3][0], 100.0, 110.0, 100.0, 100.0, 1.0)
        Z._IDX["PKUSDT"] = (bars, {b[0]: i for i, b in enumerate(bars)})

    def tearDown(self):
        Z._IDX.pop("PKUSDT", None)

    def test_the_decision_bar_is_excluded(self):
        p = Z.forward_peak("PKUSDT", self.t0 + timedelta(minutes=7), 100.0, 5)
        self.assertAlmostEqual(p, 10.0, places=6)

    def test_a_truncated_window_is_dropped_not_scored_short(self):
        late = self.t0 + timedelta(minutes=15 * 8)
        self.assertIsNone(Z.forward_peak("PKUSDT", late, 100.0, 5))


class TestTheVerdictTravelsWithTheFile(unittest.TestCase):
    def test_verdict_and_numbers_are_recorded(self):
        self.assertIn("VERDICT 2026-09-18: REFUTED", SRC)
        for token in ("2877", "0.94x", "4.4%", "5.9%"):
            self.assertIn(token, SRC)

    def test_the_comparability_caveat_is_printed(self):
        self.assertIn("not a matched", SRC)


if __name__ == "__main__":
    unittest.main(verbosity=2)
