"""trend_quality: forecast 0.000 = no data (TREND_15M_QUALITY_ZERO_FORECAST_AS_NO_DATA).

Spec: docs/specs/features/p0-validation-0925-spec.md (section "Выкатка E-4")
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import botlog  # noqa: E402
import config  # noqa: E402
import monitor  # noqa: E402
import pipeline_replay_validator as V  # noqa: E402

# a candidate that passes price edge / daily_range / RSI and fails the alt path
WEAK = dict(tf="15m", mode="trend", price=1.018, ema20=1.0, slope=0.22, adx=19.5,
            rsi=61.0, vol_x=1.02, daily_range=4.2)


def guard(**kw):
    return monitor._trend_entry_quality_guard_reason(**{**WEAK, **kw})


class TestGuard(unittest.TestCase):
    def test_zero_forecast_passes_when_on(self):
        self.assertIsNone(guard(forecast_return_pct=0.0, zero_forecast_as_no_data=True))
        self.assertIsNone(guard(forecast_return_pct=0.0004, zero_forecast_as_no_data=True))

    def test_zero_forecast_blocks_when_off(self):
        r = guard(forecast_return_pct=0.0, zero_forecast_as_no_data=False)
        self.assertIn("weak 15m trend", r)
        self.assertIn("forecast 0.000", r)

    def test_flag_is_read_when_not_passed(self):
        with mock.patch.object(config, "TREND_15M_QUALITY_ZERO_FORECAST_AS_NO_DATA", False):
            self.assertIsNotNone(guard(forecast_return_pct=0.0))
        with mock.patch.object(config, "TREND_15M_QUALITY_ZERO_FORECAST_AS_NO_DATA", True):
            self.assertIsNone(guard(forecast_return_pct=0.0))

    def test_a_real_low_forecast_still_blocks(self):
        for f in (0.0005, 0.05, 0.249, -0.01):
            self.assertIsNotNone(guard(forecast_return_pct=f, zero_forecast_as_no_data=True), f)

    def test_the_earlier_checks_still_block_at_zero_forecast(self):
        r = guard(forecast_return_pct=0.0, zero_forecast_as_no_data=True, price=1.10)   # 10% edge
        self.assertIn("price edge", r)
        r = guard(forecast_return_pct=0.0, zero_forecast_as_no_data=True, rsi=90.0)
        self.assertIn("RSI", r)

    def test_other_modes_untouched(self):
        self.assertIsNone(guard(forecast_return_pct=0.0, zero_forecast_as_no_data=False, mode="alignment"))

    def test_flag_on_with_rollback(self):
        self.assertIs(config.TREND_15M_QUALITY_ZERO_FORECAST_AS_NO_DATA, True)


class TestWiring(unittest.TestCase):
    def setUp(self):
        self.src = (HERE / "monitor.py").read_text(encoding="utf-8")

    def test_relaxed_admissions_are_tagged(self):
        self.assertIn("zero_forecast_as_no_data=False", self.src)
        self.assertIn("botlog.log_tq_zero_forecast_pass(", self.src)
        self.assertIn("tq_zero_forecast_relaxed=True if tq_zero_forecast_relaxed else None", self.src)
        body = self.src[self.src.index("async def _poll_coin("):]
        self.assertLess(body.index("tq_zero_forecast_relaxed = False"), body.index("tq_zero_forecast_relaxed = True"))

    def test_replay_mirror_follows_the_flag(self):
        src = (HERE / "replay_backtest.py").read_text(encoding="utf-8")
        self.assertIn("TREND_15M_QUALITY_ZERO_FORECAST_AS_NO_DATA", src)


class TestLogging(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = mock.patch.object(botlog, "LOG_FILE", Path(self.tmp.name) / "ev.jsonl")
        self.p.start()

    def tearDown(self):
        self.p.stop()
        self.tmp.cleanup()

    def rows(self):
        return [json.loads(x) for x in botlog.LOG_FILE.read_text(encoding="utf-8").splitlines()]

    def test_pass_event(self):
        botlog.log_tq_zero_forecast_pass("AUSDT", "15m", 1.5, bar_ts=123)
        r = self.rows()[0]
        self.assertEqual((r["event"], r["sym"], r["bar_ts"]), ("tq_zero_forecast_pass", "AUSDT", 123))

    def test_entry_field_only_when_relaxed(self):
        base = dict(sym="AUSDT", tf="15m", mode="trend", price=1.0, ema20=1.0, slope=0.1, rsi=50.0,
                    adx=20.0, vol_x=1.0, macd_hist=0.0, daily_range=3.0, trail_k=2.0, max_hold_bars=16)
        botlog.log_entry(**base)
        botlog.log_entry(**base, tq_zero_forecast_relaxed=True)
        a, b = self.rows()
        self.assertNotIn("tq_zero_forecast_relaxed", a)
        self.assertIs(b["tq_zero_forecast_relaxed"], True)


class TestReplayValidator(unittest.TestCase):
    ROW = {"reason": "trend quality guard: weak 15m trend (forecast 0.000 < 0.250, vol 1.02, ADX 19.5, slope 0.220)"}
    CFG = {"TREND_15M_QUALITY_FORECAST_MIN": 0.25, "TREND_15M_QUALITY_ALT_VOL_MIN": 1.2,
           "TREND_15M_QUALITY_ALT_ADX_MIN": 24.0, "TREND_15M_QUALITY_ALT_SLOPE_MIN": 0.35}

    def test_zero_forecast_row_follows_the_flag(self):
        self.assertIs(V._tq_blocks(self.ROW, {**self.CFG, "TREND_15M_QUALITY_ZERO_FORECAST_AS_NO_DATA": True}), False)
        self.assertIs(V._tq_blocks(self.ROW, {**self.CFG, "TREND_15M_QUALITY_ZERO_FORECAST_AS_NO_DATA": False}), True)

    def test_nonzero_forecast_row_unchanged(self):
        row = {"reason": self.ROW["reason"].replace("forecast 0.000", "forecast 0.120")}
        self.assertIs(V._tq_blocks(row, {**self.CFG, "TREND_15M_QUALITY_ZERO_FORECAST_AS_NO_DATA": True}), True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
