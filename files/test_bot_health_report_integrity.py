"""Focused contracts for honest bot-health reporting.

Run:
    python -X utf8 files/test_bot_health_report_integrity.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bot_health_report as H


def _write_critic(root: Path, day: date, phase: str, **summary) -> None:
    p = root / f"top_gainer_critic_{day.isoformat()}_{phase}.json"
    p.write_text(json.dumps({
        "target_day_local": day.isoformat(),
        "phase": phase,
        "summary": summary,
    }), encoding="utf-8")


class CompletedCriticTests(unittest.TestCase):
    def test_morning_report_falls_back_to_previous_final(self):
        today = date(2026, 8, 13)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_critic(root, today - timedelta(days=1), "final",
                          watchlist_top_count=10,
                          watchlist_top_bought=6)
            with mock.patch.object(H.PL, "REPORTS", root):
                got = H.collect_critic(today)

        self.assertTrue(got["available"])
        self.assertEqual(got["data"]["_phase_used"], "final")
        self.assertEqual(got["data"]["_critic_target_date"], "2026-08-12")
        self.assertEqual(got["data"]["_fallback_days"], 1)

    def test_missing_critic_is_a_red_flag(self):
        flags = H.detect_red_flags(
            {"available": False}, {"available": False},
            {"available": False}, {"available": False},
            {"available": False},
        )
        self.assertIn("RF_critic_unavailable", {f["id"] for f in flags})

    def test_midday_only_critic_is_a_data_quality_flag(self):
        flags = H.detect_red_flags(
            {"available": True}, {"available": False},
            {"available": False}, {"available": False},
            {"available": True, "data": {
                "_phase_used": "midday", "_fallback_days": 0,
            }},
        )
        self.assertIn("RF_critic_partial", {f["id"] for f in flags})

    def test_extra_partial_days_are_an_uptime_flag(self):
        flags = H.detect_red_flags(
            {"available": True}, {"available": False},
            {"available": False}, {"available": False},
            {"available": True, "data": {
                "_phase_used": "final", "_fallback_days": 1,
            }},
            {"metrics": {"NS_EarlyCapture_top20": {
                "days_window": 14, "days_full": 10,
                "days_down_or_partial": 4,
            }}},
        )
        by_id = {f["id"]: f for f in flags}
        self.assertEqual(by_id["RF_uptime_gap"]["value"], 3.0)
        self.assertEqual(by_id["RF_uptime_gap"]["severity"], "critical")


class ComparableProgressTests(unittest.TestCase):
    def test_progress_refuses_differently_filled_working_day_windows(self):
        rows = [
            {"ts": "2026-04-29T07:00:00Z",
             "_compute_early_capture.py": {"metric": "NS_EarlyCapture_top20",
                                             "early_capture": 0.07}},
            {"ts": "2026-08-07T07:00:00Z",
             "_compute_early_capture.py": {"metric": "NS_EarlyCapture_top20",
                                             "days_window": 14, "days_full": 5,
                                             "early_capture": 0.1088}},
            {"ts": "2026-08-07T09:00:00Z",
             "_compute_early_capture.py": {"metric": "NS_EarlyCapture_top20",
                                             "days_window": 14, "days_full": 5,
                                             "early_capture": 0.1088}},
            {"ts": "2026-08-13T08:00:00Z",
             "_compute_early_capture.py": {"metric": "NS_EarlyCapture_top20",
                                             "days_window": 14, "days_full": 10,
                                             "early_capture": 0.0702}},
        ]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "metrics.jsonl"
            p.write_text("\n".join(json.dumps(x) for x in rows) + "\n",
                         encoding="utf-8")
            with mock.patch.object(H.PL, "METRICS_DAILY", p):
                history = H._ns_history()
                verdict = H._progress_verdict()

        self.assertEqual(history, [("2026-08-07", 0.1088),
                                   ("2026-08-13", 0.0702)])
        self.assertEqual(verdict[1], "РАНО СУДИТЬ")
        self.assertIn("5 рабочим дням", verdict[2])
        self.assertIn("10", verdict[2])

    def test_progress_can_compare_equally_filled_current_schema_windows(self):
        rows = [
            {"ts": "2026-08-07T07:00:00Z",
             "_compute_early_capture.py": {"metric": "NS_EarlyCapture_top20",
                                             "days_window": 14, "days_full": 10,
                                             "early_capture": 0.1088}},
            {"ts": "2026-08-13T08:00:00Z",
             "_compute_early_capture.py": {"metric": "NS_EarlyCapture_top20",
                                             "days_window": 14, "days_full": 10,
                                             "early_capture": 0.0702}},
        ]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "metrics.jsonl"
            p.write_text("\n".join(json.dumps(x) for x in rows) + "\n",
                         encoding="utf-8")
            with mock.patch.object(H.PL, "METRICS_DAILY", p):
                verdict = H._progress_verdict()

        self.assertEqual(verdict[1], "СТАЛО ХУЖЕ")
        self.assertIn("~11 → ~7", verdict[2])


class RenderingIntegrityTests(unittest.TestCase):
    @staticmethod
    def _report() -> dict:
        return {
            "target_date": "2026-08-13",
            "deployment_health": {
                "available": True,
                "critic_target_date": "2026-08-12",
                "phase": "final",
            },
            "metrics_daily_latest": {"metrics": {
                "NS_EarlyCapture_top20": {
                    "early_capture": 0.0702,
                    "decomp_coverage": 0.6154,
                    "decomp_capture_mean": 0.1702,
                    "decomp_time_lead_mean": 0.7083,
                    "days_window": 14,
                    "days_full": 10,
                },
                "C1_C2_coverage_funnel": {
                    "coverage_pct_raw": 61.54,
                    "silent_miss_pct": 11.54,
                },
            }},
            "mode_curtail": {"available": False},
            "red_flags": [],
        }

    def test_capture_is_named_as_main_bottleneck(self):
        with mock.patch.object(H, "_progress_verdict",
                               return_value=("📉", "СТАЛО ХУЖЕ", "~11 → ~7")), \
             mock.patch.object(H, "_action_needed_count", return_value=0), \
             mock.patch.object(H, "_past_decisions_resume", return_value=[]):
            text = H.render_telegram(self._report())

        self.assertIn("Главный тормоз", text)
        self.assertIn("17%", text)
        self.assertIn("каждую ~9-ю", text)
        self.assertIn("final critic: 2026-08-12", text)

    def test_unavailable_deployment_cannot_render_no_alerts(self):
        report = self._report()
        report["deployment_health"] = {"available": False}
        with mock.patch.object(H, "_progress_verdict",
                               return_value=("❔", "ПОКА НЕ ЯСНО", "нет данных")), \
             mock.patch.object(H, "_action_needed_count", return_value=0), \
             mock.patch.object(H, "_past_decisions_resume", return_value=[]):
            text = H.render_telegram(report)

        self.assertNotIn("✅ Тревог нет", text)
        self.assertIn("статус тревог неизвестен", text)

    def test_all_insufficient_attribution_is_not_called_harm(self):
        result = {
            "verdict": "miss",
            "rationale": ["precision: insufficient_data",
                          "recall: insufficient_data"],
        }
        status = H._attribution_status(result)
        self.assertEqual(status, "insufficient_data")


if __name__ == "__main__":
    unittest.main(verbosity=2)
