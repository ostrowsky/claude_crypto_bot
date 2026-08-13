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

    def test_training_gap_is_fail_closed_without_temporal_holdout(self):
        gap = H.compute_training_to_live_gap(
            {"evaluation_scope": "in_sample_post_fit", "recall_at_20": 1.0},
            {"watchlist_top_bought_pct": 0.25},
        )
        self.assertFalse(gap["available"])
        self.assertEqual(gap["reason"], "training_metric_not_out_of_sample")

    def test_invalid_training_evidence_is_a_red_flag(self):
        flags = H.detect_red_flags(
            {"available": True}, {"available": False},
            {"available": False, "reason": "training_metric_not_out_of_sample"},
            {"available": False},
            {"available": True, "data": {"_phase_used": "final", "_fallback_days": 1}},
            training={"available": True, "evaluation_scope": "in_sample_post_fit"},
        )
        self.assertIn("RF_training_evidence_invalid", {f["id"] for f in flags})

    def test_zero_denominator_is_unknown_not_divided_by_one(self):
        critic = {"available": True, "data": {
            "_phase_used": "final", "_critic_target_date": "2026-08-12",
            "_fallback_days": 1, "_source_file": "fixture.json",
            "summary": {
                "watchlist_top_count": 0, "watchlist_top_bought": 0,
                "watchlist_top_early_captured": 0,
                "bot_unique_buys": 0, "bot_false_positive_buys": 0,
            },
        }}
        with mock.patch.object(H, "collect_training_health", return_value={"available": False}), \
             mock.patch.object(H, "collect_critic", return_value=critic), \
             mock.patch.object(H, "collect_critic_baseline", return_value={"available": False}), \
             mock.patch.object(H, "collect_per_mode_signals", return_value={"available": False}), \
             mock.patch.object(H, "collect_metrics_daily_latest", return_value={"available": False}), \
             mock.patch.object(H, "collect_scout_gates", return_value={"available": False}), \
             mock.patch.object(H, "collect_mode_curtail", return_value={"available": False}), \
             mock.patch.object(H.PL, "load_do_not_touch", return_value={}):
            report = H.build_report(date(2026, 8, 13))

        self.assertIsNone(report["deployment_health"]["watchlist_top_bought_pct"])
        self.assertIsNone(report["deployment_health"]["false_positive_rate"])
        self.assertEqual(report["north_star"]["metric"], "EarlyCapture@top20")
        self.assertEqual(report["north_star"]["status"], "provisional")
        self.assertIn("deployment_critic_diagnostic", report)
        self.assertIn("RF_top_mover_denominator_unknown",
                      {f["id"] for f in report["red_flags"]})


class TrainingEvidenceTests(unittest.TestCase):
    def test_legacy_recall_without_context_is_suppressed(self):
        row = {
            "ts": "2026-08-13T02:00:00Z",
            "bandit_recall_top20": 1.0,
            "model_auc_top20": 0.99,
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "learning.jsonl"
            p.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with mock.patch.object(H.PL, "LEARNING_PROGRESS", p):
                got = H.collect_training_health(date(2026, 8, 13))

        self.assertIsNone(got["recall_at_20"])
        self.assertTrue(got["legacy_ratio_suppressed"])

    def test_contextual_postfit_diagnostic_is_kept_but_not_promoted(self):
        row = {
            "ts": "2026-08-13T02:00:00Z",
            "bandit_recall_top20": 0.9,
            "bandit_action_rate": 0.6,
            "bandit_top20_base_rate": 0.1,
            "bandit_recall_lift": 1.5,
            "bandit_precision": 0.15,
            "bandit_evaluation_scope": "in_sample_post_fit",
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "learning.jsonl"
            p.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with mock.patch.object(H.PL, "LEARNING_PROGRESS", p):
                got = H.collect_training_health(date(2026, 8, 13))

        self.assertEqual(got["recall_at_20"], 0.9)
        gap = H.compute_training_to_live_gap(got, {"watchlist_top_bought_pct": 0.4})
        self.assertFalse(gap["available"])

    def test_scorecard_keeps_portfolio_alpha_unknown_and_prioritizes_ground_truth(self):
        score = H.build_canonical_scorecard({"metrics": {
            "NS_EarlyCapture_top20": {"early_capture": 0.07, "n": 26,
                                         "days_window": 14, "days_full": 10},
        }})
        self.assertIsNone(score["portfolio_alpha"]["value"])
        self.assertEqual(score["north_star"]["status"], "provisional")
        steps = H.derive_next_steps(score, {}, {}, date(2026, 8, 13))
        self.assertEqual(steps[0]["id"], "restore_eod_ground_truth")


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
                verdict = H._progress_verdict(ground_truth_verified=True)

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
                verdict = H._progress_verdict(ground_truth_verified=True)

        self.assertEqual(verdict[1], "СТАЛО ХУЖЕ")
        self.assertIn("~11 → ~7", verdict[2])

    def test_progress_is_not_claimed_for_provisional_labels(self):
        verdict = H._progress_verdict(ground_truth_verified=False)
        self.assertEqual(verdict[1], "МЕТРИКА ПРЕДВАРИТЕЛЬНАЯ")
        self.assertIn("immutable later-EOD", verdict[2])


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
        self.assertIn("каждое 9-е", text)
        self.assertIn("critic: 2026-08-12 · final", text)
        self.assertIn("составной score", text)
        self.assertIn("Ground truth пока provisional", text)

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

    def test_learning_block_names_diagnostic_and_refuses_gap(self):
        report = {
            "training_health": {
                "available": True,
                "recall_at_20": 1.0,
                "action_rate": 0.73,
                "base_rate": 0.20,
                "lift": 1.37,
                "precision": 0.27,
                "evaluation_scope": "in_sample_post_fit",
                "auc": 0.99,
                "model_evaluation_scope": "time_sorted_row_holdout_same_snapshot_label",
                "model_label_timing": "same_snapshot_current_24h_leaderboard",
                "model_label_encoding_features": ["tg_return_since_open"],
            },
            "training_to_live_gap": {
                "available": False,
                "reason": "training_metric_not_out_of_sample",
            },
        }
        text = H._render_learning_block(report)
        self.assertIn("diagnostic", text)
        self.assertIn("ENTER=73.0%", text)
        self.assertIn("gap неизвестен", text)
        self.assertNotIn("модель: находит", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
