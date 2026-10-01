"""Morning report carries no invalid statements (report-validity-1001-spec.md)."""
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import bot_health_report as H

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IMM = "immutable_later_eod_klines"


class TestGroundTruthStatus(unittest.TestCase):
    def test_one_definition(self):
        self.assertTrue(H.ns_ground_truth_verified({"label_provenance": IMM}))
        self.assertFalse(H.ns_ground_truth_verified({"label_provenance": IMM, "primary_degraded": True}))
        self.assertFalse(H.ns_ground_truth_verified({"label_provenance": "rolling_24h_same_snapshot"}))

    def test_no_label_step_when_verified(self):
        sc = {"north_star": {"status": "verified", "days_full": 13, "days_window": 14},
              "portfolio_alpha": {"value": -1.0}, "realized_potential": {"value": 0.6, "target": 0.5},
              "signal_precision": {}, "message_rate": {}}
        steps = H.derive_next_steps(sc, {"evaluation_scope": "out_of_sample_time_holdout"}, {}, date(2026, 10, 1))
        ids = [s["id"] for s in steps]
        self.assertNotIn("restore_eod_ground_truth", ids)
        self.assertNotIn("restore_measurement_coverage", ids)      # past uptime is not an action


class TestSteps(unittest.TestCase):
    def test_unknown_ex1_is_not_a_step_and_numbers_are_rounded(self):
        sc = {"north_star": {"status": "verified"}, "portfolio_alpha": {"value": -1.0},
              "realized_potential": {"value": None, "reason": "thin"},
              "signal_precision": {"value": 13.566739606126914, "target": 35.0},
              "message_rate": {"value": 31.133333333333333, "target_max": 10.0}}
        steps = {s["id"]: s for s in H.derive_next_steps(sc, {"evaluation_scope": "out_of_sample_time_holdout"},
                                                         {}, date(2026, 10, 1))}
        # operator 2026-10-01: only worthwhile steps -- an unknown EX1 is not one
        self.assertNotIn("choose_ex1_reference", steps)
        self.assertNotIn("restore_canonical_ex1", steps)
        self.assertIn("precision=13.6%", steps["honest_alert_budget_ranker"]["evidence"])
        self.assertIn("messages=31.1/d", steps["honest_alert_budget_ranker"]["evidence"])


class TestPastDecisions(unittest.TestCase):
    RES = {"decision_id": "d1", "verdict": "regression",
           "expected_metrics": ["realert_rate", "precision"],
           "unmeasured_expected": ["realert_rate", "precision"],
           "expected_misses": ["realert_rate", "precision"],
           "rationale": ["realert_rate: insufficient_data", "MULTI_OBJ violation"],
           "portfolio_objectives": {"violations": [{"constraint": "maxdd_abs_growth",
                                                    "limit": 0.1, "observed": 0.1003}]}}

    def test_unmeasured_metrics_never_read_as_harm(self):
        self.assertEqual(H._attribution_status(self.RES), "guard_only")
        r = dict(self.RES, portfolio_objectives={})
        self.assertEqual(H._attribution_status(r), "insufficient_data")

    def test_measured_miss_is_still_harm(self):
        r = dict(self.RES, unmeasured_expected=["precision"],
                 rationale=["realert_rate: miss", "precision: insufficient_data"])
        self.assertEqual(H._attribution_status(r), "harmed")


class TestLearningBlock(unittest.TestCase):
    def test_bandit_off_is_said(self):
        import config
        r = {"training_health": {"available": True, "recall_at_20": 0.03, "action_rate": 0.006,
                                 "base_rate": 0.038, "lift": 5.27, "precision": 0.2,
                                 "evaluation_scope": "out_of_sample_time_holdout"},
             "training_to_live_gap": {"available": False, "reason": "bandit_off_live"}}
        with mock.patch.object(config, "BANDIT_ENABLED", False):
            txt = H._render_learning_block(r)
        self.assertIn("бандит выключен в живом пути", txt)
        self.assertIn("не применимо", txt)
        self.assertNotIn("согласованы", txt)


class TestMetricsV2(unittest.TestCase):
    def test_scripts_publish_immutable_versions_with_legacy_beside(self):
        for f, name in (("_backtest_signal_precision.py", "D1_D2_precision_msgrate_v2"),
                        ("_backtest_time_to_signal.py", "E1_time_to_signal_v2"),
                        ("_backtest_top20_coverage_funnel.py", "C1_C2_coverage_funnel_v2")):
            src = (HERE / f).read_text(encoding="utf-8")
            self.assertIn('"%s"' % name, src, f)
            self.assertIn('"label_provenance": IMMUTABLE', src, f)
            self.assertIn('"legacy_label_provenance"', src, f)
            self.assertIn("rank_before_filter=True", src, f)

    def test_scorecard_reads_v2_status_and_provenance(self):
        md = {"metrics": {"D1_D2_precision_msgrate": {"metric": "D1_D2_precision_msgrate_v2",
                                                      "label_provenance": IMM, "precision_pct": 13.6},
                          "E1_time_to_signal": {"metric": "E1_time_to_signal_v2", "label_provenance": IMM,
                                                "median_h": 3.5, "definition": "x"}}}
        sc = H.build_canonical_scorecard(md)
        self.assertEqual(sc["signal_precision"]["status"], "measured")
        self.assertEqual(sc["time_to_signal"]["status"], "measured")
        self.assertEqual(sc["signal_precision"]["source"], "D1_D2_precision_msgrate_v2")

    def test_documented(self):
        spec = ROOT / "docs/specs/features/report-validity-1001-spec.md"
        self.assertTrue(spec.exists())
        for doc in ("CLAUDE.md", "PROJECT_CONTEXT.md"):
            self.assertIn("report-validity-1001-spec.md", (ROOT / doc).read_text(encoding="utf-8"), doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
