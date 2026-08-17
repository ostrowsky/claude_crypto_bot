"""Focused contracts for the project Truth Harness."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import truth_harness as H


def _ids(audit: H.Audit) -> set[str]:
    return {f.check_id for f in audit.findings}


class ChangeEvidenceTests(unittest.TestCase):
    def test_source_without_spec_or_test_is_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audit = H.Audit("change")
            H.audit_change_set(audit, ["files/bot_health_report.py"], root)

        self.assertIn("TH12_CHANGE_EVIDENCE", _ids(audit))
        self.assertTrue(audit.blocking)

    def test_metric_change_with_spec_test_and_living_spec_passes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec = root / "docs/specs/features/truth-harness-spec.md"
            spec.parent.mkdir(parents=True)
            spec.write_text("Applies TH-01 and TH-10.\n", encoding="utf-8")
            auto = root / "docs/specs/features/auto-improvement-loop-spec.md"
            auto.write_text("updated\n", encoding="utf-8")
            audit = H.Audit("change")
            H.audit_change_set(audit, [
                "files/bot_health_report.py",
                "files/test_bot_health_report_integrity.py",
                "docs/specs/features/truth-harness-spec.md",
                "docs/specs/features/auto-improvement-loop-spec.md",
            ], root)

        self.assertFalse(audit.blocking, audit.findings)


class ProvenanceTests(unittest.TestCase):
    def test_same_snapshot_label_and_return_feature_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = root / "files"
            files.mkdir()
            (files / "daily_learning.py").write_text(
                'rank_gainers(tickers)\n"label_top20": int(sym in top20)\n',
                encoding="utf-8",
            )
            (files / "top_gainer_model.py").write_text(
                'FEATURE_NAMES = ["tg_return_since_open"]\n', encoding="utf-8")
            (files / "train_top_gainer.py").write_text("", encoding="utf-8")
            (files / "offline_rl.py").write_text("", encoding="utf-8")
            (files / "_compute_early_capture.py").write_text(
                'load_winners(path, label_field="label_top20", cut_dt=cut)\n',
                encoding="utf-8",
            )
            audit = H.Audit("full")
            H.audit_model_provenance(audit, root)

        self.assertIn("TH03_TOP_GAINER_TARGET", _ids(audit))
        self.assertIn("TH03_NORTH_STAR_TARGET", _ids(audit))

    def test_row_split_without_day_group_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = root / "files"
            files.mkdir()
            for name in ("daily_learning.py", "top_gainer_model.py", "offline_rl.py"):
                (files / name).write_text("", encoding="utf-8")
            (files / "train_top_gainer.py").write_text(
                "split_idx = int(len(X) * (1 - val_ratio))\n", encoding="utf-8")
            audit = H.Audit("full")
            H.audit_model_provenance(audit, root)

        self.assertIn("TH04_DAY_GROUP_SPLIT", _ids(audit))


class HealthTruthTests(unittest.TestCase):
    def _write_health(self, root: Path, report: dict, telegram: str = "") -> None:
        health = root / ".runtime/pipeline/health"
        health.mkdir(parents=True)
        path = health / "health-2026-08-13.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        (health / "health-2026-08-13.tg.txt").write_text(telegram, encoding="utf-8")

    @staticmethod
    def _base_report() -> dict:
        return {
            "metrics_daily_latest": {"metrics": {
                "NS_EarlyCapture_top20": {
                    "early_capture": 0.07, "n": 26,
                    "days_window": 14, "days_full": 10,
                },
                "EX1_realized_potential": {},
                "D1_D2_precision_msgrate": {},
                "E1_time_to_signal": {},
                "Q2_whipsaw_rate": {},
                "Q1_Q3_fast_reversal": {},
            }},
            "training_health": {"available": False},
            "training_to_live_gap": {"available": False},
            "canonical_scorecard": {
                "north_star": {"status": "verified"},
                "portfolio_alpha": {"value": 0.01},
                "realized_potential": {"value": 0.55},
            },
            "do_not_touch": {},
        }

    def test_in_sample_recall_without_base_and_lift_is_rejected(self):
        report = self._base_report()
        report["training_health"] = {
            "recall_at_20": 1.0,
            "evaluation_scope": "in_sample_post_fit",
        }
        report["training_to_live_gap"] = {"available": True, "value": 0.73}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_health(root, report, "модель: находит 100% ракет")
            audit = H.Audit("full")
            H.audit_health_report(audit, root, today=date(2026, 8, 13))

        ids = _ids(audit)
        self.assertIn("TH01_RATIO_CONTEXT", ids)
        self.assertIn("TH04_REPORT_SCOPE", ids)
        self.assertIn("TH02_INVALID_GAP", ids)
        self.assertIn("TH02_PROXY_AS_PROGRESS", ids)

    def test_expired_gate_evidence_is_rejected(self):
        report = self._base_report()
        report["do_not_touch"] = {
            "last_verified": "2026-05-28", "verify_every_days": 30,
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_health(root, report)
            audit = H.Audit("full")
            H.audit_health_report(audit, root, today=date(2026, 8, 13))

        self.assertIn("TH10_EVIDENCE_EXPIRY", _ids(audit))

    def test_honest_fixture_has_no_health_blockers(self):
        report = self._base_report()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_health(root, report)
            audit = H.Audit("full")
            H.audit_health_report(audit, root, today=date(2026, 8, 13))

        self.assertFalse(audit.blocking, audit.findings)


class TestChecksObserveTheDeployedArtifact(unittest.TestCase):
    """A check that cannot see its own repair is worse than no check: it keeps
    reporting a solved finding and teaches the reader to skip the red line."""

    def test_facts_come_from_the_model_blob(self):
        import truth_harness as TH
        facts = TH._trained_model_facts(TH.ROOT)
        self.assertNotIn("_unreadable", facts,
                         "the blob must be readable; a swallowed error used to "
                         "return {} and report 'no immutable label' about a "
                         "model that had one")
        for key in ("label_timing", "evaluation_scope"):
            self.assertIn(key, facts)

    def test_unreadable_blob_is_reported_not_silently_empty(self):
        import truth_harness as TH
        facts = TH._trained_model_facts(Path("no", "such", "root"))
        self.assertIn("_unreadable", facts)

    def test_th03_clears_when_the_deployed_model_carries_immutable_labels(self):
        import truth_harness as TH
        facts = TH._trained_model_facts(TH.ROOT)
        audit = TH.Audit("full")
        TH.audit_model_provenance(audit)
        ids = {f.check_id for f in audit.findings if f.severity == "error"}
        if facts.get("label_timing") == "immutable_later_eod_close":
            self.assertNotIn("TH03_TOP_GAINER_TARGET", ids)
        if str(facts.get("evaluation_scope", "")).startswith("day_grouped"):
            self.assertNotIn("TH04_DAY_GROUP_SPLIT", ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
