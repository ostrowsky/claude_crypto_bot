"""Contracts for the artifact freshness SLO.

The roadmap gate for E3 is literally "verify the alarm fires on a deliberately
stale copy", so that is what these tests build: real files with backdated
mtimes, not mocked clocks.

The second half matters as much as the first. A checker that reports a
deliberately disabled artifact as stale gets ignored within a week, and then the
next real stall goes unnoticed again — which is the failure this module exists
to prevent.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import artifact_freshness as AF  # noqa: E402


class TestArtifactFreshness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _make(self, rel: str, age_h: float) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
        stamp = time.time() - age_h * 3600
        os.utime(p, (stamp, stamp))
        return p

    def _only(self, art: AF.Artifact) -> list[dict]:
        with patch.object(AF, "ARTIFACTS", (art,)):
            return AF.check(root=self.root)

    def test_fresh_artifact_is_ok(self):
        art = AF.Artifact("x", "files/x.jsonl", 6, "why")
        self._make("files/x.jsonl", age_h=1)
        row = self._only(art)[0]
        self.assertEqual(row["status"], "ok")
        self.assertLess(row["age_h"], 6)

    def test_alarm_fires_on_a_deliberately_stale_copy(self):
        art = AF.Artifact("x", "files/x.jsonl", 6, "why")
        self._make("files/x.jsonl", age_h=58 * 24)   # the backfill outage, to scale
        row = self._only(art)[0]
        self.assertEqual(row["status"], "stale")
        self.assertGreater(row["age_h"], 6)

    def test_boundary_is_not_stale_until_past_the_limit(self):
        art = AF.Artifact("x", "files/x.jsonl", 6, "why")
        self._make("files/x.jsonl", age_h=5.9)
        self.assertEqual(self._only(art)[0]["status"], "ok")

    def test_missing_artifact_is_reported_not_skipped(self):
        art = AF.Artifact("x", "files/never_written.jsonl", 6, "why")
        row = self._only(art)[0]
        self.assertEqual(row["status"], "missing")
        self.assertIsNone(row["age_h"])

    def test_disabled_flag_reports_disabled_not_stale(self):
        # fast_reversal_catboost.cbm is 46 days old on purpose; calling that
        # stale trains people to ignore the report.
        art = AF.Artifact("fr", "files/fr.cbm", 36, "why", flag="SOME_FLAG")
        self._make("files/fr.cbm", age_h=46 * 24)
        with patch.object(AF, "_flag_enabled", return_value=False):
            row = self._only(art)[0]
        self.assertEqual(row["status"], "disabled")

    def test_enabled_flag_still_reports_stale(self):
        art = AF.Artifact("fr", "files/fr.cbm", 36, "why", flag="SOME_FLAG")
        self._make("files/fr.cbm", age_h=46 * 24)
        with patch.object(AF, "_flag_enabled", return_value=True):
            self.assertEqual(self._only(art)[0]["status"], "stale")

    def test_unreadable_config_assumes_enabled(self):
        # Failing open here means a real stall is never hidden by a config error.
        with patch.dict(sys.modules, {"config": None}):
            self.assertTrue(AF._flag_enabled("ANY_FLAG"))

    def test_directory_artifact_uses_its_newest_file(self):
        art = AF.Artifact("d", ".runtime/health", 36, "why", is_dir=True)
        self._make(".runtime/health/old.json", age_h=90 * 24)
        self._make(".runtime/health/new.json", age_h=2)
        row = self._only(art)[0]
        self.assertEqual(row["status"], "ok")
        self.assertLess(row["age_h"], 3)

    def test_empty_directory_is_missing(self):
        art = AF.Artifact("d", ".runtime/health", 36, "why", is_dir=True)
        (self.root / ".runtime" / "health").mkdir(parents=True)
        self.assertEqual(self._only(art)[0]["status"], "missing")

    def test_thresholds_are_declared_for_every_artifact(self):
        for art in AF.ARTIFACTS:
            with self.subTest(artifact=art.name):
                self.assertGreater(art.max_age_h, 0)
                self.assertTrue(art.why.strip(), "a threshold without a reason rots")

    def test_render_names_the_stall_rather_than_only_counting_it(self):
        rows = [{"name": "critic_dataset", "path": "p", "max_age_h": 6,
                 "why": "written as decisions happen", "age_h": 1389.0,
                 "status": "stale", "detail": "d"}]
        text = AF.render(rows)
        self.assertIn("ПРОСРОЧЕНО", text)
        self.assertIn("critic_dataset", text)
        self.assertIn("written as decisions happen", text)


if __name__ == "__main__":
    unittest.main()
