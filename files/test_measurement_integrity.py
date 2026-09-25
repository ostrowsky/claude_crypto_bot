"""Guards for four measurement defects found on 2026-09-25.

Each one made a number say something the data did not:

  * the ml_proba field of blocked events carried the RANKER's quality_proba
    on every block from trend_quality onward -- the "bimodal ml_proba";
  * EarlyCapture@move_lead scored only misses (entered winner-days without an
    intraday deadline were dropped, missed ones kept) and printed 0.000;
  * L7 counted a decision whose metrics no report carries as a MISS, so
    hit_rate read 0.0 for eleven weeks and 11 rollbacks were recommended weekly;
  * the L3 ML replay graded that ranker number against an ML floor.

Spec: docs/specs/features/measurement-integrity-0925-spec.md
"""
from __future__ import annotations

import re
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

MON_SRC = (HERE / "monitor.py").read_text(encoding="utf-8")


class TestBlockedEventsLogTheRealMlProba(unittest.TestCase):
    def test_single_definition(self):
        self.assertEqual(MON_SRC.count("def _build_block_context("), 1)

    def test_no_block_site_passes_the_ranker_as_ml_proba(self):
        calls = re.findall(r"_build_block_context\((.*?)\)", MON_SRC, re.S)
        self.assertGreaterEqual(len(calls), 11)
        for c in calls:
            self.assertNotIn("ranker_proba", c)

    def test_the_field_comes_from_the_explicit_argument(self):
        import numpy as np
        import monitor as MON
        feat = {"rsi": np.array([50.0, 60.0]), "close": np.array([1.0, 1.1]),
                "ema_fast": np.array([1.0, 1.0])}
        ctx = MON._build_block_context(feat=feat, i=1, ml_proba=0.21,
                                       ranker_info={"quality_proba": 0.397})
        self.assertEqual(ctx["ml_proba"], 0.21)
        self.assertEqual(ctx["ranker_quality_proba"], 0.397)
        ctx = MON._build_block_context(feat=feat, i=1, ranker_info={"quality_proba": 0.397})
        self.assertNotIn("ml_proba", ctx)


class TestMoveLeadDoesNotScoreOnlyMisses(unittest.TestCase):
    def test_days_without_timing_are_excluded_whatever_the_outcome(self):
        import _compute_early_capture as E
        t = datetime(2026, 9, 20, 10, tzinfo=timezone.utc)
        winners = {("2026-09-20", "AUSDT"), ("2026-09-20", "BUSDT")}
        first_entry = {("2026-09-20", "AUSDT"): (t, 1.0)}          # A entered, B missed
        r = E.compute_north_star(winners, {}, first_entry, {}, "x", lead_mode="move", deadlines={})
        self.assertEqual(r["n"], 0)
        self.assertEqual(r["winners_without_deadline"], 2)

    def test_the_metric_is_published_as_none_not_zero(self):
        src = (HERE / "_compute_early_capture.py").read_text(encoding="utf-8")
        self.assertIn('res_move["early_capture"] if res_move["n"] else None', src)


class TestL7DoesNotCountUnmeasuredAsMiss(unittest.TestCase):
    def setUp(self):
        import pipeline_monitor as PM
        self.PM = PM
        self._prev = PM._load_health_in_window
        # four reports on each side that carry only an unrelated metric
        PM._load_health_in_window = lambda a, b: [{"deployment_health": {}} for _ in range(4)]

    def tearDown(self):
        self.PM._load_health_in_window = self._prev

    def test_all_metrics_missing_is_unmeasurable(self):
        d = {"ts": "2026-06-01T00:00:00Z", "expected": {"no_such_metric": "0.01..0.02"}}
        self.assertEqual(self.PM.evaluate_decision(d)["verdict"], "unmeasurable")

    def test_unmeasurable_stays_out_of_hit_rate(self):
        decs = [{"decision_id": "a", "stage": "approved"}, {"decision_id": "b", "stage": "approved"}]
        meta = self.PM.compute_pipeline_metameetrics(
            decs, {"a": {"verdict": "unmeasurable"}, "b": {"verdict": "hit"}})
        self.assertEqual((meta["hits"], meta["misses"], meta["unmeasurable"]), (1, 0, 1))
        self.assertEqual(meta["hit_rate"], 1.0)


class TestTheMlReplaySkipsRankerRows(unittest.TestCase):
    def test_a_row_whose_ml_proba_is_the_ranker_is_unreplayable(self):
        import pipeline_replay_validator as V
        cfg = {"ML_GENERAL_HARD_BLOCK_MIN": 0.15, "ML_GENERAL_HARD_BLOCK_BULL_DAY_MIN": 0.15}
        self.assertIsNone(V._ml_blocks({"ml_proba": 0.397, "ranker_quality_proba": 0.397}, cfg))
        self.assertTrue(V._ml_blocks({"ml_proba": 0.10, "ranker_quality_proba": 0.397}, cfg))


if __name__ == "__main__":
    unittest.main(verbosity=2)
