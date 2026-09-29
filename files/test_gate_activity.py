"""Over-block flags only for gates that still block (2026-09-29).

entry_score has blocked nothing since June, yet the scout (whole critic history)
kept a daily RF_overblock_entry_score flag from 2 475 March-May rows and the L2
agent built hypotheses on it.
"""
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pipeline_hypothesis as L2  # noqa: E402
import pipeline_lib as PL  # noqa: E402

NOW = datetime(2026, 9, 29, 12, tzinfo=timezone.utc)


class TestRecentlyActiveGates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "bot_events.jsonl"
        rows = [
            {"ts": "2026-09-28T10:00:00Z", "event": "blocked", "reason_code": "ml_zone", "signal_type": "ml_proba_zone"},
            {"ts": "2026-05-20T10:00:00Z", "event": "blocked", "reason_code": "entry_score", "signal_type": "entry_score"},
            {"ts": "2026-09-28T11:00:00Z", "event": "entry", "reason_code": "entry_score"},
        ]
        self.path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_recent_blocks_count(self):
        act = PL.recently_active_gates(events_path=self.path, now=NOW)
        self.assertEqual(act, {"ml_zone", "ml_proba_zone"})

    def test_unreadable_log_means_unknown(self):
        self.assertIsNone(PL.recently_active_gates(events_path=Path(self.tmp.name) / "missing.jsonl", now=NOW))


class TestFilters(unittest.TestCase):
    OBS = [{"gate": "entry_score", "miss_pct": 0.123}, {"gate": "ml_zone", "miss_pct": 0.2}]

    def test_health_keeps_only_active_gates(self):
        keep, drop = PL.active_over_blockers(self.OBS, {"ml_zone"})
        self.assertEqual([o["gate"] for o in keep], ["ml_zone"])
        self.assertEqual([o["gate"] for o in drop], ["entry_score"])

    def test_unknown_activity_keeps_old_behaviour(self):
        keep, drop = PL.active_over_blockers(self.OBS, None)
        self.assertEqual(len(keep), 2)
        self.assertEqual(drop, [])

    def test_l2_drops_stale_flags_from_old_reports(self):
        aggs = {"RF_overblock_entry_score": {}, "RF_overblock_ml_zone": {}, "RF_early_capture": {}}
        out, stale = L2.drop_inactive_overblock_flags(aggs, {"ml_zone"})
        self.assertEqual(set(out), {"RF_overblock_ml_zone", "RF_early_capture"})
        self.assertEqual(stale, ["entry_score"])
        out, stale = L2.drop_inactive_overblock_flags(aggs, None)
        self.assertEqual(len(out), 3)

    def test_wiring(self):
        hr = (HERE / "bot_health_report.py").read_text(encoding="utf-8")
        self.assertIn("PL.active_over_blockers(scout.get(\"over_blocking\") or [], PL.recently_active_gates())", hr)
        l2 = (HERE / "pipeline_hypothesis.py").read_text(encoding="utf-8")
        self.assertIn("drop_inactive_overblock_flags(aggs, PL.recently_active_gates())", l2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
