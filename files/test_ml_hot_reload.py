"""ML model hot reload (agent-tasks-0929-spec.md §8)."""
import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import botlog
import config
import monitor as M
import readouts as RO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UTC = timezone.utc


def _payload(name):
    return {"model_name": name, "label_version": "peak5_2.0", "feature_names": ["a"], "model": {"type": "logistic"},
            "calibration": {"floor": 0.2, "target_recall": 0.8}}


class TestHotReload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.file = Path(self.tmp.name) / "ml_signal_model.json"
        self.saved = (M._ML_MODEL_FILE, M._ML_MODEL_CACHE, M._ML_MODEL_MTIME, M._ML_MODEL_CHECKED_AT, botlog._write,
                      getattr(config, "ML_MODEL_HOT_RELOAD_ENABLED"), config.ML_MODEL_RELOAD_CHECK_SEC,
                      config.ML_MODEL_RELOAD_MIN_AGE_SEC)
        self.events = []
        botlog._write = self.events.append
        M._ML_MODEL_FILE = self.file
        M._ML_MODEL_CACHE, M._ML_MODEL_MTIME, M._ML_MODEL_CHECKED_AT = None, None, 0.0
        config.ML_MODEL_HOT_RELOAD_ENABLED = True
        config.ML_MODEL_RELOAD_CHECK_SEC = 0
        config.ML_MODEL_RELOAD_MIN_AGE_SEC = 0

    def tearDown(self):
        (M._ML_MODEL_FILE, M._ML_MODEL_CACHE, M._ML_MODEL_MTIME, M._ML_MODEL_CHECKED_AT, botlog._write,
         config.ML_MODEL_HOT_RELOAD_ENABLED, config.ML_MODEL_RELOAD_CHECK_SEC,
         config.ML_MODEL_RELOAD_MIN_AGE_SEC) = self.saved
        self.tmp.cleanup()

    def _write(self, obj, age=0.0):
        self.file.write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")
        t = time.time() - age
        os.utime(self.file, (t, t))
        M._ML_MODEL_CHECKED_AT = 0.0

    def test_first_load_then_swap(self):
        self._write(_payload("old"), age=100)
        self.assertEqual(M._load_ml_model_payload()["model_name"], "old")
        self._write(_payload("new"), age=50)
        self.assertEqual(M._load_ml_model_payload()["model_name"], "new")
        kinds = [e["event"] for e in self.events]
        self.assertIn("ml_model_reload", kinds)
        rel = [e for e in self.events if e["event"] == "ml_model_reload"][0]
        self.assertEqual((rel["old_model"], rel["new_model"]), ("old", "new"))
        self.assertEqual(kinds.count("ml_floor_calibration"), 2)       # at load and at the swap

    def test_unchanged_file_is_not_reread(self):
        self._write(_payload("old"), age=100)
        M._load_ml_model_payload()
        n = len(self.events)
        M._ML_MODEL_CHECKED_AT = 0.0
        M._load_ml_model_payload()
        self.assertEqual(len(self.events), n)

    def test_broken_or_incomplete_file_keeps_the_model(self):
        self._write(_payload("old"), age=100)
        M._load_ml_model_payload()
        self._write('{"model_name": "half', age=50)                     # half-written
        self.assertEqual(M._load_ml_model_payload()["model_name"], "old")
        self._write({"model_name": "no_model"}, age=40)                 # parses, but unusable
        self.assertEqual(M._load_ml_model_payload()["model_name"], "old")
        self._write(_payload("fixed"), age=30)                          # retried on the next check
        self.assertEqual(M._load_ml_model_payload()["model_name"], "fixed")

    def test_file_still_being_written_waits(self):
        self._write(_payload("old"), age=100)
        M._load_ml_model_payload()
        config.ML_MODEL_RELOAD_MIN_AGE_SEC = 60
        self._write(_payload("new"), age=5)
        self.assertEqual(M._load_ml_model_payload()["model_name"], "old")

    def test_flag_off_never_reloads(self):
        self._write(_payload("old"), age=100)
        M._load_ml_model_payload()
        config.ML_MODEL_HOT_RELOAD_ENABLED = False
        self._write(_payload("new"), age=50)
        self.assertEqual(M._load_ml_model_payload()["model_name"], "old")

    def test_catboost_cache_is_cleared_on_swap(self):
        import ml_signal_model as MS
        self._write(_payload("old"), age=100)
        M._load_ml_model_payload()
        MS._CATBOOST_CACHE["x"] = object()
        self._write(_payload("new"), age=50)
        M._load_ml_model_payload()
        self.assertNotIn("x", MS._CATBOOST_CACHE)

    def test_on_by_default(self):
        self.assertTrue(self.saved[5])


class TestReloadReadout(unittest.TestCase):
    def _run(self, blocked_share_day2, reloads=1):
        t0 = datetime(2026, 10, 1, tzinfo=UTC)
        evs = [{"event": "ml_model_reload", "ts": t0.isoformat(), "_dt": t0, "old_model": "a", "new_model": "b",
                "file_mtime": 0}] * reloads
        for day, share in ((0, 0.2), (1, blocked_share_day2)):
            for i in range(100):
                dt = t0 + timedelta(days=day, minutes=10 * i)
                blocked = i < share * 100
                e = {"event": "blocked" if blocked else "entry", "ts": dt.isoformat(), "_dt": dt,
                     "sym": "S%d" % i, "tf": "15m"}
                if blocked:
                    e["reason_code"] = "ml_zone"
                evs.append(e)
        old = RO.scan_events
        RO.scan_events = lambda since, kinds, path=None: evs
        try:
            return RO.readout_ml_reload({"window_from": "2026-10-01"}, {"days": ["2026-10-01", "2026-10-02"], "now": None})
        finally:
            RO.scan_events = old

    def test_blackout_is_an_anomaly(self):
        r = self._run(0.9)
        self.assertTrue(r["_anomaly"])
        self.assertIn("2026-10-02", r["anomalies"][0])

    def test_normal_days(self):
        r = self._run(0.3)
        self.assertFalse(r["_anomaly"])
        self.assertEqual(r["_verdict"], "KEEP")

    def test_threshold_catches_08_20_but_not_the_next_worst_day(self):
        self.assertLessEqual(RO.BLACKOUT_SHARE, 0.886)
        self.assertGreater(RO.BLACKOUT_SHARE, 0.791)

    def test_registered(self):
        r = [x for x in RO.load_registry() if x["id"] == "ML-HOT-RELOAD"][0]
        self.assertEqual(r["fn"], "readout_ml_reload")
        self.assertIn("readout_ml_reload", RO.FNS)

    def test_documented(self):
        for doc in ("CLAUDE.md", "PROJECT_CONTEXT.md", "docs/specs/features/agent-tasks-0929-spec.md"):
            self.assertIn("ML_MODEL_HOT_RELOAD_ENABLED", (ROOT / doc).read_text(encoding="utf-8"), doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
