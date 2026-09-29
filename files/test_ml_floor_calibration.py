"""Per-model ML floor in shadow (agent-tasks-0929-spec.md §7)."""
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ml_signal_model as MS
import readouts as RO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UTC = timezone.utc


class TestCalibrateFloor(unittest.TestCase):
    def test_admits_the_target_share_of_positives(self):
        y = [1] * 10 + [0] * 10
        s = [i / 10 for i in range(10)] + [0.05] * 10
        for target in (0.8, 0.9, 1.0):
            fl = MS.calibrate_floor(y, s, target)
            rec = sum(1 for v, t in zip(s, y) if t and v >= fl) / 10
            self.assertGreaterEqual(rec, target - 1e-9, target)
        self.assertEqual(MS.calibrate_floor([0, 0], [0.3, 0.4], 0.8), 0.0)

    def test_scale_invariance(self):
        # the same ranking on another scale gives the same operating point
        y = [1, 0, 1, 0, 1, 1, 0, 1]
        s = [0.9, 0.2, 0.7, 0.4, 0.6, 0.8, 0.1, 0.5]
        s2 = [v * 0.3 for v in s]
        adm1 = [v >= MS.calibrate_floor(y, s, 0.8) for v in s]
        adm2 = [v >= MS.calibrate_floor(y, s2, 0.8) for v in s2]
        self.assertEqual(adm1, adm2)

    def test_calibration_block(self):
        b = MS._calibration_block([1, 0, 1, 0], [0.9, 0.1, 0.6, 0.5])
        self.assertEqual(b["target_recall"], 0.80)
        self.assertEqual(set(b["grid"]), {"0.80", "0.85", "0.90", "0.95"})
        self.assertEqual(b["positives"], 2)

    def test_trainer_writes_it_and_refreshes_a_kept_incumbent(self):
        src = (HERE / "ml_signal_model.py").read_text(encoding="utf-8")
        self.assertIn('"calibration": calibration,', src)
        self.assertIn('inc["calibration"] = _calibration_block(holdout["y"], s_inc)', src)


class TestLiveGate(unittest.TestCase):
    def test_shadow_by_default(self):
        import config
        self.assertFalse(config.ML_FLOOR_CALIBRATION_ENABLED)
        self.assertEqual(config.ML_FLOOR_TARGET_RECALL, 0.80)
        self.assertLess(config.ML_FLOOR_CALIBRATED_MIN, config.ML_FLOOR_CALIBRATED_MAX)

    def test_calibrated_floor_is_clamped(self):
        import monitor as M
        old = M._ML_MODEL_CACHE
        try:
            M._ML_MODEL_CACHE = {"calibration": {"floor": 0.001}}
            self.assertAlmostEqual(M._ml_floor_calibrated(), 0.02)
            M._ML_MODEL_CACHE = {"calibration": {"floor": 0.95}}
            self.assertAlmostEqual(M._ml_floor_calibrated(), 0.60)
            M._ML_MODEL_CACHE = {"calibration": {"floor": 0.17}}
            self.assertAlmostEqual(M._ml_floor_calibrated(), 0.17)
            M._ML_MODEL_CACHE = {}
            self.assertIsNone(M._ml_floor_calibrated())
        finally:
            M._ML_MODEL_CACHE = old

    def test_gate_uses_it_only_behind_the_flag(self):
        src = (HERE / "monitor.py").read_text(encoding="utf-8")
        i = src.index('if getattr(config, "ML_FLOOR_CALIBRATION_ENABLED", False):')
        j = src.index('_max = float(getattr(config, "ML_GENERAL_HARD_BLOCK_MAX"', i)
        self.assertIn("_min = _cal", src[i:j])

    def test_load_is_logged(self):
        import botlog
        got = []
        old = botlog._write
        botlog._write = got.append
        try:
            botlog.log_ml_floor_calibration("catboost", "peak5_2.0", {"floor": 0.18, "target_recall": 0.8}, 0.15, False)
        finally:
            botlog._write = old
        self.assertEqual(got[0]["event"], "ml_floor_calibration")
        self.assertEqual(got[0]["calibrated_floor"], 0.18)
        self.assertFalse(got[0]["calibrated_floor_live"])


class TestShadowReadout(unittest.TestCase):
    def _run(self, rows, winners):
        t0 = datetime(2026, 10, 1, tzinfo=UTC)
        evs = [{"event": "ml_floor_calibration", "ts": t0.isoformat(), "_dt": t0, "calibrated_floor": 0.30, "live_floor": 0.15}]
        for i, (sym, p, y) in enumerate(rows):
            dt = t0 + timedelta(hours=1 + i)
            evs.append({"event": "entry", "ts": dt.isoformat(), "_dt": dt, "sym": sym, "tf": "15m",
                        "price": 1.0, "ml_proba": p, "_y": y})
        labels = {e["_dt"]: e.get("_y") for e in evs}
        old = (RO.scan_events, RO._peak_label)
        import goal_validator as GV
        old_w = GV.winner_days
        RO.scan_events = lambda since, kinds, path=None: evs
        RO._peak_label = lambda sym, tf, dt, price, h, thr: labels.get(dt)
        GV.winner_days = lambda since, full=None: [("2026-10-01", s, t0, t0 + timedelta(days=1)) for s in winners]
        try:
            return RO.readout_ml_floor_shadow({"window_from": "2026-10-01"}, {"days": ["2026-10-01"], "now": t0})
        finally:
            RO.scan_events, RO._peak_label = old
            GV.winner_days = old_w

    def test_fewer_winner_days_means_keep_the_fixed_floor(self):
        r = self._run([("AUSDT", 0.20, 1.0), ("BUSDT", 0.50, 0.0)], ["AUSDT"])
        self.assertEqual(r["reached_fixed"], 1)
        self.assertEqual(r["reached_calibrated"], 0)
        self.assertEqual(r["_verdict"], "ROLLBACK_SUGGESTED")

    def test_same_reach_better_precision_switches(self):
        r = self._run([("AUSDT", 0.50, 1.0), ("BUSDT", 0.20, 0.0)], ["AUSDT"])
        self.assertEqual(r["calibrated_floor"]["precision"], 1.0)
        self.assertEqual(r["_verdict"], "KEEP")

    def test_no_calibrated_model_yet(self):
        old = RO.scan_events
        RO.scan_events = lambda since, kinds, path=None: []
        try:
            r = RO.readout_ml_floor_shadow({"window_from": "2026-10-01"}, {"days": [], "now": None})
        finally:
            RO.scan_events = old
        self.assertEqual(r["_n_trades"], 0)

    def test_registered(self):
        r = [x for x in RO.load_registry() if x["id"] == "ML-FLOOR-SHADOW"][0]
        self.assertEqual(r["fn"], "readout_ml_floor_shadow")
        self.assertIn("readout_ml_floor_shadow", RO.FNS)
        self.assertGreaterEqual(r["min_days"], 14)

    def test_documented(self):
        for doc in ("CLAUDE.md", "PROJECT_CONTEXT.md", "docs/specs/features/agent-tasks-0929-spec.md"):
            self.assertIn("ML_FLOOR_CALIBRATION_ENABLED", (ROOT / doc).read_text(encoding="utf-8"), doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
