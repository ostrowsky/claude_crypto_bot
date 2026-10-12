"""GOAL-label ML model in shadow (ml-goal-shadow-1012-spec.md), written from the spec first."""
import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import numpy as np

import botlog
import config
import ml_goal_shadow as S
import ml_signal_model as M
import readouts as R

ROOT = Path(__file__).resolve().parent.parent
D0 = datetime(2026, 10, 11, tzinfo=timezone.utc)
H = timedelta(hours=1)


class TestLabel(unittest.TestCase):
    def setUp(self):
        self.win = {("2026-10-11", "ICPUSDT")}
        self.dl = {("2026-10-11", "ICPUSDT"): (D0, D0 + 15 * H), ("2026-10-11", "BTCUSDT"): (D0, D0 + 3 * H)}

    def test_early_winner_window(self):
        self.assertTrue(S.early_winner("ICPUSDT", D0 + 13 * H, self.win, self.dl))
        self.assertFalse(S.early_winner("ICPUSDT", D0 + 15 * H, self.win, self.dl))   # at the crossing: late
        self.assertFalse(S.early_winner("BTCUSDT", D0 + 1 * H, self.win, self.dl))    # crossed, not a winner

    def test_resolved_needs_both_tables(self):
        eod = {("2026-10-11", "ICPUSDT"): 16.0, ("2026-10-12", "ICPUSDT"): 1.0}
        self.assertEqual(S.resolved_days(eod, self.dl), {"2026-10-11"})

    def test_eligible_needs_resolved_day_and_one_day_of_age(self):
        now = D0 + 40 * H
        self.assertTrue(S.eligible(D0 + 13 * H, {"2026-10-11"}, now))
        self.assertFalse(S.eligible(D0 + 13 * H, set(), now))
        self.assertFalse(S.eligible(D0 + 13 * H, {"2026-10-11"}, D0 + 30 * H))


class TestPayload(unittest.TestCase):
    def test_live_format_and_atomic_save(self):
        names = M.safe_feature_names()
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, len(names)))
        y = (X[:, 0] > 1).astype(float)
        p = S.fit_payload(X, y, names, meta={"n_rows": 300})
        self.assertEqual(p["label_version"], S.LABEL_VERSION)
        self.assertEqual(p["model"]["type"], "catboost")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.json"
            S.save_payload(p, path)
            self.assertFalse((Path(d) / "m.json.part").exists())
            back = json.loads(path.read_text(encoding="utf-8"))
        rec = {"f": {}, "seq": [], "signal_type": "trend", "tf": "15m", "is_bull_day": True,
               "hour_utc": 13, "dow": 6}
        v = M.predict_proba_from_payload(back, rec)
        self.assertTrue(0.0 <= v <= 1.0)


class TestScore(unittest.TestCase):
    def _args(self):
        return dict(sym="X", tf="15m", signal_type="trend", is_bull_day=True, feat={}, data=None, i=0,
                    btc_vs_ema50=0.0)

    def test_flag_off_returns_none(self):
        with mock.patch.object(config, "ML_GOAL_SHADOW_ENABLED", False):
            self.assertIsNone(S.score(**self._args()))

    def test_missing_file_fails_open(self):
        with mock.patch.object(config, "ML_GOAL_SHADOW_ENABLED", True):
            self.assertIsNone(S.score(path=Path("no/such/model.json"), **self._args()))

    def test_reload_on_mtime_change(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.json"
            path.write_text(json.dumps({"feature_names": ["a"], "model": {"type": "x"}, "trained_at": "t1"}))
            S._reset()
            with mock.patch.object(config, "ML_GOAL_SHADOW_RELOAD_CHECK_SEC", 0):
                self.assertEqual(S._load(path)["trained_at"], "t1")
                path.write_text(json.dumps({"feature_names": ["a"], "model": {"type": "x"}, "trained_at": "t2"}))
                os.utime(path, (time.time() + 5, time.time() + 5))
                self.assertEqual(S._load(path)["trained_at"], "t2")
            S._reset()


class TestBotlog(unittest.TestCase):
    def test_event_fields(self):
        got = []
        with mock.patch.object(botlog, "_write", side_effect=got.append):
            botlog.log_ml_goal_shadow("ICPUSDT", "15m", 1760187600000, "trend", 3.129, 0.4, 0.023, 0.15, 1.01,
                                      False, "2026-10-12T01:00:00Z")
        e = got[0]
        self.assertEqual(e["event"], "ml_goal_shadow")
        for k in ("sym", "tf", "bar_ts", "mode", "price", "goal_score", "ml_proba", "ml_min", "ml_max",
                  "live_pass", "model_trained_at"):
            self.assertIn(k, e)


def srow(sym, dec, live, score):
    return {"sym": sym, "tf": "15m", "dec": dec, "live_pass": live, "goal_score": score}


class TestReadout(unittest.TestCase):
    def test_matched_weekly_admission(self):
        rows = [srow("A", D0 + k * H, live=(k < 2), score=s) for k, s in enumerate([0.1, 0.2, 0.9, 0.8])]
        rows.append(srow("B", D0 + 5 * H, live=False, score=None))
        # live admits 2 of 5 this week -> shadow admits the top 2 by score, never a null score
        self.assertEqual(R.goal_shadow_admit(rows), {2, 3})

    def test_goal_contributions(self):
        W = [("2026-10-11", "A", D0, D0 + 10 * H),   # early live entry from a row the shadow blocks -> lost
             ("2026-10-11", "B", D0, D0 + 10 * H),   # no entry, 2 shadow-only rows -> gained 1-(1-p)^2
             ("2026-10-11", "C", D0, D0 + 10 * H)]   # early entry with no shadow row -> kept
        rows = [srow("A", D0 + 2 * H, True, 0.1), srow("B", D0 + 3 * H, False, 0.9),
                srow("B", D0 + 4 * H, False, 0.8)]
        entries = [("A", D0 + 2 * H, 0), ("C", D0 + 1 * H, None)]
        d, now = R.goal_shadow_days(W, entries, rows, shadow_pass={1, 2}, in_position=lambda s, t: False, p=0.5)
        self.assertEqual(now, 2)
        self.assertEqual(d, [-1.0, 0.75, 0.0])

    def test_verdict(self):
        self.assertEqual(R.goal_shadow_verdict(1.0, 0.1, -0.05, 0.3), "KEEP")
        self.assertEqual(R.goal_shadow_verdict(0.0, -1.0, 0.0, 0.3), "ROLLBACK_SUGGESTED")
        self.assertEqual(R.goal_shadow_verdict(2.0, 0.5, -0.5, -0.2), "ROLLBACK_SUGGESTED")
        self.assertEqual(R.goal_shadow_verdict(1.0, -0.1, -0.05, 0.3), "INCONCLUSIVE")

    def test_anomaly(self):
        rows = [srow("A", D0, True, 0.5)] * 19 + [srow("A", D0, True, None)] * 2
        self.assertTrue(R.goal_shadow_anomaly(rows, ["2026-10-11"], model_age_days=0.5))       # 9.5% null
        ok = [srow("A", D0, True, 0.5)] * 20
        self.assertFalse(R.goal_shadow_anomaly(ok, ["2026-10-11"], model_age_days=0.5))
        self.assertTrue(R.goal_shadow_anomaly(ok, ["2026-10-11", "2026-10-12", "2026-10-13"], 0.5))
        self.assertTrue(R.goal_shadow_anomaly(ok, ["2026-10-11"], model_age_days=4.0))


class TestWiring(unittest.TestCase):
    def test_config_defaults(self):
        self.assertIs(config.ML_GOAL_SHADOW_ENABLED, True)
        self.assertEqual(config.ML_GOAL_SHADOW_MODEL_FILE, "ml_goal_shadow_model.json")

    def test_registry_entry(self):
        reg = {r["id"]: r for r in R.load_registry()}
        r = reg["ML-GOAL-SHADOW"]
        self.assertEqual((r["fn"], r["window_from"], r["due"], r["min_days"], r["min_trades"]),
                         ("readout_ml_goal_shadow", "2026-10-13", "2026-11-23", 28, 20))
        self.assertIn("readout_ml_goal_shadow", R.FNS)

    def test_monitor_and_nightly_call_it(self):
        mon = (ROOT / "files" / "monitor.py").read_text(encoding="utf-8")
        self.assertIn("_ml_goal_shadow_log", mon)
        dl = (ROOT / "files" / "daily_learning.py").read_text(encoding="utf-8")
        self.assertIn("retrain_ml_goal_shadow", dl)

    def test_documented(self):
        for doc in ("CLAUDE.md", "PROJECT_CONTEXT.md"):
            self.assertIn("ml-goal-shadow-1012-spec.md", (ROOT / doc).read_text(encoding="utf-8"), doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
