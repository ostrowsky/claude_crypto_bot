"""model_health.py -- input drift by PSI, quality vs the model's own target, run manifests
(model-health-1005-spec.md)."""
import json
import random
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import model_health as MH


def _rows(days, sign=1.0, seed=0, shift=0.0):
    """ml_proba ranks the peak label in the direction of `sign`; inputs shifted by `shift`."""
    rnd = random.Random(seed)
    out = []
    for d in days:
        for _ in range(80):
            p = rnd.random()
            peak = 2.0 + 3.0 * sign * (p - 0.5) + rnd.gauss(0, 1.0)
            r = {"day": d, "ts": d + "T00:00:00Z", "sym": "AAAUSDT", "tf": "15m", "ml_proba": p,
                 "peak": peak, "ret_10": rnd.gauss(0, 1)}
            for k in MH.INPUTS:
                r[k] = rnd.gauss(shift, 1)
            out.append(r)
    return out


def _days(start, n):
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


class Stats(unittest.TestCase):
    def test_psi_stable_and_shifted(self):
        rnd = random.Random(1)
        ref = [rnd.gauss(0, 1) for _ in range(5000)]
        self.assertLess(MH.psi(ref, [rnd.gauss(0, 1) for _ in range(2000)]), MH.PSI_WATCH)
        self.assertGreaterEqual(MH.psi(ref, [rnd.gauss(1.0, 1) for _ in range(2000)]), MH.PSI_SHIFT)
        self.assertIsNone(MH.psi(ref[:10], ref))              # too few rows -> no verdict

    def test_spearman_and_auc(self):
        x = list(range(100))
        self.assertAlmostEqual(MH.spearman(x, [2 * v + 1 for v in x]), 1.0)
        self.assertAlmostEqual(MH.spearman(x, [-v for v in x]), -1.0)
        labels = [v >= 50 for v in x]
        self.assertAlmostEqual(MH.auc(x, labels), 1.0)
        self.assertAlmostEqual(MH.auc([-v for v in x], labels), 0.0)
        rnd = random.Random(3)
        noise = [rnd.random() for _ in range(4000)]
        self.assertLess(abs(MH.auc(noise, [rnd.random() < .3 for _ in noise]) - 0.5), 0.05)


class Report(unittest.TestCase):
    def test_quality_against_own_target(self):
        days = _days(date(2026, 9, 1), 30)
        good = MH.finalize(MH.report_for(_rows(days, +1.0), date(2026, 10, 1), set()), [])
        bad = MH.finalize(MH.report_for(_rows(days, -1.0), date(2026, 10, 1), set()), [])
        self.assertEqual(good["quality"]["level"], "ok")
        self.assertNotIn("quality:inverted", good["alerts"])
        self.assertEqual(bad["quality"]["level"], "inverted")
        self.assertIn("quality:inverted", bad["alerts"])

    def test_uses_only_days_before_today(self):
        rows = _rows(_days(date(2026, 9, 1), 30)) + _rows(["2026-10-01"], -1.0, seed=9)
        rep = MH.report_for(rows, date(2026, 10, 1), set())
        self.assertNotIn("2026-10-01", rep["quality"]["daily"])

    def test_input_shift_alerts_only_when_it_persists(self):
        ref = _rows(_days(date(2026, 8, 1), 31))
        moved = _rows(_days(date(2026, 9, 1), 10), seed=4, shift=1.5)
        hist = []
        for i, today in enumerate((date(2026, 9, 9), date(2026, 9, 10), date(2026, 9, 11))):
            rep = MH.finalize(MH.report_for(ref + moved, today, set()), hist)
            self.assertIn("atr_pct", rep["shifted"])
            if i < MH.PERSIST_DAYS - 1:
                self.assertNotIn("input_drift:atr_pct", rep["alerts"])
            else:
                self.assertIn("input_drift:atr_pct", rep["alerts"])
            hist.append(rep)

    def test_regime_indicator_never_alerts(self):
        hist = [{"shifted": ["btc_vs_ema50"]}] * 5
        rep = {"shifted": ["btc_vs_ema50"], "quality": {"level": "ok"}}
        self.assertEqual(MH.finalize(rep, hist)["alerts"], [])

    def test_render_block(self):
        rep = {"day": "2026-10-05", "input_drift": {"atr_pct": 0.31}, "shifted": ["atr_pct"], "watched": [],
               "quality": {"level": "ok", "days": 14, "auc_mean": 0.61, "auc_lo95": 0.58, "auc_hi95": 0.64,
                           "target_base_rate": 0.14, "ic_ret10": {"mean": -0.1, "lo95": -0.2, "hi95": -0.05}},
               "alerts": ["input_drift:atr_pct"], "watch": []}
        hist = [{"shifted": ["atr_pct"]}, {"shifted": ["atr_pct"]}]
        txt = MH.render_block(rep, hist, [{"model": "ml_signal_model", "changed": True}])
        self.assertIn("AUC 0.610", txt)
        self.assertIn("atr_pct 0.31 (3 дн.)", txt)
        self.assertIn("это не цель модели", txt)
        self.assertIn("обновлены ml_signal_model", txt)


class Manifest(unittest.TestCase):
    def test_record_retrain_writes_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / "m.json"
            model.write_text("{}", encoding="utf-8")
            before = MH.sha256_file(model)
            model.write_text('{"v": 2}', encoding="utf-8")
            out = Path(d) / "manifests.jsonl"
            rec = MH.record_retrain("ml_signal_model", model, before, dataset_path=model,
                                    result={"status": "ok"}, path=out)
            self.assertTrue(rec["changed"])
            self.assertEqual(rec["sha_before"], before)
            line = json.loads(out.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(line["model"], "ml_signal_model")
            self.assertGreater(line["dataset"]["bytes"], 0)

    def test_record_retrain_never_raises(self):
        with tempfile.TemporaryDirectory() as d:
            rec = MH.record_retrain("x", Path(d) / "missing.json", None, path=Path(d) / "m.jsonl")
            self.assertFalse(rec.get("changed", False))


if __name__ == "__main__":
    unittest.main()
