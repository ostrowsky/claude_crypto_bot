"""Matched-admission ML promotion (ml-admission-match-1009-spec.md)."""
import json
import random
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ml_promotion_gate as MPG

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _rows(n=600, seed=3, bull_every=0):
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        x = rnd.random()
        rows.append({"x": x, "is_bull_day": bool(bull_every and i % bull_every == 0),
                     "_dt": T0 + timedelta(minutes=15 * i), "sym": "S%d" % (i % 7)})
    # noisy label so precision is not trivially 1
    y = [1.0 if r["x"] + rnd.gauss(0, 0.15) > 0.7 else 0.0 for r in rows]
    return rows, y


def _cfg(**kw):
    base = dict(ML_GENERAL_HARD_BLOCK_MIN=0.5, ML_GENERAL_HARD_BLOCK_BULL_DAY_MIN=0.5)
    base.update(kw)
    return types.SimpleNamespace(**base)


PRED = lambda payload, r: payload["f"](r["x"])  # noqa: E731


def _evaluate(inc, cand, rows, y, cfg, prev=None):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "inc.json"
        p.write_text(json.dumps({k: v for k, v in inc.items() if k != "f"}), encoding="utf-8")
        real = MPG.json.loads

        def loads(s):
            out = real(s)
            out["f"] = inc["f"]
            return out
        MPG.json.loads = loads
        try:
            return MPG.evaluate(p, cand, {"rows": rows, "y": y}, cfg=cfg, predict=PRED, prev_candidate=prev)
        finally:
            MPG.json.loads = real


INC = {"label_version": "peak", "model_name": "catboost", "f": lambda x: x}
# same ranking, scores shifted up: the 10-02..10-08 situation
SHIFTED = {"label_version": "peak", "model_name": "logistic", "f": lambda x: 0.5 * x + 0.4}


class TestMatchedFloors(unittest.TestCase):
    def test_floor_for_share(self):
        self.assertAlmostEqual(MPG.floor_for_share([0.1, 0.2, 0.3, 0.4], 0.5), 0.3)
        self.assertIsNone(MPG.floor_for_share([], 0.5))

    def test_per_day_type(self):
        rows = [{"is_bull_day": b} for b in (False, False, True, True)]
        mf = MPG.matched_floors(rows, [0.6, 0.4, 0.9, 0.8], [0.3, 0.2, 0.5, 0.45], [0.5] * 4)
        self.assertAlmostEqual(mf["default"], 0.3)      # incumbent admits 1 of 2 normal rows
        self.assertAlmostEqual(mf["bull_day"], 0.45)    # and 2 of 2 bull rows
        only = MPG.matched_floors(rows[:2], [0.6, 0.4], [0.3, 0.2], [0.5, 0.5])
        self.assertEqual(only["default"], only["bull_day"])

    def test_effective_floors(self):
        rows = [{"is_bull_day": True}, {"is_bull_day": False}]
        cfg = _cfg()
        self.assertEqual(MPG.effective_floors({}, rows, cfg), [0.5, 0.5])
        self.assertEqual(MPG.effective_floors({"admission_floor": {"default": 0.2, "bull_day": 0.1}}, rows, cfg),
                         [0.1, 0.2])
        self.assertEqual(MPG.effective_floors({"admission_floor": {"default": 0.2}}, rows, cfg), [0.5, 0.5])


class TestGateDecision(unittest.TestCase):
    def test_shadow_keeps_fixed_decision_and_logs_matched(self):
        rows, y = _rows()
        r = _evaluate(INC, SHIFTED, rows, y, _cfg())
        self.assertEqual(r["decision"], "keep_incumbent")          # fixed floor: admits more, precision falls
        self.assertFalse(r["matched_live"])
        self.assertEqual(r["matched"]["decision"], "promote")      # equal model at equal share
        cm = r["matched"]["comparison"]
        self.assertAlmostEqual(cm["admitted_new"], cm["admitted_old"], delta=0.01)
        self.assertAlmostEqual(r["admission_floor"]["default"], 0.65, delta=0.02)
        self.assertEqual(r["admission_floor"]["holdout_to"], rows[-1]["_dt"].isoformat())

    def test_flag_on_uses_matched_decision(self):
        rows, y = _rows()
        r = _evaluate(INC, SHIFTED, rows, y, _cfg(ML_PROMOTION_MATCHED_ADMISSION_ENABLED=True))
        self.assertTrue(r["matched_live"])
        self.assertEqual(r["decision"], "promote")
        self.assertEqual(r["fixed_floor_decision"]["decision"], "keep_incumbent")

    def test_worse_model_still_kept_when_matched(self):
        rows, y = _rows()
        bad = {"label_version": "peak", "f": lambda x: 1 - x}
        r = _evaluate(INC, bad, rows, y, _cfg(ML_PROMOTION_MATCHED_ADMISSION_ENABLED=True))
        self.assertEqual(r["decision"], "keep_incumbent")

    def test_incumbent_judged_at_its_own_floor_only_when_live(self):
        rows, y = _rows()
        inc = dict(INC, admission_floor={"default": 0.9, "bull_day": 0.9})
        off = _evaluate(inc, SHIFTED, rows, y, _cfg())
        on = _evaluate(inc, SHIFTED, rows, y, _cfg(ML_PROMOTION_MATCHED_ADMISSION_ENABLED=True))
        self.assertGreater(off["comparison"]["admitted_old"], 0.4)  # config floor 0.5
        self.assertLess(on["comparison"]["admitted_old"], 0.15)     # its own 0.9

    def test_transfer_rows_are_only_new_rows(self):
        rows, y = _rows()
        since = rows[499]["_dt"].isoformat()
        prev = dict(SHIFTED, admission_floor={"default": 0.65, "bull_day": 0.65, "holdout_to": since})
        r = _evaluate(INC, SHIFTED, rows, y, _cfg(), prev=prev)
        tr = r["transfer"]
        self.assertEqual(len(tr["rows"]), 100)
        self.assertTrue(all(row[0] > since[:19] for row in tr["rows"]))
        ts, sym, yy, a_old, a_new = tr["rows"][0]
        self.assertEqual(a_old, int(rows[500]["x"] >= 0.5))
        self.assertEqual(a_new, int(0.5 * rows[500]["x"] + 0.4 >= 0.65))
        # no stored floor -> no transfer block
        self.assertNotIn("transfer", _evaluate(INC, SHIFTED, rows, y, _cfg(), prev=dict(SHIFTED)))

    def test_shadow_error_does_not_break_decision(self):
        rows, y = _rows()
        orig = MPG.matched_floors
        MPG.matched_floors = lambda *a: 1 / 0
        try:
            r = _evaluate(INC, SHIFTED, rows, y, _cfg())
        finally:
            MPG.matched_floors = orig
        self.assertEqual(r["decision"], "keep_incumbent")
        self.assertIn("matched_error", r)


class TestReadout(unittest.TestCase):
    def _run(self, transfer_rows, winners=frozenset()):
        import readouts as R
        import exit_validator as EV
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "log.jsonl"
            with open(p, "w", encoding="utf-8") as fh:
                for i, chunk in enumerate(transfer_rows):
                    fh.write(json.dumps({"ts": "2026-10-%02dT01:00:00+00:00" % (11 + i), "decision": "keep_incumbent",
                                         "matched": {"decision": "promote"}, "admission_floor": {"default": 0.18},
                                         "transfer": {"rows": chunk}}) + "\n")
            old_log, old_lab = R.PROMOTION_LOG, EV.day_labels
            R.PROMOTION_LOG, EV.day_labels = p, (lambda: (set(), set(winners)))
            try:
                return R.readout_ml_admission_match({"window_from": "2026-10-10"}, {"days": []})
            finally:
                R.PROMOTION_LOG, EV.day_labels = old_log, old_lab

    def _chunk(self, night, n, new_admits, seed):
        rnd = random.Random(seed)
        out = []
        for i in range(n):
            y = int(rnd.random() < 0.25)
            a = int(rnd.random() < 0.66)
            out.append(["2026-10-%02dT%02d:%02d:00" % (10 + night, i // 4, (i % 4) * 15), "S%d" % i, y, a,
                        a if new_admits is None else new_admits(a, rnd)])
        return out

    def test_identical_admission_keeps(self):
        res = self._run([self._chunk(k, 60, None, k) for k in range(10)])
        self.assertEqual(res["days"], 10)
        self.assertEqual(res["rows"], 600)
        self.assertEqual(res["_verdict"], "KEEP")

    def test_drifting_share_rolls_back(self):
        res = self._run([self._chunk(k, 60, lambda a, rnd: 1, k) for k in range(10)])   # admits everything
        self.assertEqual(res["_verdict"], "ROLLBACK_SUGGESTED")

    def test_share_outside_band_is_anomaly(self):
        import readouts as R
        import exit_validator as EV
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "log.jsonl"
            p.write_text(json.dumps({"ts": "2026-10-11T01:00:00+00:00", "decision": "promote",
                                     "admission_floor": {"default": 0.3, "incumbent_share": {"default": 0.31}},
                                     "transfer": {"rows": self._chunk(0, 20, None, 2)}}) + "\n", encoding="utf-8")
            old_log, old_lab = R.PROMOTION_LOG, EV.day_labels
            R.PROMOTION_LOG, EV.day_labels = p, (lambda: (set(), set()))
            try:
                res = R.readout_ml_admission_match({"window_from": "2026-10-10"}, {"days": []})
            finally:
                R.PROMOTION_LOG, EV.day_labels = old_log, old_lab
        self.assertTrue(res["_anomaly"])
        self.assertEqual(len(res["share_off_band"]), 1)

    def test_duplicates_counted_once_and_errors_flagged(self):
        c = self._chunk(0, 30, None, 1)
        res = self._run([c, c])
        self.assertEqual(res["rows"], 30)
        self.assertEqual(res["days"], 1)


class TestMorningLine(unittest.TestCase):
    def test_shadow_verdict_in_morning_line(self):
        import pipeline_notify as N
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "log.jsonl"
            MPG.log_decision({"decision": "keep_incumbent", "reason": "r", "matched": {"decision": "promote"},
                              "matched_live": False}, p)
            self.assertIn("на равной доле пропуска была бы принята", N.build_ml_gate_line(p))
            MPG.log_decision({"decision": "keep_incumbent", "reason": "r"}, p)
            self.assertNotIn("равной доле", N.build_ml_gate_line(p))


class TestWiring(unittest.TestCase):
    def test_flag_off_by_default(self):
        import config
        self.assertFalse(config.ML_PROMOTION_MATCHED_ADMISSION_ENABLED)

    def test_trainer_and_monitor_wired(self):
        src = (HERE / "ml_signal_model.py").read_text(encoding="utf-8")
        self.assertIn("prev_candidate=prev_cand", src)
        self.assertIn('payload["admission_floor"] = gate["admission_floor"]', src)
        mon = (HERE / "monitor.py").read_text(encoding="utf-8")
        self.assertIn("_ml_floor_admission(bool(is_bull_day_now))", mon)
        self.assertLess(mon.index("ML_PROMOTION_MATCHED_ADMISSION_ENABLED"), mon.index("_cal = _ml_floor_calibrated()"))

    def test_registered_and_documented(self):
        reg = json.loads((HERE / "readout_registry.json").read_text(encoding="utf-8"))
        e = [x for x in reg["readouts"] if x["id"] == "ML-ADMISSION-MATCH"]
        self.assertEqual(len(e), 1)
        import readouts as R
        self.assertIn(e[0]["fn"], R.FNS)
        for doc in ("CLAUDE.md", "PROJECT_CONTEXT.md", "docs/specs/features/ml-admission-match-1009-spec.md"):
            self.assertIn("ML_PROMOTION_MATCHED_ADMISSION_ENABLED", (ROOT / doc).read_text(encoding="utf-8"), doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
