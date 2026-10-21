"""Tests for the pre-registered recheck LEADER-POSITIONING-OI (spec:
docs/specs/features/leader-positioning-oi-1021-spec.md). Written before the code."""
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _readout_leader_positioning_oi as R  # noqa: E402
import readouts  # noqa: E402

UTC = timezone.utc


class TestFrozen(unittest.TestCase):
    def test_rule_is_frozen_before_the_new_window(self):
        self.assertEqual(R.SEEN_FROM, "2026-08-21")
        self.assertEqual(R.SEEN_TO, "2026-10-19")
        self.assertGreater(R.NEW_FROM, R.SEEN_TO)          # judged only on days nobody has looked at
        self.assertIsInstance(R.OI1H_MEDIAN_A, float)
        self.assertIsInstance(R.OI1H_MEDIAN_D, float)

    def test_rule(self):
        self.assertTrue(R.passes({"oi_1h": R.OI1H_MEDIAN_A}, R.OI1H_MEDIAN_A))
        self.assertFalse(R.passes({"oi_1h": R.OI1H_MEDIAN_A - 1e-9}, R.OI1H_MEDIAN_A))
        self.assertFalse(R.passes({"oi_1h": None}, R.OI1H_MEDIAN_A))
        self.assertFalse(R.passes(None, R.OI1H_MEDIAN_A))


class TestVerdictA(unittest.TestCase):
    def test_too_early_below_30_positives(self):
        self.assertEqual(R.verdict_a(pos_pop=29, n_pop=150, k_rule=20, n_rule=40), "TOO_EARLY")

    def test_refuted_when_not_above_base(self):
        # base 30/150 = 0.20; rule 8/40 = 0.20
        self.assertEqual(R.verdict_a(pos_pop=30, n_pop=150, k_rule=8, n_rule=40), "REFUTED")

    def test_refuted_when_rule_selects_nothing(self):
        self.assertEqual(R.verdict_a(pos_pop=30, n_pop=150, k_rule=0, n_rule=0), "REFUTED")

    def test_supported_when_wilson_lower_bound_above_base(self):
        # base 0.20; rule 25/50 = 0.50, Wilson lower ~0.37
        self.assertEqual(R.verdict_a(pos_pop=40, n_pop=200, k_rule=25, n_rule=50), "SUPPORTED")

    def test_inconclusive_in_between(self):
        # base 0.20; rule 12/45 = 0.27, Wilson lower ~0.16
        self.assertEqual(R.verdict_a(pos_pop=40, n_pop=200, k_rule=12, n_rule=45), "INCONCLUSIVE")


class TestVerdictD(unittest.TestCase):
    ok = dict(n_rule=60, prec=0.25, bot_prec=0.20, pnl_lo=-0.20, bot_mean=-0.15, reach=7, n_late=60)

    def test_supported_needs_all_three(self):
        self.assertEqual(R.verdict_d(**self.ok), "SUPPORTED")

    def test_too_early_below_60(self):
        self.assertEqual(R.verdict_d(**{**self.ok, "n_rule": 59}), "TOO_EARLY")

    def test_each_criterion_can_refute(self):
        for k, v in (("prec", 0.19), ("pnl_lo", -0.26), ("reach", 5)):
            self.assertEqual(R.verdict_d(**{**self.ok, k: v}), "REFUTED", k)


class TestStage(unittest.TestCase):
    op = datetime(2026, 11, 1, tzinfo=UTC)
    t25 = op + timedelta(hours=5)

    def ev(self, kind, h):
        return {"event": kind, "_dt": self.op + timedelta(hours=h)}

    def test_stages(self):
        self.assertEqual(R.stage_of([], self.op, self.t25), "none")
        self.assertEqual(R.stage_of([self.ev("entry", 2)], self.op, self.t25), "early")
        self.assertEqual(R.stage_of([self.ev("entry", 7)], self.op, self.t25), "late")
        self.assertEqual(R.stage_of([self.ev("entry", -3), self.ev("entry", 7)], self.op, self.t25), "held")
        self.assertEqual(R.stage_of([self.ev("entry", -3), self.ev("exit", -1), self.ev("entry", 7)],
                                    self.op, self.t25), "late")
        self.assertEqual(R.stage_of([self.ev("entry", 25)], self.op, self.t25), "none")   # next day


class TestWiring(unittest.TestCase):
    def test_registry_entry(self):
        reg = {r["id"]: r for r in readouts.load_registry()}
        r = reg["LEADER-POSITIONING-OI"]
        self.assertEqual(r["kind"], "computed")
        self.assertEqual(r["fn"], "readout_leader_positioning_oi")
        self.assertEqual(r["window_from"], R.NEW_FROM)
        self.assertEqual(r["due"], R.DUE)
        self.assertIn(r["fn"], readouts.FNS)

    def test_new_verdicts_are_final(self):
        reg = {r["id"]: r for r in readouts.load_registry()}["LEADER-POSITIONING-OI"]
        for v in ("SUPPORTED", "REFUTED"):
            self.assertEqual(readouts.status_of(reg, {"days": 99, "_n_trades": 99, "_verdict": v}, "2027-01-01"), v)
            self.assertIn(v, readouts.FINAL_STATUSES)

    def test_readout_reads_the_result_file(self):
        reg = {r["id"]: r for r in readouts.load_registry()}["LEADER-POSITIONING-OI"]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            p.write_text(json.dumps({"computed": "2026-12-16", "A": {"verdict": "REFUTED", "pos_pop": 31},
                                     "D": {"verdict": "TOO_EARLY", "n_rule": 40}}), encoding="utf-8")
            with unittest.mock.patch.object(R, "RESULT", p):
                res = readouts.readout_leader_positioning_oi(reg, {"days": ["2026-10-20"] * 40})
        self.assertEqual(res["A"]["verdict"], "REFUTED")
        self.assertEqual(res["_verdict"], "TOO_EARLY")         # final only when BOTH parts are decided

    def test_overall(self):
        self.assertEqual(R.overall("REFUTED", "REFUTED"), "REFUTED")
        self.assertEqual(R.overall("SUPPORTED", "REFUTED"), "SUPPORTED")
        self.assertEqual(R.overall("INCONCLUSIVE", "REFUTED"), "INCONCLUSIVE")
        self.assertEqual(R.overall("TOO_EARLY", "SUPPORTED"), "TOO_EARLY")

    def test_missing_result_is_too_early(self):
        reg = {r["id"]: r for r in readouts.load_registry()}["LEADER-POSITIONING-OI"]
        with unittest.mock.patch.object(R, "RESULT", Path(tempfile.gettempdir()) / "no_such_lp_oi.json"):
            res = readouts.readout_leader_positioning_oi(reg, {"days": []})
        self.assertEqual(res["_verdict"], "TOO_EARLY")

    def test_if_due_does_nothing_before_due(self):
        self.assertFalse(R.should_run("2026-12-15"))
        self.assertTrue(R.should_run(R.DUE))

    def test_pipeline_runs_it_before_readouts(self):
        src = (HERE / "pipeline_run.py").read_text(encoding="utf-8")
        a = src.index("_readout_leader_positioning_oi.py")
        b = src.index('run_step("readouts"')
        self.assertLess(a, b)


import unittest.mock  # noqa: E402

if __name__ == "__main__":
    unittest.main(verbosity=2)
