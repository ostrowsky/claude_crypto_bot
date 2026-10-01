"""Report steps of 2026-10-01 researched on the maximum period (valid-proposals-1001-spec.md):
gate locks re-judged by the goal, and alert-budget ranking."""
import json
import os
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import bot_health_report as H
import goal_validator as GV
import _backtest_alert_budget as AB
import _replay_locked_gates_goal as RL

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UTC = timezone.utc


def _events(rows):
    fd, p = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return Path(p)


class TestGateOff(unittest.TestCase):
    def test_only_rows_the_gate_blocked_are_admitted(self):
        t = "2026-09-10T0{h}:05:00+00:00"
        rows = [
            # blocked by mode_range_quality (stage 3) -> admitted when the gate is removed
            {"ts": t.format(h=3), "event": "blocked", "sym": "AAAUSDT", "tf": "1h", "price": 100.0,
             "signal_type": "mode_range_quality", "reason_code": "mode_range_quality"},
            # blocked EARLIER by the ML gate (stage 1) -> never reached the gate under test
            {"ts": t.format(h=3), "event": "blocked", "sym": "BBBUSDT", "tf": "1h", "price": 100.0,
             "signal_type": "ml_proba_zone", "reason_code": "ml_zone"},
        ]
        p = _events(rows)
        d0 = datetime(2026, 9, 10, tzinfo=UTC)
        try:
            start = datetime(2026, 9, 1, tzinfo=UTC)
            bars = [(start + timedelta(hours=i), 100 + 0.1 * i, 100.5 + 0.1 * i, 99.5 + 0.1 * i, 100 + 0.1 * i)
                    for i in range(400)]
            r = GV.validate_gate_off(("mode_range_quality",), "2026-09-01", events_path=p,
                                     bars_loader=lambda s, tf: bars,
                                     winners=[("2026-09-10", "AAAUSDT", d0, d0 + timedelta(hours=10)),
                                              ("2026-09-10", "BBBUSDT", d0, d0 + timedelta(hours=10))])
        finally:
            os.unlink(p)
        self.assertEqual(r["goal"]["newly_reachable_days"], 1)
        self.assertEqual(r["verdict"], "needs_data")
        self.assertTrue(r["config_key"].startswith("GATE_OFF:"))

    def test_a_gate_without_a_stage_is_not_replayable(self):
        self.assertEqual(GV.validate_gate_off(("mtf",))["verdict"], "needs_data")

    def test_ml_lock_is_judged_on_the_current_model_only(self):
        self.assertEqual(RL.GATES["ml_proba_zone"]["since"], "2026-09-07")
        self.assertEqual(RL.STATUS, {"reject": "confirmed", "accept": "unsupported"})
        self.assertEqual(RL.NOT_REPLAYABLE, ("mtf", "cooldown"))


class TestReportSteps(unittest.TestCase):
    SC = {"north_star": {"status": "verified"}, "portfolio_alpha": {"value": -1.0},
          "realized_potential": {"value": 0.6, "target": 0.5},
          "signal_precision": {"value": 13.6, "target": 35.0}, "message_rate": {"value": 31.0, "target_max": 10.0}}
    TR = {"evaluation_scope": "out_of_sample_time_holdout"}

    def test_contested_lock_is_an_operator_decision(self):
        dnt = {"contested": [{"name": "mode_range_quality", "goal_evidence": "goal +3.43 pp"}]}
        with mock.patch.object(H, "_alert_budget_tested", return_value={"date": "2026-10-01"}):
            ids = {s["id"]: s for s in H.derive_next_steps(self.SC, self.TR, dnt, date(2026, 10, 1))}
        self.assertEqual(ids["decide_contested_gate_mode_range_quality"]["priority"], "P1")
        self.assertNotIn("honest_alert_budget_ranker", ids)        # tested and refuted -> not re-offered

    def test_untested_budget_step_still_offered(self):
        with mock.patch.object(H, "_alert_budget_tested", return_value=None):
            ids = [s["id"] for s in H.derive_next_steps(self.SC, self.TR, {}, date(2026, 10, 1))]
        self.assertIn("honest_alert_budget_ranker", ids)


class TestAlertBudget(unittest.TestCase):
    def test_pre_registration(self):
        self.assertEqual((AB.START, AB.SPLIT), ("2026-05-01", "2026-08-01"))
        self.assertEqual(AB.BUDGETS, (3, 5, 10))
        self.assertIn("logit_all", AB.SCORES)

    def test_wilson_and_logit(self):
        lo, hi = AB.wilson(50, 100)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        rows = [{"a": float(i), "win": 1.0 if i > 5 else 0.0} for i in range(12)]
        f = AB.fit_logit(rows, ["a"])
        self.assertGreater(f({"a": 11.0}), f({"a": 0.0}))

    def test_documented(self):
        spec = (ROOT / "docs/specs/features/valid-proposals-1001-spec.md").read_text(encoding="utf-8")
        self.assertIn("REFUTED", spec)
        for doc in ("CLAUDE.md", "PROJECT_CONTEXT.md"):
            self.assertIn("valid-proposals-1001-spec.md", (ROOT / doc).read_text(encoding="utf-8"), doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
