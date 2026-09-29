"""trend-scout L3 gate and the ML promotion gate (agent-tasks-0929-spec.md §5, §6)."""
import json
import random
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ml_promotion_gate as MPG
import trend_scout as TS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


# ------------------------------------------------------------------ trend-scout

def _vr(param="CLONE_SIGNAL_GUARD_MAX_SIMILAR", cur=22, new=23, verdict="approve", risk="low"):
    rule = types.SimpleNamespace(risk=risk, is_integer=True)
    p = types.SimpleNamespace(config_param=param, current_value=cur, proposed_value=new,
                              affected_syms=["AUSDT"], rule=rule, rationale="")
    return types.SimpleNamespace(proposal=p, verdict=verdict, new_entries_count=40,
                                 new_entries_avg_ret5=0.3, new_entries_win_rate=0.5,
                                 new_entries_avg_ret10=0.4, reason="")


class TestScoutL3Gate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = (TS.L3_CACHE_FILE, TS.CHANGELOG_FILE, TS._apply_config_change, TS.l3_verdict)
        TS.L3_CACHE_FILE = Path(self.tmp.name) / "cache.json"
        TS.CHANGELOG_FILE = Path(self.tmp.name) / "changelog.jsonl"
        self.applied = []
        TS._apply_config_change = lambda param, v, is_integer=False: self.applied.append((param, v)) or True

    def tearDown(self):
        TS.L3_CACHE_FILE, TS.CHANGELOG_FILE, TS._apply_config_change, TS.l3_verdict = self.old
        self.tmp.cleanup()

    def test_no_replay_is_held(self):
        r = TS.l3_verdict("CLONE_SIGNAL_GUARD_MAX_SIMILAR", 23, validators=[({"OTHER"}, None)])
        self.assertEqual(r["verdict"], "no_goal_replay")

    def test_routes_to_the_validator_and_caches(self):
        calls = []

        def fake(h):
            calls.append(h)
            return {"verdict": "accept", "validator": "goal_validator", "reason": "ok"}
        v = [({"ML_GENERAL_HARD_BLOCK_MIN"}, fake)]
        r1 = TS.l3_verdict("ML_GENERAL_HARD_BLOCK_MIN", 0.12, validators=v)
        r2 = TS.l3_verdict("ML_GENERAL_HARD_BLOCK_MIN", 0.12, validators=v)
        self.assertEqual(r1["verdict"], "accept")
        self.assertTrue(r2.get("cached"))
        self.assertEqual(len(calls), 1)
        # an expired cache entry is re-validated
        later = datetime.now(timezone.utc) + timedelta(days=TS.L3_CACHE_DAYS + 1)
        TS.l3_verdict("ML_GENERAL_HARD_BLOCK_MIN", 0.12, validators=v, now=later)
        self.assertEqual(len(calls), 2)

    def test_apply_only_on_accept(self):
        import config
        old = getattr(config, "TREND_SCOUT_AUTO_APPLY_REQUIRES_L3", None)
        config.TREND_SCOUT_AUTO_APPLY_REQUIRES_L3 = True
        try:
            TS.l3_verdict = lambda p, v: {"verdict": "no_goal_replay", "reason": "x"}
            self.assertEqual(TS.apply_approved_changes([_vr()]), [])
            self.assertEqual(self.applied, [])
            self.assertEqual(TS.LAST_HELD[0]["l3_verdict"], "no_goal_replay")
            log = [json.loads(l) for l in TS.CHANGELOG_FILE.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(log[-1]["held"])
            TS.l3_verdict = lambda p, v: {"verdict": "accept"}
            out = TS.apply_approved_changes([_vr()])
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0]["l3_verdict"], "accept")
            self.assertEqual(self.applied, [("CLONE_SIGNAL_GUARD_MAX_SIMILAR", 23)])
        finally:
            config.TREND_SCOUT_AUTO_APPLY_REQUIRES_L3 = old

    def test_flag_off_restores_proxy_path(self):
        import config
        old = getattr(config, "TREND_SCOUT_AUTO_APPLY_REQUIRES_L3", None)
        config.TREND_SCOUT_AUTO_APPLY_REQUIRES_L3 = False
        try:
            TS.l3_verdict = lambda p, v: self.fail("L3 must not run when the flag is off")
            self.assertEqual(len(TS.apply_approved_changes([_vr()])), 1)
        finally:
            config.TREND_SCOUT_AUTO_APPLY_REQUIRES_L3 = old

    def test_flag_on_by_default(self):
        import config
        self.assertTrue(config.TREND_SCOUT_AUTO_APPLY_REQUIRES_L3)


# ------------------------------------------------------------------ ML promotion gate

class TestPromotionGate(unittest.TestCase):
    def test_metrics(self):
        y = [1, 1, 0, 0]
        self.assertAlmostEqual(MPG.auc(y, [0.9, 0.8, 0.2, 0.1]), 1.0)
        self.assertAlmostEqual(MPG.auc(y, [0.5, 0.5, 0.5, 0.5]), 0.5)
        self.assertIsNone(MPG.auc([1, 1], [0.1, 0.2]))
        f = [0.5] * 4
        self.assertAlmostEqual(MPG.recall_at_floor(y, [0.9, 0.1, 0.9, 0.1], f), 0.5)
        self.assertAlmostEqual(MPG.precision_at_floor(y, [0.9, 0.1, 0.9, 0.1], f), 0.5)

    def test_floors_follow_the_day(self):
        cfg = types.SimpleNamespace(ML_GENERAL_HARD_BLOCK_MIN=0.15, ML_GENERAL_HARD_BLOCK_BULL_DAY_MIN=0.10)
        self.assertEqual(MPG.floors_for([{"is_bull_day": True}, {}], cfg), [0.10, 0.15])

    def test_decide(self):
        self.assertEqual(MPG.decide({"auc_diff_ci95": [-0.05, -0.01]})[0], "keep_incumbent")
        self.assertEqual(MPG.decide({"auc_diff_ci95": [-0.01, 0.02], "recall_diff_ci95": [-0.2, -0.05]})[0],
                         "keep_incumbent")
        self.assertEqual(MPG.decide({"auc_diff_ci95": [-0.01, 0.02], "recall_diff_ci95": [0.05, 0.2],
                                     "precision_diff_ci95": [-0.05, -0.02]})[0], "keep_incumbent")
        # a tie is promoted: learning must not freeze
        self.assertEqual(MPG.decide({"auc_diff_ci95": [-0.01, 0.01], "recall_diff_ci95": [-0.02, 0.02],
                                     "precision_diff_ci95": [-0.02, 0.02]})[0], "promote")

    def _holdout(self, n=400):
        rnd = random.Random(3)
        rows = [{"x": rnd.random(), "is_bull_day": False} for _ in range(n)]
        y = [1.0 if r["x"] > 0.7 else 0.0 for r in rows]
        return {"rows": rows, "y": y}

    def _eval(self, inc, cand, holdout):
        cfg = types.SimpleNamespace(ML_GENERAL_HARD_BLOCK_MIN=0.5, ML_GENERAL_HARD_BLOCK_BULL_DAY_MIN=0.5)
        predict = lambda payload, r: payload["f"](r["x"])  # noqa: E731
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "inc.json"
            if inc is not None:
                p.write_text(json.dumps({k: v for k, v in inc.items() if k != "f"}), encoding="utf-8")
                real = MPG.json.loads

                def loads(s, _inc=inc):
                    out = real(s)
                    out["f"] = _inc["f"]
                    return out
                MPG.json.loads = loads
            try:
                return MPG.evaluate(p, cand, holdout, cfg=cfg, predict=predict)
            finally:
                if inc is not None:
                    MPG.json.loads = real

    def test_clearly_worse_candidate_is_not_promoted(self):
        good = {"label_version": "peak", "f": lambda x: x}
        bad = {"label_version": "peak", "f": lambda x: 1 - x}
        r = self._eval(good, bad, self._holdout())
        self.assertEqual(r["decision"], "keep_incumbent")

    def test_equal_candidate_is_promoted(self):
        good = {"label_version": "peak", "f": lambda x: x}
        r = self._eval(good, dict(good), self._holdout())
        self.assertEqual(r["decision"], "promote")

    def test_ungated_cases(self):
        cand = {"label_version": "peak", "f": lambda x: x}
        self.assertTrue(self._eval(None, cand, self._holdout())["ungated"])                   # no incumbent
        old = {"label_version": "close", "f": lambda x: x}
        self.assertIn("label version", self._eval(old, cand, self._holdout())["reason"])
        self.assertTrue(self._eval(dict(cand), cand, self._holdout(50))["ungated"])           # tiny holdout

    def test_log_and_consecutive_keeps(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "log.jsonl"
            for dec in ("keep_incumbent", "promote", "keep_incumbent", "keep_incumbent"):
                MPG.log_decision({"decision": dec, "reason": "r"}, p)
            self.assertEqual(MPG.consecutive_keeps(p), 2)
            import pipeline_notify as N
            self.assertIn("2 ноч", N.build_ml_gate_line(p))

    def test_trainer_is_gated(self):
        src = (HERE / "ml_signal_model.py").read_text(encoding="utf-8")
        self.assertIn("ML_PROMOTION_GATE_ENABLED", src)
        self.assertIn('"_holdout"', src)
        self.assertIn("MPG.evaluate(args.model_out, payload, holdout)", src)
        import config
        self.assertTrue(config.ML_PROMOTION_GATE_ENABLED)

    def test_documented(self):
        for doc in ("CLAUDE.md", "PROJECT_CONTEXT.md", "docs/specs/features/agent-tasks-0929-spec.md"):
            txt = (ROOT / doc).read_text(encoding="utf-8")
            self.assertIn("ml_promotion_gate.py", txt, doc)
            self.assertIn("TREND_SCOUT_AUTO_APPLY_REQUIRES_L3", txt, doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
