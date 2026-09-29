"""exit_validator.py -- L3 for exit / leader-mode keys (agent-tasks-0929-spec.md, task 3)."""
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import exit_validator as EV

ROOT = Path(__file__).resolve().parent.parent
T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _series(path):
    """[(t, o, h, l, c, v)] from a list of closes, 15m bars from T0."""
    out = []
    prev = path[0]
    for i, c in enumerate(path):
        out.append((T0 + timedelta(minutes=15 * i), prev, max(prev, c) * 1.001, min(prev, c) * 0.999, c, 1.0))
        prev = c
    return out


def _grid():
    n = 96 * 3
    rocket = [100.0] * 96 + [100.0 * (1 + 0.004 * i) if i < 50 else 120.0 for i in range(96)] + [120.0] * 96
    flat = [50.0] * n
    slow = [10.0 * (1 + 0.0002 * i) for i in range(n)]
    return EV.Grid({"AUSDT": _series(rocket), "BUSDT": _series(flat), "CUSDT": _series(slow)}, 14, min_bars=10)


def _trade(g, ie=110, ix=130, real=1.0, live_leader=False, mode="impulse_speed", day="2026-09-02"):
    return {"sym": "AUSDT", "j": g.col["AUSDT"], "ie": ie, "ix": ix, "ep": float(g.C[ie, g.col["AUSDT"]]),
            "real": real, "k": 2.0, "mode": mode, "day": day, "leader_exit_live": live_leader}


P = {"rank_max": 1, "min_ret": 0.03, "floor": 0.08, "lost_rank": 10, "lost_min_bars": 4, "max_hold_bars": 96}


class TestLeaderReplay(unittest.TestCase):
    def setUp(self):
        self.g = _grid()

    def test_rank_is_the_day_return_rank(self):
        j = self.g.col["AUSDT"]
        self.assertEqual(self.g.RANK[140, j], 1)        # +17% on day 2 leads the three coins
        self.assertAlmostEqual(self.g.RET[96 + 49, j], 0.196, places=2)

    def test_switch_holds_the_rocket(self):
        vals, sw = EV.leader_arm(self.g, [_trade(self.g)], P)
        self.assertIsNotNone(sw[0])
        self.assertGreater(vals[0], 5.0)                # held on the wide trail, not the +1% real exit

    def test_no_switch_keeps_the_real_exit(self):
        vals, sw = EV.leader_arm(self.g, [_trade(self.g)], dict(P, min_ret=0.50))
        self.assertIsNone(sw[0])
        self.assertEqual(vals[0], 1.0)

    def test_live_leader_exit_without_switch_is_unobservable(self):
        vals, _ = EV.leader_arm(self.g, [_trade(self.g, live_leader=True)], dict(P, min_ret=0.50))
        self.assertIsNone(vals[0])

    def test_lost_rank_exit(self):
        t = _trade(self.g)
        s = 110
        p = EV.wide_from(self.g, t, s, dict(P, lost_rank=0, lost_min_bars=4))   # "out of top-0" at once
        j = self.g.col["AUSDT"]
        self.assertAlmostEqual(p, (self.g.C[s + 4, j] / t["ep"] - 1) * 100, places=6)

    def test_plain_trail_floor_widens(self):
        t = _trade(self.g, ie=96 + 55)                 # after the rise, flat: floor decides nothing breaks
        a, b = EV.plain_trail(self.g, t, 0.0), EV.plain_trail(self.g, t, 0.08)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)


class TestDecision(unittest.TestCase):
    def s(self, n, lo, hi, mean=0.0):
        return {"n": n, "lo95": lo, "hi95": hi, "mean": mean}

    def test_rules(self):
        ok_all = self.s(500, -0.05, 0.05)
        self.assertEqual(EV.decide(ok_all, self.s(40, 0.1, 0.5), 0.7, 100), "accept")
        self.assertEqual(EV.decide(ok_all, self.s(40, 0.1, 0.5), 0.5, 100), "needs_review")   # months disagree
        self.assertEqual(EV.decide(self.s(500, -0.2, 0.1), self.s(40, 0.1, 0.5), 1.0, 100), "reject")
        self.assertEqual(EV.decide(ok_all, self.s(40, -0.5, -0.1), 1.0, 100), "reject")
        self.assertEqual(EV.decide(ok_all, self.s(40, 0.1, 0.5), 1.0, 0), "reject")           # touches nothing
        self.assertEqual(EV.decide(ok_all, self.s(20, 0.1, 0.5), 1.0, 100), "needs_data")
        self.assertEqual(EV.decide(ok_all, self.s(40, 0.1, 0.5), 1.0, 10), "needs_data")      # < MIN_CHANGED
        self.assertEqual(EV.decide(self.s(100, 0, 1), self.s(40, 0.1, 0.5), 1.0, 100), "needs_data")


class TestValidate(unittest.TestCase):
    def test_unknown_key_and_same_value(self):
        cfg = types.SimpleNamespace(LEADER_EXIT_RANK_MAX=5, ATR_PERIOD=14)
        self.assertEqual(EV.validate({"config_key": "NOPE", "diff": {"to": 1}}, cfg_module=cfg)["verdict"], "reject")
        self.assertEqual(EV.validate({"config_key": "LEADER_EXIT_RANK_MAX", "diff": {"to": 5}},
                                     cfg_module=cfg)["verdict"], "reject")

    def test_leader_key_report(self):
        g = _grid()
        cfg = types.SimpleNamespace(LEADER_EXIT_RANK_MAX=1, LEADER_EXIT_MIN_RET=0.50, LEADER_EXIT_FLOOR_PCT=0.08,
                                    LEADER_EXIT_LOST_RANK=10, LEADER_EXIT_LOST_MIN_BARS=4,
                                    LEADER_EXIT_MAX_HOLD_BARS=96, ATR_PERIOD=14)
        trades = [_trade(g)]
        r = EV.validate({"config_key": "LEADER_EXIT_MIN_RET", "diff": {"to": 0.03}}, cfg_module=cfg, grid=g,
                        trades=trades, labels=({("2026-09-02", "AUSDT")}, set()),
                        rocket_days={("2026-09-02", "AUSDT")})
        self.assertEqual(r["changed_trades"], 1)
        self.assertGreater(r["rocket_day_diff_pp"]["mean"], 0)
        self.assertEqual(r["leader_coverage"]["held_in_leader_mode_now"], 0)
        self.assertEqual(r["leader_coverage"]["held_in_leader_mode_after"], 1)
        self.assertEqual(r["verdict"], "needs_data")                      # one trade is not evidence

    def test_keys(self):
        self.assertIn("LEADER_EXIT_LOST_RANK", EV.KEYS)
        self.assertIn("TRAIL_MIN_BUFFER_PCT_ALIGNMENT", EV.KEYS)
        import config
        for k in EV.KEYS:
            self.assertTrue(hasattr(config, k), k)


class TestWiring(unittest.TestCase):
    def test_dispatch_routes_exit_keys(self):
        import config
        import pipeline_validator as PV
        self.assertIsNone(PV._exit_verdict({"config_key": "TREND_1H_CHOP_ADX_MIN"}))
        old = config.L3_EXIT_VALIDATOR_ENABLED
        config.L3_EXIT_VALIDATOR_ENABLED = False
        try:
            self.assertIsNone(PV._exit_verdict({"config_key": "LEADER_EXIT_LOST_RANK", "diff": {"to": 15}}))
        finally:
            config.L3_EXIT_VALIDATOR_ENABLED = old
        self.assertTrue(config.L3_EXIT_VALIDATOR_ENABLED)

    def test_l2_is_offered_exit_keys(self):
        import pipeline_hypothesis as PH
        keys = PH._validatable_with_values()
        self.assertIn("LEADER_EXIT_LOST_RANK", keys)
        self.assertIn("TRAIL_MIN_BUFFER_PCT_TREND", keys)
        self.assertIn("LEADER_EXIT_", PH._CLAUDE_SYSTEM)

    def test_documented(self):
        for doc in ("CLAUDE.md", "PROJECT_CONTEXT.md", "docs/specs/features/agent-tasks-0929-spec.md"):
            self.assertIn("exit_validator.py", (ROOT / doc).read_text(encoding="utf-8"), doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
