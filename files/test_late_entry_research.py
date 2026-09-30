"""Late-entry research scripts (late-entry-0930-spec.md): the exit replay they share
and the time split the regime look depends on."""
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import exit_validator as EV
import _backtest_crossing_entry as X
import _backtest_crossing_entry_regime as XR

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _series(path):
    out, prev = [], path[0]
    for i, c in enumerate(path):
        out.append((T0 + timedelta(minutes=15 * i), prev, max(prev, c) * 1.001, min(prev, c) * 0.999, c, 1.0))
        prev = c
    return out


def _grid(rise=True):
    n = 96 * 3
    a = [100.0] * 96 + ([100.0 * (1 + 0.004 * i) if i < 50 else 120.0 for i in range(96)] if rise
                        else [100.0 * (1 + 0.004 * i) if i < 10 else 90.0 for i in range(96)]) + [110.0] * 96
    return EV.Grid({"AUSDT": _series(a), "BUSDT": _series([50.0] * n), "CUSDT": _series([10.0] * n)}, 14, min_bars=10)


P = {"rank_max": 1, "min_ret": 0.03, "floor": 0.08, "lost_rank": 10, "lost_min_bars": 4, "max_hold_bars": 96}


class TestSharedExit(unittest.TestCase):
    def test_leader_switch_holds_the_rise(self):
        g = _grid(True)
        pnl, switched = X.sim_exit(g, g.col["AUSDT"], 96 + 7, 2.0, 0.015, P)   # entry around +2.8%
        self.assertTrue(switched)
        self.assertGreater(pnl, 5.0)

    def test_collapse_exits_on_the_trail_without_a_switch(self):
        g = _grid(False)
        pnl, switched = X.sim_exit(g, g.col["AUSDT"], 96 + 5, 2.0, 0.015, dict(P, min_ret=0.50))
        self.assertFalse(switched)
        self.assertLess(pnl, 0.0)


class TestPreRegistration(unittest.TestCase):
    def test_trigger_and_variants(self):
        self.assertEqual(X.TRIG, 0.025)
        self.assertEqual(X.KS, (None, 10, 5, 3))

    def test_regime_choice_never_sees_the_test_months(self):
        self.assertEqual(XR.TRAIN_END, "2026-07-01")
        src = (HERE / "_backtest_crossing_entry_regime.py").read_text(encoding="utf-8")
        choose = src[src.index("best = None"):src.index("chosen on TRAIN")]
        self.assertIn("pick(tr,", choose)
        self.assertNotIn("pick(te,", choose)

    def test_negative_result_is_documented(self):
        spec = (ROOT / "docs/specs/features/late-entry-0930-spec.md").read_text(encoding="utf-8")
        self.assertIn("REFUTED", spec)
        for doc in ("CLAUDE.md", "PROJECT_CONTEXT.md"):
            self.assertIn("late-entry-0930-spec.md", (ROOT / doc).read_text(encoding="utf-8"), doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
