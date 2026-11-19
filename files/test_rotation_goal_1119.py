"""Rotation when the portfolio is full (rotation-goal-1119-spec.md), written from the spec first."""
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import _backtest_rotation_goal as RG

ROOT = Path(__file__).resolve().parent.parent
T = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
M15 = timedelta(minutes=15)


def pos(sym="A", tf="15m", bars=4, ev=-0.5, entry=1.0):
    return {"sym": sym, "tf": tf, "entry_dt": T - bars * (M15 if tf == "15m" else timedelta(hours=1)),
            "exit_dt": T + timedelta(hours=5), "entry_price": entry, "ranker_ev": ev}


class TestEligibility(unittest.TestCase):
    def test_live_rules(self):
        self.assertTrue(RG.eligible(pos(), T, 1.0, None))
        self.assertFalse(RG.eligible(pos(bars=2), T, 1.0, None))                     # too fresh
        self.assertFalse(RG.eligible(pos(), T, 1.006, None))                         # +0.6% protected
        self.assertFalse(RG.eligible(pos(), T, 1.0, T - M15))                        # leader before t
        self.assertTrue(RG.eligible(pos(), T, 1.0, T + M15))                         # leader only later
        self.assertFalse(RG.eligible(pos(ev=-0.40), T, 1.0, None))                   # not below -0.40
        self.assertFalse(RG.eligible(pos(ev=None), T, 1.0, None))                    # unknown ev
        self.assertTrue(RG.eligible(pos(tf="1h", bars=3), T, 1.0, None))             # bars in its own tf

    def test_weakest_is_lowest_ev(self):
        ps = [pos("A", ev=-0.5), pos("B", ev=-1.2), pos("C", ev=-0.9)]
        self.assertEqual(RG.weakest(ps)["sym"], "B")
        self.assertIsNone(RG.weakest([]))


class TestSelection(unittest.TestCase):
    def test_top_share_per_month(self):
        rows = [{"month": "2026-10", "ml_proba": p} for p in (0.1, 0.2, 0.3, 0.4)] + \
               [{"month": "2026-11", "ml_proba": p} for p in (0.9, 0.05, 0.06, 0.07)]
        self.assertEqual(RG.top_share_mask(rows, 0.25), [False, False, False, True, True, False, False, False])

    def test_unknown_proba_never_selected(self):
        rows = [{"month": "2026-10", "ml_proba": None}, {"month": "2026-10", "ml_proba": 0.2}]
        self.assertEqual(RG.top_share_mask(rows, 0.5), [False, True])


class TestGoal(unittest.TestCase):
    def test_gained_days(self):
        op = T.replace(hour=0)
        W = [("2026-10-15", "X", op, op + timedelta(hours=14)),     # no real early, added early -> gained
             ("2026-10-15", "Y", op, op + timedelta(hours=14)),     # real early exists -> not gained
             ("2026-10-15", "Z", op, op + timedelta(hours=10))]     # added after the crossing -> not gained
        real = {"Y": [op + timedelta(hours=3)]}
        added = {"X": [T], "Y": [T], "Z": [T]}
        self.assertEqual(RG.gained(W, real, added), [1, 0, 0])


class TestVerdict(unittest.TestCase):
    def test_rule(self):
        self.assertEqual(RG.verdict(29, 50, 1.0, 0.2, 0.0), "needs_data")
        self.assertEqual(RG.verdict(100, 19, 1.0, 0.2, 0.0), "needs_data")
        self.assertEqual(RG.verdict(100, 50, 1.0, 0.2, -0.05), "supported")
        self.assertEqual(RG.verdict(100, 50, 0.0, -0.2, 0.0), "refuted")
        self.assertEqual(RG.verdict(100, 50, 1.0, 0.2, -0.2), "refuted")
        self.assertEqual(RG.verdict(100, 50, 0.4, 0.1, 0.0), "inconclusive")

    def test_pre_registration(self):
        self.assertEqual((RG.WEAK_BARS, RG.PROTECT_PCT, RG.EV_MAX, RG.TOP_SHARE, RG.CI),
                         (3, 0.5, -0.40, 0.25, 0.975))

    def test_documented(self):
        self.assertIn("rotation-goal-1119",
                      (ROOT / "docs/specs/features/rotation-goal-1119-spec.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
