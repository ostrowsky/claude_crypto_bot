"""E4 scale-out on top of X4 (e4-scale-out-1118-spec.md §5). Written BEFORE the code. Synthetic, offline.

Interface the tests fix:
  trading_policy.Exits(kind="p1_scale", floor_override, max_bars, take_pct, take_frac=0.5)
  trading_policy.exit_plan(policy, grid, cand) -> (exit_bar, raw_price, reason, stop_dist)
  experiment_e4s.BASE (= X4), candidates() -> {name: policy}, admit(res), select(train) -> name | None,
  pipeline(validate, select, test, marker) -> {"verdict", "winner", ...}
"""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from test_execution_replay import FakeGrid, cand


def x4():
    import trading_policy as TP
    return replace(TP.CHAMPION, name="X4_p1_floor8", exits=TP.Exits(kind="p1_trail", floor_override=0.08))


def scale(take, frac=0.5):
    import trading_policy as TP
    return replace(TP.CHAMPION, name="s", exits=TP.Exits(kind="p1_scale", floor_override=0.08, take_pct=take, take_frac=frac))


def rising_then_drop(n=40, peak_at=10, top=120.0):
    """100 at entry, linear rise to `top` at bar peak_at, then a fall that trips the 8% trail."""
    c = []
    for i in range(n):
        if i <= peak_at:
            c.append(100.0 + (top - 100.0) * i / peak_at)
        else:
            c.append(top * (1 - 0.03 * (i - peak_at)))
    return c


class TestEpoch(unittest.TestCase):
    def test_running_policies_keep_their_epoch(self):
        import trading_policy as TP
        self.assertEqual(x4().epoch(), "0bccb4b2cf6d04f9")
        self.assertEqual(TP.CHAMPION.epoch(), "e196b96e7004b46b")

    def test_scale_policy_has_another_epoch(self):
        self.assertNotEqual(scale(0.08).epoch(), x4().epoch())


class TestPlan(unittest.TestCase):
    def test_no_crossing_equals_x4(self):
        import trading_policy as TP
        g = FakeGrid(rising_then_drop(top=105.0), atr=0.5)        # high never reaches +8%
        c = cand(0, 100.0)
        self.assertEqual(TP.exit_plan(scale(0.08), g, c), TP.exit_plan(x4(), g, c))

    def test_crossing_blends_take_and_runner(self):
        import trading_policy as TP
        g = FakeGrid(rising_then_drop(top=120.0), atr=0.5)
        c = cand(0, 100.0)
        base = TP.exit_plan(x4(), g, c)
        bar, px, reason, dist = TP.exit_plan(scale(0.08), g, c)
        level = 100.0 * 1.08
        self.assertEqual(bar, base[0])
        self.assertEqual(reason, "scale")
        self.assertAlmostEqual(px, 0.5 * level + 0.5 * base[1])
        self.assertEqual(dist, base[3])
        bar2, px2, _, _ = TP.exit_plan(scale(0.08, frac=0.25), g, c)
        self.assertAlmostEqual(px2, 0.25 * level + 0.75 * base[1])

    def test_touch_without_trade_through_is_no_fill(self):
        import trading_policy as TP
        g = FakeGrid(rising_then_drop(top=104.0), atr=0.5)
        g.H = g.C.copy()
        g.H[3, 0] = 100.0 * 1.08                                   # touches the level exactly
        c = cand(0, 100.0)
        self.assertEqual(TP.exit_plan(scale(0.08), g, c), TP.exit_plan(x4(), g, c))
        g.H[3, 0] = 100.0 * 1.08 * (1 + 2 * TP.TP_TICK)            # trades through
        self.assertEqual(TP.exit_plan(scale(0.08), g, c)[2], "scale")

    def test_crossing_after_the_x4_exit_is_ignored(self):
        import trading_policy as TP
        g = FakeGrid(rising_then_drop(top=104.0), atr=0.5)
        g.H = g.C.copy()
        c = cand(0, 100.0)
        base = TP.exit_plan(x4(), g, c)
        g.H[base[0] + 1, 0] = 200.0                                # far above the level, but after the exit
        self.assertEqual(TP.exit_plan(scale(0.08), g, c), base)

    def test_hourly_trade_follows_x4(self):
        import trading_policy as TP
        g = FakeGrid(rising_then_drop(top=120.0), atr=0.5)
        c = cand(0, 100.0, tf="1h")
        self.assertEqual(TP.exit_plan(scale(0.08), g, c), TP.exit_plan(x4(), g, c))


class TestCandidates(unittest.TestCase):
    def test_three_registered(self):
        import experiment_e4s as E
        c = E.candidates()
        self.assertEqual(sorted(c), ["S_t08", "S_t15", "S_t25"])
        for n, take in (("S_t08", 0.08), ("S_t15", 0.15), ("S_t25", 0.25)):
            ex = c[n].exits
            self.assertEqual((ex.kind, ex.take_pct, ex.take_frac, ex.floor_override, ex.max_bars), ("p1_scale", take, 0.5, 0.08, 96))
        self.assertEqual(E.BASE.epoch(), x4().epoch())


def res(mean, lo, changed):
    return {"paired": {"mean_pct": mean, "lo95": lo}, "changed": changed}


class TestSelection(unittest.TestCase):
    def test_admit_needs_both(self):
        import experiment_e4s as E
        self.assertTrue(E.admit(res(0.02, 0.001, 100)))
        self.assertFalse(E.admit(res(0.02, 0.001, 99)))
        self.assertFalse(E.admit(res(0.02, 0.0, 500)))
        self.assertFalse(E.admit(res(0.02, None, 500)))

    def test_select_largest_admitted_mean(self):
        import experiment_e4s as E
        tr = {"S_t08": res(0.01, 0.001, 300), "S_t15": res(0.05, -0.01, 300), "S_t25": res(0.02, 0.002, 150)}
        self.assertEqual(E.select(tr), "S_t25")
        self.assertIsNone(E.select({"S_t08": res(0.01, -0.001, 300)}))

    def test_stop_writes_no_marker(self):
        import experiment_e4s as E
        m = Path(tempfile.mkdtemp()) / "E4S_TEST_OPENED.json"
        calls = []
        r = E.pipeline(lambda: {"S_t08": res(0.01, -0.001, 300)}, E.select, lambda w: calls.append(w), m)
        self.assertTrue(r["verdict"].startswith("STOP"))
        self.assertFalse(m.exists())
        self.assertEqual(calls, [])

    def test_winner_opens_once(self):
        import experiment_e4s as E
        m = Path(tempfile.mkdtemp()) / "E4S_TEST_OPENED.json"
        calls = []
        tr = lambda: {"S_t08": res(0.01, 0.001, 300)}   # noqa: E731
        r = E.pipeline(tr, E.select, lambda w: calls.append(w) or {"ok": 1}, m)
        self.assertEqual(r["winner"], "S_t08")
        self.assertTrue(m.exists())
        with self.assertRaises(RuntimeError):
            E.pipeline(tr, E.select, lambda w: calls.append(w), m)
        self.assertEqual(calls, ["S_t08"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
