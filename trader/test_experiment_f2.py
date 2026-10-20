"""F2 runner -- tests written from trader-learning-loop-f-1020-spec.md "F2 runner -- контракт" BEFORE the code
(operator rule 2026-10-10: spec -> tests -> code). Synthetic grids, offline."""
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

import trading_policy as TP
from test_execution_replay import FakeGrid, cand

X4 = replace(TP.CHAMPION, name="X4_p1_floor8", exits=TP.Exits(kind="p1_trail", floor_override=0.08))


def up_then_down(n=400, ie=150, k=20):
    """Flat to ie, +0.5%/bar for k bars, then -1%/bar: X4 (8% floor) exits ~9 bars after the peak."""
    c = [100.0] * (ie + 1)
    for _ in range(k):
        c.append(c[-1] * 1.005)
    while len(c) < n:
        c.append(c[-1] * 0.99)
    return c


class StubExit:
    def __init__(self, bar):
        self.bar = bar

    def exit_bar(self, path):
        return self.bar


class TestConstants(unittest.TestCase):
    def test_windows_fixed_and_disjoint(self):
        import experiment_f2 as F2
        self.assertEqual(F2.TRAIN_TO, "2026-01-01")
        self.assertEqual((F2.VALID_FROM, F2.VALID_TO), ("2026-03-01", "2026-07-01"))
        self.assertEqual(F2.TEST_FROM, "2026-07-01")
        self.assertLessEqual(F2.TRAIN_TO, F2.VALID_FROM)
        self.assertEqual(F2.DELTAS, (0.0, 0.002, 0.005, 0.01))
        self.assertEqual((F2.N_RANDOM, F2.MAX_PATHS), (20, 30000))
        self.assertAlmostEqual(F2.ROUNDTRIP, 2 * (0.001 + 0.0008))


class TestPathFromGrid(unittest.TestCase):
    def test_path_and_x4_exit_match_exit_plan(self):
        import experiment_f2 as F2
        closes = up_then_down()
        g = FakeGrid(closes)
        c = cand(150, closes[150])
        p = F2.path_from_grid(g, c)
        plan = TP.exit_plan(X4, g, c, floor_of=lambda m: 0.0)
        self.assertEqual(p["x4_exit"], plan[0] - 150)
        np.testing.assert_allclose(p["close"], np.array(closes[150: 150 + 97]))
        self.assertAlmostEqual(p["cost"], F2.ROUNDTRIP)
        self.assertEqual(p["key"], f"{c['sym']}|{c['tf']}|{c['ts'].isoformat()}")

    def test_gap_before_the_x4_exit_gives_none(self):
        import experiment_f2 as F2
        closes = up_then_down()
        closes[160] = math.nan
        g = FakeGrid(closes)
        self.assertIsNone(F2.path_from_grid(g, cand(150, closes[150])))


class TestLearnedExitInE1(unittest.TestCase):
    def setUp(self):
        self.closes = up_then_down()
        self.g = FakeGrid(self.closes)
        self.c = cand(150, self.closes[150])

    def test_learned_exit_bar_and_price(self):
        import experiment_f2 as F2
        lid = F2.register(StubExit(3), "stub3")
        pol = replace(X4, exits=TP.Exits(kind="learned", floor_override=0.08, learned_id=lid))
        q, px, reason, _ = TP.exit_plan(pol, self.g, self.c, floor_of=lambda m: 0.0)
        self.assertEqual(q, 153)
        self.assertAlmostEqual(px, self.closes[153])
        self.assertEqual(reason, "learned")

    def test_never_later_than_x4(self):
        import experiment_f2 as F2
        lid = F2.register(StubExit(90), "stub90")
        pol = replace(X4, exits=TP.Exits(kind="learned", floor_override=0.08, learned_id=lid))
        x4 = TP.exit_plan(X4, self.g, self.c, floor_of=lambda m: 0.0)
        q, px, reason, _ = TP.exit_plan(pol, self.g, self.c, floor_of=lambda m: 0.0)
        self.assertEqual((q, px), (x4[0], x4[1]))
        self.assertEqual(reason, x4[2])                                 # X4's own exit, not "learned"

    def test_unknown_id_raises(self):
        pol = replace(X4, exits=TP.Exits(kind="learned", floor_override=0.08, learned_id="nope"))
        with self.assertRaises(KeyError):
            TP.exit_plan(pol, self.g, self.c, floor_of=lambda m: 0.0)

    def test_existing_epochs_unchanged(self):
        """The new field must not re-version running policies (values measured 2026-10-15)."""
        self.assertIsNone(TP.Exits().learned_id)
        self.assertEqual(TP.CHAMPION.epoch(), "e196b96e7004b46b")
        self.assertEqual(X4.epoch(), "0bccb4b2cf6d04f9")
        self.assertNotEqual(X4.epoch(), replace(X4, exits=TP.Exits(kind="learned", floor_override=0.08,
                                                                   learned_id="a")).epoch())


class TestRandomExitControl(unittest.TestCase):
    def paths(self, n=2000):
        return [{"key": f"S{i}USDT|15m|2026-07-01T00:{i % 60:02d}:00+00:00", "x4_exit": 40, "close": np.ones(97),
                 "cost": 0.0036} for i in range(n)]

    def test_share_and_bounds(self):
        import experiment_f2 as F2
        r = F2.RandomEarly(p=0.3, seed=1)
        bars = [r.exit_bar(p) for p in self.paths()]
        early = [b for b in bars if b < 40]
        self.assertAlmostEqual(len(early) / len(bars), 0.3, delta=0.04)
        self.assertTrue(all(1 <= b <= 39 for b in early))
        self.assertTrue(all(b == 40 for b in bars if b >= 40))

    def test_deterministic_by_seed(self):
        import experiment_f2 as F2
        ps = self.paths(300)
        a = [F2.RandomEarly(0.4, 7).exit_bar(p) for p in ps]
        self.assertEqual(a, [F2.RandomEarly(0.4, 7).exit_bar(p) for p in ps])
        self.assertNotEqual(a, [F2.RandomEarly(0.4, 8).exit_bar(p) for p in ps])


class TestSelectionAndVerdict(unittest.TestCase):
    def test_select_delta(self):
        import experiment_f2 as F2
        v = {0.0: {"paired": {"mean_pct": -0.1}}, 0.002: {"paired": {"mean_pct": 0.2}},
             0.005: {"paired": {"mean_pct": 0.35}}, 0.01: {"paired": {"mean_pct": None}}}
        self.assertEqual(F2.select_delta(v), 0.005)
        self.assertIsNone(F2.select_delta({0.0: {"paired": {"mean_pct": 0.0}}, 0.01: {"paired": {"mean_pct": -0.2}}}))

    def test_verdict(self):
        import experiment_f2 as F2
        self.assertEqual(F2.verdict(True, True, True), "accepted")
        self.assertEqual(F2.verdict(True, True, False), "shadow_candidate")
        self.assertEqual(F2.verdict(True, False, True), "rejected")
        self.assertEqual(F2.verdict(False, True, True), "rejected")

    def test_pipeline_stop_opens_nothing(self):
        import experiment_f2 as F2
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "F2_TEST_OPENED.json"
            calls = []
            out = F2.pipeline(train=lambda: calls.append("train") or "pol", validate=lambda pol: calls.append("valid") or {},
                              select=lambda v: None, test=lambda pol, delta: calls.append("test"), marker=m)
            self.assertEqual(calls, ["train", "valid"])
            self.assertFalse(m.exists())
            self.assertIn("STOP", out["verdict"])

    def test_pipeline_opens_once(self):
        import experiment_f2 as F2
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "F2_TEST_OPENED.json"
            got = []
            F2.pipeline(train=lambda: "pol", validate=lambda pol: {}, select=lambda v: 0.002,
                        test=lambda pol, delta: got.append(delta) or {"verdict": "rejected"}, marker=m)
            self.assertEqual(got, [0.002])
            self.assertTrue(m.exists())
            with self.assertRaises(RuntimeError):
                F2.pipeline(train=lambda: "pol", validate=lambda pol: {}, select=lambda v: 0.002,
                            test=lambda pol, delta: None, marker=m)


class TestSamplingAndSpeed(unittest.TestCase):
    def test_sample_paths_deterministic_order_free(self):
        import experiment_f2 as F2
        items = [{"key": f"K{i}"} for i in range(1000)]
        a = F2.sample_paths(items, 100)
        b = F2.sample_paths(list(reversed(items)), 100)
        self.assertEqual(len(a), 100)
        self.assertEqual(sorted(x["key"] for x in a), sorted(x["key"] for x in b))
        self.assertEqual(F2.sample_paths(items[:50], 100), sorted(items[:50], key=lambda x: x["key"]))

    def test_vectorised_exit_bars_equal_single(self):
        import exit_q as XQ
        from test_learning_f import peak_paths, tiny_model
        tr, te = peak_paths(150, 21), peak_paths(40, 22)
        pol = XQ.ExitPolicy(model_factory=tiny_model, delta=0.001).fit(tr)
        self.assertEqual(pol.exit_bars(te), [pol.exit_bar(p) for p in te])


if __name__ == "__main__":
    unittest.main(verbosity=2)
