"""E8 positioning filter in money -- tests written from e8-positioning-money-1012-spec.md §6 BEFORE the code
(operator rule 2026-10-10: spec -> tests -> code)."""
import inspect
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = datetime(2026, 9, 1, tzinfo=timezone.utc)


def write_pos(path, rows, tail=""):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + tail, encoding="utf-8")


class TestAsOfJoin(unittest.TestCase):
    """§6.1."""

    def test_latest_not_after_and_not_stale(self):
        import experiment_e8 as E
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pos.jsonl"
            write_pos(p, [{"ts": (T + timedelta(hours=h)).isoformat(), "sym": "A", "oi_4h": h} for h in (0, 2, 4)])
            idx = E.load_positioning(p)
            self.assertEqual(E.asof(idx, "A", T + timedelta(hours=3))["oi_4h"], 2)        # latest <= entry
            self.assertEqual(E.asof(idx, "A", T + timedelta(hours=4))["oi_4h"], 4)        # equal time allowed
            self.assertIsNone(E.asof(idx, "A", T - timedelta(minutes=1)))                  # nothing before
            self.assertIsNone(E.asof(idx, "A", T + timedelta(hours=7)))                    # 3 h old > 2.5 h
            self.assertIsNone(E.asof(idx, "B", T + timedelta(hours=1)))                    # other coin


class TestRobustRead(unittest.TestCase):
    """§6.2."""

    def test_broken_and_partial_lines_skipped(self):
        import experiment_e8 as E
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pos.jsonl"
            write_pos(p, [{"ts": T.isoformat(), "sym": "A", "oi_4h": 1}], tail='{broken\n{"ts": "2026-09-0')
            idx = E.load_positioning(p)
            self.assertEqual(len(idx["A"]), 1)


class TestThresholdsAndRules(unittest.TestCase):
    """§6.3–6.4."""

    def test_quantile_train_only(self):
        import experiment_e8 as E
        rows = [{"ts": T + timedelta(days=i), "pos": {"oi_4h": float(i)}} for i in range(20)]
        split = (T + timedelta(days=10)).strftime("%Y-%m-%d")
        q = E.train_threshold(rows, "oi_4h", 0.5, split)
        rows2 = [dict(r, pos={"oi_4h": 999.0}) if r["ts"].strftime("%Y-%m-%d") >= split else r for r in rows]
        self.assertEqual(q, E.train_threshold(rows2, "oi_4h", 0.5, split))
        self.assertLess(q, 10)

    def test_directions(self):
        import experiment_e8 as E
        self.assertTrue(E.passes({"oi_4h": 2.0}, ("oi_4h", "above"), 2.0))
        self.assertFalse(E.passes({"oi_4h": 1.9}, ("oi_4h", "above"), 2.0))
        self.assertTrue(E.passes({"retail": 1.0}, ("retail", "below"), 1.0))
        self.assertFalse(E.passes({"retail": 1.1}, ("retail", "below"), 1.0))
        self.assertTrue(E.passes({"flow": "longs_opening"}, ("flow", "eq"), "longs_opening"))
        self.assertFalse(E.passes({"flow": "flat"}, ("flow", "eq"), "longs_opening"))
        self.assertFalse(E.passes({}, ("oi_4h", "above"), 0.0))                           # missing -> no

    def test_rule_grid(self):
        import experiment_e8 as E
        self.assertEqual(len(E.RULES), 13)
        self.assertEqual(sum(1 for r in E.RULES if r["direction"] == "below"), 4)


class TestRandomControl(unittest.TestCase):
    """§6.5."""

    def test_percentile(self):
        import experiment_e8 as E
        ctrl = list(range(50))
        self.assertAlmostEqual(E.random_percentile(47.5, ctrl), 0.96)
        self.assertTrue(E.beats_random(47.5, ctrl))
        self.assertFalse(E.beats_random(40.0, ctrl))

    def test_seeds_and_frac(self):
        import experiment_e8 as E
        pols = E.random_policies(frac=0.3)
        self.assertEqual(len(pols), 50)
        self.assertEqual({p.admission.random_frac for p in pols}, {0.3})
        self.assertEqual(len({p.admission.random_seed for p in pols}), 50)


class TestSelection(unittest.TestCase):
    """§6.6."""

    def test_select_train_only_and_stop(self):
        import experiment_e8 as E
        tr = {"r1": {"R": -0.1, "trades": 80, "beats_random_train": True},
              "r2": {"R": 0.2, "trades": 40, "beats_random_train": True},     # too few trades
              "r3": {"R": 0.1, "trades": 90, "beats_random_train": False},    # not better than random
              "r4": {"R": 0.05, "trades": 70, "beats_random_train": True}}
        self.assertEqual(E.select_winner(tr), "r4")
        self.assertEqual(list(inspect.signature(E.select_winner).parameters), ["train_results"])
        calls = []
        E.pipeline(validate=lambda: calls.append("train") or {}, select=lambda v: calls.append("select") or None,
                   test=lambda w: calls.append("test"))
        self.assertEqual(calls, ["train", "select"])


class TestNoData(unittest.TestCase):
    """§6.7."""

    def test_entries_without_snapshot_are_counted_and_excluded(self):
        import experiment_e8 as E
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pos.jsonl"
            write_pos(p, [{"ts": T.isoformat(), "sym": "A", "oi_4h": 1.0}])
            idx = E.load_positioning(p)
            cands = [{"sym": "A", "ts": T + timedelta(hours=1)}, {"sym": "A", "ts": T + timedelta(hours=5)},
                     {"sym": "B", "ts": T + timedelta(hours=1)}]
            kept, missing = E.attach_positioning(cands, idx)
            self.assertEqual(len(kept), 1)
            self.assertEqual(missing, 2)
            self.assertEqual(kept[0]["pos"]["oi_4h"], 1.0)


class TestBoundary(unittest.TestCase):
    """§6.8."""

    def test_read_only_and_runtime(self):
        src = (HERE / "experiment_e8.py").read_text(encoding="utf-8")
        self.assertNotIn('"' + '.runtime' + '"', src)
        self.assertIn("TC.RUNTIME", src)
        self.assertNotIn('open(path, "w"', src)
        self.assertNotIn('open(TC.POSITIONING, "w"', src)
        import trader_config as TC
        self.assertEqual(TC.POSITIONING.name, "positioning_history.jsonl")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestAucReference(unittest.TestCase):
    """Spec §5 reference (added 2026-10-13, before the 10-18 run): AUC of the rule's feature vs y > 0,
    y = net result of the X4 trade the replay would make. Train: every rule; test: the winner only."""

    def test_trade_label_equals_engine_x4_trade(self):
        import experiment_e8 as E
        import execution_replay as R
        import fill_model as FM
        from test_execution_replay import FakeGrid, cand
        closes = [100] * 150 + [100 + 0.4 * i for i in range(40)] + [116 - 0.9 * i for i in range(60)] + [60] * 150
        g = FakeGrid(closes)
        c = cand(149, 100.0)
        fm = FM.model("taker", 1)
        y = E.trade_label(g, c, fm, floor_of=lambda m: 0.0)
        r = R.run(E.X4, fm, g, [c], floor_of=lambda m: 0.0)
        self.assertEqual(len(r["trades"]), 1)
        self.assertAlmostEqual(y, r["trades"][0]["ret_pct"], places=9)

    def test_feature_auc_directions(self):
        import experiment_e8 as E
        rows = [{"pos": {"oi_4h": float(i), "retail": float(-i), "flow": "longs_opening" if i >= 5 else "flat"},
                 "y": 1.0 if i >= 5 else -1.0} for i in range(10)]
        self.assertAlmostEqual(E.feature_auc(rows, "oi_4h", "above")["auc"], 1.0)
        self.assertAlmostEqual(E.feature_auc(rows, "retail", "below")["auc"], 1.0)      # low retail = positive
        self.assertAlmostEqual(E.feature_auc(rows, "flow", "eq")["auc"], 1.0)
        self.assertAlmostEqual(E.feature_auc(rows, "oi_4h", "above")["base_rate"], 0.5)
        self.assertIsNone(E.feature_auc([], "oi_4h", "above")["auc"])

    def test_auc_report_train_all_test_winner_only(self):
        import experiment_e8 as E
        mk = lambda d, i: {"ts": T + timedelta(days=d), "y": 1.0 if i % 2 else -1.0,  # noqa: E731
                           "pos": {"oi_4h": float(i), "taker": float(i), "taker_trend": 1.0, "top": 1.0, "retail": 1.0,
                                   "funding_bp": 1.0, "flow": "flat"}}
        rows = [mk(d, i) for d in range(20) for i in range(4)]
        split = (T + timedelta(days=10)).strftime("%Y-%m-%d")
        rep = E.auc_report(rows, split, winner=None)
        self.assertEqual(set(rep["train"]), {r["name"] for r in E.RULES})
        self.assertEqual(rep["test"], {})
        rep2 = E.auc_report(rows, split, winner="oi_4h_above_q0.5")
        self.assertEqual(set(rep2["test"]), {"oi_4h_above_q0.5"})
