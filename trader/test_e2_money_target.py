"""E2 money-target model -- tests written from e2-money-target-1010-spec.md §8 BEFORE the code
(operator rule 2026-10-10: spec -> tests -> code)."""
import random
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np

import execution_replay as R
import fill_model as FM
import trading_policy as TP
from test_execution_replay import FakeGrid, cand, T0, STEP

HERE = Path(__file__).resolve().parent


def _grid_with_ret_rank(closes):
    g = FakeGrid(closes)
    n = g.N
    opens = np.repeat(g.O[: n: 96], 96, axis=0)[:n]
    g.RET = g.C / opens - 1
    g.RANK = np.ones_like(g.C, dtype=int)
    return g


class TestLabel(unittest.TestCase):
    """§8.1: the label equals the E1 engine's trade result for the same candidate."""

    def test_label_equals_engine_trade(self):
        import experiment_e2 as E
        closes = [100] * 150 + [100 + i * 0.5 for i in range(30)] + [110 - i for i in range(40)] + [90] * 100
        g = _grid_with_ret_rank(closes)
        c = cand(149, 100.0, kind="blocked")
        fm = FM.model("taker", 1)
        y = E.label(g, c, fm, floor_of=lambda m: 0.0)
        pol = replace(TP.CHAMPION, admission=TP.Admission(kind="all_candidates"), exits=TP.Exits(kind="p1_trail"))
        r = R.run(pol, fm, g, [c], floor_of=lambda m: 0.0)
        self.assertEqual(len(r["trades"]), 1)
        self.assertAlmostEqual(y, r["trades"][0]["ret_pct"], places=9)


class TestNoLookAhead(unittest.TestCase):
    """§8.2."""

    def test_features_ignore_bars_after_ie(self):
        import experiment_e2 as E
        closes = [100 + (i % 7) for i in range(300)]
        g1, g2 = _grid_with_ret_rank(closes), _grid_with_ret_rank(closes[:150] + [500] * 150)
        c = cand(149, 100.0, kind="blocked", ml_proba=0.4)
        self.assertEqual(E.row_features(c, g1), E.row_features(c, g2))

    def test_label_ignores_bars_before_entry(self):
        import experiment_e2 as E
        closes = [100] * 300
        a = E.label(_grid_with_ret_rank(closes), cand(149, 100.0), FM.model("taker", 0), lambda m: 0.0)
        b = E.label(_grid_with_ret_rank([7] * 148 + closes[148:]), cand(149, 100.0), FM.model("taker", 0), lambda m: 0.0)
        self.assertEqual(a, b)


class TestEmbargo(unittest.TestCase):
    """§8.3 and §8.8."""

    def test_train_rows_end_one_day_before(self):
        import experiment_e2 as E
        rows = [{"ts": T0 + timedelta(hours=6 * i)} for i in range(200)]
        start = "2026-03-20"
        tr = E.embargo_cut(rows, start)
        self.assertTrue(tr)
        self.assertTrue(all(r["ts"].strftime("%Y-%m-%d") < "2026-03-19" for r in tr))

    def test_monthly_refit_plan(self):
        import experiment_e2 as E
        plan = E.monthly_refit_plan("2026-07-01", "2026-10-09")
        self.assertEqual([m for m, _ in plan], ["2026-07", "2026-08", "2026-09", "2026-10"])
        self.assertEqual(plan[0][1], "2026-07-01")


class TestSelectionAndStop(unittest.TestCase):
    """§8.4 and §8.5."""

    def test_select_on_validation_only(self):
        import experiment_e2 as E
        import inspect
        val = {"a": {"R": 0.05, "trades": 40}, "b": {"R": 0.10, "trades": 20}, "c": {"R": 0.08, "trades": 31}}
        self.assertEqual(E.select_on_validation(val), "c")       # b has too few trades
        self.assertEqual(list(inspect.signature(E.select_on_validation).parameters), ["val_results"])

    def test_stop_when_winner_not_positive(self):
        import experiment_e2 as E
        self.assertIsNone(E.select_on_validation({"a": {"R": -0.01, "trades": 100}, "b": {"R": 0.0, "trades": 100}}))

    def test_pipeline_never_touches_test_before_selection_and_stops(self):
        import experiment_e2 as E
        calls = []
        E.pipeline(validate=lambda: calls.append("validate") or {"a": {"R": -0.2, "trades": 100}},
                   select=lambda v: calls.append("select") or E.select_on_validation(v),
                   test=lambda w: calls.append("test"))
        self.assertEqual(calls, ["validate", "select"])           # STOP: test never called
        calls.clear()
        E.pipeline(validate=lambda: calls.append("validate") or {"a": {"R": 0.2, "trades": 100}},
                   select=lambda v: calls.append("select") or E.select_on_validation(v),
                   test=lambda w: calls.append("test"))
        self.assertEqual(calls, ["validate", "select", "test"])


class TestModels(unittest.TestCase):
    """§8.6 and §8.7."""

    def test_logreg_separable_and_l2(self):
        import experiment_e2 as E
        rnd = np.random.default_rng(1)
        X = rnd.normal(size=(600, 3))
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)
        m = E.LogReg(l2=0.01).fit(X, y)
        p = m.predict_proba(X)
        self.assertGreater(E.auc(y, p), 0.95)
        m2 = E.LogReg(l2=10.0).fit(X, y)
        self.assertLess(np.linalg.norm(m2.w), np.linalg.norm(m.w))

    def test_ev_formula(self):
        import experiment_e2 as E
        self.assertAlmostEqual(E.ev_from_proba(0.6, 2.0, 1.0), 0.6 * 2.0 - 0.4 * 1.0)


class TestBoundary(unittest.TestCase):
    """§8.9."""

    def test_stays_in_trader(self):
        src = (HERE / "experiment_e2.py").read_text(encoding="utf-8")
        self.assertNotIn('"' + '.runtime' + '"', src)
        self.assertIn("TC.RUNTIME", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
