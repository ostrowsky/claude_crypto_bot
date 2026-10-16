"""H2 selectivity family -- tests written from h2-selectivity-1010-spec.md §7 BEFORE the code
(operator rule 2026-10-10: spec -> tests -> code)."""
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import daily_revenue as DR
import execution_replay as R
import fill_model as FM
import trading_policy as TP
from test_execution_replay import FakeGrid, cand, T0, STEP

HERE = Path(__file__).resolve().parent


def far_exit(ie, bars=40):
    return dict(ix=ie + bars, exit_ts=T0 + (ie + bars + 1) * STEP, exit_price=100)


class TestEngineRules(unittest.TestCase):
    """§7.1–7.2: one position per symbol, cooldown after exit."""

    def test_one_per_symbol(self):
        g = FakeGrid([100] * 400)
        cs = [cand(150, 100, **far_exit(150)), cand(152, 100, kind="blocked")]
        pol = replace(TP.CHAMPION, admission=TP.Admission(kind="all_candidates"),
                      exits=TP.Exits(kind="p1_trail", max_bars=60),
                      risk=TP.Risk(max_open=10, one_per_symbol=True))
        r = R.run(pol, FM.model("taker", 0), g, cs)
        self.assertEqual(len(r["trades"]), 1)
        off = replace(pol, risk=TP.Risk(max_open=10, one_per_symbol=False))
        self.assertEqual(len(R.run(off, FM.model("taker", 0), g, cs)["trades"]), 2)

    def test_one_per_symbol_default_off(self):
        self.assertFalse(TP.Risk().one_per_symbol)

    def test_cooldown_after_exit(self):
        g = FakeGrid([100] * 400)
        pol = replace(TP.CHAMPION, admission=TP.Admission(kind="all_candidates"),
                      exits=TP.Exits(kind="p1_trail", max_bars=5),
                      risk=TP.Risk(max_open=10, one_per_symbol=True, cooldown_bars=19))
        # first exits at bar 155 (time exit after 5 bars); second at 160 is inside the cooldown, third at 180 after it
        cs = [cand(150, 100), cand(160, 100, kind="blocked"), cand(180, 100, kind="blocked")]
        r = R.run(pol, FM.model("taker", 0), g, cs)
        self.assertEqual(len(r["trades"]), 2)


class TestThresholdAndCap(unittest.TestCase):
    """§7.3–7.4."""

    def _rows(self):
        import experiment_h2 as H
        rows = []
        for d in range(10):
            day = T0 + timedelta(days=d)
            for i in range(10):
                rows.append({"ts": day + timedelta(hours=i), "ml_proba": (i + 1) / 10.0 + (100 if d >= 5 else 0),
                             "kind": "entry"})
        return H, rows

    def test_quantile_uses_train_only(self):
        H, rows = self._rows()
        split = (T0 + timedelta(days=5)).strftime("%Y-%m-%d")
        q = H.train_quantile(rows, "ml_proba", 0.5, split)
        self.assertLess(q, 1.0)                      # test rows (+100) would push it far above 1
        rows2 = [dict(r, ml_proba=r["ml_proba"] + 1000) if r["ts"].strftime("%Y-%m-%d") >= split else r for r in rows]
        self.assertEqual(q, H.train_quantile(rows2, "ml_proba", 0.5, split))

    def test_max_per_day_counts_in_time_order_and_resets(self):
        g = FakeGrid([100] * (96 * 4))
        cs = []
        for day in range(2):
            for i in range(5):
                ie = 120 + day * 96 + i * 3
                cs.append(cand(ie, 100, kind="blocked"))
        pol = replace(TP.CHAMPION, admission=TP.Admission(kind="all_candidates", max_per_day=3),
                      exits=TP.Exits(kind="p1_trail", max_bars=2), risk=TP.Risk(max_open=10))
        r = R.run(pol, FM.model("taker", 0), g, cs)
        days = [t["entry_ts"].strftime("%Y-%m-%d") for t in r["trades"]]
        self.assertEqual(sorted(set(days.count(d) for d in set(days))), [3])
        first = sorted(t["entry_ts"] for t in r["trades"])[:3]
        self.assertEqual(first, sorted(c["ts"] for c in cs)[:3])


class TestSelection(unittest.TestCase):
    """§7.5: the winner is chosen from train results only."""

    def test_select_uses_train_only(self):
        import experiment_h2 as H
        train = {"a": {"paired": {"mean_pct": 0.3, "lo95": 0.1}, "trades": 150},
                 "b": {"paired": {"mean_pct": 0.5, "lo95": 0.2}, "trades": 150},
                 "c": {"paired": {"mean_pct": 0.9, "lo95": -0.1}, "trades": 150},     # not admitted (lo <= 0)
                 "d": {"paired": {"mean_pct": 1.2, "lo95": 0.5}, "trades": 50}}       # too few trades
        self.assertEqual(H.select_winner(train), "b")
        import inspect
        self.assertEqual(list(inspect.signature(H.select_winner).parameters), ["train_results"])

    def test_stop_when_none_passes(self):
        import experiment_h2 as H
        self.assertIsNone(H.select_winner({"a": {"paired": {"mean_pct": 0.3, "lo95": -0.1}, "trades": 500}}))

    def test_grid_size(self):
        import experiment_h2 as H
        self.assertEqual(len(H.candidate_grid()), 54)


class TestAggregateAlpha(unittest.TestCase):
    """§7.6."""

    def test_wider_at_bonferroni(self):
        import random
        rnd = random.Random(1)
        s = [(f"2026-01-{d:02d}", rnd.gauss(0.1, 1.0)) for d in range(1, 29)]
        a, b = DR.aggregate(s, alpha=0.05), DR.aggregate(s, alpha=0.05 / 10)
        self.assertLess(b["lo95"], a["lo95"])
        self.assertGreater(b["hi95"], a["hi95"])
        self.assertEqual(a, DR.aggregate(s))           # default unchanged


class TestMoneyValidator(unittest.TestCase):
    """§7.7."""

    def _agg(self, lo, mean=0.1, dd=-5.0):
        return {"n_days": 100, "mean_pct": mean, "lo95": lo, "hi95": mean + 0.2, "max_dd_pct": dd}

    def test_improves(self):
        import money_validator as MV
        months = {"2026-07": 0.1, "2026-08": 0.2, "2026-09": -0.1, "2026-10": 0.05}
        ok = MV.improves(paired=self._agg(0.01), cand_dd=-10.0, base_dd=-8.0, month_diffs=months, n_trades=250)
        self.assertTrue(ok["verdict"])
        self.assertFalse(MV.improves(self._agg(-0.01), -10, -8, months, 250)["verdict"])        # CI spans 0
        self.assertFalse(MV.improves(self._agg(0.01), -20, -8, months, 250)["verdict"])         # DD 12 pp worse
        self.assertFalse(MV.improves(self._agg(0.01), -10, -8, months, 150)["verdict"])         # too few trades
        bad_m = {"2026-07": -0.1, "2026-08": -0.2, "2026-09": 0.1, "2026-10": -0.05}
        self.assertFalse(MV.improves(self._agg(0.01), -10, -8, bad_m, 250)["verdict"])          # months 25%

    def test_profitable_and_trade_ci(self):
        import money_validator as MV
        ci = MV.trade_ci([0.5] * 300 + [-0.1] * 100)
        self.assertGreater(ci["lo95"], 0)
        self.assertTrue(MV.profitable(day_agg={"mean_pct": 0.05}, trade_ci=ci)["verdict"])
        self.assertFalse(MV.profitable(day_agg={"mean_pct": -0.05}, trade_ci=ci)["verdict"])
        neg = MV.trade_ci([0.1, -0.2] * 200)
        self.assertFalse(MV.profitable(day_agg={"mean_pct": 0.05}, trade_ci=neg)["verdict"])

    def test_trials_ledger_append_only(self):
        import money_validator as MV
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "trials.jsonl"
            MV.log_trial({"family": "H2", "name": "a"}, p)
            MV.log_trial({"family": "H2", "name": "b"}, p)
            lines = p.read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(x)["name"] for x in lines], ["a", "b"])
            self.assertTrue(all("ts" in json.loads(x) for x in lines))


class TestBoundary(unittest.TestCase):
    """§7.8."""

    def test_h2_modules_stay_in_trader(self):
        for name in ("experiment_h2.py", "money_validator.py"):
            src = (HERE / name).read_text(encoding="utf-8")
            self.assertNotIn('"' + '.runtime' + '"', src)
            self.assertNotIn("sys.path.insert(0, str(TC.BOT_FILES", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestRandomControl(unittest.TestCase):
    """h2 spec §8 (reference control) -- written before the code."""

    def _cands(self, n):
        return [{"kind": "blocked", "sym": "S%d" % (i % 37), "ts": T0 + timedelta(minutes=15 * i)} for i in range(n)]

    def test_deterministic_and_fraction(self):
        pol = replace(TP.CHAMPION, admission=TP.Admission(kind="all_candidates", random_frac=0.3, random_seed=7))
        cs = self._cands(4000)
        a = [TP.decide_entry(pol, c, {"open": 0}) is not None for c in cs]
        b = [TP.decide_entry(pol, c, {"open": 0}) is not None for c in cs]
        self.assertEqual(a, b)
        self.assertAlmostEqual(sum(a) / len(a), 0.3, delta=0.03)
        other = replace(pol, admission=TP.Admission(kind="all_candidates", random_frac=0.3, random_seed=8))
        self.assertNotEqual(a, [TP.decide_entry(other, c, {"open": 0}) is not None for c in cs])

    def test_cap_applies_on_top(self):
        pol = replace(TP.CHAMPION, admission=TP.Admission(kind="all_candidates", random_frac=1.0, random_seed=1,
                                                          max_per_day=3))
        c = self._cands(1)[0]
        self.assertIsNone(TP.decide_entry(pol, c, {"open": 0, "day_count": 3}))
        self.assertIsNotNone(TP.decide_entry(pol, c, {"open": 0, "day_count": 2}))
