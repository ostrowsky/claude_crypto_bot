"""E1 replay engine (execution-replay-e1-1010-spec.md §6, §10). Synthetic grid, offline."""
import math
from pathlib import Path
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np

import daily_revenue as DR
import execution_replay as R
import fill_model as FM
import trading_policy as TP

T0 = datetime(2026, 3, 1, tzinfo=timezone.utc)
STEP = timedelta(minutes=15)


class FakeGrid:
    def __init__(self, closes, syms=("AUSDT",), atr=1.0):
        c = np.array(closes, dtype=float).reshape(len(closes), -1)
        self.syms = list(syms)
        self.col = {s: j for j, s in enumerate(self.syms)}
        self.t0 = T0
        self.N = c.shape[0]
        self.C, self.O = c, np.vstack([c[:1], c[:-1]])
        self.H, self.L = c * 1.001, c * 0.999
        self.V = np.full(c.shape, 1e6)
        self.ATR = np.full(c.shape, atr)

    def idx(self, dt):
        return int((dt - self.t0) / STEP)


def cand(ie, price, kind="entry", **kw):
    ts = T0 + (ie + 1) * STEP + timedelta(minutes=1)
    d = {"kind": kind, "sym": "AUSDT", "tf": "15m", "ts": ts, "price": price, "mode": "trend", "trail_k": 2.0,
         "j": 0, "ie": ie, "ix": None, "exit_ts": None, "exit_price": None}
    d.update(kw)
    return d


class TestFillModel(unittest.TestCase):
    def test_costs(self):
        fm = FM.FillModel(half_spread=0.001)
        self.assertAlmostEqual(fm.buy(100), 100.1)
        self.assertAlmostEqual(fm.sell(100), 99.9)
        self.assertIsNone(fm.tp_fill(110, 110.0))              # touch only: no fill
        self.assertEqual(fm.tp_fill(110, 110.5), 110)
        self.assertIsNone(fm.sl_fill(90, 95, 91))
        self.assertAlmostEqual(fm.sl_fill(90, 85, 84), 85 * 0.999)   # gap through the stop -> open
        self.assertEqual(FM.model("taker", 0).half_spread, 0.0)

    def test_provisional_levels_are_bps_scale(self):
        for lvl in (1, 2):
            hs, k = FM.SLIP[lvl]
            self.assertLess(hs, 0.003)
            self.assertEqual(k, 0.0)


class TestExitPlan(unittest.TestCase):
    def test_sl_before_tp_same_bar(self):
        g = FakeGrid([100] * 200)
        g.H[151, 0], g.L[151, 0] = 200, 50                    # both touched in one bar
        pol = replace(TP.CHAMPION, exits=TP.Exits(kind="oco_atr", tp_atr=2, sl_atr=1.5))
        q, px, reason, _ = TP.exit_plan(pol, g, cand(150, 100))
        self.assertEqual((q, reason), (151, "sl"))

    def test_no_look_ahead(self):
        """V4: nothing after the exit bar and nothing before the entry bar changes the plan."""
        closes = [100] * 150 + [101, 102, 103, 95] + [100] * 50
        pol = replace(TP.CHAMPION, exits=TP.Exits(kind="p1_trail"))
        a = TP.exit_plan(pol, FakeGrid(closes), cand(149, 100))
        later = closes[:154] + [1000] * 50                    # change only bars after the exit
        b = TP.exit_plan(pol, FakeGrid(later), cand(149, 100))
        earlier = [5] * 100 + closes[100:]                    # change only bars before the entry bar
        c = TP.exit_plan(pol, FakeGrid(earlier), cand(149, 100))
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(a[2], "trail")

    def test_hourly_positions_judged_on_hour_closes(self):
        closes = [100] * 150 + [80] + [100] * 60
        pol = replace(TP.CHAMPION, exits=TP.Exits(kind="p1_trail"))
        q15 = TP.exit_plan(pol, FakeGrid(closes), cand(149, 100))[0]
        q1h = TP.exit_plan(pol, FakeGrid(closes), cand(149, 100, tf="1h"))[0]
        self.assertEqual(q15, 150)
        self.assertNotEqual(q1h, 150)


class TestEngine(unittest.TestCase):
    def test_conservation_and_daily_series(self):
        closes = [100 + math.sin(i / 7) * 3 for i in range(96 * 6)]
        g = FakeGrid(closes)
        cs = []
        for k, ie in enumerate(range(120, 96 * 5, 37)):
            ix = ie + 10
            cs.append(cand(ie, closes[ie], ix=ix, exit_ts=T0 + (ix + 1) * STEP + timedelta(minutes=1),
                           exit_price=closes[ix]))
        r = R.run(TP.CHAMPION, FM.model("taker", 1), g, cs)
        self.assertTrue(r["v3_conservation_ok"])
        self.assertEqual(len(r["trades"]), len(cs))
        self.assertTrue(all(isinstance(v, float) for _, v in r["series"]))

    def test_slots_and_daily_loss_limit(self):
        g = FakeGrid([100] * 400)
        far = lambda ie: dict(ix=ie + 200, exit_ts=T0 + (ie + 201) * STEP, exit_price=100)  # noqa: E731
        cs = [cand(120 + i, 100, **far(120 + i)) for i in range(15)]
        r = R.run(TP.CHAMPION, FM.model("taker", 0), g, cs)
        self.assertEqual(len(r["trades"]), 10)                # 11th+ skipped while 10 are open
        pol = replace(TP.CHAMPION, risk=TP.Risk(max_open=10, daily_loss_limit_pct=0.0001))
        r2 = R.run(pol, FM.model("taker", 2), g, cs)
        self.assertLess(len(r2["trades"]), 10)               # fees+costs put the day below the limit

    def test_cash_policy_is_zero(self):
        g = FakeGrid([100] * 400)
        cs = [cand(150, 100, ix=160, exit_ts=T0 + 161 * STEP, exit_price=110)]
        r = R.run(R.POLICIES["cash"], FM.model("taker", 1), g, cs)
        self.assertEqual(r["trades"], [])


class TestAggregate(unittest.TestCase):
    def test_aggregate_and_paired(self):
        s = [(f"2026-01-{d:02d}", 1.0) for d in range(1, 21)]
        a = DR.aggregate(s)
        self.assertAlmostEqual(a["mean_pct"], 1.0)
        self.assertEqual(DR.verdict_text(a), "выше нуля")
        z = DR.aggregate([(f"2026-01-{d:02d}", (-1) ** d * 1.0) for d in range(1, 21)])
        self.assertEqual(DR.verdict_text(z), "рано судить")
        p = DR.paired(s, [(d, 0.5) for d, _ in s[:10]])
        self.assertEqual(p["n_days"], 10)
        self.assertAlmostEqual(p["mean_pct"], 0.5)

    def test_policy_epoch_changes_with_params(self):
        self.assertNotEqual(TP.CHAMPION.epoch(), replace(TP.CHAMPION, risk=TP.Risk(max_open=5)).epoch())


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestV1SameData(unittest.TestCase):
    """V1 must compare the two implementations on the SAME events, not against a frozen JSON
    (the live bot keeps appending to bot_events.jsonl: 5 753 -> 5 757 trades broke a stale V1)."""

    def _events(self, path):
        import json as _j
        rows = []
        t = T0
        for i in range(30):
            sym = "S%d" % (i % 4)
            rows.append({"event": "entry", "sym": sym, "tf": "15m", "ts": (t + timedelta(hours=3 * i)).isoformat(), "price": 100})
            rows.append({"event": "exit", "sym": sym, "tf": "15m", "ts": (t + timedelta(hours=3 * i + 2)).isoformat(),
                         "pnl_pct": (-1) ** i * 0.7, "exit_price": 100.7})
        path.write_text("\n".join(_j.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def test_baseline_compute_matches_realized_simple(self):
        import tempfile
        from pathlib import Path as _P
        import _backtest_daily_revenue_baseline as B
        with tempfile.TemporaryDirectory() as d:
            p = _P(d) / "ev.jsonl"
            self._events(p)
            ref = B.compute(events_path=p, slots=10, since="2026-03-01")
            trades, _ = R.load_events(since="2026-03-01", path=p)
            mine = R.realized_simple(trades, 10, 0.20)
            self.assertEqual(mine["taken"], ref["taken"])
            self.assertAlmostEqual(mine["agg"]["mean_pct"], ref["fees"]["taker_0.20rt"]["mean_day_pct"], places=4)

    def test_verify_uses_fresh_baseline(self):
        src = (Path(R.__file__)).read_text(encoding="utf-8")
        self.assertIn("B.compute(", src)
        self.assertNotIn('json.loads((TC.RUNTIME / "daily_revenue_baseline.json")', src)
