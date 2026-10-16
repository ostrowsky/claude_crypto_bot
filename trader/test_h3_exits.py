"""H3 exits in money -- tests written from h3-exits-money-1011-spec.md §6 BEFORE the code
(operator rule 2026-10-10: spec -> tests -> code)."""
import inspect
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np

import trader_config as TC
import trading_policy as TP
from test_execution_replay import FakeGrid, cand, T0, STEP

HERE = Path(__file__).resolve().parent
TC.bot_readonly()

RSI_REASON = "WEAK: RSI дивергенция (цена выше, RSI ниже)"
LEADER = {"rank_max": 5, "min_ret": 0.03, "floor": 0.08, "lost_rank": 10, "lost_min_bars": 4, "max_hold_bars": 672}


def grid_rr(closes, rank=None):
    g = FakeGrid(closes)
    n = g.N
    opens = np.repeat(g.O[: n: 96], 96, axis=0)[:n]
    g.RET = g.C / opens - 1
    g.RANK = np.full(g.C.shape, 50, dtype=int) if rank is None else np.asarray(rank, dtype=int).reshape(-1, 1)
    return g


def pol(kind, **kw):
    return replace(TP.CHAMPION, exits=TP.Exits(kind=kind, **kw))


def real_cand(ie, ix, price, reason, closes, tf="15m"):
    return cand(ie, price, ix=ix, exit_ts=T0 + (ix + 1) * STEP, exit_price=closes[ix], exit_reason=reason, tf=tf,
                trail_k=2.0)


class TestRsiHold(unittest.TestCase):
    """§6.1."""

    def _closes(self):
        # entry at 100, rises to ~103 by the RSI exit at bar 160, keeps rising, then falls through the trail
        return [100.0] * 150 + [100 + 0.3 * i for i in range(1, 31)] + [109 - 1.5 * i for i in range(30)] + [60.0] * 140

    def test_matches_bot_hold_on_trail(self):
        import _backtest_weak_exits as W
        closes = self._closes()
        g = grid_rr(closes)
        c = real_cand(149, 160, 100.0, RSI_REASON, closes)
        q, px, reason, _ = TP.exit_plan(pol("rsi_hold"), g, c, lambda m: 0.0)
        ref = W.hold_on_trail(g, {"j": 0, "ie": 149, "ix": 160, "k": 2.0, "ep": 100.0}, 0.0)
        self.assertEqual(reason, "rsi_hold")
        self.assertAlmostEqual((px / 100.0 - 1) * 100, ref, places=9)
        self.assertGreaterEqual(q, 160)
        self.assertLessEqual(q, 160 + 96)

    def test_only_rsi_and_only_in_profit(self):
        closes = self._closes()
        g = grid_rr(closes)
        other = real_cand(149, 160, 100.0, "WEAK: EMA веер сжимается", closes)
        self.assertEqual(TP.exit_plan(pol("rsi_hold"), g, other, lambda m: 0.0)[2], "real")
        flat = [100.0] * 400
        loss = real_cand(149, 160, 100.0, RSI_REASON, flat)          # profit 0 < 0.5%
        self.assertEqual(TP.exit_plan(pol("rsi_hold"), grid_rr(flat), loss, lambda m: 0.0)[2], "real")


class TestLeader(unittest.TestCase):
    """§6.2."""

    def test_matches_bot_wide_from(self):
        import exit_validator as EV
        closes = [100.0] * 150 + [100 + 0.4 * i for i in range(1, 41)] + [116 - 0.8 * i for i in range(60)] + [60.0] * 150
        rank = [50] * 155 + [2] * 40 + [20] * 205
        g = grid_rr(closes, rank)
        c = real_cand(149, 180, 100.0, "WEAK: EMA веер сжимается", closes)
        q, px, reason, _ = TP.exit_plan(pol("leader", leader=tuple(sorted(LEADER.items()))), g, c, lambda m: 0.0)
        self.assertEqual(reason, "leader")
        mask = (g.RANK[:, 0] <= LEADER["rank_max"]) & (g.RET[:, 0] >= LEADER["min_ret"])
        s = 149 + int(np.where(mask[149:180])[0][0])
        ref = EV.wide_from(g, {"j": 0, "ep": 100.0, "k": 2.0}, s, LEADER)
        self.assertAlmostEqual((px / 100.0 - 1) * 100, ref, places=9)

    def test_no_switch_keeps_real_exit(self):
        closes = [100.0] * 400
        g = grid_rr(closes)
        c = real_cand(149, 170, 100.0, "trail", closes)
        q, px, reason, _ = TP.exit_plan(pol("leader", leader=tuple(sorted(LEADER.items()))), g, c, lambda m: 0.0)
        self.assertEqual((q, reason), (170, "real"))


class TestCombined(unittest.TestCase):
    """§6.3."""

    def test_leader_first_else_rsi_hold(self):
        closes = [100.0] * 150 + [100 + 0.3 * i for i in range(1, 31)] + [109 - 1.5 * i for i in range(30)] + [60.0] * 140
        g = grid_rr(closes)                                           # rank 50 everywhere: never a leader
        c = real_cand(149, 160, 100.0, RSI_REASON, closes)
        p = pol("leader_rsi_hold", leader=tuple(sorted(LEADER.items())))
        self.assertEqual(TP.exit_plan(p, g, c, lambda m: 0.0)[2], "rsi_hold")
        g2 = grid_rr(closes, [50] * 155 + [1] * 205)
        self.assertEqual(TP.exit_plan(p, g2, c, lambda m: 0.0)[2], "leader")


class TestFloor8(unittest.TestCase):
    """§6.4."""

    def test_two_percent_dip_does_not_stop(self):
        closes = [100.0] * 150 + [101, 102, 103, 101, 104, 106] + [106.0] * 200
        c = cand(149, 100.0, trail_k=2.0)
        q, px, reason, dist = TP.exit_plan(pol("p1_trail", floor_override=0.08), grid_rr(closes), c, lambda m: 0.0)
        self.assertEqual(reason, "time")                               # never stopped out
        self.assertGreaterEqual(dist, 0.08 - 1e-12)


class TestHourlyAndCache(unittest.TestCase):
    """§6.5–6.6."""

    def test_1h_keeps_real_exit(self):
        closes = [100.0] * 150 + [100 + 0.3 * i for i in range(1, 31)] + [100.0] * 220
        g = grid_rr(closes, [1] * 400)
        c = real_cand(149, 160, 100.0, RSI_REASON, closes, tf="1h")
        for kind in ("rsi_hold", "leader", "leader_rsi_hold"):
            plan = TP.exit_plan(pol(kind, leader=tuple(sorted(LEADER.items()))), g, c, lambda m: 0.0)
            self.assertEqual(plan[2], "real", kind)

    def test_exits_hashable_and_distinct(self):
        lp = tuple(sorted(LEADER.items()))
        es = {TP.Exits(kind="rsi_hold"), TP.Exits(kind="leader", leader=lp), TP.Exits(kind="leader_rsi_hold", leader=lp),
              TP.Exits(kind="p1_trail", floor_override=0.08), TP.Exits(kind="real")}
        self.assertEqual(len(es), 5)


class TestSelection(unittest.TestCase):
    """§6.7."""

    def test_select_and_stop(self):
        import experiment_h3 as H
        tr = {"X1": {"paired": {"mean_pct": 0.2, "lo95": 0.05}, "changed": 150},
              "X2": {"paired": {"mean_pct": 0.4, "lo95": 0.10}, "changed": 90},       # too few changed
              "X3": {"paired": {"mean_pct": 0.3, "lo95": 0.01}, "changed": 300}}
        self.assertEqual(H.select_winner(tr), "X3")
        self.assertEqual(list(inspect.signature(H.select_winner).parameters), ["train_results"])
        self.assertIsNone(H.select_winner({"X1": {"paired": {"mean_pct": 0.2, "lo95": -0.01}, "changed": 500}}))

    def test_validate_output_holds_only_candidates(self):
        """Bug 2026-10-11: an auxiliary '_X0' entry reached select_winner -> KeyError 'changed'."""
        import experiment_h3 as H
        tr = {"X1": {"paired": {"mean_pct": 0.2, "lo95": 0.05}, "changed": 150}}
        res = H.split_train_results(dict(tr, _X0={"net_per_trade": -0.4}))
        self.assertEqual(set(res["candidates"]), {"X1"})
        self.assertEqual(H.select_winner(res["candidates"]), "X1")

    def test_test_not_opened_on_stop(self):
        import experiment_h3 as H
        calls = []
        H.pipeline(validate=lambda: calls.append("train") or {},
                   select=lambda v: calls.append("select") or None, test=lambda w: calls.append("test"))
        self.assertEqual(calls, ["train", "select"])


class TestBoundary(unittest.TestCase):
    """§6.8."""

    def test_stays_in_trader(self):
        src = (HERE / "experiment_h3.py").read_text(encoding="utf-8")
        self.assertNotIn('"' + '.runtime' + '"', src)
        self.assertIn("TC.RUNTIME", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestThinningControl(unittest.TestCase):
    """h3 spec §7 (reference control) -- written before the code."""

    def test_match_frac_hits_target(self):
        import experiment_h3 as H
        count = lambda f: int(round(2400 * f))            # monotone synthetic: trades as a function of frac
        f = H.match_frac(count, 865)
        self.assertLessEqual(abs(count(f) - 865), 0.05 * 865)
        self.assertTrue(0 < f < 1)
