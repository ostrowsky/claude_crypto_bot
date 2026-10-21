"""RSI-divergence hold in SHADOW + readout WEAK-HOLD-SHADOW (weak-exit-shadow-1020-spec.md).

Written from the spec before the code (operator rule: spec -> tests -> code).
"""
import json
import re
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import botlog
import config
import exit_validator as EV
import monitor as M
import readouts as RO
import _backtest_weak_exits as BW

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UTC = timezone.utc


def _strategy_rsi_div_reason() -> str:
    src = (HERE / "strategy.py").read_text(encoding="utf-8")
    m = re.search(r'return "(⚠️ WEAK: RSI дивергенция[^"]*)"', src)
    assert m, "RSI-divergence exit string not found in strategy.py"
    return m.group(1)


RSI_DIV = _strategy_rsi_div_reason()
OTHER_REASONS = [
    "⚠️ WEAK: объёмное истощение — покупатели заканчиваются",
    "⚠️ WEAK: веер EMA сжался",
    "RSI перекуплен (82.1)",
    "ATR-трейл пробит (стоп 1.23)",
    "2 закрытия подряд ниже EMA20",
    "EMA20 разворачивается вниз (slope -0.12%)",
    "режим лидера: широкий трейл пробит (стоп 1.0)",
]


# ---------------------------------------------------------------- 1-4: config and helper

class TestConfig(unittest.TestCase):
    def test_flags(self):
        self.assertTrue(hasattr(config, "RSI_DIV_HOLD_SHADOW_ENABLED"))
        self.assertTrue(config.RSI_DIV_HOLD_SHADOW_ENABLED)      # switched on by the operator 2026-10-21
        self.assertEqual(config.RSI_DIV_HOLD_MIN_PNL_PCT, 0.5)

    def test_no_unwired_live_flag(self):
        # a live flag the code never reads would be a false claim (CLAUDE.md 0a rule 13)
        self.assertFalse(hasattr(config, "RSI_DIV_HOLD_ENABLED"))


class TestHelper(unittest.TestCase):
    def test_threshold_boundary(self):
        self.assertIs(M._rsi_div_hold_decision(RSI_DIV, 0.50, "15m"), True)
        self.assertIs(M._rsi_div_hold_decision(RSI_DIV, 0.49, "15m"), False)
        self.assertIs(M._rsi_div_hold_decision(RSI_DIV, -1.0, "15m"), False)

    def test_out_of_scope(self):
        self.assertIsNone(M._rsi_div_hold_decision(RSI_DIV, 2.0, "1h"))
        self.assertIsNone(M._rsi_div_hold_decision(None, 2.0, "15m"))
        self.assertIsNone(M._rsi_div_hold_decision("", 2.0, "15m"))
        for r in OTHER_REASONS:
            self.assertIsNone(M._rsi_div_hold_decision(r, 2.0, "15m"), r)

    def test_fixed_threshold_not_h5_per_mode(self):
        with mock.patch.object(config, "H5_BREAK_EVEN_PCT_IMPULSE_SPEED", 0.3, create=True), \
             mock.patch.object(config, "H5_BREAK_EVEN_PCT_STRONG_TREND", 0.7, create=True), \
             mock.patch.object(config, "H5_BREAK_EVEN_PCT", 0.9, create=True):
            self.assertIs(M._rsi_div_hold_decision(RSI_DIV, 0.4, "15m"), False)
            self.assertIs(M._rsi_div_hold_decision(RSI_DIV, 0.6, "15m"), True)
        self.assertNotIn("mode", M._rsi_div_hold_decision.__code__.co_varnames)

    def test_agrees_with_replay_category(self):
        for r in [RSI_DIV, "WEAK: RSI divergence"] + OTHER_REASONS:
            self.assertEqual(M._rsi_div_hold_decision(r, 1.0, "15m") is not None,
                             BW.category(r) == "rsi_divergence", r)


# ---------------------------------------------------------------- 5-7: wiring and event

class TestWiring(unittest.TestCase):
    src = (HERE / "monitor.py").read_text(encoding="utf-8")

    def test_hook_inside_exit_block_before_send(self):
        s = self.src
        h5 = s.index("H5 SHADOW would-suppress")
        blk = s.index("\n    if reason:\n", h5)
        hook = s.index("_rsi_div_hold_decision(reason", blk)
        send = s.index("await send(", blk)
        self.assertLess(hook, send)
        start = s.index('getattr(config, "RSI_DIV_HOLD_SHADOW_ENABLED"', blk)
        self.assertLess(start, hook)
        body = s[start:s.index("pnl_icon", start)]
        self.assertIn("log_weak_exit_shadow", body)
        self.assertNotIn("reason = None", body)
        self.assertNotIn("return", body)
        self.assertIsNone(re.search(r"\bpos\.\w+\s*=[^=]", body))

    def test_leader_mode_returns_before_exit_conditions(self):
        s = self.src
        step = s.index("leader_exit.step(")
        hold = s.index('if _lx_action == "hold":', step)
        ret = s.index("return", hold)
        chk = s.index("reason = check_exit_conditions(", step)
        self.assertLess(step, hold)
        self.assertLess(ret, chk)


class TestEvent(unittest.TestCase):
    def test_event_fields(self):
        got = []
        with mock.patch.object(botlog, "_write", got.append):
            botlog.log_weak_exit_shadow("AUSDT", "15m", "trend", 1760000000000, 1.234, 0.73, True,
                                        0.5, 1.20, 2.0, False, RSI_DIV)
        e = got[0]
        self.assertEqual(e["event"], "weak_exit_shadow")
        for k in ("sym", "tf", "mode", "bar_ts", "price", "pnl_pct", "would_hold", "threshold",
                  "trail_stop", "trail_k", "leader_mode", "reason"):
            self.assertIn(k, e)
        self.assertIs(e["would_hold"], True)
        self.assertIs(e["leader_mode"], False)


# ---------------------------------------------------------------- 8: readout on a synthetic grid

def _grid(closes: np.ndarray, rank=None, ret=None, atr=1.0):
    n = len(closes)
    C = closes.reshape(n, 1).astype(float)
    R = np.full((n, 1), 999, dtype=int) if rank is None else rank.reshape(n, 1)
    T = np.zeros((n, 1)) if ret is None else ret.reshape(n, 1)
    g = SimpleNamespace(C=C, ATR=np.full((n, 1), float(atr)), RANK=R, RET=T, N=n, col={"AUSDT": 0})
    g.lead = lambda rank_max, min_ret: (g.RANK <= rank_max) & (g.RET >= min_ret)
    return g


def _trade(ie=100, ix=110, reason=RSI_DIV, day="2026-10-10", ep=100.0, k=2.0):
    return {"sym": "AUSDT", "j": 0, "ie": ie, "ix": ix, "ep": ep, "k": k, "mode": "trend",
            "day": day, "reason": reason, "real": 0.0}


P = {"rank_max": 5, "min_ret": 0.03, "floor": 0.08, "lost_rank": 10, "lost_min_bars": 4, "max_hold_bars": 672}


class TestArm(unittest.TestCase):
    def test_trail_runs_up_then_exits(self):
        c = np.full(400, 100.0)
        c[110] = 101.0
        c[111:131] = np.linspace(102, 110, 20)
        c[131:] = 105.0
        y, st = RO.weak_hold_arm(_grid(c), _trade(), 0.0, None)
        self.assertEqual(st, "trail")
        self.assertAlmostEqual(y, 5.0, places=6)
        self.assertAlmostEqual(y, BW.hold_on_trail(_grid(c), _trade(), 0.0), places=9)

    def test_exit_at_the_signal_bar_when_already_below_stop(self):
        c = np.full(400, 100.0)
        c[105] = 110.0
        c[110] = 101.0                       # stop 108 > close -> exits at ix, no change
        y, st = RO.weak_hold_arm(_grid(c), _trade(), 0.0, None)
        self.assertAlmostEqual(y, 1.0, places=6)

    def test_pending_when_hold_unresolved(self):
        c = np.linspace(100, 130, 150)       # still rising at the end of data, < 96 bars after ix
        y, st = RO.weak_hold_arm(_grid(c), _trade(), 0.0, None)
        self.assertIsNone(y)
        self.assertEqual(st, "pending")

    def test_full_96_bars_is_resolved(self):
        c = np.linspace(100, 130, 400)
        y, st = RO.weak_hold_arm(_grid(c), _trade(), 0.0, None)
        self.assertEqual(st, "max_bars")
        self.assertAlmostEqual(y, (c[110 + 96] / 100 - 1) * 100, places=6)

    def test_leader_switch_after_signal_bar(self):
        c = np.full(1200, 100.0)
        c[110] = 101.0
        c[111:130] = np.linspace(102, 112, 19)
        c[130:] = 111.0
        rank = np.full(1200, 999)
        ret = np.zeros(1200)
        rank[115:] = 1
        ret[115:] = 0.10
        rank[140:] = 20                      # loses the lead later -> lost-rank exit
        g = _grid(c, rank, ret)
        y, st = RO.weak_hold_arm(g, _trade(), 0.0, P)
        self.assertEqual(st, "leader")
        self.assertAlmostEqual(y, EV.wide_from(g, _trade(), 115, P), places=9)

    def test_leader_before_signal_bar_is_ignored(self):
        c = np.full(400, 100.0)
        c[110] = 101.0
        c[111:] = 95.0
        rank = np.full(400, 999)
        ret = np.zeros(400)
        rank[105:110] = 1
        ret[105:110] = 0.10
        y, st = RO.weak_hold_arm(_grid(c, rank, ret), _trade(), 0.0, P)
        self.assertEqual(st, "trail")

    def test_leader_pending(self):
        c = np.linspace(100, 140, 300)
        rank = np.full(300, 1)
        ret = np.full(300, 0.10)
        y, st = RO.weak_hold_arm(_grid(c, rank, ret), _trade(), 0.0, P)
        self.assertIsNone(y)
        self.assertEqual(st, "pending")


class TestCompare(unittest.TestCase):
    def _c(self):
        c = np.full(600, 100.0)
        c[110] = 101.0
        c[111:131] = np.linspace(102, 110, 20)
        c[131:] = 105.0
        return c

    def test_changed_unchanged_pending_and_window(self):
        g = _grid(self._c())
        trades = [_trade(),                                      # changed, +4
                  _trade(reason=OTHER_REASONS[0]),               # other reason -> unchanged
                  _trade(ix=105),                                # profit 0 at signal -> unchanged
                  _trade(day="2026-10-05"),                      # before the window -> excluded
                  _trade(ie=400, ix=560)]                        # would hold, not resolved -> pending
        g.C[400:560, 0] = 100.0
        g.C[560, 0] = 101.0
        g.C[561:, 0] = np.linspace(102, 120, 39)
        days = {"2026-10-10"}
        rows, info = RO.weak_hold_compare(g, trades, days, 0.5, P, cfg=config, floor_fn=lambda m: 0.0)
        self.assertEqual(info["pending"], 1)
        self.assertEqual(info["changed"], 1)
        self.assertEqual(len(rows["primary"]), 3)
        diffs = sorted(y - x for _t, x, y in rows["primary"])
        self.assertAlmostEqual(diffs[-1], 4.0, places=6)
        self.assertEqual(diffs[:2], [0.0, 0.0])


# ---------------------------------------------------------------- 9: verdict, anomalies, status

def _ev(sym="AUSDT", tf="15m", dt=None, would_hold=True, leader_mode=False):
    dt = dt or datetime(2026, 10, 22, 10, 7, tzinfo=UTC)
    return {"event": "weak_exit_shadow", "sym": sym, "tf": tf, "ts": dt.isoformat(), "_dt": dt,
            "would_hold": would_hold, "leader_mode": leader_mode}


def _exit(sym="AUSDT", tf="15m", dt=None):
    dt = dt or datetime(2026, 10, 22, 10, 7, 2, tzinfo=UTC)
    return {"event": "exit", "sym": sym, "tf": tf, "ts": dt.isoformat(), "_dt": dt, "pnl_pct": 0.7}


class TestVerdict(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(RO.weak_hold_verdict("accept"), "KEEP")
        self.assertEqual(RO.weak_hold_verdict("reject"), "ROLLBACK_SUGGESTED")
        self.assertEqual(RO.weak_hold_verdict("needs_data"), "INCONCLUSIVE")
        self.assertEqual(RO.weak_hold_verdict("needs_review"), "INCONCLUSIVE")

    def test_clean_events_no_anomaly(self):
        self.assertEqual(RO.weak_hold_anomalies([_ev()], [_exit()], {}), [])

    def test_leader_mode_event(self):
        self.assertTrue(RO.weak_hold_anomalies([_ev(leader_mode=True)], [_exit()], {}))

    def test_wrong_tf(self):
        self.assertTrue(RO.weak_hold_anomalies([_ev(tf="1h")], [_exit(tf="1h")], {}))

    def test_event_without_exit(self):
        self.assertTrue(RO.weak_hold_anomalies([_ev()], [], {}))
        late = _exit(dt=datetime(2026, 10, 22, 10, 30, tzinfo=UTC))
        self.assertTrue(RO.weak_hold_anomalies([_ev()], [late], {}))

    def test_disagreement(self):
        t0 = datetime(2026, 10, 22, 0, 7, tzinfo=UTC)
        evs = [_ev(dt=t0 + timedelta(hours=k), would_hold=True) for k in range(20)]
        exs = [_exit(dt=t0 + timedelta(hours=k, seconds=1)) for k in range(20)]
        keys = {RO.weak_hold_key(e): True for e in evs}
        self.assertEqual(RO.weak_hold_anomalies(evs, exs, keys), [])
        bad = dict(keys)
        for e in evs[:2]:                               # 2 of 20 = 10% > 5%
            bad[RO.weak_hold_key(e)] = False
        self.assertTrue(RO.weak_hold_anomalies(evs, exs, bad))
        few = {RO.weak_hold_key(e): False for e in evs[:5]}   # fewer than 20 matched: not judged
        self.assertEqual(RO.weak_hold_anomalies(evs[:5], exs[:5], few), [])

    def test_status(self):
        r = {"due": "2026-11-30", "min_days": 42, "min_trades": 40}
        self.assertEqual(RO.status_of(r, {"days": 50, "_n_trades": 60, "_verdict": "KEEP"}, "2026-11-01"), "COLLECTING")
        self.assertEqual(RO.status_of(r, {"days": 30, "_n_trades": 60, "_verdict": "KEEP"}, "2026-12-01"), "TOO_EARLY")
        self.assertEqual(RO.status_of(r, {"days": 50, "_n_trades": 39, "_verdict": "KEEP"}, "2026-12-01"), "TOO_EARLY")
        self.assertEqual(RO.status_of(r, {"days": 50, "_n_trades": 40, "_verdict": "KEEP"}, "2026-12-01"), "KEEP")
        self.assertEqual(RO.status_of(r, {"days": 1, "_anomaly": True}, "2026-10-22"), "ANOMALY")


class TestReadoutEndToEnd(unittest.TestCase):
    def test_runs_on_mocks(self):
        c = np.full(600, 100.0)
        c[110] = 101.0
        c[111:131] = np.linspace(102, 110, 20)
        c[131:] = 105.0
        g = _grid(c)
        trades = [_trade(), _trade(reason=OTHER_REASONS[0])]
        with mock.patch.object(EV, "load_grid", lambda p: g), \
             mock.patch.object(RO, "weak_hold_trades", lambda grid, since: trades), \
             mock.patch.object(EV, "day_labels", lambda: ({("2026-10-10", "AUSDT")}, set())), \
             mock.patch.object(EV, "floor_of", lambda m, cfg: 0.0), \
             mock.patch.object(RO, "scan_events", lambda since, kinds, path=None: []):
            out = RO.readout_weak_exit_shadow({"window_from": "2026-10-06"},
                                              {"days": ["2026-10-10"], "now": datetime(2026, 10, 22, tzinfo=UTC)})
        self.assertEqual(out["changed"], 1)
        self.assertEqual(out["_n_trades"], 1)
        self.assertFalse(out["_anomaly"])
        self.assertIn("primary", out)
        self.assertIn("plain_p1", out)
        self.assertIn(out["_verdict"], ("KEEP", "ROLLBACK_SUGGESTED", "INCONCLUSIVE"))


# ---------------------------------------------------------------- 10-11: registry and docs

class TestRegistryAndDocs(unittest.TestCase):
    def test_registry(self):
        reg = json.loads((HERE / "readout_registry.json").read_text(encoding="utf-8"))["readouts"]
        r = [x for x in reg if x["id"] == "WEAK-HOLD-SHADOW"]
        self.assertEqual(len(r), 1)
        r = r[0]
        self.assertEqual(r["fn"], "readout_weak_exit_shadow")
        self.assertIn(r["fn"], RO.FNS)
        self.assertEqual(r["flag"], "RSI_DIV_HOLD_SHADOW_ENABLED")
        self.assertEqual(r["window_from"], "2026-10-06")
        self.assertEqual(r["due"], "2026-11-30")
        self.assertEqual(r["min_days"], 42)
        self.assertEqual(r["min_trades"], 40)
        self.assertTrue((ROOT / r["spec"]).exists())
        for k in ("keep", "rollback", "otherwise"):
            self.assertIn(k, r["criteria"])

    def test_documented(self):
        for f in ("CLAUDE.md", "PROJECT_CONTEXT.md"):
            self.assertTrue("RSI_DIV_HOLD_SHADOW_ENABLED" in (ROOT / f).read_text(encoding="utf-8"), f)


if __name__ == "__main__":
    unittest.main()
