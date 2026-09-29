"""Improvement-agent tasks of 2026-09-29 (agent-tasks-0929-spec.md):
1 incident analyst, 2 goal validator in L3, 4 pre-registered readouts."""
import io
import json
import os
import random
import tempfile
import types
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import goal_validator as GV
import incident_analyst as IA
import pipeline_replay_validator as RV
import readouts as RO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UTC = timezone.utc


def _ev(ts, **kw):
    return dict({"ts": ts}, **kw)


def _write_events(rows):
    fd, p = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return Path(p)


def _cfg(**over):
    c = types.SimpleNamespace(
        TREND_1H_CHOP_ADX_MIN=25.0, TREND_1H_CHOP_SLOPE_MIN=0.7, TREND_1H_CHOP_VOL_MIN=1.3,
        TREND_1H_CHOP_ADX_MIN_BULL_DAY=22.0, TREND_1H_CHOP_SLOPE_MIN_BULL_DAY=1.0,
        TREND_1H_CHOP_VOL_MIN_BULL_DAY=1.2, TREND_1H_CHOP_USE_BULL_DAY_RELAX=True,
        ATR_PERIOD=14, COOLDOWN_BARS=8, TRAIL_MIN_BUFFER_PCT_ENABLED=False)
    for k, v in over.items():
        setattr(c, k, v)
    return c


def _bars_1h(start, n=200, p0=100.0, step=0.2):
    out = []
    for i in range(n):
        c = p0 + step * i
        out.append((start + timedelta(hours=i), c - 0.1, c + 0.5, c - 0.5, c))
    return out


# ------------------------------------------------------------------ goal validator

class TestGoalValidatorRules(unittest.TestCase):
    def test_relax_rules(self):
        self.assertEqual(GV.decide("relax", 0.0, 0.0, 1.0, 1.0), "reject")        # no goal gain
        self.assertEqual(GV.decide("relax", 2.0, -0.2, 1.0, 1.0), "reject")       # fails non-inferiority
        self.assertEqual(GV.decide("relax", 0.6, -0.05, 1.0, 0.5), "accept")
        self.assertEqual(GV.decide("relax", 0.3, -0.05, 1.0, 1.0), "needs_review")
        self.assertEqual(GV.decide("relax", 0.6, -0.05, 1.0, 0.0), "needs_review")

    def test_tighten_rules(self):
        self.assertEqual(GV.decide("tighten", -0.5, 0.1, 0.5, 1.0), "reject")     # loses goal
        self.assertEqual(GV.decide("tighten", 0.0, -0.3, -0.01, 1.0), "reject")   # clearly worse trades
        self.assertEqual(GV.decide("tighten", 0.0, 0.05, 0.4, 1.0), "accept")
        self.assertEqual(GV.decide("tighten", -0.2, -0.1, 0.4, 1.0), "needs_review")

    def test_combined_minus_current(self):
        self.assertAlmostEqual(GV.combined_minus_current([1.0, 1.0], [4.0], weight=1.0), 1.0)
        self.assertAlmostEqual(GV.combined_minus_current([1.0, 1.0], [4.0], weight=0.5), 0.6)
        # removing the only losing trade from [2, 2, -4]: 2 - 0 = +2
        self.assertAlmostEqual(GV.combined_minus_current([2.0, 2.0, -4.0], [-4.0], remove=True), 2.0)

    def test_bootstrap_interval_contains_point(self):
        pt, lo, hi = GV.bootstrap([1.0, 2.0, 3.0, 0.0] * 5, [0.5, -1.0] * 5, 0.4, random.Random(1))
        self.assertLessEqual(lo, pt)
        self.assertLessEqual(pt, hi)

    def test_live_trail_exits_below_stop(self):
        import numpy as np
        c = np.array([100, 101, 102, 95, 96], dtype=float)
        atr = np.array([1, 1, 1, 1, 1], dtype=float)
        j, pnl = GV.live_trail(c, atr, 0, 2.0)
        self.assertEqual(j, 3)
        self.assertAlmostEqual(pnl, -5.0)

    def test_combine_goal_decides_peak_kept(self):
        out = GV.combine({"verdict": "reject", "reason": "goal"}, {"verdict": "accept", "reason": "peak", "band": {"n": 40}})
        self.assertEqual(out["verdict"], "reject")
        self.assertEqual(out["peak_replay"]["verdict"], "accept")
        out = GV.combine({"error": "boom"}, {"verdict": "needs_review", "reason": "peak"})
        self.assertEqual(out["verdict"], "needs_review")
        self.assertIn("goal validator unavailable", out["reason"])


class TestOnlyJudgedRowsFlip(unittest.TestCase):
    """2026-09-29 fix: a row the gate never judged (blocked LATER by impulse_guard)
    must not count as admitted by relaxing the gate -- in either validator."""

    def setUp(self):
        t = "2026-09-10T0{h}:05:00+00:00"
        self.rows = [
            # A: blocked by chop on a bull day, ADX 21 -> admitted by 22 -> 20
            _ev(t.format(h=3), event="blocked", sym="AAAUSDT", tf="1h", price=100.0, signal_type="trend_1h_chop",
                reason_code="trend_chop", adx=21.0, slope_pct=1.5, vol_x=1.5, is_bull_day=True),
            # B: passed chop (not a trend candidate), blocked later by impulse_guard; ADX 21 too
            _ev(t.format(h=3), event="blocked", sym="BBBUSDT", tf="1h", price=100.0, signal_type="impulse_guard",
                reason_code="impulse_guard", adx=21.0, slope_pct=1.5, vol_x=1.5, is_bull_day=True),
        ]
        self.path = _write_events(self.rows)
        d0 = datetime(2026, 9, 10, tzinfo=UTC)
        self.winners = [("2026-09-10", "AAAUSDT", d0, d0 + timedelta(hours=10)),
                        ("2026-09-10", "BBBUSDT", d0, d0 + timedelta(hours=10))]

    def tearDown(self):
        os.unlink(self.path)

    def test_goal_validator_counts_only_blocked_here(self):
        start = datetime(2026, 9, 1, tzinfo=UTC)
        res = GV.validate({"config_key": "TREND_1H_CHOP_ADX_MIN_BULL_DAY", "diff": {"to": 20}},
                          since="2026-09-01", cfg_module=_cfg(), events_path=self.path,
                          bars_loader=lambda s, tf: _bars_1h(start), winners=self.winners)
        self.assertEqual(res["direction"], "relax")
        self.assertEqual(res["goal"]["newly_reachable_days"], 1)        # AAA only, never BBB
        self.assertEqual(res["verdict"], "needs_data")                  # 2 winner-days, 1 trade

    def test_peak_replay_counts_only_blocked_here(self):
        old_ev, old_fp = RV.EVENTS, RV.forward_peak
        RV.EVENTS, RV.forward_peak = self.path, (lambda *a, **k: 1.0)
        try:
            res = RV.validate({"config_key": "TREND_1H_CHOP_ADX_MIN_BULL_DAY", "diff": {"to": 20}},
                              since="2026-09-01", cfg_module=_cfg())
        finally:
            RV.EVENTS, RV.forward_peak = old_ev, old_fp
        self.assertEqual(res["band"]["n"], 1)


class TestL3Wiring(unittest.TestCase):
    def test_flag_off_keeps_peak(self):
        import config
        import pipeline_validator as PV
        old = getattr(config, "L3_GOAL_VALIDATOR_ENABLED", None)
        config.L3_GOAL_VALIDATOR_ENABLED = False
        try:
            peak = {"verdict": "accept", "reason": "peak"}
            self.assertIs(PV._with_goal_verdict({"config_key": "TREND_1H_CHOP_ADX_MIN"}, peak), peak)
        finally:
            config.L3_GOAL_VALIDATOR_ENABLED = old

    def test_unvalidatable_key_keeps_peak_reject(self):
        import pipeline_validator as PV
        peak = {"verdict": "reject", "reason": "unvalidatable"}
        self.assertIs(PV._with_goal_verdict({"config_key": "NOPE_KEY"}, peak), peak)

    def test_flag_on_by_default(self):
        import config
        self.assertTrue(config.L3_GOAL_VALIDATOR_ENABLED)


# ------------------------------------------------------------------ incident analyst

def _bars15(day, closes):
    d0 = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC)
    return [(d0 + timedelta(minutes=15 * i), closes[i - 1] if i else closes[0], c * 1.001, c * 0.999, c)
            for i, c in enumerate(closes)]


class TestIncidentAnalyst(unittest.TestCase):
    DAY = "2026-09-28"

    def prof(self):
        closes = [100.0] * 8 + [101, 103, 106, 111, 115, 118] + [117.0] * 82   # +2.5% at bar 9, +10% at bar 11
        return IA.day_profile(_bars15(self.DAY, closes), self.DAY)

    def test_day_profile(self):
        p = self.prof()
        self.assertEqual(p["t25"], "2026-09-28T02:30:00+00:00")
        self.assertEqual(p["t10"], "2026-09-28T03:00:00+00:00")
        self.assertTrue(p["rocket"])
        self.assertIsNone(IA.day_profile([], self.DAY))

    def _dt(self, hhmm):
        return datetime.fromisoformat(f"{self.DAY}T{hhmm}:00+00:00")

    def case(self, evs, hb=(), hb_have=True, hours=24):
        for e in evs:
            e.setdefault("_dt", self._dt(e.pop("t")))
        return IA.classify_case(self.DAY, "XUSDT", self.prof(), list(evs), list(hb), hb_have, hours)

    def test_bot_down_is_not_a_miss(self):
        c = self.case([], hours=10)
        self.assertEqual(c["stage"], "bot_down")
        self.assertEqual(c["missed_pct"], 0.0)
        s = IA.summarise([c])
        self.assertEqual(s["goal_entered_before_crossing"]["of"], 0)

    def test_caught_early_and_exit(self):
        c = self.case([{"t": "01:00", "event": "entry", "tf": "15m", "mode": "trend", "price": 100.5},
                       {"t": "03:30", "event": "exit", "tf": "15m", "pnl_pct": 5.0, "exit_price": 105.5,
                        "reason": "⚠️ WEAK: RSI дивергенция"}])
        self.assertEqual(c["stage"], "caught_early")
        self.assertAlmostEqual(c["lead_hours"], 1.5)
        self.assertEqual(c["exit"]["class"], "WEAK: RSI divergence")
        self.assertGreater(c["exit"]["left_after_exit_pct"], 5)
        self.assertAlmostEqual(c["missed_pct"], round(c["move_pct"] - 5.0, 2))

    def test_caught_late(self):
        c = self.case([{"t": "05:00", "event": "entry", "tf": "1h", "mode": "impulse_speed", "price": 117.0}])
        self.assertEqual(c["stage"], "caught_late")
        self.assertLess(c["lead_hours"], 0)
        self.assertEqual(c["exit"], {"open": True})

    def test_held_from_before(self):
        c = self.case([{"_dt": self._dt("00:00") - timedelta(days=1), "t": "00:00", "event": "entry", "tf": "15m"}])
        self.assertEqual(c["stage"], "held_from_before")

    def test_blocked_first_gate(self):
        c = self.case([{"t": "01:00", "event": "blocked", "signal_type": "trend_quality",
                        "reason": "weak 15m trend (forecast 0.000 < 0.25"},
                       {"t": "02:00", "event": "blocked", "signal_type": "correlation_guard"}])
        self.assertEqual(c["stage"], "blocked")
        self.assertEqual(c["first_gate"], "trend_quality: forecast0")
        self.assertEqual(c["gates_birth"], ["correlation_guard", "trend_quality: forecast0"])

    def test_heartbeat_stages(self):
        self.assertEqual(self.case([], hb_have=False)["stage"], "no_heartbeat")
        self.assertEqual(self.case([], hb=[])["stage"], "not_polled")
        ev = {"bar_utc": f"{self.DAY}T01:00", "tf": "15m", "stage": "evaluated", "fired": [],
              "reasons": {"entry": "RSI 78 вне зоны", "impulse": "slope 0.1 < 0.5", "retest": "x"}}
        c = self.case([], hb=[ev])
        self.assertEqual(c["stage"], "no_rule_fired")
        self.assertIn("entry: RSI out of zone", c["rule_reasons"])
        c = self.case([], hb=[dict(ev, fired=["impulse"])])
        self.assertEqual(c["stage"], "rule_fired_no_candidate")
        c = self.case([], hb=[{"bar_utc": f"{self.DAY}T01:00", "tf": "15m", "stage": "cooldown"}])
        self.assertEqual(c["stage"], "cooldown")

    def test_sanitize_groups(self):
        res = {"groups": [{"title": "a", "mechanism": "exits on WEAK", "case_ids": ["d:X", "invented"]},
                          {"title": "b", "mechanism": "m", "case_ids": ["nope"]},
                          {"title": "c", "mechanism": "you should lower the floor", "case_ids": ["d:X"]}]}
        g = IA.sanitize_groups(res, {"d:X"})
        self.assertEqual([x["case_ids"] for x in g], [["d:X"], ["d:X"]])
        self.assertFalse(g[0]["remedy_wording"])
        self.assertTrue(g[1]["remedy_wording"])

    def test_seek_offset(self):
        rows = [_ev(f"2026-09-{d:02d}T00:00:00+00:00", event="x", pad="p" * 400) for d in range(1, 29) for _ in range(40)]
        p = _write_events(rows)
        try:
            with io.open(p, "rb") as fh:
                off = IA.seek_offset(fh, "2026-09-20")
                fh.seek(off)
                first = [json.loads(l)["ts"] for l in fh if json.loads(l)["ts"] >= "2026-09-20"][0]
            self.assertTrue(first.startswith("2026-09-20"))
            self.assertLessEqual(off, p.stat().st_size)
        finally:
            os.unlink(p)

    def test_rollup_and_render(self):
        c1 = self.case([{"t": "01:00", "event": "entry", "tf": "15m", "price": 100.5}])
        c2 = self.case([{"t": "01:00", "event": "blocked", "signal_type": "trend_quality", "reason": "RSI 80"}])
        c2["case_id"] = "2026-09-28:YUSDT"
        c2["sym"] = "YUSDT"
        rep = {"day": self.DAY, "status": "ok", "cases": [c1, c2], "summary": IA.summarise([c1, c2]), "llm_groups": []}
        with tempfile.TemporaryDirectory() as d:
            IA.write_report(rep, Path(d))
            r = IA.rollup(days=3, until="2026-09-29", out_dir=Path(d))
        self.assertEqual(r["cases"], 2)
        self.assertEqual(r["goal_entered_before_crossing"]["n"], 1)
        self.assertEqual(r["first_gate"], {"trend_quality: RSI": 1})
        b = IA.render_block(rep)
        self.assertIn("Взяли до +2.5%: 1 из 2", b)


# ------------------------------------------------------------------ readouts

class TestReadouts(unittest.TestCase):
    def test_registry_is_pre_registered(self):
        reg = RO.load_registry()
        ids = [r["id"] for r in reg]
        self.assertEqual(len(ids), len(set(ids)))
        for r in reg:
            self.assertIn(r["kind"], ("computed", "delegated"))
            self.assertTrue(r["due"] and r["window_from"] and r["min_days"] > 0, r["id"])
            self.assertIn("keep", r["criteria"], r["id"])
            self.assertTrue((ROOT / r["spec"]).exists(), r["spec"])
            if r["kind"] == "computed":
                self.assertIn(r["fn"], RO.FNS)
                self.assertIn("rollback", r["criteria"])
                self.assertTrue(r.get("rollback"))
                # the window may not open before the change was live
                self.assertGreaterEqual(r["window_from"], r["switched_on_utc"][:10])

    def test_status_logic(self):
        r = {"kind": "computed", "due": "2026-10-09", "min_days": 14, "min_trades": 5}
        self.assertEqual(RO.status_of(r, {"days": 20, "_n_trades": 9, "_verdict": "KEEP"}, "2026-10-01"), "COLLECTING")
        self.assertEqual(RO.status_of(r, {"days": 10, "_n_trades": 9, "_verdict": "KEEP"}, "2026-10-09"), "TOO_EARLY")
        self.assertEqual(RO.status_of(r, {"days": 14, "_n_trades": 4, "_verdict": "KEEP"}, "2026-10-09"), "TOO_EARLY")
        self.assertEqual(RO.status_of(r, {"days": 14, "_n_trades": 5, "_verdict": "KEEP"}, "2026-10-09"), "KEEP")
        # an anomaly is raised at once, before the due date
        self.assertEqual(RO.status_of(r, {"_anomaly": True}, "2026-09-30"), "ANOMALY")

    def test_wilson(self):
        lo, hi = RO.wilson(3, 3)
        self.assertAlmostEqual(lo, 0.438, places=2)
        self.assertEqual(hi, 1.0)
        self.assertEqual(RO.wilson(0, 0), (0.0, 1.0))

    def test_pair_trades_and_rocket(self):
        t = lambda h: datetime(2026, 9, 28, h, tzinfo=UTC)
        evs = [{"event": "entry", "sym": "A", "tf": "15m", "_dt": t(1)}, {"event": "exit", "sym": "A", "tf": "1h", "_dt": t(2)},
               {"event": "exit", "sym": "A", "tf": "15m", "_dt": t(3)}]
        pr = RO.pair_trades(evs)
        self.assertEqual(len(pr), 1)
        self.assertEqual(pr[0][1]["_dt"], t(3))
        self.assertTrue(RO.is_rocket({"open": 1, "high": 1.2, "close": 1.15}))
        self.assertFalse(RO.is_rocket({"open": 1, "high": 1.2, "close": 1.05}))

    def test_render(self):
        res = [{"id": "E-4", "status": "TOO_EARLY", "due": "2026-10-09", "min_days": 14, "date": "2026-10-09",
                "final": False, "result": {"days": 9}},
               {"id": "OLD", "status": "KEEP", "due": "2026-09-01", "date": "2026-09-02", "final": True, "result": {}}]
        b = RO.render_block(res, today="2026-10-09")
        self.assertIn("E-4: рано судить (9/14", b)
        self.assertNotIn("OLD", b)          # a final verdict is reported on the day it is reached only


# ------------------------------------------------------------------ wiring & docs

class TestWiring(unittest.TestCase):
    def test_flags(self):
        import config
        for f in ("INCIDENT_ANALYST_ENABLED", "READOUTS_ENABLED", "L3_GOAL_VALIDATOR_ENABLED"):
            self.assertTrue(getattr(config, f), f)
        self.assertGreater(config.INCIDENT_L2_WINDOW_DAYS, 0)

    def test_daily_run_order(self):
        src = (HERE / "pipeline_run.py").read_text(encoding="utf-8")
        i, r, n = src.index('"incidents"'), src.index('"readouts"'), src.index('run_step("notify"')
        self.assertLess(i, n)
        self.assertLess(r, n)

    def test_l2_receives_incidents(self):
        import pipeline_hypothesis as PH
        self.assertIn("incidents_14d", PH._CLAUDE_SYSTEM)
        self.assertIn("case_ids", PH._CLAUDE_SCHEMA_HINT)
        src = (HERE / "pipeline_hypothesis.py").read_text(encoding="utf-8")
        self.assertIn('"incidents_14d": _incidents_rollup()', src)

    def test_notify_blocks_never_raise(self):
        import pipeline_notify as N
        old = IA.OUT_DIR, RO.OUT_DIR
        with tempfile.TemporaryDirectory() as d:
            IA.OUT_DIR = RO.OUT_DIR = Path(d)
            try:
                self.assertEqual(N.build_agent_blocks(date(2026, 9, 29)), [])
            finally:
                IA.OUT_DIR, RO.OUT_DIR = old

    def test_documented(self):
        for doc in ("CLAUDE.md", "PROJECT_CONTEXT.md"):
            txt = (ROOT / doc).read_text(encoding="utf-8")
            for name in ("incident_analyst.py", "goal_validator.py", "readouts.py"):
                self.assertIn(name, txt, (doc, name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
