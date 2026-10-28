"""LA-3 -- money test of announcement events on train (listing-announcements-la-1028-spec.md §4).
Written BEFORE experiment_la.py. Synthetic bars and events, offline.

Interface the tests fix:
  TRADE_TYPES                         event types that may open a long on Binance spot
  build_events(records, spot_first_bar, t_from, t_to) -> [{sym, t_pub, etype, title}]
        tradable only, one per (etype, sym) within 24 h, train purge: t_pub < t_to - 1 day
  candidates() -> {name: {exit: "h1"|"h4"|"h24"|"x4", skip_ext: bool}}      8 per type
  trade(arr, t_pub, cand, fm, atr) -> {entry_ms, exit_ms, entry_raw, exit_raw, ret_pct, reason} | None
  simulate(events, bars, cand, fm, day_from, day_to, slots=10) -> {series, agg, trades, skipped_full, conservation_ok}
  control_events(events, bars, kind, seed, ...) -> events of the same count: "time" (same coin, random time without
        events) or "coin" (random coin without events at the same time)
  admit(res) / select(res_by_name, pbo) / open_test(marker, info) / main(["test"]) == 2
"""
import json
import random
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BAR = 15 * 60 * 1000
HOUR = 3_600_000
DAY = 86_400_000


def ms(s):
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)


T0 = ms("2025-01-01 00:00")


def bars(closes, t0=T0, opens=None):
    c = np.asarray(closes, dtype=float)
    o = np.asarray(opens, dtype=float) if opens is not None else np.r_[c[0], c[:-1]]
    return {"t": np.arange(len(c), dtype=np.int64) * BAR + t0, "o": o, "h": np.maximum(o, c) * 1.001,
            "l": np.minimum(o, c) * 0.999, "c": c, "v": np.ones(len(c))}


def zero_fm():
    import fill_model as FM
    return FM.FillModel(taker_fee=0.0, maker_fee=0.0, half_spread=0.0, impact_k=0.0)


def rec(source, title, t, catalog=None, rid=None):
    return {"source": source, "id": rid or f"{source}|{title}|{t}", "catalog": catalog, "title": title, "url": "",
            "t_pub": t, "raw": {}}


class TestEvents(unittest.TestCase):
    def test_build_events_tradable_dedup_purge(self):
        import experiment_la as EX
        first = {"BMTUSDT": T0 - 10 * DAY, "XUSDT": T0 - 10 * DAY}
        t = ms("2025-03-17 11:31")
        recs = [rec("binance", "Binance Futures Will Launch USDⓈ-Margined BMTUSDT and MUBARAKUSDT Perpetual Contracts", t, 48),
                rec("bybit", "New Listing: BMTUSDT Perpetual Contract", t + HOUR),               # other type: kept
                rec("binance", "Binance Futures Will Launch USDⓈ-Margined BMTUSDT Perpetual Contract", t + 2 * HOUR, 48,
                    rid="dup"),                                                                  # same type < 24 h: dropped
                rec("binance", "Binance Will List Newcoin (NEWC) with Seed Tag Applied", t, 48),  # not on spot yet
                rec("binance", "Binance Will Delist XUSDT on 2025-04-01", t, 161),                 # not a long event
                rec("okx", "OKX to list perpetual futures for X crypto", ms("2025-12-31 06:00"))]  # inside the purge day
        ev = EX.build_events(recs, first, T0 - 30 * DAY, ms("2026-01-01 00:00"))
        self.assertEqual(sorted((e["etype"], e["sym"]) for e in ev), [("BN_PERP_NEW", "BMTUSDT"), ("BYBIT_PERP_NEW", "BMTUSDT")])
        self.assertTrue(set(EX.TRADE_TYPES) >= {"BN_PERP_NEW", "OKX_SPOT_NEW", "OKX_PERP_NEW", "BYBIT_SPOT_NEW",
                                                "BYBIT_PERP_NEW"})
        self.assertNotIn("BN_DELIST", EX.TRADE_TYPES)

    def test_grid(self):
        import experiment_la as EX
        c = EX.candidates()
        self.assertEqual(len(c), 8)
        self.assertEqual(sorted({v["exit"] for v in c.values()}), ["h1", "h24", "h4", "x4"])
        self.assertEqual(sorted({v["skip_ext"] for v in c.values()}), [False, True])


class TestTrade(unittest.TestCase):
    def test_time_exit_and_entry_bar(self):
        import experiment_la as EX
        a = bars([100.0 + i for i in range(300)])
        t_pub = T0 + 10 * BAR + 7 * 60 * 1000                   # inside bar 10
        r = EX.trade(a, t_pub, {"exit": "h4", "skip_ext": False}, zero_fm(), None)
        self.assertEqual(r["entry_ms"], T0 + 11 * BAR)          # bar 11 opens after t_pub + 1 min
        self.assertAlmostEqual(r["entry_raw"], a["c"][11])
        self.assertAlmostEqual(r["exit_raw"], a["c"][11 + 16])
        self.assertAlmostEqual(r["ret_pct"], (a["c"][27] / a["c"][11] - 1) * 100, places=9)
        self.assertEqual(r["reason"], "time")

    def test_trail_exit(self):
        import experiment_la as EX
        c = [100.0] * 20 + [110.0] * 5 + [100.0] * 100            # rises, then falls 9% -> below the 8% floor stop
        a = bars(c)
        atr = np.full(len(c), 0.1)
        r = EX.trade(a, T0 + 5 * BAR, {"exit": "x4", "skip_ext": False}, zero_fm(), atr)
        self.assertEqual(r["reason"], "trail")
        self.assertEqual(r["exit_ms"], T0 + 25 * BAR)            # first close under 110 * 0.92
        flat = bars([100.0] * 200)
        r2 = EX.trade(flat, T0 + 5 * BAR, {"exit": "x4", "skip_ext": False}, zero_fm(), np.full(200, 0.1))
        self.assertEqual(r2["reason"], "time")
        self.assertEqual(r2["exit_ms"], T0 + (6 + 96) * BAR)

    def test_skip_extended(self):
        import experiment_la as EX
        c = [100.0] * 4 + [112.0] * 200                           # +12% since the UTC open at the entry bar
        a = bars(c, opens=[100.0] * 204)
        self.assertIsNone(EX.trade(a, T0 + 5 * BAR, {"exit": "h1", "skip_ext": True}, zero_fm(), None))
        self.assertIsNotNone(EX.trade(a, T0 + 5 * BAR, {"exit": "h1", "skip_ext": False}, zero_fm(), None))


class TestBook(unittest.TestCase):
    def test_slots_marks_conservation(self):
        import experiment_la as EX
        n = 96 * 6
        rnd = np.random.default_rng(3)
        B = {f"S{k}USDT": bars(list(100 * np.exp(np.cumsum(0.003 * rnd.standard_normal(n))))) for k in range(12)}
        evs = [{"sym": f"S{k}USDT", "t_pub": T0 + DAY + 3 * BAR, "etype": "BN_PERP_NEW", "title": ""} for k in range(12)]
        r = EX.simulate(evs, B, {"exit": "h24", "skip_ext": False}, zero_fm(), "2025-01-01", "2025-01-06", slots=10,
                        start_equity=1000.0)
        self.assertEqual(len(r["trades"]), 10)
        self.assertEqual(r["skipped_full"], 2)
        self.assertTrue(r["conservation_ok"])
        self.assertAlmostEqual(r["trades"][0]["cost"], 100.0, places=6)
        self.assertEqual(len(r["series"]), 5)


class TestControls(unittest.TestCase):
    def test_controls_shape_and_reproducible(self):
        import experiment_la as EX
        n = 96 * 40
        B = {f"S{k}USDT": bars([100.0] * n) for k in range(6)}
        evs = [{"sym": "S0USDT", "t_pub": T0 + 10 * DAY, "etype": "BN_PERP_NEW", "title": ""},
               {"sym": "S1USDT", "t_pub": T0 + 12 * DAY, "etype": "BN_PERP_NEW", "title": ""}]
        all_ev = {"S0USDT": [T0 + 10 * DAY], "S1USDT": [T0 + 12 * DAY, T0 + 13 * DAY]}
        a = EX.control_events(evs, B, "time", 5, all_ev)
        self.assertEqual([e["sym"] for e in a], ["S0USDT", "S1USDT"])
        self.assertTrue(all(abs(e["t_pub"] - x) >= 48 * HOUR for e in a for x in all_ev[e["sym"]]))
        self.assertEqual(a, EX.control_events(evs, B, "time", 5, all_ev))
        b = EX.control_events(evs, B, "coin", 5, all_ev)
        self.assertEqual([e["t_pub"] for e in b], [e["t_pub"] for e in evs])
        self.assertTrue(all(e["sym"] not in ("S0USDT",) or abs(e["t_pub"] - T0 - 10 * DAY) >= 48 * HOUR for e in b))


class TestAdmission(unittest.TestCase):
    def good(self, **kw):
        r = {"agg": {"mean_pct": 0.02, "lo95": 0.001}, "rand_time": [0.0] * 50, "rand_coin": [0.0] * 50,
             "n_trades": 100, "r_by_year": {"2023": 0.01, "2024": -0.01, "2025": 0.02}}
        r.update(kw)
        return r

    def test_admit_and_select(self):
        import experiment_la as EX
        self.assertTrue(EX.admit(self.good())[0])
        for k, v in (("rand_time", [0.03] * 50), ("rand_coin", [0.03] * 50), ("n_trades", 99),
                     ("agg", {"mean_pct": 0.02, "lo95": -0.001}),
                     ("r_by_year", {"2023": -0.01, "2024": -0.01, "2025": 0.02})):
            self.assertFalse(EX.admit(self.good(**{k: v}))[0], k)
        res = {"a": dict(self.good(), admitted=True), "b": dict(self.good(agg={"mean_pct": 0.05, "lo95": 0.01}), admitted=True),
               "c": dict(self.good(), admitted=False)}
        self.assertEqual(EX.select(res, 0.3), "b")
        self.assertIsNone(EX.select(res, 0.5))

    def test_marker_and_operator(self):
        import experiment_la as EX
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "M.json"
            EX.open_test(m, {})
            with self.assertRaises(RuntimeError):
                EX.open_test(m, {})
        self.assertEqual(EX.main(["test"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
