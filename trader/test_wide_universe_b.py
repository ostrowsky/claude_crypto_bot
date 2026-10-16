"""Phase B (wide universe) -- tests written from wide-universe-b-1014-spec.md §6 BEFORE the code
(operator rule 2026-10-10: spec -> tests -> code)."""
import json
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
MS = lambda d: int(d.timestamp() * 1000)  # noqa: E731
BAR = 15 * 60 * 1000


def bars15(n, start=T0, base=100.0, step=0.1, vol=1.0, skip=()):
    t, o, h, l, c, v = [], [], [], [], [], []
    for i in range(n):
        if i in skip:
            continue
        p = base + i * step
        t.append(MS(start) + i * BAR)
        o.append(p)
        h.append(p + 0.5)
        l.append(p - 0.5)
        c.append(p + 0.05)
        v.append(vol + i)
    return {k: np.array(x, dtype=float if k != "t" else np.int64) for k, x in zip("tohlcv", (t, o, h, l, c, v))}


# ======================================================================================== §6.1 history

class TestHistory(unittest.TestCase):
    def test_agg_1h_equals_ohlcv_of_four_15m(self):
        import history_store as HS
        a = bars15(8)
        r = HS.agg_1h(a)
        self.assertEqual(list(r["t"]), [MS(T0), MS(T0) + 4 * BAR])
        self.assertEqual(r["o"][0], a["o"][0])
        self.assertEqual(r["h"][0], max(a["h"][:4]))
        self.assertEqual(r["l"][0], min(a["l"][:4]))
        self.assertEqual(r["c"][0], a["c"][3])
        self.assertAlmostEqual(r["v"][0], sum(a["v"][:4]))
        self.assertTrue(all(r["complete"]))

    def test_incomplete_hour_flagged_and_droppable(self):
        import history_store as HS
        a = bars15(8, skip=(5,))
        r = HS.agg_1h(a)
        self.assertEqual(list(r["complete"]), [True, False])
        self.assertEqual(len(HS.agg_1h(a, complete_only=True)["t"]), 1)

    def test_merge_append_only_no_duplicates(self):
        import history_store as HS
        a, b = bars15(10), bars15(15)
        m, added = HS.merge(a, b)
        self.assertEqual(added, 5)
        self.assertEqual(len(m["t"]), 15)
        self.assertTrue((np.diff(m["t"]) > 0).all())
        m2, added2 = HS.merge(m, b)
        self.assertEqual((added2, len(m2["t"])), (0, 15))

    def test_gaps_recorded_not_filled(self):
        import history_store as HS
        a = bars15(10, skip=(4, 5))
        g = HS.gaps(a["t"])
        self.assertEqual(len(g), 1)
        self.assertEqual(g[0]["missing"], 2)
        self.assertEqual(len(a["t"]), 8)                       # nothing invented

    def test_fetch_pages_and_resumes(self):
        import history_store as HS

        class C:
            def __init__(self, n):
                self.n, self.calls = n, []

            def klines_from(self, symbol, interval, start_ms, limit=1000, base=None):
                self.calls.append((symbol, interval, start_ms, base))
                rows = []
                for k in range(self.n):
                    ts = MS(T0) + k * BAR
                    if ts >= start_ms and len(rows) < limit:
                        rows.append([ts, "1", "2", "0.5", "1.5", "10", ts + BAR - 1])
                return rows

        c = C(2500)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "XUSDT_15m.npz"
            st = HS.fetch_symbol(c, "XUSDT", MS(T0), MS(T0) + 2500 * BAR, p)
            self.assertEqual(st["added"], 2500)
            self.assertEqual(st["requests"], 3)                 # 1000 + 1000 + 500
            self.assertTrue(all(x[1] == "15m" and "api.binance.com" in x[3] for x in c.calls))
            c.calls.clear()
            st2 = HS.fetch_symbol(c, "XUSDT", MS(T0), MS(T0) + 2500 * BAR, p)
            self.assertEqual(st2["added"], 0)
            self.assertTrue(c.calls[0][2] == MS(T0) + 2500 * BAR)   # resumes after the last stored bar

    def test_network_errors_retried_and_progress_kept(self):
        """Found 2026-10-14 on the first live run: one SSL handshake timeout lost the whole pair."""
        import exchange_client as X
        import history_store as HS

        class C:
            def __init__(self, fail_at):
                self.fail_at, self.calls = dict(fail_at), 0

            def klines_from(self, symbol, interval, start_ms, limit=1000, base=None):
                self.calls += 1
                k = self.fail_at.get(self.calls)
                if k:
                    raise X.ExchangeError(None, "network", "handshake timed out", "/api/v3/klines")
                rows = []
                for i in range(2500):
                    ts = MS(T0) + i * BAR
                    if ts >= start_ms and len(rows) < limit:
                        rows.append([ts, "1", "2", "0.5", "1.5", "10", ts + BAR - 1])
                return rows
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "XUSDT_15m.npz"
            st = HS.fetch_symbol(C({2: True}), "XUSDT", MS(T0), MS(T0) + 2500 * BAR, p, sleep=lambda s: None)
            self.assertEqual(st["added"], 2500)                       # one failure retried
            p2 = Path(d) / "YUSDT_15m.npz"
            with self.assertRaises(X.ExchangeError):
                HS.fetch_symbol(C({2: 1, 3: 1, 4: 1, 5: 1}), "YUSDT", MS(T0), MS(T0) + 2500 * BAR, p2, sleep=lambda s: None)
            self.assertEqual(len(HS.load(p2)["t"]), 1000)             # the first page is kept

    def test_backtest_universe_includes_delisted_excludes_stables(self):
        import history_store as HS
        xinfo = {"symbols": [
            {"symbol": "AUSDT", "baseAsset": "A", "quoteAsset": "USDT", "status": "TRADING", "isSpotTradingAllowed": True},
            {"symbol": "BUSDT", "baseAsset": "B", "quoteAsset": "USDT", "status": "BREAK", "isSpotTradingAllowed": False},
            {"symbol": "USDCUSDT", "baseAsset": "USDC", "quoteAsset": "USDT", "status": "TRADING", "isSpotTradingAllowed": True},
            {"symbol": "ABTC", "baseAsset": "A", "quoteAsset": "BTC", "status": "TRADING", "isSpotTradingAllowed": True}]}
        self.assertEqual(HS.backtest_universe(xinfo), ["AUSDT", "BUSDT"])
        self.assertEqual(HS.live_universe(xinfo), ["AUSDT"])

    def test_gold_token_is_tradable_not_a_stablecoin(self):
        """PAXG (gold-backed) moves with gold; operator decision #2 is "all coins tradable on spot" (2026-10-14)."""
        import history_store as HS
        xinfo = {"symbols": [{"symbol": "PAXGUSDT", "baseAsset": "PAXG", "quoteAsset": "USDT", "status": "TRADING", "isSpotTradingAllowed": True}]}
        self.assertEqual(HS.backtest_universe(xinfo), ["PAXGUSDT"])
        self.assertEqual(HS.live_universe(xinfo), ["PAXGUSDT"])


# ======================================================================================== §6.2 generator

CFG = types.SimpleNamespace(ATR_TRAIL_K=2.0, ATR_TRAIL_K_STRONG=2.5, ATR_TRAIL_K_RETEST=1.5, ATR_TRAIL_K_BREAKOUT=1.5,
                            ATR_TRAIL_K_CROSS=2.5, ATR_TRAIL_K_TREND_SURGE=2.5, TREND_SURGE_PRECEDENCE_ENABLED=False,
                            EMA_CROSS_ENABLED=False, CROSS_COOLDOWN_BARS=6, ENTRY_BLOCK_HOURS=[],
                            EMA_SLOW=50, SLOPE_LOOKBACK=5, ADX_PERIOD=14, ADX_GROW_BARS=3, VOL_LOOKBACK=20)


def F(**kw):
    f = {"entry": False, "breakout": False, "retest": False, "surge": False, "impulse": False, "alignment": False, "ema_cross": False}
    f.update(kw)
    return f


class TestModeSelection(unittest.TestCase):
    """= monitor.py ~L5621-5694: breakout > retest > (surge if precedence) > entry > surge > impulse > alignment > ema_cross."""

    def test_priority_and_trail_k(self):
        import candidate_gen as CG
        sel = lambda flags, em="trend", cfg=CFG: CG.select_mode(flags, em, cfg)   # noqa: E731
        self.assertEqual(sel(F(breakout=True, retest=True, entry=True)), ("breakout", 1.5))
        self.assertEqual(sel(F(retest=True, entry=True)), ("retest", 1.5))
        self.assertEqual(sel(F(entry=True, surge=True)), ("trend", 2.0))
        self.assertEqual(sel(F(entry=True), "strong_trend"), ("strong_trend", 2.5))
        self.assertEqual(sel(F(entry=True), "impulse_speed"), ("impulse_speed", 2.5))
        self.assertEqual(sel(F(surge=True, impulse=True)), ("impulse_speed", 2.5))
        self.assertEqual(sel(F(impulse=True, alignment=True)), ("impulse", 2.0))
        self.assertEqual(sel(F(alignment=True)), ("alignment", 2.0))
        self.assertIsNone(sel(F(ema_cross=True)))                                  # EMA_CROSS_ENABLED False
        on = types.SimpleNamespace(**{**vars(CFG), "EMA_CROSS_ENABLED": True, "TREND_SURGE_PRECEDENCE_ENABLED": True})
        self.assertEqual(sel(F(ema_cross=True), cfg=on), ("impulse_cross", 2.5))
        self.assertEqual(sel(F(entry=True, surge=True), cfg=on), ("trend_surge", 2.5))
        self.assertIsNone(sel(F()))


class TestGenerator(unittest.TestCase):
    def rules(self, fire_at):
        """Fake bot rules: entry fires at bar indices in fire_at."""
        return {"features": lambda o, h, l, c, v: {"n": len(c)},
                "entry": lambda feat, i, c, tf: (i in fire_at, "x"), "breakout": lambda feat, i: (False, ""),
                "retest": lambda feat, i: (False, ""), "surge": lambda feat, i: (False, ""), "impulse": lambda feat, i: (False, ""),
                "alignment": lambda feat, i, tf: (False, ""), "ema_cross": lambda feat, i: (False, ""),
                "entry_mode": lambda feat, i, c, tf: ("trend", "")}

    def test_event_format_close_time_and_price(self):
        import candidate_gen as CG
        a = bars15(200)
        ev = CG.generate("XUSDT", "15m", a, cfg=CFG, rules=self.rules({150}))
        self.assertEqual(len(ev), 1)
        e = ev[0]
        self.assertEqual((e["event"], e["sym"], e["tf"], e["mode"], e["trail_k"], e["source"]), ("entry", "XUSDT", "15m", "trend", 2.0, "wide_gen"))
        self.assertEqual(e["ts"], datetime.fromtimestamp((a["t"][150] + BAR) / 1000, timezone.utc).isoformat())
        self.assertEqual(e["price"], float(a["c"][150]))

    def test_only_closed_bars_and_warmup(self):
        import candidate_gen as CG
        a = bars15(200)
        now = datetime.fromtimestamp((a["t"][150] + BAR // 2) / 1000, timezone.utc)      # bar 150 still forming
        self.assertEqual(CG.generate("XUSDT", "15m", a, cfg=CFG, rules=self.rules({150}), now=now), [])
        w = CG.warmup(CFG)
        self.assertEqual(CG.generate("XUSDT", "15m", a, cfg=CFG, rules=self.rules({w - 1})), [])

    def test_block_hours(self):
        import candidate_gen as CG
        a = bars15(200)
        hour = datetime.fromtimestamp(a["t"][150] / 1000, timezone.utc).hour
        cfg = types.SimpleNamespace(**{**vars(CFG), "ENTRY_BLOCK_HOURS": [hour]})
        self.assertEqual(CG.generate("XUSDT", "15m", a, cfg=cfg, rules=self.rules({150})), [])

    def test_bot_functions_called_not_copied(self):
        src = (HERE / "candidate_gen.py").read_text(encoding="utf-8")
        for name in ("compute_features", "check_entry_conditions", "check_breakout_conditions", "check_retest_conditions",
                     "check_impulse_conditions", "check_alignment_conditions", "check_trend_surge_conditions", "get_effective_entry_mode"):
            self.assertIn(name, src, name)
            self.assertNotIn("def " + name, src, name)


# ======================================================================================== §6.3 calibration

class TestCalibration(unittest.TestCase):
    def test_recall_and_width(self):
        import gen_calibration as GC
        b = lambda s, tf, k: {"sym": s, "tf": tf, "ts": T0 + timedelta(minutes=15 * k, seconds=40)}   # noqa: E731
        c = lambda s, tf, k: {"sym": s, "tf": tf, "ts": T0 + timedelta(minutes=15 * k)}               # noqa: E731
        bot = [b("A", "15m", 10), b("A", "15m", 50), b("B", "1h", 4), b("C", "15m", 7)]
        cands = [c("A", "15m", 11), c("A", "15m", 53), c("B", "15m", 16), c("C", "15m", 7), c("C", "15m", 30), c("D", "15m", 3)]
        r = GC.match(bot, cands, tol_bars=1)
        self.assertEqual((r["n_bot"], r["n_matched"]), (4, 2))            # A@10 (±1), C@7; A@50 is 3 bars off; B other tf
        self.assertAlmostEqual(r["recall"], 0.5)
        self.assertEqual((r["n_cands"], r["n_taken"]), (6, 2))
        self.assertFalse(GC.ok(r))
        self.assertTrue(GC.ok({"recall": 0.80}))
        # operator decision 2026-10-14: threshold 0.80 -> 0.79 (spec §8)
        self.assertEqual(GC.RECALL_MIN, 0.79)
        self.assertTrue(GC.ok({"recall": 0.79}))
        self.assertFalse(GC.ok({"recall": 0.7899}))
        self.assertTrue(GC.ok({"recall": 4868 / 6160}))

    def test_bot_entries_reader(self):
        import gen_calibration as GC
        rows = [{"event": "entry", "sym": "AUSDT", "tf": "15m", "ts": "2026-03-05T10:01:00+00:00", "price": 1},
                {"event": "exit", "sym": "AUSDT", "tf": "15m", "ts": "2026-03-05T11:01:00+00:00"},
                {"event": "entry", "sym": "ZUSDT", "tf": "1h", "ts": "2026-03-06T10:01:00+00:00", "price": 1},
                {"event": "blocked", "sym": "AUSDT", "tf": "15m", "ts": "2026-03-06T10:01:00+00:00"},
                {"event": "entry", "sym": "AUSDT", "tf": "1h", "ts": "2026-02-01T10:01:00+00:00", "price": 1}]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ev.jsonl"
            p.write_text("".join(json.dumps(r) + "\n" for r in rows) + "{broken\n", encoding="utf-8")
            got = GC.bot_entries(p, since="2026-03-01", until="2026-10-13", syms={"AUSDT", "ZUSDT"})
            self.assertEqual([(g["sym"], g["tf"]) for g in got], [("AUSDT", "15m"), ("ZUSDT", "1h")])
            self.assertEqual(len(GC.bot_entries(p, since="2026-03-01", until="2026-10-13", syms={"AUSDT"})), 1)


# ======================================================================================== §4a lazy grid

class TestLazyGrid(unittest.TestCase):
    def test_run_parity_with_dense_grid(self):
        import copy
        import execution_replay as R
        import fill_model as FM
        import lazy_grid as LG
        import shadow_portfolio as S
        import trading_policy as TP
        from test_execution_replay import T0 as RT0
        from test_shadow_portfolio import two_sym_grid, grid_events, write_events
        g = two_sym_grid()
        arrays = {s: {"C": g.C[:, j], "O": g.O[:, j], "H": g.H[:, j], "L": g.L[:, j], "V": g.V[:, j], "ATR": g.ATR[:, j]}
                  for s, j in g.col.items()}
        lz = LG.LazyGrid.from_arrays(arrays, {s: 0 for s in g.syms}, RT0, g.N)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ev.jsonl"
            write_events(p, grid_events(g, n_trades=40, seed=11))
            tr, _ = R.load_events(since="2026-03-01", path=p)
        for pol in (S.X4, TP.CHAMPION):
            a, b = copy.deepcopy(tr), copy.deepcopy(tr)
            R.attach_grid(a, g)
            R.attach_grid(b, lz)
            ra = R.run(pol, FM.model("taker", 1), g, a, floor_of=lambda m: 0.0)
            rb = R.run(pol, FM.model("taker", 1), lz, b, floor_of=lambda m: 0.0)
            ka = [(t["sym"], t["entry_ts"], t["reason"], round(t["ret_pct"], 9)) for t in ra["trades"]]
            kb = [(t["sym"], t["entry_ts"], t["reason"], round(t["ret_pct"], 9)) for t in rb["trades"]]
            self.assertEqual(ka, kb, pol.name)
            self.assertTrue(ka, pol.name)
            self.assertEqual([(d_, round(v, 9)) for d_, v in ra["series"]], [(d_, round(v, 9)) for d_, v in rb["series"]])
            self.assertAlmostEqual(ra["final_equity"], rb["final_equity"], places=6)

    def test_cache_dense_with_nan_gaps(self):
        import lazy_grid as LG
        a = bars15(12, skip=(4, 5))
        with tempfile.TemporaryDirectory() as d:
            LG.build_cache("XUSDT", a, Path(d), atr_fn=lambda h, l, c: np.arange(len(c), dtype=float) + 1)
            lz = LG.LazyGrid.load(Path(d), ["XUSDT"], T0 - timedelta(minutes=30))
            j = lz.col["XUSDT"]
            ix0 = lz.idx(T0)
            self.assertEqual(ix0, 2)
            self.assertAlmostEqual(lz.C[ix0, j], a["c"][0])
            self.assertTrue(np.isnan(lz.C[ix0 + 4, j]) and np.isnan(lz.ATR[ix0 + 5, j]))   # the gap
            self.assertAlmostEqual(lz.C[ix0 + 6, j], a["c"][4])                            # 7th bar = 5th stored
            self.assertAlmostEqual(lz.ATR[ix0 + 6, j], 5.0)                                # ATR over bars without gaps
            self.assertTrue(np.isnan(lz.C[0, j]) and np.isnan(lz.C[ix0 + 100, j]))        # outside the pair's range
            lz.close()

    def test_atr_is_the_bots(self):
        src = (HERE / "lazy_grid.py").read_text(encoding="utf-8")
        self.assertIn("_atr", src)
        self.assertNotIn("def _atr", src)


# ======================================================================================== §6.4 money

class TestMoney(unittest.TestCase):
    def test_arms_and_thinning(self):
        import experiment_b3 as B3
        cands = [{"sym": s, "ts": T0 + timedelta(minutes=k)} for k, s in enumerate(["A", "B", "N1", "N2", "A", "N3"])]
        arms = B3.split_arms(cands, {"A", "B"})
        self.assertEqual([x["sym"] for x in arms["W93"]], ["A", "B", "A"])
        self.assertEqual([x["sym"] for x in arms["WNEW"]], ["N1", "N2", "N3"])
        self.assertEqual(len(arms["WALL"]), 6)
        t1, t2 = B3.thin(cands, 3, seed=7), B3.thin(cands, 3, seed=7)
        self.assertEqual(t1, t2)
        self.assertEqual(len(t1), 3)
        self.assertEqual(t1, sorted(t1, key=lambda x: x["ts"]))

    def test_verdict_inclusive_rules(self):
        import experiment_b3 as B3
        v = B3.verdict(paired={"lo95": 0.01, "mean_pct": 0.2}, wall={"lo95": -0.1, "mean_pct": 0.1}, random_means=[0.0, 0.05, 0.2])
        self.assertEqual((v["improves"], v["profitable"], v["beats_random"]), (True, False, True))
        v = B3.verdict(paired={"lo95": 0.0, "mean_pct": 0.2}, wall={"lo95": 0.0, "mean_pct": 0.05}, random_means=[0.0, 0.05, 0.2])
        self.assertEqual((v["improves"], v["profitable"], v["beats_random"]), (False, False, False))   # strict > 0 and > median

    def test_test_half_opened_once(self):
        import experiment_b3 as B3
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "B3_TEST_OPENED.json"
            B3.open_test(m, {"winner": "WALL"})
            with self.assertRaises(RuntimeError):
                B3.open_test(m, {"winner": "WALL"})

    def test_split_dates_fixed(self):
        import experiment_b3 as B3
        self.assertEqual((B3.TRAIN_FROM, B3.TEST_FROM), ("2023-01-01", "2026-01-01"))


# ======================================================================================== §4b runner

def _synthetic_world(n=2200, syms=("AUSDT", "BUSDT", "NUSDT")):
    """Lazy grid of random walks across the train/test boundary + generator-style events on both sides."""
    import random as _r
    import lazy_grid as LG
    t0 = datetime(2025, 12, 20, tzinfo=timezone.utc)
    arrays, rnd, events = {}, _r.Random(3), []
    for k, s in enumerate(syms):
        p, C = 100.0 + k, []
        for _ in range(n):
            p *= 1 + rnd.gauss(0.0002, 0.006)
            C.append(p)
        C = np.array(C)
        arrays[s] = {"C": C, "O": np.r_[C[:1], C[:-1]], "H": C * 1.002, "L": C * 0.998, "V": np.full(n, 1e6), "ATR": np.full(n, 0.5)}
        for i in range(120, n - 120, 37 + 5 * k):
            events.append({"event": "entry", "sym": s, "tf": "15m", "ts": (t0 + timedelta(minutes=15 * (i + 1))).isoformat(),
                           "price": float(C[i]), "mode": "trend", "trail_k": 2.0, "source": "wide_gen"})
    return LG.LazyGrid.from_arrays(arrays, {s: 0 for s in syms}, t0, n), events


class TestB3Runner(unittest.TestCase):
    def test_to_replay_and_policy(self):
        import experiment_b3 as B3
        ev = [{"event": "entry", "sym": "A", "tf": "1h", "ts": "2026-01-02T10:00:00+00:00", "price": 2.0, "mode": "impulse_speed", "trail_k": 2.5},
              {"event": "entry", "sym": "B", "tf": "15m", "ts": "2026-01-01T10:00:00+00:00", "price": 1.0, "mode": "trend", "trail_k": 2.0}]
        r = B3.to_replay(ev)
        self.assertEqual([x["sym"] for x in r], ["B", "A"])
        self.assertTrue(all(x["kind"] == "entry" and isinstance(x["ts"], datetime) and x["exit_ts"] is None for x in r))
        pol = B3.b3_policy(cooldown_bars=19)
        self.assertEqual((pol.exits.kind, pol.exits.floor_override, pol.exits.max_bars), ("p1_trail", 0.08, 96))
        self.assertEqual((pol.risk.max_open, pol.risk.one_per_symbol, pol.risk.cooldown_bars), (10, True, 19))

    def test_candidate_cache(self):
        import experiment_b3 as B3
        calls = []

        def gen(sym, arr):
            calls.append(sym)
            return [{"event": "entry", "sym": sym, "tf": "15m", "ts": "2026-01-01T00:15:00+00:00", "price": 1.0, "mode": "trend", "trail_k": 2.0}]
        with tempfile.TemporaryDirectory() as d:
            a = B3.cands_for("AUSDT", None, Path(d), gen, "sha1")
            b = B3.cands_for("AUSDT", None, Path(d), gen, "sha1")
            c = B3.cands_for("AUSDT", None, Path(d), gen, "sha2")
        self.assertEqual(calls, ["AUSDT", "AUSDT"])                 # reused once, recomputed on a new sha
        self.assertEqual(a, b)
        self.assertEqual(len(c), 1)

    def test_train_does_not_see_test_and_test_once(self):
        import experiment_b3 as B3
        grid, ev = _synthetic_world()
        cands = B3.to_replay(ev)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with self.assertRaises(RuntimeError):
                B3.run_test(cands, grid, {"AUSDT", "BUSDT"}, d / "B3_TRAIN.json", d / "B3_TEST.json", d / "MARK.json",
                            trial_log=d / "trials.jsonl")
            tr = B3.run_train(cands, grid, {"AUSDT", "BUSDT"}, d / "B3_TRAIN.json")
            self.assertEqual(set(tr["arms"]), {"W93", "WALL", "WNEW"})
            self.assertTrue(all(t < B3.TEST_FROM for arm in tr["arms"].values() for t in arm["trade_days"]))
            te = B3.run_test(cands, grid, {"AUSDT", "BUSDT"}, d / "B3_TRAIN.json", d / "B3_TEST.json", d / "MARK.json",
                             trial_log=d / "trials.jsonl")
            self.assertTrue({"improves", "profitable", "beats_random", "random_median"} <= set(te["verdict"]))
            self.assertEqual(len(te["random_means"]), 20)
            self.assertTrue(all(t >= B3.TEST_FROM for arm in te["arms"].values() for t in arm["trade_days"]))
            self.assertTrue((d / "trials.jsonl").exists())
            with self.assertRaises(RuntimeError):
                B3.run_test(cands, grid, {"AUSDT", "BUSDT"}, d / "B3_TRAIN.json", d / "B3_TEST2.json", d / "MARK.json",
                            trial_log=d / "trials.jsonl")

    def test_compact_cand_parity_and_size(self):
        import copy
        import sys as _sys
        import execution_replay as R
        import experiment_b3 as B3
        import fill_model as FM
        grid, ev = _synthetic_world()
        objs = B3.to_replay(ev)
        dicts = [dict(c) for c in objs]
        self.assertTrue(all(isinstance(c, B3.Cand) for c in objs))
        a, b = [c.copy() for c in objs], copy.deepcopy(dicts)
        R.attach_grid(a, grid)
        R.attach_grid(b, grid)
        pol = B3.b3_policy(19)
        ra = R.run(pol, FM.model("taker", 1), grid, a, floor_of=lambda m: 0.0, plan_cache={})
        rb = R.run(pol, FM.model("taker", 1), grid, b, floor_of=lambda m: 0.0, plan_cache={})
        self.assertEqual([(t["sym"], t["entry_ts"], round(t["ret_pct"], 9)) for t in ra["trades"]],
                         [(t["sym"], t["entry_ts"], round(t["ret_pct"], 9)) for t in rb["trades"]])
        self.assertTrue(ra["trades"])
        self.assertLess(_sys.getsizeof(objs[0]), _sys.getsizeof(dicts[0]) / 2)

    def test_random_control_reproducible(self):
        import experiment_b3 as B3
        grid, ev = _synthetic_world()
        cands = B3.to_replay(ev)
        a = B3.random_control(cands, 10, grid, B3.TEST_FROM, None, n=5)
        b = B3.random_control(cands, 10, grid, B3.TEST_FROM, None, n=5)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 5)


# ======================================================================================== §6.5 boundary

class TestBoundary(unittest.TestCase):
    def test_runtime_only_unsigned_no_telegram(self):
        import history_store as HS
        import trader_config as TC
        self.assertTrue(str(HS.HIST_DIR).startswith(str(TC.RUNTIME)))
        for name in ("history_store.py", "candidate_gen.py", "gen_calibration.py", "experiment_b3.py"):
            src = (HERE / name).read_text(encoding="utf-8")
            self.assertNotIn('"' + '.runtime' + '"', src, name)
            self.assertNotIn("telegram", src.lower(), name)
            self.assertNotIn("BINANCE_DEMO", src, name)
            self.assertNotIn("place_order", src, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
