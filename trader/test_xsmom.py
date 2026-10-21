"""XSMOM -- daily cross-sectional momentum in money: tests written from xsmom-daily-money-1021-spec.md §11
BEFORE the code (operator rule 2026-10-10: spec -> tests -> code). Synthetic data, offline.

Modules are imported inside each test, so before the code exists every test fails on its own
(ModuleNotFoundError) instead of the whole file failing at import."""
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import fill_model as FM
import trading_policy as TP

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DAY_MS = 86_400_000
BAR_MS = 15 * 60 * 1000
BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)
FM1 = FM.model("taker", 1)                                   # taker 0.10% + 8 bps per side
FREE = FM.FillModel(taker_fee=0.0, maker_fee=0.0)            # no costs: cash flows by hand


def _xs():
    import xsmom
    return xsmom


def ds(k, base=BASE):
    return (base + timedelta(days=k)).strftime("%Y-%m-%d")


def arr15(day_closes, base=BASE, bars=None, no_first=(), vol=1.0, day_vol=None, intraday=0.0):
    """15m arrays (history_store layout). Day k has `bars[k]` bars (default 96) with close
    day_closes[k] * (1 + intraday * i); None = no bars that day; days in no_first lack the 00:00 bar;
    volume per bar = day_vol[k] if given else vol."""
    t0 = int(base.timestamp() * 1000)
    t, c, v = [], [], []
    for k, p in enumerate(day_closes):
        if p is None:
            continue
        n = 96 if bars is None else bars.get(k, 96)
        idx = [i for i in range(n) if not (k in no_first and i == 0)]
        for i in idx:
            t.append(t0 + k * DAY_MS + i * BAR_MS)
            c.append(p * (1 + intraday * i))
            v.append((day_vol or {}).get(k, vol))
    c = np.array(c, dtype=float)
    return {"t": np.array(t, dtype=np.int64), "o": c.copy(), "h": c * 1.001, "l": c * 0.999, "c": c,
            "v": np.array(v, dtype=float)}


def panel(cols):
    XS = _xs()
    return XS.Panel.from_daily({s: XS.daily_from_15m(a) for s, a in cols.items()})


def walk(seed, names, n):
    """Random-walk closes for several pairs, same seed -> same series."""
    rng = np.random.default_rng(seed)
    return {s: list(100 * np.cumprod(1 + rng.normal(0, 0.04, n))) for s in names}


# --------------------------------------------------------------------------- §11.1 daily bars
class TestDailyBars(unittest.TestCase):
    def test_close_first_close_qv(self):
        XS = _xs()
        d = XS.daily_from_15m(arr15([100.0, 200.0], intraday=0.001, vol=2.0))
        day0 = int(BASE.timestamp() * 1000) // DAY_MS
        self.assertEqual([int(x) for x in d["day"]], [day0, day0 + 1])
        self.assertAlmostEqual(d["close"][0], 100.0 * (1 + 0.001 * 95))        # last 15m close of the day
        self.assertAlmostEqual(d["first_close"][1], 200.0)                       # close of the 00:00-00:15 bar
        self.assertAlmostEqual(d["qv"][0], sum(100.0 * (1 + 0.001 * i) * 2.0 for i in range(96)))
        self.assertTrue(bool(d["complete"][0]))
        self.assertTrue(bool(d["has_first"][0]))

    def test_complete_threshold(self):
        XS = _xs()
        d = XS.daily_from_15m(arr15([100.0, 100.0], bars={0: 89, 1: 90}))
        self.assertFalse(bool(d["complete"][0]))
        self.assertTrue(bool(d["complete"][1]))
        self.assertEqual(XS.COMPLETE_MIN_BARS, 90)

    def test_no_first_bar(self):
        XS = _xs()
        d = XS.daily_from_15m(arr15([100.0], no_first={0}))
        self.assertFalse(bool(d["has_first"][0]))
        self.assertTrue(np.isnan(d["first_close"][0]))


# --------------------------------------------------------------------------- §11.2 no look-ahead
class TestNoLookAhead(unittest.TestCase):
    N, D = 40, 35
    NAMES = ("AUSDT", "BUSDT", "CUSDT", "DUSDT")

    def _panel(self, how=None):
        w = walk(3, self.NAMES, self.N)
        cols = {}
        for i, s in enumerate(self.NAMES):
            p = w[s]
            if how == "scale":
                p = p[:self.D] + [x * 3.0 for x in p[self.D:]]
            elif how == "drop":
                p = p[:self.D] + [None] * (self.N - self.D)
            cols[s] = arr15(p, vol=1.0 + i)
        return panel(cols)

    def test_universe_scores_market_ignore_day_d_and_later(self):
        XS = _xs()
        cfg = XS.Config(L=7, H=1, K=2, n_liq=3, filter_days=7, filter_min_pairs=1)
        d = ds(self.D)
        ref = self._panel()
        u0 = XS.universe(ref, d, cfg)
        s0 = XS.scores(ref, d, cfg.L, u0)
        m0 = XS.market_on(ref, d, u0, cfg)
        self.assertEqual(len(u0), 3)
        for how in ("scale", "drop"):
            p = self._panel(how)
            u = XS.universe(p, d, cfg)
            self.assertEqual(u, u0, how)
            self.assertEqual(XS.scores(p, d, cfg.L, u), s0, how)
            self.assertEqual(XS.market_on(p, d, u, cfg), m0, how)


# --------------------------------------------------------------------------- §11.3 universe
class TestUniverse(unittest.TestCase):
    D = 34

    def cfg(self, **kw):
        XS = _xs()
        base = dict(L=3, H=1, K=2)
        base.update(kw)
        return XS.Config(**base)

    def test_leveraged_excluded(self):
        XS = _xs()
        cols = {s: arr15([100.0] * 35) for s in ("BTCUSDT", "BTCUPUSDT", "BTCDOWNUSDT", "ETHUSDT")}
        self.assertEqual(sorted(XS.universe(panel(cols), ds(self.D), self.cfg())), ["BTCUSDT", "ETHUSDT"])

    def test_needs_complete_signal_days(self):
        XS = _xs()
        d, L = self.D, 3
        cols = {"AUSDT": arr15([100.0] * 35, bars={d - 1: 50}),            # day d-1 incomplete
                "BUSDT": arr15([100.0] * 35, bars={d - 1 - L: 50}),        # day d-1-L incomplete
                "CUSDT": arr15([100.0] * 35)}
        self.assertEqual(XS.universe(panel(cols), ds(d), self.cfg(L=L)), ["CUSDT"])

    def test_liquidity_needs_20_complete_days_of_30(self):
        XS = _xs()
        d = self.D
        cols = {"AUSDT": arr15([100.0] * 35, bars={k: 10 for k in range(d - 30, d - 19)}),   # 11 incomplete -> 19
                "BUSDT": arr15([100.0] * 35, bars={k: 10 for k in range(d - 30, d - 20)})}   # 10 incomplete -> 20
        self.assertEqual(XS.universe(panel(cols), ds(d), self.cfg()), ["BUSDT"])
        self.assertEqual((XS.LIQ_DAYS, XS.LIQ_MIN_DAYS, XS.N_LIQ), (30, 20, 100))

    def test_top_n_by_median_liquidity_ties_by_name(self):
        XS = _xs()
        d = self.D
        cols = {"AUSDT": arr15([100.0] * 35, vol=1.0),
                "BUSDT": arr15([100.0] * 35, vol=3.0),
                "CUSDT": arr15([100.0] * 35, vol=2.0),
                "DUSDT": arr15([100.0] * 35, vol=2.0),
                "EUSDT": arr15([100.0] * 35, vol=1.0, day_vol={d - 1: 1000.0})}   # huge mean, small median
        self.assertEqual(XS.universe(panel(cols), ds(d), self.cfg(n_liq=3)), ["BUSDT", "CUSDT", "DUSDT"])


# --------------------------------------------------------------------------- §11.4 selection
class TestSelection(unittest.TestCase):
    def test_top_k_order_and_ties(self):
        XS = _xs()
        s = {"DUSDT": -0.1, "CUSDT": 0.3, "AUSDT": 0.1, "BUSDT": 0.3}
        self.assertEqual(XS.top_k(s, 2), ["BUSDT", "CUSDT"])
        self.assertEqual(XS.top_k(s, 10), ["BUSDT", "CUSDT", "AUSDT", "DUSDT"])

    def test_score_is_close_d_minus_1_over_close_d_minus_1_minus_L(self):
        XS = _xs()
        p = panel({"AUSDT": arr15([100.0, 100.0, 100.0, 120.0, 999.0])})
        self.assertAlmostEqual(XS.scores(p, ds(4), 2, ["AUSDT"])["AUSDT"], 0.2)

    def test_default_chooser_is_momentum(self):
        XS = _xs()
        cfg = XS.Config(L=7, H=1, K=2)
        ctx = XS.Ctx(day=ds(10), tranche=0, held=frozenset(), universe=["AUSDT", "BUSDT", "CUSDT"],
                     scores={"AUSDT": 0.1, "BUSDT": 0.5, "CUSDT": 0.3}, cfg=cfg)
        self.assertEqual(XS.momentum_chooser(ctx), ["BUSDT", "CUSDT"])
        w = walk(5, ("AUSDT", "BUSDT", "CUSDT", "DUSDT"), 60)
        p = panel({s: arr15(v, vol=1.0 + i) for i, (s, v) in enumerate(w.items())})
        cfg = XS.Config(L=7, H=1, K=2, liq_min_days=20)
        a = XS.rotate(p, cfg, FM1, ds(40), ds(60))
        b = XS.rotate(p, cfg, FM1, ds(40), ds(60), chooser=XS.momentum_chooser)
        self.assertEqual(a["series"], b["series"])


# --------------------------------------------------------------------------- §11.5 money
class TestRotateMoney(unittest.TestCase):
    def test_single_coin_by_hand(self):
        XS = _xs()
        p = panel({"AUSDT": arr15([100.0, 100.0, 110.0, 121.0])})
        r = XS.rotate(p, XS.Config(L=1, H=1, K=1), FM1, ds(1), ds(3), chooser=lambda ctx: ["AUSDT"])
        qty = 10_000.0 / (100.0 * 1.0008) * 0.999
        self.assertEqual([d for d, _ in r["series"]], [ds(1), ds(2)])
        self.assertAlmostEqual(r["series"][0][1], (qty * 100.0 / 10_000.0 - 1) * 100, places=9)
        self.assertAlmostEqual(r["series"][1][1], (110.0 / 100.0 - 1) * 100, places=9)
        proceeds = qty * 110.0 * (1 - 0.0008) * 0.999            # window edge: sold at close of the last day
        self.assertEqual(len(r["trades"]), 1)
        self.assertAlmostEqual(r["trades"][0]["pnl"], proceeds - 10_000.0, places=6)
        self.assertAlmostEqual(r["final_equity"], proceeds, places=6)
        self.assertTrue(r["conservation_ok"])

    def test_conservation_and_aggregate(self):
        XS = _xs()
        import daily_revenue as DR
        w = walk(7, [f"{c}USDT" for c in "ABCDEFGH"], 60)
        p = panel({s: arr15(v, vol=1.0 + i) for i, (s, v) in enumerate(w.items())})
        r = XS.rotate(p, XS.Config(L=7, H=1, K=3, liq_min_days=20), FM1, ds(40), ds(60))
        self.assertGreater(len(r["trades"]), 0)
        self.assertTrue(r["conservation_ok"])
        self.assertAlmostEqual(r["final_equity"] - 10_000.0, sum(t["pnl"] for t in r["trades"]), places=6)
        self.assertEqual(r["agg"], DR.aggregate(r["series"]))
        self.assertGreaterEqual(r["turnover_per_day"], 0.0)

    def test_kept_names_not_traded_and_freed_cash_split(self):
        XS = _xs()
        cols = {s: arr15([100.0] * 6) for s in ("AUSDT", "BUSDT", "CUSDT")}
        sched = {ds(1): ["AUSDT", "BUSDT"], ds(2): ["AUSDT", "CUSDT"], ds(3): ["AUSDT", "CUSDT"]}
        r = XS.rotate(panel(cols), XS.Config(L=1, H=1, K=2), FREE, ds(1), ds(4), chooser=lambda ctx: sched[ctx.day])
        self.assertEqual(r["moves"][(0, ds(1))], (0, 2))
        self.assertEqual(r["moves"][(0, ds(2))], (1, 1))
        self.assertEqual(r["moves"][(0, ds(3))], (0, 0))
        a = [t for t in r["trades"] if t["sym"] == "AUSDT"]
        self.assertEqual(len(a), 1)                                 # one holding, closed only at the window edge
        self.assertEqual(a[0]["reason"], "flush")
        b = [t for t in r["trades"] if t["sym"] == "BUSDT"][0]
        self.assertEqual((b["entry_day"], b["exit_day"], b["reason"]), (ds(1), ds(2), "rotate"))
        c = [t for t in r["trades"] if t["sym"] == "CUSDT"][0]
        self.assertAlmostEqual(c["cost"], 5_000.0, places=6)        # B's proceeds, no costs, flat price

    def test_min_trade(self):
        XS = _xs()
        cols = {s: arr15([100.0] * 4) for s in ("AUSDT", "BUSDT")}
        r = XS.rotate(panel(cols), XS.Config(L=1, H=1, K=2), FM1, ds(1), ds(3),
                      chooser=lambda ctx: ["AUSDT", "BUSDT"], start_equity=15.0)
        self.assertEqual(r["trades"], [])
        self.assertTrue(all(abs(v) < 1e-12 for _, v in r["series"]))
        self.assertEqual(XS.MIN_TRADE_USDT, 10.0)

    def test_weekly_tranches(self):
        XS = _xs()
        calls = []

        def ch(ctx):
            calls.append((ctx.tranche, ctx.day))
            return ["AUSDT"]
        r = XS.rotate(panel({"AUSDT": arr15([100.0] * 25)}), XS.Config(L=1, H=7, K=1), FREE, ds(1), ds(22), chooser=ch)
        self.assertTrue({(i, ds(1)) for i in range(7)} <= set(calls))       # every tranche buys on the first day
        later = [c for c in calls if c[1] != ds(1)]
        for tr, day in later:
            k = (datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc) - BASE).days - 1
            self.assertEqual(k % 7, tr)
        self.assertEqual(len([c for c in later if c[0] == 3]), 3)            # offsets 3, 10, 17 inside [1, 22)
        self.assertEqual(len(r["trades"]), 7)
        self.assertTrue(all(abs(t["cost"] - 10_000.0 / 7) < 1e-6 for t in r["trades"]))

    def test_market_filter_sells_all_and_rebuys(self):
        XS = _xs()
        prices = [100.0, 101.0, 102.0, 103.0, 90.0, 91.0, 92.0, 93.0]
        cols = {s: arr15(prices) for s in ("AUSDT", "BUSDT")}
        cfg = XS.Config(L=1, H=7, K=1, market_filter=True, n_liq=2, liq_min_days=1, filter_days=1, filter_min_pairs=1)
        r = XS.rotate(panel(cols), cfg, FM1, ds(2), ds(8), chooser=lambda ctx: ["AUSDT"])
        filt = [t for t in r["trades"] if t["reason"] == "filter"]
        self.assertEqual(len(filt), 7)                                       # day 5: close4/close3 < 0 -> off
        self.assertTrue(all(t["exit_day"] == ds(5) for t in filt))
        self.assertEqual(len([t for t in r["trades"] if t["entry_day"] == ds(6)]), 7)   # on again -> all rebuy
        self.assertAlmostEqual(r["exposure_share"], 5 / 6)

    def test_market_on_rules(self):
        XS = _xs()
        p = panel({s: arr15([100.0, 100.0, 90.0, 90.0]) for s in ("AUSDT", "BUSDT")})
        cfg = XS.Config(L=1, H=1, K=1, filter_days=1, filter_min_pairs=1)
        self.assertFalse(XS.market_on(p, ds(3), ["AUSDT", "BUSDT"], cfg))                  # close2 / close1 = 0.9
        self.assertTrue(XS.market_on(p, ds(3), ["AUSDT", "BUSDT"], replace(cfg, filter_min_pairs=3)))  # no data, no block
        self.assertEqual((XS.FILTER_DAYS, XS.FILTER_MIN_PAIRS), (28, 20))

    def test_delisting_haircut_without_fee(self):
        XS = _xs()
        cols = {"AUSDT": arr15([100.0, 100.0, 100.0, 80.0]), "BUSDT": arr15([100.0] * 8)}
        r = XS.rotate(panel(cols), XS.Config(L=1, H=1, K=1), FM1, ds(1), ds(7), chooser=lambda ctx: ["AUSDT"])
        t = [x for x in r["trades"] if x["reason"] == "delisted"]
        self.assertEqual(len(t), 1)
        self.assertEqual(t[0]["exit_day"], ds(4))                            # the day after the last bar
        qty = 10_000.0 / (100.0 * 1.0008) * 0.999
        self.assertAlmostEqual(t[0]["pnl"], qty * 80.0 * (1 - XS.DELIST_HAIRCUT) - 10_000.0, places=6)
        self.assertEqual(XS.DELIST_HAIRCUT, 0.5)
        self.assertTrue(r["conservation_ok"])

    def test_gap_inside_window_is_not_delisting(self):
        XS = _xs()
        cols = {"AUSDT": arr15([100.0, 100.0, 100.0, None, None, 120.0, 120.0, 120.0]), "BUSDT": arr15([100.0] * 8)}
        r = XS.rotate(panel(cols), XS.Config(L=1, H=1, K=1), FM1, ds(1), ds(8), chooser=lambda ctx: ["AUSDT"])
        self.assertFalse(any(t["reason"] == "delisted" for t in r["trades"]))
        self.assertEqual(len([t for t in r["trades"] if t["sym"] == "AUSDT"]), 1)
        self.assertAlmostEqual(dict(r["series"])[ds(3)], 0.0, places=9)     # marked at the last known close

    def test_sell_waits_for_a_first_bar_and_buy_needs_one(self):
        XS = _xs()
        sched = {ds(k): (["AUSDT"] if k == 1 else ["BUSDT"]) for k in range(1, 6)}
        cols = {"AUSDT": arr15([100.0] * 6, no_first={2}), "BUSDT": arr15([100.0] * 6)}
        r = XS.rotate(panel(cols), XS.Config(L=1, H=1, K=1), FM1, ds(1), ds(6), chooser=lambda ctx: sched[ctx.day])
        self.assertEqual([t for t in r["trades"] if t["sym"] == "AUSDT"][0]["exit_day"], ds(3))
        self.assertEqual([t for t in r["trades"] if t["sym"] == "BUSDT"][0]["entry_day"], ds(3))
        r2 = XS.rotate(panel({"CUSDT": arr15([100.0] * 6, no_first={1})}), XS.Config(L=1, H=1, K=1), FM1, ds(1), ds(3),
                       chooser=lambda ctx: ["CUSDT"])
        self.assertEqual(r2["trades"][0]["entry_day"], ds(2))

    def test_window_edge_reads_nothing_beyond(self):
        XS = _xs()
        base = [100.0, 100.0, 105.0, 110.0, 120.0]
        cfg = XS.Config(L=1, H=1, K=1)
        a = XS.rotate(panel({"AUSDT": arr15(base + [130.0, 140.0])}), cfg, FM1, ds(1), ds(5), chooser=lambda ctx: ["AUSDT"])
        b = XS.rotate(panel({"AUSDT": arr15(base + [1.0, None])}), cfg, FM1, ds(1), ds(5), chooser=lambda ctx: ["AUSDT"])
        self.assertEqual(a["series"], b["series"])
        self.assertAlmostEqual(a["final_equity"], b["final_equity"], places=9)
        self.assertEqual(a["trades"][-1]["reason"], "flush")

    def test_panel_cut(self):
        XS = _xs()
        p = panel({"AUSDT": arr15([100.0] * 10, base=datetime(2025, 12, 26, tzinfo=timezone.utc))})
        q = p.cut("2026-01-01")
        self.assertEqual(XS.day_str(q.days[-1]), "2025-12-31")
        self.assertEqual(q.close.shape[0], len(q.days))


# --------------------------------------------------------------------------- §11.6 parity with E1
class TestE1Parity(unittest.TestCase):
    def test_k1_rotation_equals_execution_replay(self):
        XS = _xs()
        import execution_replay as R
        from test_execution_replay import FakeGrid, T0
        A = arr15([100.0, 102.0, 101.0, 105.0, 103.0, 108.0], base=T0, intraday=0.0002)
        B = arr15([50.0, 49.0, 52.0, 53.0, 55.0, 54.0], base=T0, intraday=-0.0001)
        sched = {2: ["AUSDT"], 3: ["BUSDT"], 4: ["BUSDT"], 5: ["AUSDT"]}

        def ch(ctx):
            return sched[(datetime.strptime(ctx.day, "%Y-%m-%d").replace(tzinfo=timezone.utc) - T0).days]
        rot = XS.rotate(panel({"AUSDT": A, "BUSDT": B}), XS.Config(L=1, H=1, K=1), FM1, ds(2, T0), ds(6, T0), chooser=ch)
        grid = FakeGrid(np.column_stack([A["c"], B["c"]]).tolist(), syms=("AUSDT", "BUSDT"))
        cands = XS.to_e1_candidates(rot["trades"])
        R.attach_grid(cands, grid)
        pol = replace(TP.CHAMPION, name="xsmom_parity", sizing=TP.Sizing(max_position_pct=100.0), risk=TP.Risk(max_open=1))
        e1 = R.run(pol, FM1, grid, cands)
        a, b = dict(rot["series"]), dict(e1["series"])
        common = sorted(set(a) & set(b))
        self.assertEqual(common, [ds(k, T0) for k in range(2, 6)])
        for d in common:
            self.assertAlmostEqual(a[d], b[d], places=9, msg=d)
        self.assertAlmostEqual(rot["final_equity"], e1["final_equity"], places=6)


# --------------------------------------------------------------------------- §11.7 random_matched
class TestRandomMatched(unittest.TestCase):
    def _setup(self):
        XS = _xs()
        w = walk(11, [f"{c}USDT" for c in "ABCDEFGHIJ"], 70)
        p = panel({s: arr15(v, vol=1.0 + i) for i, (s, v) in enumerate(w.items())})
        cfg = XS.Config(L=7, H=1, K=3, n_liq=8, liq_min_days=20)
        return XS, p, cfg, XS.rotate(p, cfg, FM1, ds(40), ds(70))

    def test_same_moves_as_the_candidate(self):
        XS, p, cfg, base = self._setup()
        r = XS.rotate(p, cfg, FM1, ds(40), ds(70), chooser=XS.matched_random(base["moves"], seed=0))
        self.assertEqual(r["moves"], base["moves"])

    def test_deterministic_by_seed(self):
        XS, p, cfg, base = self._setup()
        syms = lambda s: [t["sym"] for t in XS.rotate(p, cfg, FM1, ds(40), ds(70),
                                                      chooser=XS.matched_random(base["moves"], seed=s))["trades"]]
        self.assertEqual(syms(0), syms(0))
        self.assertNotEqual(syms(0), syms(1))

    def test_buys_only_from_the_universe(self):
        XS, p, cfg, base = self._setup()
        seen = []
        inner = XS.matched_random(base["moves"], seed=3)

        def spy(ctx):
            out = inner(ctx)
            seen.append((set(out) - set(ctx.held), set(ctx.universe)))
            return out
        XS.rotate(p, cfg, FM1, ds(40), ds(70), chooser=spy)
        self.assertTrue(seen)
        for new, uni in seen:
            self.assertTrue(new <= uni)

    def test_random_control_n_seeds(self):
        XS, p, cfg, base = self._setup()
        m = XS.random_control(p, cfg, FM1, ds(40), ds(70), base["moves"], n=5)
        self.assertEqual(len(m), 5)
        self.assertEqual(m, XS.random_control(p, cfg, FM1, ds(40), ds(70), base["moves"], n=5))


# --------------------------------------------------------------------------- §11.8 ewu
class TestEwu(unittest.TestCase):
    def test_holds_the_universe_with_equal_new_buys(self):
        XS = _xs()
        cols = {s: arr15([100.0] * 40, vol=v) for s, v in (("AUSDT", 1.0), ("BUSDT", 2.0), ("CUSDT", 3.0), ("DUSDT", 0.5))}
        cfg = XS.Config(L=1, H=1, K=10, n_liq=3, liq_min_days=20)
        r = XS.rotate(panel(cols), cfg, FREE, ds(31), ds(33), chooser=XS.ewu_chooser)
        first = [t for t in r["trades"] if t["entry_day"] == ds(31)]
        self.assertEqual(sorted(t["sym"] for t in first), ["AUSDT", "BUSDT", "CUSDT"])
        for t in first:
            self.assertAlmostEqual(t["cost"], 10_000.0 / 3, places=6)


# --------------------------------------------------------------------------- §11.9 procedure
class TestProcedure(unittest.TestCase):
    def test_candidates_exactly_section_5(self):
        import experiment_xsmom as XE
        c = XE.candidates()
        names = {f"{row}_L{L}_H{H}_K{K}" for row in ("M", "MF") for L in (7, 14, 28) for H in (1, 7) for K in (5, 10)}
        self.assertEqual(set(c), names)
        for name, cfg in c.items():
            self.assertEqual(cfg.name, name)
            self.assertEqual(cfg.market_filter, name.startswith("MF_"))
        self.assertEqual(set(XE.ROWS), {"M", "MF"})
        self.assertEqual(sorted(sum(XE.ROWS.values(), [])), sorted(c))

    def test_constants(self):
        import experiment_xsmom as XE
        self.assertEqual((XE.TRAIN_FROM, XE.TEST_FROM), ("2023-03-01", "2026-01-01"))
        self.assertEqual((XE.N_RANDOM, XE.MIN_TRADES, XE.PBO_MAX), (50, 200, 0.5))

    def _res(self, **kw):
        r = {"mean_pct": 0.30, "random_means": [0.0] * 40 + [0.1] * 10, "paired_ewu": {"mean_pct": 0.05},
             "paired_ewu_by_year": {"2023": 0.1, "2024": -0.1, "2025": 0.2}, "n_trades": 250}
        r.update(kw)
        return r

    def test_admit_needs_every_condition(self):
        import experiment_xsmom as XE
        self.assertTrue(XE.admit(self._res())[0])
        self.assertFalse(XE.admit(self._res(mean_pct=0.05))[0])                       # not above random p95 (0.1)
        self.assertFalse(XE.admit(self._res(paired_ewu={"mean_pct": -0.01}))[0])
        self.assertFalse(XE.admit(self._res(paired_ewu_by_year={"2023": 0.1, "2024": -0.1, "2025": -0.2}))[0])
        self.assertFalse(XE.admit(self._res(n_trades=199))[0])

    def test_pbo_gate_and_selection(self):
        import experiment_xsmom as XE
        self.assertTrue(XE.row_open(0.49))
        self.assertFalse(XE.row_open(0.5))
        res = {"M_L7_H1_K5": {"admitted": True, "paired_ewu": {"mean_pct": 0.1}},
               "M_L14_H1_K5": {"admitted": True, "paired_ewu": {"mean_pct": 0.3}},
               "M_L28_H1_K5": {"admitted": False, "paired_ewu": {"mean_pct": 0.9}},
               "MF_L7_H1_K5": {"admitted": False, "paired_ewu": {"mean_pct": 0.5}}}
        self.assertEqual(XE.select_by_row(res, {"M": 0.2, "MF": 0.1}), {"M": "M_L14_H1_K5"})
        self.assertEqual(XE.select_by_row(res, {"M": 0.6, "MF": 0.1}), {})

    def test_stop_and_test_opened_once(self):
        import experiment_xsmom as XE
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d) / "XSMOM_TEST_OPENED.json"
            called = []
            out = XE.pipeline(lambda: {}, lambda w, a: called.append((w, a)), marker)
            self.assertTrue(out["stop"])
            self.assertEqual(called, [])
            self.assertFalse(marker.exists())
            XE.pipeline(lambda: {"M": "M_L14_H7_K10", "MF": "MF_L7_H7_K5"}, lambda w, a: called.append((w, a)), marker)
            self.assertEqual(len(called), 1)
            self.assertAlmostEqual(called[0][1], 0.025)                                 # alpha = 0.05 / k, k = 2
            self.assertTrue(marker.exists())
            with self.assertRaises(RuntimeError):
                XE.pipeline(lambda: {"M": "M_L14_H7_K10"}, lambda w, a: None, marker)

    def test_require_train_and_alpha(self):
        import experiment_xsmom as XE
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                XE.require_train(Path(d) / "XSMOM_TRAIN.json")
        self.assertAlmostEqual(XE.alpha_for(1), 0.05)
        self.assertAlmostEqual(XE.alpha_for(2), 0.025)

    def test_train_panel_has_no_test_days(self):
        import experiment_xsmom as XE
        XS = _xs()
        p = XE.train_panel(panel({"AUSDT": arr15([100.0] * 10, base=datetime(2025, 12, 26, tzinfo=timezone.utc))}))
        self.assertLess(XS.day_str(p.days[-1]), XE.TEST_FROM)

    def test_verdicts(self):
        import experiment_xsmom as XE
        v = XE.verdict(mean_pct=0.2, random_means=[0.0] * 50, paired_ewu={"lo95": 0.01},
                       day_agg={"mean_pct": 0.2}, trade_ci={"lo95": -0.1})
        self.assertEqual((v["beats_random"], v["beats_market"], v["profitable"]), (True, True, False))
        v = XE.verdict(mean_pct=-0.1, random_means=[0.0] * 50, paired_ewu={"lo95": -0.01},
                       day_agg={"mean_pct": -0.1}, trade_ci={"lo95": -0.3})
        self.assertEqual((v["beats_random"], v["beats_market"], v["profitable"]), (False, False, False))


# --------------------------------------------------------------------------- §11.10 boundary
class TestBoundary(unittest.TestCase):
    def test_bot_does_not_import_xsmom(self):
        for f in (ROOT / "files").glob("*.py"):
            txt = f.read_text(encoding="utf-8", errors="replace")
            self.assertNotIn("import xsmom", txt, f.name)
            self.assertNotIn("import experiment_xsmom", txt, f.name)

    def test_outputs_in_trader_runtime(self):
        import experiment_xsmom as XE
        import trader_config as TC
        for p in (XE.OUT_DIR, XE.TEST_MARKER.parent):
            self.assertTrue(str(p).startswith(str(TC.RUNTIME)), p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
