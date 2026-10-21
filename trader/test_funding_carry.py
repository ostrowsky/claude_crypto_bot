"""FC funding carry -- tests written from carry-funding-money-1021-spec.md §11 BEFORE the code
(operator rule 2026-10-10: spec -> tests -> code). Synthetic data, offline.

Modules are imported inside each test, so before the code exists every test fails on its own."""
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from test_xsmom import BASE, ROOT, _xs, arr15, ds, panel

DAY_MS = 86_400_000
HOUR_MS = 3_600_000
BASE_MS = int(BASE.timestamp() * 1000)


def _carry():
    import carry
    return carry


def _fs():
    import funding_store
    return funding_store


def events(days, rate=0.0001, hours=(0, 8, 16), base_ms=BASE_MS):
    """Funding events at the given UTC hours of each day index; rate may be a callable (day, hour) -> rate."""
    t, r = [], []
    for d in days:
        for h in hours:
            t.append(base_ms + d * DAY_MS + h * HOUR_MS)
            r.append(rate(d, h) if callable(rate) else rate)
    return {"t": np.array(t, dtype=np.int64), "rate": np.array(r, dtype=float)}


def market(spot_cols, perp_cols, funding, mult=None):
    C = _carry()
    return C.Market(spot=panel(spot_cols),
                    perp=C.PerpPanel.from_daily({s: C.daily_perp(a) for s, a in perp_cols.items()}, mult=mult or {}),
                    funding=funding)


def one_pair(spot_prices, perp_prices=None, n_days_funding=8):
    perp_prices = perp_prices or spot_prices
    return market({"AUSDT": arr15(spot_prices)}, {"AUSDT": arr15(perp_prices)},
                  {"AUSDT": events(range(n_days_funding))})


def cfg1(**kw):
    C = _carry()
    base = dict(W=1, K=1, theta_in=0.0, liq_min_days=1)
    base.update(kw)
    return C.CarryConfig(**base)



# --------------------------------------------------------------------------- §11.1 perp <-> spot mapping
class TestMapping(unittest.TestCase):
    def test_perp_to_spot(self):
        FS = _fs()
        spot = {"BTCUSDT", "1INCHUSDT", "1000SATSUSDT", "PEPEUSDT", "BABYDOGEUSDT", "MOGUSDT"}
        self.assertEqual(FS.perp_to_spot("BTCUSDT", spot), ("BTCUSDT", 1))
        self.assertEqual(FS.perp_to_spot("1INCHUSDT", spot), ("1INCHUSDT", 1))           # a name, not a prefix
        self.assertEqual(FS.perp_to_spot("1000SATSUSDT", spot), ("1000SATSUSDT", 1))     # exact spot name wins
        self.assertEqual(FS.perp_to_spot("1000PEPEUSDT", spot), ("PEPEUSDT", 1000))
        self.assertEqual(FS.perp_to_spot("1000000MOGUSDT", spot), ("MOGUSDT", 1_000_000))
        self.assertEqual(FS.perp_to_spot("1MBABYDOGEUSDT", spot), ("BABYDOGEUSDT", 1_000_000))
        self.assertIsNone(FS.perp_to_spot("XYZUSDT", spot))

    def test_perp_prices_are_divided_by_the_multiplier(self):
        C = _carry()
        pp = C.PerpPanel.from_daily({"PEPEUSDT": C.daily_perp(arr15([1.0, 2.0]))}, mult={"PEPEUSDT": 1000})
        self.assertAlmostEqual(float(pp.close[0, 0]), 0.001)
        self.assertAlmostEqual(float(pp.close[1, 0]), 0.002)


# --------------------------------------------------------------------------- §11.2 funding store
class TestFundingStore(unittest.TestCase):
    def test_parse_and_merge(self):
        FS = _fs()
        rows = [{"symbol": "BTCUSDT", "fundingTime": 3000, "fundingRate": "0.0001", "markPrice": ""},
                {"symbol": "BTCUSDT", "fundingTime": 1000, "fundingRate": "-0.0002", "markPrice": "100.5"}]
        a = FS.parse_funding(rows)
        self.assertEqual(list(a["t"]), [1000, 3000])                    # sorted by time
        self.assertAlmostEqual(a["rate"][0], -0.0002)
        self.assertAlmostEqual(a["mark"][0], 100.5)
        self.assertTrue(np.isnan(a["mark"][1]))                          # empty markPrice -> NaN
        b = FS.parse_funding([{"fundingTime": 3000, "fundingRate": "0.0009", "markPrice": ""},
                              {"fundingTime": 5000, "fundingRate": "0.0003", "markPrice": ""}])
        m, added = FS.merge(a, b)
        self.assertEqual(list(m["t"]), [1000, 3000, 5000])
        self.assertEqual(added, 1)
        self.assertAlmostEqual(m["rate"][1], 0.0001)                     # the stored row is kept (append-only)


# --------------------------------------------------------------------------- §11.3 signal
class TestSignal(unittest.TestCase):
    def test_annualized_window_excludes_decision_time_and_later(self):
        C = _carry()
        d0 = BASE_MS + 10 * DAY_MS
        t = np.array([d0 - 5 * DAY_MS, d0 - 2 * DAY_MS, d0 - 3 * HOUR_MS, d0, d0 + HOUR_MS], dtype=np.int64)
        r = np.array([0.5, 0.0002, 0.0001, 0.9, 0.9])
        self.assertAlmostEqual(C.annualized_funding(t, r, d0, 3), (0.0002 + 0.0001) * 365 / 3)
        self.assertIsNone(C.annualized_funding(t, r, d0 - 6 * DAY_MS, 1))

    def test_intervals_add_up_by_event(self):
        C = _carry()
        d0 = BASE_MS + 10 * DAY_MS
        e4 = events([9], rate=0.0001, hours=(0, 4, 8, 12, 16, 20))
        e8 = events([9], rate=0.0001)
        self.assertAlmostEqual(C.annualized_funding(e4["t"], e4["rate"], d0, 1), 6 * 0.0001 * 365)
        self.assertAlmostEqual(C.annualized_funding(e8["t"], e8["rate"], d0, 1), 3 * 0.0001 * 365)

    def test_no_look_ahead_in_signal_and_eligibility(self):
        C = _carry()
        cols = {s: arr15([100.0] * 40, vol=1.0 + i) for i, s in enumerate(("AUSDT", "BUSDT", "CUSDT"))}
        f1 = {s: events(range(40), rate=0.0001 * (i + 1)) for i, s in enumerate(cols)}
        f2 = {s: {"t": v["t"], "rate": np.where(v["t"] >= BASE_MS + 30 * DAY_MS, 9.9, v["rate"])} for s, v in f1.items()}
        cfg = C.CarryConfig(W=3, K=2, theta_in=0.1, liq_min_days=20)
        m1, m2 = market(cols, cols, f1), market(cols, cols, f2)
        self.assertEqual(C.eligible(m1, ds(30), cfg), C.eligible(m2, ds(30), cfg))
        u = C.eligible(m1, ds(30), cfg)
        self.assertEqual(C.signal(m1, ds(30), cfg, u), C.signal(m2, ds(30), cfg, u))


# --------------------------------------------------------------------------- §11.4 hysteresis and selection
class TestSelection(unittest.TestCase):
    def test_hysteresis_and_cap(self):
        C = _carry()
        XS = _xs()
        cfg = C.CarryConfig(W=3, K=3, theta_in=0.25)
        self.assertAlmostEqual(cfg.theta_out, 0.125)
        sc = {"AUSDT": 0.30, "BUSDT": 0.15, "CUSDT": 0.12, "DUSDT": 0.05}
        ctx = XS.Ctx(day=ds(5), tranche=0, held=frozenset({"BUSDT", "DUSDT"}), universe=list(sc), scores=sc, cfg=cfg)
        self.assertEqual(set(C.funding_chooser(ctx)), {"AUSDT", "BUSDT"})      # keep B, drop D, C below theta_in
        sc2 = {f"{c}USDT": 0.5 for c in "PQRS"}
        ctx2 = XS.Ctx(day=ds(5), tranche=0, held=frozenset(), universe=list(sc2), scores=sc2, cfg=cfg)
        self.assertEqual(C.funding_chooser(ctx2), ["PUSDT", "QUSDT", "RUSDT"])  # K cap, ties by name

    def test_passive_chooser(self):
        C = _carry()
        XS = _xs()
        ctx = XS.Ctx(day=ds(5), tranche=0, held=frozenset(), universe=[], scores={}, cfg=C.CarryConfig(W=3, K=2, theta_in=0.0))
        self.assertEqual(C.passive_chooser(ctx), ["BTCUSDT", "ETHUSDT"])


# --------------------------------------------------------------------------- §11.5 eligibility
class TestEligibility(unittest.TestCase):
    def test_needs_perp_funding_and_a_complete_perp_day(self):
        C = _carry()
        spot = {s: arr15([100.0] * 12, vol=1.0 + i) for i, s in enumerate(("AUSDT", "BUSDT", "CUSDT", "DUSDT"))}
        perp = {"AUSDT": arr15([100.0] * 12), "BUSDT": arr15([100.0] * 12),
                "DUSDT": arr15([100.0] * 12, bars={9: 40})}                         # D: day d-1 incomplete
        fund = {"AUSDT": events(range(12)), "DUSDT": events(range(12)), "CUSDT": events(range(12))}
        m = market(spot, perp, fund)
        self.assertEqual(C.eligible(m, ds(10), cfg1(W=3)), ["AUSDT"])          # B no funding, C no perp


# --------------------------------------------------------------------------- §11.6 money
class TestMoney(unittest.TestCase):
    def test_one_pair_without_costs_by_hand(self):
        C = _carry()
        free = C.Fees(0.0, 0.0, 0.0)
        r = C.simulate(one_pair([100.0] * 6), cfg1(), free, ds(1), ds(4), chooser=lambda ctx: ["AUSDT"], start_equity=10_000.0)
        f = 50 * 100.0 * 0.0001                         # notional C/2 = 5000 -> 50 coins
        self.assertEqual([d for d, _ in r["series"]], [ds(1), ds(2), ds(3)])
        self.assertAlmostEqual(r["series"][0][1], 3 * f / 10_000.0 * 100, places=9)
        self.assertAlmostEqual(r["series"][1][1], 3 * f / (10_000.0 + 3 * f) * 100, places=9)
        self.assertAlmostEqual(r["funding_income"], 9 * f, places=9)    # entry 00:15 day 1 .. edge 00:00 day 4
        self.assertAlmostEqual(r["final_equity"], 10_000.0 + 9 * f, places=6)
        self.assertEqual(r["trades"][0]["reason"], "flush")
        self.assertTrue(r["conservation_ok"])

    def test_one_pair_with_costs_on_both_legs(self):
        C = _carry()
        fees = C.Fees(0.001, 0.0005, 0.0008)
        r = C.simulate(one_pair([100.0] * 6), cfg1(), fees, ds(1), ds(4), chooser=lambda ctx: ["AUSDT"], start_equity=10_000.0)
        q = 5000.0 / (100.0 * 1.0008) * (1 - 0.001)
        perp_in, perp_out = 100.0 * (1 - 0.0008), 100.0 * (1 + 0.0008)
        expected = (10_000.0 - 5000.0 + q * 100.0 * (1 - 0.0008) * (1 - 0.001)          # spot round trip
                    + q * (perp_in - perp_out)                                          # short round trip
                    - q * perp_in * 0.0005 - q * perp_out * 0.0005                      # perp fees
                    + 9 * q * 100.0 * 0.0001)                                           # funding
        self.assertAlmostEqual(r["final_equity"], expected, places=6)
        self.assertTrue(r["conservation_ok"])

    def test_margin_stop(self):
        C = _carry()
        free = C.Fees(0.0, 0.0, 0.0)
        r = C.simulate(one_pair([100.0, 100.0, 160.0, 160.0, 160.0, 160.0]), cfg1(), free, ds(1), ds(4),
                       chooser=lambda ctx: ["AUSDT"], start_equity=10_000.0)
        t = [x for x in r["trades"] if x["reason"] == "margin_stop"]
        self.assertEqual(len(t), 1)
        self.assertEqual(t[0]["exit_day"], ds(3))
        self.assertEqual(r["margin_stops"], 1)
        f = 3 * 50 * 100.0 * 0.0001 + 3 * 50 * 160.0 * 0.0001          # day-1 periods at 100, day-2 periods at 160
        self.assertAlmostEqual(r["final_equity"], 8000.0 + 2000.0 + f, places=6)
        self.assertEqual(len(r["trades"]), 1)                           # no re-entry the same day
        self.assertTrue(r["conservation_ok"])

    def test_liquidation(self):
        C = _carry()
        free = C.Fees(0.0, 0.0, 0.0)
        prices = [100.0, 100.0, 200.0, 200.0, 200.0, 200.0]
        r = C.simulate(one_pair(prices), cfg1(), free, ds(1), ds(4), chooser=lambda ctx: ["AUSDT"], start_equity=10_000.0)
        t = [x for x in r["trades"] if x["reason"] == "liquidation"]
        self.assertEqual(len(t), 1)
        self.assertEqual(t[0]["exit_day"], ds(2))
        self.assertEqual(r["liquidations"], 1)
        self.assertAlmostEqual(C.liq_price(100.0, 1), 100.0 * (1 + 1 - C.MMR))
        f = 3 * 50 * 100.0 * 0.0001                                    # day-1 periods only: none on the liquidation day
        self.assertAlmostEqual(r["final_equity"], 50 * 200.0 + 0.0 + f, places=6)   # margin lost, spot sold at 200
        self.assertTrue(r["conservation_ok"])


# --------------------------------------------------------------------------- §11.7 controls
class TestControls(unittest.TestCase):
    def _setup(self):
        C = _carry()
        rng = np.random.default_rng(3)
        names = [f"{c}USDT" for c in "ABCDEFGH"]
        cols = {s: arr15(list(100 * np.cumprod(1 + rng.normal(0, 0.01, 60))), vol=1.0 + i) for i, s in enumerate(names)}
        fund = {s: events(range(60), rate=lambda d, h, i=i: 0.00005 * (1 + (d + i) % 4)) for i, s in enumerate(names)}
        m = market(cols, cols, fund)
        cfg = C.CarryConfig(W=3, K=3, theta_in=0.05, liq_min_days=20)
        return C, m, cfg, C.simulate(m, cfg, C.fees(1), ds(40), ds(60))

    def test_random_matched_same_moves_and_deterministic(self):
        XS = _xs()
        C, m, cfg, base = self._setup()
        self.assertGreater(len(base["trades"]), 0)
        a = C.simulate(m, cfg, C.fees(1), ds(40), ds(60), chooser=XS.matched_random(base["moves"], seed=0))
        b = C.simulate(m, cfg, C.fees(1), ds(40), ds(60), chooser=XS.matched_random(base["moves"], seed=0))
        self.assertEqual(a["moves"], base["moves"])
        self.assertEqual(a["series"], b["series"])
        rc = C.random_control(m, cfg, C.fees(1), ds(40), ds(60), base["moves"], n=3)
        self.assertEqual(len(rc), 3)


# --------------------------------------------------------------------------- §11.8 procedure
class TestProcedure(unittest.TestCase):
    def test_candidates_and_passive(self):
        import experiment_fc as FC
        c = FC.candidates()
        names = {f"FC_W{W}_K{K}_T{T}" for W in (3, 14) for K in (5, 10) for T in (10, 25)}
        self.assertEqual(set(c), names)
        for n, cfg in c.items():
            self.assertEqual(cfg.name, n)
            self.assertAlmostEqual(cfg.theta_out, cfg.theta_in / 2)
            self.assertEqual((cfg.lev, cfg.passive), (1, False))
        p = FC.passive_config()
        self.assertEqual((p.name, p.passive, p.K), ("FC_PASSIVE", True, 2))

    def test_constants(self):
        import experiment_fc as FC
        self.assertEqual((FC.TRAIN_FROM, FC.TEST_FROM), ("2023-03-01", "2026-01-01"))
        self.assertEqual((FC.N_RANDOM, FC.MIN_TRADES, FC.PBO_MAX, FC.DSR_MIN, FC.N_TRIALS), (50, 50, 0.5, 0.90, 8))
        self.assertEqual(FC.START_EQUITY, 1_000_000.0)
        self.assertEqual(FC.TEST_MARKER.name, "FC_TEST_OPENED.json")

    def _res(self, **kw):
        r = {"mean_pct": 0.02, "r_by_year": {"2023": 0.03, "2024": 0.01, "2025": -0.01},
             "random_means": [0.0] * 45 + [0.01] * 5, "paired_passive": {"mean_pct": 0.005}, "liquidations": 0, "n_trades": 60}
        r.update(kw)
        return r

    def test_admission(self):
        import experiment_fc as FC
        self.assertTrue(FC.admit(self._res())[0])
        for kw in ({"mean_pct": -0.01}, {"r_by_year": {"2023": 0.03, "2024": -0.01, "2025": -0.01}}, {"mean_pct": 0.01},
                   {"paired_passive": {"mean_pct": -0.001}}, {"liquidations": 1}, {"n_trades": 49}):
            self.assertFalse(FC.admit(self._res(**kw))[0], kw)
        self.assertTrue(FC.admit_passive({"mean_pct": 0.01, "r_by_year": {"2023": 0.0, "2024": 0.01, "2025": -0.1}, "liquidations": 0}))
        self.assertFalse(FC.admit_passive({"mean_pct": 0.0, "r_by_year": {"2023": 0.1, "2024": 0.1, "2025": 0.1}, "liquidations": 0}))
        self.assertFalse(FC.admit_passive({"mean_pct": 0.01, "r_by_year": {"2023": 0.1, "2024": 0.1, "2025": 0.1}, "liquidations": 1}))

    def test_winner_dsr_and_hypotheses(self):
        import experiment_fc as FC
        res = {"FC_W3_K5_T10": {"admitted": True, "mean_pct": 0.01}, "FC_W14_K10_T25": {"admitted": True, "mean_pct": 0.02},
               "FC_W3_K10_T25": {"admitted": False, "mean_pct": 0.09}}
        self.assertEqual(FC.select_winner(res, 0.3), "FC_W14_K10_T25")
        self.assertIsNone(FC.select_winner(res, 0.5))
        self.assertTrue(FC.dsr_ok(0.9))
        self.assertFalse(FC.dsr_ok(0.89))
        self.assertEqual(FC.hypotheses("FC_W3_K5_T10", True), ["FC_W3_K5_T10", "FC_PASSIVE"])
        self.assertEqual(FC.hypotheses(None, True), ["FC_PASSIVE"])
        self.assertEqual(FC.hypotheses(None, False), [])
        self.assertAlmostEqual(FC.alpha_for(2), 0.025)

    def test_stop_and_test_opened_once(self):
        import experiment_fc as FC
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d) / "FC_TEST_OPENED.json"
            called = []
            out = FC.pipeline(lambda: [], lambda names, a: called.append((names, a)), marker)
            self.assertTrue(out["stop"])
            self.assertFalse(marker.exists())
            FC.pipeline(lambda: ["FC_PASSIVE"], lambda names, a: called.append((names, a)), marker)
            self.assertEqual(called, [(["FC_PASSIVE"], 0.05)])
            with self.assertRaises(RuntimeError):
                FC.pipeline(lambda: ["FC_PASSIVE"], lambda names, a: None, marker)
        self.assertEqual(FC.main(["test"]), 2)                         # no --operator-approved

    def test_verdicts(self):
        import experiment_fc as FC
        ok = {"mean_pct": 0.02, "lo95": 0.005, "max_dd_pct": -3.0}
        self.assertEqual(FC.verdict("fc", ok, random_means=[0.0] * 50, liquidations=0)["outcome"], "accepted")
        self.assertEqual(FC.verdict("fc", ok, random_means=[0.05] * 50, liquidations=0)["outcome"], "rejected")
        self.assertEqual(FC.verdict("fc", ok, random_means=[0.0] * 50, liquidations=1)["outcome"], "rejected")
        self.assertEqual(FC.verdict("fc", dict(ok, max_dd_pct=-12.0), random_means=[0.0] * 50, liquidations=0)["outcome"], "rejected")
        self.assertEqual(FC.verdict("passive", ok, liquidations=0)["outcome"], "accepted")
        self.assertEqual(FC.verdict("passive", dict(ok, lo95=-0.001), liquidations=0)["outcome"], "rejected")

    def test_train_panel_has_no_test_days(self):
        import experiment_fc as FC
        XS = _xs()
        p = FC.train_panel(panel({"AUSDT": arr15([100.0] * 10, base=datetime(2025, 12, 26, tzinfo=timezone.utc))}))
        self.assertLess(XS.day_str(p.days[-1]), FC.TEST_FROM)


# --------------------------------------------------------------------------- §11.9 demo readiness (F0)
class TestDemoReadiness(unittest.TestCase):
    GOOD = {"account_ok": True, "futures_permission": True, "futures_balance_usdt": 150.0, "one_way_mode": True,
            "isolated_ok": True, "leverage1_ok": True, "commission": {"taker": 0.0005, "maker": 0.0002},
            "order_test_ok": True, "lot_match_ok": True, "funding_income_seen": None, "demo_vs_live_funding_gap_bp": 0.3}

    def test_go_no_go_and_warnings(self):
        import demo_futures as DF
        r = DF.readiness(dict(self.GOOD))
        self.assertEqual(r["status"], "GO")
        self.assertEqual(r["missing"], [])
        self.assertTrue(any("funding_income" in w for w in r["warnings"]))         # not verifiable without a position
        for k, bad in (("futures_permission", False), ("futures_balance_usdt", 50.0), ("commission", None),
                       ("order_test_ok", False), ("lot_match_ok", False)):
            r = DF.readiness(dict(self.GOOD, **{k: bad}))
            self.assertEqual(r["status"], "NO_GO", k)
            self.assertIn(k, r["missing"])
        missing_key = dict(self.GOOD)
        del missing_key["isolated_ok"]
        self.assertIn("isolated_ok", DF.readiness(missing_key)["missing"])          # not checked = not verified
        r = DF.readiness(dict(self.GOOD, demo_vs_live_funding_gap_bp=4.4))
        self.assertEqual(r["status"], "GO")
        self.assertTrue(any("demo_vs_live_funding_gap_bp" in w for w in r["warnings"]))
        self.assertEqual((DF.MIN_FUTURES_BALANCE, DF.FUNDING_GAP_WARN_BP), (100.0, 1.0))


# --------------------------------------------------------------------------- §11.10 hosts
class TestHosts(unittest.TestCase):
    def test_signed_demo_only_and_public_futures_data(self):
        import exchange_client as X
        self.assertIn("demo-fapi.binance.com", X.ALLOWED_HOSTS)
        self.assertTrue(all(h.startswith("demo-") for h in X.ALLOWED_HOSTS))
        self.assertTrue({"fapi.binance.com", "data.binance.vision"} <= set(X.PUBLIC_HOSTS))


# --------------------------------------------------------------------------- §11.11 boundary
class TestBoundary(unittest.TestCase):
    def test_bot_does_not_import_fc_modules(self):
        for f in (ROOT / "files").glob("*.py"):
            txt = f.read_text(encoding="utf-8", errors="replace")
            for n in ("carry", "funding_store", "experiment_fc", "demo_futures"):
                self.assertNotIn(f"import {n}", txt, f"{f.name} imports {n}")

    def test_outputs_in_trader_runtime(self):
        import experiment_fc as FC
        import trader_config as TC
        FS = _fs()
        for p in (FC.OUT_DIR, FC.TEST_MARKER.parent, FS.FUNDING_DIR, FS.PERP_DIR):
            self.assertTrue(str(p).startswith(str(TC.RUNTIME)), p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
