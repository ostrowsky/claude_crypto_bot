"""FC_PASSIVE next step -- tests written from fc-passive-next-1022-spec.md §D BEFORE the code
(operator rule 2026-10-10: spec -> tests -> code). Synthetic data and fake transports, offline.

Modules are imported inside each test, so before the code exists every test fails on its own."""
import io
import json
import random
import tempfile
import unittest
import urllib.parse
from datetime import timedelta
from pathlib import Path

import numpy as np

from test_xsmom import BASE, ROOT, arr15, ds
from test_funding_carry import events

HERE = Path(__file__).resolve().parent
DAY = timedelta(days=1)
HOUR = timedelta(hours=1)
DAY_MS, HOUR_MS = 86_400_000, 3_600_000
BASE_MS = int(BASE.timestamp() * 1000)
N_DAYS = 46            # the store starts 35 days before S; the liquidity filter needs 20 days of history
S_DAY = 36             # first step at day 35 10:00 UTC -> S = day 36


def _shadow():
    import fc_shadow
    return fc_shadow


def data_two_pairs(n_days=N_DAYS):
    """BTC and ETH: spot and perp 15m bars drifting +0.2 %/day, funding 0.0001 at 00, 08, 16 UTC."""
    out = {}
    for sym, p0 in (("BTCUSDT", 100.0), ("ETHUSDT", 50.0)):
        prices = [p0 * (1 + 0.002 * k) for k in range(n_days)]
        out[sym] = {"spot": arr15(prices), "perp": arr15(prices), "funding": events(range(n_days))}
    return out


def fetch_of(data):
    return lambda *a, **k: data


# --------------------------------------------------------------------------- A. shadow book
class TestShadowBook(unittest.TestCase):
    def test_start_day_is_the_first_full_utc_day(self):
        S = _shadow()
        self.assertEqual(S.start_day(BASE + 10 * HOUR), ds(1))
        self.assertEqual(S.start_day(BASE), ds(0))
        self.assertEqual(S.START_EQUITY, 10_000.0)

    def test_series_equals_carry_simulate_start_is_kept_catch_up(self):
        S = _shadow()
        import carry as C
        import experiment_fc as FC
        data = data_two_pairs()
        with tempfile.TemporaryDirectory() as d:
            st, led = Path(d) / "state.json", Path(d) / "ledger.jsonl"
            S.step(BASE + 35 * DAY + 10 * HOUR, fetch_of(data), state_path=st, ledger_path=led, enabled=True)
            first = json.loads(st.read_text(encoding="utf-8"))
            self.assertEqual(first["start_day"], ds(S_DAY))
            self.assertEqual(first["series"], [])
            # the machine slept for a week: one step catches up every missed day
            out = S.step(BASE + 42 * DAY + 12 * HOUR, fetch_of(data), state_path=st, ledger_path=led, enabled=True)
            state = json.loads(st.read_text(encoding="utf-8"))
        self.assertEqual(state["start_day"], ds(S_DAY))
        ref = C.simulate(S.market_from(data), FC.passive_config(), C.fees(1), ds(S_DAY), ds(42),
                         start_equity=S.START_EQUITY)
        self.assertEqual([tuple(x) for x in state["series"]], [tuple(x) for x in ref["series"]])
        self.assertEqual(len(state["series"]), 6)
        self.assertEqual(out["days"], 6)
        self.assertGreater(state["funding_income"], 0)                    # the book really holds the carry
        self.assertAlmostEqual(state["funding_income"], ref["funding_income"], places=9)
        self.assertEqual(state["liquidations"], ref["liquidations"])

    def test_window_edge_flush_is_not_a_closed_trade(self):
        S = _shadow()
        data = data_two_pairs()
        with tempfile.TemporaryDirectory() as d:
            st, led = Path(d) / "state.json", Path(d) / "ledger.jsonl"
            S.step(BASE + 35 * DAY + 10 * HOUR, fetch_of(data), state_path=st, ledger_path=led, enabled=True)
            S.step(BASE + 42 * DAY + 1 * HOUR, fetch_of(data), state_path=st, ledger_path=led, enabled=True)
            state = json.loads(st.read_text(encoding="utf-8"))
            rows = [json.loads(x) for x in led.read_text(encoding="utf-8").splitlines()] if led.exists() else []
        self.assertEqual(state["n_trades"], 0)                            # passive: nothing closes inside the window
        self.assertFalse([r for r in rows if r.get("type") == "trade"])

    def test_idempotent_and_flag(self):
        S = _shadow()
        data = data_two_pairs()
        with tempfile.TemporaryDirectory() as d:
            st, led = Path(d) / "state.json", Path(d) / "ledger.jsonl"
            now = BASE + 41 * DAY + 1 * HOUR
            S.step(BASE + 35 * DAY + 10 * HOUR, fetch_of(data), state_path=st, ledger_path=led, enabled=True)
            S.step(now, fetch_of(data), state_path=st, ledger_path=led, enabled=True)
            s1 = json.loads(st.read_text(encoding="utf-8"))
            l1 = led.read_text(encoding="utf-8") if led.exists() else ""
            S.step(now, fetch_of(data), state_path=st, ledger_path=led, enabled=True)
            s2 = json.loads(st.read_text(encoding="utf-8"))
            l2 = led.read_text(encoding="utf-8") if led.exists() else ""
        for k in ("start_day", "series", "marks", "n_trades", "liquidations", "margin_stops", "funding_income", "anomalies"):
            self.assertEqual(s1[k], s2[k], k)
        self.assertEqual(l1, l2)
        with tempfile.TemporaryDirectory() as d:
            st = Path(d) / "state.json"
            out = S.step(BASE + 35 * DAY, fetch_of(data), state_path=st, ledger_path=Path(d) / "l.jsonl", enabled=False)
            self.assertTrue(out["skipped"])
            self.assertFalse(st.exists())
        import trader_config as TC
        self.assertIs(TC.FC_SHADOW_ENABLED, True)

    def test_data_checks(self):
        S = _shadow()
        lo, hi = ds(S_DAY), ds(43)
        self.assertEqual(S.data_checks(data_two_pairs(), lo, hi), [])

        gap = data_two_pairs()                                            # 2 h without perp bars on day 38
        a = gap["BTCUSDT"]["perp"]
        t0 = BASE_MS + 38 * DAY_MS + 10 * HOUR_MS
        keep = (a["t"] < t0) | (a["t"] >= t0 + 2 * HOUR_MS)
        gap["BTCUSDT"]["perp"] = {k: v[keep] for k, v in a.items()}
        self.assertIn("bar_gap", {x["kind"] for x in S.data_checks(gap, lo, hi)})

        fg = data_two_pairs()                                             # 24 h between funding events (limit 12 h)
        f = fg["ETHUSDT"]["funding"]
        drop = (f["t"] > BASE_MS + 39 * DAY_MS) & (f["t"] < BASE_MS + 40 * DAY_MS)
        fg["ETHUSDT"]["funding"] = {k: v[~drop] for k, v in f.items()}
        self.assertIn("funding_gap", {x["kind"] for x in S.data_checks(fg, lo, hi)})

        pr = data_two_pairs()                                             # perp close 10% above spot on day 40
        p = pr["BTCUSDT"]["perp"]
        day40 = (p["t"] >= BASE_MS + 40 * DAY_MS) & (p["t"] < BASE_MS + 41 * DAY_MS)
        pr["BTCUSDT"]["perp"] = dict(p, c=np.where(day40, p["c"] * 1.10, p["c"]))
        self.assertIn("price_ratio", {x["kind"] for x in S.data_checks(pr, lo, hi)})

        outside = data_two_pairs()                                        # a hole before S is not this book's anomaly
        a = outside["BTCUSDT"]["spot"]
        t0 = BASE_MS + 10 * DAY_MS
        keep = (a["t"] < t0) | (a["t"] >= t0 + 3 * HOUR_MS)
        outside["BTCUSDT"]["spot"] = {k: v[keep] for k, v in a.items()}
        self.assertEqual(S.data_checks(outside, lo, hi), [])

    def test_liquidation_is_an_anomaly_margin_stop_is_not(self):
        S = _shadow()
        self.assertIn("liquidation", {x["kind"] for x in S.book_checks({"liquidations": 1, "margin_stops": 0})})
        self.assertEqual(S.book_checks({"liquidations": 0, "margin_stops": 2}), [])

    def _state(self, vals, liq=0, anomalies=None):
        return {"start_day": ds(0), "series": [[ds(i), v] for i, v in enumerate(vals)], "liquidations": liq,
                "anomalies": anomalies or []}

    def test_readout_statuses(self):
        S = _shadow()
        rng = random.Random(4)
        self.assertEqual(S.MIN_DAYS, 90)
        pos = [0.01 + rng.gauss(0, 0.002) for _ in range(95)]           # 95% interval [+0.0096, +0.0104]
        neg = [-0.01 + rng.gauss(0, 0.002) for _ in range(95)]          # [-0.0100, -0.0093]
        noisy = [rng.gauss(0, 0.05) for _ in range(95)]                 # [-0.0084, +0.0078]
        self.assertEqual(S.readout(self._state([0.01] * 89))["status"], "COLLECTING")
        self.assertEqual(S.readout(self._state(pos))["status"], "KEEP")
        self.assertEqual(S.readout(self._state(neg))["status"], "REJECT")
        self.assertEqual(S.readout(self._state(noisy))["status"], "INCONCLUSIVE")
        self.assertEqual(S.readout(self._state(pos, liq=1))["status"], "REJECT")
        self.assertEqual(S.readout(self._state(pos, anomalies=[{"kind": "bar_gap"}]))["status"], "ANOMALY")
        self.assertEqual(S.readout(self._state([0.01] * 30, anomalies=[{"kind": "funding_gap"}]))["status"], "ANOMALY")
        r = S.readout(self._state(pos))
        self.assertEqual(r["n_days"], 95)
        self.assertIn("lo95", r["agg"])

    def test_boundary_and_wiring(self):
        S = _shadow()
        import trader_config as TC
        src = (HERE / "fc_shadow.py").read_text(encoding="utf-8")
        self.assertNotIn("load_credentials", src)
        self.assertNotIn("signed=True", src)
        root = TC.RUNTIME / "fc_shadow"
        self.assertEqual(Path(S.FC_SHADOW_DIR), root)
        for p in (S.DATA_DIR, S.STATE, S.LEDGER, S.READOUT):
            self.assertTrue(Path(p).resolve().is_relative_to(root.resolve()), p)
        cmd = (HERE / "run_shadow.cmd").read_text(encoding="utf-8")
        self.assertLess(cmd.index("shadow_portfolio.py step"), cmd.index("fc_shadow.py step"))
        tail = cmd[cmd.index("fc_shadow.py step"):]
        self.assertIn("errorlevel", tail.lower())                        # its own check, E7 is not blocked by it
        self.assertIn("FC SHADOW STEP FAILED", tail)


# --------------------------------------------------------------------------- B. F0
SIGNED_FUT = {"fut_account", "fut_balance", "fut_position_side", "fut_set_position_side", "fut_margin_type",
              "fut_leverage", "fut_commission", "fut_test_order", "fut_income"}
DEMO_RATE = {"BTCUSDT": "0.00002103", "ETHUSDT": "0.00002057"}
LIVE_RATE = {"BTCUSDT": "0.00002262", "ETHUSDT": "0.00006497"}
MARK = {"BTCUSDT": "84000", "ETHUSDT": "2500"}


def perp_info(sym, step, min_qty, min_notional):
    return {"symbol": sym, "filters": [{"filterType": "LOT_SIZE", "stepSize": step, "minQty": min_qty},
                                       {"filterType": "MARKET_LOT_SIZE", "stepSize": step, "minQty": min_qty},
                                       {"filterType": "MIN_NOTIONAL", "notional": min_notional}]}


def spot_info(sym, step, min_qty, min_notional):
    return {"symbol": sym, "filters": [{"filterType": "LOT_SIZE", "stepSize": step, "minQty": min_qty},
                                       {"filterType": "NOTIONAL", "minNotional": min_notional}]}


PERP_X = {"symbols": [perp_info("BTCUSDT", "0.001", "0.001", "100"), perp_info("ETHUSDT", "0.001", "0.001", "20")]}
SPOT_X = {"symbols": [spot_info("BTCUSDT", "0.00001", "0.00001", "5"), spot_info("ETHUSDT", "0.0001", "0.0001", "5")]}


class Rec:
    """Fake futures client: records method calls, returns canned answers."""

    def __init__(self, dual=False, account_error=None, margin_error=None):
        self.calls, self.orders = [], []
        self.dual, self.account_error, self.margin_error = dual, account_error, margin_error

    def fut_time(self):
        self.calls.append("fut_time")
        return {"serverTime": 1}

    def fut_account(self):
        self.calls.append("fut_account")
        if self.account_error:
            raise self.account_error
        return {"canTrade": True}

    def fut_balance(self):
        self.calls.append("fut_balance")
        return [{"asset": "USDT", "balance": "150.0", "availableBalance": "150.0"}]

    def fut_position_side(self):
        self.calls.append("fut_position_side")
        return {"dualSidePosition": self.dual}

    def fut_set_position_side(self, dual):
        self.calls.append("fut_set_position_side")
        return {"code": 200}

    def fut_margin_type(self, symbol, margin_type):
        self.calls.append("fut_margin_type")
        if self.margin_error:
            raise self.margin_error
        return {"code": 200}

    def fut_leverage(self, symbol, leverage):
        self.calls.append("fut_leverage")
        return {"symbol": symbol, "leverage": leverage}

    def fut_commission(self, symbol):
        self.calls.append("fut_commission")
        return {"symbol": symbol, "makerCommissionRate": "0.0002", "takerCommissionRate": "0.0005"}

    def fut_test_order(self, symbol, side, type_, **kw):
        self.calls.append("fut_test_order")
        self.orders.append(dict(symbol=symbol, side=side, type=type_, **kw))
        return {}

    def fut_income(self, income_type, limit=10):
        self.calls.append("fut_income")
        return []

    def futures_exchange_info(self, base=None):
        self.calls.append("futures_exchange_info")
        return PERP_X

    def premium_index(self, symbol, base=None):
        self.calls.append("premium_index")
        demo = base is not None and "demo-fapi" in base
        return {"symbol": symbol, "lastFundingRate": (DEMO_RATE if demo else LIVE_RATE)[symbol], "markPrice": MARK[symbol]}


class SpotPub:
    def __init__(self):
        self.calls = []

    def exchange_info(self, *a, **k):
        self.calls.append("exchange_info")
        return SPOT_X


class FakeResp(io.BytesIO):
    def __init__(self, body):
        super().__init__(json.dumps(body).encode())
        self.headers = {}
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def routed_opener(seen):
    """One fake transport for every client: answers by method and path, records (host, method, path, signed)."""
    def opener(req, timeout=None):
        u = urllib.parse.urlparse(req.full_url)
        q = urllib.parse.parse_qs(u.query or (req.data or b"").decode())
        host, path, method = u.hostname, u.path, req.get_method()
        seen.append((host, method, path, "signature" in q))
        demo = host == "demo-fapi.binance.com"
        sym = (q.get("symbol") or ["BTCUSDT"])[0]
        body = {("GET", "/fapi/v1/time"): {"serverTime": 1_800_000_000_000},
                ("GET", "/fapi/v2/account"): {"canTrade": True},
                ("GET", "/fapi/v2/balance"): [{"asset": "USDT", "balance": "150.0", "availableBalance": "150.0"}],
                ("GET", "/fapi/v1/positionSide/dual"): {"dualSidePosition": False},
                ("POST", "/fapi/v1/marginType"): {"code": 200, "msg": "success"},
                ("POST", "/fapi/v1/leverage"): {"symbol": sym, "leverage": 1},
                ("GET", "/fapi/v1/commissionRate"): {"symbol": sym, "makerCommissionRate": "0.0002",
                                                     "takerCommissionRate": "0.0005"},
                ("POST", "/fapi/v1/order/test"): {},
                ("GET", "/fapi/v1/income"): [],
                ("GET", "/fapi/v1/exchangeInfo"): PERP_X,
                ("GET", "/fapi/v1/premiumIndex"): {"symbol": sym, "markPrice": MARK[sym],
                                                   "lastFundingRate": (DEMO_RATE if demo else LIVE_RATE)[sym]},
                ("GET", "/api/v3/exchangeInfo"): SPOT_X}.get((method, path))
        if body is None:
            raise AssertionError(f"F0 called an unlisted endpoint: {method} {host}{path}")
        return FakeResp(body)
    return opener


F0_ENDPOINTS = {("GET", "/fapi/v1/time"), ("GET", "/fapi/v2/account"), ("GET", "/fapi/v2/balance"),
                ("GET", "/fapi/v1/positionSide/dual"), ("POST", "/fapi/v1/positionSide/dual"),
                ("POST", "/fapi/v1/marginType"), ("POST", "/fapi/v1/leverage"), ("GET", "/fapi/v1/commissionRate"),
                ("POST", "/fapi/v1/order/test"), ("GET", "/fapi/v1/income"), ("GET", "/fapi/v1/exchangeInfo"),
                ("GET", "/fapi/v1/premiumIndex"), ("GET", "/api/v3/exchangeInfo")}


class TestF0(unittest.TestCase):
    def test_without_go_no_request(self):
        import demo_futures as DF
        fut, spot = Rec(), SpotPub()
        out = DF.run_f0(go=False, fut=fut, spot=spot)
        self.assertEqual(out["status"], "PLAN")
        self.assertEqual((fut.calls, spot.calls), ([], []))

    def test_go_happy_path(self):
        import demo_futures as DF
        import trader_config as TC
        self.assertEqual(Path(DF.F0_DIR), TC.RUNTIME / "futures")
        self.assertEqual(DF.LOT_NOTIONAL_USDT, 2500.0)                          # 10 000 / 2 pairs / (1 + 1/L), L = 1
        fut, spot = Rec(), SpotPub()
        with tempfile.TemporaryDirectory() as d:
            out = DF.run_f0(go=True, fut=fut, spot=spot, out_dir=Path(d))
            f = Path(out["file"])
            self.assertTrue(f.exists())
            self.assertEqual(f.parent, Path(d))
            self.assertTrue(f.name.startswith("F0_"))
            saved = json.loads(f.read_text(encoding="utf-8"))
        self.assertEqual(saved["readiness"]["status"], "GO")
        self.assertNotIn("fut_set_position_side", fut.calls)                     # already one-way: not touched
        self.assertEqual([(o["symbol"], o["side"], o["type"], o["quantity"]) for o in fut.orders],
                         [("BTCUSDT", "SELL", "MARKET", "0.002")])              # smallest valid qty at 84 000
        res = out["results"]
        self.assertTrue(res["lot_match_ok"])
        self.assertEqual(res["futures_balance_usdt"], 150.0)
        self.assertIsNone(res["funding_income_seen"])                            # no rows yet
        self.assertAlmostEqual(res["demo_vs_live_funding_gap_bp"], abs(0.00002057 - 0.00006497) * 10_000, places=9)

    def test_go_end_to_end_endpoints_signing_and_no_real_order(self):
        import demo_futures as DF
        import exchange_client as X
        seen = []
        op = routed_opener(seen)
        fut = X.ExchangeClient(base=X.DEMO_FAPI, key="k", secret="s", opener=op, audit=None)
        spot = X.ExchangeClient(key="k", secret="s", opener=op, audit=None)
        with tempfile.TemporaryDirectory() as d:
            out = DF.run_f0(go=True, fut=fut, spot=spot, out_dir=Path(d))
        self.assertEqual(out["readiness"]["status"], "GO", out["readiness"])
        calls = {(m, p) for _, m, p, _ in seen}
        self.assertTrue(calls <= F0_ENDPOINTS, calls - F0_ENDPOINTS)
        self.assertNotIn(("POST", "/fapi/v1/order"), calls)
        self.assertIn(("POST", "/fapi/v1/order/test"), calls)
        self.assertEqual({h for h, _, _, signed in seen if signed}, {"demo-fapi.binance.com"})
        self.assertIn("fapi.binance.com", {h for h, _, p, _ in seen if p == "/fapi/v1/premiumIndex"})  # live reference

    def test_hedge_mode_is_switched_and_4046_is_ok(self):
        import demo_futures as DF
        import exchange_client as X
        fut = Rec(dual=True, margin_error=X.ExchangeError(400, -4046, "No need to change margin type.", "/fapi/v1/marginType"))
        with tempfile.TemporaryDirectory() as d:
            out = DF.run_f0(go=True, fut=fut, spot=SpotPub(), out_dir=Path(d))
        self.assertIn("fut_set_position_side", fut.calls)
        self.assertTrue(out["results"]["one_way_mode"])
        self.assertTrue(out["results"]["isolated_ok"])

    def test_permission_error_stops_signed_calls(self):
        import demo_futures as DF
        import exchange_client as X
        for status, code in ((401, -2015), (400, -2014), (400, -2008), (401, None)):
            fut = Rec(account_error=X.ExchangeError(status, code, "Invalid API-key, IP, or permissions for action.",
                                                    "/fapi/v2/account"))
            with tempfile.TemporaryDirectory() as d:
                out = DF.run_f0(go=True, fut=fut, spot=SpotPub(), out_dir=Path(d))
            self.assertEqual(out["readiness"]["status"], "NO_GO")
            self.assertIs(out["results"]["futures_permission"], False)
            self.assertIn("futures_permission", out["readiness"]["missing"])
            after = fut.calls[fut.calls.index("fut_account") + 1:]
            self.assertFalse(set(after) & SIGNED_FUT, after)                     # nothing signed after the refusal

    def test_lot_match_order_qty_gap_and_warning(self):
        import demo_futures as DF
        btc_s, btc_p = spot_info("BTCUSDT", "0.00001", "0.00001", "5"), perp_info("BTCUSDT", "0.001", "0.001", "100")
        eth_s, eth_p = spot_info("ETHUSDT", "0.0001", "0.0001", "5"), perp_info("ETHUSDT", "0.001", "0.001", "20")
        ok, _ = DF.lot_match(btc_s, btc_p, notional=2500.0, price=84_000.0)    # spot 0.02976, perp 0.029: < one step
        self.assertTrue(ok)
        below, _ = DF.lot_match(btc_s, btc_p, notional=100.0, price=84_000.0)  # perp 0.001 = 84 USDT < 100
        self.assertFalse(below)
        zero, _ = DF.lot_match(btc_s, btc_p, notional=100.0, price=150_000.0)  # perp floors to 0 < minQty
        self.assertFalse(zero)
        small, _ = DF.lot_match(eth_s, eth_p, notional=10.0, price=2_500.0)    # 10 USDT < perp MIN_NOTIONAL 20
        self.assertFalse(small)
        self.assertEqual(DF.order_test_qty(btc_p, 84_000.0), "0.002")           # >= 100 x 1.05 USDT, on the step
        self.assertEqual(DF.order_test_qty(eth_p, 2_500.0), "0.009")            # 21 / 2500 = 0.0084 -> 0.009
        gap = DF.funding_gap_bp({"BTCUSDT": 0.00002103, "ETHUSDT": 0.00002057}, {"BTCUSDT": 0.00002262, "ETHUSDT": 0.00006497})
        self.assertAlmostEqual(gap, 0.444, places=6)
        r = DF.readiness({"account_ok": True, "futures_permission": True, "futures_balance_usdt": 150.0, "one_way_mode": True,
                          "isolated_ok": True, "leverage1_ok": True, "commission": {"taker": 0.0005, "maker": 0.0002},
                          "order_test_ok": True, "lot_match_ok": True, "funding_income_seen": None,
                          "demo_vs_live_funding_gap_bp": 2.0})
        self.assertEqual(r["status"], "GO")
        self.assertTrue(any("demo_vs_live_funding_gap_bp" in w for w in r["warnings"]))


class TestClientFutures(unittest.TestCase):
    def test_time_sync_path_by_host_and_signed_futures_calls(self):
        import exchange_client as X
        seen = []

        def opener(req, timeout=None):
            seen.append((req.full_url, {k.lower(): v for k, v in req.header_items()}, req.get_method(), req.data))
            return FakeResp({"serverTime": 1_800_000_000_000} if "/time" in req.full_url else {})
        c = X.ExchangeClient(base=X.DEMO_FAPI, key="k", secret="s", opener=opener, audit=None)
        c._sync_time(force=True)
        self.assertIn("demo-fapi.binance.com/fapi/v1/time", seen[-1][0])
        c.fut_account()
        url, headers, method, _ = seen[-1]
        self.assertIn("demo-fapi.binance.com/fapi/v2/account", url)
        self.assertIn("signature=", url)
        self.assertEqual(method, "GET")
        self.assertEqual(headers.get("x-mbx-apikey"), "k")
        c.fut_test_order("BTCUSDT", "SELL", "MARKET", quantity="0.002")
        url, _, method, data = seen[-1]
        self.assertTrue(url.endswith("demo-fapi.binance.com/fapi/v1/order/test"), url)
        self.assertEqual(method, "POST")
        self.assertIn(b"signature=", data)
        # F0 had no real futures order method; fc-demo-execution-1030-spec.md adds fut_order behind _trading_allowed
        self.assertTrue(hasattr(c, "fut_order"))
        spot = X.ExchangeClient(key="k", secret="s", opener=opener, audit=None)
        spot._sync_time(force=True)
        self.assertIn("demo-api.binance.com/api/v3/time", seen[-1][0])           # spot path unchanged


# --------------------------------------------------------------------------- C. interval at 1 - alpha
class TestJudgeInterval(unittest.TestCase):
    def test_judge_uses_the_alpha_interval(self):
        import daily_revenue as DR
        import experiment_fc as FC
        found = None
        for seed in range(500):                                                  # seed 4: 95% lo +0.0238, 97.5% lo -0.0071
            rng = random.Random(seed)
            series = [(ds(i), 0.25 + rng.gauss(0, 1.0)) for i in range(60)]
            if DR.aggregate(series, alpha=0.05)["lo95"] > 0 >= DR.aggregate(series, alpha=0.025)["lo95"]:
                found = series
                break
        self.assertIsNotNone(found, "no series between the two bounds")
        e = {"_series": found, "agg": DR.aggregate(found), "random_means": [], "liquidations": 0}
        v05, v025 = FC.judge("passive", e, 0.05), FC.judge("passive", e, 0.025)
        self.assertTrue(v05["profitable"])
        self.assertFalse(v025["profitable"])
        self.assertEqual((v05["alpha"], v025["alpha"]), (0.05, 0.025))
        self.assertEqual(v025["lo"], DR.aggregate(found, alpha=0.025)["lo95"])

    def test_run_test_uses_judge(self):
        src = (HERE / "experiment_fc.py").read_text(encoding="utf-8")
        body = src[src.index("def run_test"):]
        nxt = body.find("\ndef ", 1)
        body = body if nxt < 0 else body[:nxt]
        self.assertIn("judge(", body)
        self.assertNotIn("verdict(", body)


# --------------------------------------------------------------------------- boundary
class TestBoundary(unittest.TestCase):
    def test_bot_does_not_import_new_modules(self):
        for f in (ROOT / "files").glob("*.py"):
            txt = f.read_text(encoding="utf-8", errors="replace")
            for n in ("fc_shadow", "demo_futures"):
                self.assertNotIn(f"import {n}", txt, f"{f.name} imports {n}")
                self.assertNotIn(f"from {n}", txt, f"{f.name} imports {n}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
