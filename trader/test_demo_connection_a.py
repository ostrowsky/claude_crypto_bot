"""Phase A: exchange client, universe, microstructure, equity (demo-connection-a-1010-spec.md). Offline."""
import io
import json
import types
import unittest
import urllib.error
from decimal import Decimal
from pathlib import Path

import exchange_client as X
import trading_universe as U
import microstructure_recorder as M
import equity_snapshots as EQ

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

FILTERS = {
    "PRICE_FILTER": {"filterType": "PRICE_FILTER", "minPrice": "0.01", "maxPrice": "1000000", "tickSize": "0.01"},
    "LOT_SIZE": {"filterType": "LOT_SIZE", "minQty": "0.00001", "maxQty": "9000", "stepSize": "0.00001"},
    "NOTIONAL": {"filterType": "NOTIONAL", "minNotional": "5", "applyMinToMarket": True,
                 "maxNotional": "9000000", "applyMaxToMarket": False},
}


class FakeResp(io.BytesIO):
    def __init__(self, body, headers=None):
        super().__init__(json.dumps(body).encode())
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestClientSafety(unittest.TestCase):
    def test_mainnet_signed_refused(self):
        with self.assertRaises(X.HostNotAllowed):
            X.ExchangeClient(base="https://api.binance.com", key="k", secret="s")
        c = X.ExchangeClient(key="k", secret="s", opener=lambda *a, **k: FakeResp({}))
        with self.assertRaises(X.HostNotAllowed):
            c._request("GET", "/api/v3/account", signed=True, base="https://api.binance.com")
        with self.assertRaises(X.HostNotAllowed):
            c._request("GET", "/api/v3/time", base="https://evil.example.com")

    def test_signature_vector_from_binance_docs(self):
        q = ("symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1"
             "&recvWindow=5000&timestamp=1499827319559")
        s = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
        self.assertEqual(X.sign(q, s), "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71")

    def test_redact(self):
        t = X.redact("GET /x?a=1&signature=abcdef&b=2 key=KEY123", ("KEY123",))
        self.assertNotIn("abcdef", t)
        self.assertNotIn("KEY123", t)
        self.assertIn("b=2", t)

    def test_orders_disabled_until_phase_c(self):
        c = X.ExchangeClient(key="k", secret="s", cfg=types.SimpleNamespace(TRADING_ENABLED=False, TRADING_MODE="demo"))
        with self.assertRaises(X.TradingDisabled):
            c.place_order("BTCUSDT")
        c.cfg = types.SimpleNamespace(TRADING_ENABLED=True, TRADING_MODE="paper")
        with self.assertRaises(X.TradingDisabled):
            c.place_oco("BTCUSDT")

    def test_signed_request_has_key_header_and_signature(self):
        seen = {}

        def opener(req, timeout=None):
            seen["url"], seen["headers"] = req.full_url, dict(req.headers)
            return FakeResp({"serverTime": 1} if "/time" in req.full_url else {"balances": []})
        c = X.ExchangeClient(key="KEY", secret="SEC", opener=opener)
        c.account()
        self.assertIn("signature=", seen["url"])
        self.assertIn("timestamp=", seen["url"])
        self.assertEqual(seen["headers"].get("X-mbx-apikey"), "KEY")

    def test_timestamp_error_resyncs_once(self):
        calls = {"n": 0}

        def opener(req, timeout=None):
            if "/time" in req.full_url:
                return FakeResp({"serverTime": 1_000_000_000_000})
            calls["n"] += 1
            raise urllib.error.HTTPError(req.full_url, 400, "bad", {},
                                         io.BytesIO(json.dumps({"code": -1021, "msg": "ts"}).encode()))
        c = X.ExchangeClient(key="k", secret="s", opener=opener)
        with self.assertRaises(X.ExchangeError):
            c.account()
        self.assertEqual(calls["n"], 2)        # original + one retry, no loop

    def test_credentials_only_demo_names(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".env"
            p.write_text("BINANCE_API_KEY =live\nBINANCE_DEMO_API_KEY = dk\nBINANCE_DEMO_API_SECRET=ds\n", encoding="utf-8")
            self.assertEqual(X.load_credentials(p), ("dk", "ds"))
            p.write_text("BINANCE_API_KEY=live\nBINANCE_API_SECRET=ls\n", encoding="utf-8")
            k, s = X.load_credentials(p)
            self.assertNotEqual(k, "live")


class TestFilters(unittest.TestCase):
    def test_rounding(self):
        self.assertEqual(X.round_qty(FILTERS, "0.123456789"), Decimal("0.12345"))
        self.assertEqual(X.round_price(FILTERS, "123.4567"), Decimal("123.45"))

    def test_check_order_names_filters(self):
        self.assertEqual(X.check_order(FILTERS, "0.001", "10000"), [])
        self.assertIn("NOTIONAL_MIN", X.check_order(FILTERS, "0.0001", "10000"))
        self.assertIn("LOT_SIZE_STEP", X.check_order(FILTERS, "0.000015", "10000000"))
        self.assertIn("PRICE_FILTER_TICK", X.check_order(FILTERS, "0.001", "10000.005"))

    def test_min_order_qty_passes(self):
        q = X.min_order_qty(FILTERS, "60000")
        self.assertEqual(X.check_order(FILTERS, q, "60000"), [])


def _sym(sym, base, **kw):
    d = {"symbol": sym, "baseAsset": base, "quoteAsset": "USDT", "status": "TRADING", "isSpotTradingAllowed": True,
         "ocoAllowed": True, "orderTypes": ["LIMIT", "LIMIT_MAKER", "MARKET", "STOP_LOSS_LIMIT"], "filters": []}
    d.update(kw)
    return d


class TestUniverse(unittest.TestCase):
    def test_structural_rules(self):
        x = {"symbols": [_sym("BTCUSDT", "BTC"), _sym("JUPUSDT", "JUP"), _sym("BTCUPUSDT", "BTCUP"),
                         _sym("USDCUSDT", "USDC"), _sym("OLDUSDT", "OLD", status="BREAK"),
                         _sym("NOOCOUSDT", "NOOCO", ocoAllowed=False), _sym("ETHBTC", "ETH", quoteAsset="BTC")]}
        u = U.build(x, hist=lambda s: 5000)
        got = {p["symbol"] for p in u["pairs"]}
        self.assertEqual(got, {"BTCUSDT", "JUPUSDT"})          # JUP is not a leveraged token
        self.assertEqual(u["excluded"]["BTCUPUSDT"], "leveraged_token")
        self.assertEqual(u["excluded"]["USDCUSDT"], "stable_or_fiat")
        self.assertEqual(u["excluded"]["OLDUSDT"], "status_break")
        self.assertTrue(all(p["status"] == "liquidity_unmeasured" for p in u["pairs"]))

    def test_needs_history_and_liquidity(self):
        x = {"symbols": [_sym("AUSDT", "A"), _sym("BUSDT", "B"), _sym("CUSDT", "C")]}
        hist = {"AUSDT": 100, "BUSDT": 5000, "CUSDT": 5000}
        liq = {"BUSDT": {"quote_vol": 1e7, "spread_bps": 2}, "CUSDT": {"quote_vol": 1e4, "spread_bps": 50}}
        u = U.build(x, liq, {"min_quote_vol": 1e6, "max_spread_bps": 10}, hist=lambda s: hist[s])
        st = {p["symbol"]: p["status"] for p in u["pairs"]}
        self.assertEqual(st["AUSDT"], "needs_history")
        self.assertEqual(st["BUSDT"], "tradable")
        self.assertTrue(st["CUSDT"].startswith("illiquid:"))

    def test_watchlist_untouched(self):
        src = (HERE / "trading_universe.py").read_text(encoding="utf-8")
        self.assertNotIn("watchlist.json\").write", src)
        self.assertIn('TC.RUNTIME / "trading" / "universe.json"', src)


class TestMicrostructure(unittest.TestCase):
    def test_walk_cost(self):
        asks = [["100.0", "1"], ["101.0", "10"]]
        self.assertAlmostEqual(M.walk_cost(asks, 50, 99.95, "buy"), (100 / 99.95 - 1) * 1e4, places=2)
        c = M.walk_cost(asks, 300, 99.95, "buy")             # walks into the second level
        self.assertGreater(c, (100 / 99.95 - 1) * 1e4)
        self.assertIsNone(M.walk_cost(asks, 10 ** 6, 99.95, "buy"))   # book too thin
        bids = [["99.9", "1"], ["99.0", "5"]]
        self.assertGreater(M.walk_cost(bids, 200, 99.95, "sell"), 0)

    def test_weight_budget_from_exchange_info(self):
        x = {"rateLimits": [{"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 6000}]}
        self.assertEqual(M.weight_budget(x), (600, 4800))


class TestEquity(unittest.TestCase):
    def test_value(self):
        b = [{"asset": "USDT", "free": "100", "locked": "50"}, {"asset": "ETH", "free": "1", "locked": "0"},
             {"asset": "XYZ", "free": "2", "locked": "0"}, {"asset": "ABC", "free": "10", "locked": "0"}]
        mids = {"ETHUSDT": 2000.0, "BTCUSDT": 50000.0, "XYZBTC": 0.001}
        v = EQ.value(b, mids)
        self.assertAlmostEqual(v["equity_usdt"], 150 + 2000 + 2 * 0.001 * 50000)
        self.assertEqual(v["unpriced"], {"ABC": 10.0})


class TestSeparateApp(unittest.TestCase):
    """The trader is a separate application: it must not write into, or be imported by, the signal bot."""

    def test_bot_does_not_import_trader(self):
        bot = ROOT / "files"
        names = ("exchange_client", "trading_policy", "execution_replay", "fill_model", "trading_universe",
                 "microstructure_recorder", "equity_snapshots", "daily_revenue", "trader_config")
        for f in bot.glob("*.py"):
            txt = f.read_text(encoding="utf-8", errors="replace")
            for n in names:
                self.assertNotIn(f"import {n}", txt, f"{f.name} imports {n}")
        cfg = (bot / "config.py").read_text(encoding="utf-8")
        self.assertNotIn("TRADING_ENABLED", cfg)
        self.assertNotIn("MICROSTRUCTURE_RECORDER_ENABLED", cfg)

    def test_trader_writes_only_its_runtime(self):
        import trader_config as TC
        self.assertEqual(TC.RUNTIME, HERE / "runtime")
        for f in HERE.glob("*.py"):
            if f.name.startswith("test_"):
                continue
            txt = f.read_text(encoding="utf-8")
            self.assertNotIn('"' + '.runtime' + '"', txt, f.name)


class TestWiring(unittest.TestCase):
    def test_flags(self):
        import trader_config as TC
        self.assertIn(TC.TRADING_MODE, ("paper", "demo"))           # C2 since 2026-10-20 (operator decision)
        self.assertEqual(TC.TRADING_ENABLED, TC.TRADING_MODE == "demo")
        self.assertTrue(TC.MICROSTRUCTURE_RECORDER_ENABLED)

    def test_hosts_are_code_constants(self):
        import trader_config as TC
        self.assertFalse(hasattr(TC, "TRADING_ALLOW_MAINNET"))
        self.assertTrue(all(h.startswith("demo-") for h in X.ALLOWED_HOSTS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
