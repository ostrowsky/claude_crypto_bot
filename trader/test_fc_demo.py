"""FC-DEMO -- FC_PASSIVE on the demo account (fc-demo-execution-1030-spec.md §5). Written BEFORE fc_demo.py and the
new client methods. Fake venue, offline.

Interface the tests fix:
  fc_demo.FcDemo(cfg, venue, log, state=None).step(now, control_dir) -> dict
  venue: spot_buy_quote(sym, quote, cid) -> {"qty", "cost", "fee_usdt"}; spot_sell(sym, qty, cid) -> {"proceeds", "fee_usdt"}
         perp_sell(sym, qty, cid) -> {"qty", "price", "fee_usdt"}; perp_buy_reduce(sym, qty, cid) -> {"qty", "price", "fee_usdt"}
         perp_step(sym), spot_price(sym), mark(sym), funding_since(sym, since_ms) -> USDT
  exchange_client: fut_order (POST /fapi/v1/order behind _trading_allowed), fut_positions (GET /fapi/v2/positionRisk)
"""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


def cfg(**kw):
    base = dict(FC_DEMO_ENABLED=True, FC_DEMO_MODE="paper", FC_DEMO_NOTIONAL_USDT=2500.0, FC_DEMO_KILL_USDT=300.0,
                FC_DEMO_PAIRS=("BTCUSDT", "ETHUSDT"))
    base.update(kw)
    return SimpleNamespace(**base)


class Venue:
    def __init__(self, fail_perp=False):
        self.px = {"BTCUSDT": 80000.0, "ETHUSDT": 2500.0}
        self.mk = dict(self.px)
        self.calls, self.fail_perp = [], fail_perp

    def spot_buy_quote(self, sym, quote, cid):
        self.calls.append(("spot_buy", sym, quote))
        return {"qty": quote / self.px[sym] * 0.999, "cost": quote, "fee_usdt": 0.0}

    def spot_sell(self, sym, qty, cid):
        self.calls.append(("spot_sell", sym, qty))
        return {"proceeds": qty * self.px[sym] * 0.999, "fee_usdt": 0.0}

    def perp_sell(self, sym, qty, cid):
        self.calls.append(("perp_sell", sym, qty))
        if self.fail_perp:
            raise RuntimeError("rejected")
        return {"qty": qty, "price": self.mk[sym], "fee_usdt": qty * self.mk[sym] * 0.0005}

    def perp_buy_reduce(self, sym, qty, cid):
        self.calls.append(("perp_buy", sym, qty))
        return {"qty": qty, "price": self.mk[sym], "fee_usdt": qty * self.mk[sym] * 0.0005}

    def perp_step(self, sym):
        return 0.001

    def spot_price(self, sym):
        return self.px[sym]

    def mark(self, sym):
        return self.mk[sym]

    def funding_since(self, sym, since_ms):
        return 0.5


def at(h, m=0, day=30):
    return datetime(2026, 10, day, h, m, tzinfo=timezone.utc)


class Log:
    def __init__(self):
        self.rows = []

    def __call__(self, kind, now, **kw):
        self.rows.append({"kind": kind, **kw})

    def kind(self, k):
        return [r for r in self.rows if r["kind"] == k]


class TestFcDemo(unittest.TestCase):
    def test_disabled_does_nothing(self):
        import fc_demo as F
        v, lg = Venue(), Log()
        r = F.FcDemo(cfg(FC_DEMO_ENABLED=False), v, lg).step(at(1), None)
        self.assertEqual(r.get("skipped"), "disabled")
        self.assertEqual(v.calls, [])

    def test_entry_once_per_day_after_0015(self):
        import fc_demo as F
        v, lg = Venue(), Log()
        e = F.FcDemo(cfg(), v, lg)
        e.step(at(0, 10), None)
        self.assertEqual(v.calls, [])                                    # before the first 15m close
        e.step(at(0, 20), None)
        buys = [c for c in v.calls if c[0] == "spot_buy"]
        sells = [c for c in v.calls if c[0] == "perp_sell"]
        self.assertEqual(len(buys), 2)
        self.assertEqual(len(sells), 2)
        q_spot = 2500.0 / 80000.0 * 0.999
        self.assertAlmostEqual(next(c[2] for c in sells if c[1] == "BTCUSDT"), int(q_spot / 0.001) * 0.001, places=9)
        ent = lg.kind("fc_entry")
        self.assertEqual(len(ent), 2)
        self.assertTrue(all(len(r["client_order_ids"]) == 2 for r in ent))
        self.assertGreater(e.state["pos"]["BTCUSDT"]["dust"], 0.0)
        n = len(v.calls)
        e.step(at(6), None)
        self.assertEqual(len([c for c in v.calls[n:] if c[0] in ("spot_buy", "perp_sell")]), 0)

    def test_one_leg_unwind(self):
        import fc_demo as F
        v, lg = Venue(fail_perp=True), Log()
        with tempfile.TemporaryDirectory() as d:
            e = F.FcDemo(cfg(), v, lg)
            e.step(at(0, 20), Path(d))
            self.assertTrue((Path(d) / "PAUSE").exists())
            self.assertTrue(any(c[0] == "spot_sell" for c in v.calls))
            self.assertEqual(e.state["pos"], {})
            self.assertTrue(any(r.get("code") == "one_leg" for r in lg.kind("fc_alert")))

    def test_margin_stop_and_no_reentry_same_day(self):
        import fc_demo as F
        v, lg = Venue(), Log()
        e = F.FcDemo(cfg(), v, lg)
        e.step(at(0, 20), None)
        v.mk["BTCUSDT"] = 80000.0 * 1.51
        v.px["BTCUSDT"] = 80000.0 * 1.51
        n = len(v.calls)
        e.step(at(3), None)
        seq = [c[0] for c in v.calls[n:] if c[1] == "BTCUSDT"]
        self.assertEqual(seq[:2], ["perp_buy", "spot_sell"])             # perp first
        self.assertNotIn("BTCUSDT", e.state["pos"])
        self.assertIn("ETHUSDT", e.state["pos"])
        n = len(v.calls)
        e.step(at(5), None)
        self.assertFalse(any(c[0] == "spot_buy" for c in v.calls[n:]))
        e.step(at(0, 20, day=31), None)
        self.assertIn("BTCUSDT", e.state["pos"])

    def test_pause_flatten_kill(self):
        import fc_demo as F
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "PAUSE").write_text("x")
            v, lg = Venue(), Log()
            e = F.FcDemo(cfg(), v, lg)
            e.step(at(0, 20), d)
            self.assertEqual(v.calls, [])
            (d / "PAUSE").unlink()
            e.step(at(0, 25), d)
            self.assertEqual(len(e.state["pos"]), 2)
            (d / "FLATTEN").write_text("x")
            e.step(at(1), d)
            self.assertEqual(e.state["pos"], {})
            self.assertTrue((d / "PAUSE").exists())
            self.assertFalse((d / "FLATTEN").exists())
        v, lg = Venue(), Log()
        e = F.FcDemo(cfg(), v, lg)
        e.step(at(0, 20), None)
        v.px["BTCUSDT"] = 80000.0 * 0.80                                 # spot -20% while the perp mark stays
        e.step(at(2), None)
        self.assertEqual(e.state["pos"], {})
        self.assertTrue(any(r.get("code") == "kill_switch" for r in lg.kind("fc_alert")))

    def test_daily_mark(self):
        import fc_demo as F
        v, lg = Venue(), Log()
        e = F.FcDemo(cfg(), v, lg)
        e.step(at(0, 20), None)
        e.step(at(0, 20, day=31), None)
        m = lg.kind("fc_mark")
        self.assertTrue(m)
        r = m[-1]
        for k in ("spot_value", "perp_pnl", "margin", "funding", "hedge_delta", "equity_change"):
            self.assertIn(k, r)


class TestVenueBuild(unittest.TestCase):
    def test_paper_venue_constructs(self):
        """Bug 10-30: PaperVenue built a client on api.binance.com, which ExchangeClient refuses (demo hosts only)."""
        import fc_demo as F
        v = F.PaperVenue()
        self.assertIsNotNone(v.spot)


class TestClient(unittest.TestCase):
    def test_fut_order_guarded_and_demo_only(self):
        import exchange_client as X
        seen = []

        import io

        class R(io.BytesIO):
            headers, status = {}, 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def opener(req, timeout=None):
            seen.append((req.full_url, req.get_method()))
            body = {"serverTime": 1_800_000_000_000} if "/time" in req.full_url else ([] if "positionRisk" in req.full_url else {})
            return R(json.dumps(body).encode())
        off = SimpleNamespace(TRADING_ENABLED=False, TRADING_MODE="demo")
        c = X.ExchangeClient(base=X.DEMO_FAPI, key="k", secret="s", opener=opener, audit=None, cfg=off)
        with self.assertRaises(X.TradingDisabled):
            c.fut_order("BTCUSDT", "SELL", "MARKET", quantity="0.031")
        on = SimpleNamespace(TRADING_ENABLED=True, TRADING_MODE="demo")
        c = X.ExchangeClient(base=X.DEMO_FAPI, key="k", secret="s", opener=opener, audit=None, cfg=on)
        c.fut_order("BTCUSDT", "SELL", "MARKET", quantity="0.031")
        self.assertTrue(seen[-1][0].split("?")[0].endswith("demo-fapi.binance.com/fapi/v1/order"))
        self.assertEqual(seen[-1][1], "POST")
        c.fut_positions()
        self.assertIn("demo-fapi.binance.com/fapi/v2/positionRisk", seen[-1][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
