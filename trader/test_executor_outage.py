"""C1 incident 2026-10-14 07:58:50 UTC -- reproducing tests written BEFORE the fix
(operator rule: a bug fix starts with a reproducing test; demo-execution-c-1013-spec.md §9 work log).

What happened (audit journals of 10-14): the machine lost the network (getaddrinfo failed, SSL handshake
timeouts; 484 failed executor requests that day). Then:
  D1 equity() valued every coin it could not price at 0 -> equity == the USDT balance exactly (4800.375454)
     -> a FALSE kill switch -> flatten + PAUSE; the executor stood paused from then on;
  D2 PaperExchange.market_sell took the coins off the balance BEFORE asking the price; the price request
     failed, _place retried -> -2010 'insufficient balance'; ~200 USDT of coins vanished without proceeds;
  D3 the next poll saw the stops CANCELED and _protect placed a stop for qty 0 and called it protected;
  D4 _finish wrote pnl = None for a close with 0 proceeds, so the loss never reached `realized`
     (the daily loss limit could not see it).
"""
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import exchange_client as X
import fill_model as FM
from test_executor import CFG, NOW, StubPaper, W, filt, closed

SYMS = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT")


class Net:
    """Live mid prices behind a switchable network outage (the failure the client raises: status None)."""

    def __init__(self, px):
        self.px, self.down = dict(px), False

    def __call__(self, sym):
        if self.down:
            raise X.ExchangeError(None, "network", "URLError(gaierror(11001, 'getaddrinfo failed'))",
                                  "/api/v3/ticker/bookTicker")
        return self.px[sym]


def paper_executor(cfg=None):
    import executor as E
    E.FC_STATE = Path(__file__).with_name("_no_fc_state_in_tests.json")   # never the live runtime/fc_demo state
    net = Net({s: 2.0 for s in SYMS})
    pe = E.PaperExchange(net, FM.model("taker", 1), {s: filt() for s in SYMS}, usdt=5000.0)
    paper, w = StubPaper(), W()
    exe = E.Executor(paper, pe, cfg or CFG(), w)
    return E, net, pe, paper, w, exe


def open_four(paper, exe):
    keys = [paper.enter(sym=s, ts=f"2026-10-14T00:4{i}:00+00:00", price=2.0, notional=1000.0, stop=1.84)
            for i, s in enumerate(SYMS)]
    exe.cycle([], NOW)
    return keys


class TestOutageNoFalseKillSwitch(unittest.TestCase):
    """D1."""

    def test_network_outage_does_not_trip_kill_switch(self):
        E, net, pe, paper, w, exe = paper_executor()
        keys = open_four(paper, exe)
        self.assertTrue(all(exe.positions[k]["state"] == "PROTECTED" for k in keys))
        with tempfile.TemporaryDirectory() as d:
            ctl = Path(d)
            net.down = True
            exe.cycle([], NOW + timedelta(minutes=1), control_dir=ctl)
            self.assertNotIn("kill_switch", [a.get("code") for a in w.kind("alert")])
            self.assertFalse((ctl / "PAUSE").exists())
            self.assertFalse(exe.killed)
            for k in keys:
                self.assertIn(k, exe.positions)                   # nothing flattened
            net.down = False                                      # network back: business as usual
            exe.cycle([], NOW + timedelta(minutes=2), control_dir=ctl)
            self.assertFalse(exe.killed)

    def test_paper_equity_refuses_to_guess(self):
        E, net, pe, paper, w, exe = paper_executor()
        open_four(paper, exe)
        net.down = True
        with self.assertRaises(Exception):
            pe.equity()                                           # an unpriced coin is unknown, not 0

    def test_real_drawdown_still_trips(self):
        E, net, pe, paper, w, exe = paper_executor()
        keys = open_four(paper, exe)
        with tempfile.TemporaryDirectory() as d:
            ctl = Path(d)
            for s in SYMS:
                net.px[s] = 0.5                                   # -75% on 80 USDT of coins: -60 USDT
            pe.bal["USDT"] -= 10.0
            exe.cycle([], NOW + timedelta(minutes=1), control_dir=ctl)
            self.assertIn("kill_switch", [a.get("code") for a in w.kind("alert")])
            self.assertTrue((ctl / "PAUSE").exists())
            for k in keys:
                self.assertIsNotNone(closed(exe, k))

    def test_orders_equity_missing_price_row_raises(self):
        """Live adapter: a held coin whose pair is known but whose price row is missing is not worth 0."""
        bal = [{"asset": "USDT", "free": "100", "locked": "0"}, {"asset": "ETH", "free": "1", "locked": "0"},
               {"asset": "BTC", "free": "0", "locked": "2"}]

        class C:
            def account(self):
                return {"balances": bal}

            def book_tickers(self, symbols=None, base=None):
                return [{"symbol": "ETHUSDT", "bidPrice": "10", "askPrice": "10"}]      # BTCUSDT row missing

        xinfo = {"symbols": [{"symbol": s, "filters": list(filt().values())} for s in ("BTCUSDT", "ETHUSDT")]}
        o = X.Orders(C(), xinfo)
        with self.assertRaises(Exception):
            o.equity()


class TestPaperSellAtomic(unittest.TestCase):
    """D2."""

    def test_failed_price_leaves_balance_intact(self):
        E, net, pe, paper, w, exe = paper_executor()
        pe.market_buy("AAAUSDT", 20.0, "cb_b")
        q, usdt = pe.free("AAA"), pe.free("USDT")
        net.down = True
        with self.assertRaises(X.ExchangeError):
            pe.market_sell("AAAUSDT", q, "cb_x")
        self.assertAlmostEqual(pe.free("AAA"), q)                 # the coins are still there
        self.assertAlmostEqual(pe.free("USDT"), usdt)
        net.down = False
        o = pe.market_sell("AAAUSDT", q, "cb_x2")                 # and sell once the price is back
        self.assertEqual(o["status"], "FILLED")
        self.assertGreater(pe.free("USDT"), usdt + 19.0)

    def test_flatten_during_outage_loses_no_coins(self):
        """The whole 10-14 chain: a flatten while prices fail must not make coins vanish."""
        E, net, pe, paper, w, exe = paper_executor()
        keys = open_four(paper, exe)
        coins = {s: pe.free(s[:-4]) + pe.locked.get(s[:-4], 0.0) for s in SYMS}
        net.down = True
        exe.flatten(NOW + timedelta(minutes=1), "flatten")
        for s in SYMS:
            b = s[:-4]
            held = pe.free(b) + pe.locked.get(b, 0.0)
            gone = [p for p in exe.closed if p["sym"] == s and (p.get("sold_qty") or 0) > 0]
            self.assertTrue(abs(held - coins[s]) < 1e-9 or gone, (s, held, coins[s]))
        self.assertNotIn(-2010, [a.get("err_code") for a in w.kind("alert")])
        self.assertTrue(all(k in exe.positions or closed(exe, k) is not None for k in keys))


class TestNoZeroQtyProtection(unittest.TestCase):
    """D3."""

    def test_no_coins_is_not_protected(self):
        E, net, pe, paper, w, exe = paper_executor()
        key = paper.enter(sym="AAAUSDT", stop=1.84)
        exe.cycle([], NOW)
        p = exe.positions[key]
        pe.cancel("AAAUSDT", p["stop_order_id"])                  # stop gone ...
        pe.bal["AAA"] = 0.0                                       # ... and the coins too
        exe.cycle([], NOW + timedelta(minutes=1))
        p2 = exe.positions.get(key) or closed(exe, key)
        self.assertFalse(p2.get("state") == "PROTECTED" and not p2.get("qty"),
                         "a position with no coins was reported protected with qty 0")
        zero_stops = [o for o in pe.orders.values() if o["type"] == "STOP_LOSS" and float(o["origQty"]) == 0]
        self.assertEqual(zero_stops, [])


class TestZeroProceedsCounted(unittest.TestCase):
    """D4."""

    def test_close_with_nothing_sold_is_a_full_loss(self):
        E, net, pe, paper, w, exe = paper_executor()
        key = paper.enter(sym="AAAUSDT", stop=1.84)
        exe.cycle([], NOW)
        cost = exe.positions[key]["cost"]
        exe._finish(exe.positions[key], NOW + timedelta(minutes=1), "no_balance", 0.0, dust=None, sold=0.0)
        p = closed(exe, key)
        self.assertAlmostEqual(p["pnl"], -cost)
        self.assertAlmostEqual(exe.realized[NOW.strftime("%Y-%m-%d")], -cost)
        rec = [r for r in w.kind("real_exit") if r.get("key") == key][-1]
        self.assertAlmostEqual(rec["pnl"], -cost)

    def test_unknown_proceeds_stay_unknown(self):
        """Reconcile 'position gone from the account': the coins may have been sold by hand, proceeds are
        in myTrades (phase D), not known here -> pnl None, not an invented full loss."""
        E, net, pe, paper, w, exe = paper_executor()
        key = paper.enter(sym="AAAUSDT", stop=1.84)
        exe.cycle([], NOW)
        exe._finish(exe.positions[key], NOW + timedelta(minutes=1), "gone_from_exchange", None, dust=None, sold=0.0)
        self.assertIsNone(closed(exe, key)["pnl"])
        self.assertNotIn(NOW.strftime("%Y-%m-%d"), exe.realized)

    def test_reconcile_gone_passes_unknown(self):
        import inspect
        import executor as E
        src = inspect.getsource(E.Executor.reconcile)
        self.assertIn('"gone_from_exchange", None', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
