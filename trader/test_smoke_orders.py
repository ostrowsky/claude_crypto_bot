"""C0 smoke of demo orders -- tests written from demo-execution-c-1013-spec.md §6 (C0) BEFORE the code
(operator rule 2026-10-10: spec -> tests -> code)."""
import unittest

from test_executor import filt

XINFO = {"symbols": [{"symbol": "ABCUSDT", "quoteAssetPrecision": 8, "filters": list(filt().values())}]}


class FakeClient:
    """Records the raw client calls the smoke makes; holds one symbol's balance and orders."""

    def __init__(self, fail_at=None):
        self.calls, self.fail_at = [], fail_at
        self.base, self.orders, self.lists, self.nid = 0.0, {}, {}, 10

    def _f(self, name):
        self.calls.append(name)
        if name == self.fail_at:
            import exchange_client as X
            raise X.ExchangeError(400, -1013, "Filter failure", "/api/v3/" + name)

    def account(self):
        self._f("account")
        return {"balances": [{"asset": "ABC", "free": str(self.base), "locked": "0"},
                             {"asset": "USDT", "free": "5000", "locked": "0"}]}

    def book_tickers(self, base=None, symbols=None):
        raise AssertionError("C0 must not ask bookTicker for all pairs (4.6-4.8 s on demo, 2026-10-13)")

    def book_ticker(self, symbol, base=None):
        return {"symbol": symbol, "bidPrice": "1.999", "askPrice": "2.001"}

    def place_order(self, symbol, side=None, type_=None, **kw):
        self._f(f"order_{side}_{type_}" + ("_trailing" if "trailingDelta" in kw else ""))
        self.nid += 1
        if type_ == "MARKET" and side == "BUY":
            self.base += 4.995
            return {"orderId": self.nid, "status": "FILLED", "executedQty": "5", "cummulativeQuoteQty": "10"}
        if type_ == "MARKET" and side == "SELL":
            self.base -= float(kw["quantity"])
            return {"orderId": self.nid, "status": "FILLED", "executedQty": kw["quantity"]}
        self.orders[self.nid] = {"orderId": self.nid, "symbol": symbol, "status": "NEW"}
        return {"orderId": self.nid, "status": "NEW"}

    def cancel_replace(self, symbol, cancel_order_id, type_, side="SELL", **kw):
        self._f("cancel_replace")
        self.orders.pop(cancel_order_id, None)
        self.nid += 1
        self.orders[self.nid] = {"orderId": self.nid, "symbol": symbol, "status": "NEW"}
        return {"cancelResult": "SUCCESS", "newOrderResult": "SUCCESS", "newOrderResponse": {"orderId": self.nid}}

    def cancel_order(self, symbol, order_id=None, orig_client_order_id=None):
        self._f("cancel_order")
        self.orders.pop(order_id, None)
        return {"orderId": order_id, "status": "CANCELED"}

    def place_oco(self, symbol, side=None, **kw):
        self._f("oco")
        self.nid += 1
        self.lists[self.nid] = [self.nid + 100, self.nid + 101]
        for oid in self.lists[self.nid]:
            self.orders[oid] = {"orderId": oid, "symbol": symbol, "status": "NEW"}
        return {"orderListId": self.nid, "orderReports": [{"orderId": o} for o in self.lists[self.nid]]}

    def cancel_order_list(self, symbol, order_list_id):
        self._f("cancel_oco")
        for oid in self.lists.pop(order_list_id, []):
            self.orders.pop(oid, None)
        return {"orderListId": order_list_id, "listOrderStatus": "ALL_DONE"}

    def open_orders(self, symbol=None):
        self._f("open_orders")
        return list(self.orders.values())


class TestSmoke(unittest.TestCase):
    def test_all_steps_in_order_and_clean_end(self):
        import _smoke_demo_orders as SM
        c = FakeClient()
        rep = SM.run_smoke(c, XINFO, "ABCUSDT", quote=10.0, ledger_sync=lambda: 3)
        names = [s["step"] for s in rep["steps"]]
        self.assertEqual(names, ["buy", "stop", "replace", "cancel_stop", "oco", "cancel_oco", "trailing_stop",
                                 "cancel_trailing", "sell", "ledger_sync", "no_open_orders"])
        self.assertTrue(all(s["ok"] for s in rep["steps"]), rep["steps"])
        self.assertTrue(rep["ok"])
        self.assertEqual(c.orders, {})
        self.assertLess(c.base, 0.01)                     # sold down to dust

    def test_failure_midway_cleans_up(self):
        import _smoke_demo_orders as SM
        c = FakeClient(fail_at="oco")
        rep = SM.run_smoke(c, XINFO, "ABCUSDT", quote=10.0, ledger_sync=lambda: 0)
        self.assertFalse(rep["ok"])
        self.assertEqual(rep["failed_step"], "oco")
        self.assertEqual(c.orders, {})                    # every open order cancelled
        self.assertLess(c.base, 0.01)                     # the coin sold back
        self.assertTrue(rep["cleanup"]["ok"])

    def test_refuses_without_go(self):
        import _smoke_demo_orders as SM
        self.assertNotEqual(SM.main(["--symbol", "ABCUSDT"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
