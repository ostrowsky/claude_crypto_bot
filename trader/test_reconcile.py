"""Phase C reconciliation -- tests written from demo-execution-c-1013-spec.md §5.2 BEFORE the code
(operator rule 2026-10-10: spec -> tests -> code)."""
import unittest

from test_executor import CFG, FakeOrders, floor_step

KEY = "ABCUSDT|15m|2026-10-14T00:45:00+00:00"


def setup(stop_alive=True):
    ex = FakeOrders({"ABCUSDT": 2.0, "XYZUSDT": 2.0})
    ex.bal["ABC"] = 9.99
    o = ex.stop_loss("ABCUSDT", 9.99, 1.7848, "cb_t1013a01_abc_s1")
    if not stop_alive:
        ex.cancel("ABCUSDT", o["orderId"])
    ex.calls.clear()
    pos = {KEY: {"key": KEY, "sym": "ABCUSDT", "state": "PROTECTED", "qty": 9.99, "stop_order_id": o["orderId"],
                 "stop_price": 1.7848}}
    return ex, pos


def acts(ex, pos, **kw):
    import reconcile as R
    return R.run(pos, ex, lambda k: 1.84, CFG(), **kw)


class TestReconcile(unittest.TestCase):
    def test_consistent_account_needs_nothing(self):
        ex, pos = setup()
        self.assertEqual(acts(ex, pos), [])
        self.assertEqual([c for c in ex.calls if c[0] not in ("get_order",)], [])   # reconcile only reads

    def test_position_without_stop_gets_one(self):
        ex, pos = setup(stop_alive=False)
        a = acts(ex, pos)
        self.assertEqual([x["action"] for x in a], ["place_stop"])
        self.assertEqual(a[0]["key"], KEY)
        self.assertAlmostEqual(a[0]["stop"], floor_step(1.84 * 0.97, "0.0001"))

    def test_orphan_balance(self):
        ex, pos = setup()
        ex.bal["XYZ"] = 10.0                                   # 20 USDT, not dust, no position
        a = [x for x in acts(ex, pos) if x["action"] == "orphan_balance"]
        self.assertEqual(len(a), 1)
        self.assertEqual(a[0]["sym"], "XYZUSDT")
        self.assertAlmostEqual(a[0]["stop"], floor_step(2.0 * 0.89, "0.0001"))

    def test_dust_and_stablecoins_ignored(self):
        ex, pos = setup()
        ex.bal["XYZ"] = 1.0                                    # 2 USDT < minNotional 5
        ex.bal["USDC"] = 5000.0
        self.assertEqual(acts(ex, pos), [])

    def test_own_orphan_order_cancelled_foreign_only_alerted(self):
        ex, pos = setup()
        ex.bal["XYZ"] = 10.0
        mine = ex.stop_loss("XYZUSDT", 5.0, 1.5, "cb_t1013a01_xyz_s1")
        foreign = ex.stop_loss("XYZUSDT", 5.0, 1.4, "web_manual_1")
        a = acts(ex, pos)
        cancels = [x for x in a if x["action"] == "cancel_order"]
        self.assertEqual([x["order_id"] for x in cancels], [mine["orderId"]])
        self.assertIn("foreign_order", [x.get("code") for x in a if x["action"] == "alert"])
        self.assertNotIn(foreign["orderId"], [x.get("order_id") for x in cancels])

    def test_position_gone_from_exchange(self):
        ex, pos = setup(stop_alive=False)
        ex.bal["ABC"] = 0.0
        a = acts(ex, pos)
        self.assertEqual([x["action"] for x in a], ["close_from_trades"])
        self.assertEqual(a[0]["key"], KEY)

    def test_reconcile_fail_blocks_entries(self):
        ex, pos = setup()
        self.assertEqual([x["action"] for x in acts(ex, pos, day_status="RECONCILE_FAIL")], ["block_entries"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
