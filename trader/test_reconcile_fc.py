"""Reconcile must leave the FC-DEMO legs alone (fc-demo-execution-1030-spec.md §6, 2026-11-03, fifth blocker), and the
report's no-fills line must not claim demo orders are off (demo-ledger-d-1013-spec.md, amendment 2026-11-03).
Written BEFORE the fix. Reproduces 11-02: reconcile put a STOP_LOSS on the whole FC spot leg every cycle and cancelled
it on the next one (570 orphan_order + 570 orphan_balance alerts)."""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from test_executor import CFG, FakeOrders, make
from test_reconcile import setup

NOW = datetime(2026, 11, 2, 0, 0, 10, tzinfo=timezone.utc)


def fc_account():
    ex, pos = setup()
    ex.px["BTCUSDT"] = 70_000.0
    ex.bal["BTC"] = 0.03085                                # FC spot leg, ~2 160 USDT, no X4 position
    return ex, pos


class TestReconcileReserved(unittest.TestCase):
    def acts(self, ex, pos, reserved):
        import reconcile as R
        return R.run(pos, ex, lambda k: 1.84, CFG(), reserved=reserved)

    def test_fc_balance_is_not_an_orphan(self):
        ex, pos = fc_account()
        self.assertIn("orphan_balance", [a["action"] for a in self.acts(ex, pos, frozenset())])    # the 11-02 bug
        self.assertEqual(self.acts(ex, pos, frozenset({"BTCUSDT"})), [])

    def test_fc_orders_untouched_x4_leftover_cancelled_once(self):
        ex, pos = fc_account()
        fc = ex.stop_loss("BTCUSDT", 0.01, 60_000.0, "fc_btc_202611020000_s")
        left = ex.stop_loss("BTCUSDT", 0.02, 62_000.0, "cb_c1013a01_2e5d813a00f0_o")
        a = self.acts(ex, pos, frozenset({"BTCUSDT"}))
        self.assertNotIn(fc["orderId"], [x.get("order_id") for x in a])          # no cancel, no foreign_order alert
        c = [x for x in a if x["action"] == "cancel_order"]
        self.assertEqual([x["order_id"] for x in c], [left["orderId"]])
        self.assertTrue(c[0].get("reserved"))
        self.assertNotIn("orphan_balance", [x["action"] for x in a])

    def test_other_pairs_unchanged(self):
        ex, pos = fc_account()
        ex.bal["XYZ"] = 10.0
        a = self.acts(ex, pos, frozenset({"BTCUSDT"}))
        self.assertEqual([(x["action"], x["sym"]) for x in a], [("orphan_balance", "XYZUSDT")])


class TestExecutorPassesReserved(unittest.TestCase):
    def test_no_stop_on_fc_leg_and_leftover_alert(self):
        E, paper, ex, w, exe = make(prices={"ABCUSDT": 2.0, "BTCUSDT": 70_000.0})
        with tempfile.TemporaryDirectory() as d:
            st = Path(d) / "state.json"
            st.write_text(json.dumps({"mode": "demo", "pos": {"BTCUSDT": {"q_spot": 0.03085, "q_perp": 0.0308}}}),
                          encoding="utf-8")
            E.FC_STATE = st
            ex.bal["BTC"] = 0.03085
            left = ex.stop_loss("BTCUSDT", 0.03085, 62_000.0, "cb_c1013a01_2e5d813a00f0_o")
            ex.calls.clear()
            exe.reconcile(NOW)
            codes = [r.get("code") for r in w.kind("alert")]
            self.assertNotIn("orphan_balance", codes)
            self.assertNotIn("orphan_order", codes)
            self.assertIn("fc_leftover_cancel", codes)
            self.assertNotIn("stop_loss", [c[0] for c in ex.calls])
            self.assertNotIn(left["orderId"], [o["orderId"] for o in ex.open_orders()])
            w.rows.clear()
            exe.reconcile(NOW)                                                      # next cycle: silent
            self.assertEqual([r.get("code") for r in w.kind("alert")], [])
        E.FC_STATE = Path(__file__).with_name("_no_fc_state_in_tests.json")


class TestReportNoFillsLine(unittest.TestCase):
    def test_report_no_fills_line(self):
        import trader_report as TR
        txt = TR.build("2026-11-02", [], demo_fills=[])
        self.assertIn("демо-счёт: исполнений за день нет", txt)
        self.assertNotIn("ещё не включены", txt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
