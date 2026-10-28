"""Phase D: ledger, daily revenue from the demo account, daily report -- tests written from
demo-ledger-d-1013-spec.md §8 BEFORE the code (operator rule 2026-10-10: spec -> tests -> code)."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
D0 = datetime(2026, 10, 14, tzinfo=timezone.utc)
MS = lambda d: int(d.timestamp() * 1000)  # noqa: E731


def snap(t, holdings, ts_fmt=True):
    eq = sum(h["qty"] * h["px"] for h in holdings.values())
    return {"ts": t.isoformat() if ts_fmt else t, "equity_usdt": eq, "holdings": holdings,
            "positions": {a: h["qty"] * h["px"] for a, h in holdings.items()}}


def day_snaps(day, hold_fn, every_min=5, start=None, end=None):
    out, t = [], (start or day)
    stop = end or (day + timedelta(days=1))
    while t <= stop:
        out.append(snap(t, hold_fn(t)))
        t += timedelta(minutes=every_min)
    return out


class FakeEx:
    def __init__(self, trades):
        self.trades, self.calls = trades, []

    def my_trades(self, symbol, from_id=None, limit=500):
        self.calls.append((symbol, from_id))
        return [t for t in self.trades.get(symbol, []) if from_id is None or t["id"] >= from_id][:limit]


def mt(i, sym, side, price, qty, comm, comm_asset, t, order_id=1, maker=False):
    return {"id": i, "orderId": order_id, "symbol": sym, "price": str(price), "qty": str(qty), "quoteQty": str(price * qty),
            "commission": str(comm), "commissionAsset": comm_asset, "time": MS(t), "isBuyer": side == "BUY",
            "isMaker": maker}


class TestLedger(unittest.TestCase):
    """§8.1."""

    def test_sync_once_idempotent_fromid_bnb_and_decision(self):
        import trade_ledger as L
        trades = {"ABCUSDT": [mt(10, "ABCUSDT", "BUY", 2.0, 100, 0.05, "BNB", D0 + timedelta(hours=1), order_id=7),
                              mt(11, "ABCUSDT", "SELL", 2.1, 100, 0.21, "USDT", D0 + timedelta(hours=3), order_id=8)]}
        ex = FakeEx(trades)
        with tempfile.TemporaryDirectory() as d:
            led, st = Path(d) / "ledger.jsonl", {}
            omap = {7: "cb_x_ABC_1_entry", 8: "cb_x_ABC_1_exit"}
            dmap = {"cb_x_ABC_1_entry": "dec-1"}
            price_at = lambda asset, t: (600.0, "test")   # noqa: E731  BNB = 600 USDT
            n1 = L.sync(ex, ["ABCUSDT"], st, led, omap, dmap, price_at)
            n2 = L.sync(ex, ["ABCUSDT"], st, led, omap, dmap, price_at)
            rows = [json.loads(x) for x in led.read_text(encoding="utf-8").splitlines()]
            self.assertEqual((n1, n2, len(rows)), (2, 0, 2))
            self.assertEqual(st["from_id"]["ABCUSDT"], 12)
            buy = rows[0]
            self.assertAlmostEqual(buy["commission_usdt"], 0.05 * 600.0)
            self.assertEqual(buy["commission_price_source"], "test")
            self.assertEqual(buy["client_order_id"], "cb_x_ABC_1_entry")
            self.assertEqual(buy["decision_id"], "dec-1")
            self.assertAlmostEqual(rows[1]["commission_usdt"], 0.21)

    def test_symbols_to_sync(self):
        import trade_ledger as L
        audit = [{"path": "/api/v3/order", "params": {"symbol": "XUSDT"}}, {"path": "/api/v3/depth", "params": {"symbol": "ZUSDT"}}]
        syms = L.symbols_to_sync(audit, open_orders=[{"symbol": "YUSDT"}],
                                 balances=[{"asset": "USDT", "free": "5", "locked": "0"},
                                           {"asset": "QQ", "free": "1", "locked": "0"},
                                           {"asset": "USDC", "free": "5", "locked": "0"}])
        self.assertEqual(syms, ["QQUSDT", "XUSDT", "YUSDT"])


class TestOrderMapPhaseC(unittest.TestCase):
    """demo-execution-c-1013-spec.md §7: the ledger links fills of orders placed by cancelReplace and OCO."""

    def test_cancel_replace_and_oco_bodies(self):
        import trade_ledger as L
        rows = [{"path": "/api/v3/order", "orderId": 1, "clientOrderId": "cb_b"},
                {"path": "/api/v3/order/cancelReplace",
                 "body": {"cancelResult": "SUCCESS", "newOrderResult": "SUCCESS",
                          "cancelResponse": {"orderId": 2, "clientOrderId": "cb_s1"},
                          "newOrderResponse": {"orderId": 3, "clientOrderId": "cb_s2"}}},
                {"path": "/api/v3/orderList/oco",
                 "body": {"orderListId": 9, "orderReports": [{"orderId": 4, "clientOrderId": "cb_tp"},
                                                             {"orderId": 5, "clientOrderId": "cb_sl"}]}}]
        m = L.order_client_map(rows)
        self.assertEqual(m, {1: "cb_b", 2: "cb_s1", 3: "cb_s2", 4: "cb_tp", 5: "cb_sl"})


class TestMidnightAndReconcile(unittest.TestCase):
    """§8.2–8.6."""

    def _hold(self, usdt=5000.0, usdc_px=1.0, abc=0.0, abc_px=2.0):
        h = {"USDT": {"qty": usdt, "px": 1.0}, "USDC": {"qty": 5000.0, "px": usdc_px}}
        if abc:
            h["ABC"] = {"qty": abc, "px": abc_px}
        return h

    def test_mark_within_10_min(self):
        import daily_revenue as DR
        snaps = [snap(D0 + timedelta(minutes=7), self._hold()), snap(D0 + timedelta(minutes=30), self._hold())]
        self.assertEqual(DR.midnight_mark(snaps, D0)["ts"], snaps[0]["ts"])
        self.assertIsNone(DR.midnight_mark([snap(D0 + timedelta(minutes=11), self._hold())], D0))

    def test_no_trades_flat_full_day(self):
        import daily_revenue as DR
        snaps = day_snaps(D0, lambda t: self._hold(usdc_px=1.0 + 0.0002 * (t - D0).total_seconds() / 86400))
        rec = DR.compute_day(D0.strftime("%Y-%m-%d"), snaps, fills=[])
        self.assertEqual(rec["status"], "FULL")
        self.assertAlmostEqual(rec["stable_contrib_pct"], 5000 * 0.0002 / 10000 * 100, places=6)
        self.assertLess(abs(rec["R_policy_pct"]), 0.005)
        self.assertEqual(rec["provenance"], "equity_snapshots")

    def test_unknown_without_marks(self):
        import daily_revenue as DR
        rec = DR.compute_day(D0.strftime("%Y-%m-%d"), [snap(D0 + timedelta(hours=5), self._hold())], fills=[])
        self.assertEqual(rec["status"], "UNKNOWN")

    def test_trade_reconciles_and_reset_detected(self):
        import daily_revenue as DR
        t_buy = D0 + timedelta(hours=2)
        fills = [{"symbol": "ABCUSDT", "side": "BUY", "qty": 100.0, "quote_qty": 200.0, "commission": 0.2,
                  "commission_asset": "USDT", "time": t_buy.isoformat()}]

        def hold(t):
            if t < t_buy:
                return self._hold()
            return self._hold(usdt=5000 - 200.2, abc=100, abc_px=2.06)       # position marked at +3%
        snaps = day_snaps(D0, hold)
        rec = DR.compute_day(D0.strftime("%Y-%m-%d"), snaps, fills)
        self.assertEqual(rec["status"], "FULL")                                # +3% revaluation is not a reset
        self.assertLess(abs(rec["P_minus_L_pct"]), 1e-9)

        def hold_reset(t):
            return self._hold() if t < D0 + timedelta(hours=10) else self._hold(usdt=5100)   # +1% from nowhere
        rec2 = DR.compute_day(D0.strftime("%Y-%m-%d"), day_snaps(D0, hold_reset), fills=[])
        self.assertEqual(rec2["status"], "RESET")

        def hold_small(t):
            return self._hold() if t < D0 + timedelta(hours=10) else self._hold(usdt=5002)   # 0.02% unexplained
        rec3 = DR.compute_day(D0.strftime("%Y-%m-%d"), day_snaps(D0, hold_small), fills=[])
        self.assertEqual(rec3["status"], "RECONCILE_FAIL")

    def test_bot_down_by_coverage(self):
        import daily_revenue as DR
        snaps = day_snaps(D0, lambda t: self._hold(), every_min=10)          # 50% of 5-min snapshots
        rec = DR.compute_day(D0.strftime("%Y-%m-%d"), snaps, fills=[])
        self.assertEqual(rec["status"], "BOT_DOWN")


class TestBenchAndAppend(unittest.TestCase):
    """§8.7–8.8."""

    def test_benchmarks(self):
        import daily_revenue as DR
        b = DR.benchmarks(btc_open_close=(100.0, 103.0), universe_open_close=[(10.0, 11.0), (20.0, 19.0)])
        self.assertAlmostEqual(b["bench_btc_pct"], 3.0)
        self.assertAlmostEqual(b["bench_universe_pct"], (10.0 - 5.0) / 2)
        self.assertEqual(b["bench_usdt_pct"], 0.0)

    def test_append_revision(self):
        import daily_revenue as DR
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "dr.jsonl"
            DR.append_day(p, {"day": "2026-10-14", "R_pct": 0.1})
            DR.append_day(p, {"day": "2026-10-14", "R_pct": 0.2})
            rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["revision"] for r in rows], [1, 2])
            self.assertEqual(DR.latest_by_day(p)["2026-10-14"]["R_pct"], 0.2)


class TestReport(unittest.TestCase):
    """§8.9 and §8.12."""

    def test_report_text(self):
        import trader_report as TR
        days = [{"day": f"2026-10-{d:02d}", "status": "FULL", "R_pct": 0.0, "R_policy_pct": ((-1) ** d) * 0.01,
                 "stable_contrib_pct": 0.0, "bench_btc_pct": 1.0, "bench_universe_pct": 0.5} for d in range(1, 15)]
        shadow = [{"type": "exit", "book": "x4_floor8", "sym": "AUSDT", "entry_ts": "2026-10-14T05:00:00+00:00",
                   "exit_time": "2026-10-14T09:00:00+00:00", "reason": "trail", "ret_pct": -2.5, "pnl": -25.0},
                  {"type": "exit", "book": "champion", "sym": "BUSDT", "entry_ts": "2026-10-13T22:00:00+00:00",
                   "exit_time": "2026-10-14T01:00:00+00:00", "reason": "real", "ret_pct": 1.0, "pnl": 10.0},
                  {"type": "exit", "book": "champion", "sym": "CUSDT", "entry_ts": "2026-10-12T22:00:00+00:00",
                   "exit_time": "2026-10-13T01:00:00+00:00", "reason": "real", "ret_pct": 9.0, "pnl": 90.0}]
        txt = TR.build("2026-10-14", days, shadow_rows=shadow, demo_fills=[], exchange_rows=[], snapshot_share=0.99)
        self.assertIn("рано судить", txt)
        self.assertNotIn("растёт", txt)
        self.assertNotIn("падает", txt)
        self.assertIn("Сделки предыдущего дня", txt)
        self.assertIn("нет сделок", txt)                         # demo account has none yet
        self.assertIn("AUSDT", txt)
        self.assertIn("BUSDT", txt)
        self.assertNotIn("CUSDT", txt)                           # exited the day before
        self.assertIn("x4_floor8", txt)


class TestFewDays(unittest.TestCase):
    """Found on the first real run 2026-10-13: one counted day gives a point CI [x, x] and the report
    said "выше нуля". A direction needs at least two 5-day bootstrap blocks."""

    def test_one_day_is_too_early(self):
        import daily_revenue as DR
        import trader_report as TR
        a = DR.aggregate([("2026-10-12", 0.0075)])
        self.assertEqual(DR.verdict_text(a), "рано судить")
        nine = DR.aggregate([(f"2026-10-{d:02d}", 1.0) for d in range(1, 10)])
        self.assertEqual(DR.verdict_text(nine), "рано судить")
        ten = DR.aggregate([(f"2026-10-{d:02d}", 1.0) for d in range(1, 11)])
        self.assertEqual(DR.verdict_text(ten), "выше нуля")
        txt = TR.build("2026-10-12", [{"day": "2026-10-12", "status": "FULL", "R_pct": 0.0075, "R_policy_pct": None}])
        self.assertNotIn("выше нуля", txt)
        self.assertIn("рано судить", txt)


class TestReportExecutor(unittest.TestCase):
    """demo-execution-c-1013-spec.md §6: the 09:00 report shows the executor's trades of the day (C1 paper / C2 demo)."""

    def test_executor_section(self):
        import trader_report as TR
        rows = [{"kind": "real_entry", "mode": "paper", "ts": "2026-10-14T03:00:00+00:00", "action": "enter",
                 "key": "AUSDT|15m|2026-10-14T02:59:00+00:00", "sym": "AUSDT", "cost": 20.0, "slippage_bps": 4.0},
                {"kind": "real_entry", "mode": "paper", "ts": "2026-10-14T04:00:00+00:00", "action": "skip",
                 "reason": "exposure_cap", "sym": "BUSDT"},
                {"kind": "real_exit", "mode": "paper", "ts": "2026-10-14T09:00:00+00:00", "action": "exit",
                 "key": "AUSDT|15m|2026-10-14T02:59:00+00:00", "sym": "AUSDT", "reason": "trail", "pnl": -0.6, "ret_pct": -3.0},
                {"kind": "real_exit", "mode": "paper", "ts": "2026-10-13T09:00:00+00:00", "action": "exit",
                 "sym": "OLDUSDT", "reason": "trail", "pnl": 5.0, "ret_pct": 25.0},
                {"kind": "alert", "mode": "paper", "ts": "2026-10-14T05:00:00+00:00", "code": "stop_missing"}]
        txt = TR.build("2026-10-14", [], executor_rows=rows)
        self.assertIn("Исполнитель", txt)
        self.assertIn("paper", txt)
        self.assertIn("AUSDT", txt)
        self.assertIn("exposure_cap", txt)
        self.assertIn("stop_missing", txt)
        self.assertNotIn("OLDUSDT", txt)                         # closed the day before


class TestSnapshotHoldings(unittest.TestCase):
    """§8.11."""

    def test_value_returns_holdings(self):
        import equity_snapshots as EQ
        v = EQ.value([{"asset": "USDT", "free": "10", "locked": "0"}, {"asset": "ETH", "free": "1", "locked": "0"}],
                     {"ETHUSDT": 2000.0})
        self.assertEqual(v["holdings"]["ETH"], {"qty": 1.0, "px": 2000.0})
        self.assertEqual(v["holdings"]["USDT"], {"qty": 10.0, "px": 1.0})


class TestBoundary(unittest.TestCase):
    """§8.10."""

    def test_runtime_only_no_telegram(self):
        for name in ("trade_ledger.py", "trader_report.py"):
            src = (HERE / name).read_text(encoding="utf-8")
            self.assertNotIn('"' + '.runtime' + '"', src)
            self.assertNotIn("telegram", src.lower())
            self.assertNotIn("api.telegram.org", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestReportIntervalBelowTenDays(unittest.TestCase):
    """Gate C readout 2026-10-28: the 7-day window printed [-0.061, -0.033] around a mean of -0.035 with sd 0.040 --
    with a 5-day block and n = 7 the bootstrap has 3 block starts and the interval is degenerate. Below
    MIN_DAYS_DIRECTION the report must not print bootstrap bounds at all."""

    def rows(self, n):
        vals = [-0.016, -0.106, -0.069, -0.036, -0.025, 0.015, -0.012, 0.02, -0.03, 0.01, -0.04, 0.005]
        return [{"day": f"2026-10-{d + 1:02d}", "status": "FULL", "R_pct": 0.0, "R_policy_pct": vals[d],
                 "stable_contrib_pct": 0.0} for d in range(n)]

    def test_seven_days_no_bounds(self):
        import trader_report as TR
        line = TR._window(self.rows(7), "2026-10-07", 7)
        self.assertNotIn("[", line)
        self.assertNotIn("-0.033", line)
        self.assertIn("ДИ не определён", line)
        self.assertIn("-0.036", line)                       # the mean is still shown
        self.assertIn("рано судить", line)

    def test_ten_days_bounds_shown(self):
        import trader_report as TR
        line = TR._window(self.rows(10), "2026-10-10", None)
        self.assertIn("[", line)
        self.assertNotIn("ДИ не определён", line)
