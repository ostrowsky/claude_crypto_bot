"""FC-DEMO blockers before the demo mode (fc-demo-execution-1030-spec.md §6, 2026-10-30). Written BEFORE the fixes.

  1. executor.fc_reserved_pairs(cfg, fc_state) -> set: FC pairs are closed to X4 while FC runs in demo mode;
     executor.fc_spot_value(fc_state, price) -> USDT of the FC spot legs (excluded from the X4 kill-switch equity)
  2. fc_demo.fc_spot_qty_at(decisions, ts) -> {asset: qty} of the demo FC spot legs at a moment (from fc_entry / fc_exit)
     fc_demo.fc_equity_at(decisions, ts) -> FC equity change of the first demo fc_mark at or after ts, or None
  3. daily_revenue.compute_day(day, snaps, fills, fc=...) books the FC spot leg outside the policy revenue:
     fc_spot_contrib_pct, and R_fc_pct = FC both legs from the marks
  4. FC_DEMO_NOTIONAL_USDT fits the free spot USDT (2 x N <= 4 900)
"""
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace


class TestExecutorReserve(unittest.TestCase):
    def test_reserved_only_in_demo_mode(self):
        import executor as E
        cfg = SimpleNamespace(FC_DEMO_ENABLED=True, FC_DEMO_MODE="demo", FC_DEMO_PAIRS=("BTCUSDT", "ETHUSDT"))
        self.assertEqual(E.fc_reserved_pairs(cfg, {"mode": "demo", "pos": {}}), {"BTCUSDT", "ETHUSDT"})
        self.assertEqual(E.fc_reserved_pairs(SimpleNamespace(**{**vars(cfg), "FC_DEMO_MODE": "paper"}), {"mode": "paper"}), set())
        self.assertEqual(E.fc_reserved_pairs(SimpleNamespace(**{**vars(cfg), "FC_DEMO_ENABLED": False}), None), set())
        # FC positions left on the account still reserve their pairs even if the flag was switched off
        self.assertEqual(E.fc_reserved_pairs(SimpleNamespace(**{**vars(cfg), "FC_DEMO_ENABLED": False}),
                                             {"mode": "demo", "pos": {"BTCUSDT": {"q_spot": 0.03}}}), {"BTCUSDT"})

    def test_spot_value(self):
        import executor as E
        st = {"mode": "demo", "pos": {"BTCUSDT": {"q_spot": 0.03}, "ETHUSDT": {"q_spot": 1.0}}}
        px = {"BTCUSDT": 80000.0, "ETHUSDT": 2500.0}
        self.assertAlmostEqual(E.fc_spot_value(st, px.get), 0.03 * 80000 + 2500)
        self.assertEqual(E.fc_spot_value({"mode": "paper", "pos": st["pos"]}, px.get), 0.0)
        self.assertEqual(E.fc_spot_value(None, px.get), 0.0)


def row(kind, ts, mode="demo", **kw):
    return {"kind": kind, "book": "fc_demo", "mode": mode, "ts": ts, **kw}


class TestFcJournal(unittest.TestCase):
    def test_qty_and_equity_at(self):
        import fc_demo as F
        dec = [row("fc_entry", "2026-10-30T00:20:00+00:00", sym="BTCUSDT", q_spot=0.03),
               row("fc_entry", "2026-10-30T00:20:01+00:00", sym="ETHUSDT", q_spot=1.0),
               row("fc_entry", "2026-10-30T00:20:01+00:00", mode="paper", sym="ETHUSDT", q_spot=9.0),
               row("fc_exit", "2026-10-30T09:00:00+00:00", sym="BTCUSDT", q_spot=0.03),
               row("fc_mark", "2026-10-31T00:20:00+00:00", equity_change=4.5),
               row("fc_mark", "2026-11-01T00:20:00+00:00", equity_change=6.0)]
        self.assertEqual(F.fc_spot_qty_at(dec, datetime(2026, 10, 30, 12, tzinfo=timezone.utc)), {"ETH": 1.0})
        self.assertEqual(F.fc_spot_qty_at(dec, datetime(2026, 10, 30, 8, tzinfo=timezone.utc)), {"BTC": 0.03, "ETH": 1.0})
        self.assertEqual(F.fc_equity_at(dec, datetime(2026, 10, 31, tzinfo=timezone.utc)), 4.5)
        self.assertIsNone(F.fc_equity_at(dec, datetime(2026, 11, 2, tzinfo=timezone.utc)))


class TestDailyRevenueFc(unittest.TestCase):
    def snaps(self, e0, e1, btc0, btc1, q_btc):
        def s(ts, e, px):
            return {"ts": ts, "equity_usdt": e, "unpriced_count": 0,
                    "holdings": {"USDT": {"qty": e - q_btc * px, "px": 1.0}, "BTC": {"qty": q_btc, "px": px}}}
        return [s("2026-10-30T00:00:30+00:00", e0, btc0), s("2026-10-31T00:00:30+00:00", e1, btc1)]

    def test_fc_spot_leg_outside_policy(self):
        import daily_revenue as DR
        q = 0.03
        e0 = 10000.0
        e1 = e0 + q * (82000 - 80000)                                  # only the FC spot leg moved
        sn = self.snaps(e0, e1, 80000.0, 82000.0, q)
        base = DR.compute_day("2026-10-30", sn, [])
        self.assertAlmostEqual(base["R_policy_pct"], 0.6, places=6)    # without FC info: booked as policy
        r = DR.compute_day("2026-10-30", sn, [], fc={"spot_qty": {"BTC": q}, "equity_start": 1.0, "equity_end": 3.5})
        self.assertAlmostEqual(r["fc_spot_contrib_pct"], 0.6, places=6)
        self.assertAlmostEqual(r["R_policy_pct"], 0.0, places=6)
        self.assertAlmostEqual(r["R_fc_pct"], 2.5 / e0 * 100, places=6)
        r2 = DR.compute_day("2026-10-30", sn, [], fc={"spot_qty": {"BTC": q}, "equity_start": None, "equity_end": 3.5})
        self.assertIsNone(r2["R_fc_pct"])


class TestReportLine(unittest.TestCase):
    def test_fc_line_in_report(self):
        import trader_report as TR
        rows = {"2026-10-30": {"day": "2026-10-30", "status": "FULL", "R_pct": 0.6, "stable_contrib_pct": 0.0,
                               "R_policy_pct": 0.0, "fc_spot_contrib_pct": 0.6, "R_fc_pct": 0.025}}
        txt = TR.build("2026-10-30", rows)
        self.assertIn("FC на демо", txt)
        self.assertIn("демо-фандинг", txt)
        self.assertNotIn("FC на демо", TR.build("2026-10-30", {"2026-10-30": {"day": "2026-10-30", "status": "FULL"}}))


class TestLock(unittest.TestCase):
    def test_single_step_at_a_time(self):
        import tempfile
        from pathlib import Path
        import fc_demo as F
        now = datetime(2026, 10, 30, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            lk = Path(d) / "step.lock"
            self.assertTrue(F.acquire_lock(lk, now))
            self.assertFalse(F.acquire_lock(lk, now))                    # a second step while the first runs
            F.release_lock(lk)
            self.assertTrue(F.acquire_lock(lk, now))
            from datetime import timedelta
            self.assertTrue(F.acquire_lock(lk, now + timedelta(minutes=20)))   # a stale lock (crashed step) is taken


class TestPerpFillBug(unittest.TestCase):
    """Bug 2026-10-30 (first demo step): demo fut_order RESULT answers FILLED + executedQty WITHOUT avgPrice; the code
    read price 0 as 'not filled', unwound the spot leg and left naked perp shorts (closed by hand the same minute)."""

    def test_filled_without_avg_price_is_a_fill(self):
        import fc_demo as F

        class Fut:
            def fut_order(self, sym, side, type_, **kw):
                return {"status": "FILLED", "executedQty": "0.0308", "orderId": 7}

            def fut_get_order(self, sym, order_id):
                return {"status": "FILLED", "executedQty": "0.0308", "avgPrice": "77775.0"}
        v = F.DemoVenue.__new__(F.DemoVenue)
        v.fut_c, v._setup = Fut(), {"BTCUSDT"}
        r = v._perp("BTCUSDT", "SELL", 0.0308, "cid", False)
        self.assertAlmostEqual(r["qty"], 0.0308)
        self.assertAlmostEqual(r["price"], 77775.0)

    def test_unwind_closes_an_opened_perp(self):
        import fc_demo as F
        from test_fc_demo import Venue, Log, cfg, at

        class V(Venue):
            def perp_sell(self, sym, qty, cid):
                self.calls.append(("perp_sell", sym, qty))
                self.opened = qty
                raise RuntimeError("answer unreadable")

            def perp_open_qty(self, sym):
                return getattr(self, "opened", 0.0)
        v, lg = V(), Log()
        e = F.FcDemo(cfg(FC_DEMO_PAIRS=("BTCUSDT",)), v, lg)
        e.step(at(0, 20), None)
        kinds = [c[0] for c in v.calls]
        self.assertIn("perp_buy", kinds)                                 # the opened short is closed
        self.assertIn("spot_sell", kinds)
        self.assertEqual(e.state["pos"], {})


class TestNotional(unittest.TestCase):
    def test_fits_free_spot_usdt(self):
        import trader_config as TC
        self.assertLessEqual(2 * TC.FC_DEMO_NOTIONAL_USDT, 4900.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
