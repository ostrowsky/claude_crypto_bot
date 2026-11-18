"""Bug 2026-11-18: the FC-DEMO spot leg leaked into the policy revenue from day 2026-11-04 on.

daily_revenue.compute_main rebuilt the FC spot quantity from the decision journals of the last 5 days RELATIVE TO NOW
(audit_report.load("decisions", 5)). FC-DEMO entered on 2026-10-30, so from the day computed on 11-05 on the fc_entry
rows were outside the window: spot_qty = {} -> fc_spot_contrib_pct 0 -> the hedged spot leg's price move was booked
as policy revenue (11-04 +2.53 %, 11-05 -3.35 %, 11-08 -1.60 %, 11-12 -1.23 % with no X4 position open).

Written BEFORE the fix (fc-demo-execution-1030-spec.md §6, entry 2026-11-18):
  1. compute_main(day) separates the FC spot leg however old the fc_entry row is, and for a day computed later (a
     recompute) it reads that day's own fc_marks: the journal is read in full, not as a window relative to now.
  2. load_fc_rows(root) reads every decision journal and keeps only the fc_demo rows.
  3. fc_context(rows, d0) -> the fc input of compute_day, or None when FC never ran in demo mode.
  4. A day whose fc_mark shows FC spot value > 0 while the journal knows no FC spot quantity is UNKNOWN
     (reason fc_spot_qty_unknown), never booked as policy revenue: the silent zero is what hid this for 14 days.
  5. A recompute without live benchmarks (bench=False) keeps the benchmarks of the previous revision of that day.
"""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

Q_BTC, Q_ETH = 0.03, 1.0


def _iso(dt):
    return dt.isoformat()


def _row(kind, ts, **kw):
    return {"kind": kind, "book": "fc_demo", "process": "fc_demo", "mode": "demo", "ts": _iso(ts), **kw}


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _snap(ts, btc_px, eth_px, usdt=2000.0):
    h = {"USDT": {"qty": usdt, "px": 1.0}, "BTC": {"qty": Q_BTC, "px": btc_px}, "ETH": {"qty": Q_ETH, "px": eth_px}}
    e = sum(v["qty"] * v["px"] for v in h.values())
    return {"ts": _iso(ts), "equity_usdt": e, "unpriced_count": 0, "holdings": h}


class _Runtime:
    """A temporary trader runtime: FC entered `entry_age` days before today, the day computed is `day_age` days ago."""

    def __init__(self, entry_age=12, day_age=7):
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        self.d0 = today - timedelta(days=day_age)
        self.day = self.d0.strftime("%Y-%m-%d")
        self.entry = today - timedelta(days=entry_age) + timedelta(hours=9, minutes=26)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        dec = self.root / "audit" / "decisions"
        _write_jsonl(dec / f"{self.entry:%Y-%m-%d}.jsonl",
                     [_row("fc_entry", self.entry, sym="BTCUSDT", q_spot=Q_BTC, q_perp=0.03),
                      _row("fc_entry", self.entry, sym="ETHUSDT", q_spot=Q_ETH, q_perp=1.0),
                      {"kind": "entry_decision", "book": "x4", "process": "shadow", "ts": _iso(self.entry)}])
        d = self.entry.replace(hour=0, minute=0) + timedelta(days=1)
        while d <= today:                                    # one fc_mark a day at 00:15, equity_change grows by 1
            n = (d - self.entry.replace(hour=0, minute=0)).days
            _write_jsonl(dec / f"{d:%Y-%m-%d}.jsonl",
                         [_row("fc_mark", d + timedelta(minutes=15), spot_value=4900.0, equity_change=float(n))])
            d += timedelta(days=1)
        # only the FC spot legs move during the day: BTC +2000, ETH -50 -> +10 USDT, all of it FC
        _write_jsonl(self.root / "trading" / "equity_snapshots.jsonl",
                     [_snap(self.d0 + timedelta(seconds=30), 80000.0, 2500.0),
                      _snap(self.d0 + timedelta(days=1, seconds=30), 82000.0, 2450.0)])
        self.e0 = 2000.0 + Q_BTC * 80000.0 + Q_ETH * 2500.0
        self.fc_usdt = Q_BTC * 2000.0 + Q_ETH * -50.0

    def patches(self):
        import audit as A
        import trader_config as TC
        return mock.patch.object(TC, "RUNTIME", self.root), mock.patch.object(A, "ROOT", self.root / "audit")

    def close(self):
        self.tmp.cleanup()


class TestComputeMainFcHistory(unittest.TestCase):
    def run_day(self, rt, **kw):
        import daily_revenue as DR
        p1, p2 = rt.patches()
        with p1, p2:
            return DR.compute_main(rt.day, bench=False, **kw)

    def test_old_fc_entry_still_separates_the_spot_leg(self):
        rt = _Runtime(entry_age=12, day_age=7)               # the entry is 12 days old, as 10-30 was on 11-11
        try:
            r = self.run_day(rt)
            self.assertIsNotNone(r.get("fc_spot_contrib_pct"), "FC spot leg not separated (journal window too short)")
            self.assertAlmostEqual(r["fc_spot_contrib_pct"], rt.fc_usdt / rt.e0 * 100, places=6)
            self.assertAlmostEqual(r["R_policy_pct"], 0.0, places=6)     # no X4 position: no policy revenue
            # both FC legs from that day's own marks (start of day = mark n, end = n + 1), not from marks near now
            self.assertAlmostEqual(r["R_fc_pct"], 1.0 / rt.e0 * 100, places=6)
        finally:
            rt.close()

    def test_recent_entry_unchanged(self):
        rt = _Runtime(entry_age=3, day_age=2)                # inside the old 5-day window: worked before, still works
        try:
            r = self.run_day(rt)
            self.assertAlmostEqual(r["fc_spot_contrib_pct"], rt.fc_usdt / rt.e0 * 100, places=6)
            self.assertAlmostEqual(r["R_policy_pct"], 0.0, places=6)
        finally:
            rt.close()

    def test_recompute_keeps_previous_benchmarks(self):
        rt = _Runtime(entry_age=12, day_age=7)
        try:
            import daily_revenue as DR
            path = rt.root / "metrics" / "daily_revenue.jsonl"
            DR.append_day(path, {"day": rt.day, "status": "FULL", "R_pct": 0.1, "bench_btc_pct": 0.42,
                                 "bench_universe_pct": -1.5, "bench_universe_n": 400, "bench_universe_of": 410,
                                 "bench_usdt_pct": 0.0})
            r = self.run_day(rt)
            self.assertEqual(r["revision"], 2)
            self.assertEqual(r["bench_btc_pct"], 0.42)
            self.assertEqual(r["bench_universe_pct"], -1.5)
            self.assertEqual(r["bench_universe_n"], 400)
            self.assertTrue(r.get("bench_carried"))
        finally:
            rt.close()


class TestFcRowsAndContext(unittest.TestCase):
    def test_load_fc_rows_reads_every_journal(self):
        import daily_revenue as DR
        rt = _Runtime(entry_age=40, day_age=7)
        try:
            rows = DR.load_fc_rows(rt.root / "audit" / "decisions")
            self.assertTrue(rows and all(r.get("book") == "fc_demo" for r in rows))
            self.assertEqual(sum(r["kind"] == "fc_entry" for r in rows), 2)
            self.assertEqual(sum(r["kind"] == "fc_mark" for r in rows), 40)
        finally:
            rt.close()

    def test_context(self):
        import daily_revenue as DR
        e = datetime(2026, 10, 30, 9, 26, tzinfo=timezone.utc)
        rows = [_row("fc_entry", e, sym="BTCUSDT", q_spot=Q_BTC), _row("fc_entry", e, sym="ETHUSDT", q_spot=Q_ETH),
                _row("fc_mark", datetime(2026, 11, 8, 0, 15, tzinfo=timezone.utc), spot_value=4000.0, equity_change=3.0),
                _row("fc_mark", datetime(2026, 11, 9, 0, 15, tzinfo=timezone.utc), spot_value=4010.0, equity_change=4.0)]
        c = DR.fc_context(rows, datetime(2026, 11, 8, tzinfo=timezone.utc))
        self.assertEqual(c["spot_qty"], {"BTC": Q_BTC, "ETH": Q_ETH})
        self.assertEqual((c["equity_start"], c["equity_end"]), (3.0, 4.0))
        self.assertFalse(c.get("spot_qty_unknown"))
        self.assertIsNone(DR.fc_context([dict(r, mode="paper") for r in rows], datetime(2026, 11, 8, tzinfo=timezone.utc)))
        self.assertIsNone(DR.fc_context([], datetime(2026, 11, 8, tzinfo=timezone.utc)))

    def test_context_flags_a_journal_without_the_legs(self):
        import daily_revenue as DR
        rows = [_row("fc_mark", datetime(2026, 11, 8, 0, 15, tzinfo=timezone.utc), spot_value=4000.0, equity_change=3.0)]
        c = DR.fc_context(rows, datetime(2026, 11, 8, tzinfo=timezone.utc))
        self.assertTrue(c["spot_qty_unknown"])
        # an entry between midnight and the mark is not a gap: the mark's own moment knows the legs
        e = datetime(2026, 11, 8, 0, 5, tzinfo=timezone.utc)
        rows2 = [_row("fc_entry", e, sym="BTCUSDT", q_spot=Q_BTC)] + rows
        self.assertFalse(DR.fc_context(rows2, datetime(2026, 11, 8, tzinfo=timezone.utc)).get("spot_qty_unknown"))


class TestUnknownFcLegIsNotPolicyRevenue(unittest.TestCase):
    def test_unknown_leg_makes_the_day_unknown(self):
        import daily_revenue as DR
        d0 = datetime(2026, 11, 8, tzinfo=timezone.utc)
        sn = [_snap(d0 + timedelta(seconds=30), 80000.0, 2500.0), _snap(d0 + timedelta(days=1, seconds=30), 82000.0, 2450.0)]
        r = DR.compute_day("2026-11-08", sn, [], fc={"spot_qty": {}, "spot_qty_unknown": True,
                                                     "equity_start": 3.0, "equity_end": 4.0})
        self.assertEqual(r["status"], "UNKNOWN")
        self.assertEqual(r["reason"], "fc_spot_qty_unknown")
        self.assertIsNone(r["R_policy_pct"])
        self.assertIsNotNone(r["R_pct"])                     # the account return itself is still known


if __name__ == "__main__":
    unittest.main()
