"""Entry price gap -- tests written from entry-price-gap-1013-spec.md §6 BEFORE the code
(operator rule 2026-10-10: spec -> tests -> code)."""
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from test_demo_connection_a import FakeResp

HERE = Path(__file__).resolve().parent
T = datetime(2026, 10, 13, 17, 31, 34, tzinfo=timezone.utc)
MS = lambda d: int(d.timestamp() * 1000)  # noqa: E731


def minute(d):
    return d.replace(second=0, microsecond=0)


class TestPriceAndDelay(unittest.TestCase):
    """§6.1–6.2."""

    def test_price_at_minute_containing_t(self):
        import experiment_entry_gap as G
        kl = {MS(minute(T)): (0.1395, 0.1399), MS(minute(T) - timedelta(minutes=1)): (0.1380, 0.1385)}
        self.assertEqual(G.price_at(kl, T), (0.1395, 0.1399, (0.1395 + 0.1399) / 2))
        self.assertIsNone(G.price_at(kl, T + timedelta(minutes=5)))

    def test_bar_close_and_delay(self):
        import experiment_entry_gap as G
        self.assertEqual(G.bar_close(T, "15m"), datetime(2026, 10, 13, 17, 30, tzinfo=timezone.utc))
        self.assertEqual(G.bar_close(T, "1h"), datetime(2026, 10, 13, 17, 0, tzinfo=timezone.utc))
        self.assertAlmostEqual(G.delay_sec(T, "15m"), 94.0)


class TestWindows(unittest.TestCase):
    """§6.3."""

    def test_windows_cover_all_entries_minimal(self):
        import experiment_entry_gap as G
        ents = [("AUSDT", T), ("AUSDT", T + timedelta(minutes=500)), ("AUSDT", T + timedelta(minutes=1200)),
                ("BUSDT", T)]
        w = G.windows(ents)
        self.assertEqual(sorted(s for s, _ in w), ["AUSDT", "AUSDT", "BUSDT"])
        for sym, t in ents:                                   # every entry minute lies in one of its windows
            m = MS(minute(t))
            self.assertTrue(any(s == sym and st <= m < st + 1000 * 60_000 for s, st in w), (sym, t))


class TestSummary(unittest.TestCase):
    """§6.4."""

    def test_summary_numbers_and_determinism(self):
        import experiment_entry_gap as G
        rows = [{"day": f"2026-10-{d:02d}", "gap_bps": v} for d in range(1, 21) for v in (10.0, 30.0)]
        a, b = G.summary(rows, seed=7), G.summary(rows, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(a["n"], 40)
        self.assertAlmostEqual(a["mean"], 20.0)
        self.assertAlmostEqual(a["median"], 20.0)
        self.assertAlmostEqual(a["pos_share"], 1.0)
        self.assertLessEqual(a["lo95"], 20.0)
        self.assertGreaterEqual(a["hi95"], 20.0)
        self.assertGreater(a["lo95"], 0)
        z = G.summary([{"day": f"2026-10-{d:02d}", "gap_bps": (-1) ** d * 5.0} for d in range(1, 21)], seed=7)
        self.assertLess(z["lo95"], 0)
        self.assertGreater(z["hi95"], 0)

    def test_verdict_rule(self):
        import experiment_entry_gap as G
        self.assertTrue(G.systematic({"lo95": 9.0, "hi95": 30.0, "mean": 20.0}))
        self.assertFalse(G.systematic({"lo95": 1.0, "hi95": 9.0, "mean": 5.0}))      # |mean| < 8 bps
        self.assertFalse(G.systematic({"lo95": -2.0, "hi95": 30.0, "mean": 14.0}))   # CI spans 0
        self.assertTrue(G.systematic({"lo95": -30.0, "hi95": -9.0, "mean": -20.0}))


class TestApplyGap(unittest.TestCase):
    """§6.5."""

    def test_copies_with_market_price(self):
        import experiment_entry_gap as G
        tr = [{"sym": "AUSDT", "tf": "15m", "ts": T, "price": 0.13815},
              {"sym": "BUSDT", "tf": "15m", "ts": T, "price": 2.0}]
        out, missing = G.apply_gap(tr, {G.trade_key(tr[0]): 0.1397})
        self.assertEqual(out[0]["price"], 0.1397)
        self.assertEqual(out[1]["price"], 2.0)
        self.assertEqual(missing, 1)
        self.assertEqual(tr[0]["price"], 0.13815)                                    # originals untouched
        self.assertIsNot(out[0], tr[0])


class TestCacheAndClient(unittest.TestCase):
    """§6.6–6.7."""

    def test_cache_second_run_offline(self):
        import experiment_entry_gap as G

        class C:
            def __init__(self):
                self.n = 0

            def klines_from(self, symbol, interval, start_ms, limit=1000, base=None):
                self.n += 1
                return [[start_ms + i * 60_000, "1.0", "1.1", "0.9", str(1.0 + i / 1000), "5", start_ms + (i + 1) * 60_000 - 1]
                        for i in range(3)]
        c = C()
        with tempfile.TemporaryDirectory() as d:
            st = MS(minute(T))
            k1 = G.fetch_window(c, "AUSDT", st, Path(d))
            k2 = G.fetch_window(c, "AUSDT", st, Path(d))
            self.assertEqual(c.n, 1)
            self.assertEqual(k1, k2)
            self.assertEqual(k1[st], (1.0, 1.0))

    def test_broken_cache_line_is_refetched(self):
        """A process killed mid-append leaves a partial last line: it must not break the next run."""
        import experiment_entry_gap as G

        class C:
            n = 0

            def klines_from(self, symbol, interval, start_ms, limit=1000, base=None):
                C.n += 1
                return [[start_ms, "1.0", "1.1", "0.9", "1.05", "5", start_ms + 59_999]]
        with tempfile.TemporaryDirectory() as d:
            st = MS(minute(T))
            (Path(d) / "AUSDT.jsonl").write_text('{"start": %d, "rows": [[%d, 1.0' % (st, st), encoding="utf-8")
            k = G.fetch_window(C(), "AUSDT", st, Path(d))
            self.assertEqual(C.n, 1)
            self.assertEqual(k[st], (1.0, 1.05))

    def test_measure_parallel_same_result(self):
        import experiment_entry_gap as G

        class C:
            def klines_from(self, symbol, interval, start_ms, limit=1000, base=None):
                base_px = {"AUSDT": 1.0, "BUSDT": 2.0}[symbol]
                return [[start_ms + i * 60_000, str(base_px), "0", "0", str(base_px * 1.01), "1", 0] for i in range(1000)]
        tr = [{"sym": s, "tf": "15m", "ts": T + timedelta(minutes=m), "price": px, "mode": "trend"}
              for s, px in (("AUSDT", 1.0), ("BUSDT", 2.0)) for m in (0, 30, 2000)]
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            a = G.measure(tr, C(), Path(d1), log=lambda *_: None, workers=1)
            b = G.measure(tr, C(), Path(d2), log=lambda *_: None, workers=4)
        self.assertEqual(a, b)
        self.assertTrue(all(abs(r["gap_bps"] - 50.0) < 1e-6 for r in a))

    def test_client_klines_from_url(self):
        import exchange_client as X
        seen = []

        def opener(req, timeout=None):
            seen.append(req.full_url)
            return FakeResp([])
        c = X.ExchangeClient(key="", secret="", opener=opener)
        c.klines_from("AUSDT", "1m", 1_760_000_000_000, 1000, base=X.LIVE_REST)
        self.assertIn("api.binance.com/api/v3/klines?", seen[-1])
        for part in ("symbol=AUSDT", "interval=1m", "startTime=1760000000000", "limit=1000"):
            self.assertIn(part, seen[-1])

    def test_boundary(self):
        src = (HERE / "experiment_entry_gap.py").read_text(encoding="utf-8")
        self.assertNotIn('"' + '.runtime' + '"', src)
        self.assertNotIn("telegram", src.lower())
        self.assertNotIn("place_order", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
