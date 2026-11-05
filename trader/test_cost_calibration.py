"""Mainnet cost calibration (cost-calibration-mainnet-1105-spec.md §6). Written BEFORE cost_calibration.py.
Synthetic records, offline.

Interface the tests fix:
  cost_calibration.bucket(sym, rank) -> "B1" | "B2" | "B3" | "B4"
  cost_calibration.pair_stats(values, min_n=50) -> {"median", "p90", "none_share", "n"} | None
  cost_calibration.bucket_stats(stats_list) -> {"median", "p90", "none_share", "n_pairs"}
  cost_calibration.nearest_size(notional) -> one of SIZES
  cost_calibration.drift(entries, book_index, window_sec=1800) -> (drifts, n_excluded)
       entries: [{"sym", "ts" (epoch s), "slippage_bps", "notional"}]
       book_index: {sym: [(ts epoch s, {size: buy_bps})]} sorted by ts
  cost_calibration.trade_delta_pp(entry_bps, exit_bps, model_bps=8.0) -> extra cost per round trip, pp
  cost_calibration.iter_records(dir, first_day, last_day) -> records of files with first_day <= day <= last_day
"""
import json
import tempfile
import unittest
from pathlib import Path


class TestBuckets(unittest.TestCase):
    def test_bucket(self):
        import cost_calibration as C
        self.assertEqual(C.bucket("BTCUSDT", 1), "B1")
        self.assertEqual(C.bucket("ETHUSDT", 2), "B1")
        self.assertEqual(C.bucket("ETHUSDT", 7), "B1")                 # by name, whatever the rank
        self.assertEqual(C.bucket("SOLUSDT", 3), "B2")
        self.assertEqual(C.bucket("XUSDT", 20), "B2")
        self.assertEqual(C.bucket("XUSDT", 21), "B3")
        self.assertEqual(C.bucket("XUSDT", 100), "B3")
        self.assertEqual(C.bucket("XUSDT", 101), "B4")


class TestStats(unittest.TestCase):
    def test_pair_stats(self):
        import cost_calibration as C
        vals = [float(v) for v in range(1, 101)] + [None] * 25
        s = C.pair_stats(vals, min_n=50)
        self.assertAlmostEqual(s["median"], 50.5)
        self.assertAlmostEqual(s["p90"], 90.1, places=1)
        self.assertAlmostEqual(s["none_share"], 0.2)
        self.assertEqual(s["n"], 125)
        self.assertIsNone(C.pair_stats([1.0] * 49, min_n=50))

    def test_bucket_stats(self):
        import cost_calibration as C
        b = C.bucket_stats([{"median": 2.0, "p90": 5.0, "none_share": 0.0, "n": 60},
                            {"median": 4.0, "p90": 9.0, "none_share": 0.5, "n": 60},
                            {"median": 9.0, "p90": 20.0, "none_share": 0.1, "n": 60}])
        self.assertAlmostEqual(b["median"], 4.0)
        self.assertGreaterEqual(b["p90"], 9.0)
        self.assertEqual(b["n_pairs"], 3)
        self.assertAlmostEqual(b["none_share"], 0.2)


class TestDrift(unittest.TestCase):
    def test_drift_window_and_size(self):
        import cost_calibration as C
        self.assertEqual(C.nearest_size(80), 100)
        self.assertEqual(C.nearest_size(700), 500)
        self.assertEqual(C.nearest_size(900), 1000)
        idx = {"AUSDT": [(1000.0, {100: 3.0, 500: 4.0, 1000: 6.0, 5000: 12.0}),
                         (5000.0, {100: 1.0, 500: 2.0, 1000: 3.0, 5000: 9.0})]}
        entries = [{"sym": "AUSDT", "ts": 1200.0, "slippage_bps": 10.0, "notional": 950.0},    # snap 1000: 10 - 6
                   {"sym": "AUSDT", "ts": 4900.0, "slippage_bps": 5.0, "notional": 120.0},     # snap 5000: 5 - 1
                   {"sym": "AUSDT", "ts": 9000.0, "slippage_bps": 5.0, "notional": 120.0},     # > 30 min away
                   {"sym": "BUSDT", "ts": 1000.0, "slippage_bps": 5.0, "notional": 120.0}]     # no book
        d, excl = C.drift(entries, idx, window_sec=1800)
        self.assertEqual(sorted(d), [4.0, 4.0])
        self.assertEqual(excl, 2)

    def test_trade_delta(self):
        import cost_calibration as C
        self.assertAlmostEqual(C.trade_delta_pp(8.0, 8.0), 0.0)
        self.assertAlmostEqual(C.trade_delta_pp(10.0, 12.0), 0.06)
        self.assertAlmostEqual(C.trade_delta_pp(3.0, 3.0), -0.10)


class TestRead(unittest.TestCase):
    def test_iter_records_window_and_missing_depth(self):
        import cost_calibration as C
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "2026-10-10.jsonl").write_text(json.dumps({"ts": "2026-10-10T00:00:00+00:00",
                                                            "depth": {"live": {"AUSDT": {}}}}) + "\n\n", encoding="utf-8")
            (p / "2026-11-04.jsonl").write_text(json.dumps({"ts": "2026-11-04T00:00:00+00:00"}) + "\n", encoding="utf-8")
            (p / "2026-11-05.jsonl").write_text(json.dumps({"ts": "2026-11-05T00:00:00+00:00"}) + "\n", encoding="utf-8")
            (p / "_state.json").write_text("{}", encoding="utf-8")
            recs = list(C.iter_records(p, "2026-10-10", "2026-11-04"))
            self.assertEqual([r["ts"][:10] for r in recs], ["2026-10-10", "2026-11-04"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
