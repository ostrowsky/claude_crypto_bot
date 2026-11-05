"""W-SHADOW readout with per-bucket costs (wide-scan-shadow-w1-1028-spec.md, entry of 2026-11-05). Written BEFORE
cost_adjust.py. Offline.

Interface the tests fix:
  cost_adjust.trade_delta_pp(sym, notional, table, ranks) -> extra round-trip cost vs the model's 8 bp, pp
  cost_adjust.adjusted_series(series, exits, day_start_eq, table, ranks) -> [(day, R %)] with
       R_d - sum over exits on day d of delta_pp * cost / equity_d
  cost_adjust.load(cost_cal_path) -> (table, ranks)   from COST_CAL.json (tables.live, ranks)
"""
import json
import tempfile
import unittest
from pathlib import Path


def table():
    def side(m):
        return {str(s): {"median": m * (1 + (s >= 5000))} for s in (100, 500, 1000, 5000)}
    return {"B1": {"buy": side(0.01), "sell": side(0.01)}, "B2": {"buy": side(0.6), "sell": side(0.6)},
            "B3": {"buy": side(3.3), "sell": side(3.3)}, "B4": {"buy": side(11.5), "sell": side(11.5)}}


class TestDelta(unittest.TestCase):
    def test_bucket_and_size(self):
        import cost_adjust as CA
        ranks = {"BTCUSDT": 1, "SOLUSDT": 3, "AUSDT": 50, "TAILUSDT": 300}
        t = table()
        self.assertAlmostEqual(CA.trade_delta_pp("BTCUSDT", 1000, t, ranks), (0.02 - 16) / 100)
        self.assertAlmostEqual(CA.trade_delta_pp("AUSDT", 1000, t, ranks), (6.6 - 16) / 100)
        self.assertAlmostEqual(CA.trade_delta_pp("TAILUSDT", 1000, t, ranks), (23.0 - 16) / 100)
        self.assertAlmostEqual(CA.trade_delta_pp("TAILUSDT", 4000, t, ranks), (46.0 - 16) / 100)   # nearest 5 000
        self.assertAlmostEqual(CA.trade_delta_pp("NEWUSDT", 1000, t, ranks), (23.0 - 16) / 100)    # unranked -> B4


class TestSeries(unittest.TestCase):
    def test_adjusted_series(self):
        import cost_adjust as CA
        ranks = {"TAILUSDT": 300, "BTCUSDT": 1}
        series = [("2026-11-01", 1.0), ("2026-11-02", -0.5)]
        exits = [{"sym": "TAILUSDT", "cost": 1000.0, "exit_time": "2026-11-01T10:00:00+00:00"},
                 {"sym": "BTCUSDT", "cost": 1000.0, "exit_time": "2026-11-02T03:00:00+00:00"}]
        eq = {"2026-11-01": 10_000.0, "2026-11-02": 9_000.0}
        s = dict(CA.adjusted_series(series, exits, eq, table(), ranks))
        self.assertAlmostEqual(s["2026-11-01"], 1.0 - 0.07 * 1000 / 10_000)
        self.assertAlmostEqual(s["2026-11-02"], -0.5 - (0.02 - 16) / 100 * 1000 / 9_000)

    def test_load(self):
        import cost_adjust as CA
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "COST_CAL.json"
            p.write_text(json.dumps({"tables": {"live": table()}, "ranks": {"BTCUSDT": 1}}), encoding="utf-8")
            t, r = CA.load(p)
            self.assertEqual(r["BTCUSDT"], 1)
            self.assertIn("B4", t)


class TestLoadAndReadout(unittest.TestCase):
    def test_load_without_ranks_refuses(self):
        import cost_adjust as CA
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "COST_CAL.json"
            p.write_text(json.dumps({"tables": {"live": table()}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                CA.load(p)

    def test_adjusted_paired_on_synthetic_books(self):
        import wide_scan as W
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cc = root / "COST_CAL.json"
            cc.write_text(json.dumps({"tables": {"live": table()}, "ranks": {"BTCUSDT": 1}}), encoding="utf-8")
            days = [f"2026-11-{i:02d}" for i in range(1, 13)]
            for name, book, sym in (("w", "x4_wide", "TAILUSDT"), ("e", "x4_floor8", "BTCUSDT")):
                (root / name).mkdir()
                (root / name / "state.json").write_text(json.dumps(
                    {"books": {book: {"day_start_eq": {dd: 10_000.0 for dd in days}}}}), encoding="utf-8")
                (root / name / "ledger.jsonl").write_text("".join(json.dumps(
                    {"type": "exit", "book": book, "sym": sym, "cost": 1000.0, "exit_time": dd + "T05:00:00+00:00"})
                    + "\n" for dd in days), encoding="utf-8")
            flat = {dd: 0.0 for dd in days}
            pr = W.adjusted_paired(days, flat, dict(flat), root / "w", root / "e", cost_cal=cc)
            # wide pays the tail (+0.07 pp per trade), E7 saves on BTC (-0.1598 pp): both shift the gap down
            self.assertAlmostEqual(pr["mean_pct"], -(0.07 + 0.1598) * 1000 / 10_000, delta=1e-4)   # aggregate rounds to 4


if __name__ == "__main__":
    unittest.main(verbosity=2)
