"""EX-RL1 phase 0 -- market vs limit entry on the aggTrades tape (execution-bandit-exrl1-1029-spec.md §5).
Written BEFORE exec_tape.py / experiment_exrl1.py. Synthetic tapes, offline, no network.

Interface the tests fix:
  exec_tape.a0_price(tape, t, lag)                       first buyer-taker trade at or after t + lag | None
  exec_tape.limit_ref(tape, t, lag)                      last seller-taker price at or before t + lag | None
  exec_tape.a1_price(tape, t, lag, timeout)              limit if a trade < limit in (t+lag, t+lag+timeout], else market
  exec_tape.a2_fill(tape, t, lag, timeout)               (filled, price | None)
  exec_tape.pair_diff_bp(a0, arm, ret_pct, skipped)      money difference vs A0, bp of the trade's capital
  exec_tape.calibration(sim, actual)                     {median_abs_bp, ok}
  exec_tape.fetch_tape(client, sym, t0, t1) / tape_path / have_tape
  experiment_exrl1.admit(res)
tape = {"T": ms array, "p": price array, "m": isBuyerMaker bool array}   (m True = seller was the taker)
"""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

S = 1000
T0 = 1_800_000_000_000


def tape(rows):
    T, p, m = zip(*rows)
    return {"T": np.array(T, dtype=np.int64), "p": np.array(p, float), "m": np.array(m, bool)}


class TestArms(unittest.TestCase):
    def test_a0_first_buyer_taker_after_lag(self):
        import exec_tape as E
        tp = tape([(T0 + 10 * S, 100.0, False), (T0 + 31 * S, 99.9, True), (T0 + 32 * S, 100.2, False),
                   (T0 + 40 * S, 100.3, False)])
        self.assertEqual(E.a0_price(tp, T0, 30 * S), 100.2)
        self.assertIsNone(E.a0_price(tape([(T0 + 31 * S, 99.9, True)]), T0, 30 * S))

    def test_limit_ref_and_a1(self):
        import exec_tape as E
        rows = [(T0 + 5 * S, 99.8, True), (T0 + 20 * S, 99.9, True), (T0 + 35 * S, 100.1, False),
                (T0 + 60 * S, 99.9, True),                            # AT the limit: queue not proven passed
                (T0 + 200 * S, 100.4, False)]
        tp = tape(rows)
        self.assertEqual(E.limit_ref(tp, T0, 30 * S), 99.9)
        self.assertEqual(E.a1_price(tp, T0, 30 * S, 120 * S), 100.4)    # no trade < 99.9 in 2 min -> market after
        filled = tape(rows[:4] + [(T0 + 90 * S, 99.85, True), (T0 + 200 * S, 100.4, False)])
        self.assertEqual(E.a1_price(filled, T0, 30 * S, 120 * S), 99.9)
        no_ref = tape([(T0 + 35 * S, 100.1, False)])
        self.assertEqual(E.a1_price(no_ref, T0, 30 * S, 120 * S), 100.1)  # nothing to rest on -> same as A0

    def test_a2(self):
        import exec_tape as E
        tp = tape([(T0 + 20 * S, 99.9, True), (T0 + 400 * S, 99.8, True)])
        self.assertEqual(E.a2_fill(tp, T0, 30 * S, 900 * S), (True, 99.9))
        tp = tape([(T0 + 20 * S, 99.9, True), (T0 + 1000 * S, 99.8, True)])
        self.assertEqual(E.a2_fill(tp, T0, 30 * S, 900 * S), (False, None))

    def test_pair_diff(self):
        import exec_tape as E
        self.assertAlmostEqual(E.pair_diff_bp(100.2, 100.0, 0.0, False), 20.0, places=6)    # cheaper entry: +
        self.assertAlmostEqual(E.pair_diff_bp(100.0, 100.0, 5.0, False), 0.0)
        self.assertAlmostEqual(E.pair_diff_bp(100.0, None, -1.5, True), 150.0)            # skipped loser: +
        self.assertAlmostEqual(E.pair_diff_bp(100.0, None, 2.0, True), -200.0)            # skipped winner: -

    def test_calibration(self):
        import exec_tape as E
        r = E.calibration([100.0, 50.0, 10.0], [100.1, 50.0, 10.02])
        self.assertAlmostEqual(r["median_abs_bp"], 10.0, places=3)
        self.assertTrue(r["ok"])
        self.assertFalse(E.calibration([100.0, 50.0], [101.0, 50.5])["ok"])


class TestFetch(unittest.TestCase):
    def test_fetch_pages_by_time_without_duplicates(self):
        import exec_tape as E
        rows = [{"a": i, "p": str(100 + i / 1000), "q": "1", "T": T0 + i * 100, "m": bool(i % 2)} for i in range(2500)]
        calls = []

        class Fake:
            def agg_trades(self, symbol, start_ms, end_ms, limit=1000):
                assert end_ms - start_ms <= 3_600_000
                calls.append((start_ms, end_ms))
                return [r for r in rows if start_ms <= r["T"] <= end_ms][:limit]
        tp = E.fetch_tape(Fake(), "XUSDT", T0, T0 + 2499 * 100)
        self.assertEqual(len(tp["T"]), 2500)
        self.assertTrue(np.all(np.diff(tp["T"]) >= 0))
        self.assertEqual(len(set(tp["a"].tolist())), 2500)
        self.assertGreaterEqual(len(calls), 3)

    def test_store_skips_existing(self):
        import exec_tape as E
        with tempfile.TemporaryDirectory() as d:
            p = E.tape_path("XUSDT", T0, root=Path(d))
            self.assertFalse(E.have_tape("XUSDT", T0, root=Path(d)))
            E.save_tape(p, tape([(T0, 1.0, True)]))
            self.assertTrue(E.have_tape("XUSDT", T0, root=Path(d)))
            self.assertEqual(len(E.load_tape(p)["T"]), 1)


class TestAdmission(unittest.TestCase):
    def test_admit(self):
        import experiment_exrl1 as X
        good = {"calibration_ok": True, "agg": {"lo95": 0.1}, "n_changed": 100,
                "by_month": {f"2026-0{m}": 1.0 for m in range(3, 9)} | {"2026-09": -1.0, "2026-10": -1.0}}
        self.assertTrue(X.admit(good)[0])
        for k, v in (("calibration_ok", False), ("agg", {"lo95": -0.1}), ("n_changed", 99),
                     ("by_month", {f"2026-0{m}": (1.0 if m < 7 else -1.0) for m in range(3, 10)} | {"2026-10": -1.0})):
            self.assertFalse(X.admit(dict(good, **{k: v}))[0], k)


if __name__ == "__main__":
    unittest.main(verbosity=2)
