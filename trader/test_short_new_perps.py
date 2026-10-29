"""SN -- short new Binance perps after launch (short-new-perps-n6-1029-spec.md §8). Written BEFORE shortperp.py /
experiment_sn.py. Synthetic 15m perp bars, offline.

Interface the tests fix:
  launches(first_bar: {sym: ms}, cutoff_ms) -> {sym: launch ms}           first bar after the cut-off
  entry_ms_for(arr, launch_ms, d) -> open ms of bar d after the first one | None
  trade(arr, funding, entry_ms, H_days, fees) -> {entry_close_ms, exit_close_ms, gross_mult, marks, reason, ...} | None
        gross_mult = value of 1 USDT of margin at exit; marks = {00:00 ms: value} while open (experiment_la.book format)
  book = experiment_la.book (reused)
  old_pool(first_bar, launches, t_ms) -> perps first listed >= 180 days before t and not newly launched within +-30 days
  control(entries, first_bar, launches, seed) -> entries with the coin replaced, same time, reproducible
  in_train(trade, test_from_ms) / admit(res) / select(res, pbo) / decide(agg, rand, liq_share, n) / main(["test"]) == 2
"""
import random
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BAR = 15 * 60 * 1000
DAY = 86_400_000


def ms(s):
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)


T0 = ms("2025-01-01 00:00")


def bars(closes, t0=T0, highs=None):
    c = np.asarray(closes, dtype=float)
    return {"t": np.arange(len(c), dtype=np.int64) * BAR + t0, "o": c.copy(),
            "h": np.asarray(highs, float) if highs is not None else c.copy(), "l": c.copy(), "c": c, "v": np.ones(len(c))}


def zero_fees():
    import carry as C
    return C.Fees(spot_taker=0.0, perp_taker=0.0, slip=0.0)


class TestLaunchesAndTiming(unittest.TestCase):
    def test_launches_after_cutoff(self):
        import shortperp as S
        fb = {"OLDUSDT": ms("2022-06-01 00:00"), "NEWUSDT": ms("2024-05-01 08:00"), "EDGEUSDT": ms("2023-01-08 00:00")}
        self.assertEqual(S.launches(fb, ms("2023-01-08 00:00")), {"NEWUSDT": ms("2024-05-01 08:00")})

    def test_entry_bar_and_full_window(self):
        import shortperp as S
        a = bars([100.0] * (96 * 3))
        self.assertEqual(S.entry_ms_for(a, T0, 0), T0)
        self.assertEqual(S.entry_ms_for(a, T0, 96), T0 + 96 * BAR)
        self.assertIsNone(S.entry_ms_for(a, T0, 96 * 3))
        self.assertIsNotNone(S.trade(a, None, T0, 1, zero_fees()))
        self.assertIsNone(S.trade(a, None, T0 + 96 * BAR * 2, 1, zero_fees()))        # 1 day does not fit
        x = S.trade(a, None, T0, 1, zero_fees())
        self.assertEqual(x["exit_close_ms"], T0 + 96 * BAR + BAR)
        self.assertTrue(S.in_train(x, T0 + 3 * DAY))
        self.assertFalse(S.in_train(x, T0 + DAY))


class TestMoney(unittest.TestCase):
    def test_short_pnl_and_funding(self):
        import shortperp as S
        n = 96 * 2
        down = bars([100.0] + [100.0] * 95 + [90.0] * (n - 96))
        x = S.trade(down, None, T0, 1, zero_fees())
        self.assertAlmostEqual(x["gross_mult"], 1.10, places=9)
        up = bars([100.0] + [100.0] * 95 + [110.0] * (n - 96))
        self.assertAlmostEqual(S.trade(up, None, T0, 1, zero_fees())["gross_mult"], 0.90, places=9)
        flat = bars([100.0] * n)
        f = {"t": np.array([T0 + 8 * 3600 * 1000], dtype=np.int64), "rate": np.array([0.0001])}
        self.assertAlmostEqual(S.trade(flat, f, T0, 1, zero_fees())["gross_mult"], 1.0001, places=9)
        self.assertEqual(x["reason"], "time")

    def test_liquidation(self):
        import shortperp as S
        n = 96 * 2
        c = [100.0] * n
        hi = list(c)
        hi[10] = 200.0                                   # >= 100 * (1 + 1 - 0.01)
        x = S.trade(bars(c, highs=hi), None, T0, 1, zero_fees())
        self.assertEqual(x["reason"], "liquidation")
        self.assertAlmostEqual(x["gross_mult"], 0.0, places=9)
        self.assertEqual(x["exit_close_ms"], T0 + 10 * BAR + BAR)

    def test_book_is_reused_and_conserves(self):
        import experiment_la as EL
        import shortperp as S
        self.assertIs(S.book, EL.book)
        n = 96 * 4
        trades = []
        for k in range(12):
            x = S.trade(bars([100.0] * 96 + [95.0] * (n - 96)), None, T0, 1, zero_fees())
            trades.append({**x, "sym": f"S{k}USDT"})
        r = S.book(trades, "2025-01-01", "2025-01-04", slots=10, start_equity=1000.0)
        self.assertEqual(len(r["trades"]), 10)
        self.assertEqual(r["skipped_full"], 2)
        self.assertTrue(r["conservation_ok"])


class TestControl(unittest.TestCase):
    def test_old_pool_and_reproducible(self):
        import shortperp as S
        t = ms("2025-06-01 00:00")
        fb = {"OLDAUSDT": ms("2023-01-01 00:00"), "OLDBUSDT": ms("2024-01-01 00:00"), "YOUNGUSDT": ms("2025-03-01 00:00"),
              "NEWUSDT": ms("2025-05-25 00:00")}
        lau = {"YOUNGUSDT": fb["YOUNGUSDT"], "NEWUSDT": fb["NEWUSDT"]}
        self.assertEqual(S.old_pool(fb, lau, t), ["OLDAUSDT", "OLDBUSDT"])
        ent = [{"sym": "NEWUSDT", "entry_ms": t}]
        a = S.control(ent, fb, lau, seed=4)
        self.assertEqual(a, S.control(ent, fb, lau, seed=4))
        self.assertIn(a[0]["sym"], ("OLDAUSDT", "OLDBUSDT"))
        self.assertEqual(a[0]["entry_ms"], t)


class TestDecisions(unittest.TestCase):
    def good(self, **kw):
        r = {"agg": {"mean_pct": 0.02, "lo95": 0.001}, "rand": [0.0] * 50, "n_trades": 100, "liq_share": 0.05,
             "r_by_year": {"2023": 0.01, "2024": -0.01, "2025": 0.02}}
        r.update(kw)
        return r

    def test_admit_select(self):
        import experiment_sn as EX
        self.assertEqual(len(EX.candidates()), 8)
        self.assertTrue(EX.admit(self.good())[0])
        for k, v in (("agg", {"mean_pct": 0.02, "lo95": -0.001}), ("rand", [0.03] * 50), ("n_trades", 99),
                     ("liq_share", 0.051), ("r_by_year", {"2023": -0.01, "2024": -0.01, "2025": 0.02})):
            self.assertFalse(EX.admit(self.good(**{k: v}))[0], k)
        res = {"a": dict(self.good(), admitted=True), "b": dict(self.good(agg={"mean_pct": 0.05, "lo95": 0.01}), admitted=True)}
        self.assertEqual(EX.select(res, 0.3), "b")
        self.assertIsNone(EX.select(res, 0.5))

    def test_decide_and_operator(self):
        import experiment_sn as EX
        self.assertEqual(EX.decide({"mean_pct": 0.05, "lo95": 0.01}, [0.0] * 50, 0.0, 14)["verdict"], "TOO_EARLY")
        self.assertEqual(EX.decide({"mean_pct": 0.05, "lo95": 0.01}, [0.0] * 50, 0.0, 15)["verdict"], "ACCEPTED")
        self.assertEqual(EX.decide({"mean_pct": 0.05, "lo95": -0.01}, [0.0] * 50, 0.0, 15)["verdict"], "REJECTED")
        self.assertEqual(EX.decide({"mean_pct": 0.05, "lo95": 0.01}, [0.0] * 50, 0.06, 15)["verdict"], "REJECTED")
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "M.json"
            EX.open_test(m, {})
            with self.assertRaises(RuntimeError):
                EX.open_test(m, {})
        self.assertEqual(EX.main(["test"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
