"""LQ -- rebound after a long-liquidation flush (liquidation-flush-lq-1030-spec.md §6). Written BEFORE lq.py /
experiment_lq.py. Synthetic arrays, offline.

Interface the tests fix:
  lq.oi_change_1h(m, ts) -> array      log OI change over 1 h; a row with create_time T is known at T + 5 min; > 30 min old -> NaN
  lq.price_change_1h(c) -> array       log(close[i] / close[i - 4]) on 15m closes
  lq.events(t15, dp, doi, thr_oi, thr_p, kind) -> [bar index]   kind "PO" (both) | "P" (price drop, doi > -1%); 1 per 24 h
  lq.trade(t15, c15, i, hold_bars, fm, delisted) -> {ret_pct, entry_close_ms, exit_close_ms, gross_mult, marks} | None
  lq.random_bars(t15, events_idx, n, seed) -> bars of the same months, same count, reproducible
  experiment_lq.candidates() (8), admit(res)
"""
import math
import unittest
from datetime import datetime, timezone

import numpy as np

BAR = 15 * 60 * 1000
MIN5 = 5 * 60 * 1000


def ms(s):
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)


T0 = ms("2024-03-01 00:00")


def zero_fm():
    import fill_model as FM
    return FM.FillModel(taker_fee=0.0, maker_fee=0.0, half_spread=0.0, impact_k=0.0)


class TestSignals(unittest.TestCase):
    def test_oi_change_availability(self):
        import lq
        n = 60
        oi = 100.0 * np.exp(-np.arange(n) * 0.01)
        m = {"t": T0 + np.arange(n, dtype=np.int64) * MIN5, "oi": oi, "oi_val": oi, "top_pos": np.ones(n),
             "ls_all": np.ones(n), "taker": np.ones(n), "top_acc": np.ones(n)}
        t = T0 + 30 * MIN5
        v = lq.oi_change_1h(m, np.array([t]))[0]
        self.assertAlmostEqual(v, math.log(oi[29] / oi[17]), places=9)      # last known row 29, an hour back 17
        m2 = {k: (a.copy() if k != "t" else a) for k, a in m.items()}
        m2["oi"][30:] = 1.0
        self.assertAlmostEqual(lq.oi_change_1h(m2, np.array([t]))[0], v, places=12)
        self.assertTrue(np.isnan(lq.oi_change_1h(m, np.array([t + 4 * 3600 * 1000]))[0]))   # store ended: stale

    def test_price_change_and_events(self):
        import lq
        c = np.full(400, 100.0)
        c[100:] = 95.0                                                   # -5.1% drop at bar 100
        c[200:] = 90.0                                                   # another drop 25 h later
        t = T0 + np.arange(400, dtype=np.int64) * BAR
        dp = lq.price_change_1h(c)
        self.assertAlmostEqual(dp[100], math.log(95 / 100))
        self.assertTrue(np.isnan(dp[2]))
        doi = np.zeros(400)
        doi[100:104] = -0.08
        doi[200:204] = -0.08
        ev = lq.events(t, dp, doi, -0.05, -0.03, "PO")
        self.assertEqual(ev, [100, 200])                                # one per 24 h, both thresholds
        self.assertEqual(lq.events(t, dp, doi, -0.10, -0.03, "PO"), [])
        doi2 = np.zeros(400)
        self.assertEqual(lq.events(t, dp, doi2, -0.05, -0.03, "P"), [100, 200])
        self.assertEqual(lq.events(t, dp, doi, -0.05, -0.03, "P"), [])  # OI dropped: not a price-only event


class TestTradeAndControl(unittest.TestCase):
    def test_trade(self):
        import lq
        c = np.linspace(100, 110, 200)
        t = T0 + np.arange(200, dtype=np.int64) * BAR
        x = lq.trade(t, c, 10, 16, zero_fm(), delisted=False)
        self.assertAlmostEqual(x["ret_pct"], (c[27] / c[11] - 1) * 100, places=9)     # entry next close, exit +16 bars
        self.assertEqual(x["entry_close_ms"], int(t[11]) + BAR)
        self.assertEqual(x["exit_close_ms"], int(t[27]) + BAR)
        self.assertIsNone(lq.trade(t, c, 190, 16, zero_fm(), delisted=False))       # window does not fit
        d = lq.trade(t, c, 190, 16, zero_fm(), delisted=True)
        self.assertAlmostEqual(d["ret_pct"], -50.0)

    def test_random_bars(self):
        import lq
        t = T0 + np.arange(96 * 60, dtype=np.int64) * BAR                # March + April 2024
        ev = [10, 20, 96 * 40]
        a = lq.random_bars(t, ev, 3, seed=1)
        self.assertEqual(a, lq.random_bars(t, ev, 3, seed=1))
        self.assertEqual(len(a), 3)
        month = lambda i: datetime.fromtimestamp(int(t[i]) / 1000, timezone.utc).strftime("%Y-%m")
        self.assertEqual(sorted(month(i) for i in a), sorted(month(i) for i in ev))


class TestTradeMean(unittest.TestCase):
    def test_trade_mean_is_per_trade_with_day_blocks(self):
        """Spec §5.1: the MEAN TRADE, bootstrapped by blocks of days -- not the mean of daily means. The first train run
        reported the mean of daily means; on clustered crash days the two differ in sign."""
        import lq
        tr = [{"ret_pct": 10.0, "entry_close_ms": T0}, {"ret_pct": 10.0, "entry_close_ms": T0 + BAR},
              {"ret_pct": 10.0, "entry_close_ms": T0 + 2 * BAR}, {"ret_pct": -2.0, "entry_close_ms": T0 + 86_400_000}]
        a = lq.trade_mean_ci(tr, alpha=0.05)
        self.assertAlmostEqual(a["mean_pct"], 7.0)                       # (30 - 2) / 4, not (10 - 2) / 2
        self.assertEqual(a["n_trades"], 4)
        self.assertEqual(a["n_days"], 2)
        rnd = np.random.default_rng(3)
        many = [{"ret_pct": float(v), "entry_close_ms": T0 + int(i // 3) * 86_400_000}
                for i, v in enumerate(rnd.normal(0.5, 2.0, 600))]
        w = lq.trade_mean_ci(many, alpha=0.05)
        n = lq.trade_mean_ci(many, alpha=0.05 / 8)
        self.assertLess(w["lo95"], w["mean_pct"])
        self.assertGreater(w["hi95"], w["mean_pct"])
        self.assertLessEqual(n["lo95"], w["lo95"])                        # smaller alpha -> wider interval
        self.assertEqual(lq.trade_mean_ci(many, alpha=0.05), w)           # reproducible
        self.assertEqual(lq.trade_mean_ci([], alpha=0.05)["n_trades"], 0)


class TestAdmission(unittest.TestCase):
    def test_admit(self):
        import experiment_lq as X
        self.assertEqual(len(X.candidates()), 8)
        good = {"agg": {"lo95": 0.01, "mean_pct": 0.5}, "rand_p95": 0.1, "vs_p": {"lo95": 0.01}, "n_events": 100,
                "by_year": {"2023": 0.1, "2024": -0.1, "2025": 0.2}}
        self.assertTrue(X.admit(good)[0])
        for k, v in (("agg", {"lo95": -0.01, "mean_pct": 0.5}), ("rand_p95", 0.6), ("vs_p", {"lo95": -0.01}),
                     ("n_events", 99), ("by_year", {"2023": -0.1, "2024": -0.1, "2025": 0.2})):
            self.assertFalse(X.admit(dict(good, **{k: v}))[0], k)


if __name__ == "__main__":
    unittest.main(verbosity=2)
