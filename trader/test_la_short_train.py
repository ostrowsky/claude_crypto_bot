"""LA narrow short, train step (la-short-feasibility-1105-spec.md, entry of 2026-11-05). Written BEFORE la_short_train.py.

Interface the tests fix:
  la_short_train.short_trade(bars, t_pub, hold_bars, funding) -> {"ret_pct", "funding_pct", "liquidated",
       "entry_close_ms"} | None
       bars: {"t" (bar open ms), "h", "c"}; funding: {"t", "rate"} | None
       entry: close of the first bar opening >= t_pub + 1 min; exit: close hold_bars later
       ret = (entry - exit) / entry - 2 * (0.05% + 8 bp) + sum of funding rates in (entry close, exit close];
       a high >= entry * (1 + 1 - 0.01) inside the hold -> liquidated, ret = -100%
  la_short_train.select(rows) -> rows with status perp_live and etype OKX_PERP_NEW / BYBIT_PERP_NEW
  la_short_train.day_cluster_ci(trades, alpha, boot, seed) -> {"n", "mean", "lo", "hi"} (days resampled)
  la_short_train.power(mu, sd, n, alpha=0.05) -> one-sided power of a z test
"""
import math
import unittest

import numpy as np

BAR = 15 * 60 * 1000
T0 = 1_700_000_000_000 - (1_700_000_000_000 % BAR)


def bars(closes, highs=None):
    n = len(closes)
    return {"t": np.array([T0 + i * BAR for i in range(n)], dtype=np.int64), "c": np.array(closes, dtype=float),
            "h": np.array(highs if highs is not None else closes, dtype=float)}


class TestTrade(unittest.TestCase):
    def test_short_sign_and_costs(self):
        import la_short_train as L
        b = bars([100.0] * 3 + [90.0] * 10)
        r = L.short_trade(b, T0 + BAR + 30_000, 4, None)          # first bar opening >= t_pub+1min is bar 2
        self.assertAlmostEqual(r["ret_pct"], (100 - 90) / 100 * 100 - 2 * 0.13, places=6)
        self.assertFalse(r["liquidated"])
        self.assertEqual(r["entry_close_ms"], int(b["t"][2]) + BAR)

    def test_funding_inside_the_hold_only(self):
        import la_short_train as L
        b = bars([100.0] * 12)
        entry_close = int(b["t"][2]) + BAR
        f = {"t": np.array([entry_close - 1, entry_close + 1, entry_close + 4 * BAR, entry_close + 4 * BAR + 1]),
             "rate": np.array([0.01, 0.001, 0.002, 0.05])}
        r = L.short_trade(b, T0 + BAR + 30_000, 4, f)
        self.assertAlmostEqual(r["funding_pct"], (0.001 + 0.002) * 100)
        self.assertAlmostEqual(r["ret_pct"], -0.26 + 0.3, places=6)

    def test_liquidation_and_missing_bars(self):
        import la_short_train as L
        b = bars([100.0] * 10, highs=[100.0] * 4 + [199.5] + [100.0] * 5)
        r = L.short_trade(b, T0 + BAR + 30_000, 4, None)
        self.assertTrue(r["liquidated"])
        self.assertEqual(r["ret_pct"], -100.0)
        self.assertIsNone(L.short_trade(bars([100.0] * 4), T0 + BAR + 30_000, 4, None))   # exit bar not there


class TestSelectStats(unittest.TestCase):
    def test_select(self):
        import la_short_train as L
        rows = [{"etype": "OKX_PERP_NEW", "status": "perp_live"}, {"etype": "OKX_PERP_NEW", "status": "perp_later"},
                {"etype": "BN_PERP_NEW", "status": "perp_live"}, {"etype": "BYBIT_PERP_NEW", "status": "perp_live"}]
        self.assertEqual([r["etype"] for r in L.select(rows)], ["OKX_PERP_NEW", "BYBIT_PERP_NEW"])

    def test_day_cluster_ci_and_power(self):
        import la_short_train as L
        tr = [{"ret_pct": 1.0 + (i % 3), "entry_close_ms": T0 + (i // 2) * 86_400_000} for i in range(40)]
        ci = L.day_cluster_ci(tr, 0.05, 2000, 29)
        self.assertEqual(ci["n"], 40)
        self.assertAlmostEqual(ci["mean"], np.mean([t["ret_pct"] for t in tr]))
        self.assertLess(ci["lo"], ci["mean"])
        self.assertGreater(ci["hi"], ci["mean"])
        self.assertAlmostEqual(L.power(1.0, 1.0, 9), 0.5 * (1 + math.erf((3 - 1.6448536) / math.sqrt(2))), places=4)
        self.assertLess(L.power(0.2, 1.0, 30), L.power(0.2, 1.0, 300))


class TestEntryBar(unittest.TestCase):
    def test_gap_at_the_entry_bar_gives_none(self):
        import la_short_train as L
        b = bars([100.0] * 40)
        gap = {k: np.delete(v, [2, 3, 4]) for k, v in b.items()}     # the first bar after t_pub is missing
        self.assertIsNone(L.short_trade(gap, T0 + BAR + 30_000, 4, None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
