"""H4 -- beta of the X4 book on the bot's entries (bot-beta-h4-1105-spec.md §4). Written BEFORE bot_beta.py. Offline.

Interface the tests fix:
  bot_beta.factor_returns(grid, alt_syms, btc_sym) -> {day: (alt_ret_pct, btc_ret_pct)}  (close of the day's last
       15m bar vs the previous day's; NaN coins are left out of the mean)
  bot_beta.ols(y, X) -> coefficients [alpha, b1, b2, ...]
  bot_beta.boot_ols(y, X, block=5, boot=2000, seed=29) -> {"coef", "lo", "hi"}
  bot_beta.hedged(series, factors, b_alt, b_btc) -> [(day, R - b_alt*ALT - b_btc*BTC)] on common days
  bot_beta.exposure(trades, series, start_equity) -> mean share of equity held in positions per day
"""
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

import test_execution_replay as TER


class TestFactors(unittest.TestCase):
    def test_daily_close_to_close(self):
        import bot_beta as BB
        n = 96 * 3
        a = np.full(n, 100.0)
        a[96 * 2 - 1] = 110.0                       # day 2 closes +10%
        a[96 * 3 - 1] = 99.0                        # day 3 closes -10%
        b = np.full(n, 50.0)
        b[96 * 2 - 1] = np.nan                      # missing close: left out of day 2's mean
        c = np.full(n, 200.0)
        c[96 * 3 - 1] = 220.0
        g = TER.FakeGrid(np.column_stack([a, b, c]), syms=("AUSDT", "BUSDT", "BTCUSDT"))
        f = BB.factor_returns(g, ["AUSDT", "BUSDT"], "BTCUSDT")
        d2, d3 = (TER.T0 + timedelta(days=1)).strftime("%Y-%m-%d"), (TER.T0 + timedelta(days=2)).strftime("%Y-%m-%d")
        self.assertAlmostEqual(f[d2][0], 10.0)      # only AUSDT has both closes
        self.assertAlmostEqual(f[d2][1], 0.0)
        self.assertAlmostEqual(f[d3][1], 10.0)
        self.assertAlmostEqual(f[d3][0], -10.0)     # AUSDT -10%; BUSDT has no close on day 2


class TestRegression(unittest.TestCase):
    def synth(self, n=400, seed=5):
        rnd = np.random.default_rng(seed)
        X = rnd.normal(0, 2, size=(n, 2))
        y = -0.3 + 0.5 * X[:, 0] + 0.2 * X[:, 1] + rnd.normal(0, 0.5, n)
        return y, X

    def test_ols(self):
        import bot_beta as BB
        y, X = self.synth()
        c = BB.ols(y, X)
        self.assertAlmostEqual(c[0], -0.3, delta=0.08)
        self.assertAlmostEqual(c[1], 0.5, delta=0.03)
        self.assertAlmostEqual(c[2], 0.2, delta=0.03)

    def test_boot_ols_covers_truth_and_is_seeded(self):
        import bot_beta as BB
        y, X = self.synth()
        r = BB.boot_ols(y, X, boot=400)
        for k, true in enumerate((-0.3, 0.5, 0.2)):
            self.assertLess(r["lo"][k], true)
            self.assertGreater(r["hi"][k], true)
        self.assertEqual(BB.boot_ols(y, X, boot=400)["lo"], r["lo"])


class TestHedgeAndExposure(unittest.TestCase):
    def test_hedged(self):
        import bot_beta as BB
        s = [("2026-01-01", 1.0), ("2026-01-02", -2.0), ("2026-01-03", 0.5)]
        f = {"2026-01-01": (2.0, 1.0), "2026-01-02": (-4.0, 0.0)}
        h = BB.hedged(s, f, 0.5, 1.0)
        self.assertEqual([d for d, _ in h], ["2026-01-01", "2026-01-02"])
        self.assertAlmostEqual(h[0][1], 1.0 - 1.0 - 1.0)
        self.assertAlmostEqual(h[1][1], -2.0 + 2.0)

    def test_exposure(self):
        import bot_beta as BB
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        trades = [{"entry_ts": t0 + timedelta(hours=12), "exit_time": t0 + timedelta(hours=36), "cost": 1000.0}]
        series = [("2026-01-01", 0.0), ("2026-01-02", 0.0)]
        e = BB.exposure(trades, series, 10_000.0)
        self.assertAlmostEqual(e, 0.05)              # half a day at 10% on each of two days


class TestShiftDays(unittest.TestCase):
    """Control RC2 (spec entry of 2026-11-05): the same coin and time of day, a random day within +-7 days.
    (RC, a random bar of the same day, was withdrawn: it looked ahead.)"""

    def grid(self):
        return TER.FakeGrid(np.arange(1, 96 * 30 + 1, dtype=float))

    def test_shift_is_whole_days_within_a_week(self):
        import bot_beta as BB
        g = self.grid()
        c15 = TER.cand(96 * 15 + 40, float(96 * 15 + 41))
        c1h = TER.cand(96 * 16 + 51, float(96 * 16 + 52), tf="1h")
        out = BB.shift_days([c15, c1h], g, seed=3)
        self.assertEqual(len(out), 2)
        self.assertEqual([c["ts"] for c in out], sorted(c["ts"] for c in out))
        for orig in (c15, c1h):
            new = [c for c in out if c["tf"] == orig["tf"]][0]
            d = new["ie"] - orig["ie"]
            self.assertEqual(d % 96, 0)
            self.assertTrue(1 <= abs(d) // 96 <= 7)
            self.assertEqual(new["sym"], orig["sym"])
            self.assertEqual(new["price"], g.C[new["ie"], 0])
        self.assertEqual((out[[c["tf"] for c in out].index("1h")]["ie"] + 1) % 4, 0)

    def test_seeded_and_consistent_with_attach_grid(self):
        import bot_beta as BB
        import execution_replay as R
        g = self.grid()
        cs = [TER.cand(96 * 10 + 7, 1.0), TER.cand(96 * 20 + 70, 1.0)]
        a, b = BB.shift_days(cs, g, seed=11), BB.shift_days(cs, g, seed=11)
        self.assertEqual([c["ie"] for c in a], [c["ie"] for c in b])
        c = dict(a[0])
        R.attach_grid([c], g)
        self.assertEqual(c["ie"], a[0]["ie"])

    def test_edges_pick_another_shift(self):
        import bot_beta as BB
        g = self.grid()
        near_end = TER.cand(96 * 29 + 5, 1.0)                 # only negative shifts fit
        for sd in range(20):
            out = BB.shift_days([near_end], g, seed=sd)
            self.assertLess(out[0]["ie"], near_end["ie"])


class TestForwardOnly(unittest.TestCase):
    """Control RC3: shifts after the signal only (k in 1..7)."""

    def test_forward_only(self):
        import bot_beta as BB
        g = TER.FakeGrid(np.arange(1, 96 * 30 + 1, dtype=float))
        cs = [TER.cand(96 * d + 33, 1.0) for d in (3, 9, 15)]
        for sd in range(25):
            out = BB.shift_days(cs, g, seed=sd, forward_only=True)
            for orig, new in zip(cs, out):
                k = (new["ie"] - orig["ie"]) // 96
                self.assertTrue(1 <= k <= 7, k)


if __name__ == "__main__":
    unittest.main(verbosity=2)
