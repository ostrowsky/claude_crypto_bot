"""LA sign check (fourth-review-1105-spec.md §2). Written BEFORE la_sign_check.py. Offline.

Interface the tests fix:
  la_sign_check.gross_pair(spot, perp, t_pub, hold_bars) -> {"long_pct", "short_pct", "day"} | None
       entry on each market: close of the first 15m bar opening >= t_pub + 1 min; exit hold_bars later; no costs
  la_sign_check.split_diff(rows, alpha, boot, seed) -> {"live", "rest", "diff", "lo", "hi", "n_live", "n_rest"}
       rows: {"group": "live" | "rest", "long_pct", "day"}; diff = rest - live; days are resampled
  la_sign_check.claim_holds(d) -> True only if hi < 0
"""
import unittest

import numpy as np

BAR = 15 * 60 * 1000
T0 = 1_700_000_000_000 - (1_700_000_000_000 % BAR)


def bars(closes):
    return {"t": np.array([T0 + i * BAR for i in range(len(closes))], dtype=np.int64), "c": np.array(closes, dtype=float)}


class TestPair(unittest.TestCase):
    def test_mirror_on_the_same_prices(self):
        import la_sign_check as L
        b = bars([100.0] * 3 + [90.0] * 10)
        r = L.gross_pair(b, b, T0 + BAR + 30_000, 4)
        self.assertAlmostEqual(r["long_pct"], -10.0)
        self.assertAlmostEqual(r["short_pct"], 10.0)
        self.assertAlmostEqual(r["long_pct"] + r["short_pct"], 0.0)

    def test_each_market_uses_its_own_prices(self):
        import la_sign_check as L
        r = L.gross_pair(bars([100.0] * 3 + [110.0] * 10), bars([200.0] * 3 + [180.0] * 10), T0 + BAR + 30_000, 4)
        self.assertAlmostEqual(r["long_pct"], 10.0)
        self.assertAlmostEqual(r["short_pct"], 10.0)

    def test_missing_exit_bar(self):
        import la_sign_check as L
        self.assertIsNone(L.gross_pair(bars([100.0] * 4), bars([100.0] * 20), T0 + BAR + 30_000, 4))


class TestSplit(unittest.TestCase):
    def rows(self, live, rest):
        r = [{"group": "live", "long_pct": v, "day": i // 2} for i, v in enumerate(live)]
        r += [{"group": "rest", "long_pct": v, "day": 100 + i // 2} for i, v in enumerate(rest)]
        return r

    def test_clear_difference_holds(self):
        import la_sign_check as L
        d = L.split_diff(self.rows([1.0, 1.1, 0.9] * 10, [-5.0, -5.1, -4.9] * 10), 0.05, 1000, 29)
        self.assertAlmostEqual(d["diff"], d["rest"] - d["live"])
        self.assertLess(d["hi"], 0)
        self.assertTrue(L.claim_holds(d))
        self.assertEqual((d["n_live"], d["n_rest"]), (30, 30))

    def test_no_difference_does_not_hold(self):
        import la_sign_check as L
        rnd = np.random.default_rng(0)
        a, b = list(rnd.normal(0, 3, 60)), list(rnd.normal(0, 3, 60))
        self.assertFalse(L.claim_holds(L.split_diff(self.rows(a, b), 0.05, 1000, 29)))

    def test_empty_group_does_not_hold(self):
        import la_sign_check as L
        d = L.split_diff(self.rows([1.0] * 10, []), 0.05, 200, 29)
        self.assertFalse(L.claim_holds(d))


class TestSameBar(unittest.TestCase):
    """Reproduces the defect of 2026-11-05: a market not yet trading at t_pub silently entered at a later bar
    (TRUMPUSDT 2025-01-19: spot at 53.67 against perp at 32.81)."""

    def test_market_without_the_entry_bar_gives_none(self):
        import la_sign_check as L
        perp = bars([100.0] * 40)
        late = {"t": perp["t"][20:], "c": perp["c"][20:]}          # spot starts 5 hours after the announcement
        self.assertIsNone(L.gross_pair(late, perp, T0 + BAR + 30_000, 4))
        self.assertIsNone(L.gross_pair(perp, late, T0 + BAR + 30_000, 4))

    def test_both_entries_are_the_same_bar(self):
        import la_sign_check as L
        a = bars([100.0] * 40)
        self.assertIsNotNone(L.gross_pair(a, a, T0 + BAR + 30_000, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
