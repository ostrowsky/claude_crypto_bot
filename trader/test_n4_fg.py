"""N4 G1 and G2 (n4-1106-spec.md §4). Written BEFORE n4_fg.py. Synthetic, offline.

Interface the tests fix:
  n4_fg.forecast(prem, t_prev, t_next, decision_ms, i8=1e-4) -> (fhat, n_bars) or (None, 0); prem = {"t", "c"}
  n4_fg.g1_trade(bars, t_fund, rate, fhat, cost_side) -> {"net_bp", "price_bp", "funding_bp", "cost_bp", "side", "entry_ms"} or None
  n4_fg.placebo_times(t_fund, all_funding) -> [t']   no accrual inside t' +- 15 min, none for a 1 h interval
  n4_fg.spot_of(perp, spot_syms) -> (spot, mult) or None
  n4_fg.g2_trades(prem, perp, spot, thr_of_bar, funding, cost_of_bar, excluded_until=0, entry_lag=0, last_ms=None) -> (trades, dropped)
"""
import unittest

import numpy as np

BAR = 900_000
H = 3_600_000


def prem_of(t0, values):
    return {"t": t0 + np.arange(len(values), dtype=np.int64) * BAR, "c": np.array(values, dtype=float)}


class TestForecast(unittest.TestCase):
    def test_hand_example_8h_inside_the_clamp(self):
        import n4_fg as N
        # 4 closed bars with premium 1,1,3,5 bp, weights 1,2,3,4: (1+2+9+20)/10 = 3.2 bp; clamp(1 - 3.2) = -2.2 bp -> 1 bp
        p = prem_of(0, [1e-4, 1e-4, 3e-4, 5e-4])
        f, n = N.forecast(p, 0, 8 * H, 4 * BAR)
        self.assertEqual(n, 4)
        self.assertAlmostEqual(f, 1e-4, places=12)

    def test_hand_example_clamp_binds(self):
        import n4_fg as N
        # P = 30 bp: clamp(1 - 30 = -29 bp) to -5 bp -> 25 bp
        f, _ = N.forecast(prem_of(0, [30e-4] * 4), 0, 8 * H, 4 * BAR)
        self.assertAlmostEqual(f, 25e-4, places=12)
        # P = -30 bp: clamp(+31 bp) to +5 bp -> -25 bp
        f, _ = N.forecast(prem_of(0, [-30e-4] * 4), 0, 8 * H, 4 * BAR)
        self.assertAlmostEqual(f, -25e-4, places=12)

    def test_shorter_interval_divides_everything(self):
        import n4_fg as N
        f, _ = N.forecast(prem_of(0, [30e-4] * 4), 0, 4 * H, 4 * BAR)
        self.assertAlmostEqual(f, 12.5e-4, places=12)                  # 25 bp * 4 / 8

    def test_one_hour_weights_are_equal(self):
        import n4_fg as N
        f, n = N.forecast(prem_of(0, [10e-4, 20e-4, 60e-4]), 0, 1 * H, 3 * BAR)
        self.assertEqual(n, 3)
        self.assertAlmostEqual(f, (30e-4 - 5e-4) / 8, places=12)       # P = 30 bp, kept equal-weighted

    def test_only_bars_closed_by_the_decision(self):
        import n4_fg as N
        p = prem_of(0, [10e-4, 10e-4, 500e-4])                         # the last bar closes after the decision
        f, n = N.forecast(p, 0, 8 * H, 2 * BAR)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(f, 10e-4 - 5e-4, places=12)

    def test_only_bars_of_this_interval(self):
        import n4_fg as N
        p = prem_of(0, [500e-4, 10e-4, 10e-4])
        f, n = N.forecast(p, BAR, BAR + 8 * H, 3 * BAR)
        self.assertEqual(n, 2)

    def test_no_bars(self):
        import n4_fg as N
        self.assertEqual(N.forecast(prem_of(100 * BAR, [1e-4]), 0, 8 * H, BAR), (None, 0))


def bars(t0, closes):
    return {"t": t0 + np.arange(len(closes), dtype=np.int64) * BAR, "c": np.array(closes, dtype=float)}


class TestG1Trade(unittest.TestCase):
    T = 8 * H

    def _b(self, entry, exit_):
        # bars opened at T-45, T-30, T-15, T, T+15
        return bars(self.T - 3 * BAR, [100.0, entry, 999.0, exit_, 999.0])

    def test_bars_used(self):
        import n4_fg as N
        r = N.g1_trade(self._b(100.0, 101.0), self.T, 0.0, 1e-3, 0.0)
        self.assertEqual(r["entry_ms"], self.T - BAR)                  # the close of the bar opened at T-30
        self.assertAlmostEqual(r["price_bp"], -100.0)                  # short, price up 1%, exit at the close of the bar opened at T

    def test_short_receives_a_positive_rate(self):
        import n4_fg as N
        r = N.g1_trade(self._b(100.0, 100.0), self.T, 2e-3, 1e-3, 0.0)
        self.assertEqual(r["side"], -1)
        self.assertAlmostEqual(r["funding_bp"], 20.0)

    def test_long_receives_a_negative_rate(self):
        import n4_fg as N
        r = N.g1_trade(self._b(100.0, 100.0), self.T, -2e-3, -1e-3, 0.0)
        self.assertEqual(r["side"], 1)
        self.assertAlmostEqual(r["funding_bp"], 20.0)

    def test_wrong_forecast_pays(self):
        import n4_fg as N
        r = N.g1_trade(self._b(100.0, 100.0), self.T, -2e-3, 1e-3, 0.0)
        self.assertEqual(r["side"], -1)
        self.assertAlmostEqual(r["funding_bp"], -20.0)

    def test_money_adds_up(self):
        import n4_fg as N
        r = N.g1_trade(self._b(100.0, 99.0), self.T, 2e-3, 1e-3, 5e-4)
        self.assertAlmostEqual(r["cost_bp"], 10.0)
        self.assertAlmostEqual(r["net_bp"], r["price_bp"] + r["funding_bp"] - r["cost_bp"])
        self.assertAlmostEqual(r["net_bp"], 100.0 + 20.0 - 10.0)

    def test_missing_exact_bar_is_dropped(self):
        import n4_fg as N
        b = self._b(100.0, 101.0)
        keep = b["t"] != self.T
        self.assertIsNone(N.g1_trade({"t": b["t"][keep], "c": b["c"][keep]}, self.T, 1e-3, 1e-3, 0.0))
        keep = b["t"] != self.T - 2 * BAR
        self.assertIsNone(N.g1_trade({"t": b["t"][keep], "c": b["c"][keep]}, self.T, 1e-3, 1e-3, 0.0))

    def test_jitter_of_the_event_is_floored(self):
        import n4_fg as N
        r = N.g1_trade(self._b(100.0, 101.0), self.T + 7, 0.0, 1e-3, 0.0)
        self.assertAlmostEqual(r["price_bp"], -100.0)


class TestPlacebo(unittest.TestCase):
    def test_both_sides_on_8h(self):
        import n4_fg as N
        ft = np.array([0, 8, 16, 24], dtype=np.int64) * H
        self.assertEqual(N.placebo_times(8 * H, ft), [6 * H, 10 * H])

    def test_window_never_contains_an_accrual(self):
        import n4_fg as N
        ft = np.array([0, 2, 4, 6, 8], dtype=np.int64) * H             # a 2 h interval: T +- 2 h is an accrual
        self.assertEqual(N.placebo_times(4 * H, ft), [])

    def test_none_for_a_1h_interval(self):
        import n4_fg as N
        ft = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64) * H
        self.assertEqual(N.placebo_times(3 * H, ft), [])

    def test_accrual_near_the_edge_blocks(self):
        import n4_fg as N
        ft = np.array([0, 8 * H - 2 * H + 10 * 60_000, 8 * H], dtype=np.int64)
        self.assertEqual(N.placebo_times(8 * H, ft), [10 * H])


class TestMapping(unittest.TestCase):
    def test_multiplier(self):
        import n4_fg as N
        self.assertEqual(N.spot_of("1000PEPEUSDT", {"PEPEUSDT"}), ("PEPEUSDT", 1000))
        self.assertEqual(N.spot_of("BTCUSDT", {"BTCUSDT"}), ("BTCUSDT", 1))
        self.assertIsNone(N.spot_of("XYZUSDT", {"BTCUSDT"}))

    def test_usdc_version_finds_the_usdt_spot(self):
        import n4_fg as N
        self.assertEqual(N.spot_of("1000PEPEUSDC", {"PEPEUSDT"}), ("PEPEUSDT", 1000))
        self.assertEqual(N.spot_of("BTCUSDC", {"BTCUSDT"}), ("BTCUSDT", 1))


def g2_set(prem_vals, perp_vals=None, spot_vals=None):
    n = len(prem_vals)
    t = np.arange(n, dtype=np.int64) * BAR
    perp = {"t": t, "c": np.array(perp_vals if perp_vals is not None else [100.0] * n, dtype=float)}
    spot = {"t": t, "c": np.array(spot_vals if spot_vals is not None else [100.0] * n, dtype=float)}
    return {"t": t, "c": np.array(prem_vals, dtype=float)}, perp, spot


THR = 0.004


def thr(_):
    return THR


def zero(_):
    return 0.0


class TestG2(unittest.TestCase):
    def test_exit_at_the_first_bar_below_a_third(self):
        import n4_fg as N
        p, a, b = g2_set([0, 0.005, 0.003, 0.001, 0.0, 0, 0], [100, 100, 100, 99, 99, 99, 99], [100, 100, 100, 100, 100, 100, 100])
        tr, _ = N.g2_trades(p, a, b, thr, None, zero)
        self.assertEqual(len(tr), 1)
        self.assertEqual(tr[0]["reason"], "converged")
        self.assertEqual(tr[0]["bars"], 2)
        self.assertAlmostEqual(tr[0]["price_bp"], 100.0)               # short perp -1%, spot flat

    def test_equal_to_the_threshold_is_not_a_signal(self):
        import n4_fg as N
        p, a, b = g2_set([0, THR, 0, 0, 0])
        self.assertEqual(N.g2_trades(p, a, b, thr, None, zero)[0], [])

    def test_time_exit_after_32_bars(self):
        import n4_fg as N
        p, a, b = g2_set([0.005] * 40)
        tr, _ = N.g2_trades(p, a, b, thr, None, zero)
        self.assertEqual(tr[0]["reason"], "time")
        self.assertEqual(tr[0]["bars"], 32)

    def test_one_position_per_symbol(self):
        import n4_fg as N
        p, a, b = g2_set([0, 0.005, 0.005, 0.005, 0.001, 0, 0.005, 0.001, 0])
        tr, _ = N.g2_trades(p, a, b, thr, None, zero)
        self.assertEqual([x["entry_ms"] for x in tr], [2 * BAR, 7 * BAR])

    def test_missing_entry_bar_is_dropped_and_counted(self):
        import n4_fg as N
        p, a, b = g2_set([0, 0.005, 0.001, 0])
        k = b["t"] != BAR
        tr, dropped = N.g2_trades(p, a, {"t": b["t"][k], "c": b["c"][k]}, thr, None, zero)
        self.assertEqual(tr, [])
        self.assertEqual(dropped["no_entry_bar"], 1)

    def test_missing_exit_bar_is_dropped_and_counted(self):
        import n4_fg as N
        p, a, b = g2_set([0, 0.005, 0.001, 0, 0])
        k = b["t"] != 2 * BAR
        tr, dropped = N.g2_trades(p, a, {"t": b["t"][k], "c": b["c"][k]}, thr, None, zero)
        self.assertEqual(tr, [])
        self.assertEqual(dropped["no_exit_bar"], 1)

    def test_listing_filter(self):
        import n4_fg as N
        p, a, b = g2_set([0, 0.005, 0.001, 0, 0.005, 0.001, 0])
        tr, dropped = N.g2_trades(p, a, b, thr, None, zero, excluded_until=3 * BAR)
        self.assertEqual([x["entry_ms"] for x in tr], [5 * BAR])
        self.assertEqual(dropped["listing"], 1)

    def test_funding_window_and_sign(self):
        import n4_fg as N
        p, a, b = g2_set([0, 0.005, 0.003, 0.003, 0.001, 0, 0])
        # the entry bar opens at BAR and closes at 2 BAR; the exit bar opens at 4 BAR and closes at 5 BAR
        f = {"t": np.array([2 * BAR, 3 * BAR, 5 * BAR, 6 * BAR], dtype=np.int64), "rate": np.array([9.0, 1e-3, 2e-3, 9.0])}
        tr, _ = N.g2_trades(p, a, b, thr, f, zero)
        self.assertAlmostEqual(tr[0]["funding_bp"], 30.0)              # (close of entry, close of exit]

    def test_money_adds_up(self):
        import n4_fg as N
        p, a, b = g2_set([0, 0.005, 0.001, 0], [100, 100, 99, 99], [100, 100, 101, 101])
        f = {"t": np.array([3 * BAR], dtype=np.int64), "rate": np.array([1e-3])}
        tr, _ = N.g2_trades(p, a, b, thr, f, lambda _: 0.003)
        x = tr[0]
        self.assertAlmostEqual(x["price_bp"], 100.0 + 100.0)
        self.assertAlmostEqual(x["cost_bp"], 30.0)
        self.assertAlmostEqual(x["net_bp"], x["price_bp"] + x["funding_bp"] - x["cost_bp"])

    def test_delisted_closes_at_the_last_price(self):
        import n4_fg as N
        p, a, b = g2_set([0, 0.005, 0.005, 0.005, 0.005] + [0.005] * 40, [100.0] * 45, [100.0] * 45)
        cut = {"t": a["t"][:4], "c": np.array([100.0, 100.0, 100.0, 90.0])}
        tr, _ = N.g2_trades(p, cut, b, thr, None, zero, last_ms=1000 * BAR)
        self.assertEqual(tr[0]["reason"], "delisted")
        self.assertAlmostEqual(tr[0]["price_bp"], 1000.0)              # no -50% penalty

    def test_data_ending_with_the_window_is_not_a_delisting(self):
        import n4_fg as N
        p, a, b = g2_set([0, 0.005, 0.005, 0.005, 0.005])
        tr, dropped = N.g2_trades(p, a, b, thr, None, zero, last_ms=4 * BAR)
        self.assertEqual(tr, [])
        self.assertEqual(dropped["no_exit_bar"], 1)

    def test_next_bar_sensitivity_uses_the_later_close(self):
        import n4_fg as N
        p, a, b = g2_set([0, 0.005, 0.005, 0.001, 0, 0], [100, 100, 90, 90, 90, 90])
        tr, _ = N.g2_trades(p, a, b, thr, None, zero, entry_lag=1)
        self.assertEqual(tr[0]["entry_ms"], 3 * BAR)
        self.assertAlmostEqual(tr[0]["price_bp"], 0.0)


class TestWindowCut(unittest.TestCase):
    def test_bars_outside_the_window_never_load(self):
        import tempfile
        from pathlib import Path
        import n4_fg as N
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.npz"
            np.savez(p, t=np.array([0, BAR, 2 * BAR, 3 * BAR], dtype=np.int64), c=np.arange(4.0))
            a = N._load_cut([p], BAR, 3 * BAR)
            self.assertEqual(a["t"].tolist(), [BAR, 2 * BAR])
            self.assertIsNone(N._load_cut([p], 9 * BAR, 10 * BAR))

    def test_two_files_merge_without_duplicates(self):
        import tempfile
        from pathlib import Path
        import n4_fg as N
        with tempfile.TemporaryDirectory() as d:
            p, q = Path(d) / "a.npz", Path(d) / "b.npz"
            np.savez(p, t=np.array([0, BAR], dtype=np.int64), c=np.array([1.0, 2.0]))
            np.savez(q, t=np.array([BAR, 2 * BAR], dtype=np.int64), c=np.array([2.0, 3.0]))
            self.assertEqual(N._load_cut([p, q], 0, 3 * BAR)["t"].tolist(), [0, BAR, 2 * BAR])


if __name__ == "__main__":
    unittest.main(verbosity=2)
