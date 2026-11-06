"""N1-A/B/C USDC perps (n1-1106-spec.md §4). Written BEFORE n1_usdc.py. Synthetic, offline.

Interface the tests fix:
  n1_usdc.parse_daily(text) -> {"t", "qv", "count"}                 1d kline CSV, ms or microseconds
  n1_usdc.parse_funding_full(text) -> {"t", "rate", "hours"}        hours = nan when the file has no interval column
  n1_usdc.daily_sums(t, rate, hours) -> {day: pct}                  only complete UTC days (spec §3)
  n1_usdc.pool(pairs) -> [(day, mean pct)]                          pairs = {coin: {"usdc": {day: pct}, "usdt": {...}, "from": day}}
  n1_usdc.months_upto(keys, last="2025-12") -> sorted months
  n1_usdc.usdt_of(sym) -> usdt symbol
  n1_usdc.episodes(t, x, thr, gap_min=5) -> [{"start", "minutes", "peak"}]
  n1_usdc.admit(days, first_day, launch_day) -> {"median", "n_days", "admitted"}
"""
import unittest

MIN = 60_000
DAY = 86_400_000


class TestParsers(unittest.TestCase):
    def test_daily_columns_and_microseconds(self):
        import n1_usdc as N
        txt = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
               "1700000000000,1,2,0.5,1.5,10,1700086399999,33.5,44,5,6,0\n"
               "1700086400000000,1,2,0.5,1.5,10,1700172799999,7.5,9,5,6,0\n")
        a = N.parse_daily(txt)
        self.assertEqual(a["t"].tolist(), [1700000000000, 1700086400000])
        self.assertEqual(a["qv"].tolist(), [33.5, 7.5])
        self.assertEqual(a["count"].tolist(), [44, 9])

    def test_funding_interval_duplicates_order(self):
        import n1_usdc as N
        a = N.parse_funding_full("calc_time,funding_interval_hours,last_funding_rate\n"
                                 "2000,8,0.0002\n1000,4,0.0001\n2000,8,0.0002\n")
        self.assertEqual(a["t"].tolist(), [1000, 2000])
        self.assertEqual(a["rate"].tolist(), [0.0001, 0.0002])
        self.assertEqual(a["hours"].tolist(), [4.0, 8.0])

    def test_funding_without_interval_column(self):
        import numpy as np
        import n1_usdc as N
        a = N.parse_funding_full("1000,0.0001\n")
        self.assertEqual(a["rate"].tolist(), [0.0001])
        self.assertTrue(np.isnan(a["hours"][0]))


class TestDailySums(unittest.TestCase):
    def test_complete_day_in_percent(self):
        import numpy as np
        import n1_usdc as N
        t = np.array([0, 8, 16]) * 3_600_000
        s = N.daily_sums(t, np.array([0.0001, 0.0002, 0.0001]), np.array([8.0, 8.0, 8.0]))
        self.assertAlmostEqual(s["1970-01-01"], 0.04)

    def test_incomplete_day_dropped_not_zero(self):
        import numpy as np
        import n1_usdc as N
        t = np.array([0, 8]) * 3_600_000
        s = N.daily_sums(t, np.array([0.0001, 0.0001]), np.array([8.0, 8.0]))
        self.assertEqual(s, {})

    def test_mixed_intervals_add_up(self):
        import numpy as np
        import n1_usdc as N
        t = np.array([0, 4, 8, 12, 16, 20]) * 3_600_000
        s = N.daily_sums(t, np.full(6, 0.0001), np.full(6, 4.0))
        self.assertAlmostEqual(s["1970-01-01"], 0.06)

    def test_no_interval_column_uses_event_count(self):
        import numpy as np
        import n1_usdc as N
        nan = float("nan")
        s = N.daily_sums(np.array([0, 8]) * 3_600_000, np.full(2, 0.0001), np.array([nan, nan]))
        self.assertAlmostEqual(s["1970-01-01"], 0.02)
        s = N.daily_sums(np.array([0]) * 3_600_000, np.full(1, 0.0001), np.array([nan]))
        self.assertEqual(s, {})


class TestPool(unittest.TestCase):
    def test_equal_weight_each_day_and_launch_cut(self):
        import n1_usdc as N
        pairs = {"A": {"usdc": {"2024-02-01": 0.01, "2024-02-02": 0.01}, "usdt": {"2024-02-01": 0.03, "2024-02-02": 0.03}, "from": "2024-01-01"},
                 "B": {"usdc": {"2024-02-01": 0.00, "2024-02-02": 0.01}, "usdt": {"2024-02-01": 0.00, "2024-02-02": 0.01}, "from": "2024-02-02"}}
        self.assertEqual(N.pool(pairs), [("2024-02-01", -0.02), ("2024-02-02", -0.01)])

    def test_day_missing_on_one_side_is_skipped(self):
        import n1_usdc as N
        pairs = {"A": {"usdc": {"2024-02-01": 0.01}, "usdt": {"2024-02-02": 0.03}, "from": "2024-01-01"}}
        self.assertEqual(N.pool(pairs), [])


class TestScope(unittest.TestCase):
    def test_2026_months_are_never_requested(self):
        import n1_usdc as N
        keys = ["x/A-fundingRate-2025-11.zip", "x/A-fundingRate-2026-01.zip", "x/A-fundingRate-2025-12.zip"]
        self.assertEqual(N.months_upto(keys), ["2025-11", "2025-12"])

    def test_usdt_of_keeps_the_multiplier(self):
        import n1_usdc as N
        self.assertEqual(N.usdt_of("1000PEPEUSDC"), "1000PEPEUSDT")
        self.assertEqual(N.usdt_of("BTCUSDC"), "BTCUSDT")


class TestEpisodes(unittest.TestCase):
    def _x(self, minutes, on):
        import numpy as np
        t = np.arange(minutes) * MIN
        x = np.ones(minutes)
        for i in on:
            x[i] = 1.002
        return t, x

    def test_gap_of_five_merges(self):
        import n1_usdc as N
        t, x = self._x(30, [1, 2, 8, 9])                     # 3..7 are five quiet minutes
        e = N.episodes(t, x, 0.001)
        self.assertEqual(len(e), 1)
        self.assertEqual(e[0]["minutes"], 9)

    def test_gap_of_six_splits(self):
        import n1_usdc as N
        t, x = self._x(30, [1, 2, 9, 10])                    # 3..8 are six quiet minutes
        self.assertEqual(len(N.episodes(t, x, 0.001)), 2)

    def test_missing_minutes_are_not_an_episode(self):
        import numpy as np
        import n1_usdc as N
        t = np.array([0, 100]) * MIN
        self.assertEqual(N.episodes(t, np.array([1.0, 1.0]), 0.001), [])

    def test_peak_is_the_largest_deviation(self):
        import numpy as np
        import n1_usdc as N
        t = np.arange(3) * MIN
        e = N.episodes(t, np.array([0.99, 1.0, 1.0]), 0.001)
        self.assertAlmostEqual(e[0]["peak"], 0.01)


class TestAdmit(unittest.TestCase):
    def _days(self, n, v):
        return {f"2025-{m:02d}-{d:02d}": v for m, d in [(10 + (i // 30), (i % 28) + 1) for i in range(n)]}

    def test_threshold(self):
        import n1_usdc as N
        self.assertTrue(N.admit(self._days(70, 25e6), "2025-10-03", "2024-01-01")["admitted"])
        self.assertFalse(N.admit(self._days(70, 15e6), "2025-10-03", "2024-01-01")["admitted"])

    def test_too_few_days(self):
        import n1_usdc as N
        self.assertFalse(N.admit(self._days(40, 25e6), "2025-10-03", "2024-01-01")["admitted"])

    def test_launched_inside_the_window_is_out(self):
        import n1_usdc as N
        self.assertFalse(N.admit(self._days(70, 25e6), "2025-10-03", "2025-11-01")["admitted"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
