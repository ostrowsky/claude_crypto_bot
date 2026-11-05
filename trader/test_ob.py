"""OB -- futures order-book depth imbalance, step 1 (orderbook-depth-ob-1105-spec.md §6). Written BEFORE
bookdepth.py / experiment_ob.py. Synthetic CSV and arrays, offline.

Interface the tests fix:
  bookdepth.LEVELS = (0.2, 1, 2, 3, 4, 5)
  bookdepth.parse_day(csv_text) -> {"t": int64 ms (n,), "bid": (n, 6), "ask": (n, 6)}   USDT notional per level
  bookdepth.hourly(parsed, day) -> {"t": hour-close ms (m,), "bid": (m, 6), "ask": (m, 6), "imb1_mean": (m,)}
       hour H in (day 01:00 .. next day 00:00]; last snapshot in [H - 5 min, H); hours without one are skipped
  bookdepth.imb(bid, ask) -> (bid - ask) / (bid + ask)
  bookdepth.fetch_day(perp, day, get) -> hourly dict | None (404)
  bookdepth.fetch(jobs, get, store_dir) -> totals; a day already done is not fetched again
  experiment_ob.features(h) -> {name: (m,)} for the 6 features of §3
  experiment_ob.ic(f, y, min_n) / partial_ic(f, past, y, min_n) / quintile_spread(f, y)
  experiment_ob.moment_hours(t_ms, step_h) -> mask of hours on the 4 h / 24 h grid
  experiment_ob.admit(res)
"""
import json
import math
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

H = 3600 * 1000


def ms(s):
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)


def csv_day(day="2024-03-01", snaps=None):
    """snaps: [(time 'HH:MM:SS', bid_mult, ask_mult)]; level k has bid = k * 100 * bid_mult, ask = k * 100 * ask_mult."""
    rows = ["timestamp,percentage,depth,notional"]
    for hhmmss, bm, am in snaps:
        for lv in (5.0, 4.0, 3.0, 2.0, 1.0, 0.2):
            rows.append(f"{day} {hhmmss},-{lv:.2f},1.0,{lv * 100 * bm}")
        for lv in (0.2, 1.0, 2.0, 3.0, 4.0, 5.0):
            rows.append(f"{day} {hhmmss},{lv:.2f},1.0,{lv * 100 * am}")
    return "\n".join(rows) + "\n"


class TestParse(unittest.TestCase):
    def test_parse_day(self):
        import bookdepth as B
        p = B.parse_day(csv_day(snaps=[("00:30:00", 2.0, 1.0), ("00:00:01", 1.0, 1.0)]))
        self.assertEqual(B.LEVELS, (0.2, 1, 2, 3, 4, 5))
        self.assertEqual(p["t"].tolist(), [ms("2024-03-01 00:00:01"), ms("2024-03-01 00:30:00")])   # sorted
        self.assertAlmostEqual(p["bid"][1, 1], 200.0)            # level 1%, bid side, snapshot 00:30
        self.assertAlmostEqual(p["ask"][1, 0], 20.0)             # level 0.2%, ask side
        self.assertAlmostEqual(B.imb(3.0, 1.0), 0.5)

    def test_hourly(self):
        import bookdepth as B
        snaps = [("00:10:00", 1.0, 1.0), ("00:56:00", 3.0, 1.0), ("00:58:30", 1.0, 3.0),     # hour 01:00
                 ("01:30:00", 1.0, 1.0),                                                    # 02:00: none in last 5 min
                 ("23:57:00", 2.0, 2.0)]                                                     # hour 24:00 = next day
        h = B.hourly(B.parse_day(csv_day(snaps=snaps)), "2024-03-01")
        self.assertEqual(h["t"].tolist(), [ms("2024-03-01 01:00:00"), ms("2024-03-02 00:00:00")])
        self.assertAlmostEqual(B.imb(h["bid"][0, 1], h["ask"][0, 1]), -0.5)                 # last snapshot 00:58:30
        self.assertAlmostEqual(h["imb1_mean"][0], np.mean([0.0, 0.5, -0.5]))               # all snapshots of the hour


class TestFetch(unittest.TestCase):
    def test_fetch_day_and_resume(self):
        import io
        import zipfile
        import bookdepth as B
        calls = []

        def zipped(day):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr(f"XUSDT-bookDepth-{day}.csv", csv_day(day, [("00:57:00", 1.0, 1.0)]))
            return buf.getvalue()

        def get(url):
            calls.append(url)
            self.assertTrue(url.startswith("https://data.binance.vision/data/futures/um/daily/bookDepth/XUSDT/"))
            if "2024-03-02" in url:
                raise B.NotFound(url)
            return zipped(url.split("bookDepth-")[1][:10])

        self.assertEqual(len(B.fetch_day("XUSDT", "2024-03-01", get)["t"]), 1)
        self.assertIsNone(B.fetch_day("XUSDT", "2024-03-02", get))
        with tempfile.TemporaryDirectory() as d:
            calls.clear()
            tot = B.fetch({"XUSDT": ["2024-03-01", "2024-03-02"]}, get, Path(d), workers=1)
            self.assertEqual((tot["files"], tot["missing"]), (1, 1))
            calls.clear()
            B.fetch({"XUSDT": ["2024-03-01", "2024-03-02", "2024-03-03"]}, get, Path(d), workers=1)
            self.assertEqual([c.split("bookDepth-")[1][:10] for c in calls], ["2024-03-03"])
            a = B.load("XUSDT", Path(d))
            self.assertEqual(len(a["t"]), 2)


class TestShard(unittest.TestCase):
    def test_shards_partition_the_pairs(self):
        import bookdepth as B
        perps = [f"P{k}USDT" for k in range(30)]
        parts = [B.shard(perps, i, 4) for i in range(4)]
        self.assertEqual(sorted(sum(parts, [])), sorted(perps))
        self.assertEqual(sum(len(x) for x in parts), 30)
        self.assertEqual(B.shard(perps, 0, 1), perps)

class TestStats(unittest.TestCase):
    def test_features(self):
        import experiment_ob as X
        n = 8
        bid = np.tile(np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), (n, 1))
        ask = np.ones((n, 6))
        ask[4:, 1] = 3.0                                            # imb_1 drops to 0 from hour 4 on
        h = {"t": np.arange(n, dtype=np.int64) * H + H, "bid": bid, "ask": ask, "imb1_mean": np.full(n, 0.1)}
        f = X.features(h)
        self.assertEqual(sorted(f), sorted(X.FEATURES))
        self.assertAlmostEqual(f["imb_02"][0], 1 / 3)
        self.assertAlmostEqual(f["imb_1"][0], 0.5)
        self.assertAlmostEqual(f["imb_5"][0], 6 / 8)
        self.assertTrue(np.isnan(f["d_imb_1_4h"][3]))               # no hour 4 h back
        self.assertAlmostEqual(f["d_imb_1_4h"][4], -0.5)
        self.assertAlmostEqual(f["imb_1_mean1h"][2], 0.1)

    def test_ic_partial_spread_moments(self):
        import experiment_ob as X
        rnd = np.random.default_rng(2)
        f = rnd.normal(size=30)
        y = f + rnd.normal(scale=0.1, size=30)
        self.assertGreater(X.ic(f, y, 10), 0.9)
        self.assertIsNone(X.ic(f[:5], y[:5], 10))
        past = f.copy()                                             # the feature is all past return
        self.assertLess(abs(X.partial_ic(f, past, y, 10) or 0.0), 0.2)
        self.assertGreater(X.quintile_spread(f, y), 1.0)
        t = np.array([ms("2024-03-01 00:00:00"), ms("2024-03-01 01:00:00"), ms("2024-03-01 04:00:00"),
                      ms("2024-03-02 00:00:00")])
        self.assertEqual(X.moment_hours(t, 4).tolist(), [True, False, True, True])
        self.assertEqual(X.moment_hours(t, 24).tolist(), [True, False, False, True])

    def test_admit(self):
        import experiment_ob as X
        good = {"ic": {"lo95": 0.01}, "partial": {"lo95": 0.005}, "halves": [0.02, 0.01], "spread_pct": 0.40}
        self.assertTrue(X.admit(good)[0])
        for k, v in (("ic", {"lo95": -0.01}), ("partial", {"lo95": -0.001}), ("halves", [0.02, -0.01]),
                     ("spread_pct", 0.30)):
            self.assertFalse(X.admit(dict(good, **{k: v}))[0], k)


if __name__ == "__main__":
    unittest.main(verbosity=2)
