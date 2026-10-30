"""BH19 -- frozen rules on the unseen 2018-2022 holdout (backward-holdout-bh19-1030-spec.md §4). Written BEFORE
archive_bh.py / experiment_bh19.py. Offline, no network.

Interface the tests fix:
  archive_bh.parse_klines(text) -> {t, o, h, l, c, v} (ms; header or not; microseconds -> ms; deduped, sorted)
  archive_bh.parse_funding(text) -> {t, rate, mark}
  archive_bh.parse_listing(xml) -> (prefixes or keys, next_marker | None)
  archive_bh.months_in_range(keys, first="2017-11", last="2022-12") -> ["YYYY-MM", ...]
  archive_bh.http_get(url) -> refuses any host but data.binance.vision and the S3 listing of it
  archive_bh.daily_panel(daily_by_sym, last_day="2022-12-31") -> xsmom.Panel without days after last_day
  experiment_bh19.train_mask(days, lo, hi) / overall(results) -> {"confirmed": [...], "not_confirmed": [...]}
"""
import unittest

import numpy as np

ROW = "{t},100.0,101.0,99.0,100.5,10.0,{t2},1005.0,7,5.0,502.0,0\n"


class TestParse(unittest.TestCase):
    def test_klines_formats(self):
        import archive_bh as A
        t0 = 1551398400000
        plain = "".join(ROW.format(t=t0 + i * 900000, t2=t0 + i * 900000 + 899999) for i in range(3))
        plain += ROW.format(t=t0, t2=t0 + 899999)                       # duplicate
        a = A.parse_klines(plain)
        self.assertEqual(len(a["t"]), 3)
        self.assertTrue(np.all(np.diff(a["t"]) == 900000))
        self.assertAlmostEqual(a["c"][0], 100.5)
        head = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
        self.assertEqual(len(A.parse_klines(head + plain)["t"]), 3)
        micro = "".join(ROW.format(t=(t0 + i * 900000) * 1000, t2=0) for i in range(2))
        self.assertEqual(int(A.parse_klines(micro)["t"][1]), t0 + 900000)

    def test_funding(self):
        import archive_bh as A
        f = A.parse_funding("calc_time,funding_interval_hours,last_funding_rate\n1583020800000,8,0.00049317\n"
                            "1583049600000,8,0.00027933\n")
        self.assertEqual(list(f["t"]), [1583020800000, 1583049600000])
        self.assertAlmostEqual(f["rate"][0], 0.00049317)

    def test_listing_and_months(self):
        import archive_bh as A
        xml = ("<ListBucketResult><IsTruncated>true</IsTruncated><NextMarker>data/x/B/</NextMarker>"
               "<CommonPrefixes><Prefix>data/x/A/</Prefix></CommonPrefixes>"
               "<CommonPrefixes><Prefix>data/x/B/</Prefix></CommonPrefixes></ListBucketResult>")
        self.assertEqual(A.parse_listing(xml), (["data/x/A/", "data/x/B/"], "data/x/B/"))
        xml2 = ("<ListBucketResult><IsTruncated>false</IsTruncated>"
                "<Contents><Key>p/AUSDT-15m-2017-10.zip</Key></Contents>"
                "<Contents><Key>p/AUSDT-15m-2017-11.zip</Key></Contents>"
                "<Contents><Key>p/AUSDT-15m-2017-11.zip.CHECKSUM</Key></Contents>"
                "<Contents><Key>p/AUSDT-15m-2023-01.zip</Key></Contents></ListBucketResult>")
        keys, nxt = A.parse_listing(xml2)
        self.assertIsNone(nxt)
        self.assertEqual(A.months_in_range(keys), ["2017-11"])

    def test_hosts(self):
        import archive_bh as A
        with self.assertRaises(ValueError):
            A.http_get("https://api.binance.com/api/v3/time")
        with self.assertRaises(ValueError):
            A.http_get("https://data.binance.vision/x", method="POST")


class TestPanelAndRules(unittest.TestCase):
    def test_panel_stops_at_2022(self):
        import archive_bh as A
        import xsmom as XS
        day = 86_400_000
        t0 = XS.day_num("2022-12-30") * day
        t = np.arange(4 * 96, dtype=np.int64) * 900000 + t0              # 2022-12-30 .. 2023-01-02
        c = np.linspace(1, 2, len(t))
        arr = {"t": t, "o": c, "h": c, "l": c, "c": c, "v": np.ones(len(t))}
        p = A.daily_panel({"XUSDT": XS.daily_from_15m(arr)}, last_day="2022-12-31")
        self.assertEqual(XS.day_str(p.d0 + p.n - 1), "2022-12-31")

    def test_stablecoins_and_fiat_excluded(self):
        """Live history_store drops stablecoins / fiat (trading_universe.STABLE_BASES); the 2017-2022 archive also has
        bases delisted before that list existed (PAX, USDS, USDSB, SUSD, AUD, BKRW, IDRT, BIDR, ...)."""
        import archive_bh as A
        import xsmom as XS
        for s in ("USDCUSDT", "BUSDUSDT", "TUSDUSDT", "PAXUSDT", "USDSUSDT", "USDSBUSDT", "SUSDUSDT", "EURUSDT",
                  "GBPUSDT", "AUDUSDT", "DAIUSDT", "USTUSDT", "BKRWUSDT", "IDRTUSDT", "BIDRUSDT", "USDPUSDT"):
            self.assertFalse(A.keep_symbol(s), s)
        for s in ("BTCUSDT", "AUDIOUSDT", "PAXGUSDT", "SUSHIUSDT", "ETHUSDT"):
            self.assertTrue(A.keep_symbol(s), s)
        day = 86_400_000
        t = np.arange(3 * 96, dtype=np.int64) * 900000 + XS.day_num("2022-12-01") * day
        c = np.linspace(1, 2, len(t))
        arr = XS.daily_from_15m({"t": t, "o": c, "h": c, "l": c, "c": c, "v": np.ones(len(t))})
        p = A.daily_panel({"XUSDT": arr, "USDCUSDT": arr, "PAXUSDT": arr})
        self.assertEqual(list(p.syms), ["XUSDT"])

    def test_train_mask_and_overall(self):
        import experiment_bh19 as E
        days = np.array(["2019-05-01", "2023-03-01", "2025-12-31", "2026-01-01"])
        self.assertEqual(E.train_mask(days, "2023-03-01", "2026-01-01").tolist(), [False, True, True, False])
        r = E.overall({"DIR_1d": {"accepted": False}, "FC_PASSIVE": {"accepted": True}})
        self.assertEqual(r["confirmed"], ["FC_PASSIVE"])
        self.assertEqual(r["not_confirmed"], ["DIR_1d"])
        self.assertAlmostEqual(E.ALPHA, 0.01)


    def test_fc_uses_carry_fees(self):
        """FC_PASSIVE is judged with the cost model of its own train/test (carry.fees(1)); fill_model has no .slip and
        crashed the first BH19 run before any number was computed."""
        import carry as C
        import experiment_bh19 as E
        f = E.fc_costs()
        self.assertIsInstance(f, C.Fees)
        self.assertEqual(f, C.fees(1))

if __name__ == "__main__":
    unittest.main(verbosity=2)
