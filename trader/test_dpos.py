"""DPOS -- derivatives positioning added to the 4h / 1d direction forecast (dir-positioning-dpos-1029-spec.md §5).
Written BEFORE metrics_store.py / experiment_dpos.py. Synthetic data, offline, no network.

Interface the tests fix:
  metrics_store.parse_csv(text) -> {t, oi, oi_val, top_acc, top_pos, ls_all, taker}   t = create_time ms, sorted, deduped
  metrics_store.merge(parsed_by_day: {day: arrays | None}) -> (arrays, missing_days)
  metrics_store.url_for(perp, day) / allowed_day(day) / plan(days, have_days) / fetch_day(perp, day, get)
  experiment_dpos.pos_features(m, t_ms, h_ms, spot_qv_med) -> {oi_c1, oi_c6, oi_qv, ls_top, ls_all, tk}
  experiment_dpos.funding_features(f, t_ms) -> {fund_last, fund_3d}
  experiment_dpos.has_pos(P) -> row mask used for BOTH arms
  experiment_dpos.paired_auc_ci(y, sa, sb, days, alpha, seed, boot) -> (delta, lo, hi)
  experiment_dpos.pit_check(pairs) -> {spearman_oi, med_rel_top, ok}
  experiment_dpos.outcome(res) -> "admit" | "info_only" | "stop";  open_test(marker, info);  main(["test"]) == 2
"""
import math
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MIN5 = 5 * 60 * 1000
H1 = 3600 * 1000
HEAD = ("create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
        "sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n")


def ms(s):
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)


def csv_day(day, n=288, oi0=100.0, step=1.0):
    lines = [HEAD]
    t0 = ms(day + " 00:00:00")
    for i in range(n):
        t = datetime.fromtimestamp((t0 + i * MIN5) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"{t},XUSDT,{oi0 + i * step},{(oi0 + i * step) * 10},1.5,1.2,2.0,{1.0 + i / 1000}\n")
    return "".join(lines)


def arrays(n, t0, oi=None, taker=None):
    t = t0 + np.arange(n, dtype=np.int64) * MIN5
    oi = np.full(n, 100.0) if oi is None else np.asarray(oi, float)
    return {"t": t, "oi": oi, "oi_val": oi * 10, "top_acc": np.full(n, 1.5), "top_pos": np.full(n, 1.2),
            "ls_all": np.full(n, 2.0), "taker": np.full(n, 1.0) if taker is None else np.asarray(taker, float)}


class TestStore(unittest.TestCase):
    def test_parse_and_dedupe(self):
        import metrics_store as M
        text = csv_day("2024-06-01", n=3)
        text += text.splitlines(keepends=True)[2]                     # duplicate of row 2 at the end
        a = M.parse_csv(text)
        self.assertEqual(len(a["t"]), 3)
        self.assertTrue(np.all(np.diff(a["t"]) == MIN5))
        self.assertAlmostEqual(a["oi"][1], 101.0)
        self.assertAlmostEqual(a["top_pos"][0], 1.2)
        self.assertAlmostEqual(a["ls_all"][0], 2.0)
        self.assertAlmostEqual(a["taker"][2], 1.002)

    def test_merge_records_missing_days(self):
        import metrics_store as M
        d1 = M.parse_csv(csv_day("2024-06-01", n=2))
        d3 = M.parse_csv(csv_day("2024-06-03", n=2))
        a, missing = M.merge({"2024-06-01": d1, "2024-06-02": None, "2024-06-03": d3})
        self.assertEqual(missing, ["2024-06-02"])
        self.assertEqual(len(a["t"]), 4)                                # nothing filled in for the missing day

    def test_downloader_scope(self):
        import metrics_store as M
        u = M.url_for("1000BONKUSDT", "2024-06-01")
        self.assertTrue(u.startswith("https://data.binance.vision/data/futures/um/daily/metrics/1000BONKUSDT/"))
        self.assertTrue(u.endswith("1000BONKUSDT-metrics-2024-06-01.zip"))
        self.assertTrue(M.allowed_day("2025-12-31"))
        self.assertFalse(M.allowed_day("2026-01-01"))
        self.assertTrue(M.allowed_day("2026-08-21"))                   # PIT window only
        self.assertFalse(M.allowed_day("2026-10-29"))
        self.assertEqual(M.plan(["2024-06-01", "2024-06-02", "2026-03-01"], {"2024-06-01"}), ["2024-06-02"])
        calls = []

        def get(url, method="GET"):
            calls.append((url, method))
            return None                                                 # 404 -> day missing
        self.assertIsNone(M.fetch_day("XUSDT", "2024-06-01", get))
        self.assertEqual(calls[0][1], "GET")
        with self.assertRaises(ValueError):
            M.fetch_day("XUSDT", "2026-02-01", get)

    def test_one_request_per_day_and_crc(self):
        import io
        import zipfile
        import metrics_store as M
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("XUSDT-metrics-2024-06-01.csv", csv_day("2024-06-01", n=4))
        raw = buf.getvalue()
        calls = []

        def get(url, method="GET"):
            calls.append(url)
            return raw
        a = M.fetch_day("XUSDT", "2024-06-01", get)
        self.assertEqual(len(calls), 1)                                 # no .CHECKSUM round trip
        self.assertEqual(len(a["t"]), 4)
        bad = bytearray(raw)
        pos = raw.index(b"PK\x01\x02")                                  # corrupt a byte of the compressed data
        bad[pos // 2] ^= 0xFF
        with self.assertRaises(Exception):
            M.fetch_day("XUSDT", "2024-06-01", lambda url, method="GET": bytes(bad))


class TestFeatures(unittest.TestCase):
    def test_availability_staleness_no_lookahead(self):
        import experiment_dpos as E
        t0 = ms("2024-06-01 00:00:00")
        oi = 100.0 * np.exp(np.arange(400) * 0.001)
        m = arrays(400, t0, oi=oi)
        t = t0 + 300 * MIN5                                             # decision time
        f = E.pos_features(m, t, H1, spot_qv_med=500.0)
        # row with create_time T is known at T + 5 min -> last usable row is index 299
        self.assertAlmostEqual(f["oi_c1"], math.log(oi[299] / oi[299 - 12]), places=9)
        self.assertAlmostEqual(f["oi_c6"], math.log(oi[299] / oi[299 - 72]), places=9)
        self.assertAlmostEqual(f["oi_qv"], math.log(oi[299] * 10 / 500.0), places=9)
        self.assertAlmostEqual(f["ls_top"], 1.2)
        self.assertAlmostEqual(f["ls_all"], 2.0)
        m2 = {k: v.copy() for k, v in m.items()}
        for k in ("oi", "oi_val", "taker"):
            m2[k][300:] *= 7.0
        self.assertEqual(E.pos_features(m2, t, H1, 500.0), f)
        stale = {k: v[:290] for k, v in m.items()}                     # last row known 45 min before t
        self.assertTrue(all(math.isnan(v) for v in E.pos_features(stale, t, H1, 500.0).values()))

    def test_taker_mean_over_last_bar(self):
        import experiment_dpos as E
        t0 = ms("2024-06-01 00:00:00")
        tk = np.arange(100, dtype=float)
        m = arrays(100, t0, taker=tk)
        t = t0 + 60 * MIN5
        f = E.pos_features(m, t, H1, 500.0)
        self.assertAlmostEqual(f["tk"], float(np.mean(tk[48:60])))       # rows known in (t - 1h, t]

    def test_funding(self):
        import experiment_dpos as E
        t0 = ms("2024-06-01 00:00:00")
        f = {"t": np.array([t0 + i * 8 * H1 for i in range(12)], dtype=np.int64), "rate": np.arange(12) / 1e4}
        t = t0 + 11 * 8 * H1 + 1
        r = E.funding_features(f, t)
        self.assertAlmostEqual(r["fund_last"], 11 / 1e4)
        self.assertAlmostEqual(r["fund_3d"], float(np.mean(np.arange(3, 12) / 1e4)))   # events in (t - 3d, t]

    def test_same_rows_for_both_arms(self):
        import experiment_dpos as E
        P = np.array([[0.1, 1.2, np.nan], [np.nan, np.nan, np.nan], [np.nan, 1.0, 2.0]])
        self.assertEqual(E.has_pos(P).tolist(), [True, False, True])


class TestStatsAndDecisions(unittest.TestCase):
    def test_paired_auc(self):
        import experiment_dpos as E
        rng = np.random.default_rng(7)
        n = 3000
        y = rng.integers(0, 2, n)
        days = np.repeat(np.arange(300), 10)
        sa = rng.normal(0, 1, n)
        d, lo, hi = E.paired_auc_ci(y, sa, sa.copy(), days, alpha=0.05, seed=29, boot=200)
        self.assertAlmostEqual(d, 0.0)
        self.assertLessEqual(lo, 0.0)
        self.assertGreaterEqual(hi, 0.0)
        d, lo, hi = E.paired_auc_ci(y, sa, y + rng.normal(0, 0.01, n), days, alpha=0.05, seed=29, boot=200)
        self.assertGreater(lo, 0.3)

    def test_pit_check(self):
        import experiment_dpos as E
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, 500)
        good = [{"bot_oi_1h": v, "arc_oi_1h": v + rng.normal(0, 0.01), "bot_top": 1.5, "arc_top": 1.51} for v in x]
        r = E.pit_check(good)
        self.assertTrue(r["ok"])
        bad = [{"bot_oi_1h": v, "arc_oi_1h": rng.normal(0, 1), "bot_top": 1.5, "arc_top": 1.51} for v in x]
        self.assertFalse(E.pit_check(bad)["ok"])
        drift = [{"bot_oi_1h": v, "arc_oi_1h": v, "bot_top": 1.5, "arc_top": 1.6} for v in x]
        self.assertFalse(E.pit_check(drift)["ok"])

    def test_pit2_compare(self):
        import experiment_dpos as E
        t0 = ms("2026-10-21 00:00:00")
        arc = arrays(200, t0, oi=np.linspace(100, 120, 200))
        sh = MIN5                                                       # REST timestamp = archive create_time + 5 min
        rest = {"oi": {int(t) + sh: float(v) for t, v in zip(arc["t"], arc["oi"])},
                "top": {int(t) + sh: 1.2 for t in arc["t"]}, "glob": {int(t) + sh: 2.0 for t in arc["t"]}}
        r = E.pit2_compare(arc, rest)
        self.assertTrue(r["ok"])
        self.assertAlmostEqual(r["coverage"], 1.0)
        bad = {k: dict(v) for k, v in rest.items()}
        for t in list(bad["oi"])[:5]:                                   # 2.5% of OI rows differ
            bad["oi"][t] *= 1.001
        self.assertFalse(E.pit2_compare(arc, bad)["ok"])
        thin = {k: dict(list(v.items())[:150]) for k, v in rest.items()}  # 75% coverage
        r = E.pit2_compare(arc, thin)
        self.assertFalse(r["ok"])
        self.assertAlmostEqual(r["coverage"], 0.75)

    def test_pit2_fetch_pages_backwards_and_shift(self):
        """Bug 10-29: the REST returns the LATEST `limit` rows of [start, end]; paging forward kept 500 of 2304.
        Archive create_time T == REST timestamp T + 5 min (500/500 exact on BTCUSDT 10-27..28)."""
        import experiment_dpos as E
        t0 = ms("2026-10-21 00:00:00")
        allrows = [{"timestamp": t0 + i * MIN5, "sumOpenInterest": str(100 + i)} for i in range(1200)]

        class Fake:
            def futures_data(self, path, symbol, period, start, end, limit):
                sel = [r for r in allrows if start <= r["timestamp"] <= end]
                return sel[-limit:]
        got = E.rest_series(Fake(), "/futures/data/openInterestHist", "sumOpenInterest", "XUSDT", t0, t0 + 1199 * MIN5)
        self.assertEqual(len(got), 1200)
        arc = arrays(1199, t0, oi=np.arange(1199) + 101.0)              # archive row T = REST value at T + 5 min
        r = E.pit2_compare(arc, {"oi": got, "top": {t + 0: 1.2 for t in got}, "glob": {t: 2.0 for t in got}})
        self.assertAlmostEqual(r["coverage"], 1.0)
        self.assertAlmostEqual(r["exact_share"]["oi"], 1.0)

    def test_pit2b_rounding_aware(self):
        """REST ratios have 4 decimals; the global ratio is computed from 4-decimal account shares."""
        import experiment_dpos as E
        t0 = ms("2026-10-21 00:00:00")
        n = 100
        arc = arrays(n, t0)
        la = 0.70573
        ratio = la / (1 - la)                                           # 2.39829...
        arc["top_pos"] = np.full(n, 1.919031)
        arc["ls_all"] = np.full(n, ratio)
        T = [int(t) + MIN5 for t in arc["t"]]
        la4, sa4 = round(la, 4), round(1 - la, 4)
        rest = {"oi": {t: 100.0 for t in T}, "top": {t: 1.9190 for t in T}, "glob": {t: round(la4 / sa4, 4) for t in T},
                "glob_long": {t: la4 for t in T}, "glob_short": {t: sa4 for t in T}}
        r = E.pit2b_compare(arc, rest)
        self.assertTrue(r["ok"], r)
        bad = dict(rest, top={t: 1.9195 for t in T})
        self.assertFalse(E.pit2b_compare(arc, bad)["ok"])
        bad = dict(rest, glob={t: 2.41 for t in T})
        self.assertFalse(E.pit2b_compare(arc, bad)["ok"])

    def test_pit2_perps_and_train_gate(self):
        import experiment_dpos as E
        counts = {"AUSDT": 10, "BUSDT": 30, "CUSDT": 30, "DUSDT": 5}
        self.assertEqual(E.pit2_perps(counts, 3), ["BUSDT", "CUSDT", "AUSDT"])
        self.assertTrue(E.train_allowed({"ok": False}, {"ok": True}))
        self.assertFalse(E.train_allowed({"ok": False}, {"ok": False}))
        self.assertFalse(E.train_allowed({"ok": False}, None))

    def test_outcome_marker_operator(self):
        import experiment_dpos as E
        money = {"sel": {"lo95": 0.01}, "paired": {"lo95": 0.001}, "sel_by_year": {"2023": 0.1, "2024": 0.1, "2025": -1}}
        no_money = {"sel": {"lo95": -0.01}, "paired": {"lo95": 0.001}, "sel_by_year": {"2023": 0.1, "2024": 0.1, "2025": 0.1}}
        self.assertEqual(E.outcome({"d_auc_lo": 0.001, "B": money}), "admit")
        self.assertEqual(E.outcome({"d_auc_lo": 0.001, "B": no_money}), "info_only")
        self.assertEqual(E.outcome({"d_auc_lo": -0.001, "B": money}), "stop")
        with tempfile.TemporaryDirectory() as d:
            mk = Path(d) / "DPOS_TEST_OPENED.json"
            E.open_test(mk, {})
            with self.assertRaises(RuntimeError):
                E.open_test(mk, {})
        self.assertEqual(E.main(["test"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
