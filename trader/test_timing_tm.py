"""TM -- BTC/ETH spot timing (btc-eth-timing-tm-1028-spec.md §8). Written BEFORE timing.py / experiment_tm.py."""
import json
import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import xsmom as XS

D0 = "2023-01-01"
DAY_MS = 86_400_000


def panel_from(closes: dict, first=None):
    """closes: sym -> list of daily closes from D0; first_close = previous close unless given."""
    daily = {}
    d0 = XS.day_num(D0)
    for s, c in closes.items():
        c = np.asarray(c, dtype=float)
        n = len(c)
        fc = np.asarray(first[s], dtype=float) if first else np.r_[c[0], c[:-1]]
        daily[s] = {"day": np.arange(d0, d0 + n), "close": c, "first_close": fc, "qv": np.ones(n) * 1e9,
                    "n_bars": np.full(n, 96), "complete": np.ones(n, dtype=bool), "has_first": np.ones(n, dtype=bool)}
    return XS.Panel.from_daily(daily)


def walk(n, seed, drift=0.0, vol=0.03):
    rnd = np.random.default_rng(seed)
    return list(100 * np.exp(np.cumsum(drift + vol * rnd.standard_normal(n))))


def funding_from(rates_per_day: dict):
    """sym -> list of per-day rates; three payments at 00, 08, 16 UTC with that rate."""
    out = {}
    d0 = XS.day_num(D0)
    for s, rs in rates_per_day.items():
        t, r = [], []
        for i, v in enumerate(rs):
            for h in (0, 8, 16):
                t.append((d0 + i) * DAY_MS + h * 3_600_000)
                r.append(v)
        out[s] = {"t": np.asarray(t, dtype=np.int64), "rate": np.asarray(r, dtype=float)}
    return out


def zero_fm():
    import fill_model as FM
    return FM.FillModel(taker_fee=0.0, maker_fee=0.0, half_spread=0.0, impact_k=0.0)


SYMS = ("BTCUSDT", "ETHUSDT")


class TestNoLookahead(unittest.TestCase):
    def test_target_ignores_rows_from_d(self):
        import timing as TM
        n = 400
        base = {"BTCUSDT": walk(n, 1), "ETHUSDT": walk(n, 2)}
        fund = funding_from({s: list(np.random.default_rng(3).normal(1e-4, 5e-5, n)) for s in SYMS})
        r = 300
        for cand in TM.candidates().values():
            p1 = panel_from(base)
            alt = {s: v[:r] + [x * 3.0 for x in v[r:]] for s, v in base.items()}
            p2 = panel_from(alt)
            f2 = {s: {"t": f["t"], "rate": np.where(f["t"] >= (XS.day_num(D0) + r) * DAY_MS, 0.05, f["rate"])}
                  for s, f in fund.items()}
            for s in SYMS:
                self.assertAlmostEqual(TM.target(p1, fund, cand, s, r), TM.target(p2, f2, cand, s, r), places=12,
                                       msg=cand.name)

    def test_funding_paid_at_midnight_of_d_is_not_used(self):
        import timing as TM
        n = 200
        p = panel_from({s: walk(n, 5) for s in SYMS})
        rates = [1e-4] * n
        cand = TM.Cand(name="F_3_0.8", kind="F", W=3, q=0.80)
        r = 150
        base = TM.funding_signal(funding_from({"BTCUSDT": rates}), "BTCUSDT", XS.day_num(D0) + r, 3)
        spiked = list(rates)
        spiked[r] = 0.5                                     # payments at 00/08/16 of day d itself
        got = TM.funding_signal(funding_from({"BTCUSDT": spiked}), "BTCUSDT", XS.day_num(D0) + r, 3)
        self.assertAlmostEqual(base, got, places=15)
        _ = (p, cand)


class TestMoney(unittest.TestCase):
    def test_buy_and_hold_reproduces_price_ratio(self):
        import timing as TM
        n = 120
        c = {"BTCUSDT": walk(n, 7), "ETHUSDT": walk(n, 8)}
        p = panel_from(c)
        r0, r1 = 10, 100
        tg = {s: np.ones(r1 - r0) for s in SYMS}
        res = TM.simulate(p, tg, zero_fm(), XS.day_str(p.d0 + r0), XS.day_str(p.d0 + r1), start_equity=1000.0)
        # bought at first_close[r0] = close[r0 - 1], sold at the last mark = close[r1 - 1]
        exp = sum(500.0 * c[s][r1 - 1] / c[s][r0 - 1] for s in SYMS)
        self.assertAlmostEqual(res["final_equity"], exp, places=6)

    def test_on_off_is_product_of_in_market_days(self):
        import timing as TM
        n = 60
        c = walk(n, 11)
        p = panel_from({"BTCUSDT": c, "ETHUSDT": c})
        r0, r1 = 5, 50
        w = np.array([1.0 if (i // 4) % 2 == 0 else 0.0 for i in range(r1 - r0)])
        res = TM.simulate(p, {"BTCUSDT": w, "ETHUSDT": w}, zero_fm(), XS.day_str(p.d0 + r0), XS.day_str(p.d0 + r1),
                          start_equity=1000.0)
        g = 1.0
        for i in range(r1 - r0):
            r = r0 + i
            if w[i] == 1.0:
                nxt = c[r] if r + 1 < r1 else c[r1 - 1]
                g *= nxt / c[r - 1]            # held from first_close[r] = close[r-1] to first_close[r+1] = close[r]
        self.assertAlmostEqual(res["final_equity"], 1000.0 * g, places=6)

    def test_conservation_and_flat_after_window(self):
        import fill_model as FM
        import timing as TM
        n = 300
        p = panel_from({"BTCUSDT": walk(n, 21), "ETHUSDT": walk(n, 22)})
        res = TM.run(p, {}, TM.candidates()["T_ret_14"], FM.model("taker", 1), XS.day_str(p.d0 + 120),
                     XS.day_str(p.d0 + 290), start_equity=1000.0)
        self.assertTrue(res["conservation_ok"])
        self.assertAlmostEqual(res["final_equity"] - 1000.0, sum(t["pnl"] for t in res["trades"] if t["side"] == "SELL"),
                               places=6)

    def test_band(self):
        import timing as TM
        n = 40
        p = panel_from({"BTCUSDT": [100.0] * n, "ETHUSDT": [100.0] * n})
        r0, r1 = 5, 30
        w = np.full(r1 - r0, 0.5)
        w[10:] = 0.55
        res = TM.simulate(p, {"BTCUSDT": w, "ETHUSDT": w}, zero_fm(), XS.day_str(p.d0 + r0), XS.day_str(p.d0 + r1),
                          start_equity=1000.0)
        buys_after = [t for t in res["trades"] if t["day"] > XS.day_str(p.d0 + r0) and t["reason"] == "rebalance"]
        self.assertEqual(buys_after, [])                  # 0.05 inside the band
        w2 = np.full(r1 - r0, 0.5)
        w2[10:] = 0.70
        res2 = TM.simulate(p, {"BTCUSDT": w2, "ETHUSDT": w2}, zero_fm(), XS.day_str(p.d0 + r0), XS.day_str(p.d0 + r1),
                           start_equity=1000.0)
        later = [t for t in res2["trades"] if t["day"] == XS.day_str(p.d0 + r0 + 10)]
        self.assertEqual(len(later), 2)                   # 0.20 outside the band, both sleeves


class TestControls(unittest.TestCase):
    def test_shift_keeps_exposure_and_switches(self):
        import timing as TM
        rnd = np.random.default_rng(4)
        w = (np.cumsum(rnd.standard_normal(500)) > 0).astype(float)
        sh = TM.shift_targets({"BTCUSDT": w, "ETHUSDT": w}, seed=9)
        for s in SYMS:
            self.assertAlmostEqual(sh[s].mean(), w.mean(), places=12)
            sw = lambda x: int(np.sum(x[1:] != x[:-1]) + (x[0] != x[-1]))  # noqa: E731  circular switch count
            self.assertEqual(sw(sh[s]), sw(w))
            self.assertFalse(np.array_equal(sh[s], w))
        self.assertFalse(np.array_equal(sh["BTCUSDT"], sh["ETHUSDT"]))   # independent shifts per sleeve

    def test_ce_is_the_mean(self):
        import timing as TM
        w = {"BTCUSDT": np.array([0.0, 1.0, 1.0, 0.0]), "ETHUSDT": np.array([0.2, 0.4, 0.6, 0.8])}
        ce = TM.ce_targets(w)
        self.assertTrue(np.allclose(ce["BTCUSDT"], 0.5))
        self.assertTrue(np.allclose(ce["ETHUSDT"], 0.5))


class TestGrid(unittest.TestCase):
    def test_twenty_candidates_three_rows(self):
        import experiment_tm as EX
        c = EX.candidates()
        self.assertEqual(len(c), 20)
        self.assertEqual({k: len(v) for k, v in EX.ROWS.items()}, {"T": 10, "V": 6, "F": 4})


class TestAdmissionAndPipeline(unittest.TestCase):
    def good(self, **kw):
        r = {"mean_pct": 0.05, "random_means": [0.0] * 50, "paired_ce": {"mean_pct": 0.01},
             "paired_ce_by_year": {"2023": 0.01, "2024": -0.01, "2025": 0.02}, "n_rebal": 20}
        r.update(kw)
        return r

    def test_admit(self):
        import experiment_tm as EX
        self.assertTrue(EX.admit(self.good())[0])
        self.assertFalse(EX.admit(self.good(random_means=[0.06] * 50))[0])
        self.assertFalse(EX.admit(self.good(paired_ce={"mean_pct": -0.001}))[0])
        self.assertFalse(EX.admit(self.good(paired_ce_by_year={"2023": -0.01, "2024": -0.01, "2025": 0.02}))[0])
        self.assertFalse(EX.admit(self.good(n_rebal=11))[0])

    def test_select_by_row(self):
        import experiment_tm as EX
        names = EX.ROWS["T"]
        res = {n: {"admitted": False} for n in EX.candidates()}
        res[names[0]] = {"admitted": True, "paired_ce": {"mean_pct": 0.01}}
        res[names[1]] = {"admitted": True, "paired_ce": {"mean_pct": 0.02}}
        self.assertEqual(EX.select_by_row(res, {"T": 0.3, "V": 0.3, "F": 0.3}), {"T": names[1]})
        self.assertEqual(EX.select_by_row(res, {"T": 0.6, "V": 0.3, "F": 0.3}), {})

    def test_pipeline_marker(self):
        import experiment_tm as EX
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "M.json"
            r = EX.pipeline(lambda: {}, lambda w, a: {"x": 1}, m)
            self.assertTrue(r["stop"])
            self.assertFalse(m.exists())
            r = EX.pipeline(lambda: {"T": "T_ret_14"}, lambda w, a: {"alpha": a}, m)
            self.assertTrue(m.exists())
            self.assertAlmostEqual(r["alpha"], 0.05)
            with self.assertRaises(RuntimeError):
                EX.pipeline(lambda: {"T": "T_ret_14"}, lambda w, a: {}, m)
            with self.assertRaises(RuntimeError):
                EX.require_train(Path(d) / "nope.json")

    def test_test_needs_operator(self):
        import experiment_tm as EX
        self.assertEqual(EX.main(["test"]), 2)

    def test_decide(self):
        import experiment_tm as EX
        ok = EX.decide({"mean_pct": 0.01, "lo95": 0.001}, {"lo95": 0.001}, 0.05, [0.0] * 50)
        self.assertTrue(ok["accepted"])
        self.assertFalse(EX.decide({"mean_pct": 0.01, "lo95": -0.001}, {"lo95": 0.001}, 0.05, [0.0] * 50)["accepted"])
        self.assertFalse(EX.decide({"mean_pct": 0.01, "lo95": 0.001}, {"lo95": -0.001}, 0.05, [0.0] * 50)["accepted"])
        self.assertFalse(EX.decide({"mean_pct": 0.01, "lo95": 0.001}, {"lo95": 0.001}, 0.05, [0.06] * 50)["accepted"])


class TestTrainPanelCut(unittest.TestCase):
    def test_no_2026_rows(self):
        import experiment_tm as EX
        n = (XS.day_num("2026-03-01") - XS.day_num(D0))
        p = panel_from({s: walk(n, 31) for s in SYMS})
        cut = EX.train_panel(p)
        self.assertLess(XS.day_str(cut.d0 + cut.n - 1), "2026-01-01")
        f = funding_from({s: [1e-4] * n for s in SYMS})
        fc = EX.train_funding(f)
        for s in SYMS:
            self.assertLess(int(fc[s]["t"].max()), XS.day_num("2026-01-01") * DAY_MS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
