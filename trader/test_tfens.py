"""TF-ENS Donchian trend ensemble (tf-ensemble-tfens-1118-spec.md §7). Written BEFORE the code. Synthetic, offline.

Interface the tests fix:
  tfens.channel_state(x, n, with_stop=False) -> np.ndarray of 0/1 (and the stop array), the state after the close of each t
  tfens.ensemble(x) -> np.ndarray, mean state over LOOKBACKS
  tfens.vol_mult(x, t) -> float (0.0 without enough history), from closes x[.. t]
  tfens.normalize(w) -> dict, sum <= 1
  tfens.need_trade(w_target, w_now, band) -> bool
  tfens.universe(panel, day) -> list (recomputed on the first day of the month only)
  tfens.Signals(panel) .weights(day, mode="tf"|"always", shift=None) -> {sym: w}
  tfens.shifts(syms, n_rows, seed) -> {sym: k}, SHIFT_MIN <= k <= n_rows - SHIFT_MIN
  tfens.simulate(panel, weight_fn, fm, day_from, day_to, band, start_equity) -> {"series", "trades", "n_rebal", "end_equity"}
       weight_fn(day) -> {sym: w}
  experiment_tfens.candidates(), admit(res), select(train), pipeline(validate, select, test, marker)
"""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import xsmom as XS

D0 = "2024-01-01"


def panel_from(closes: dict, day0: str = D0, qv=None, first=None):
    """Daily panel: closes = {sym: list}; first_close = close of the previous day unless given; qv constant per sym."""
    n = max(len(v) for v in closes.values())
    lo = XS.day_num(day0)
    daily = {}
    for s, v in closes.items():
        c = np.array(v, dtype=float)
        ok = np.isfinite(c)
        days = np.arange(lo, lo + len(c))[ok]
        fc = first[s] if first and s in first else np.r_[c[:1], c[:-1]]
        daily[s] = {"day": days, "close": c[ok], "first_close": np.asarray(fc, dtype=float)[ok],
                    "qv": np.full(ok.sum(), (qv or {}).get(s, 1e6)), "n_bars": np.full(ok.sum(), 96),
                    "complete": np.ones(ok.sum(), dtype=bool), "has_first": np.ones(ok.sum(), dtype=bool)}
    return XS.Panel.from_daily(daily)


def zero_fm():
    import fill_model as FM
    return FM.FillModel(taker_fee=0.0, maker_fee=0.0, half_spread=0.0, impact_k=0.0)


class TestChannel(unittest.TestCase):
    def test_entry_only_above_the_prior_max(self):
        import tfens as T
        x = np.array([10, 11, 12, 11, 12, 13], dtype=float)
        s = T.channel_state(x, 3)
        self.assertEqual(s[:3].tolist(), [0, 0, 0])      # fewer than n prior closes
        self.assertEqual(s[3], 0)                          # 11 < max(10, 11, 12)
        self.assertEqual(s[4], 0)                          # 12 is not above 12
        self.assertEqual(s[5], 1)                          # 13 > max(12, 11, 12)

    def test_stop_is_the_midpoint_and_only_rises(self):
        import tfens as T
        # enter at t=3 (13 > 12): window x[1..3] = 11, 12, 13 -> stop 12; t=4 close 12.5 holds; t=5 11.9 < 12 exits
        x = np.array([10, 11, 12, 13, 12.5, 11.9], dtype=float)
        s = T.channel_state(x, 3)
        self.assertEqual(s.tolist(), [0, 0, 0, 1, 1, 0])
        st, stop = T.channel_state(x, 3, with_stop=True)
        self.assertAlmostEqual(stop[3], 12.0)                    # (13 + 11) / 2
        self.assertAlmostEqual(stop[4], 12.5)                    # max(12, (13 + 12) / 2)
        # property on random walks: within a long spell the stop never falls, it is >= the channel midpoint,
        # and every exit is a close below the previous day's stop
        rng = np.random.default_rng(11)
        for seed in range(20):
            x = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.03, 300)))
            for n in (5, 20, 60):
                st, stop = T.channel_state(x, n, with_stop=True)
                for t in range(n + 1, len(x)):
                    if st[t] and st[t - 1]:
                        self.assertGreaterEqual(stop[t] + 1e-12, stop[t - 1])
                        w = x[t - n + 1:t + 1]
                        self.assertGreaterEqual(stop[t] + 1e-12, (w.max() + w.min()) / 2)
                    if st[t - 1] and not st[t]:
                        self.assertLess(x[t], stop[t - 1])

    def test_missing_close_changes_nothing(self):
        import tfens as T
        x = np.array([10, 11, 12, 13, np.nan, 12.5], dtype=float)
        s = T.channel_state(x, 3)
        self.assertEqual(s[3], 1)
        self.assertEqual(s[4], 1)
        self.assertEqual(s[5], 1)

    def test_ensemble_is_the_share_of_lookbacks_long(self):
        import tfens as T
        x = np.linspace(100, 200, 400)                    # steady rise: every lookback long once it has history
        e = T.ensemble(x)
        self.assertAlmostEqual(e[-1], 1.0)
        self.assertAlmostEqual(e[7], 1 / 9)               # only n = 5 has 5 prior closes at t = 7 (t >= n)
        self.assertEqual(len(T.LOOKBACKS), 9)


class TestSizing(unittest.TestCase):
    def test_vol_mult_cap_and_history(self):
        import tfens as T
        rng = np.random.default_rng(1)
        calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 200)))
        self.assertAlmostEqual(T.vol_mult(calm, 199), 2.0)            # sigma tiny -> capped
        wild = 100 * np.exp(np.cumsum(rng.normal(0, 0.05, 200)))
        v = T.vol_mult(wild, 199)
        r = np.diff(np.log(wild[199 - 90:200]))
        self.assertAlmostEqual(v, 0.25 / (np.std(r, ddof=1) * math.sqrt(365)), places=6)
        self.assertEqual(T.vol_mult(wild, 50), 0.0)                    # fewer than 60 returns

    def test_normalize(self):
        import tfens as T
        self.assertEqual(T.normalize({"A": 0.3, "B": 0.2}), {"A": 0.3, "B": 0.2})
        w = T.normalize({"A": 1.0, "B": 1.0})
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertAlmostEqual(w["A"], 0.5)

    def test_need_trade(self):
        import tfens as T
        self.assertFalse(T.need_trade(0.050, 0.048, 0.10))
        self.assertTrue(T.need_trade(0.050, 0.044, 0.10))
        self.assertTrue(T.need_trade(0.001, 0.0, 0.50))                # a full entry always trades
        self.assertTrue(T.need_trade(0.0, 0.001, 0.50))                # a full exit always trades
        self.assertFalse(T.need_trade(0.0, 0.0, 0.10))


class TestUniverse(unittest.TestCase):
    def _panel(self, n_days=500):
        closes, qv = {}, {}
        for i in range(25):
            s = "C%02dUSDT" % i
            closes[s] = [100.0] * n_days
            qv[s] = 1e6 * (i + 1)
        closes["BTCUPUSDT"] = [100.0] * n_days                       # leveraged token: never in
        qv["BTCUPUSDT"] = 1e12
        closes["BTCUSDT"] = [100.0] * n_days                         # its base, so the token is recognised; illiquid here
        qv["BTCUSDT"] = 1.0
        closes["YNGUSDT"] = [np.nan] * 300 + [100.0] * (n_days - 300)  # first bar 300 days in: too young until day 665
        qv["YNGUSDT"] = 1e12
        return panel_from(closes, qv=qv)

    def test_top20_no_leveraged_no_young(self):
        import tfens as T
        p = self._panel()
        day = XS.day_str(XS.day_num(D0) + 400)
        u = T.universe(p, day)
        self.assertEqual(len(u), 20)
        self.assertNotIn("BTCUPUSDT", u)
        self.assertNotIn("YNGUSDT", u)
        self.assertEqual(set(u), {"C%02dUSDT" % i for i in range(5, 25)})

    def test_reselected_on_the_first_day_of_the_month_only(self):
        import tfens as T
        p = self._panel()
        j = p.col["C00USDT"]
        p.qv[p.row("2025-02-05"):, j] = 1e13                    # C00 becomes the most liquid after Feb 1
        feb = T.universe(p, "2025-02-01")
        self.assertNotIn("C00USDT", feb)
        self.assertEqual(T.universe(p, "2025-02-17"), feb)
        self.assertIn("C00USDT", T.universe(p, "2025-03-01"))


class TestNoLookAhead(unittest.TestCase):
    def test_future_closes_do_not_change_todays_weights(self):
        import tfens as T
        rng = np.random.default_rng(3)
        base = {"C%02dUSDT" % i: list(100 * np.exp(np.cumsum(rng.normal(0.001, 0.03, 760)))) for i in range(22)}
        day = XS.day_str(XS.day_num(D0) + 700)
        w1 = T.Signals(panel_from(base)).weights(day)
        alt = {s: v[:700] + [x * 3 for x in v[700:]] for s, v in base.items()}
        w2 = T.Signals(panel_from(alt)).weights(day)
        self.assertEqual(w1.keys(), w2.keys())
        for s in w1:
            self.assertAlmostEqual(w1[s], w2[s])
        self.assertTrue(any(v > 0 for v in w1.values()))
        self.assertLessEqual(sum(w1.values()), 1.0 + 1e-12)

    def test_always_mode_has_full_ensemble(self):
        import tfens as T
        rng = np.random.default_rng(4)
        base = {"C%02dUSDT" % i: list(100 * np.exp(np.cumsum(rng.normal(0, 0.03, 760)))) for i in range(22)}
        sig = T.Signals(panel_from(base))
        day = XS.day_str(XS.day_num(D0) + 700)
        tf, al = sig.weights(day), sig.weights(day, mode="always")
        self.assertEqual(set(al), set(T.universe(sig.panel, day)))
        for s in al:
            self.assertGreaterEqual(al[s] + 1e-12, tf.get(s, 0.0))


class TestShift(unittest.TestCase):
    def test_bounds_and_seed(self):
        import tfens as T
        syms = ["A", "B", "C", "D"]
        k = T.shifts(syms, 731, seed=7)
        self.assertEqual(k, T.shifts(syms, 731, seed=7))
        self.assertNotEqual(k, T.shifts(syms, 731, seed=8))
        for v in k.values():
            self.assertTrue(T.SHIFT_MIN <= v <= 731 - T.SHIFT_MIN)


class TestSimulate(unittest.TestCase):
    def test_money_is_conserved_at_flat_prices_without_costs(self):
        import tfens as T
        p = panel_from({"AUSDT": [100.0] * 60, "BUSDT": [50.0] * 60})
        ws = [{"AUSDT": 0.5, "BUSDT": 0.3}, {"AUSDT": 0.2}, {}]
        fn = lambda day: ws[XS.day_num(day) % 3]                          # noqa: E731
        r = T.simulate(p, fn, zero_fm(), XS.day_str(XS.day_num(D0) + 5), XS.day_str(XS.day_num(D0) + 50), 0.10, 1_000_000.0)
        self.assertAlmostEqual(r["end_equity"], 1_000_000.0, places=4)
        for _, v in r["series"]:
            self.assertAlmostEqual(v, 0.0, places=9)
        self.assertGreater(r["n_rebal"], 10)

    def test_band_skips_small_resizes(self):
        import tfens as T
        p = panel_from({"AUSDT": [100.0] * 30})
        seq = {0: 0.50, 1: 0.52, 2: 0.60}
        d0 = XS.day_num(D0) + 5
        fn = lambda day: {"AUSDT": seq.get(XS.day_num(day) - d0, 0.60)}  # noqa: E731
        r = T.simulate(p, fn, zero_fm(), XS.day_str(d0), XS.day_str(d0 + 4), 0.10, 1_000_000.0)
        buys = [t for t in r["trades"] if t["side"] == "BUY"]
        self.assertEqual([t["day"] for t in buys], [XS.day_str(d0), XS.day_str(d0 + 2)])

    def test_gain_follows_the_weight(self):
        import tfens as T
        p = panel_from({"AUSDT": [100.0] * 10 + [110.0] * 10})          # +10% on the close of day 10
        d0 = XS.day_num(D0) + 5
        r = T.simulate(p, lambda day: {"AUSDT": 0.5}, zero_fm(), XS.day_str(d0), XS.day_str(d0 + 10), 0.10, 1_000_000.0)
        self.assertAlmostEqual(r["end_equity"], 1_050_000.0, places=2)

    def test_delisting_haircut(self):
        import tfens as T
        p = panel_from({"AUSDT": [100.0] * 10 + [np.nan] * 10, "BUSDT": [1.0] * 20})
        d0 = XS.day_num(D0) + 5
        r = T.simulate(p, lambda day: {"AUSDT": 0.5}, zero_fm(), XS.day_str(d0), XS.day_str(d0 + 12), 0.10, 1_000_000.0)
        self.assertAlmostEqual(r["end_equity"], 1_000_000.0 - 0.5 * 1_000_000.0 * T.DELIST_HAIRCUT, places=2)
        self.assertTrue(any(t["reason"] == "delisted" for t in r["trades"]))


def res(mean, p95, pa, years, n_rebal=50):
    return {"mean_pct": mean, "random_p95": p95, "paired_always": {"mean_pct": pa}, "paired_always_by_year": years, "n_rebal": n_rebal}


class TestSelection(unittest.TestCase):
    def test_candidates(self):
        import experiment_tfens as E
        c = E.candidates()
        self.assertEqual(sorted(c), ["TF_B10", "TF_B50"])
        self.assertEqual((c["TF_B10"].band, c["TF_B50"].band), (0.10, 0.50))

    def test_admit_needs_all(self):
        import experiment_tfens as E
        ok = res(0.05, 0.03, 0.01, {"2024": 0.02, "2025": 0.001})
        self.assertTrue(E.admit(ok)[0])
        for bad in (res(-0.01, -0.03, 0.01, {"2024": 0.02, "2025": 0.01}), res(0.05, 0.06, 0.01, {"2024": 0.02, "2025": 0.01}),
                    res(0.05, 0.03, -0.01, {"2024": 0.02, "2025": 0.01}), res(0.05, 0.03, 0.01, {"2024": 0.02, "2025": -0.001}),
                    res(0.05, 0.03, 0.01, {"2024": 0.02, "2025": 0.01}, n_rebal=11)):
            self.assertFalse(E.admit(bad)[0], bad)

    def test_select_and_stop(self):
        import experiment_tfens as E
        tr = {"TF_B10": {**res(0.05, 0.03, 0.01, {"2024": 0.1, "2025": 0.1}), "admitted": True},
              "TF_B50": {**res(0.07, 0.03, 0.01, {"2024": 0.1, "2025": 0.1}), "admitted": True}}
        self.assertEqual(E.select(tr), "TF_B50")
        tr["TF_B50"]["admitted"] = False
        self.assertEqual(E.select(tr), "TF_B10")
        m = Path(tempfile.mkdtemp()) / "TFENS_TEST_OPENED.json"
        calls = []
        r = E.pipeline(lambda: {"TF_B10": {"admitted": False}}, E.select, lambda w: calls.append(w), m)
        self.assertTrue(r["verdict"].startswith("STOP"))
        self.assertFalse(m.exists())
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
