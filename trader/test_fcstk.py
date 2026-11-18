"""FC-STK staking leg in the carry (fc-stk-staking-leg-1110-spec.md §6). Written BEFORE fcstk.py. Synthetic, offline.

Interface the tests fix:
  fcstk.simulate(market, cfg, fees, day_from, day_to, hold=None, chooser=None, slip_hold=SLIP_HOLD, band=BAND,
                 ratio_days=RATIO_DAYS, start_equity=...) -> the keys of carry.simulate plus
       "rehedges" (int), "rehedge_skipped" (int: days a re-hedge was due and the wallet could not pay its margin),
       "max_residual_delta" (max over days of |q_hold * rho_d - q_perp| / q_perp, after that day's re-hedge decision),
       "expected" [(day, pct)] on the days of "series": weight of the held leg at the mark x growth of the close ratio
  hold: {coin: instrument held on the long leg}; None -> fcstk.HOLD; {} -> plain FC
  rho_d: median of (instrument close / coin close) over the ratio_days days before d; none known -> first 15m closes of d
  fcstk.judge(e) -> {"passes", "anomaly", "failed"}                       (spec §5)
  fcstk.intraday_gaps(hold15, coin15, ratio_by_day, levels) -> {"n_bars": {level: n}, "worst", "worst_t", "bars"}
"""
import unittest

from test_funding_carry import events, market
from test_xsmom import ROOT, arr15, ds

DAY = 86_400_000


def _carry():
    import carry
    return carry


def _f():
    import fcstk
    return fcstk


def cfg(K=1):
    return _carry().CarryConfig(W=1, K=K, theta_in=0.0, liq_min_days=1)


def free():
    return _carry().Fees(0.0, 0.0, 0.0)


def only_eth(ctx):
    return ["ETHUSDT"]


def mk(e, w=None, funding=None, perp=None):
    spot = {"ETHUSDT": arr15(e)}
    if w is not None:
        spot["WBETHUSDT"] = arr15(w)
    return market(spot, {"ETHUSDT": arr15(perp or e)}, funding or {})


def two_pairs(with_copy=False):
    """BTC rises 60% (a margin stop), ETH drifts; perps off the spot by 0.1%; funding of both signs; 10 days."""
    btc = [100.0, 100.0, 102.0, 160.0, 158.0, 161.0, 150.0, 155.0, 152.0, 150.0]
    eth = [50.0, 50.0, 51.0, 49.0, 52.0, 53.0, 50.0, 51.0, 52.0, 50.0]
    spot = {"BTCUSDT": arr15(btc), "ETHUSDT": arr15(eth)}
    if with_copy:
        spot["WBETHUSDT"] = arr15(eth)
    perp = {"BTCUSDT": arr15([p * 1.001 for p in btc]), "ETHUSDT": arr15([p * 0.999 for p in eth])}
    rate = lambda d, h: 0.0001 * (1 + d % 3) - (0.0002 if h == 8 else 0.0)      # noqa: E731
    fund = {"BTCUSDT": events(range(10), rate=rate), "ETHUSDT": events(range(10), rate=rate)}
    return market(spot, perp, fund)


def passive():
    return _carry().CarryConfig(W=1, K=2, theta_in=0.0, passive=True, liq_min_days=1)


# --------------------------------------------------------------------------- §6.1-2 the engine is FC
class TestIdentity(unittest.TestCase):
    def assert_same(self, a, b):
        self.assertEqual([d for d, _ in a["series"]], [d for d, _ in b["series"]])
        for (_, x), (_, y) in zip(a["series"], b["series"]):
            self.assertAlmostEqual(x, y, places=9)
        self.assertAlmostEqual(a["final_equity"], b["final_equity"], places=6)
        self.assertAlmostEqual(a["funding_income"], b["funding_income"], places=6)
        self.assertEqual(a["margin_stops"], b["margin_stops"])
        self.assertEqual(a["liquidations"], b["liquidations"])
        self.assertEqual([(t["sym"], t["entry_day"], t["exit_day"], t["reason"]) for t in a["trades"]],
                         [(t["sym"], t["entry_day"], t["exit_day"], t["reason"]) for t in b["trades"]])
        for x, y in zip(a["trades"], b["trades"]):
            self.assertAlmostEqual(x["pnl"], y["pnl"], places=6)
        self.assertAlmostEqual(a["turnover_per_day"], b["turnover_per_day"], places=9)
        self.assertAlmostEqual(a["exposure_share"], b["exposure_share"], places=12)
        self.assertTrue(a["conservation_ok"] and b["conservation_ok"])

    def test_no_hold_is_carry_simulate(self):
        C, F = _carry(), _f()
        m, fees = two_pairs(), C.fees(1)
        a = C.simulate(m, passive(), fees, ds(1), ds(9), start_equity=10_000.0)
        self.assertGreaterEqual(a["margin_stops"], 1)                       # the stop path is exercised
        b = F.simulate(m, passive(), fees, ds(1), ds(9), hold={}, start_equity=10_000.0)
        self.assert_same(a, b)
        self.assertEqual(b["rehedges"], 0)
        self.assertTrue(all(abs(v) < 1e-12 for _, v in b["expected"]))

    def test_exact_copy_at_fc_slippage_is_fc(self):
        C, F = _carry(), _f()
        fees = C.fees(1)
        a = C.simulate(two_pairs(), passive(), fees, ds(1), ds(9), start_equity=10_000.0)
        b = F.simulate(two_pairs(with_copy=True), passive(), fees, ds(1), ds(9), slip_hold=fees.slip, start_equity=10_000.0)
        self.assert_same(a, b)
        self.assertEqual(b["rehedges"], 0)

    def test_coin_without_hold_trades_as_in_fc(self):
        C, F = _carry(), _f()
        btc, eth = [100.0] * 6, [50.0] * 6
        spot = {"BTCUSDT": arr15(btc), "ETHUSDT": arr15(eth), "WBETHUSDT": arr15([p * 1.1 for p in eth])}
        m = market(spot, {"BTCUSDT": arr15(btc), "ETHUSDT": arr15(eth)}, {})
        fees = C.fees(1)
        a = {t["sym"]: t for t in C.simulate(m, passive(), fees, ds(1), ds(4), start_equity=10_000.0)["trades"]}
        b = {t["sym"]: t for t in F.simulate(m, passive(), fees, ds(1), ds(4), start_equity=10_000.0)["trades"]}
        self.assertAlmostEqual(a["BTCUSDT"]["pnl"], b["BTCUSDT"]["pnl"], places=9)
        self.assertLess(b["ETHUSDT"]["pnl"], a["ETHUSDT"]["pnl"])           # 20 bp a side on the held leg, 8 bp in FC


# --------------------------------------------------------------------------- §6.3-4 hedge by the rate, the staking income
class TestHedgeAndIncome(unittest.TestCase):
    def test_value_does_not_depend_on_the_coin_price(self):
        F = _f()
        e = [100.0, 100.0, 130.0, 70.0, 100.0, 100.0]
        r = F.simulate(mk(e, [p * 1.1 for p in e]), cfg(), free(), ds(1), ds(5), chooser=only_eth, slip_hold=0.0, start_equity=10_000.0)
        self.assertEqual(len(r["series"]), 4)
        for _, v in r["series"]:                           # perp quantity = 1.1 x held quantity, or +-30% would show
            self.assertAlmostEqual(v, 0.0, places=9)
        self.assertAlmostEqual(r["final_equity"], 10_000.0, places=6)
        self.assertAlmostEqual(r["max_residual_delta"], 0.0, places=9)
        self.assertEqual(r["rehedges"], 0)

    def test_ratio_growth_is_earned_and_expected_matches(self):
        C, F = _carry(), _f()
        m = mk([100.0] * 6, [110.0, 110.0, 111.1, 112.2, 113.3, 113.3])
        r = F.simulate(m, cfg(), free(), ds(1), ds(4), chooser=only_eth, slip_hold=0.0, band=1.0, start_equity=10_000.0)
        fc = C.simulate(m, cfg(), free(), ds(1), ds(4), chooser=only_eth, start_equity=10_000.0)
        got = [v for _, v in r["series"]]
        # held leg 5000 of 10000; closes 110 -> 111.1 -> 112.2: marks 10000, 10000, 10050, 10100
        self.assertAlmostEqual(got[0], 0.0, places=9)
        self.assertAlmostEqual(got[1], 0.5, places=9)
        self.assertAlmostEqual(got[2], (10_100.0 / 10_050.0 - 1) * 100, places=9)
        self.assertAlmostEqual(r["final_equity"], 10_100.0, places=6)
        self.assertEqual([d for d, _ in r["expected"]], [d for d, _ in r["series"]])
        for (_, x), (_, y), (_, z) in zip(r["expected"], r["series"], fc["series"]):
            self.assertAlmostEqual(x, y - z, places=9)     # with no costs the paired difference is the expected gain
            self.assertAlmostEqual(z, 0.0, places=9)


# --------------------------------------------------------------------------- §6.5 re-hedge
UP = [110.0, 110.0, 110.66, 110.66, 110.66, 110.66, 110.66, 110.66]         # ratio 1.1 -> 1.1066: +0.6%


class TestRehedge(unittest.TestCase):
    def run_k2(self, w, fees=None, e=None, day_to=6, **kw):
        F = _f()
        return F.simulate(mk(e or [100.0] * len(w), w), cfg(2), fees or free(), ds(1), ds(day_to), chooser=only_eth, slip_hold=0.0,
                          ratio_days=1, start_equity=10_000.0, **kw)

    def test_only_at_the_band(self):
        below = self.run_k2([110.0, 110.0, 110.44, 110.44, 110.44, 110.44, 110.44, 110.44])     # +0.4%
        self.assertEqual(below["rehedges"], 0)
        self.assertAlmostEqual(below["max_residual_delta"], 0.004, places=9)
        above = self.run_k2(UP)
        self.assertEqual(above["rehedges"], 1)             # once; the ratio then stays
        self.assertEqual(above["rehedge_skipped"], 0)
        self.assertAlmostEqual(above["max_residual_delta"], 0.0, places=9)

    def test_only_upward(self):
        r = self.run_k2([110.0, 110.0, 108.9, 108.9, 108.9, 108.9, 108.9, 108.9])               # -1%
        self.assertEqual(r["rehedges"], 0)
        self.assertEqual(len(r["trades"]), 1)

    def test_value_is_neutral_again_after_it(self):
        e = [100.0, 100.0, 100.0, 100.0, 120.0, 80.0, 100.0, 100.0]
        w = [110.0, 110.0, 110.66, 110.66, 132.792, 88.528, 110.66, 110.66]                     # ratio 1.1066 from day 2
        r = self.run_k2(w, e=e, day_to=7)
        got = dict(r["series"])
        self.assertAlmostEqual(got[ds(2)], 0.15, places=9)                  # 2500 x 0.6% on 10000
        for d in (3, 4, 5, 6):
            self.assertAlmostEqual(got[ds(d)], 0.0, places=9)               # +20% and -20% of the coin change nothing
        self.assertAlmostEqual(r["final_equity"], 10_015.0, places=6)
        loose = self.run_k2(w, e=e, day_to=7, band=1.0)                     # without it the 0.6% of the leg is exposed
        self.assertGreater(abs(dict(loose["series"])[ds(4)]), 0.01)

    def test_cost_only_on_the_addition(self):
        C = _carry()
        r = self.run_k2(UP, fees=C.Fees(0.0, 0.001, 0.0))
        entry = 25.0 * 100.0 * 0.001                       # N 2500: 22.7273 held, 25 short
        added = 0.15 * 100.0 * 0.001                       # 25 -> 25.15
        exit_ = 25.15 * 100.0 * 0.001
        self.assertAlmostEqual(r["final_equity"], 10_000.0 + 15.0 - entry - added - exit_, places=6)
        self.assertTrue(r["conservation_ok"])

    def test_skipped_and_counted_when_the_wallet_is_empty(self):
        F = _f()
        r = F.simulate(mk([100.0] * 8, UP), cfg(1), free(), ds(1), ds(6), chooser=only_eth, slip_hold=0.0, ratio_days=1,
                       start_equity=10_000.0)              # K = 1: the whole equity is in the pair, no funding
        self.assertEqual(r["rehedges"], 0)
        self.assertEqual(r["rehedge_skipped"], 3)          # due on days 3, 4 and 5
        self.assertAlmostEqual(r["max_residual_delta"], 0.006, places=9)


# --------------------------------------------------------------------------- §6.6 the rate has no look-ahead
class TestRatio(unittest.TestCase):
    SPIKE = [110.0, 110.0, 110.0, 115.5, 110.0, 110.0, 110.0, 110.0]         # day 3 closes 5% high

    def sim(self, w, day_to, **kw):
        F = _f()
        return F.simulate(mk([100.0] * len(w), w), cfg(2), free(), ds(1), ds(day_to), chooser=only_eth, slip_hold=0.0,
                          start_equity=10_000.0, **kw)

    def test_the_close_of_day_d_is_not_used_on_day_d(self):
        self.assertEqual(self.sim(self.SPIKE, 4, ratio_days=1)["rehedges"], 0)      # decisions on days 1..3
        self.assertEqual(self.sim(self.SPIKE, 5, ratio_days=1)["rehedges"], 1)      # day 4 sees the close of day 3

    def test_one_day_outlier_does_not_move_the_median(self):
        self.assertEqual(self.sim(self.SPIKE, 7)["rehedges"], 0)                    # default: 7 days

    def test_data_at_or_after_day_to_changes_nothing(self):
        F = _f()
        a = mk([100.0] * 8, UP)
        b = mk([100.0] * 5 + [10.0] * 3, UP[:5] + [300.0] * 3)
        ra = F.simulate(a, cfg(2), free(), ds(1), ds(5), chooser=only_eth, slip_hold=0.0, ratio_days=1, start_equity=10_000.0)
        rb = F.simulate(b, cfg(2), free(), ds(1), ds(5), chooser=only_eth, slip_hold=0.0, ratio_days=1, start_equity=10_000.0)
        self.assertEqual(ra["series"], rb["series"])
        self.assertEqual(ra["final_equity"], rb["final_equity"])
        self.assertEqual(ra["rehedges"], rb["rehedges"])


# --------------------------------------------------------------------------- §6.7 the held instrument's own price
class TestDepeg(unittest.TestCase):
    def test_no_forced_sale_and_the_mark_recovers(self):
        F = _f()
        w = [110.0, 110.0, 110.0, 55.0, 110.0, 110.0, 110.0, 110.0]
        r = F.simulate(mk([100.0] * 8, w), cfg(), free(), ds(1), ds(7), chooser=only_eth, slip_hold=0.0, start_equity=10_000.0)
        got = dict(r["series"])
        self.assertAlmostEqual(got[ds(3)], -25.0, places=9)                 # half of the book halves for a day
        self.assertAlmostEqual(got[ds(4)], (10_000.0 / 7_500.0 - 1) * 100, places=9)
        self.assertEqual([t["reason"] for t in r["trades"]], ["flush"])
        self.assertEqual((r["margin_stops"], r["liquidations"], r["rehedges"]), (0, 0, 0))
        self.assertAlmostEqual(r["final_equity"], 10_000.0, places=6)


# --------------------------------------------------------------------------- §6.8-9 costs, funding
class TestMoney(unittest.TestCase):
    def test_held_leg_pays_spot_fee_and_its_own_slippage(self):
        C, F = _carry(), _f()
        fees = C.Fees(0.001, 0.0005, 0.0008)
        m = mk([100.0] * 6, [110.0] * 6, funding={"ETHUSDT": events(range(8))})
        r = F.simulate(m, cfg(), fees, ds(1), ds(4), chooser=only_eth, slip_hold=0.002, start_equity=10_000.0)
        q = 5000.0 / (110.0 * 1.002) * (1 - 0.001)
        qp = q * 1.1
        perp_in, perp_out = 100.0 * (1 - 0.0008), 100.0 * (1 + 0.0008)
        expected = (10_000.0 - 5000.0 + q * 110.0 * (1 - 0.002) * (1 - 0.001)             # held leg round trip
                    + qp * (perp_in - perp_out)                                           # short round trip
                    - qp * perp_in * 0.0005 - qp * perp_out * 0.0005                      # perp fees
                    + 9 * qp * 100.0 * 0.0001)                                            # funding on the short
        self.assertAlmostEqual(r["final_equity"], expected, places=6)
        self.assertTrue(r["conservation_ok"])
        self.assertAlmostEqual(r["final_equity"] - 10_000.0, sum(t["pnl"] for t in r["trades"]), places=6)

    def test_funding_is_paid_on_the_short_and_grows_with_it(self):
        F = _f()
        m = mk([100.0] * 8, UP, funding={"ETHUSDT": events(range(8))})
        r = F.simulate(m, cfg(2), free(), ds(1), ds(6), chooser=only_eth, slip_hold=0.0, ratio_days=1, start_equity=10_000.0)
        self.assertEqual(r["rehedges"], 1)                                  # at 00:15 of day 3
        self.assertAlmostEqual(r["funding_income"], 6 * 25.0 * 100.0 * 0.0001 + 9 * 25.15 * 100.0 * 0.0001, places=9)


# --------------------------------------------------------------------------- §6.10-11 stop, missing instrument
class TestStopAndMissing(unittest.TestCase):
    def test_margin_stop_closes_both_legs_and_reenters_at_the_new_rate(self):
        F = _f()
        e = [100.0, 100.0, 160.0, 160.0, 160.0, 200.0, 160.0, 160.0]
        w = [110.0, 110.0, 192.0, 192.0, 192.0, 240.0, 192.0, 192.0]         # ratio 1.1, then 1.2 from day 2
        r = F.simulate(mk(e, w), cfg(), free(), ds(1), ds(8), chooser=only_eth, slip_hold=0.0, ratio_days=1, start_equity=10_000.0)
        self.assertEqual([(t["reason"], t["exit_day"]) for t in r["trades"]], [("margin_stop", ds(3)), ("flush", ds(8))])
        self.assertEqual(r["trades"][1]["entry_day"], ds(4))
        self.assertEqual(r["margin_stops"], 1)
        eq = 10_000.0 + 5000.0 / 110.0 * (192.0 - 110.0) - 50.0 * 60.0       # held leg +82 a unit, short -60 a coin
        got = dict(r["series"])
        self.assertAlmostEqual(got[ds(2)], (eq / 10_000.0 - 1) * 100, places=9)
        for d in (3, 4, 5, 6, 7):                          # day 5: the coin +25%; neutral only at 1.2 short per unit
            self.assertAlmostEqual(got[ds(d)], 0.0, places=9)
        self.assertAlmostEqual(r["final_equity"], eq, places=6)
        self.assertTrue(r["conservation_ok"])

    def test_pair_is_not_opened_without_the_instrument(self):
        F = _f()
        r = F.simulate(mk([100.0] * 6), cfg(), free(), ds(1), ds(4), chooser=only_eth, slip_hold=0.0, start_equity=10_000.0)
        self.assertEqual(r["trades"], [])
        self.assertAlmostEqual(r["final_equity"], 10_000.0, places=9)
        self.assertEqual(r["exposure_share"], 0.0)


# --------------------------------------------------------------------------- §6.12 the criterion
def passing():
    return {"paired": {"20": {"lo95": 0.0012}, "60": {"lo95": 0.0009}}, "paired_by_year": {"2024": 0.0020, "2025": 0.0015},
            "margin_stops": 3, "fc_margin_stops": 3, "liquidations": 0, "fc_liquidations": 0,
            "max_dd_pct": -0.20, "fc_max_dd_pct": -0.17, "max_residual_delta": 0.006, "explained_share": 1.05}


class TestJudge(unittest.TestCase):
    def test_all_six_pass(self):
        self.assertEqual(_f().judge(passing()), {"passes": True, "anomaly": False, "failed": []})

    def test_each_of_the_first_five_stops(self):
        F = _f()
        cases = {"ci": lambda e: e["paired"]["60"].update(lo95=-0.0001),
                 "years": lambda e: e["paired_by_year"].update({"2025": -0.0001}),
                 "stops": lambda e: e.update(margin_stops=4),
                 "drawdown": lambda e: e.update(max_dd_pct=-0.30),
                 "residual_delta": lambda e: e.update(max_residual_delta=0.012)}
        for name, change in cases.items():
            e = passing()
            change(e)
            j = F.judge(e)
            self.assertFalse(j["passes"], name)
            self.assertFalse(j["anomaly"], name)
            self.assertEqual(j["failed"], [name])

    def test_edges(self):
        F = _f()
        e = passing()
        e["paired"]["20"]["lo95"] = None                    # too few days for an interval
        self.assertEqual(F.judge(e)["failed"], ["ci"])
        e = passing()
        e["paired"]["20"]["lo95"] = 0.0                     # the bound must be above zero
        self.assertEqual(F.judge(e)["failed"], ["ci"])
        e = passing()
        del e["paired_by_year"]["2024"]                     # both years are required
        self.assertEqual(F.judge(e)["failed"], ["years"])
        e = passing()
        e.update(liquidations=1)
        self.assertEqual(F.judge(e)["failed"], ["stops"])
        e = passing()
        e.update(max_dd_pct=-0.26)                          # deeper by 0.09 pp: inside the 0.10 pp allowance
        self.assertTrue(F.judge(e)["passes"])

    def test_unexplained_gain_is_an_anomaly_not_a_verdict(self):
        F = _f()
        for share in (0.5, 1.5, None):
            e = passing()
            e["explained_share"] = share
            j = F.judge(e)
            self.assertTrue(j["anomaly"], share)
            self.assertFalse(j["passes"], share)
            self.assertIn("explained", j["failed"])
        for share in (0.75, 1.35):
            e = passing()
            e["explained_share"] = share
            self.assertTrue(F.judge(e)["passes"], share)


# --------------------------------------------------------------------------- §6.13 intraday gaps
class TestIntradayGaps(unittest.TestCase):
    def test_finds_the_bar_and_counts_the_levels(self):
        F = _f()
        coin, hold = arr15([100.0] * 3), arr15([110.0] * 3)
        hold["l"][100] = 77.0                               # -30% against 1.1 x 100
        hold["l"][150] = 110.0 * 0.97                       # -3%
        g = F.intraday_gaps(hold, coin, {ds(0): 1.1, ds(1): 1.1, ds(2): 1.1}, levels=(0.01, 0.05, 0.20))
        self.assertEqual(g["bars"], 3 * 96)
        self.assertEqual(g["n_bars"], {0.01: 2, 0.05: 1, 0.20: 1})
        self.assertAlmostEqual(g["worst"], -0.30, places=9)
        self.assertEqual(g["worst_t"], int(hold["t"][100]))

    def test_days_without_a_rate_are_left_out(self):
        F = _f()
        coin, hold = arr15([100.0] * 3), arr15([110.0] * 3)
        hold["l"][10] = 55.0                                # day 0 has no rate: not compared
        g = F.intraday_gaps(hold, coin, {ds(1): 1.1, ds(2): 1.1}, levels=(0.01,))
        self.assertEqual(g["bars"], 2 * 96)
        self.assertEqual(g["n_bars"], {0.01: 0})


# --------------------------------------------------------------------------- procedure, boundary
class TestProcedure(unittest.TestCase):
    def test_constants(self):
        F = _f()
        self.assertEqual(F.SPEC, "fc-stk-staking-leg-1110-spec.md")
        self.assertEqual(F.HOLD, {"ETHUSDT": "WBETHUSDT"})
        self.assertEqual((F.BAND, F.RATIO_DAYS, F.SLIP_HOLD), (0.005, 7, 0.0020))
        self.assertEqual(F.WINDOW, ("2024-01-01", "2026-01-01"))            # day_to is exclusive: through 2025-12-31
        self.assertEqual(tuple(F.BLOCKS), (20, 60))
        self.assertEqual(tuple(F.SLIP_SENSITIVITY), (0.0008, 0.0020, 0.0050))

    def test_window_does_not_reach_2026(self):
        import experiment_fc as EF
        self.assertLessEqual(_f().WINDOW[1], EF.TEST_FROM)

    def test_output_in_trader_runtime(self):
        import trader_config as TC
        self.assertTrue(str(_f().OUT).startswith(str(TC.RUNTIME)))


class TestBoundary(unittest.TestCase):
    def test_bot_does_not_import_fcstk(self):
        for f in (ROOT / "files").glob("*.py"):
            self.assertNotIn("import fcstk", f.read_text(encoding="utf-8", errors="replace"), f.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
