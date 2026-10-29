"""QB -- quarterly basis carry (quarterly-basis-qb-1029-spec.md §7). Written BEFORE quarterly.py / experiment_qb.py.

Interface the tests fix:
  expiry_of(year, q) -> ms of 08:00 UTC on the last Friday of Mar / Jun / Sep / Dec
  symbol_of(pair, expiry_ms) / parse_expiry(symbol)
  pick_contract(decision_ms, expiries, min_days=14, max_expiry_ms=None) -> expiry ms | None
  basis_ann(F, S, decision_ms, expiry_ms)
  simulate(spot, fut, theta, fees, day_from, day_to, pairs, start_equity, max_expiry_ms) -> {series, agg, trades,
        liquidations, margin_stops, conservation_ok}
      spot: pair -> 1h arrays; fut: symbol -> 1h arrays (t open ms, o, h, l, c)
"""
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

H = 3_600_000
D = 86_400_000


def ms(s):
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)


def h1(t0, closes, highs=None):
    c = np.asarray(closes, dtype=float)
    return {"t": np.arange(len(c), dtype=np.int64) * H + t0, "o": c.copy(), "h": np.asarray(highs, float) if highs is not None
            else c * 1.0, "l": c * 1.0, "c": c, "v": np.ones(len(c))}


def zero_fees():
    import carry as C
    return C.Fees(spot_taker=0.0, perp_taker=0.0, slip=0.0)


class TestCalendar(unittest.TestCase):
    def test_expiry_and_symbols(self):
        import quarterly as Q
        self.assertEqual(Q.expiry_of(2023, 1), ms("2023-03-31 08:00"))
        self.assertEqual(Q.expiry_of(2024, 2), ms("2024-06-28 08:00"))
        self.assertEqual(Q.expiry_of(2025, 3), ms("2025-09-26 08:00"))
        self.assertEqual(Q.expiry_of(2026, 4), ms("2026-12-25 08:00"))
        self.assertEqual(Q.symbol_of("BTCUSDT", ms("2025-03-28 08:00")), "BTCUSDT_250328")
        self.assertEqual(Q.parse_expiry("ETHUSDT_231229"), ms("2023-12-29 08:00"))

    def test_pick_contract(self):
        import quarterly as Q
        ex = [Q.expiry_of(2025, q) for q in (1, 2, 3, 4)] + [Q.expiry_of(2026, 1)]
        self.assertEqual(Q.pick_contract(ms("2025-03-10 01:00"), ex), Q.expiry_of(2025, 1))   # 18 days left
        self.assertEqual(Q.pick_contract(ms("2025-03-20 01:00"), ex), Q.expiry_of(2025, 2))   # 8 days left -> next
        self.assertIsNone(Q.pick_contract(ms("2025-12-20 01:00"), ex, max_expiry_ms=ms("2026-01-01 00:00")))
        self.assertEqual(Q.pick_contract(ms("2025-12-01 01:00"), ex, max_expiry_ms=ms("2026-01-01 00:00")),
                         Q.expiry_of(2025, 4))

    def test_basis_ann(self):
        import quarterly as Q
        self.assertAlmostEqual(Q.basis_ann(101.0, 100.0, 0, 73 * D), 0.01 * 365 / 73, places=12)


class TestMoney(unittest.TestCase):
    def setup(self, b=0.02, highs_factor=1.0):
        import quarterly as Q
        exp = Q.expiry_of(2025, 1)                       # 2025-03-28 08:00
        t0 = ms("2025-03-01 00:00")
        n = int((exp - t0) / H) + 48
        s = [100.0] * n
        k_exp = int((exp - H - t0) / H)                  # the 07:00 bar of the expiry day
        f = [100.0 * (1 + b) if i < k_exp else 100.0 for i in range(n)]
        hi = [x * highs_factor for x in f]
        spot = {"BTCUSDT": h1(t0, s)}
        fut = {Q.symbol_of("BTCUSDT", exp): h1(t0, f, hi)}
        return Q, spot, fut, exp

    def test_locked_basis_without_costs(self):
        Q, spot, fut, exp = self.setup(b=0.02)
        r = Q.simulate(spot, fut, 0.0, zero_fees(), "2025-03-02", "2025-03-30", pairs=("BTCUSDT",), start_equity=1000.0,
                       expiries=[exp])
        self.assertEqual(len([t for t in r["trades"] if t["reason"] == "expiry"]), 1)
        self.assertAlmostEqual(r["final_equity"], 1000.0 * (1 + 0.02 / 2), places=6)   # pnl = N b, capital = 2N
        self.assertTrue(r["conservation_ok"])
        self.assertEqual(r["liquidations"], 0)

    def test_threshold_blocks_entry(self):
        Q, spot, fut, exp = self.setup(b=0.001)          # 0.1%: at most 0.001 * 365 / 14 = 2.6%/yr, under 5%
        r = Q.simulate(spot, fut, 0.05, zero_fees(), "2025-03-02", "2025-03-30", pairs=("BTCUSDT",), start_equity=1000.0,
                       expiries=[exp])
        self.assertEqual(r["trades"], [])
        self.assertAlmostEqual(r["final_equity"], 1000.0, places=9)

    def test_liquidation_loses_the_margin(self):
        Q, spot, fut, exp = self.setup(b=0.02, highs_factor=2.5)   # the short's liquidation price is hit
        r = Q.simulate(spot, fut, 0.0, zero_fees(), "2025-03-02", "2025-03-30", pairs=("BTCUSDT",), start_equity=1000.0,
                       expiries=[exp])
        self.assertGreaterEqual(r["liquidations"], 1)
        self.assertLess(r["final_equity"], 1000.0)
        self.assertTrue(r["conservation_ok"])

    def test_decision_sees_only_closed_bars(self):
        import quarterly as Q
        exp = Q.expiry_of(2025, 1)
        t0 = ms("2025-03-01 00:00")
        n = int((exp - t0) / H) + 48
        s = [100.0] * n
        f = [100.0] * n
        k = int((ms("2025-03-05 00:00") - t0) / H)
        f[k + 1] = 150.0                                  # the 01:00 bar: after the decision
        r = Q.simulate({"BTCUSDT": h1(t0, s)}, {Q.symbol_of("BTCUSDT", exp): h1(t0, f)}, 0.05, zero_fees(), "2025-03-05",
                       "2025-03-06", pairs=("BTCUSDT",), start_equity=1000.0, expiries=[exp])
        self.assertEqual(r["trades"], [])


class TestAdmission(unittest.TestCase):
    def test_admit_select_marker(self):
        import experiment_qb as EX
        g = {"agg": {"mean_pct": 0.01, "lo95": 0.001}, "liquidations": 0, "r_by_year": {"2023": 0.01, "2024": -0.01,
                                                                                         "2025": 0.02},
             "theta": 0.05, "paired_t0": {"mean_pct": 0.001}}
        self.assertTrue(EX.admit(g)[0])
        for k, v in (("agg", {"mean_pct": 0.01, "lo95": -0.001}), ("liquidations", 1), ("paired_t0", {"mean_pct": -0.001}),
                     ("r_by_year", {"2023": -0.01, "2024": -0.01, "2025": 0.02})):
            self.assertFalse(EX.admit(dict(g, **{k: v}))[0], k)
        self.assertTrue(EX.admit(dict(g, theta=0.0, paired_t0=None))[0])          # T0 is not compared with itself
        res = {"QB_T0": dict(g, admitted=True), "QB_T5": dict(g, admitted=True, agg={"mean_pct": 0.02, "lo95": 0.01})}
        self.assertEqual(EX.select(res, 0.2), "QB_T5")
        self.assertIsNone(EX.select(res, 0.6))
        self.assertEqual(sorted(EX.candidates()), ["QB_T0", "QB_T10", "QB_T15", "QB_T5"])
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "M.json"
            EX.open_test(m, {})
            with self.assertRaises(RuntimeError):
                EX.open_test(m, {})
        self.assertEqual(EX.main(["test"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestT0SingleLook(unittest.TestCase):
    """§9: the 2026 look runs QB_T0 only, all contracts allowed, window to the last full spot day."""

    def test_last_full_day_and_decide(self):
        import experiment_qb as EX
        t0 = ms("2026-01-01 00:00")
        spot = {"BTCUSDT": h1(t0, [100.0] * (24 * 3 + 5)), "ETHUSDT": h1(t0, [100.0] * (24 * 2 + 1))}
        self.assertEqual(EX.last_full_day(spot), "2026-01-02")       # ETH has 2 full days
        ok = EX.decide_t0({"lo95": 0.001, "max_dd_pct": -1.0}, 0)
        self.assertTrue(ok["accepted"])
        self.assertFalse(EX.decide_t0({"lo95": -0.001, "max_dd_pct": -1.0}, 0)["accepted"])
        self.assertFalse(EX.decide_t0({"lo95": 0.001, "max_dd_pct": -1.0}, 1)["accepted"])
        self.assertFalse(EX.decide_t0({"lo95": 0.001, "max_dd_pct": -10.5}, 0)["accepted"])
