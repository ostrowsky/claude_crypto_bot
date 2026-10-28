"""FS -- BTC+ETH carry switched on by funding (fc-switch-n4-1028-spec.md §6). Written BEFORE fc_switch.py / experiment_fs.py."""
import unittest

import numpy as np

from test_funding_carry import BASE_MS, DAY_MS, events, market
from test_xsmom import arr15, ds

SYMS = ("BTCUSDT", "ETHUSDT")


def levels_events(levels, hours=(0, 8, 16)):
    """Day k carries three events whose annualised W=1 sum is levels[k] (rate = level / 365 / 3)."""
    return events(range(len(levels)), rate=lambda d, h: levels[d] / 365.0 / len(hours), hours=hours)


class TestSchedule(unittest.TestCase):
    def test_hysteresis(self):
        import fc_switch as FSW
        # day k's signal (W = 1) is the level of day k-1
        lv = [0.20, 0.08, 0.06, 0.04, 0.08, 0.12, 0.051, 0.049, 0.30]
        f = {"BTCUSDT": levels_events(lv)}
        days = [ds(k) for k in range(1, len(lv) + 1)]
        s = FSW.schedule(f, ["BTCUSDT"], days, W=1, theta_in=0.10)["BTCUSDT"]
        # signal: 0.20 0.08 0.06 0.04 0.08 0.12 0.051 0.049 0.30
        self.assertEqual(list(s), [1, 1, 1, 0, 0, 1, 1, 0, 1])

    def test_no_data_is_off_and_no_lookahead(self):
        import fc_switch as FSW
        lv = [0.20] * 10
        f = {"BTCUSDT": levels_events(lv)}
        days = [ds(k) for k in range(0, 10)]
        s = FSW.schedule(f, ["BTCUSDT", "ETHUSDT"], days, W=1, theta_in=0.10)
        self.assertEqual(int(s["BTCUSDT"][0]), 0)                    # nothing paid before day 0
        self.assertTrue(np.all(s["ETHUSDT"] == 0))                   # no funding at all
        g = {"BTCUSDT": {"t": f["BTCUSDT"]["t"],
                         "rate": np.where(f["BTCUSDT"]["t"] >= BASE_MS + 5 * DAY_MS, -1.0, f["BTCUSDT"]["rate"])}}
        s2 = FSW.schedule(g, ["BTCUSDT"], days, W=1, theta_in=0.10)["BTCUSDT"]
        self.assertTrue(np.array_equal(s["BTCUSDT"][:6], s2[:6]))    # day 5 sees only events before 00:00 of day 5
        self.assertEqual(int(s2[6]), 0)

    def test_switch_count(self):
        import fc_switch as FSW
        self.assertEqual(FSW.n_switches({"A": np.array([0, 1, 1, 0, 1]), "B": np.array([1, 1, 1, 1, 1])}), 4)


def three_pair_market(n=60):
    rnd = np.random.default_rng(5)
    cols = {}
    for i, s in enumerate(("BTCUSDT", "ETHUSDT", "XUSDT")):
        p = list(100 * np.exp(np.cumsum(0.01 * rnd.standard_normal(n))))
        cols[s] = arr15(p)
    fund = {s: events(range(n), rate=0.0001 * (i + 1)) for i, s in enumerate(cols)}
    return market(cols, cols, fund)


class TestMarketAndChooser(unittest.TestCase):
    def test_subset_market_reproduces_passive(self):
        import carry as C
        import experiment_fc as EF
        import fc_switch as FSW
        m = three_pair_market()
        sub = FSW.subset_market(m, SYMS)
        self.assertEqual(sorted(sub.spot.syms), list(SYMS))
        self.assertEqual(sorted(sub.funding), list(SYMS))
        a = C.simulate(m, EF.passive_config(), C.fees(1), ds(5), ds(55))
        b = C.simulate(sub, EF.passive_config(), C.fees(1), ds(5), ds(55))
        self.assertEqual(a["series"], b["series"])
        self.assertEqual([(t["sym"], t["entry_day"], t["exit_day"]) for t in a["trades"]],
                         [(t["sym"], t["entry_day"], t["exit_day"]) for t in b["trades"]])

    def test_schedule_chooser(self):
        import fc_switch as FSW
        days = [ds(k) for k in range(4)]
        sched = {"BTCUSDT": np.array([1, 0, 1, 0]), "ETHUSDT": np.array([0, 0, 1, 1])}
        ch = FSW.schedule_chooser(sched, days)

        class Ctx:
            def __init__(self, day):
                self.day = day
        self.assertEqual(ch(Ctx(ds(0))), ["BTCUSDT"])
        self.assertEqual(ch(Ctx(ds(1))), [])
        self.assertEqual(sorted(ch(Ctx(ds(2)))), list(SYMS))
        self.assertEqual(ch(Ctx(ds(3))), ["ETHUSDT"])

    def test_schedule_drives_the_simulation(self):
        import carry as C
        import experiment_fc as EF
        import fc_switch as FSW
        m = FSW.subset_market(three_pair_market(), SYMS)
        days = [ds(k) for k in range(5, 55)]
        on = np.array([1 if 10 <= k < 30 else 0 for k in range(5, 55)])
        sched = {"BTCUSDT": on, "ETHUSDT": np.zeros(len(days), dtype=int)}
        r = C.simulate(m, EF.passive_config(), C.fees(1), ds(5), ds(55), chooser=FSW.schedule_chooser(sched, days))
        trades = [t for t in r["trades"] if t["reason"] != "margin_stop"]
        self.assertEqual([(t["sym"], t["entry_day"], t["exit_day"]) for t in trades], [("BTCUSDT", ds(10), ds(30))])


class TestShift(unittest.TestCase):
    def test_shift_keeps_share_and_switches(self):
        import fc_switch as FSW
        import timing as TM
        rnd = np.random.default_rng(2)
        s = {"BTCUSDT": (np.cumsum(rnd.standard_normal(400)) > 0).astype(float),
             "ETHUSDT": (np.cumsum(rnd.standard_normal(400)) > 0).astype(float)}
        sh = TM.shift_targets(s, seed=3)
        for k in s:
            self.assertAlmostEqual(sh[k].mean(), s[k].mean())
        circ = lambda x: int(np.sum(x[1:] != x[:-1]) + (x[0] != x[-1]))  # noqa: E731
        self.assertEqual(sum(circ(v) for v in sh.values()), sum(circ(v) for v in s.values()))
        _ = FSW


class TestAdmission(unittest.TestCase):
    def good(self, **kw):
        r = {"mean_pct": 0.02, "random_means": [0.0] * 50, "paired_passive": {"mean_pct": 0.005},
             "paired_passive_by_year": {"2023": 0.01, "2024": -0.001, "2025": 0.002}, "liquidations": 0,
             "n_switches": 6}
        r.update(kw)
        return r

    def test_admit(self):
        import experiment_fs as EX
        self.assertTrue(EX.admit(self.good())[0])
        self.assertFalse(EX.admit(self.good(paired_passive={"mean_pct": -0.0001}))[0])
        self.assertFalse(EX.admit(self.good(paired_passive_by_year={"2023": -0.01, "2024": -0.001, "2025": 0.002}))[0])
        self.assertFalse(EX.admit(self.good(random_means=[0.03] * 50))[0])
        self.assertFalse(EX.admit(self.good(liquidations=1))[0])
        self.assertFalse(EX.admit(self.good(n_switches=5))[0])

    def test_select(self):
        import experiment_fs as EX
        c = list(EX.candidates())
        self.assertEqual(len(c), 9)
        res = {n: {"admitted": False} for n in c}
        res[c[0]] = {"admitted": True, "paired_passive": {"mean_pct": 0.001}}
        res[c[3]] = {"admitted": True, "paired_passive": {"mean_pct": 0.004}}
        self.assertEqual(EX.select_winner(res, 0.2), c[3])
        self.assertIsNone(EX.select_winner(res, 0.5))

    def test_no_test_mode(self):
        import experiment_fs as EX
        self.assertEqual(EX.main(["test"]), 2)

    def test_train_market_cut(self):
        import experiment_fs as EX
        import fc_switch as FSW
        m = FSW.subset_market(three_pair_market(800), SYMS)
        cut = EX.train_market(m, test_from=ds(700))
        self.assertLess(cut.spot.n, 701)
        for s in SYMS:
            self.assertTrue(np.all(cut.funding[s]["t"] <= BASE_MS + 700 * DAY_MS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
