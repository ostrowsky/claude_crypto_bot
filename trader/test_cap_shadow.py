"""CAP-SHADOW -- forward paper book of the frozen CAP_W4_X50_H24 (cap-shadow-1104-spec.md §5). Written BEFORE
cap_shadow.py. Synthetic arrays, offline.

Interface the tests fix:
  cap_shadow.step(bars, uni, state, now_ms, fm) -> (state, rows)      pure: no IO
      bars {sym: (t15, c15)}, uni {day_number: set}, state {} on the first call
      rows: {"kind": "event", "t_ms", "share"} and {"kind": "result", "t_ms", "ret_pct", "n_pairs", "rand_mean_pct"}
  cap_shadow.run_step(enabled, ...)   flag False -> nothing written
  cap_shadow.readout(results, today) -> {"status": COLLECTING | KEEP | REJECT | INCONCLUSIVE | TOO_FEW, ...}
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_cap import BAR, DAY, T0, market, zero_fm

H = 96


def crash_market(crash_bars=(300, 700)):
    grid, pairs, uni = market(n_pairs=60, n_bars=96 * 12)
    for k, (t, c) in pairs.items():
        if int(k[1:3]) < 40:                                   # 40 of 60 pairs (67%) fall 12% twice
            for b in crash_bars:
                c[b:] *= 0.88
    return grid, pairs, uni


def now_after(grid, i):
    """'now' right after the close of bar i."""
    return int(grid[i]) + BAR + 1000


class TestStep(unittest.TestCase):
    def test_events_and_results_match_cap(self):
        import cap
        import cap_shadow as S
        grid, pairs, uni = crash_market()
        st, rows = S.step(pairs, uni, {}, now_after(grid, len(grid) - 1), zero_fm())
        share = cap.breadth(grid, pairs, uni, 16, 0.10)
        exp = cap.events(share, 0.5, H)
        ev = [r for r in rows if r["kind"] == "event"]
        self.assertEqual([r["t_ms"] for r in ev], [int(grid[i]) for i in exp])
        res = [r for r in rows if r["kind"] == "result"]
        self.assertEqual(len(res), len(exp))
        for r, i in zip(res, exp):
            b = cap.basket(grid, i, H, pairs, sorted(uni[int(grid[i] // DAY)]), zero_fm())
            self.assertAlmostEqual(r["ret_pct"], b["ret_pct"], places=9)
            self.assertEqual(r["n_pairs"], b["n_pairs"])
            self.assertIsNotNone(r["rand_mean_pct"])

    def test_repeat_step_changes_nothing(self):
        import cap_shadow as S
        grid, pairs, uni = crash_market()
        now = now_after(grid, len(grid) - 1)
        st, rows = S.step(pairs, uni, {}, now, zero_fm())
        st2, rows2 = S.step(pairs, uni, json.loads(json.dumps(st)), now, zero_fm())
        self.assertEqual(rows2, [])
        self.assertEqual(st2, json.loads(json.dumps(st)))

    def test_catch_up_equals_incremental(self):
        import cap_shadow as S
        grid, pairs, uni = crash_market()
        _, once = S.step(pairs, uni, {}, now_after(grid, len(grid) - 1), zero_fm())
        st, inc = {}, []
        for i in range(250, len(grid), 37):
            st, r = S.step(pairs, uni, st, now_after(grid, i), zero_fm())
            inc += r
        st, r = S.step(pairs, uni, st, now_after(grid, len(grid) - 1), zero_fm())
        inc += r
        key = lambda rs: sorted((x["kind"], x["t_ms"], round(x.get("ret_pct", 0), 9)) for x in rs)
        self.assertEqual(key(inc), key(once))

    def test_event_waits_for_bar_after(self):
        import cap_shadow as S
        grid, pairs, uni = crash_market()
        _, r = S.step(pairs, uni, {}, now_after(grid, 300), zero_fm())       # bar 301 not closed yet
        self.assertEqual([x for x in r if x["kind"] == "event"], [])
        st, r = S.step(pairs, uni, {}, now_after(grid, 301), zero_fm())
        self.assertEqual(len([x for x in r if x["kind"] == "event"]), 1)
        self.assertEqual([x for x in r if x["kind"] == "result"], [])        # exit bar not closed


class TestRunStepFlag(unittest.TestCase):
    def test_disabled_writes_nothing(self):
        import cap_shadow as S
        with tempfile.TemporaryDirectory() as d:
            out = S.run_step(enabled=False, out_dir=Path(d))
            self.assertEqual(out, {"enabled": False})
            self.assertEqual(list(Path(d).iterdir()), [])


class TestReadout(unittest.TestCase):
    def res(self, rets, rand):
        return [{"kind": "result", "t_ms": T0 + k * DAY, "ret_pct": r, "rand_mean_pct": q, "n_pairs": 90}
                for k, (r, q) in enumerate(zip(rets, rand))]

    def test_rules(self):
        import cap_shadow as S
        rnd = np.random.default_rng(1)
        self.assertEqual(S.readout(self.res([5.0] * 14, [0.0] * 14), "2027-01-01")["status"], "COLLECTING")
        good = self.res(list(rnd.normal(4, 2, 20)), list(rnd.normal(0, 1, 20)))
        self.assertEqual(S.readout(good, "2027-01-01")["status"], "KEEP")
        bad = self.res(list(rnd.normal(-4, 2, 20)), list(rnd.normal(0, 1, 20)))
        self.assertEqual(S.readout(bad, "2027-01-01")["status"], "REJECT")
        mid = self.res(list(rnd.normal(0.2, 6, 20)), list(rnd.normal(0, 1, 20)))
        self.assertEqual(S.readout(mid, "2027-01-01")["status"], "INCONCLUSIVE")
        self.assertEqual(S.readout(self.res([5.0] * 9, [0.0] * 9), "2027-11-04")["status"], "TOO_FEW")


if __name__ == "__main__":
    unittest.main(verbosity=2)
