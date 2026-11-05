"""H1 step 0 -- could the coin be shorted at the announcement? (la-short-feasibility-1105-spec.md §4)
Written BEFORE la_short_feasibility.py. Offline.

Interface the tests fix:
  la_short_feasibility.status(t_pub_ms, perp_first_ms | None) -> ("perp_live" | "perp_later" | "no_perp", delay_h | None)
  la_short_feasibility.summary(rows) -> {etype: {"n", "perp_live", "perp_later", "no_perp", "unavailable_share",
       "median_delay_h", "out": bool}}     rows: [{"etype", "status", "delay_h"}]
"""
import unittest

H = 3_600_000


class TestStatus(unittest.TestCase):
    def test_status(self):
        import la_short_feasibility as F
        t = 1_700_000_000_000
        self.assertEqual(F.status(t, t - 3 * H), ("perp_live", None))
        self.assertEqual(F.status(t, t - 15 * 60 * 1000), ("perp_live", None))          # bar opened 15 min before
        self.assertEqual(F.status(t, t - 5 * 60 * 1000)[0], "perp_later")               # bar not closed at t_pub
        st, d = F.status(t, t + 2 * H)
        self.assertEqual(st, "perp_later")
        self.assertAlmostEqual(d, 2.0)
        self.assertEqual(F.status(t, None), ("no_perp", None))

    def test_summary(self):
        import la_short_feasibility as F
        rows = ([{"etype": "A", "status": "perp_live", "delay_h": None}] * 8
                + [{"etype": "A", "status": "perp_later", "delay_h": 1.0}] * 2
                + [{"etype": "B", "status": "perp_later", "delay_h": 3.0}] * 3
                + [{"etype": "B", "status": "no_perp", "delay_h": None}] * 1)
        s = F.summary(rows)
        self.assertEqual(s["A"]["n"], 10)
        self.assertAlmostEqual(s["A"]["unavailable_share"], 0.2)
        self.assertFalse(s["A"]["out"])
        self.assertAlmostEqual(s["B"]["unavailable_share"], 1.0)
        self.assertTrue(s["B"]["out"])
        self.assertAlmostEqual(s["B"]["median_delay_h"], 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
