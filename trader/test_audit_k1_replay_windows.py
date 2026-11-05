"""K1 check of the external review (docs/audit/review-verification-2026-11-05.md): the replay is compared with the
live books on EQUAL windows (TH-04). Offline, a fake replay."""
import unittest


class TestK1Windows(unittest.TestCase):
    def test_windows_match_the_live_books(self):
        import _audit_k1_replay_windows as K
        w = {name: (lo, hi) for name, lo, hi in K.WINDOWS}
        self.assertEqual(w["h3_test"], ("2026-07-01", "2026-10-11"))      # H3 split 07-01, record ends 10-10
        self.assertEqual(w["e7_window"][0], "2026-10-13")                  # READOUT_E7 'from'
        self.assertEqual(w["demo_window"][0], "2026-10-20")                # C2 demo orders since 10-20

    def test_x4_policy_is_the_h3_winner(self):
        import _audit_k1_replay_windows as K
        p = K.x4_policy()
        self.assertEqual((p.exits.kind, p.exits.floor_override), ("p1_trail", 0.08))

    def test_run_windows_passes_each_window_and_summarises(self):
        import _audit_k1_replay_windows as K
        seen = []

        def run(lo, hi):
            seen.append((lo, hi))
            return {"agg": {"n_days": 3, "mean_pct": -1.0, "lo95": -2.0, "hi95": 0.0}, "trades": [1, 2]}
        rows = K.run_windows(run, lambda t: {"n": len(t), "net_mean_pct": -0.5, "gross_mean_pct": -0.1},
                             windows=(("a", "2026-01-01", "2026-01-04"),))
        self.assertEqual(seen, [("2026-01-01", "2026-01-04")])
        self.assertEqual(rows[0]["trades"], 2)
        self.assertEqual((rows[0]["mean_pct"], rows[0]["net_per_trade"]), (-1.0, -0.5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
