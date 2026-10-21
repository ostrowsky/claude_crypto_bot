"""S1 reversal -- tests written from s1-reversal-universe-1021-spec.md §10 BEFORE the code
(operator rule 2026-10-10: spec -> tests -> code). Synthetic data, offline.

Modules are imported inside each test, so before the code exists every test fails on its own."""
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from test_xsmom import FM1, FREE, ROOT, _xs, arr15, ds, panel, walk


def _s1():
    import experiment_s1
    return experiment_s1


# --------------------------------------------------------------------------- §10.1 reversal signal
class TestReversalSignal(unittest.TestCase):
    def test_bottom_k_order_and_ties(self):
        XS = _xs()
        s = {"DUSDT": -0.1, "CUSDT": 0.3, "AUSDT": -0.1, "BUSDT": 0.5}
        self.assertEqual(XS.bottom_k(s, 2), ["AUSDT", "DUSDT"])
        self.assertEqual(XS.bottom_k(s, 10), ["AUSDT", "DUSDT", "CUSDT", "BUSDT"])

    def test_reversal_chooser_names_and_defaults(self):
        XS = _xs()
        rev = XS.Config(L=7, H=1, K=2, side="reversal")
        self.assertEqual(rev.name, "R_L7_H1_K2")
        self.assertEqual(XS.Config(L=7, H=7, K=10, side="reversal", market_filter=True).name, "RF_L7_H7_K10")
        mom = XS.Config(L=7, H=1, K=2)
        self.assertEqual((mom.side, mom.min_age_days, mom.name), ("momentum", 0, "M_L7_H1_K2"))   # XSMOM unchanged
        ctx = XS.Ctx(day=ds(10), tranche=0, held=frozenset(), universe=["AUSDT", "BUSDT", "CUSDT"],
                     scores={"AUSDT": 0.1, "BUSDT": 0.5, "CUSDT": -0.3}, cfg=rev)
        self.assertEqual(XS.reversal_chooser(ctx), ["CUSDT", "AUSDT"])
        self.assertIs(XS.default_chooser(rev), XS.reversal_chooser)
        self.assertIs(XS.default_chooser(mom), XS.momentum_chooser)

    def test_rotate_default_follows_the_side(self):
        XS = _xs()
        w = walk(5, ("AUSDT", "BUSDT", "CUSDT", "DUSDT"), 60)
        p = panel({s: arr15(v, vol=1.0 + i) for i, (s, v) in enumerate(w.items())})
        rev = XS.Config(L=7, H=1, K=2, liq_min_days=20, side="reversal")
        mom = XS.Config(L=7, H=1, K=2, liq_min_days=20)
        a = XS.rotate(p, rev, FM1, ds(40), ds(60))
        self.assertEqual(a["series"], XS.rotate(p, rev, FM1, ds(40), ds(60), chooser=XS.reversal_chooser)["series"])
        c = XS.rotate(p, mom, FM1, ds(40), ds(60))
        self.assertEqual(c["series"], XS.rotate(p, mom, FM1, ds(40), ds(60), chooser=XS.momentum_chooser)["series"])
        self.assertNotEqual(a["series"], c["series"])


# --------------------------------------------------------------------------- §10.2 pair age
class TestMinAge(unittest.TestCase):
    def test_younger_than_min_age_is_out_and_default_unchanged(self):
        XS = _xs()
        d = 100
        cols = {"AUSDT": arr15([100.0] * (d + 1)),
                "BUSDT": arr15([None] * (d - 59) + [100.0] * 60),      # first bar on d-59
                "CUSDT": arr15([None] * (d - 60) + [100.0] * 61)}      # first bar on d-60: exactly 60 days
        p = panel(cols)
        rev = XS.Config(L=3, H=1, K=2, side="reversal", min_age_days=60)
        self.assertEqual(sorted(XS.universe(p, ds(d), rev)), ["AUSDT", "CUSDT"])
        self.assertEqual(sorted(XS.universe(p, ds(d), XS.Config(L=3, H=1, K=2))), ["AUSDT", "BUSDT", "CUSDT"])


# --------------------------------------------------------------------------- §10.3 references
class TestReferences(unittest.TestCase):
    def test_btc_is_held_the_whole_window(self):
        XS = _xs()
        cols = {"BTCUSDT": arr15([100.0, 100.0, 110.0, 105.0, 120.0]), "AUSDT": arr15([100.0] * 5)}
        r = XS.rotate(panel(cols), XS.Config(L=1, H=1, K=1), FREE, ds(1), ds(5), chooser=XS.btc_chooser)
        self.assertEqual([t["sym"] for t in r["trades"]], ["BTCUSDT"])
        self.assertAlmostEqual(r["final_equity"], 10_000.0 * 120.0 / 100.0, places=6)

    def test_delisted_later_share(self):
        XS = _xs()
        full = panel({"AUSDT": arr15([100.0] * 10), "BUSDT": arr15([100.0] * 6), "CUSDT": arr15([100.0] * 10)})
        trades = [{"sym": "AUSDT", "pnl": 5.0}, {"sym": "BUSDT", "pnl": -2.0}, {"sym": "BUSDT", "pnl": 1.0},
                  {"sym": "CUSDT", "pnl": 0.0}]
        out = XS.delisted_later_share(trades, full)
        self.assertEqual(out["pairs"], ["BUSDT"])
        self.assertEqual(out["n_trades"], 2)
        self.assertAlmostEqual(out["share_trades"], 0.5)
        self.assertAlmostEqual(out["pnl"], -1.0)


# --------------------------------------------------------------------------- §10.4 procedure
class TestProcedure(unittest.TestCase):
    def test_candidates_exactly_section_4(self):
        S1 = _s1()
        c = S1.candidates()
        names = {f"{row}_L{L}_H{H}_K10" for row in ("R", "RF") for L in (1, 7, 14, 28) for H in (1, 7)}
        self.assertEqual(set(c), names)
        for name, cfg in c.items():
            self.assertEqual(cfg.name, name)
            self.assertEqual((cfg.side, cfg.min_age_days, cfg.K), ("reversal", 60, 10))
            self.assertEqual(cfg.market_filter, name.startswith("RF_"))
        self.assertEqual(set(S1.ROWS), {"R", "RF"})
        self.assertEqual(sorted(sum(S1.ROWS.values(), [])), sorted(c))

    def test_constants(self):
        S1 = _s1()
        import experiment_xsmom as XE
        self.assertEqual((S1.TRAIN_FROM, S1.TEST_FROM), ("2023-03-01", "2026-01-01"))
        self.assertEqual((S1.N_RANDOM, S1.MIN_TRADES, S1.PBO_MAX, S1.DSR_MIN), (50, 200, 0.5, 0.90))
        self.assertEqual(S1.N_TRIALS_FAMILY, len(XE.candidates()) + len(S1.candidates()))
        self.assertEqual(S1.N_TRIALS_FAMILY, 40)
        self.assertEqual(S1.START_EQUITY, XE.START_EQUITY)

    def test_admission_is_the_xsmom_rule(self):
        S1 = _s1()
        import experiment_xsmom as XE
        base = {"mean_pct": 0.30, "random_means": [0.0] * 40 + [0.1] * 10, "paired_ewu": {"mean_pct": 0.05},
                "paired_ewu_by_year": {"2023": 0.1, "2024": -0.1, "2025": 0.2}, "n_trades": 250}
        for kw in ({}, {"mean_pct": 0.05}, {"paired_ewu": {"mean_pct": -0.01}},
                   {"paired_ewu_by_year": {"2023": 0.1, "2024": -0.1, "2025": -0.2}}, {"n_trades": 199}):
            r = dict(base, **kw)
            self.assertEqual(S1.admit(r)[0], XE.admit(r)[0], kw)
        self.assertTrue(S1.admit(base)[0])

    def test_single_winner_and_dsr_gate(self):
        S1 = _s1()
        res = {"R_L1_H1_K10": {"admitted": True, "paired_ewu": {"mean_pct": 0.10}},
               "R_L7_H7_K10": {"admitted": True, "paired_ewu": {"mean_pct": 0.20}},
               "RF_L14_H1_K10": {"admitted": True, "paired_ewu": {"mean_pct": 0.30}},
               "RF_L28_H7_K10": {"admitted": False, "paired_ewu": {"mean_pct": 0.90}}}
        self.assertEqual(S1.select_winner(res, {"R": 0.2, "RF": 0.3}), "RF_L14_H1_K10")
        self.assertEqual(S1.select_winner(res, {"R": 0.2, "RF": 0.6}), "R_L7_H7_K10")    # RF row closed by PBO
        self.assertIsNone(S1.select_winner(res, {"R": 0.5, "RF": 0.6}))
        self.assertTrue(S1.dsr_ok(0.90))
        self.assertFalse(S1.dsr_ok(0.899))
        self.assertFalse(S1.dsr_ok(None))

    def test_stop_once_and_one_window_for_the_family(self):
        S1 = _s1()
        with tempfile.TemporaryDirectory() as d:
            marker, xsmom = Path(d) / "S1_TEST_OPENED.json", Path(d) / "XSMOM_TEST_OPENED.json"
            called = []
            out = S1.pipeline(lambda: None, lambda w, a: called.append((w, a)), marker, (xsmom,))
            self.assertTrue(out["stop"])
            self.assertEqual(called, [])
            self.assertFalse(marker.exists())
            xsmom.write_text("{}", encoding="utf-8")                  # the window was opened by XSMOM
            with self.assertRaises(RuntimeError):
                S1.pipeline(lambda: "R_L1_H1_K10", lambda w, a: called.append((w, a)), marker, (xsmom,))
            self.assertEqual(called, [])
            self.assertFalse(marker.exists())
            xsmom.unlink()
            S1.pipeline(lambda: "R_L1_H1_K10", lambda w, a: called.append((w, a)), marker, (xsmom,))
            self.assertEqual(called, [("R_L1_H1_K10", 0.05)])
            self.assertTrue(marker.exists())
            with self.assertRaises(RuntimeError):
                S1.pipeline(lambda: "R_L1_H1_K10", lambda w, a: None, marker, (xsmom,))

    def test_family_markers_and_cli_guard(self):
        S1 = _s1()
        import experiment_xsmom as XE
        self.assertIn(XE.TEST_MARKER, S1.FAMILY_MARKERS)
        self.assertIn(S1.TEST_MARKER, S1.FAMILY_MARKERS)
        self.assertEqual(S1.main(["test"]), 2)                         # no --operator-approved: nothing opened

    def test_verdict_outcomes(self):
        S1 = _s1()
        rnd = [0.0] * 50
        self.assertEqual(S1.verdict(0.3, rnd, {"lo95": 0.01}, {"lo95": 0.02})["outcome"], "accepted")
        self.assertEqual(S1.verdict(0.3, rnd, {"lo95": 0.01}, {"lo95": -0.02})["outcome"], "shadow_candidate")
        self.assertEqual(S1.verdict(0.3, rnd, {"lo95": -0.01}, {"lo95": 0.02})["outcome"], "rejected")
        self.assertEqual(S1.verdict(-0.1, rnd, {"lo95": 0.01}, {"lo95": 0.02})["outcome"], "rejected")

    def test_train_panel_has_no_test_days(self):
        S1 = _s1()
        XS = _xs()
        p = S1.train_panel(panel({"AUSDT": arr15([100.0] * 10, base=datetime(2025, 12, 26, tzinfo=timezone.utc))}))
        self.assertLess(XS.day_str(p.days[-1]), S1.TEST_FROM)


# --------------------------------------------------------------------------- §10.5 boundary
class TestBoundary(unittest.TestCase):
    def test_bot_does_not_import_s1(self):
        for f in (ROOT / "files").glob("*.py"):
            self.assertNotIn("import experiment_s1", f.read_text(encoding="utf-8", errors="replace"), f.name)

    def test_outputs_in_trader_runtime(self):
        S1 = _s1()
        import trader_config as TC
        for p in (S1.OUT_DIR, S1.TEST_MARKER.parent):
            self.assertTrue(str(p).startswith(str(TC.RUNTIME)), p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
