"""X4 family in money -- tests written from x4-family-money-1015-spec.md §7 BEFORE the code
(operator rule 2026-10-10: spec -> tests -> code). Synthetic grids, offline."""
import inspect
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np

import fill_model as FM
import execution_replay as R
import trader_config as TC
import trading_policy as TP
from test_execution_replay import FakeGrid, cand, T0, STEP

HERE = Path(__file__).resolve().parent
X4 = replace(TP.CHAMPION, name="X4_p1_floor8", exits=TP.Exits(kind="p1_trail", floor_override=0.08))


def multi_grid(cols: dict, n: int):
    """FakeGrid with several symbols; cols = {sym: list of n closes} (np.nan allowed)."""
    syms = list(cols)
    closes = np.column_stack([np.asarray(cols[s], dtype=float) for s in syms])
    return FakeGrid(closes.tolist(), syms=syms)


def flat_pairs(k: int, n: int, up: bool):
    """k pairs whose last close sits above (up=True) or below their own long EMA."""
    out = {}
    for i in range(k):
        base = np.full(n, 100.0)
        base[-30:] = 110.0 if up else 90.0
        out[f"P{i}USDT"] = base
    return out


# --------------------------------------------------------------------------- §7.1 regime flags
class TestRegimeFlags(unittest.TestCase):
    N = 400

    def test_btc_down_from_close_vs_ema(self):
        import regime as RG
        btc_dn = np.full(self.N, 100.0)
        btc_dn[-30:] = 90.0
        g = multi_grid({"BTCUSDT": btc_dn, **flat_pairs(25, self.N, up=True)}, self.N)
        f = RG.flags(g)
        self.assertTrue(f["btc_down"][self.N - 1])
        self.assertFalse(f["btc_down"][self.N - 40])          # before the drop the close equals the EMA
        btc_up = np.full(self.N, 100.0)
        btc_up[-30:] = 110.0
        g2 = multi_grid({"BTCUSDT": btc_up, **flat_pairs(25, self.N, up=True)}, self.N)
        self.assertFalse(RG.flags(g2)["btc_down"][self.N - 1])

    def test_breadth_low(self):
        import regime as RG
        btc = np.full(self.N, 100.0)
        down = multi_grid({"BTCUSDT": btc, **flat_pairs(25, self.N, up=False)}, self.N)
        up = multi_grid({"BTCUSDT": btc, **flat_pairs(25, self.N, up=True)}, self.N)
        self.assertTrue(RG.flags(down)["breadth_low"][self.N - 1])
        self.assertFalse(RG.flags(up)["breadth_low"][self.N - 1])

    def test_no_look_ahead(self):
        """Changing bars AFTER ie leaves the flags at ie unchanged."""
        import regime as RG
        cols = {"BTCUSDT": np.full(self.N, 100.0), **flat_pairs(25, self.N, up=False)}
        g1 = multi_grid(cols, self.N)
        cols2 = {k: v.copy() for k, v in cols.items()}
        for v in cols2.values():
            v[300:] = 500.0
        g2 = multi_grid(cols2, self.N)
        f1, f2 = RG.flags(g1), RG.flags(g2)
        for key in ("btc_down", "breadth_low"):
            np.testing.assert_array_equal(f1[key][:300], f2[key][:300])

    def test_nan_excluded_and_min_pairs(self):
        import regime as RG
        btc = np.full(self.N, 100.0)
        down = flat_pairs(21, self.N, up=False)
        up = {k.replace("P", "U"): v for k, v in flat_pairs(30, self.N, up=True).items()}
        for v in up.values():                                 # the 30 "up" pairs have no close on the last bar
            v[-1] = np.nan
        g = multi_grid({"BTCUSDT": btc, **down, **up}, self.N)
        # excluded: 0 of 22 valid above -> low; forward-filled they would make 30 of 52 above -> not low
        self.assertTrue(RG.flags(g)["breadth_low"][self.N - 1])
        few = flat_pairs(15, self.N, up=False)                # only 15 pairs + BTC = 16 < 20
        g2 = multi_grid({"BTCUSDT": btc, **few}, self.N)
        self.assertFalse(RG.flags(g2)["breadth_low"][self.N - 1])  # no data -> no block
        self.assertEqual(RG.MIN_PAIRS, 20)

    def test_warmup_no_block(self):
        """Before EMA_BARS finite closes the EMA is not valid: no btc_down, pair not counted."""
        import regime as RG
        n = 150
        btc = np.full(n, 100.0)
        btc[-10:] = 50.0
        g = multi_grid({"BTCUSDT": btc, **flat_pairs(25, n, up=False)}, n)
        f = RG.flags(g)
        self.assertEqual(RG.EMA_BARS, 200)
        self.assertFalse(f["btc_down"].any())
        self.assertFalse(f["breadth_low"].any())

    def test_missing_btc_no_block(self):
        import regime as RG
        g = multi_grid(flat_pairs(25, self.N, up=False), self.N)
        self.assertFalse(RG.flags(g)["btc_down"].any())

    def test_blocks_table(self):
        import regime as RG
        t, f = True, False
        cases = {"R1": {(t, f): t, (f, t): f, (t, t): t, (f, f): f},
                 "R2": {(t, f): f, (f, t): t, (t, t): t, (f, f): f},
                 "R3": {(t, f): f, (f, t): f, (t, t): t, (f, f): f},
                 "R4": {(t, f): t, (f, t): t, (t, t): t, (f, f): f}}
        for name, table in cases.items():
            for (b, w), want in table.items():
                self.assertEqual(RG.blocks(name, {"btc_down": b, "breadth_low": w}), want, (name, b, w))
        self.assertFalse(RG.blocks("R4", None))               # no regime data -> no block
        with self.assertRaises(ValueError):
            RG.blocks("R9", {"btc_down": True, "breadth_low": True})


# --------------------------------------------------------------------------- §7.2 entry decision
class TestEntryDecision(unittest.TestCase):
    STATE = {"open": 0, "open_syms": set(), "day_count": 0, "day_return_pct": 0.0, "cooldown_until": {}, "bar": 0,
             "equity": 1e4}

    def test_default_regime_none_unchanged(self):
        self.assertIsNone(TP.Admission().regime)
        c = {"kind": "entry", "sym": "AUSDT", "ts": T0, "mode": "trend"}
        for pol in (TP.CHAMPION, X4):
            st = dict(self.STATE, regime={"btc_down": True, "breadth_low": True})
            self.assertEqual(TP.decide_entry_explained(pol, c, st), (0.1, "enter"))

    def test_regime_blocks_with_reason(self):
        c = {"kind": "entry", "sym": "AUSDT", "ts": T0, "mode": "trend"}
        pol = replace(X4, admission=TP.Admission(regime="R1"))
        st = dict(self.STATE, regime={"btc_down": True, "breadth_low": False})
        self.assertEqual(TP.decide_entry_explained(pol, c, st), (None, "regime_block"))
        st2 = dict(self.STATE, regime={"btc_down": False, "breadth_low": True})
        self.assertEqual(TP.decide_entry_explained(pol, c, st2), (0.1, "enter"))
        self.assertEqual(TP.decide_entry_explained(pol, c, dict(self.STATE)), (0.1, "enter"))   # no data

    def test_epoch_changes(self):
        self.assertNotEqual(X4.epoch(), replace(X4, admission=TP.Admission(regime="R3")).epoch())

    def test_existing_epochs_unchanged(self):
        """The new field must not re-version running policies (E7 shadow and executor journals key on the epoch).
        Values measured before the change, 2026-10-15."""
        self.assertEqual(TP.CHAMPION.epoch(), "e196b96e7004b46b")
        self.assertEqual(X4.epoch(), "0bccb4b2cf6d04f9")

    def test_engine_identical_without_regime_and_blocks_with_it(self):
        n = 96 * 5
        closes = [100.0 + (i % 50) * 0.1 for i in range(n)]
        g = FakeGrid(closes)
        cs = []
        for ie in range(220, n - 120, 40):
            cs.append(cand(ie, closes[ie], ix=ie + 8, exit_ts=T0 + (ie + 9) * STEP, exit_price=closes[ie + 8]))
        fm = FM.model("taker", 1)
        r0 = R.run(TP.CHAMPION, fm, g, [dict(c) for c in cs])
        flags = {"btc_down": np.ones(n, dtype=bool), "breadth_low": np.zeros(n, dtype=bool)}
        r1 = R.run(TP.CHAMPION, fm, g, [dict(c) for c in cs], regime_flags=flags)
        self.assertEqual(json.dumps(r0["series"]), json.dumps(r1["series"]))      # regime=None: flags ignored
        self.assertEqual(len(r0["trades"]), len(r1["trades"]))
        blocked = R.run(replace(TP.CHAMPION, admission=TP.Admission(regime="R1")), fm, g, [dict(c) for c in cs],
                        regime_flags=flags)
        self.assertEqual(blocked["trades"], [])
        passed = R.run(replace(TP.CHAMPION, admission=TP.Admission(regime="R2")), fm, g, [dict(c) for c in cs],
                       regime_flags=flags)
        self.assertEqual(len(passed["trades"]), len(r0["trades"]))


# --------------------------------------------------------------------------- candidates (spec §3)
class TestCandidates(unittest.TestCase):
    def test_family_registered(self):
        import experiment_x4f as X
        c = X.candidates()
        rows = {}
        for name, (row, pol) in c.items():
            rows.setdefault(row, []).append(pol)
        self.assertEqual({k: len(v) for k, v in rows.items()}, {"W": 6, "O": 6, "R": 4, "L": 3})
        self.assertNotIn("S", rows)                                         # H7 is not run (§2)
        floors = sorted(p.exits.floor_override for p in rows["W"] if p.exits.max_bars == 96)
        self.assertEqual(floors, [0.04, 0.06, 0.10, 0.12])
        self.assertEqual(sorted(p.exits.max_bars for p in rows["W"] if p.exits.floor_override == 0.08), [48, 192])
        self.assertEqual(sorted((p.exits.tp_atr, p.exits.sl_atr) for p in rows["O"]),
                         [(1.5, 1), (1.5, 1.5), (2, 1), (2, 1.5), (3, 1), (3, 1.5)])
        self.assertTrue(all(p.exits.kind == "oco_atr" and p.exits.max_bars == 96 for p in rows["O"]))
        self.assertEqual(sorted(p.admission.regime for p in rows["R"]), ["R1", "R2", "R3", "R4"])
        self.assertEqual(sorted(p.risk.daily_loss_limit_pct for p in rows["L"]), [1, 2, 3])
        for p in rows["R"] + rows["L"]:                                      # filters keep the X4 book
            self.assertEqual(p.exits, X4.exits)
        self.assertEqual(X.BASE.exits, X4.exits)
        self.assertEqual(X.SPLIT, "2026-07-01")


# --------------------------------------------------------------------------- §7.5 changed trades
def tr(sym, ie, ix, px):
    """Shape of execution_replay.run()['trades']: no exit price, gross_ret_pct carries it."""
    return {"sym": sym, "entry_ts": T0 + ie * STEP, "exit_time": T0 + ix * STEP, "gross_ret_pct": px, "ret_pct": 0.0}


class TestChanged(unittest.TestCase):
    def test_exit_rows_count_different_exits(self):
        import experiment_x4f as X
        base = [tr("A", 1, 5, 100.0), tr("B", 2, 6, 50.0), tr("C", 3, 7, 10.0)]
        cand_ = [tr("A", 1, 5, 100.0), tr("B", 2, 9, 50.0), tr("C", 3, 7, 11.0), tr("D", 4, 8, 1.0)]
        self.assertEqual(X.changed("W", base, cand_), 3)                     # B bar, C price, D new entry
        self.assertEqual(X.changed("O", base, base), 0)

    def test_filter_rows_count_skipped_entries(self):
        import experiment_x4f as X
        base = [tr("A", 1, 5, 100.0), tr("B", 2, 6, 50.0), tr("C", 3, 7, 10.0)]
        self.assertEqual(X.changed("R", base, [base[0]]), 2)
        self.assertEqual(X.changed("L", base, base), 0)


# --------------------------------------------------------------------------- §7.3 train selection
def res(mean, lo, changed, agg_mean=-0.2, thin=None):
    return {"paired": {"mean_pct": mean, "lo95": lo}, "changed": changed, "agg": {"mean_pct": agg_mean},
            "thin_median": thin}


class TestSelection(unittest.TestCase):
    def test_admit_conditions(self):
        import experiment_x4f as X
        self.assertTrue(X.admit("W", res(0.2, 0.05, 150)))
        self.assertFalse(X.admit("W", res(0.2, -0.01, 150)))                 # lo95 not > 0
        self.assertFalse(X.admit("O", res(0.2, 0.05, 99)))                   # < 100 changed
        self.assertFalse(X.admit("R", res(0.2, 0.05, 150, -0.2, None)))      # no thinning control -> no
        self.assertFalse(X.admit("R", res(0.2, 0.05, 150, -0.2, -0.1)))      # below random thinning
        self.assertTrue(X.admit("L", res(0.2, 0.05, 150, -0.05, -0.1)))      # beats random thinning
        self.assertEqual(X.MIN_CHANGED, 100)

    def test_one_winner_per_row(self):
        import experiment_x4f as X
        tr_ = {"W_f04": ("W", res(0.20, 0.05, 150)), "W_f12": ("W", res(0.30, 0.02, 150)),
               "O_t2s1": ("O", res(0.50, -0.01, 400)),
               "R1": ("R", res(0.40, 0.10, 300, -0.1, -0.2)), "R2": ("R", res(0.60, 0.10, 300, -0.3, -0.2)),
               "L1": ("L", res(0.10, 0.01, 50, 0.0, -0.5))}
        w = X.select_by_row(tr_)
        self.assertEqual(w, {"W": "W_f12", "O": None, "R": "R1", "L": None})
        self.assertEqual(list(inspect.signature(X.select_by_row).parameters), ["train_results"])

    def test_alpha_bonferroni(self):
        import experiment_x4f as X
        self.assertAlmostEqual(X.alpha_for(1), 0.05)
        self.assertAlmostEqual(X.alpha_for(4), 0.0125)
        with self.assertRaises(ValueError):
            X.alpha_for(0)

    def test_paired_k_wider(self):
        import experiment_x4f as X
        rnd = np.random.default_rng(3)
        days = [(T0 + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(120)]
        a = list(zip(days, rnd.normal(0.1, 1.0, 120).tolist()))
        b = list(zip(days, rnd.normal(0.0, 1.0, 120).tolist()))
        p1, p4 = X.paired_k(a, b, 1), X.paired_k(a, b, 4)
        self.assertLessEqual(p4["lo95"], p1["lo95"])
        self.assertGreaterEqual(p4["hi95"], p1["hi95"])
        self.assertAlmostEqual(p1["mean_pct"], p4["mean_pct"])

    def test_thinning_median(self):
        import experiment_x4f as X
        seen = []

        def run(frac, seed):                      # trades ~ 1000*frac, mean depends on seed
            seen.append(seed)
            return int(round(1000 * frac)), -0.5 + seed * 0.01
        out = X.thinning_median(run, target=400)
        self.assertEqual(sorted(set(seen) - {0}) + [0], list(range(1, 20)) + [0])
        self.assertAlmostEqual(out["median"], -0.5 + 9.5 * 0.01)
        self.assertLessEqual(abs(out["n_trades"] - 400), 0.05 * 400)
        self.assertEqual(X.N_THIN, 20)

    def test_validate_reads_train_days_only(self):
        import experiment_x4f as X
        calls = []

        def go(pol, lo, hi):
            calls.append((lo, hi))
            return {"series": [("2026-03-02", -0.1)], "agg": {"mean_pct": -0.1}, "trades": []}
        X.validate(go, {"W_f04": ("W", replace(X4, name="w"))}, thin=lambda *a: {"median": None, "n_trades": 0})
        self.assertTrue(calls)
        self.assertTrue(all(hi == X.SPLIT and (lo is None or lo < X.SPLIT) for lo, hi in calls))


# --------------------------------------------------------------------------- §7.4 test half
class TestTestHalf(unittest.TestCase):
    def test_stop_without_admitted_opens_nothing(self):
        import experiment_x4f as X
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d) / "X4F_TEST_OPENED.json"
            calls = []
            out = X.pipeline(validate=lambda: calls.append("train") or {},
                             select=lambda t: {"W": None, "O": None, "R": None, "L": None},
                             test=lambda winners, k: calls.append(("test", k)), marker=marker)
            self.assertEqual(calls, ["train"])
            self.assertFalse(marker.exists())
            self.assertIn("STOP", out["verdict"])

    def test_opens_once_with_k(self):
        import experiment_x4f as X
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d) / "X4F_TEST_OPENED.json"
            got = []
            X.pipeline(validate=lambda: {}, select=lambda t: {"W": "W_f12", "O": None, "R": "R1", "L": None},
                       test=lambda winners, k: got.append((sorted(winners), k)), marker=marker)
            self.assertEqual(got, [(["R1", "W_f12"], 2)])
            self.assertTrue(marker.exists())
            with self.assertRaises(RuntimeError):
                X.pipeline(validate=lambda: {}, select=lambda t: {"W": "W_f12", "O": None, "R": None, "L": None},
                           test=lambda winners, k: None, marker=marker)

    def test_requires_train_file(self):
        import experiment_x4f as X
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                X.require_train(Path(d) / "X4F_TRAIN.json")

    def test_row_verdict(self):
        import experiment_x4f as X
        self.assertEqual(X.row_verdict("W", improves=True, beats_thin=None), "accepted")
        self.assertEqual(X.row_verdict("O", improves=False, beats_thin=None), "rejected")
        self.assertEqual(X.row_verdict("R", improves=True, beats_thin=False), "less_trading_only")
        self.assertEqual(X.row_verdict("L", improves=True, beats_thin=True), "accepted")
        self.assertEqual(X.row_verdict("R", improves=False, beats_thin=True), "rejected")


# --------------------------------------------------------------------------- §7.6 boundary
class TestBoundary(unittest.TestCase):
    def test_stays_in_trader(self):
        for f in ("experiment_x4f.py", "regime.py"):
            src = (HERE / f).read_text(encoding="utf-8")
            self.assertNotIn('"' + '.runtime' + '"', src)
            self.assertNotIn("BOT_FILES", src)                    # nothing of the bot's files/ is written
        self.assertIn("TC.RUNTIME", (HERE / "experiment_x4f.py").read_text(encoding="utf-8"))

    def test_bot_does_not_import(self):
        bot = TC.BOT_FILES
        for p in bot.glob("*.py"):
            s = p.read_text(encoding="utf-8", errors="replace")
            self.assertNotIn("import regime", s, p.name)
            self.assertNotIn("experiment_x4f", s, p.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
