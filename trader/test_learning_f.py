"""Phase F (trader learning loop) -- tests written from trader-learning-loop-f-1020-spec.md §6 BEFORE the code
(operator rule 2026-10-10: spec -> tests -> code). Synthetic data, offline, deterministic."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import trader_config as TC

HERE = Path(__file__).resolve().parent
T0 = datetime(2026, 10, 21, 0, 0, tzinfo=timezone.utc)


def iso(dt):
    return dt.isoformat()


def dec(kind, action, key, ts, **kw):
    """One executor decision row as the audit journal writes it (book exec_x4)."""
    sym, tf, ets = key.split("|")
    return {"schema": "audit.decision.v1", "ts": iso(ts), "process": "executor", "kind": kind, "book": "exec_x4",
            "mode": "demo", "action": action, "key": key, "sym": sym, "decision_id": f"{kind}|{key}|{action}", **kw}


KEY_A = "AAAUSDT|15m|2026-10-21T00:45:00+00:00"
KEY_B = "BBBUSDT|15m|2026-10-21T01:00:00+00:00"


# =============================================================================== F1 §6.1 trade dataset
class TestTradeDataset(unittest.TestCase):
    def rows(self):
        return [
            dec("real_entry", "enter", KEY_A, T0 + timedelta(minutes=46), fill_price=2.02, paper_price=2.0,
                slippage_bps=100.0, latency_sec=60.0, cost=20.0, qty=9.9),
            dec("real_entry", "skip", KEY_B, T0 + timedelta(minutes=61), reason="asset_held"),
            dec("real_exit", "exit", KEY_A, T0 + timedelta(hours=5), reason="trail", pnl=-0.8, ret_pct=-4.0),
            dec("real_trail", None, KEY_A, T0 + timedelta(hours=1)),                        # not a dataset row
            {"kind": "real_entry", "book": "champion", "action": "enter", "key": KEY_B, "ts": iso(T0)},  # other book
        ]

    def test_one_row_per_decision_and_outcome_joined(self):
        import trade_dataset as TD
        out = TD.rows_from_decisions(self.rows())
        self.assertEqual([(r["kind"], r["action"]) for r in out], [("entry", "enter"), ("entry", "skip")])
        a = out[0]
        self.assertEqual(a["key"], KEY_A)
        self.assertAlmostEqual(a["slippage_bps"], 100.0)
        self.assertEqual(a["source"], "live")
        self.assertAlmostEqual(a["outcome"]["ret_pct"], -4.0)          # joined from the exit row of the same key
        self.assertEqual(a["outcome"]["exit_reason"], "trail")
        self.assertIsNone(out[1]["outcome"])                           # a skip has no live outcome

    def test_open_position_has_no_outcome_yet(self):
        import trade_dataset as TD
        out = TD.rows_from_decisions(self.rows()[:2])
        self.assertIsNone(out[0]["outcome"])

    def test_snapshot_known_at_entry_only(self):
        import trade_dataset as TD
        t = T0 + timedelta(minutes=46)
        snaps = {"AAAUSDT": [(t - timedelta(minutes=20), {"spread_bps": 9.0}),
                             (t - timedelta(minutes=3), {"spread_bps": 4.0}),
                             (t + timedelta(minutes=1), {"spread_bps": 99.0})]}          # after the entry: never
        f = TD.snapshot_at(snaps, "AAAUSDT", t, max_age_sec=300)
        self.assertEqual(f, {"spread_bps": 4.0, "age_sec": 180.0})
        self.assertIsNone(TD.snapshot_at(snaps, "AAAUSDT", t - timedelta(minutes=5), max_age_sec=300))  # 15 min old
        self.assertIsNone(TD.snapshot_at(snaps, "ZZZUSDT", t, max_age_sec=300))

    def test_append_is_idempotent_and_survives_broken_lines(self):
        import trade_dataset as TD
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "trades.jsonl"
            p.write_text('{"decision_id": "old"}\n{broken\n', encoding="utf-8")
            rows = TD.rows_from_decisions(self.rows())
            self.assertEqual(TD.append_new(p, rows), 2)
            self.assertEqual(TD.append_new(p, rows), 0)                 # the same decisions again: nothing added
            ids = [json.loads(x)["decision_id"] for x in p.read_text(encoding="utf-8").splitlines()
                   if x.startswith("{") and x.endswith("}") and "broken" not in x]
            self.assertEqual(len(ids), len(set(ids)))

    def test_outcome_update_rewrites_nothing_old(self):
        """An entry first stored open gets its outcome later as a separate 'outcome' row, append-only."""
        import trade_dataset as TD
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "trades.jsonl"
            TD.append_new(p, TD.rows_from_decisions(self.rows()[:2]))
            before = p.read_text(encoding="utf-8")
            n = TD.append_new(p, TD.rows_from_decisions(self.rows()))
            after = p.read_text(encoding="utf-8")
            self.assertTrue(after.startswith(before))
            self.assertEqual(n, 1)                                      # only the outcome of KEY_A
            last = json.loads(after.splitlines()[-1])
            self.assertEqual(last["kind"], "outcome")
            self.assertEqual(last["key"], KEY_A)

    def test_counterfactual_rows_for_not_taken(self):
        import trade_dataset as TD
        skips = [r for r in TD.rows_from_decisions(self.rows()) if r["action"] == "skip"]
        cf = TD.counterfactual_rows(skips, replay_ret=lambda row: 1.5)
        self.assertEqual(len(cf), 1)
        self.assertEqual(cf[0]["source"], "replay")
        self.assertAlmostEqual(cf[0]["outcome"]["ret_pct"], 1.5)
        self.assertNotEqual(cf[0]["decision_id"], skips[0]["decision_id"])


# =============================================================================== F1 §6.2 live vs replay
class TestLiveVsReplay(unittest.TestCase):
    def test_zero_when_execution_matches_the_bar(self):
        import trade_dataset as TD
        live = [{"key": KEY_A, "ret_pct": 3.0}]
        r = TD.live_vs_replay(live, replay_ret=lambda t: 3.0)
        self.assertAlmostEqual(r["rows"][0]["diff_pct"], 0.0)
        self.assertAlmostEqual(r["mean_diff_pct"], 0.0)

    def test_sign_and_size_with_slippage(self):
        import trade_dataset as TD
        live = [{"key": KEY_A, "ret_pct": 3.0 - 0.5}, {"key": KEY_B, "ret_pct": -1.0 - 0.5}]
        r = TD.live_vs_replay(live, replay_ret=lambda t: 3.0 if t["key"] == KEY_A else -1.0)
        self.assertAlmostEqual(r["mean_diff_pct"], -0.5)                # live worse than the replay by 0.5 pp
        self.assertEqual(r["n"], 2)

    def test_missing_replay_counted_not_dropped_silently(self):
        import trade_dataset as TD
        r = TD.live_vs_replay([{"key": KEY_A, "ret_pct": 1.0}], replay_ret=lambda t: None)
        self.assertEqual(r["n"], 0)
        self.assertEqual(r["no_replay"], 1)


# =============================================================================== §6.3 purged walk-forward
class TestPurgedWalkForward(unittest.TestCase):
    def data(self, n_days=40):
        rows = []
        for d in range(n_days):
            for h in (1, 9, 17):
                t0 = T0 + timedelta(days=d, hours=h)
                rows.append({"t0": t0, "t1": t0 + timedelta(hours=30)})        # outcome known 30 h later
        return rows

    def test_no_label_leak_and_embargo(self):
        import purged_cv as PC
        rows = self.data()
        folds = PC.walk_forward(rows, n_folds=4, embargo_days=1)
        self.assertEqual(len(folds), 4)
        for tr, te in folds:
            self.assertTrue(tr and te)
            test_start = min(rows[i]["t0"] for i in te)
            for i in tr:
                self.assertLess(rows[i]["t1"], test_start - timedelta(days=1))   # outcome known before test, + embargo
            self.assertTrue(set(tr).isdisjoint(te))

    def test_folds_move_forward_in_time(self):
        import purged_cv as PC
        rows = self.data()
        folds = PC.walk_forward(rows, n_folds=4, embargo_days=1)
        starts = [min(rows[i]["t0"] for i in te) for _, te in folds]
        self.assertEqual(starts, sorted(starts))
        ends = [max(rows[i]["t0"] for i in te) for _, te in folds]
        for a, b in zip(ends, starts[1:]):
            self.assertLess(a, b)                                       # test blocks do not overlap


# =============================================================================== F2 §6.4 learned exit
def tiny_model():
    import exit_q as XQ
    return XQ.TreeRegressor(max_depth=4, min_leaf=20)


def peak_paths(n=300, seed=1):
    """Deterministic shape: +0.5%/bar up to a peak at bar k, then -1%/bar; X4 exits 8% below the peak."""
    rnd = np.random.default_rng(seed)
    paths = []
    for _ in range(n):
        k = int(rnd.integers(5, 40))
        up = 100 * np.cumprod(np.r_[1.0, np.full(k, 1.005)])
        down = up[-1] * np.cumprod(np.full(96 - k, 0.99))
        close = np.r_[up, down][:97]
        x4 = next((t for t in range(1, 97) if close[t] < close[: t + 1].max() * 0.92), 96)
        paths.append({"close": close, "x4_exit": x4, "cost": 0.0018, "peak": k})
    return paths


def walk_paths(n=300, seed=2):
    rnd = np.random.default_rng(seed)
    paths = []
    for _ in range(n):
        close = 100 * np.exp(np.r_[0.0, np.cumsum(rnd.normal(0, 0.01, 96))])
        x4 = next((t for t in range(1, 97) if close[t] < close[: t + 1].max() * 0.92), 96)
        paths.append({"close": close, "x4_exit": x4, "cost": 0.0018})
    return paths


class TestLearnedExit(unittest.TestCase):
    def test_exits_right_after_the_peak(self):
        import exit_q as XQ
        tr, te = peak_paths(300, 1), peak_paths(100, 3)
        pol = XQ.ExitPolicy(model_factory=tiny_model, delta=0.0005).fit(tr)
        err = [abs(pol.exit_bar(p) - (p["peak"] + 1)) for p in te]
        self.assertLessEqual(float(np.median(err)), 1.0)

    def test_random_walk_rarely_exits_early(self):
        import exit_q as XQ
        pol = XQ.ExitPolicy(model_factory=tiny_model, delta=0.002).fit(walk_paths(300, 2))
        te = walk_paths(200, 4)
        early = sum(1 for p in te if pol.exit_bar(p) < p["x4_exit"]) / len(te)
        self.assertLessEqual(early, 0.2)

    def test_never_later_than_x4_nor_96_bars(self):
        import exit_q as XQ
        pol = XQ.ExitPolicy(model_factory=tiny_model, delta=0.0).fit(walk_paths(200, 5))
        for p in walk_paths(100, 6) + peak_paths(50, 7):
            b = pol.exit_bar(p)
            self.assertLessEqual(b, p["x4_exit"])
            self.assertLessEqual(b, 96)
            self.assertGreaterEqual(b, 1)

    def test_features_use_bars_up_to_t_only(self):
        import exit_q as XQ
        p = peak_paths(1, 8)[0]
        q = dict(p, close=p["close"].copy())
        q["close"][30:] = 1.0                                           # change the future
        np.testing.assert_allclose(XQ.state(p["close"], 20), XQ.state(q["close"], 20))

    def test_deterministic(self):
        import exit_q as XQ
        tr = peak_paths(200, 9)
        a = XQ.ExitPolicy(model_factory=tiny_model, delta=0.001).fit(tr)
        b = XQ.ExitPolicy(model_factory=tiny_model, delta=0.001).fit(tr)
        te = peak_paths(30, 10)
        self.assertEqual([a.exit_bar(p) for p in te], [b.exit_bar(p) for p in te])


# =============================================================================== §6.5 trials registry, PBO, DSR
class TestOverfittingControls(unittest.TestCase):
    def test_pbo_one_real_strategy_among_noise(self):
        import pbo as PB
        rnd = np.random.default_rng(11)
        R = rnd.normal(0, 1, (400, 20))
        R[:, 0] += 0.25                                                 # one real edge
        self.assertLess(PB.cscv_pbo(R, n_splits=8), 0.2)

    def test_pbo_pure_noise_about_half(self):
        """About 0.5 IN EXPECTATION: one noise realisation ranges 0.09..0.93 (measured 2026-10-20, 40 seeds,
        mean 0.578) -- the first version of this test asserted it for a single seed, which was wrong."""
        import pbo as PB
        v = np.mean([PB.cscv_pbo(np.random.default_rng(s).normal(0, 1, (400, 20)), n_splits=8) for s in range(20)])
        self.assertTrue(0.35 <= v <= 0.75, v)

    def test_deflated_sharpe_falls_with_trials(self):
        import pbo as PB
        a = PB.deflated_sharpe(sr=0.15, n_trials=1, n_obs=250)
        b = PB.deflated_sharpe(sr=0.15, n_trials=50, n_obs=250)
        self.assertGreater(a, b)
        self.assertTrue(0.0 <= b <= 1.0 and 0.0 <= a <= 1.0)


# =============================================================================== §6.6 boundary, test opened once
class TestBoundaryF(unittest.TestCase):
    def test_f2_test_half_opens_once(self):
        import experiment_f2 as F2
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "F2_TEST_OPENED.json"
            F2.open_test(m, {"policy": "x"})
            with self.assertRaises(RuntimeError):
                F2.open_test(m, {"policy": "x"})

    def test_writes_only_trader_runtime(self):
        import trade_dataset as TD
        self.assertTrue(str(TD.DATASET).startswith(str(TC.RUNTIME)))
        for f in ("trade_dataset.py", "purged_cv.py", "exit_q.py", "pbo.py", "experiment_f2.py"):
            src = (HERE / f).read_text(encoding="utf-8")
            self.assertNotIn('"' + '.runtime' + '"', src, f)
            self.assertNotIn("BOT_FILES /", src.replace('BOT_FILES / "watchlist.json"', ""), f)

    def test_bot_does_not_import(self):
        for p in TC.BOT_FILES.glob("*.py"):
            s = p.read_text(encoding="utf-8", errors="replace")
            for m in ("trade_dataset", "purged_cv", "exit_q", "experiment_f2"):
                self.assertNotIn("import " + m, s, p.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
