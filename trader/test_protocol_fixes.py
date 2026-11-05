"""Protocol fixes after the external review (protocol-fixes-1105-spec.md §3). Written BEFORE the code. Offline.

Interface the tests fix:
  provenance.events_digest(path, until) -> {"sha", "n"};  provenance.grid_digest(grid, i0, i1) -> str
  execution_replay.run(...) -> trades carry "censored"; result has "n_censored"
  daily_revenue.aggregate(series, alpha=, block=) -> also "ci_level", "ci_method"
  pbo.applicable(R, min_n=6, dup_corr=0.98) -> (bool, n_distinct);  pbo.cscv_pbo: a tie is half an overfit
  cap.book_marked(events, days) -> [(day, R %)]  (events: {"entry_close_ms", "exit_close_ms", "ret_pct", "marks"})
  cap.mean_ci(rets) with n < 10 -> lo95 / hi95 None, ci_method "too_few"
  docs/audit/alpha_ledger.json lists every trader/runtime/experiments/*_OPENED.json marker
"""
import json
import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import test_execution_replay as TER

ROOT = Path(__file__).resolve().parent.parent
DAY_MS = 86_400_000


class TestProvenance(unittest.TestCase):
    def test_events_digest_ignores_later_lines(self):
        import provenance as PV
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ev.jsonl"
            p.write_text('{"ts": "2026-10-01T00:00:00+00:00", "event": "entry"}\n'
                         '{"ts": "2026-10-02T00:00:00+00:00", "event": "exit"}\n', encoding="utf-8")
            a = PV.events_digest(p, "2026-10-05")
            with p.open("a", encoding="utf-8") as fh:
                fh.write('{"ts": "2026-10-06T00:00:00+00:00", "event": "entry"}\n')
            b = PV.events_digest(p, "2026-10-05")
            self.assertEqual(a, b)
            self.assertEqual(a["n"], 2)
            self.assertNotEqual(PV.events_digest(p, "2026-10-07")["sha"], a["sha"])

    def test_grid_digest_sees_a_price_change(self):
        import provenance as PV
        g = TER.FakeGrid([100.0] * 50)
        a = PV.grid_digest(g, 0, 40)
        g.C[45, 0] = 101.0
        self.assertEqual(PV.grid_digest(g, 0, 40), a)                # outside the window
        g.C[10, 0] = 101.0
        self.assertNotEqual(PV.grid_digest(g, 0, 40), a)


class TestBotState(unittest.TestCase):
    def test_bot_state_covers_config_overrides_and_imports(self):
        import provenance as PV
        b = PV.bot_state()
        for k in ("files/config.py", "files/exit_validator.py", "files/_backtest_trend_start_detector.py",
                  "files/_compute_early_capture.py", ".runtime/release/runtime_overrides.json", "files/watchlist.json"):
            self.assertIn(k, b["files"])
        self.assertIn("dirty", b)
        self.assertIn("bot", PV.block()["code"])


class TestCensored(unittest.TestCase):
    def test_edge_cut_trade_is_flagged(self):
        import execution_replay as R
        import fill_model as FM
        import trading_policy as TP
        from dataclasses import replace
        x4 = replace(TP.CHAMPION, name="x4", exits=TP.Exits(kind="p1_trail", floor_override=0.08))
        g = TER.FakeGrid([100.0] * 300)                               # flat: the 8% trail never fires
        full = TER.cand(100, 100.0)                                   # 96 bars fit: time exit at bar 196
        cut = TER.cand(250, 100.0)                                    # the grid ends at 299 < 250 + 96
        r = R.run(x4, FM.model("taker", 0), g, [full, cut])
        by_ie = {t["entry_ts"]: t for t in r["trades"]}
        self.assertFalse(by_ie[full["ts"]]["censored"])
        self.assertTrue(by_ie[cut["ts"]]["censored"])
        self.assertEqual(r["n_censored"], 1)


class TestInterval(unittest.TestCase):
    def series(self, n=200):
        rnd = np.random.default_rng(3)
        return [((datetime(2025, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d"), float(v))
                for i, v in enumerate(rnd.normal(0.1, 1.0, n))]

    def test_level_and_method(self):
        import daily_revenue as DR
        s = self.series()
        a = DR.aggregate(s)
        self.assertEqual(a["ci_method"], "block_bootstrap_percentile")
        self.assertAlmostEqual(a["ci_level"], 0.95)
        b = DR.aggregate(s, alpha=0.05 / 24)
        self.assertEqual(b["ci_method"], "normal_block_se")
        self.assertAlmostEqual(b["ci_level"], 1 - 0.05 / 24)
        self.assertLess(b["lo95"], a["lo95"])
        self.assertGreater(b["hi95"], a["hi95"])

    def test_too_few_and_block(self):
        import daily_revenue as DR
        a = DR.aggregate([("2026-01-01", 1.0), ("2026-01-02", 3.0)])
        self.assertIsNone(a["lo95"])
        self.assertIsNone(a["hi95"])
        self.assertEqual(a["ci_method"], "too_few")
        s = self.series(120)
        self.assertNotEqual(DR.aggregate(s, block=5)["lo95"], DR.aggregate(s, block=30)["lo95"])


class TestPbo(unittest.TestCase):
    def test_applicable(self):
        import pbo
        rnd = np.random.default_rng(1)
        x = rnd.normal(size=(300, 1))
        same = np.hstack([x, x * 1.0001])
        self.assertEqual(pbo.applicable(same)[0], False)
        many = rnd.normal(size=(300, 7))
        ok, n = pbo.applicable(many)
        self.assertTrue(ok)
        self.assertEqual(n, 7)

    def test_tie_is_half(self):
        import pbo
        x = np.random.default_rng(2).normal(size=(160, 1))
        self.assertAlmostEqual(pbo.cscv_pbo(np.hstack([x, x])), 0.5)


class TestCapBook(unittest.TestCase):
    def test_marked_book_sees_the_drawdown(self):
        import cap
        import daily_revenue as DR
        d0 = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        ev = {"entry_close_ms": d0 + DAY_MS // 2, "exit_close_ms": d0 + DAY_MS + DAY_MS // 2, "ret_pct": 5.0,
              "marks": {d0 + DAY_MS: 0.80}}
        days = ["2026-01-01", "2026-01-02", "2026-01-03"]
        book = dict(cap.book_marked([ev], days))
        self.assertAlmostEqual(book["2026-01-01"], -20.0, places=6)
        self.assertAlmostEqual(book["2026-01-02"], (1.05 / 0.80 - 1) * 100, places=6)
        self.assertAlmostEqual(book["2026-01-03"], 0.0)
        self.assertLessEqual(DR.aggregate(sorted(book.items()))["max_dd_pct"], -15.0)

    def test_mean_ci_too_few(self):
        import cap
        r = cap.mean_ci([1.2, 16.3])
        self.assertIsNone(r["lo95"])
        self.assertEqual(r["ci_method"], "too_few")
        self.assertIsNotNone(cap.mean_ci([float(i) for i in range(12)])["lo95"])


class TestAlphaLedger(unittest.TestCase):
    def test_every_opened_marker_is_in_the_ledger(self):
        exp = ROOT / "trader" / "runtime" / "experiments"
        markers = sorted(p.name for p in exp.glob("*_OPENED*")) if exp.exists() else []      # CAP_2026_OPENED has no .json
        if not markers:
            self.skipTest("no runtime markers in this checkout")
        ledger = json.loads((ROOT / "docs" / "audit" / "alpha_ledger.json").read_text(encoding="utf-8"))
        listed = {x["marker"] for x in ledger["looks"] if x.get("marker")}
        self.assertEqual([m for m in markers if m not in listed], [])
        for x in ledger["looks"]:
            self.assertTrue(0 < float(x["alpha"]) <= 0.05, x)


if __name__ == "__main__":
    unittest.main(verbosity=2)
