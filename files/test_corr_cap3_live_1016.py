"""correlation_guard cap 3 live (corr-cap3-live-1016-spec.md), written from the spec first."""
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import botlog
import config
import correlation_guard as CG
import readouts as R

ROOT = Path(__file__).resolve().parent.parent
T = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def cache(pairs):
    c = CG.CorrelationCache()
    for a, b, r in pairs:
        c.set(a, b, r)
    return c


class TestGuard(unittest.TestCase):
    def test_config(self):
        self.assertEqual(config.CORR_MAX_PER_CLUSTER, 3)
        self.assertEqual(config.CORR_CAP_RELAXED_FROM, 2)

    def test_cluster_of_two_passes_three_blocks(self):
        with mock.patch.object(config, "CORR_GUARD_ENABLED", True):
            c2 = cache([("X", "A", 0.9), ("X", "B", 0.9)])
            r = CG.check_entry("X", {"A": 1, "B": 1}, c2)
            self.assertTrue(r.allowed)
            self.assertEqual(r.cluster_size, 2)
            c3 = cache([("X", "A", 0.9), ("X", "B", 0.9), ("X", "C", 0.9)])
            self.assertFalse(CG.check_entry("X", {"A": 1, "B": 1, "C": 1}, c3).allowed)


class TestLogging(unittest.TestCase):
    def test_events(self):
        got = []
        with mock.patch.object(botlog, "_write", side_effect=got.append):
            botlog.log_corr_cap_relaxed("INJUSDT", "15m", 7.424, 2, 0.737, "WLDUSDT")
            botlog.log_corr_prune("KSMUSDT", "corr_cluster_prune: cluster 3 size 4->3")
        self.assertEqual(got[0]["event"], "corr_cap_relaxed")
        for k in ("sym", "tf", "price", "cluster_size", "max_rho", "peer_sym"):
            self.assertIn(k, got[0])
        self.assertEqual((got[1]["event"], got[1]["sym"]), ("corr_prune", "KSMUSDT"))

    def test_hooks_wired(self):
        mon = (ROOT / "files" / "monitor.py").read_text(encoding="utf-8")
        self.assertIn("botlog.log_corr_cap_relaxed(", mon)
        self.assertIn("botlog.log_corr_prune(", mon)


def ev(kind, sym, t, **kw):
    return {"event": kind, "sym": sym, "tf": "15m", "_dt": t, "ts": t.strftime("%Y-%m-%dT%H:%M:%SZ"), **kw}


class TestReadout(unittest.TestCase):
    def test_relaxed_match_window(self):
        trades = [(ev("entry", "A", T), ev("exit", "A", T + timedelta(hours=2), pnl_pct=1.0)),
                  (ev("entry", "B", T), ev("exit", "B", T + timedelta(hours=2), pnl_pct=-1.0)),
                  (ev("entry", "C", T), ev("exit", "C", T + timedelta(hours=2), pnl_pct=0.5))]
        relaxed = [ev("corr_cap_relaxed", "A", T - timedelta(seconds=5)),
                   ev("corr_cap_relaxed", "B", T - timedelta(minutes=45)),     # too old
                   ev("corr_cap_relaxed", "C", T + timedelta(minutes=1))]      # after the entry
        self.assertEqual(R.cap3_relaxed_flags(trades, relaxed), [True, False, False])

    def test_tail_with_and_without(self):
        daily = {"d1": [(-5.0, True), (1.0, False)], "d2": [(-2.0, False)], "d3": [(3.0, True)]}
        dd_w, dd_wo, worst_w, worst_wo = R.cap3_tail(daily)
        # with: -4, -2, +3 -> cum -4, -6, -3 -> dd -6, worst -4 ; without: +1, -2, 0 -> cum 1, -1, -1 -> dd -2, worst -2
        self.assertEqual((dd_w, dd_wo, worst_w, worst_wo), (-6.0, -2.0, -4.0, -2.0))

    def test_verdict(self):
        tail_ok = dict(dd_with=-105, dd_without=-100, worst_with=-50, worst_without=-48)
        self.assertEqual(R.cap3_verdict(1, -0.05, 0.2, **tail_ok), "KEEP")
        self.assertEqual(R.cap3_verdict(0, -0.05, 0.2, **tail_ok), "INCONCLUSIVE")
        self.assertEqual(R.cap3_verdict(2, -0.5, -0.2, **tail_ok), "ROLLBACK_SUGGESTED")
        self.assertEqual(R.cap3_verdict(2, -0.05, 0.2, dd_with=-120, dd_without=-100, worst_with=-50,
                                        worst_without=-48), "ROLLBACK_SUGGESTED")

    def test_registry(self):
        r = {x["id"]: x for x in R.load_registry()}["CORR-CAP3"]
        self.assertEqual((r["fn"], r["window_from"], r["due"], r["min_days"], r["min_trades"]),
                         ("readout_corr_cap3", "2026-10-17", "2026-11-13", 21, 10))
        self.assertIn("readout_corr_cap3", R.FNS)

    def test_documented(self):
        for doc in ("CLAUDE.md", "PROJECT_CONTEXT.md"):
            self.assertIn("corr-cap3-live-1016-spec.md", (ROOT / doc).read_text(encoding="utf-8"), doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
