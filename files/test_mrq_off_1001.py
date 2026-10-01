"""mode_range_quality OFF by operator decision, guard kept in shadow (mrq-off-1001-spec.md)."""
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import botlog
import config
import monitor as M
import readouts as RO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UTC = timezone.utc


class TestGuard(unittest.TestCase):
    def test_off_by_operator_decision(self):
        self.assertFalse(config.MODE_RANGE_QUALITY_GUARD_ENABLED)

    def test_shadow_still_computes_the_reason(self):
        kw = dict(mode="alignment", tf="15m", daily_range=3.0, slope=1.0)
        self.assertIsNone(M._mode_daily_range_guard_reason(**kw))
        self.assertIn("alignment/15m", M._mode_daily_range_guard_reason(ignore_flag=True, **kw))
        self.assertIsNone(M._mode_daily_range_guard_reason(ignore_flag=True, mode="alignment", tf="15m",
                                                           daily_range=6.0, slope=1.0))

    def test_live_path_logs_shadow_and_does_not_block(self):
        src = (HERE / "monitor.py").read_text(encoding="utf-8")
        i = src.index("ignore_flag=True,")
        block = src[i:i + 700]
        self.assertIn("log_mode_range_shadow", block)
        self.assertIn("mode_range_guard_reason = None", block)

    def test_shadow_event(self):
        got = []
        with mock.patch.object(botlog, "_write", got.append):
            botlog.log_mode_range_shadow("AUSDT", "15m", "trend", 1.0, "mode_range_quality: x")
        self.assertEqual(got[0]["event"], "mode_range_shadow")


class TestReadout(unittest.TestCase):
    def _run(self, n_added, added_pnl, other_pnl, days=1):
        t0 = datetime(2026, 10, 2, tzinfo=UTC)
        evs = []
        for k in range(n_added + 10):
            dt = t0 + timedelta(minutes=30 * k)
            sym = "S%d" % k
            added = k < n_added
            if added:
                evs.append({"event": "mode_range_shadow", "ts": (dt - timedelta(minutes=1)).isoformat(),
                            "_dt": dt - timedelta(minutes=1), "sym": sym, "tf": "15m"})
            evs.append({"event": "entry", "ts": dt.isoformat(), "_dt": dt, "sym": sym, "tf": "15m"})
            evs.append({"event": "exit", "ts": (dt + timedelta(hours=1)).isoformat(), "_dt": dt + timedelta(hours=1),
                        "sym": sym, "tf": "15m", "pnl_pct": added_pnl if added else other_pnl})
        import goal_validator as GV
        with mock.patch.object(RO, "scan_events", lambda since, kinds, path=None: evs), \
             mock.patch.object(GV, "winner_days", lambda since, full=None: []):
            return RO.readout_mrq_off({"window_from": "2026-10-02"},
                                      {"days": ["2026-10-0%d" % (2 + i) for i in range(days)], "now": t0})

    def test_counts_only_entries_the_gate_would_have_blocked(self):
        r = self._run(5, 1.0, 1.0)
        self.assertEqual(r["added_entries"], 5)
        self.assertEqual(r["other_entries"], 10)

    def test_much_worse_added_trades_suggest_rollback(self):
        self.assertEqual(self._run(30, -5.0, 1.0, days=2)["_verdict"], "ROLLBACK_SUGGESTED")

    def test_flood_is_an_anomaly(self):
        r = self._run(25, 0.5, 0.5)
        self.assertTrue(r["_anomaly"])

    def test_registered(self):
        r = [x for x in RO.load_registry() if x["id"] == "MRQ-OFF"][0]
        self.assertEqual(r["fn"], "readout_mrq_off")
        self.assertEqual(r["expected"]["added_messages_per_day"], 6.6)

    def test_documented(self):
        self.assertTrue((ROOT / "docs/specs/features/mrq-off-1001-spec.md").exists())
        for doc in ("CLAUDE.md", "PROJECT_CONTEXT.md"):
            self.assertIn("mrq-off-1001-spec.md", (ROOT / doc).read_text(encoding="utf-8"), doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
