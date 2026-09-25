"""Poll heartbeat: dedup per closed bar, fail-safe, logging only.

Spec: docs/specs/features/poll-heartbeat-spec.md
"""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
import poll_heartbeat as HB  # noqa: E402


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "hb"
        self.p = [mock.patch.object(HB, "DIR", self.dir), mock.patch.object(HB, "_last", {}),
                  mock.patch.object(HB, "_pruned_day", None),
                  mock.patch.object(config, "POLL_HEARTBEAT_ENABLED", True, create=True)]
        for p in self.p:
            p.start()

    def tearDown(self):
        for p in self.p:
            p.stop()
        self.tmp.cleanup()

    def lines(self):
        out = []
        for f in sorted(self.dir.glob("*.jsonl")):
            out += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]
        return out


class TestRecord(_Base):
    def test_one_line_per_coin_per_closed_bar(self):
        bar = 1_760_000_000_000
        self.assertTrue(HB.record("AUSDT", "15m", "evaluated", bar_ts=bar, rules={"entry": False}))
        self.assertFalse(HB.record("AUSDT", "15m", "evaluated", bar_ts=bar, rules={"entry": False}))
        self.assertTrue(HB.record("BUSDT", "15m", "evaluated", bar_ts=bar, rules={"entry": True}))
        self.assertTrue(HB.record("AUSDT", "15m", "evaluated", bar_ts=bar + 900_000, rules={"entry": True}))
        self.assertEqual(len(self.lines()), 3)

    def test_fields_fired_and_failed_reasons(self):
        HB.record("AUSDT", "15m", "evaluated", bar_ts=1_760_000_000_000,
                  rules={"entry": True, "retest": False, "ema_cross": None},
                  reasons={"retest": "x" * 200, "entry": ""}, price=1.5, hour_blocked=False)
        row = self.lines()[0]
        self.assertEqual(row["stage"], "evaluated")
        self.assertEqual(row["fired"], ["entry"])
        self.assertEqual(set(row["reasons"]), {"retest"})
        self.assertEqual(len(row["reasons"]["retest"]), HB.REASON_MAX)
        self.assertEqual(row["bar_utc"], "2025-10-09T08:53")
        self.assertIs(row["hour_blocked"], False)

    def test_disabled_writes_nothing(self):
        with mock.patch.object(config, "POLL_HEARTBEAT_ENABLED", False):
            self.assertFalse(HB.record("AUSDT", "15m", "cooldown", bar_ts=1))
        self.assertFalse(self.dir.exists())

    def test_failure_never_raises(self):
        self.dir.parent.mkdir(parents=True, exist_ok=True)
        self.dir.write_text("not a directory")
        self.assertFalse(HB.record("AUSDT", "15m", "no_data"))

    def test_no_bar_uses_the_wall_clock_bucket(self):
        with mock.patch.object(HB.time, "time", return_value=1_760_000_123.0):
            HB.record("AUSDT", "15m", "no_data")
            self.assertFalse(HB.record("AUSDT", "15m", "no_data"))
        self.assertEqual(self.lines()[0]["bar_ts"] % 900_000, 0)

    def test_old_days_are_pruned(self):
        self.dir.mkdir(parents=True)
        (self.dir / "2020-01-01.jsonl").write_text("{}\n")
        HB.record("AUSDT", "15m", "no_data")
        self.assertFalse((self.dir / "2020-01-01.jsonl").exists())


class TestWiring(unittest.TestCase):
    def setUp(self):
        src = (HERE / "monitor.py").read_text(encoding="utf-8")
        start = src.index("async def _poll_coin(")
        self.body = src[start:src.index("\n        any_signal = entry_ok", start)]
        self.src = src

    def test_every_silent_return_before_rule_evaluation_is_recorded(self):
        lines = self.body.splitlines()
        for n, l in enumerate(lines):
            if l.strip().startswith("return"):
                prev = "\n".join(lines[max(0, n - 3):n])
                self.assertIn("poll_heartbeat.record(", prev, "unrecorded return: " + l.strip())

    def test_all_stages_are_wired(self):
        for stage in HB.STAGES:
            self.assertIn('"%s"' % stage, self.body, stage)

    def test_logging_only_nothing_reads_it(self):
        uses = set(re.findall(r"poll_heartbeat\.(\w+)", self.src))
        self.assertEqual(uses, {"record", "enabled"})

    def test_flag_and_rollback_documented(self):
        self.assertIsInstance(config.POLL_HEARTBEAT_ENABLED, bool)
        self.assertGreater(config.POLL_HEARTBEAT_KEEP_DAYS, 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
