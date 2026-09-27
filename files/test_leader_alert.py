"""Leader-of-the-day info alert. Spec: docs/specs/features/leader-alert-spec.md"""
import asyncio
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import botlog  # noqa: E402
import config  # noqa: E402
import leader_alert as LA  # noqa: E402

DAY_MS = int(datetime(2026, 9, 27, tzinfo=timezone.utc).timestamp() * 1000)
BAR = 15 * 60 * 1000


def make_day(n_coins=25, n_bars=10, leaders=None):
    """Flat coins; `leaders` maps sym -> list of returns per bar (len n_bars)."""
    day = {}
    for k in range(n_coins):
        day["C%02dUSDT" % k] = (1.0, {DAY_MS + b * BAR: 1.0 + 0.001 * k for b in range(n_bars)})
    for s, rets in (leaders or {}).items():
        day[s] = (1.0, {DAY_MS + b * BAR: 1.0 + r for b, r in enumerate(rets)})
    return day


class TestRule(unittest.TestCase):
    def test_held_lead_for_eight_bars_alerts(self):
        day = make_day(leaders={"AUSDT": [0.02, 0.05] + [0.10] * 8})
        out = LA.find_leaders(day, 3, 0.075, 8)
        self.assertEqual([x["sym"] for x in out], ["AUSDT"])
        self.assertEqual(out[0]["rank"], 1)
        self.assertAlmostEqual(out[0]["ret"], 0.10)

    def test_seven_bars_is_not_enough(self):
        day = make_day(leaders={"AUSDT": [0.02, 0.05, 0.05] + [0.10] * 7})
        self.assertEqual(LA.find_leaders(day, 3, 0.075, 8), [])

    def test_rank_four_is_not_a_leader(self):
        lead = {s: [0.2 + i * 0.01] * 10 for i, s in enumerate(("XUSDT", "YUSDT", "ZUSDT"))}
        lead["AUSDT"] = [0.10] * 10
        out = LA.find_leaders(make_day(leaders=lead), 3, 0.075, 8)
        self.assertNotIn("AUSDT", [x["sym"] for x in out])
        self.assertEqual(len(out), 3)

    def test_below_min_return_is_not_a_leader(self):
        self.assertEqual(LA.find_leaders(make_day(leaders={"AUSDT": [0.07] * 10}), 3, 0.075, 8), [])

    def test_too_early_in_the_day(self):
        self.assertEqual(LA.find_leaders(make_day(n_bars=7, leaders={"AUSDT": [0.10] * 7}), 3, 0.075, 8), [])


class TestRunOnce(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = [mock.patch.object(LA, "STATE_FILE", Path(self.tmp.name) / "la.json"),
                  mock.patch.object(botlog, "LOG_FILE", Path(self.tmp.name) / "ev.jsonl"),
                  mock.patch.object(config, "LEADER_ALERT_ENABLED", True, create=True)]
        for p in self.p:
            p.start()
        self.sent = []

    def tearDown(self):
        for p in self.p:
            p.stop()
        self.tmp.cleanup()

    async def _send(self, text):
        self.sent.append(text)

    def run_with(self, day, now=datetime(2026, 9, 27, 3, 0, tzinfo=timezone.utc)):
        async def fake(session, sym, day_ms, now_ms, sem):
            return day.get(sym)
        with mock.patch.object(LA, "_fetch_today", fake):
            return asyncio.run(LA.run_once(None, self._send, sorted(day), now=now))

    def test_alerts_once_and_logs(self):
        day = make_day(leaders={"AUSDT": [0.10] * 10})
        self.assertEqual(len(self.run_with(day)), 1)
        self.assertEqual(len(self.run_with(day)), 0)          # same coin, same day: no repeat
        self.assertEqual(len(self.sent), 1)
        ev = [json.loads(x) for x in botlog.LOG_FILE.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(ev[0]["event"], "leader_alert")
        self.assertEqual(ev[0]["sym"], "AUSDT")

    def test_max_three_a_day(self):
        lead = {s: [0.3 - i * 0.01] * 10 for i, s in enumerate(("AUSDT", "BUSDT", "DUSDT"))}
        self.assertEqual(len(self.run_with(make_day(leaders=lead))), 3)
        more = {s: [0.5] * 10 for s in ("EUSDT",)}
        self.assertEqual(len(self.run_with(make_day(leaders=more))), 0)

    def test_new_day_resets(self):
        self.run_with(make_day(leaders={"AUSDT": [0.10] * 10}))
        nxt = datetime(2026, 9, 28, 3, 0, tzinfo=timezone.utc)
        self.assertEqual(len(self.run_with(make_day(leaders={"AUSDT": [0.10] * 10}), now=nxt)), 1)

    def test_disabled(self):
        with mock.patch.object(config, "LEADER_ALERT_ENABLED", False):
            self.assertEqual(self.run_with(make_day(leaders={"AUSDT": [0.10] * 10})), [])
        self.assertEqual(self.sent, [])

    def test_too_few_coins_skips(self):
        self.assertEqual(self.run_with(make_day(n_coins=5, leaders={"AUSDT": [0.10] * 10})), [])

    def test_message_is_info_not_signal_and_markdown_balanced(self):
        self.run_with(make_day(leaders={"AUSDT": [0.10] * 10}))
        m = self.sent[0]
        self.assertIn("не сигнал на покупку", m)
        self.assertIn("AUSDT", m)
        self.assertEqual(m.count("*") % 2, 0)
        self.assertNotIn("_", m)


class TestWiring(unittest.TestCase):
    def test_info_only(self):
        src = (HERE / "leader_alert.py").read_text(encoding="utf-8")
        for forbidden in ("positions", "open_position", "log_entry", "Position("):
            self.assertNotIn(forbidden, src)

    def test_monitor_runs_it_in_background_per_bar(self):
        src = (HERE / "monitor.py").read_text(encoding="utf-8")
        self.assertIn("import leader_alert", src)
        self.assertIn("asyncio.create_task(_run_leader_alert())", src)
        self.assertIn('state.__dict__.get("leader_alert_bar") != _la_bar', src)

    def test_flag_and_params(self):
        self.assertIsInstance(config.LEADER_ALERT_ENABLED, bool)
        self.assertEqual((config.LEADER_ALERT_RANK_MAX, config.LEADER_ALERT_MIN_RET,
                          config.LEADER_ALERT_HOLD_BARS, config.LEADER_ALERT_MAX_PER_DAY), (3, 0.075, 8, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
