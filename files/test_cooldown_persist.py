"""Post-exit cooldowns survive a restart (monitor.PersistentCooldowns / load_cooldowns).

HBAR 2026-09-28: 12 cooldown bars left at 09:45 UTC, evaluated as free right after
the 10:09 restart -- state.cooldowns was memory-only.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
import monitor  # noqa: E402

NOW = 1_790_600_000_000


class TestPersistentCooldowns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "cooldowns.json"
        self.flag = mock.patch.object(config, "COOLDOWN_PERSIST_ENABLED", True, create=True)
        self.flag.start()

    def tearDown(self):
        self.flag.stop()
        self.tmp.cleanup()

    def test_write_on_set_and_restore_after_restart(self):
        cd = monitor.load_cooldowns(self.path, now_ms=NOW)
        cd["HBARUSDT"] = NOW + 12 * 900_000
        self.assertEqual(json.loads(self.path.read_text())["HBARUSDT"], NOW + 12 * 900_000)
        again = monitor.load_cooldowns(self.path, now_ms=NOW + 900_000)       # "restart" one bar later
        self.assertEqual(again.get("HBARUSDT", 0), NOW + 12 * 900_000)

    def test_expired_entries_are_dropped(self):
        self.path.write_text(json.dumps({"OLDUSDT": NOW - 1, "LIVEUSDT": NOW + 5}))
        cd = monitor.load_cooldowns(self.path, now_ms=NOW)
        self.assertEqual(dict(cd), {"LIVEUSDT": NOW + 5})
        self.assertEqual(json.loads(self.path.read_text()), {"LIVEUSDT": NOW + 5})

    def test_delete_and_pop_persist(self):
        cd = monitor.load_cooldowns(self.path, now_ms=NOW)
        cd["A"] = NOW + 10
        cd["B"] = NOW + 10
        del cd["A"]
        cd.pop("B", None)
        self.assertEqual(json.loads(self.path.read_text()), {})

    def test_corrupt_file_starts_empty(self):
        self.path.write_text("{not json")
        self.assertEqual(dict(monitor.load_cooldowns(self.path, now_ms=NOW)), {})

    def test_flag_off_is_a_plain_memory_dict(self):
        with mock.patch.object(config, "COOLDOWN_PERSIST_ENABLED", False):
            cd = monitor.load_cooldowns(self.path, now_ms=NOW)
        cd["A"] = 1
        self.assertIs(type(cd), dict)
        self.assertFalse(self.path.exists())

    def test_the_monitor_reads_it_like_a_dict(self):
        cd = monitor.load_cooldowns(self.path, now_ms=NOW)
        cd["A"] = NOW + 900_000
        self.assertEqual(cd.get("A", 0), NOW + 900_000)
        self.assertEqual(cd.get("B", 0), 0)


class TestWiring(unittest.TestCase):
    def test_bot_restores_cooldowns_at_startup(self):
        src = (HERE / "bot.py").read_text(encoding="utf-8")
        self.assertIn("state.cooldowns = load_cooldowns()", src)
        self.assertLess(src.index("state.positions = load_positions()"), src.index("state.cooldowns = load_cooldowns()"))

    def test_flag(self):
        self.assertIs(config.COOLDOWN_PERSIST_ENABLED, True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
