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


class TestRealertReference(unittest.TestCase):
    """QNT 2026-09-28: exit 344.48 (+50.7%), restart during the cooldown reset the
    alert reference to 226.49 and the channel got "+7.3% after our exit" at 243."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = mock.patch.object(monitor, "_COOLDOWN_REFS_FILE", Path(self.tmp.name) / "refs.json")
        self.p.start()

    def tearDown(self):
        self.p.stop()
        self.tmp.cleanup()

    def test_restart_keeps_the_reference_of_the_same_cooldown(self):
        st = monitor.MonitorState()
        self.assertTrue(monitor._begin_or_resume_cooldown_ref(st, "QNTUSDT", NOW, 344.48))
        st.cooldown_realerted["QNTUSDT"] = True
        monitor._save_cooldown_refs(st)
        fresh = monitor.MonitorState()                      # the process after a restart
        self.assertFalse(monitor._begin_or_resume_cooldown_ref(fresh, "QNTUSDT", NOW, 226.49))
        self.assertEqual(fresh.cooldown_exit_px["QNTUSDT"], 344.48)
        self.assertTrue(fresh.cooldown_realerted["QNTUSDT"])

    def test_a_new_cooldown_takes_a_new_reference(self):
        st = monitor.MonitorState()
        monitor._begin_or_resume_cooldown_ref(st, "QNTUSDT", NOW, 344.48)
        st.cooldown_realerted["QNTUSDT"] = True
        self.assertTrue(monitor._begin_or_resume_cooldown_ref(st, "QNTUSDT", NOW + 86_400_000, 250.0))
        self.assertEqual(st.cooldown_exit_px["QNTUSDT"], 250.0)
        self.assertFalse(st.cooldown_realerted["QNTUSDT"])

    def test_first_cooldown_poll_uses_it_and_the_alert_is_markdown(self):
        src = (HERE / "monitor.py").read_text(encoding="utf-8")
        self.assertIn("_begin_or_resume_cooldown_ref(state, sym, cooldown_until_ms, float(c[i]))", src)
        self.assertNotIn("<b>{sym}</b> продолжает движение", src)
        self.assertIn("*{sym}* продолжает движение", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
