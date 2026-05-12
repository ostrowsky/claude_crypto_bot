"""Tests for bot_ui_cache — the snapshot cache that makes Telegram menu
render instant regardless of disk/event-loop state.

Run:
    pyembed\\python.exe files\\test_bot_ui_cache.py
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bot_ui_cache as C


class BasicTests(unittest.TestCase):

    def setUp(self):
        C._reset_for_tests()

    def test_empty_get_returns_none(self):
        self.assertIsNone(C.get())

    def test_refresh_publishes(self):
        snap = C.refresh(wl_count=105, hot_count=12, pos_count=5, running=True,
                         now=1000.0)
        self.assertEqual(snap.wl_count, 105)
        self.assertEqual(snap.hot_count, 12)
        self.assertEqual(snap.pos_count, 5)
        self.assertTrue(snap.running)
        self.assertEqual(snap.captured_at, 1000.0)
        # And get() returns the same instance
        self.assertIs(C.get(), snap)

    def test_get_or_default_uses_default_when_empty(self):
        d = C.UISnapshot(captured_at=0.0, wl_count=0, hot_count=0,
                         pos_count=0, running=False)
        self.assertIs(C.get_or_default(d), d)

    def test_get_or_default_uses_cache_when_available(self):
        C.refresh(wl_count=1, hot_count=2, pos_count=3, running=True, now=1.0)
        d = C.UISnapshot(captured_at=0.0, wl_count=99, hot_count=99,
                         pos_count=99, running=False)
        self.assertEqual(C.get_or_default(d).wl_count, 1)


class AgeAndStaleTests(unittest.TestCase):

    def setUp(self):
        C._reset_for_tests()

    def test_age_is_inf_when_empty(self):
        self.assertEqual(C.age_seconds(now=100.0), float("inf"))

    def test_age_computed_correctly(self):
        C.refresh(wl_count=1, hot_count=1, pos_count=1, running=False, now=100.0)
        self.assertAlmostEqual(C.age_seconds(now=105.5), 5.5, places=3)

    def test_is_stale_true_when_empty(self):
        self.assertTrue(C.is_stale(ttl_sec=5.0))

    def test_is_stale_false_when_fresh(self):
        now = time.time()
        C.refresh(wl_count=1, hot_count=1, pos_count=1, running=False, now=now)
        self.assertFalse(C.is_stale(ttl_sec=5.0, now=now + 1.0))

    def test_is_stale_true_when_expired(self):
        now = time.time()
        C.refresh(wl_count=1, hot_count=1, pos_count=1, running=False, now=now)
        self.assertTrue(C.is_stale(ttl_sec=5.0, now=now + 6.0))


class ImmutabilityTests(unittest.TestCase):
    """Snapshots must be safe to publish across threads — readers should
    never see torn writes."""

    def setUp(self):
        C._reset_for_tests()

    def test_dataclass_is_frozen(self):
        snap = C.refresh(wl_count=1, hot_count=1, pos_count=1, running=False, now=0.0)
        with self.assertRaises(Exception):
            snap.wl_count = 999          # frozen → AttributeError / FrozenInstanceError

    def test_concurrent_publishers_dont_corrupt(self):
        """Hammer publish() from many threads; reads must always be valid."""
        stop = threading.Event()
        errors: list[Exception] = []

        def reader() -> None:
            while not stop.is_set():
                try:
                    s = C.get()
                    if s is not None:
                        # Touch all fields; if torn, this raises or returns wrong type
                        _ = (s.wl_count, s.hot_count, s.pos_count, s.running, s.captured_at)
                        assert isinstance(s.wl_count, int)
                except Exception as e:
                    errors.append(e)

        def writer(seed: int) -> None:
            for i in range(200):
                C.refresh(wl_count=seed * 1000 + i, hot_count=i,
                          pos_count=i, running=bool(i & 1), now=float(i))

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        writer_threads = [threading.Thread(target=writer, args=(s,)) for s in range(4)]
        for t in writer_threads:
            t.start()
        for t in writer_threads:
            t.join()
        stop.set()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"reader errors: {errors[:3]}")


class HotPathPerformanceTests(unittest.TestCase):
    """The whole point of this cache is sub-millisecond hot reads. Pin
    that contract in a test so a future change that adds work to get()
    is caught."""

    def setUp(self):
        C._reset_for_tests()

    def test_get_under_1ms_for_1000_calls(self):
        C.refresh(wl_count=105, hot_count=12, pos_count=5, running=True, now=0.0)
        t0 = time.perf_counter()
        for _ in range(1000):
            C.get()
        elapsed = time.perf_counter() - t0
        # 1000 lock+read in well under 100ms even on slow Windows VMs
        self.assertLess(elapsed, 0.1,
                        f"hot read too slow: {elapsed*1000:.1f}ms for 1000 calls")


if __name__ == "__main__":
    unittest.main(verbosity=2)
