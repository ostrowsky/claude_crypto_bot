"""Contracts for the backfill cross-process lock.

The lock used to be an empty `O_CREAT|O_EXCL` file with no owner and no
timestamp, so a killed backfill left it behind permanently and every later run
returned at INFO level. It was found on 2026-08-13 holding for 1389 hours
(58 days) with 11 894 rows still unlabelled — a learning input dead since
mid-June that no report mentioned.

These tests pin the two halves of the fix: a live holder is still respected, and
a dead or expired one is taken over loudly.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import backfill_critic_labels as B  # noqa: E402


class TestBackfillLock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._orig = B._LOCK_FILE
        B._LOCK_FILE = Path(self.tmp.name) / "backfill.lock"
        self.addCleanup(lambda: setattr(B, "_LOCK_FILE", self._orig))

    def _write_lock(self, pid: int, age_sec: float) -> None:
        B._LOCK_FILE.write_text(json.dumps({"pid": pid, "ts": time.time() - age_sec}),
                                encoding="utf-8")
        stamp = time.time() - age_sec
        os.utime(B._LOCK_FILE, (stamp, stamp))

    def test_free_lock_is_acquired_and_records_owner(self):
        self.assertTrue(B._acquire_lock())
        info = json.loads(B._LOCK_FILE.read_text(encoding="utf-8"))
        self.assertEqual(info["pid"], os.getpid())
        self.assertGreater(info["ts"], 0)

    def test_live_owner_is_respected(self):
        self._write_lock(pid=4242, age_sec=30)
        with patch.object(B, "_lock_owner_alive", return_value=True):
            self.assertFalse(B._acquire_lock())
        # untouched: the running backfill keeps its claim
        self.assertEqual(json.loads(B._LOCK_FILE.read_text(encoding="utf-8"))["pid"], 4242)

    def test_dead_owner_is_taken_over(self):
        self._write_lock(pid=4242, age_sec=30)
        with patch.object(B, "_lock_owner_alive", return_value=False):
            self.assertTrue(B._acquire_lock())
        self.assertEqual(json.loads(B._LOCK_FILE.read_text(encoding="utf-8"))["pid"],
                         os.getpid())

    def test_expired_lock_is_taken_over_even_if_owner_looks_alive(self):
        # PIDs get recycled; a lock older than the TTL cannot be a real run.
        self._write_lock(pid=4242, age_sec=B._LOCK_TTL_SEC + 60)
        with patch.object(B, "_lock_owner_alive", return_value=True):
            self.assertTrue(B._acquire_lock())

    def test_legacy_empty_lock_is_not_trusted_forever(self):
        # This is the exact artefact found in production: zero bytes, no owner.
        B._LOCK_FILE.write_text("", encoding="utf-8")
        stamp = time.time() - (58 * 24 * 3600)
        os.utime(B._LOCK_FILE, (stamp, stamp))
        self.assertTrue(B._acquire_lock())

    def test_unknown_liveness_does_not_displace_a_fresh_lock(self):
        # If the process check itself fails we must not evict a working run.
        self._write_lock(pid=4242, age_sec=10)
        with patch("subprocess.run", side_effect=OSError("no tasklist")):
            self.assertFalse(B._acquire_lock())


if __name__ == "__main__":
    unittest.main()
