"""Contracts for the incremental JSONL → SQLite event store.

The store is only worth having if resuming is exact. Every test here builds real
files and re-syncs them, because the failure modes worth catching are all about
byte offsets: double-counting an appended row, resuming into a rotated file,
storing half of a line a writer is still appending.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import event_store as ES  # noqa: E402


class TestEventStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.src = self.dir / "bot_events.jsonl"
        self.db = self.dir / "store.sqlite3"
        self._orig_lock = ES.LOCK_PATH
        ES.LOCK_PATH = self.dir / "store.lock"
        self.addCleanup(lambda: setattr(ES, "LOCK_PATH", self._orig_lock))

    def _write(self, rows: list[dict], mode: str = "w") -> None:
        with self.src.open(mode, encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    def _sync(self) -> dict:
        return ES.sync(("bot_events.jsonl",), db_path=self.db, files_dir=self.dir)["sources"][0]

    def _count(self) -> int:
        conn = ES._connect(self.db)
        try:
            return conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        finally:
            conn.close()

    @staticmethod
    def _ev(n: int, event: str = "blocked") -> dict:
        return {"event": event, "sym": f"S{n}USDT", "tf": "15m",
                "reason": "entry score 1 < floor 35",
                "ts": f"2026-08-{(n % 28) + 1:02d}T10:00:00Z"}

    def test_first_sync_ingests_everything(self):
        self._write([self._ev(i) for i in range(5)])
        res = self._sync()
        self.assertEqual(res["status"], "synced")
        self.assertEqual(res["new_rows"], 5)
        self.assertEqual(self._count(), 5)

    def test_second_sync_of_unchanged_file_does_nothing(self):
        self._write([self._ev(i) for i in range(5)])
        self._sync()
        res = self._sync()
        self.assertEqual(res["status"], "up_to_date")
        self.assertEqual(res["new_rows"], 0)
        self.assertEqual(self._count(), 5)

    def test_appended_rows_are_the_only_ones_re_read(self):
        self._write([self._ev(i) for i in range(5)])
        self._sync()
        self._write([self._ev(i) for i in range(5, 8)], mode="a")
        res = self._sync()
        self.assertEqual(res["new_rows"], 3, "resume must not re-read the old bytes")
        self.assertEqual(self._count(), 8)

    def test_truncation_resets_instead_of_splicing(self):
        # A rotated or truncated source is a different file. Continuing from the
        # old offset would staple two histories together.
        self._write([self._ev(i) for i in range(10)])
        self._sync()
        self._write([self._ev(99)])           # rewrite, now much shorter
        res = self._sync()
        self.assertTrue(res["reset"])
        self.assertEqual(self._count(), 1)

    def test_partial_trailing_line_is_left_for_next_sync(self):
        # A writer mid-append leaves a line without its newline; storing it
        # would persist half a record and then skip the rest of it forever.
        self._write([self._ev(i) for i in range(3)])
        with self.src.open("a", encoding="utf-8") as fh:
            fh.write('{"event": "blocked", "sym": "PARTIA')
        res = self._sync()
        self.assertEqual(res["new_rows"], 3)
        # complete the line; the next sync picks it up whole
        with self.src.open("a", encoding="utf-8") as fh:
            fh.write('L", "ts": "2026-08-13T10:00:00Z"}\n')
        res2 = self._sync()
        self.assertEqual(res2["new_rows"], 1)
        self.assertEqual(self._count(), 4)

    def test_malformed_lines_are_counted_not_fatal(self):
        self.src.write_text(
            json.dumps(self._ev(1)) + "\nnot json at all\n" + json.dumps(self._ev(2)) + "\n",
            encoding="utf-8")
        res = self._sync()
        self.assertEqual(res["new_rows"], 2)
        self.assertEqual(res["malformed"], 1)

    def test_reingesting_the_same_offset_cannot_double_count(self):
        self._write([self._ev(i) for i in range(4)])
        self._sync()
        # force a re-read of the whole file from zero
        conn = ES._connect(self.db)
        conn.execute("UPDATE source_state SET byte_offset=0")
        conn.commit()
        conn.close()
        self._sync()
        self.assertEqual(self._count(), 4, "primary key (source, offset) must dedupe")

    def test_missing_source_is_reported_not_raised(self):
        res = self._sync()
        self.assertEqual(res["status"], "missing")

    def test_indexed_columns_are_extracted_for_querying(self):
        self._write([self._ev(1, event="entry")])
        self._sync()
        conn = ES._connect(self.db)
        try:
            row = conn.execute(
                "SELECT day, event, sym, tf FROM events").fetchone()
        finally:
            conn.close()
        self.assertEqual(row, ("2026-08-02", "entry", "S1USDT", "15m"))

    def test_schema_version_bump_rebuilds(self):
        self._write([self._ev(i) for i in range(3)])
        self._sync()
        conn = ES._connect(self.db)
        conn.execute("PRAGMA user_version=999")
        conn.commit()
        conn.close()
        self._sync()
        self.assertEqual(self._count(), 3, "rebuild must re-ingest, not lose rows")


if __name__ == "__main__":
    unittest.main()
