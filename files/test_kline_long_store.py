"""Guards for the long kline store extension.

The 15m long file stopped at 2026-08-20 while every history reader kept using
it; the peak training label resolved 0% of September's 15m rows and nothing
said so. These tests pin the append so the store cannot silently freeze again.

Spec: docs/specs/features/kline-long-store-spec.md
"""
from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _backfill_klines_history as B  # noqa: E402

HDR = "ts,open,high,low,close,volume"


def line(ts, px):
    return "%s,%s,%s,%s,%s,1" % (ts, px, px, px, px)


class TestExtendLongStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.prev = B.HISTORY_DIR
        B.HISTORY_DIR = Path(self.tmp.name)

    def tearDown(self):
        B.HISTORY_DIR = self.prev
        self.tmp.cleanup()

    def write(self, name, rows):
        with io.open(B.HISTORY_DIR / name, "w", encoding="utf-8", newline=chr(10)) as f:
            f.write(HDR + chr(10) + chr(10).join(rows) + chr(10))

    def read(self, name):
        return io.open(B.HISTORY_DIR / name, encoding="utf-8").read().splitlines()

    def ts(self, minute):
        return "2026-08-20T%02d:%02d:00+00:00" % (minute // 60, minute % 60)

    def test_appends_only_bars_after_the_long_files_end(self):
        self.write("XUSDT_15m_419d.csv", [line(self.ts(m), 1) for m in (0, 15, 30)])
        self.write("XUSDT_15m.csv", [line(self.ts(m), 2) for m in (15, 30, 45, 60)])
        n, st = B.extend_long_store("XUSDT", "15m")
        self.assertEqual((n, st), (2, "ok"))
        out = self.read("XUSDT_15m_419d.csv")
        self.assertEqual(len(out), 1 + 5)
        # the overlapping bars keep the LONG file's values: append-only means
        # results computed on the long file stay reproducible
        self.assertTrue(out[2].endswith(",1,1,1,1,1"))
        self.assertTrue(out[-1].startswith(self.ts(60)))

    def test_is_idempotent(self):
        self.write("XUSDT_15m_419d.csv", [line(self.ts(m), 1) for m in (0, 15)])
        self.write("XUSDT_15m.csv", [line(self.ts(m), 2) for m in (15, 30)])
        B.extend_long_store("XUSDT", "15m")
        self.assertEqual(B.extend_long_store("XUSDT", "15m"), (0, "current"))
        self.assertEqual(len(self.read("XUSDT_15m_419d.csv")), 1 + 3)

    def test_a_gap_is_reported_not_hidden(self):
        # the rolling window moved past the long file's end: bars in between
        # exist in neither file and must be re-fetched, so say so
        self.write("XUSDT_15m_419d.csv", [line(self.ts(0), 1)])
        self.write("XUSDT_15m.csv", [line(self.ts(120), 2)])
        n, st = B.extend_long_store("XUSDT", "15m")
        self.assertEqual(n, 1)
        self.assertTrue(st.startswith("GAP"), st)

    def test_1h_uses_its_own_long_file_and_step(self):
        self.write("XUSDT_1h_365d.csv", [line("2026-08-20T00:00:00+00:00", 1)])
        self.write("XUSDT_1h.csv", [line("2026-08-20T01:00:00+00:00", 2)])
        self.assertEqual(B.extend_long_store("XUSDT", "1h"), (1, "ok"))

    def test_missing_files_are_a_status_not_a_crash(self):
        self.assertEqual(B.extend_long_store("NOPEUSDT", "15m"), (0, "missing"))


class TestTheDailyTaskRunsIt(unittest.TestCase):
    def test_main_extends_after_the_fetch(self):
        src = (HERE / "_backfill_klines_history.py").read_text(encoding="utf-8")
        self.assertIn("if args.tf in LONG_SUFFIX:", src)
        self.assertIn("extend_all(syms, args.tf)", src)
        self.assertEqual(B.LONG_SUFFIX, {"15m": "15m_419d", "1h": "1h_365d"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
