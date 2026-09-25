"""Guards for the intraday label tier (2026-09-25).

The move-relative North Star needs the hour a coin crossed +2.5%. Hourly labels
stopped on 2026-08-12, the daily tier has no time, and the daily records are
immutable -- so the timing lives in its own immutable tier.

Spec: docs/specs/features/intraday-label-tier-spec.md
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import label_store as LS  # noqa: E402

DAY0 = datetime(2026, 9, 20, tzinfo=timezone.utc)


def write_1h(dirpath: Path, sym: str, hours: int, jump_at: int | None = None, name_suffix="_1h_365d"):
    lines = ["ts,open,high,low,close,volume"]
    for h in range(hours):
        ts = DAY0 + timedelta(hours=h)
        hi = 103.0 if jump_at is not None and h >= jump_at else 100.5
        lines.append("%s,100,%s,99.5,100,1" % (ts.isoformat(), hi))
    (dirpath / (sym + name_suffix + ".csv")).write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.hist, self.store = d / "history", d / "labels"
        self.hist.mkdir()
        self.now = int((DAY0 + timedelta(days=2, hours=1)).timestamp() * 1000)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, syms=("XUSDT",)):
        return LS.build_intraday_from_store(list(syms), since_day="2026-09-01", store_root=self.store,
                                            history=self.hist, now_ms=self.now)

    def test_a_closed_day_gets_an_hourly_record_with_its_crossing_time(self):
        write_1h(self.hist, "XUSDT", 24, jump_at=7)
        r = self.build()
        self.assertEqual(r["written"], 1)
        rec = LS.LabelStore(self.store, filename=LS.INTRADAY_FILE).records()[0]
        self.assertEqual(rec["resolution"], "1h")
        self.assertEqual(rec["early_deadline_ts"], int((DAY0 + timedelta(hours=7)).timestamp() * 1000))
        self.assertEqual(rec["provenance"]["builder_version"], LS.INTRADAY_BUILDER_VERSION)

    def test_rebuild_is_idempotent(self):
        write_1h(self.hist, "XUSDT", 24, jump_at=7)
        self.build()
        r = self.build()
        self.assertEqual((r["written"], r["already_present"], r["conflicts"]), (0, 1, 0))

    def test_a_partial_day_is_skipped_not_labelled(self):
        write_1h(self.hist, "XUSDT", 12, jump_at=7)
        r = self.build()
        self.assertEqual((r["written"], r["skipped_incomplete"]), (0, 1))

    def test_the_forming_day_is_never_written(self):
        self.now = int((DAY0 + timedelta(hours=23)).timestamp() * 1000)
        write_1h(self.hist, "XUSDT", 23, jump_at=7)
        r = self.build()
        self.assertEqual((r["written"], r["skipped_still_forming"]), (0, 1))

    def test_the_main_store_is_untouched(self):
        write_1h(self.hist, "XUSDT", 24, jump_at=7)
        self.build()
        self.assertEqual(LS.LabelStore(self.store).records(), [])

    def test_rolling_file_wins_on_overlap(self):
        write_1h(self.hist, "XUSDT", 24, jump_at=None)                   # long: never crosses
        write_1h(self.hist, "XUSDT", 24, jump_at=5, name_suffix="_1h")   # rolling: crosses at 05
        self.build()
        rec = LS.LabelStore(self.store, filename=LS.INTRADAY_FILE).records()[0]
        self.assertEqual(rec["early_deadline_ts"], int((DAY0 + timedelta(hours=5)).timestamp() * 1000))

    def test_deadlines_read_both_tiers_and_skip_days_without_a_crossing(self):
        write_1h(self.hist, "XUSDT", 24, jump_at=7)
        write_1h(self.hist, "YUSDT", 24, jump_at=None)
        self.build(("XUSDT", "YUSDT"))
        dl = LS.intraday_deadlines(self.store)
        self.assertEqual(set(dl), {("2026-09-20", "XUSDT")})


class TestWiring(unittest.TestCase):
    def test_the_north_star_reads_the_tier(self):
        src = (HERE / "_compute_early_capture.py").read_text(encoding="utf-8")
        self.assertIn("deadlines = LS.intraday_deadlines()", src)

    def test_the_nightly_cycle_extends_it(self):
        src = (HERE / "daily_learning.py").read_text(encoding="utf-8")
        self.assertIn("LS.build_intraday_from_store(wl, since_day=since)", src)

    def test_its_freshness_is_declared(self):
        import artifact_freshness as AF
        self.assertIn("label_store_intraday", {a.name for a in AF.ARTIFACTS})


if __name__ == "__main__":
    unittest.main(verbosity=2)
