"""Exit gates for the global (daily-resolution) label tier.

The store gains records it can rank globally but cannot time intraday. The
property that matters: those two kinds never get confused for one another.

Spec: docs/specs/features/global-label-universe-spec.md
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import immutable_labels as IL  # noqa: E402
import label_store as LS  # noqa: E402

DAY_MS = 86_400_000


def day_ms(offset_days: int) -> int:
    """Start of a UTC day, `offset_days` before today."""
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0,
                                             microsecond=0)
    return int((now - timedelta(days=offset_days)).timestamp() * 1000)


def daily_bar(start_ms: int, o=100.0, h=110.0, low=99.0, c=108.0):
    return [start_ms, o, h, low, c, 1234.0]


class TestDailyRecord(unittest.TestCase):
    def test_a_closed_day_is_complete_from_one_bar(self):
        # MIN_BARS_COMPLETE counts HOURLY bars; applying it to a daily bar would
        # mark every global record incomplete and drop it from every consumer.
        rec = LS.build_day_record_daily("XUSDT", day_ms(3), daily_bar(day_ms(3)),
                                        provenance={"source": "test"})
        self.assertTrue(rec["complete"])
        self.assertEqual(rec["bars_used"], 1)

    def test_todays_unfinished_day_is_not_complete(self):
        rec = LS.build_day_record_daily("XUSDT", day_ms(0), daily_bar(day_ms(0)),
                                        provenance={"source": "test"})
        self.assertFalse(rec["complete"])

    def test_intraday_fields_are_absent_not_approximated(self):
        rec = LS.build_day_record_daily("XUSDT", day_ms(3), daily_bar(day_ms(3)),
                                        provenance={"source": "test"})
        self.assertEqual(rec["resolution"], "1d")
        self.assertIsNone(rec["anchor_ts"])
        self.assertIsNone(rec["early_deadline_ts"])

    def test_returns_come_from_open_and_close(self):
        rec = LS.build_day_record_daily("XUSDT", day_ms(3),
                                        daily_bar(day_ms(3), o=100.0, h=115.0,
                                                  c=110.0),
                                        provenance={"source": "test"})
        self.assertAlmostEqual(rec["eod_return_pct"], 10.0, places=4)
        self.assertAlmostEqual(rec["max_move_pct"], 15.0, places=4)
        self.assertTrue(rec["qualifies_move5"])

    def test_hourly_records_read_as_hourly_without_carrying_the_field(self):
        # `_identity` hashes every field plus builder_version, so ADDING
        # `resolution` to the hourly builder would make a rebuild of any of the
        # 19 502 written records raise ImmutableLabelError. The field is read
        # through a normaliser instead of backfilled into immutable records.
        bars = [[day_ms(3) + i * 3_600_000, 100.0, 101.0 + i, 99.0, 100.5 + i, 1.0]
                for i in range(24)]
        rec = LS.build_day_record("XUSDT", day_ms(3), bars,
                                  provenance={"source": "test"})
        self.assertNotIn("resolution", rec)
        self.assertEqual(LS.resolution_of(rec), "1h")
        self.assertEqual(LS.resolution_of({}), "1h")

    def test_hourly_builder_version_is_unchanged(self):
        # Bumping it silently invalidates every stored record's identity.
        self.assertEqual(LS.BUILDER_VERSION, "label-store-v1")
        self.assertNotEqual(LS.DAILY_BUILDER_VERSION, LS.BUILDER_VERSION)

    def test_a_rebuilt_hourly_record_still_matches_a_stored_one(self):
        bars = [[day_ms(3) + i * 3_600_000, 100.0, 101.0, 99.0, 100.5, 1.0]
                for i in range(24)]
        prov = {"source": "test", "source_sha256": "abc"}
        a = LS.build_day_record("XUSDT", day_ms(3), bars, provenance=prov)
        b = LS.build_day_record("XUSDT", day_ms(3), bars, provenance=prov)
        self.assertEqual(LS.LabelStore._identity(a), LS.LabelStore._identity(b))


class TestMoveEventPathRefusesDailyRecords(unittest.TestCase):
    def test_daily_record_cannot_enter_the_moveevent_path(self):
        # weekly_steering filters to the watchlist today, which happens to
        # exclude these — an accident of the filter, not an invariant. A None
        # deadline would silently classify every alert as early or late.
        import weekly_steering as WS
        rec = LS.build_day_record_daily("XUSDT", day_ms(3), daily_bar(day_ms(3)),
                                        provenance={"source": "test"})
        self.assertFalse(WS.is_move_event_source(rec))

    def test_hourly_record_is_accepted(self):
        import weekly_steering as WS
        bars = [[day_ms(3) + i * 3_600_000, 100.0, 106.0, 99.0, 105.0, 1.0]
                for i in range(24)]
        rec = LS.build_day_record("XUSDT", day_ms(3), bars,
                                  provenance={"source": "test"})
        self.assertTrue(WS.is_move_event_source(rec))


class _Store(LS.LabelStore):
    def __init__(self, records):
        self._records = records

    def records(self):
        return list(self._records)


class TestTierSeparationReturns(unittest.TestCase):
    def test_global_universe_restores_distinct_tiers(self):
        # The watchlist-scoped store made top20 and top50 byte-identical: inside
        # 95 symbols the +5% floor binds long before the rank does. Over a
        # global universe the rank binds again.
        rows = []
        # 60 movers above the floor, descending, plus 300 flat symbols.
        for i in range(60):
            rows.append({"symbol": f"M{i:03d}", "utc_day": "2026-05-01",
                         "eod_return_pct": 40.0 - i * 0.5, "complete": True})
        for i in range(300):
            rows.append({"symbol": f"F{i:03d}", "utc_day": "2026-05-01",
                         "eod_return_pct": 0.1, "complete": True})
        days = ["2026-05-01"] * len(rows)
        syms = [r["symbol"] for r in rows]
        _, lab, _ = IL.tier_labels(days, syms, tiers=(20, 50), floor=5.0,
                                   store=_Store(rows))
        self.assertEqual(sum(lab["top20"]), 20)
        self.assertEqual(sum(lab["top50"]), 50)
        self.assertNotEqual(lab["top20"], lab["top50"])


class TestBuildReportsWhatItCouldNotResolve(unittest.TestCase):
    def test_summary_counts_unresolved_symbols(self):
        # TH-05: a universe silently missing delisted pairs looks like a quiet
        # market rather than missing data.
        res = LS.summarise_universe_build(resolved=["A", "B"], failed=["C"])
        self.assertEqual(res["symbols_resolved"], 2)
        self.assertEqual(res["symbols_failed"], 1)
        self.assertIn("C", res["failed_symbols"])
        self.assertIn("survivorship", res["caveat"].lower())


if __name__ == "__main__":
    unittest.main()
