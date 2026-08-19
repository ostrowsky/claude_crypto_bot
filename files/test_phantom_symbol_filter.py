"""Exit gates for the phantom-symbol filter.

A delisted pair keeps returning a 24h ticker with non-zero volume, so it arrives
with a complete, plausible feature row. EOSUSDT carried
`tg_return_since_open = 6.79` while its last candle was May 2025, and the model
ranked it first. These pin the predicate that stops it.

Spec: docs/specs/features/phantom-symbol-filter-spec.md
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import label_store as LS  # noqa: E402
import phantom_filter as PF  # noqa: E402


def day(offset: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=offset)).strftime("%Y-%m-%d")


class _Store(LS.LabelStore):
    def __init__(self, records):
        self._records = records

    def records(self):
        return list(self._records)


def rec(symbol, d, complete=True):
    return {"symbol": symbol, "utc_day": d, "eod_return_pct": 1.0,
            "complete": complete}


class TestLiveness(unittest.TestCase):
    def test_recent_label_is_live(self):
        store = _Store([rec("AUSDT", day(1))])
        self.assertTrue(PF.is_live("AUSDT", store=store))

    def test_stale_label_is_not_live(self):
        store = _Store([rec("DEADUSDT", day(400))])
        self.assertFalse(PF.is_live("DEADUSDT", store=store))

    def test_absent_and_stale_are_different_facts(self):
        store = _Store([rec("DEADUSDT", day(400))])
        self.assertEqual(PF.liveness("DEADUSDT", store=store), "stale")
        self.assertEqual(PF.liveness("NEVERUSDT", store=store), "unknown")
        # Both fail is_live, but a newly listed pair and a delisted one are not
        # the same problem and must not be reported as one number.
        self.assertFalse(PF.is_live("NEVERUSDT", store=store))

    def test_incomplete_labels_do_not_count_as_recent(self):
        store = _Store([rec("XUSDT", day(1), complete=False)])
        self.assertFalse(PF.is_live("XUSDT", store=store))

    def test_the_newest_label_decides_not_the_oldest(self):
        store = _Store([rec("XUSDT", day(400)), rec("XUSDT", day(2))])
        self.assertTrue(PF.is_live("XUSDT", store=store))


class TestFiltering(unittest.TestCase):
    def test_filter_reports_what_it_dropped(self):
        # Ten live names, so dropping two is 17% and stays under the
        # stand-down threshold: the original three-symbol fixture dropped 67%
        # and tripped the guard, which is the guard working, not a bug.
        live = [f"OK{i}USDT" for i in range(10)]
        store = _Store([rec(s, day(1)) for s in live]
                       + [rec("DEADUSDT", day(400))])
        kept, dropped = PF.filter_live(live + ["DEADUSDT", "NEVERUSDT"],
                                       store=store)
        self.assertEqual(kept, live)
        self.assertEqual(dropped["stale"], ["DEADUSDT"])
        self.assertEqual(dropped["unknown"], ["NEVERUSDT"])

    def test_flag_off_filters_nothing(self):
        store = _Store([rec("DEADUSDT", day(400))])
        kept, dropped = PF.filter_live(["DEADUSDT"], store=store, enabled=False)
        self.assertEqual(kept, ["DEADUSDT"])
        self.assertEqual(dropped["stale"], [])


class TestAgainstTheRealStore(unittest.TestCase):
    """The three phantoms that produced the finding, and the live rename."""

    def test_known_phantoms_and_the_live_rename(self):
        for dead in ("RNDRUSDT", "EOSUSDT", "ACAUSDT"):
            self.assertFalse(PF.is_live(dead), f"{dead} last traded years ago")
        self.assertTrue(PF.is_live("RENDERUSDT"),
                        "the live half of the RNDR rename")


class TestWatchlistIsClean(unittest.TestCase):
    def test_no_phantom_survives_in_the_watchlist(self):
        import json
        wl = json.loads((HERE / "watchlist.json").read_text(encoding="utf-8"))
        stale = [s for s in wl if PF.liveness(s) == "stale"]
        self.assertEqual(stale, [], f"delisted symbols still listed: {stale}")

    def test_the_rename_left_only_the_live_half(self):
        import json
        wl = set(json.loads((HERE / "watchlist.json").read_text(encoding="utf-8")))
        self.assertIn("RENDERUSDT", wl)
        self.assertNotIn("RNDRUSDT", wl)


class TestFailsOpenWhenTheStoreLags(unittest.TestCase):
    """The store is rebuilt daily. If that build breaks, every symbol goes stale
    at once and a fail-closed filter would empty the universe — a pipeline
    outage rendered as 'nothing is trading today'."""

    def test_a_mass_drop_stands_the_filter_down(self):
        store = _Store([rec(f"S{i}", day(400)) for i in range(10)])
        symbols = [f"S{i}" for i in range(10)]
        kept, dropped = PF.filter_live(symbols, store=store)
        self.assertEqual(kept, symbols, "must keep everything and flag it")
        self.assertTrue(dropped["stood_down"])
        self.assertEqual(dropped["would_have_dropped"], 10)

    def test_a_small_drop_still_filters(self):
        store = _Store([rec(f"S{i}", day(1)) for i in range(9)]
                       + [rec("DEADUSDT", day(400))])
        symbols = [f"S{i}" for i in range(9)] + ["DEADUSDT"]
        kept, dropped = PF.filter_live(symbols, store=store)
        self.assertNotIn("DEADUSDT", kept)
        self.assertNotIn("stood_down", dropped)


class TestSnapshotWriterIsGuarded(unittest.TestCase):
    """The dataset is what every model here trains on, so a phantom row costs
    more than a wasted slot in a ten-name list."""

    def test_the_snapshot_filters_before_writing(self):
        src = (HERE / "backfill_top_gainer_dataset.py").read_text(encoding="utf-8")
        self.assertIn("phantom_filter", src)
        self.assertIn("filter_live", src)

    def test_it_reports_rather_than_silently_shrinking(self):
        src = (HERE / "backfill_top_gainer_dataset.py").read_text(encoding="utf-8")
        self.assertIn("phantom filter: dropped", src)

    def test_an_unavailable_filter_does_not_stop_collection(self):
        # Losing a day of training data because a helper failed to import is a
        # worse outcome than a few phantom rows.
        src = (HERE / "backfill_top_gainer_dataset.py").read_text(encoding="utf-8")
        self.assertIn("writing the raw watchlist", src)


if __name__ == "__main__":
    unittest.main()
