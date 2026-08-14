"""Exit gates for the immutable label store and the day-grouped splitter.

Written before the implementation. The properties under test are the ones whose
absence produced two harness blockers: a label that measures itself, and a split
that puts one UTC day on both sides.

Spec: docs/specs/features/immutable-label-store-spec.md
"""
from __future__ import annotations

import ast
import json
import shutil
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import day_split as DS  # noqa: E402
import label_store as LS  # noqa: E402

HOUR_MS = 3_600_000


def _day_bars(day_start_ms: int, *, open_px: float, path: list[float],
              hours: int = 24) -> list[list]:
    """One synthetic UTC day. `path` gives each bar's high as a multiple of open."""
    bars = []
    for h in range(hours):
        mult = path[h] if h < len(path) else 1.0
        px = open_px * mult
        bars.append([day_start_ms + h * HOUR_MS, open_px, px, open_px * 0.99, px, 100.0])
    return bars


class TestLabelComputation(unittest.TestCase):
    DAY0 = 1_767_225_600_000          # 2026-01-01T00:00:00Z, an exact UTC midnight

    def _build(self, path, hours=24):
        bars = _day_bars(self.DAY0, open_px=100.0, path=path, hours=hours)
        return LS.build_day_record("TSTUSDT", self.DAY0, bars,
                                   provenance={"source": "unit-test",
                                               "source_sha256": "0" * 64})

    def test_eod_return_uses_close_not_the_high(self):
        # +9% intraday, closing at +1%: the immutable label is the close.
        rec = self._build([1.0] * 12 + [1.09] + [1.01] * 11)
        self.assertAlmostEqual(rec["max_move_pct"], 9.0, places=6)
        self.assertAlmostEqual(rec["eod_return_pct"], 1.0, places=6)

    def test_move5_qualifies_on_the_days_high(self):
        rec = self._build([1.0] * 8 + [1.06] + [1.0] * 15)
        self.assertTrue(rec["qualifies_move5"])
        self.assertFalse(self._build([1.0] * 24)["qualifies_move5"])

    def test_early_deadline_never_follows_the_anchor(self):
        # The v1 MoveEvent bug: a "midpoint" defined as half the realised move
        # put the deadline after the anchor for moves above +5%. A fixed +2.5%
        # crossing cannot.
        rec = self._build([1.0, 1.01, 1.03, 1.04, 1.06] + [1.0] * 19)
        self.assertIsNotNone(rec["early_deadline_ts"])
        self.assertIsNotNone(rec["anchor_ts"])
        self.assertLessEqual(rec["early_deadline_ts"], rec["anchor_ts"])

    def test_deadline_exists_without_an_anchor(self):
        rec = self._build([1.0, 1.03] + [1.0] * 22)     # crosses +2.5%, never +5%
        self.assertIsNotNone(rec["early_deadline_ts"])
        self.assertIsNone(rec["anchor_ts"])
        self.assertFalse(rec["qualifies_move5"])

    def test_incomplete_day_is_unknown_not_zero(self):
        rec = self._build([1.0] * 6, hours=6)
        self.assertFalse(rec["complete"])
        self.assertFalse(rec["qualifies_move5"],
                         "a partial day must not qualify — absence of data is not a fact")

    def test_every_record_carries_provenance(self):
        rec = self._build([1.0] * 24)
        self.assertEqual(len(rec["provenance"]["source_sha256"]), 64)
        self.assertIn("builder_version", rec["provenance"])
        self.assertTrue(rec["label_mature_at"])


class TestImmutability(unittest.TestCase):
    def setUp(self):
        self.root = HERE.parent / ".runtime" / "labels_test"
        shutil.rmtree(self.root, ignore_errors=True)
        self.store = LS.LabelStore(self.root)
        self.addCleanup(shutil.rmtree, self.root, True)
        self.rec = {"symbol": "TSTUSDT", "utc_day": "2026-01-01",
                    "eod_return_pct": 1.0, "complete": True,
                    "provenance": {"source_sha256": "a" * 64}}

    def test_rebuilding_an_identical_record_is_a_no_op(self):
        self.store.put(self.rec)
        self.store.put(dict(self.rec))
        self.assertEqual(len(self.store.records()), 1)

    def test_a_changed_record_is_refused_not_overwritten(self):
        self.store.put(self.rec)
        changed = dict(self.rec, eod_return_pct=99.0)
        with self.assertRaises(LS.ImmutableLabelError):
            self.store.put(changed)
        self.assertEqual(self.store.records()[0]["eod_return_pct"], 1.0)

    def test_a_changed_source_hash_is_refused(self):
        self.store.put(self.rec)
        changed = dict(self.rec, provenance={"source_sha256": "b" * 64})
        with self.assertRaises(LS.ImmutableLabelError):
            self.store.put(changed)


class TestIsolation(unittest.TestCase):
    def test_label_module_reads_no_bot_dataset(self):
        tree = ast.parse((HERE / "label_store.py").read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        forbidden = {"config", "monitor", "strategy", "critic_dataset",
                     "ml_dataset", "offline_rl", "top_gainer_critic"}
        self.assertFalse(names & forbidden,
                         "a label derived from the bot's own snapshots is not ground truth")
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for banned in ("top_gainer_dataset", "critic_dataset", "ml_dataset"):
                    self.assertNotIn(banned, node.value)


class TestCoverageHonesty(unittest.TestCase):
    """A "per day" rate computed across a union of eras is not a rate.

    The kline fetch returns whatever history exists, so a delisted or renamed
    symbol contributes days from a different year. Three of 98 symbols did
    exactly that, and the first version of `status()` divided by the union of
    all days — reporting 5.18 qualifying events per day where the real figure
    over the well-covered window is 16.27.
    """

    def _recs(self):
        current = [{"symbol": f"S{i}", "utc_day": f"2026-08-{d:02d}",
                    "qualifies_move5": i == 0}
                   for d in range(1, 11) for i in range(10)]
        stale = [{"symbol": "OLDUSDT", "utc_day": f"2024-01-{d:02d}",
                  "qualifies_move5": False} for d in range(1, 11)]
        return current + stale

    def test_rates_are_scoped_to_well_covered_days(self):
        st = LS.status_from_records(self._recs())
        self.assertEqual(st["well_covered_days"], 10)
        self.assertGreater(st["days_any_coverage"], st["well_covered_days"])
        # 1 qualifying of 10 symbols on each of 10 covered days
        self.assertAlmostEqual(st["qualifying_per_day"], 1.0, places=2)

    def test_stale_symbols_are_named_not_averaged_away(self):
        st = LS.status_from_records(self._recs())
        self.assertTrue(any("OLDUSDT" in s for s in st["stale_symbols"]))

    def test_rates_withheld_when_no_day_is_well_covered(self):
        sparse = [{"symbol": f"S{i}", "utc_day": f"2026-08-{i:02d}",
                   "qualifies_move5": False} for i in range(1, 12)]
        st = LS.status_from_records(sparse)
        self.assertEqual(st["well_covered_days"], 0)
        self.assertNotIn("qualifying_per_day", st)


class TestDaySplit(unittest.TestCase):
    ROWS = [{"utc_day": f"2026-01-{d:02d}", "i": i}
            for d in range(1, 21) for i in range(5)]

    def test_no_day_appears_on_both_sides(self):
        for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
            train, holdout = DS.split_by_day(self.ROWS, "utc_day", train_frac=frac)
            with self.subTest(frac=frac):
                self.assertFalse({r["utc_day"] for r in train} &
                                 {r["utc_day"] for r in holdout})

    def test_split_is_chronological(self):
        train, holdout = DS.split_by_day(self.ROWS, "utc_day", train_frac=0.7)
        self.assertLess(max(r["utc_day"] for r in train),
                        min(r["utc_day"] for r in holdout))

    def test_embargo_removes_boundary_days_from_training(self):
        train, holdout = DS.split_by_day(self.ROWS, "utc_day", train_frac=0.7,
                                         embargo_days=3)
        gap_start = max(r["utc_day"] for r in train)
        gap_end = min(r["utc_day"] for r in holdout)
        removed = [d for d in sorted({r["utc_day"] for r in self.ROWS})
                   if gap_start < d < gap_end]
        self.assertEqual(len(removed), 3,
                         "an embargo must actually withhold the boundary days")

    def test_deterministic(self):
        a = DS.split_by_day(self.ROWS, "utc_day", train_frac=0.6)
        b = DS.split_by_day(self.ROWS, "utc_day", train_frac=0.6)
        self.assertEqual(a, b)

    def test_refuses_when_a_side_would_be_empty(self):
        with self.assertRaises(ValueError):
            DS.split_by_day(self.ROWS, "utc_day", train_frac=1.0)


if __name__ == "__main__":
    unittest.main()
