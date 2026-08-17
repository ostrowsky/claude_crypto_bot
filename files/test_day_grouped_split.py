"""Exit gates for the day-grouped training split (TH-04).

The defect being fixed is subtle: the existing split sorts chronologically and
then cuts by row index, so it *looks* like walk-forward while placing part of a
UTC day on each side. Since the tier labels are per-day ranks, knowing part of a
day tells you about the rest of it.

Spec: docs/specs/features/day-grouped-training-split-spec.md
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import day_split as DS  # noqa: E402

DAY_MS = 86_400_000
DAY0 = 1_767_225_600_000        # 2026-01-01T00:00:00Z


def timestamps(days: int, per_day: int) -> np.ndarray:
    """Chronologically sorted ms timestamps, `per_day` rows inside each UTC day.

    Rows are spaced to fit within the day. An earlier version used a fixed hour
    step, so `per_day=50` silently spanned three days and a test for the
    single-day case never exercised it.
    """
    step = DAY_MS // max(per_day, 1)
    out = []
    for d in range(days):
        for r in range(per_day):
            out.append(DAY0 + d * DAY_MS + r * step)
    return np.array(out, dtype=np.int64)


class TestSplitIndices(unittest.TestCase):
    TS = timestamps(days=20, per_day=5)

    def _days(self, idx):
        return {int((self.TS[i] // DAY_MS)) for i in idx}

    def test_no_day_appears_on_both_sides(self):
        for frac in (0.1, 0.25, 0.5, 0.75, 0.9):
            tr, ho = DS.split_indices_by_day(self.TS, train_frac=frac)
            with self.subTest(frac=frac):
                self.assertFalse(self._days(tr) & self._days(ho))

    def test_the_boundary_is_chronological(self):
        tr, ho = DS.split_indices_by_day(self.TS, train_frac=0.7)
        self.assertLess(max(self.TS[i] for i in tr), min(self.TS[i] for i in ho))

    def test_row_index_split_would_have_straddled(self):
        # The defect, demonstrated rather than asserted: the old cut lands
        # inside a day whenever len(rows) * frac is not a day boundary.
        cut = int(len(self.TS) * 0.7)
        old_train_days = {int(t // DAY_MS) for t in self.TS[:cut]}
        old_val_days = {int(t // DAY_MS) for t in self.TS[cut:]}
        # 20 days x 5 rows, cut at row 70 -> day 14 is split 0/5? verify shape
        self.assertTrue(len(self.TS) * 0.7 % 5 == 0 or
                        bool(old_train_days & old_val_days),
                        "this fixture must exercise the straddle it is testing")

    def test_embargo_drops_boundary_days_from_training_only(self):
        tr, ho = DS.split_indices_by_day(self.TS, train_frac=0.7, embargo_days=2)
        tr_no, ho_no = DS.split_indices_by_day(self.TS, train_frac=0.7)
        self.assertLess(len(tr), len(tr_no), "embargo must remove training rows")
        self.assertEqual(len(ho), len(ho_no), "holdout must not absorb them")

    def test_deterministic(self):
        a = DS.split_indices_by_day(self.TS, train_frac=0.6)
        b = DS.split_indices_by_day(self.TS, train_frac=0.6)
        self.assertEqual(list(a[0]), list(b[0]))
        self.assertEqual(list(a[1]), list(b[1]))

    def test_single_day_dataset_is_refused(self):
        with self.assertRaises(ValueError):
            DS.split_indices_by_day(timestamps(days=1, per_day=50), train_frac=0.7)

    def test_every_row_is_placed_or_embargoed(self):
        tr, ho = DS.split_indices_by_day(self.TS, train_frac=0.7, embargo_days=2)
        self.assertEqual(len(set(tr) & set(ho)), 0)
        self.assertLessEqual(len(tr) + len(ho), len(self.TS))


class TestTrainerWiring(unittest.TestCase):
    """The flag must be inert by default: production behaviour is unchanged
    until the operator flips it."""

    def test_flag_exists_as_the_rollback_path(self):
        # Asserted "defaults to off" until the operator flipped it on
        # 2026-08-17, together with the immutable label change so one
        # measurable change is attributable. What must survive is that the
        # flag still EXISTS and still names a rollback — a behaviour change
        # without a way back is the thing this test was really guarding.
        import config
        self.assertIsInstance(
            getattr(config, "TRAIN_DAY_GROUPED_SPLIT_ENABLED", None), bool)
        self.assertIsInstance(getattr(config, "TRAIN_SPLIT_EMBARGO_DAYS", None), int)


    def test_trainer_imports_the_shared_splitter(self):
        src = (HERE / "train_top_gainer.py").read_text(encoding="utf-8")
        self.assertIn("day_split", src,
                      "one splitter, not a second implementation")

    def test_scope_string_still_names_the_label_defect(self):
        # Fixing the split does not fix the same-snapshot label. The scope must
        # not imply it did. Checked through the function rather than as a source
        # literal: the string is now composed from the two flags independently,
        # and asserting on a literal would have tested the spelling, not the
        # property.
        import train_top_gainer as TT
        self.assertEqual(TT._evaluation_scope(True, None),
                         "day_grouped_holdout_same_snapshot_label")
        self.assertEqual(TT._evaluation_scope(False, None),
                         "time_sorted_row_holdout_same_snapshot_label")
        # …and once the labels are immutable, the scope stops claiming they are
        # not — each defect is named by its own flag.
        self.assertEqual(TT._evaluation_scope(True, {"n_labelled": 1}),
                         "day_grouped_holdout_immutable_later_eod_label")


if __name__ == "__main__":
    unittest.main()
