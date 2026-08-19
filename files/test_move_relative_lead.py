"""Exit gates for move-relative lead (goal 2: signal entry as early as possible).

The clock-hour lead rewarded being early in the CALENDAR and penalised being
early in the MOVE. These pin the replacement, and in particular pin the case
that would quietly restate a data gap as bad timing.

Spec: docs/specs/features/move-relative-lead-spec.md
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _compute_early_capture as CE  # noqa: E402

OPEN = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
DEADLINE = OPEN + timedelta(hours=8)          # +2.5% crossed at 08:00


def lead(entry_dt, deadline_ts=DEADLINE, open_ts=OPEN):
    return CE.move_relative_lead(entry_dt, open_ts=open_ts, deadline_ts=deadline_ts)


class TestScale(unittest.TestCase):
    def test_entry_at_the_open_is_one(self):
        self.assertEqual(lead(OPEN), 1.0)

    def test_entry_at_the_deadline_is_zero(self):
        self.assertEqual(lead(DEADLINE), 0.0)

    def test_halfway_is_a_half(self):
        self.assertAlmostEqual(lead(OPEN + timedelta(hours=4)), 0.5, places=6)

    def test_after_the_deadline_is_clamped_not_negative(self):
        self.assertEqual(lead(DEADLINE + timedelta(hours=6)), 0.0)

    def test_before_the_open_is_clamped_to_one(self):
        self.assertEqual(lead(OPEN - timedelta(hours=1)), 1.0)


class TestTheCaseTheClockVersionGotBackwards(unittest.TestCase):
    """A coin that starts moving at 20:00 and is caught at 20:05 is the best the
    bot can do; the clock lead scored it 0.17 and scored an idle 02:00 buy 0.92."""

    def test_a_late_day_catch_just_before_the_deadline_scores_well(self):
        open_ts = datetime(2026, 5, 1, tzinfo=timezone.utc)
        deadline = open_ts + timedelta(hours=20, minutes=30)   # move starts late
        entry = open_ts + timedelta(hours=20, minutes=5)
        self.assertGreater(lead(entry, deadline, open_ts), 0.0)
        clock_lead = 1.0 - entry.hour / 24.0
        self.assertLess(clock_lead, 0.2, "the old definition punished this catch")

    def test_an_idle_early_buy_no_longer_scores_high(self):
        # 02:00 entry, move only crosses +2.5% at 03:00: barely any lead left.
        open_ts = datetime(2026, 5, 1, tzinfo=timezone.utc)
        deadline = open_ts + timedelta(hours=3)
        entry = open_ts + timedelta(hours=2)
        self.assertAlmostEqual(lead(entry, deadline, open_ts), 1 / 3, places=6)
        self.assertGreater(1.0 - entry.hour / 24.0, 0.9)


class TestUncomputableIsNotZero(unittest.TestCase):
    def test_missing_deadline_returns_none(self):
        # A daily-resolution label has no crossing time. Scoring 0.0 would say
        # "alerted late", which is a claim about the bot, not about the data.
        self.assertIsNone(CE.move_relative_lead(OPEN, open_ts=OPEN,
                                                deadline_ts=None))

    def test_zero_length_window_does_not_divide_by_zero(self):
        self.assertIsNone(CE.move_relative_lead(OPEN, open_ts=OPEN,
                                                deadline_ts=OPEN))


class TestFlagAndProvenance(unittest.TestCase):
    def test_flag_exists_and_v3_is_published_beside_v2(self):
        # Asserted False while the side-by-side reading was unpublished; it is
        # published now (lead 0.61 -> 0.02) and the flag is on. What must
        # survive: the switch exists, and v3 is an ADDITIONAL line rather than a
        # replacement — substituting it would show a 17x collapse that looks
        # like a regression and is a change of question.
        import config
        self.assertIsInstance(
            getattr(config, "NS_MOVE_RELATIVE_LEAD_ENABLED", None), bool)
        src = (HERE / "_compute_early_capture.py").read_text(encoding="utf-8")
        self.assertIn("move_lead_early_capture", src)
        self.assertIn('"early_capture": primary["early_capture"]', src)
        self.assertIn("move_lead_winners_without_deadline", src)

    def test_metric_names_its_lead_definition(self):
        src = (HERE / "_compute_early_capture.py").read_text(encoding="utf-8")
        self.assertIn("lead_definition", src)
        self.assertIn("move_relative", src)
        self.assertIn("clock_hour", src)


if __name__ == "__main__":
    unittest.main()
