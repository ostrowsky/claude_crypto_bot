"""Exit gates for the weekly steering pair.

Written before the implementation. The properties tested are the ones that make
the pair trustworthy rather than merely computable: that downtime does not read
as a miss, that the recall/precision trade is visible, and that buying precision
with lateness shows up instead of being rewarded.

Spec: docs/specs/features/weekly-steering-pair-spec.md
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import weekly_steering as WS  # noqa: E402

HOUR = 3_600_000
DAY0 = 1_767_225_600_000            # 2026-01-01T00:00:00Z


def label(symbol="AUSDT", day="2026-01-01", *, qualifies=True, deadline_h=6):
    return {
        "symbol": symbol, "utc_day": day, "complete": True,
        "qualifies_move5": qualifies,
        "early_deadline_ts": DAY0 + deadline_h * HOUR if deadline_h is not None else None,
        "anchor_ts": DAY0 + 8 * HOUR if qualifies else None,
        "max_move_pct": 7.0 if qualifies else 1.0,
    }


def alert(symbol="AUSDT", day="2026-01-01", *, hour=3):
    return {"symbol": symbol, "utc_day": day, "ts_ms": DAY0 + hour * HOUR}


def observed(days, hours=24):
    return {d: hours for d in days}


class TestEligibility(unittest.TestCase):
    def test_alert_strictly_before_the_deadline_is_eligible(self):
        r = WS.compute([label()], [alert(hour=3)], observed(["2026-01-01"]))
        self.assertEqual(r["coverage"]["value"], 1.0)

    def test_alert_in_the_same_hour_as_the_deadline_counts_late(self):
        # Hourly bars cannot order an alert against a crossing inside the same
        # hour, so the ambiguous case is scored against the bot.
        r = WS.compute([label(deadline_h=6)], [alert(hour=6)], observed(["2026-01-01"]))
        self.assertEqual(r["coverage"]["value"], 0.0)

    def test_no_deadline_means_the_whole_day_is_eligible(self):
        r = WS.compute([label(qualifies=False, deadline_h=None)],
                       [alert(hour=23)], observed(["2026-01-01"]))
        self.assertEqual(r["precision_early"]["n"], 1)


class TestThePairMovesIndependently(unittest.TestCase):
    def test_a_missed_qualifying_event_lowers_coverage_only(self):
        labels = [label("AUSDT"), label("BUSDT")]
        r = WS.compute(labels, [alert("AUSDT")], observed(["2026-01-01"]))
        self.assertEqual(r["coverage"]["value"], 0.5)
        self.assertEqual(r["precision_early"]["value"], 1.0)

    def test_an_alert_on_a_dud_lowers_precision_only(self):
        labels = [label("AUSDT"), label("BUSDT", qualifies=False)]
        r = WS.compute(labels, [alert("AUSDT"), alert("BUSDT")],
                       observed(["2026-01-01"]))
        self.assertEqual(r["coverage"]["value"], 1.0)
        self.assertEqual(r["precision_early"]["value"], 0.5)

    def test_buying_precision_with_lateness_is_visible(self):
        # Alert late on the dud: it leaves the early denominator, so
        # precision_early improves — and precision_all does not.
        labels = [label("AUSDT"), label("BUSDT", qualifies=False, deadline_h=2)]
        late = WS.compute(labels, [alert("AUSDT"), alert("BUSDT", hour=20)],
                          observed(["2026-01-01"]))
        self.assertEqual(late["precision_early"]["value"], 1.0)
        self.assertLess(late["precision_all"]["value"], 1.0,
                        "a late wrong alert must still count somewhere")


class TestDowntime(unittest.TestCase):
    def test_a_partially_observed_day_is_excluded_not_missed(self):
        r = WS.compute([label()], [], {"2026-01-01": 4})
        self.assertEqual(r["coverage"]["n"], 0)
        self.assertEqual(r["days_scored"], 0)
        self.assertEqual(r["days_excluded_no_data"], 1)

    def test_an_unobserved_day_never_enters_the_denominator(self):
        labels = [label(day="2026-01-01"), label(day="2026-01-02")]
        r = WS.compute(labels, [alert(day="2026-01-01")],
                       {"2026-01-01": 24, "2026-01-02": 2})
        self.assertEqual(r["coverage"]["n"], 1)
        self.assertEqual(r["coverage"]["value"], 1.0)


class TestRatioContext(unittest.TestCase):
    def test_every_metric_carries_n_base_rate_and_ci(self):
        labels = [label("AUSDT"), label("BUSDT", qualifies=False)]
        r = WS.compute(labels, [alert("AUSDT")], observed(["2026-01-01"]))
        for key in ("coverage", "precision_early", "precision_all"):
            with self.subTest(metric=key):
                block = r[key]
                self.assertIn("n", block)
                self.assertIn("base_rate", block)
                self.assertIn("ci", block)
                self.assertEqual(len(block["ci"]), 2)

    def test_bootstrap_resamples_days_not_rows(self):
        # 40 rows on one day must not produce a tight interval: there is one
        # independent unit, not forty.
        labels = [label(f"S{i}USDT") for i in range(40)]
        alerts = [alert(f"S{i}USDT") for i in range(20)]
        r = WS.compute(labels, alerts, observed(["2026-01-01"]), bootstrap=200)
        lo, hi = r["coverage"]["ci"]
        self.assertEqual(lo, hi, "one day is one observation; the CI cannot narrow")


class TestNoInventedContext(unittest.TestCase):
    def test_coverage_reports_no_lift_because_it_has_no_base(self):
        # Its denominator is already the qualifying events; a "lift" would be
        # the value divided by one, wearing a label it has not earned.
        r = WS.compute([label()], [alert()], observed(["2026-01-01"]))
        self.assertIsNone(r["coverage"]["lift"])
        self.assertIsNone(r["coverage"]["base_rate"])
        self.assertIsNotNone(r["precision_early"]["base_rate"])

    def test_a_day_with_no_events_at_all_is_counted_as_downtime(self):
        # Absent from the event log entirely — must be excluded loudly, not
        # silently dropped from the window.
        r = WS.compute([label(day="2026-01-01"), label(day="2026-01-02")],
                       [], {"2026-01-01": 24})
        self.assertEqual(r["days_scored"], 1)
        self.assertEqual(r["days_excluded_no_data"], 1)


class TestWindowComparison(unittest.TestCase):
    @staticmethod
    def _window(days, cover):
        labels, alerts = [], []
        for d in days:
            for i in range(10):
                labels.append(label(f"S{i}USDT", d))
                if i < cover:
                    alerts.append(alert(f"S{i}USDT", d))
        return labels, alerts, observed(days)

    def test_inconclusive_when_the_interval_spans_the_threshold(self):
        a = WS.compute(*self._window([f"2026-01-{d:02d}" for d in range(1, 8)], 5))
        b = WS.compute(*self._window([f"2026-02-{d:02d}" for d in range(1, 8)], 6))
        verdict = WS.compare(a, b, metric="coverage", practical_threshold=0.25)
        self.assertEqual(verdict["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertIn("рано судить", verdict["text"])

    def test_mismatched_windows_are_refused(self):
        a = WS.compute(*self._window([f"2026-01-{d:02d}" for d in range(1, 15)], 5))
        b = WS.compute(*self._window(["2026-02-01", "2026-02-02"], 5))
        verdict = WS.compare(a, b, metric="coverage", practical_threshold=0.1)
        self.assertEqual(verdict["verdict"], "NOT_COMPARABLE")


if __name__ == "__main__":
    unittest.main()
