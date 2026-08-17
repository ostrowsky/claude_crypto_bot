"""Exit gates for canonical EX1 (TH-02 / TH-11).

The property that matters: the daily number is the canonical one, and if it is
not, the scorecard says unknown rather than publishing the proxy as the answer.

Spec: docs/specs/features/canonical-ex1-zigzag-spec.md
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


class TestAggregatorCanPassArguments(unittest.TestCase):
    """`SCRIPTS` was a flat list of filenames and subprocess.run passed none, so
    `--use-zigzag` could not be set no matter what anyone intended."""

    def setUp(self):
        # Read SCRIPTS without importing: report_metrics_daily runs every metric
        # script at MODULE level, so importing it here spawned the whole daily
        # aggregation and made this test take five minutes.
        import ast
        tree = ast.parse((HERE / "report_metrics_daily.py").read_text(encoding="utf-8"))
        self.scripts = None
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign)
                    and any(getattr(t, "id", None) == "SCRIPTS" for t in node.targets)):
                self.scripts = ast.literal_eval(node.value)
        self.assertIsNotNone(self.scripts, "SCRIPTS not found")

    def test_script_table_carries_arguments(self):
        for entry in self.scripts:
            self.assertIsInstance(entry, tuple, "each entry is (filename, args)")
            self.assertEqual(len(entry), 2)
            self.assertIsInstance(entry[1], (list, tuple))

    def test_ex1_is_invoked_canonically(self):
        args = dict((name, list(a)) for name, a in self.scripts)
        self.assertIn("--use-zigzag", args["_backtest_ex1_realized_potential.py"])


class TestProvenanceTravelsWithTheNumber(unittest.TestCase):
    def test_metric_names_its_potential_source(self):
        src = (HERE / "_backtest_ex1_realized_potential.py").read_text(encoding="utf-8")
        self.assertIn('"potential_source"', src)
        # The counts make a mixed population visible instead of averaged away:
        # a trade with no matching uptrend falls back to the proxy.
        self.assertIn('"n_zigzag"', src)
        self.assertIn('"n_proxy"', src)


class TestScorecardRefusesAProxyValue(unittest.TestCase):
    def setUp(self):
        import bot_health_report as H
        self.H = H

    def _score(self, ex1_payload):
        return self.H.build_canonical_scorecard(
            {"metrics": {"EX1_realized_potential": ex1_payload}})

    def test_proxy_payload_stays_unknown_with_a_reason(self):
        # Publishing the proxy would answer the question with a number that
        # measures a move no single trade had to catch.
        s = self._score({"potential_source": "proxy",
                         "top20": {"n": 27, "median": 0.0027}})
        rp = s["realized_potential"]
        self.assertIsNone(rp["value"])
        self.assertEqual(rp["status"], "unknown")
        self.assertIn("proxy", str(rp.get("reason", "")).lower())

    def test_canonical_payload_is_published(self):
        s = self._score({"potential_source": "zigzag",
                         "top20": {"n": 27, "median": 0.0032,
                                   "share_ex1_ge_05": 11.1}})
        rp = s["realized_potential"]
        self.assertEqual(rp["value"], 0.0032)
        self.assertEqual(rp["n"], 27)

    def test_missing_payload_stays_unknown(self):
        rp = self._score({})["realized_potential"]
        self.assertIsNone(rp["value"])


    def test_reason_does_not_reinstate_the_refuted_cause(self):
        # The first diagnosis blamed missing 15m kline history. Instrumenting
        # the matcher showed ZERO misses from absent files: 17 of 18 are trades
        # overlapping no detected uptrend at 1h. This pins the corrected reason
        # so the refuted one cannot drift back into a user-facing report.
        s = self._score({"potential_source": "mixed", "top20_zigzag_n": 9,
                         "zigzag_coverage": 0.36,
                         "top20": {"n": 27, "median": 0.0032}})
        reason = str(s["realized_potential"].get("reason", "")).lower()
        self.assertIn("not a data gap", reason)
        self.assertNotIn("missing 15m", reason)
        self.assertIn("overlap", reason)

    def test_share_carries_its_denominator(self):
        # TH-01: 11.1% is three trades out of 27; one trade moves it 3.7pp.
        s = self._score({"potential_source": "zigzag",
                         "top20": {"n": 27, "median": 0.0032,
                                   "share_ex1_ge_05": 11.1}})
        rp = s["realized_potential"]
        self.assertEqual(rp["n"], 27)
        self.assertIn("share_ex1_ge_05", rp)


class TestMatchDiagnosisIsPerTrade(unittest.TestCase):
    """Counting failures in aggregate cannot say WHY they failed. The row must
    carry its own interval and the reason the matcher rejected it, or the two
    readings — 'the bot traded outside any uptrend' and 'the detector is too
    strict here' — stay indistinguishable."""

    def setUp(self):
        self.src = (HERE / "_backtest_ex1_realized_potential.py").read_text(
            encoding="utf-8")

    def test_row_carries_its_own_interval(self):
        for key in ('"entry_ts"', '"exit_ts"', '"hold_hours"'):
            self.assertIn(key, self.src)

    def test_row_carries_the_rejection_reason(self):
        for key in ('"zz_why"', '"zz_nearest_gap_min"', '"zz_n_trends"'):
            self.assertIn(key, self.src)

    def test_reasons_are_distinguishable(self):
        # Four outcomes, not one boolean: a missing file and a trendless stretch
        # are different facts and were both reported as "proxy".
        for why in ("no_klines", "no_uptrends", "no_overlap", "matched"):
            self.assertIn(f'"{why}"', self.src)

    def test_metric_publishes_the_failure_breakdown(self):
        self.assertIn('"match_failure"', self.src)

    def test_helper_reports_a_reason_for_a_symbol_with_no_klines(self):
        # A missing file must say so, not fall through as "no overlap": the two
        # are different facts and were both reported as "proxy".
        import _backtest_ex1_realized_potential as EX
        from datetime import datetime, timezone
        t0 = datetime(2026, 8, 1, tzinfo=timezone.utc)
        value, diag = EX._zigzag_potential_for_trade(
            "NOSUCHSYMUSDT", t0, t0, tf="1h")
        self.assertIsNone(value)
        self.assertEqual(diag["why"], "no_klines")
        # No uptrends exist, so no distance exists. Zero would read as
        # "adjacent", which is the opposite conclusion.
        self.assertIsNone(diag["nearest_gap_min"])
        self.assertEqual(diag["n_trends"], 0)


if __name__ == "__main__":
    unittest.main()
