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


if __name__ == "__main__":
    unittest.main()
