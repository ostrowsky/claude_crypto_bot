"""Tests for structured blocked-event logging (RM-1/RM-2).

Verifies that:
1. All blocked events have reason_code and gate fields
2. Context fields (features, scores) are properly populated
3. No field-type errors in the logged JSON
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import botlog


class BlockedEventLoggingTests(unittest.TestCase):
    """Test structured blocked-event logging."""

    def setUp(self):
        """Redirect log file to temp location."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_log = Path(self.temp_dir.name) / "test_events.jsonl"
        self.patcher = mock.patch.object(botlog, "LOG_FILE", self.temp_log)
        self.patcher.start()

    def tearDown(self):
        """Clean up."""
        self.patcher.stop()
        self.temp_dir.cleanup()

    def _read_logged_events(self) -> list[dict]:
        """Read all blocked events from temp log file."""
        if not self.temp_log.exists():
            return []
        
        events = []
        with self.temp_log.open("r") as f:
            for line in f:
                if line.strip():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return events

    def test_basic_blocked_event_with_reason_code(self):
        """Test that reason_code is logged when provided."""
        botlog.log_blocked(
            sym="BTCUSDT",
            tf="15m",
            price=50000.0,
            reason="ML proba outside zone",
            reason_code="ml_zone",
            gate="ml_proba_zone",
        )

        events = self._read_logged_events()
        self.assertEqual(len(events), 1)

        event = events[0]
        self.assertEqual(event["event"], "blocked")
        self.assertEqual(event["sym"], "BTCUSDT")
        self.assertEqual(event["tf"], "15m")
        self.assertEqual(event["reason_code"], "ml_zone")
        self.assertEqual(event["gate"], "ml_proba_zone")
        self.assertIn("ts", event)

    def test_blocked_event_with_all_context_fields(self):
        """Test that all context fields are logged when provided."""
        botlog.log_blocked(
            sym="ETHUSDT",
            tf="1h",
            price=3000.0,
            reason="Entry score too low",
            reason_code="entry_score",
            gate="entry_score_floor",
            rsi=72.0,
            adx=25.5,
            vol_x=1.45,
            daily_range=4.2,
            slope_pct=1.35,
            macd_hist=0.00001,
            ema20=2990.0,
            ema50=2980.0,
            ema200=2950.0,
            atr_pct=0.025,
            ml_proba=0.38,
            ranker_top_gainer_prob=0.65,
            ranker_ev=0.08,
            ranker_quality_proba=0.72,
            ranker_final_score=35.0,
            candidate_score=40.0,
            score_floor=45.0,
            is_bull_day=True,
            btc_vs_ema50=0.015,
            market_regime="bull",
        )

        events = self._read_logged_events()
        self.assertEqual(len(events), 1)

        event = events[0]
        self.assertEqual(event["rsi"], 72.0)
        self.assertEqual(event["adx"], 25.5)
        self.assertAlmostEqual(event["vol_x"], 1.45, places=2)
        self.assertEqual(event["slope_pct"], 1.35)
        self.assertAlmostEqual(event["ml_proba"], 0.38, places=2)
        self.assertAlmostEqual(event["ranker_top_gainer_prob"], 0.65, places=2)
        self.assertAlmostEqual(event["ranker_final_score"], 35.0, places=2)
        self.assertEqual(event["candidate_score"], 40.0)
        self.assertEqual(event["score_floor"], 45.0)
        self.assertTrue(event["is_bull_day"])
        self.assertEqual(event["market_regime"], "bull")

    def test_null_fields_omitted_from_json(self):
        """Test that None/null fields are omitted to keep JSON compact."""
        botlog.log_blocked(
            sym="BNBUSDT",
            tf="15m",
            price=500.0,
            reason="Trend quality guard",
            reason_code="trend_quality",
            gate="trend_15m_quality_guard",
            rsi=None,  # Explicitly None
            adx=None,
            ml_proba=0.55,  # Present
        )

        events = self._read_logged_events()
        self.assertEqual(len(events), 1)

        event = events[0]
        self.assertNotIn("rsi", event)  # Null fields omitted
        self.assertNotIn("adx", event)
        self.assertIn("ml_proba", event)  # Non-null fields present
        self.assertEqual(event["ml_proba"], 0.55)

    def test_multiple_blocked_events(self):
        """Test logging multiple blocked events."""
        for i in range(5):
            botlog.log_blocked(
                sym=f"SYM{i}USDT",
                tf="15m" if i % 2 == 0 else "1h",
                price=100.0 + i,
                reason=f"Reason {i}",
                reason_code="entry_score" if i % 2 == 0 else "ml_zone",
                gate="entry_score_floor" if i % 2 == 0 else "ml_proba_zone",
            )

        events = self._read_logged_events()
        self.assertEqual(len(events), 5)

        for i, event in enumerate(events):
            self.assertEqual(event["sym"], f"SYM{i}USDT")
            self.assertIn(event["reason_code"], ["entry_score", "ml_zone"])

    def test_backward_compat_without_reason_code(self):
        """Test that old-style calls without reason_code still work."""
        botlog.log_blocked(
            sym="ADAUSDT",
            tf="15m",
            price=1.2,
            reason="Some legacy reason",
            # No reason_code or gate
        )

        events = self._read_logged_events()
        self.assertEqual(len(events), 1)

        event = events[0]
        self.assertEqual(event["sym"], "ADAUSDT")
        self.assertNotIn("reason_code", event)  # Omitted when None
        self.assertNotIn("gate", event)

    def test_json_serialization_no_errors(self):
        """Test that all logged events are valid JSON."""
        test_cases = [
            ("ml_zone", "ml_proba_zone", 0.38),
            ("trend_quality", "trend_15m_quality_guard", 0.55),
            ("entry_score", "entry_score_floor", 40.0),
            ("ranker_hard_veto", "ml_candidate_ranker", 20.0),
        ]

        for reason_code, gate, score in test_cases:
            botlog.log_blocked(
                sym="TESTUSDT",
                tf="15m",
                price=100.0,
                reason=f"Test {reason_code}",
                reason_code=reason_code,
                gate=gate,
                candidate_score=score,
                ml_proba=0.5,
            )

        events = self._read_logged_events()
        self.assertEqual(len(events), len(test_cases))

        # Verify all are valid JSON by round-tripping
        for i, event in enumerate(events):
            json_str = json.dumps(event)
            reparsed = json.loads(json_str)
            self.assertEqual(reparsed["reason_code"], test_cases[i][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
