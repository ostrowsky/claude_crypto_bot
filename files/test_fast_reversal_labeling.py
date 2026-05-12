"""Tests for RM-3 anti-fast-reversal labeling.

Tests the label_fast_reversal computation:
- label = 1 if low(t+1..t+3) ≤ entry × (1 - atr_pct × trail_k)
- label = 0 otherwise
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np

import sys
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import critic_dataset


class FastReversalLabelingTests(unittest.TestCase):
    """Test fast-reversal label computation."""

    def setUp(self):
        """Redirect critic dataset file to temp location."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_log = Path(self.temp_dir.name) / "test_critic.jsonl"
        self.patcher = mock.patch.object(critic_dataset, "CRITIC_FILE", self.temp_log)
        self.patcher.start()

    def tearDown(self):
        """Clean up."""
        self.patcher.stop()
        self.temp_dir.cleanup()

    def _read_critic_records(self) -> list[dict]:
        """Read all critic records from temp file."""
        if not self.temp_log.exists():
            return []
        
        records = []
        with self.temp_log.open("r") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def test_fast_reversal_label_hit_threshold(self):
        """Test that label = 1 when low hits trail stop."""
        # Setup: entry at 100, ATR 2, trail_k 2.0 → trail_buffer = 2*2 = 4% → threshold = 96
        # Next 3 bars: lows = [99, 95, 98] → min=95 < 96 → label should be 1
        
        entry_price = 100.0
        atr_pct = 2.0
        trail_k = 2.0
        bar_ms = 15 * 60 * 1000  # 15 min

        # Create a record
        rec = {
            "id": "test_001",
            "sym": "TESTUSDT",
            "tf": "15m",
            "bar_ts": 1000000,
            "entry_price": entry_price,
            "entry_context": {"atr_pct": atr_pct, "trail_k": trail_k},
            "labels": {"label_fast_reversal": None},
        }

        # Write to file
        with self.temp_log.open("w") as f:
            f.write(json.dumps(rec) + "\n")

        # Create time and low arrays
        # Entry bar at t=1000000, then 3 bars at 1000000 + 15min each
        t_arr = np.array([1000000, 1000000 + bar_ms, 1000000 + 2*bar_ms, 1000000 + 3*bar_ms], dtype=int)
        l_arr = np.array([100.0, 99.0, 95.0, 98.0], dtype=float)  # min at bar 2 = 95 < 96

        # Fill pending from data
        critic_dataset.fill_pending_from_data(
            sym="TESTUSDT", tf="15m",
            t_arr=t_arr,
            c_arr=np.array([100.0, 100.5, 100.2, 100.8]),  # closes (not used for fast-reversal)
            bar_ms=bar_ms,
            l_arr=l_arr,
        )

        records = self._read_critic_records()
        self.assertEqual(len(records), 1)

        rec = records[0]
        # Should be labeled as fast reversal (1) since low 95 hit threshold 96
        self.assertEqual(rec["labels"]["label_fast_reversal"], 1)

    def test_fast_reversal_label_miss_threshold(self):
        """Test that label = 0 when low does not hit trail stop."""
        # Entry at 100, ATR 2, trail_k 2.0 → threshold = 96
        # Next 3 bars: lows = [99, 97, 98] → min=97 > 96 → label should be 0
        
        entry_price = 100.0
        atr_pct = 2.0
        trail_k = 2.0
        bar_ms = 15 * 60 * 1000

        rec = {
            "id": "test_002",
            "sym": "TESTUSDT",
            "tf": "15m",
            "bar_ts": 2000000,
            "entry_price": entry_price,
            "entry_context": {"atr_pct": atr_pct, "trail_k": trail_k},
            "labels": {"label_fast_reversal": None},
        }

        with self.temp_log.open("w") as f:
            f.write(json.dumps(rec) + "\n")

        t_arr = np.array([2000000, 2000000 + bar_ms, 2000000 + 2*bar_ms, 2000000 + 3*bar_ms], dtype=int)
        l_arr = np.array([100.0, 99.0, 97.0, 98.0], dtype=float)  # min at bar 2 = 97 > 96

        critic_dataset.fill_pending_from_data(
            sym="TESTUSDT", tf="15m",
            t_arr=t_arr,
            c_arr=np.array([100.0, 100.5, 100.2, 100.8]),
            bar_ms=bar_ms,
            l_arr=l_arr,
        )

        records = self._read_critic_records()
        self.assertEqual(len(records), 1)

        rec = records[0]
        # Should NOT be labeled as fast reversal (0) since low 97 > threshold 96
        self.assertEqual(rec["labels"]["label_fast_reversal"], 0)

    def test_fast_reversal_label_high_volatility(self):
        """Test with high ATR (larger trail buffer)."""
        # Entry at 100, ATR 5, trail_k 3.0 → buffer = 15% → threshold = 85
        # Next 3 bars: lows = [95, 88, 90] → min=88 > 85 → label should be 0
        
        entry_price = 100.0
        atr_pct = 5.0
        trail_k = 3.0
        bar_ms = 15 * 60 * 1000

        rec = {
            "id": "test_003",
            "sym": "ETHUSDT",
            "tf": "15m",
            "bar_ts": 3000000,
            "entry_price": entry_price,
            "entry_context": {"atr_pct": atr_pct, "trail_k": trail_k},
            "labels": {"label_fast_reversal": None},
        }

        with self.temp_log.open("w") as f:
            f.write(json.dumps(rec) + "\n")

        t_arr = np.array([3000000, 3000000 + bar_ms, 3000000 + 2*bar_ms, 3000000 + 3*bar_ms], dtype=int)
        l_arr = np.array([100.0, 95.0, 88.0, 90.0], dtype=float)  # min=88 > 85

        critic_dataset.fill_pending_from_data(
            sym="ETHUSDT", tf="15m",
            t_arr=t_arr,
            c_arr=np.array([100.0, 100.5, 100.2, 100.8]),
            bar_ms=bar_ms,
            l_arr=l_arr,
        )

        records = self._read_critic_records()
        self.assertEqual(len(records), 1)

        rec = records[0]
        # Should NOT be fast reversal (0) since min 88 > threshold 85
        self.assertEqual(rec["labels"]["label_fast_reversal"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
