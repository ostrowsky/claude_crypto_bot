"""Contracts for the block-reason taxonomy and the why-no-signal verdicts.

The taxonomy classifies free text, so it rots silently the moment a gate
reworded its message. These tests pin the real strings observed in
`bot_events.jsonl` — including the mangled and mixed-alphabet ones — so a
rewording fails here instead of quietly moving 50k blocks into another bucket.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import block_reasons as BR  # noqa: E402
import why_no_signal as W  # noqa: E402


class TestBlockReasons(unittest.TestCase):
    # Sampled from the live file with their observed counts, so a regression
    # here is a regression on real volume, not on invented text.
    OBSERVED = [
        ("ML proba 0.19 outside profitable zone [0.28,1.01]", "ml_proba_zone"),
        ("entry score 27.69 < floor 35.00", "entry_score"),
        ("trend quality guard: RSI 73.2 > 72.0", "trend_quality"),
        ("trend quality guard: weak 15m trend (forecast 0.000 < 0.250, vol 1.23)",
         "trend_quality"),
        ("mode_range_quality: alignment/15m daily_range 3.02% < 4.00%",
         "mode_range_quality"),
        ("bandit skip: ucbs=[0.0603, -0.0142]", "bandit_skip"),
        ("trend/1h chop: ADX 19.6<25 OR slope +0.54%<0.7%", "trend_1h_chop"),
        ("1h impulse_speed guard: RSI 88.1 > 85.0", "impulse_speed_guard"),
        ("impulse_speed regime-curtailed (trailing realized pnl < 0)",
         "impulse_speed_curtail"),
        ("портфель: портфель полон: 6/6 позиций", "portfolio_full"),
        ("ranker hard veto: final -3.1 <= -2.50 and TG 0.11 <= 0.25",
         "ranker_hard_veto"),
        ("clone signal guard: recent 15m setups in group 'Meme' already 2/2",
         "clone_signal_guard"),
        ("open cluster cap: 15m_impulse already 3/3 (COTIUSDT, UMAUSDT)",
         "open_cluster_cap"),
        ("weak 1h impulse: ADX 17.2 < 20.0", "impulse_guard"),
        ("time block: UTC hour 3 filtered", "time_block"),
        ("late 1h continuation: RSI 81, price 6% > EMA50", "late_continuation"),
        ("late impulse_speed rotation guard: RSI 84 >= 80", "late_impulse_rotation"),
    ]

    def test_observed_reasons_classify(self):
        for reason, expected in self.OBSERVED:
            with self.subTest(reason=reason[:40]):
                self.assertEqual(BR.normalize_block_reason(reason), expected)

    def test_cyrillic_and_mangled_variants_do_not_escape(self):
        # `MTF: 1м` uses a Cyrillic м while `MTF: 1m retest` uses Latin m, and
        # a cp1251 write produced the `????????` prefix on 449 rows.
        for reason in (
            "MTF: 1м MACD hist=-0.002 <= floor -0.001 (коррекция)",
            "MTF: 1м MACD hist=-2e-05 ≤ -1e-05 (коррекция)",
            "MTF: 15m deep correction: MACD=-0.01 <= hard floor -0.005, RSI=41 < 45",
        ):
            with self.subTest(reason=reason[:32]):
                self.assertEqual(BR.normalize_block_reason(reason), "mtf")
        self.assertEqual(
            BR.normalize_block_reason("????????: портфель полон: 6/6 позиций"),
            "portfolio_full",
        )

    def test_unknown_stays_visible(self):
        # An unmatched reason must not be folded into a neighbouring bucket:
        # it is the signal that a gate changed its wording.
        self.assertEqual(BR.normalize_block_reason("что-то совершенно новое"),
                         BR.UNKNOWN)
        self.assertEqual(BR.normalize_block_reason(""), BR.UNKNOWN)

    def test_codes_are_unique(self):
        codes = BR.known_codes()
        self.assertEqual(len(codes), len(set(codes)))


class TestWhyNoSignalVerdicts(unittest.TestCase):
    """Silence must not be reported as a monitoring failure on its own."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "bot_events.jsonl"
        self._orig_events, self._orig_wl = W.EVENTS, W.WATCHLIST
        W.EVENTS = self.path
        W.WATCHLIST = Path(self.tmp.name) / "watchlist.json"
        W.WATCHLIST.write_text(json.dumps(["AAAUSDT", "BBBUSDT"]), encoding="utf-8")

    def tearDown(self):
        W.EVENTS, W.WATCHLIST = self._orig_events, self._orig_wl

    def _write(self, rows):
        with self.path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    @staticmethod
    def _ts(minutes_ago: int) -> str:
        return (datetime.now(timezone.utc)
                - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def test_blocked_symbol_names_the_dominant_gate(self):
        self._write([
            {"event": "blocked", "sym": "AAAUSDT", "ts": self._ts(30),
             "reason": "trend quality guard: RSI 73.2 > 72.0"},
            {"event": "blocked", "sym": "AAAUSDT", "ts": self._ts(20),
             "reason": "trend quality guard: RSI 74.0 > 72.0"},
            {"event": "blocked", "sym": "AAAUSDT", "ts": self._ts(10),
             "reason": "entry score 27.69 < floor 35.00"},
        ])
        out = W.report("AAAUSDT", hours=6, top=5)
        self.assertEqual(out["verdict"], "blocked:trend_quality")
        self.assertEqual(out["n_blocked"], 3)

    def test_quiet_symbol_while_bot_works_is_not_called_a_monitoring_bug(self):
        rows = [{"event": "blocked", "sym": "BBBUSDT", "ts": self._ts(i),
                 "reason": "entry score 1 < floor 35"} for i in range(1, 40)]
        self._write(rows)
        out = W.report("AAAUSDT", hours=6, top=5)
        self.assertEqual(out["verdict"], "silent_bot_alive")
        self.assertGreater(out["other_events_in_window"], 0)

    def test_total_silence_is_reported_as_a_dead_process(self):
        self._write([{"event": "blocked", "sym": "BBBUSDT", "ts": self._ts(60 * 48),
                      "reason": "entry score 1 < floor 35"}])
        out = W.report("AAAUSDT", hours=6, top=5)
        self.assertEqual(out["verdict"], "bot_silent")

    def test_symbol_outside_watchlist_is_expected_not_a_fault(self):
        self._write([{"event": "blocked", "sym": "BBBUSDT", "ts": self._ts(5),
                      "reason": "entry score 1 < floor 35"}])
        out = W.report("ZZZUSDT", hours=6, top=5)
        self.assertEqual(out["verdict"], "not_in_watchlist")


if __name__ == "__main__":
    unittest.main()
