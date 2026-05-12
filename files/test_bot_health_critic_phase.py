"""Tests for bot_health_report.collect_critic — locks in the rule that
EOD (`final`) is always preferred over `midday`.

This is the contract that caused user-visible confusion before:
  - Health report ran at 03:00 local; final wasn't ready yet → used midday
  - Critic also sent Telegram for both phases (~12h apart) with different
    numbers for the same day

The fix combined disabling the critic Telegram (config flag) with making
sure health ALWAYS prefers final when it exists. These tests ensure the
preference logic doesn't regress.

Run:
    pyembed\\python.exe files\\test_bot_health_critic_phase.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bot_health_report as H
import pipeline_lib as PL


@contextmanager
def temp_reports():
    """Yield a temp dir patched into PL.REPORTS so we control which critic
    files exist for the day under test."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        with mock.patch.object(H.PL, "REPORTS", p):
            yield p


def write_critic(reports_dir: Path, day: date, phase: str, summary: dict) -> Path:
    p = reports_dir / f"top_gainer_critic_{day.isoformat()}_{phase}.json"
    p.write_text(json.dumps({"target_day_local": day.isoformat(),
                             "phase": phase,
                             "summary": summary}),
                 encoding="utf-8")
    return p


class CriticPhasePreferenceTests(unittest.TestCase):

    DAY = date(2026, 5, 12)

    def test_returns_unavailable_when_no_files(self):
        with temp_reports():
            r = H.collect_critic(self.DAY)
            self.assertFalse(r["available"])

    def test_uses_midday_when_only_midday_present(self):
        with temp_reports() as t:
            write_critic(t, self.DAY, "midday",
                         {"watchlist_top_bought": 5})
            r = H.collect_critic(self.DAY)
            self.assertTrue(r["available"])
            self.assertEqual(r["data"]["_phase_used"], "midday")
            self.assertEqual(r["data"]["summary"]["watchlist_top_bought"], 5)

    def test_prefers_final_when_both_present(self):
        """The crux of the fix — when both exist, final wins."""
        with temp_reports() as t:
            write_critic(t, self.DAY, "midday",
                         {"watchlist_top_bought": 5,    # stale midday
                          "watchlist_top_count": 15})
            write_critic(t, self.DAY, "final",
                         {"watchlist_top_bought": 11,   # EOD final
                          "watchlist_top_count": 15})
            r = H.collect_critic(self.DAY)
            self.assertEqual(r["data"]["_phase_used"], "final")
            self.assertEqual(r["data"]["summary"]["watchlist_top_bought"], 11)

    def test_uses_final_when_only_final_present(self):
        with temp_reports() as t:
            write_critic(t, self.DAY, "final",
                         {"watchlist_top_bought": 8})
            r = H.collect_critic(self.DAY)
            self.assertEqual(r["data"]["_phase_used"], "final")

    def test_source_file_path_included(self):
        """Helps debugging: collect_critic must record which file it used."""
        with temp_reports() as t:
            write_critic(t, self.DAY, "final", {"x": 1})
            r = H.collect_critic(self.DAY)
            self.assertIn("_source_file", r["data"])
            self.assertTrue(r["data"]["_source_file"].endswith(
                f"top_gainer_critic_{self.DAY.isoformat()}_final.json"
            ))


class CriticTelegramFlagTests(unittest.TestCase):
    """The Telegram-send for critic phases must be controlled by a single
    flag — flipping it off must silence BOTH midday and final."""

    def test_flag_default_is_disabled_after_fix(self):
        """After the 2026-05-12 fix this defaults to False in config.py.
        If someone flips it back to True without updating the spec or the
        operator workflow, this test fails — forcing the discussion."""
        import config
        self.assertFalse(
            getattr(config, "TOP_GAINER_CRITIC_TELEGRAM_REPORTS_ENABLED", True),
            "TOP_GAINER_CRITIC_TELEGRAM_REPORTS_ENABLED should be False — "
            "duplicate Telegram noise was confusing operators. "
            "If enabling, also unify with pipeline_notify."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
