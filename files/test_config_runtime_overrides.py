"""Contracts for the runtime-override channel.

`decisions.jsonl` is the loop's memory and also an execution channel: records in
it are applied to `config.py` globals at import time. That makes two properties
load-bearing.

First, the switch that gates the mechanism must not be reachable *through* the
mechanism — a decision record targeting `AUTO_APPLY_OVERRIDES_ENABLED` would
otherwise let the channel turn itself back on, or leave the audit snapshot
disagreeing with the value the rest of the process reads. A kill switch
reachable by the thing it kills is not a kill switch.

Second, the watchlist and the bot token must never be settable this way, for the
same reason they are immutable everywhere else.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _config_runtime_overrides as RO  # noqa: E402


class TestProtectedKeys(unittest.TestCase):
    def _apply(self, loaded: dict, globals_: dict) -> dict:
        with patch.object(RO, "load_active_overrides", return_value=dict(loaded)), \
             patch.object(RO, "APPLIED_SNAPSHOT", Path("/nonexistent/none.json")):
            return RO.apply_overrides(globals_)

    def test_ordinary_key_is_applied(self):
        g = {"ENTRY_SCORE_MIN_15M": 40.0}
        rec = self._apply({"ENTRY_SCORE_MIN_15M": 35.0}, g)
        self.assertEqual(g["ENTRY_SCORE_MIN_15M"], 35.0)
        self.assertIn("ENTRY_SCORE_MIN_15M", rec["applied"])

    def test_kill_switch_cannot_be_set_through_the_channel(self):
        g = {"AUTO_APPLY_OVERRIDES_ENABLED": False}
        rec = self._apply({"AUTO_APPLY_OVERRIDES_ENABLED": True}, g)
        self.assertIs(g["AUTO_APPLY_OVERRIDES_ENABLED"], False,
                      "the switch gating this mechanism must not be settable by it")
        self.assertIn("AUTO_APPLY_OVERRIDES_ENABLED", rec["refused_protected_key"])
        self.assertNotIn("AUTO_APPLY_OVERRIDES_ENABLED", rec["applied"])

    def test_watchlist_and_token_are_refused(self):
        g = {"DEFAULT_WATCHLIST": ["AAAUSDT"], "TELEGRAM_BOT_TOKEN": "keep"}
        rec = self._apply({"DEFAULT_WATCHLIST": ["EVIL"],
                           "TELEGRAM_BOT_TOKEN": "stolen"}, g)
        self.assertEqual(g["DEFAULT_WATCHLIST"], ["AAAUSDT"])
        self.assertEqual(g["TELEGRAM_BOT_TOKEN"], "keep")
        self.assertEqual(set(rec["refused_protected_key"]),
                         {"DEFAULT_WATCHLIST", "TELEGRAM_BOT_TOKEN"})

    def test_refusal_does_not_block_other_keys_in_the_same_batch(self):
        g = {"AUTO_APPLY_OVERRIDES_ENABLED": True, "ENTRY_SCORE_MIN_15M": 40.0}
        self._apply({"AUTO_APPLY_OVERRIDES_ENABLED": False,
                     "ENTRY_SCORE_MIN_15M": 35.0}, g)
        self.assertIs(g["AUTO_APPLY_OVERRIDES_ENABLED"], True)
        self.assertEqual(g["ENTRY_SCORE_MIN_15M"], 35.0)

    def test_unknown_key_is_reported_not_created(self):
        g = {}
        rec = self._apply({"NO_SUCH_CONSTANT": 1}, g)
        self.assertNotIn("NO_SUCH_CONSTANT", g)
        self.assertIn("NO_SUCH_CONSTANT", rec["config_key_not_present"])

    def test_active_override_is_logged_at_warning(self):
        # An on-disk file silently changing live gating is never routine INFO.
        g = {"ENTRY_SCORE_MIN_15M": 40.0}
        with self.assertLogs(RO.LOG, level="WARNING") as caught:
            self._apply({"ENTRY_SCORE_MIN_15M": 35.0}, g)
        joined = " ".join(caught.output)
        self.assertIn("ENTRY_SCORE_MIN_15M", joined)
        self.assertIn("AUTO_APPLY_OVERRIDES_ENABLED", joined,
                      "the message must name the real switch, not a neighbouring flag")

    def test_protected_set_covers_the_switch(self):
        self.assertIn("AUTO_APPLY_OVERRIDES_ENABLED", RO._NEVER_OVERRIDABLE)


if __name__ == "__main__":
    unittest.main()
