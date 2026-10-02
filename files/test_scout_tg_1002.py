"""trend-scout writes to Telegram only for changes the goal backs (scout-tg-actionable-1002-spec.md)."""
import types
import unittest
from pathlib import Path
from unittest import mock

import config
import trend_scout as TS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _vr(param="COOLDOWN_BARS", cur=19, new=15.0, risk="high"):
    rule = types.SimpleNamespace(risk=risk, is_integer=True)
    p = types.SimpleNamespace(config_param=param, current_value=cur, proposed_value=new,
                              affected_syms=["A"], rule=rule, rationale="")
    return types.SimpleNamespace(proposal=p, verdict="approve", new_entries_count=496,
                                 new_entries_avg_ret5=0.07, new_entries_win_rate=0.5,
                                 new_entries_avg_ret10=0.0, reason="")


def _report(pending_ok):
    return TS.ScoutReport(ts="t", candidates_total=67, candidates_trending=52, entered=4, blocked_trending=48,
                          proposals=[], validated=[_vr()], applied=[], has_findings=False,
                          pending_l3_ok=pending_ok)


class TestTelegramOnlyActionable(unittest.TestCase):
    def test_flag_on_by_default(self):
        self.assertTrue(config.TREND_SCOUT_TG_ONLY_ACTIONABLE)

    def test_unbacked_risky_proposal_is_not_put_to_the_operator(self):
        with mock.patch.object(config, "TREND_SCOUT_TG_ONLY_ACTIONABLE", True):
            self.assertNotIn("Требует подтверждения", TS._build_telegram_text(_report([])))
            txt = TS._build_telegram_text(_report([{"param": "COOLDOWN_BARS", "new_value": 15.0}]))
        self.assertIn("COOLDOWN_BARS", txt)

    def test_flag_off_restores_the_old_report(self):
        with mock.patch.object(config, "TREND_SCOUT_TG_ONLY_ACTIONABLE", False):
            self.assertIn("Требует подтверждения", TS._build_telegram_text(_report([])))

    def test_findings_need_an_application_or_an_l3_accept(self):
        src = (HERE / "trend_scout.py").read_text(encoding="utf-8")
        self.assertIn("findings = bool(applied or pending_l3_ok)", src)
        self.assertIn("has_findings=findings", src)

    def test_cooldown_has_no_goal_replay(self):
        # the proposal that prompted this (COOLDOWN_BARS 19 -> 15) can never be L3-accepted
        r = TS.l3_verdict("COOLDOWN_BARS", 15.0, validators=[({"ML_GENERAL_HARD_BLOCK_MIN"}, None)])
        self.assertEqual(r["verdict"], "no_goal_replay")

    def test_documented(self):
        self.assertTrue((ROOT / "docs/specs/features/scout-tg-actionable-1002-spec.md").exists())
        for doc in ("CLAUDE.md", "PROJECT_CONTEXT.md"):
            self.assertIn("TREND_SCOUT_TG_ONLY_ACTIONABLE", (ROOT / doc).read_text(encoding="utf-8"), doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
