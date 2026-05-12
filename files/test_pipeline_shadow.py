"""Unit tests for pipeline_shadow simulators (L4 counterfactual handlers).

Run:
    pyembed\\python.exe files\\test_pipeline_shadow.py

We test the SIM_HANDLERS in isolation by feeding them a synthetic
critic_dataset.jsonl in a temp directory and a fake hypothesis. The point is
to lock down the filter logic (which rows count as evidence) and the
emitted shape (canonical shadow events) without depending on real bot data.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pipeline_shadow as S
import pipeline_lib as PL


@contextmanager
def temp_critic(rows: list[dict]):
    """Write a temp critic_dataset.jsonl and patch S.CRITIC_DATASET."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "critic_dataset.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with mock.patch.object(S, "CRITIC_DATASET", p):
            yield p


def _row(*, signal_type: str, action: str, ret_5: float | None, bar_ts: str | None,
         reason_code: str | None = None, sym: str = "TESTUSDT") -> dict:
    """Construct a minimal critic_dataset row matching the production schema."""
    return {
        "id":          f"{sym}-{bar_ts}",
        "sym":         sym,
        "bar_ts":      bar_ts,
        "ts_signal":   bar_ts,
        "signal_type": signal_type,
        "decision":    {"action": action, "reason_code": reason_code or ""},
        "labels":      {"ret_5": ret_5, "trade_taken": action == "take",
                        "trade_exit_pnl": None, "label_5": None},
    }


def _today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _days_ago_iso(d: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=d)).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# sim_disable_mode
# ---------------------------------------------------------------------------


class SimDisableModeTests(unittest.TestCase):

    def _hyp(self, mode: str) -> dict:
        return {
            "hypothesis_id": f"h-test-disable_mode_{mode}",
            "rule":          f"disable_mode_{mode}",
        }

    def test_filters_to_target_mode_only(self):
        rows = [
            _row(signal_type="impulse_speed", action="take", ret_5=-0.5, bar_ts=_today_iso()),
            _row(signal_type="alignment",     action="take", ret_5=+1.0, bar_ts=_today_iso()),
            _row(signal_type="trend",         action="take", ret_5=+0.2, bar_ts=_today_iso()),
        ]
        with temp_critic(rows):
            ev = S.sim_disable_mode(self._hyp("impulse_speed"), window_days=30)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["ctx"]["mode"], "impulse_speed")

    def test_take_only_blocked_skipped(self):
        rows = [
            _row(signal_type="impulse_speed", action="take",    ret_5=-1.0, bar_ts=_today_iso()),
            _row(signal_type="impulse_speed", action="blocked", ret_5=+3.0, bar_ts=_today_iso()),
            _row(signal_type="impulse_speed", action="candidate", ret_5=+2.0, bar_ts=_today_iso()),
        ]
        with temp_critic(rows):
            ev = S.sim_disable_mode(self._hyp("impulse_speed"), window_days=30)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["ctx"]["prod_pnl_pct"], -1.0)

    def test_window_filtering(self):
        rows = [
            _row(signal_type="impulse_speed", action="take", ret_5=-0.5, bar_ts=_days_ago_iso(5)),
            _row(signal_type="impulse_speed", action="take", ret_5=+0.8, bar_ts=_days_ago_iso(40)),
        ]
        with temp_critic(rows):
            ev = S.sim_disable_mode(self._hyp("impulse_speed"), window_days=30)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["ctx"]["prod_pnl_pct"], -0.5)

    def test_canonical_shape(self):
        rows = [_row(signal_type="impulse_speed", action="take", ret_5=-0.4, bar_ts=_today_iso())]
        with temp_critic(rows):
            ev = S.sim_disable_mode(self._hyp("impulse_speed"), window_days=30)
        e = ev[0]
        self.assertEqual(e["event"], "shadow")
        self.assertEqual(e["feature_flag"], "DISABLE_MODE_IMPULSE_SPEED")
        self.assertEqual(e["prod_decision"], "take")
        self.assertEqual(e["shadow_decision"], "skip")
        self.assertEqual(e["ctx"]["shadow_pnl_pct"], 0.0)
        self.assertEqual(e["ctx"]["prod_pnl_pct"], -0.4)

    def test_wrong_rule_returns_empty(self):
        rows = [_row(signal_type="impulse_speed", action="take", ret_5=-0.4, bar_ts=_today_iso())]
        with temp_critic(rows):
            ev = S.sim_disable_mode({"rule": "relax_gate_foo"}, window_days=30)
        self.assertEqual(ev, [])

    def test_missing_ret5_skipped(self):
        rows = [_row(signal_type="impulse_speed", action="take", ret_5=None, bar_ts=_today_iso())]
        with temp_critic(rows):
            ev = S.sim_disable_mode(self._hyp("impulse_speed"), window_days=30)
        self.assertEqual(ev, [])

    def test_prefers_trade_exit_pnl_when_present(self):
        row = _row(signal_type="impulse_speed", action="take", ret_5=-0.4, bar_ts=_today_iso())
        row["labels"]["trade_exit_pnl"] = -2.5
        with temp_critic([row]):
            ev = S.sim_disable_mode(self._hyp("impulse_speed"), window_days=30)
        self.assertEqual(ev[0]["ctx"]["prod_pnl_pct"], -2.5)


# ---------------------------------------------------------------------------
# sim_widen_watchlist_match_tolerance
# ---------------------------------------------------------------------------


class SimWidenWatchlistTests(unittest.TestCase):

    _HYP = {
        "hypothesis_id": "h-test-widen_watchlist",
        "rule":          "widen_watchlist_match_tolerance",
    }

    def test_picks_only_blocked_by_score_reasons(self):
        rows = [
            _row(signal_type="trend", action="blocked", ret_5=+1.5,
                 bar_ts=_today_iso(), reason_code="entry_score_below_floor"),
            _row(signal_type="trend", action="blocked", ret_5=+2.0,
                 bar_ts=_today_iso(), reason_code="watchlist_no_match"),
            _row(signal_type="trend", action="blocked", ret_5=+5.0,
                 bar_ts=_today_iso(), reason_code="trend_quality"),
            _row(signal_type="trend", action="take",    ret_5=+1.0,
                 bar_ts=_today_iso()),
        ]
        with temp_critic(rows):
            ev = S.sim_widen_watchlist_match_tolerance(self._HYP, window_days=30)
        self.assertEqual(len(ev), 2)
        flags = {e["feature_flag"] for e in ev}
        self.assertEqual(flags, {"WIDEN_WATCHLIST_MATCH_TOLERANCE"})

    def test_caveat_present(self):
        rows = [_row(signal_type="trend", action="blocked", ret_5=+1.0,
                     bar_ts=_today_iso(), reason_code="entry_score_low")]
        with temp_critic(rows):
            ev = S.sim_widen_watchlist_match_tolerance(self._HYP, window_days=30)
        self.assertEqual(len(ev), 1)
        self.assertIn("upper-bound", ev[0]["_caveat"])

    def test_canonical_shape(self):
        rows = [_row(signal_type="trend", action="blocked", ret_5=+1.2,
                     bar_ts=_today_iso(), reason_code="entry_score_below_floor")]
        with temp_critic(rows):
            ev = S.sim_widen_watchlist_match_tolerance(self._HYP, window_days=30)
        e = ev[0]
        self.assertEqual(e["event"], "shadow")
        self.assertEqual(e["prod_decision"], "skip")
        self.assertEqual(e["shadow_decision"], "take")
        self.assertEqual(e["ctx"]["prod_pnl_pct"], 0.0)
        self.assertEqual(e["ctx"]["shadow_pnl_pct"], 1.2)

    def test_window_filtering(self):
        rows = [
            _row(signal_type="trend", action="blocked", ret_5=+1.5,
                 bar_ts=_days_ago_iso(10), reason_code="entry_score_low"),
            _row(signal_type="trend", action="blocked", ret_5=+2.0,
                 bar_ts=_days_ago_iso(40), reason_code="entry_score_low"),
        ]
        with temp_critic(rows):
            ev = S.sim_widen_watchlist_match_tolerance(self._HYP, window_days=30)
        self.assertEqual(len(ev), 1)

    def test_wrong_rule_returns_empty(self):
        rows = [_row(signal_type="trend", action="blocked", ret_5=+1.0,
                     bar_ts=_today_iso(), reason_code="score")]
        with temp_critic(rows):
            ev = S.sim_widen_watchlist_match_tolerance({"rule": "disable_mode_foo"}, window_days=30)
        self.assertEqual(ev, [])

    def test_no_dataset_returns_empty(self):
        # Patch path to non-existent file
        with mock.patch.object(S, "CRITIC_DATASET", Path("/nonexistent/path.jsonl")):
            ev = S.sim_widen_watchlist_match_tolerance(self._HYP, window_days=30)
        self.assertEqual(ev, [])


# ---------------------------------------------------------------------------
# find_simulator
# ---------------------------------------------------------------------------


class FindSimulatorTests(unittest.TestCase):

    def test_disable_mode_prefix_match(self):
        h = S.find_simulator("disable_mode_impulse_speed")
        self.assertIs(h, S.sim_disable_mode)

    def test_widen_watchlist_match(self):
        h = S.find_simulator("widen_watchlist_match_tolerance")
        self.assertIs(h, S.sim_widen_watchlist_match_tolerance)

    def test_unknown_returns_none(self):
        self.assertIsNone(S.find_simulator("tighten_proba_alignment"))
        self.assertIsNone(S.find_simulator(""))


# ---------------------------------------------------------------------------
# aggregate + verdict over sim output (end-to-end behaviour)
# ---------------------------------------------------------------------------


class AggregateAndVerdictTests(unittest.TestCase):

    def test_disable_mode_losing_yields_accept(self):
        # All taken impulse_speed entries lose money → shadow (skip) wins
        rows = [
            _row(signal_type="impulse_speed", action="take", ret_5=-0.5, bar_ts=_today_iso()),
            _row(signal_type="impulse_speed", action="take", ret_5=-1.2, bar_ts=_today_iso()),
            _row(signal_type="impulse_speed", action="take", ret_5=-0.3, bar_ts=_today_iso()),
            _row(signal_type="impulse_speed", action="take", ret_5=-0.8, bar_ts=_today_iso()),
        ]
        with temp_critic(rows):
            events = S.sim_disable_mode({"hypothesis_id": "h", "rule": "disable_mode_impulse_speed"}, 30)
        summary = S.aggregate(events)
        verdict = S.verdict(summary, min_events=3, min_delta_pct=0.0)
        v = verdict["DISABLE_MODE_IMPULSE_SPEED"]
        self.assertEqual(v["verdict"], "accept")
        self.assertGreater(summary["by_feature"]["DISABLE_MODE_IMPULSE_SPEED"]["delta_median_pnl_pct"], 0)

    def test_disable_mode_winning_yields_reject(self):
        # All taken impulse_speed entries make money → shadow (skip) loses
        rows = [
            _row(signal_type="impulse_speed", action="take", ret_5=+2.5, bar_ts=_today_iso()),
            _row(signal_type="impulse_speed", action="take", ret_5=+1.2, bar_ts=_today_iso()),
            _row(signal_type="impulse_speed", action="take", ret_5=+0.8, bar_ts=_today_iso()),
            _row(signal_type="impulse_speed", action="take", ret_5=+1.5, bar_ts=_today_iso()),
        ]
        with temp_critic(rows):
            events = S.sim_disable_mode({"hypothesis_id": "h", "rule": "disable_mode_impulse_speed"}, 30)
        verdict = S.verdict(S.aggregate(events), min_events=3, min_delta_pct=0.0)
        self.assertEqual(verdict["DISABLE_MODE_IMPULSE_SPEED"]["verdict"], "reject")

    def test_insufficient_events_yields_insufficient(self):
        rows = [_row(signal_type="impulse_speed", action="take", ret_5=-0.5, bar_ts=_today_iso())]
        with temp_critic(rows):
            events = S.sim_disable_mode({"hypothesis_id": "h", "rule": "disable_mode_impulse_speed"}, 30)
        verdict = S.verdict(S.aggregate(events), min_events=3, min_delta_pct=0.0)
        self.assertEqual(verdict["DISABLE_MODE_IMPULSE_SPEED"]["verdict"], "insufficient_data")


if __name__ == "__main__":
    unittest.main(verbosity=2)
