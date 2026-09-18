"""Guards for the L3 replay validator and the loop repairs of 2026-09-18.

Eleven weekly runs produced zero automatic verdicts because L3 matched on a
rule name L2 invents freely. These tests pin the three things that closed the
loop, and the two mistakes the first version of the validator made.

Spec: docs/specs/features/l3-replay-validator-spec.md
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pipeline_replay_validator as V  # noqa: E402
import pipeline_validator as PV  # noqa: E402


def cfg(**over):
    base = dict(ML_GENERAL_HARD_BLOCK_MIN=0.15, ML_GENERAL_HARD_BLOCK_BULL_DAY_MIN=0.15,
                TREND_15M_QUALITY_FORECAST_MIN=0.25, TREND_15M_QUALITY_ALT_VOL_MIN=1.2,
                TREND_15M_QUALITY_ALT_ADX_MIN=24.0, TREND_15M_QUALITY_ALT_SLOPE_MIN=0.35,
                TREND_1H_CHOP_ADX_MIN=25.0, TREND_1H_CHOP_SLOPE_MIN=0.7,
                TREND_1H_CHOP_VOL_MIN=1.3, TREND_1H_CHOP_ADX_MIN_BULL_DAY=22.0,
                TREND_1H_CHOP_SLOPE_MIN_BULL_DAY=1.0, TREND_1H_CHOP_VOL_MIN_BULL_DAY=1.2,
                TREND_1H_CHOP_USE_BULL_DAY_RELAX=True, COOLDOWN_BARS=19)
    base.update(over)
    return types.SimpleNamespace(**base)


class Synthetic(unittest.TestCase):
    """A private event log and a deterministic outcome, so the rule is tested
    and not the market."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.prev_events, self.prev_peak = V.EVENTS, V.forward_peak
        V.EVENTS = Path(self.tmp.name) / "bot_events.jsonl"
        self.peak_of = {}
        V.forward_peak = lambda sym, tf, dt, px, hours=4: self.peak_of.get(sym)

    def tearDown(self):
        V.EVENTS, V.forward_peak = self.prev_events, self.prev_peak
        self.tmp.cleanup()

    def write(self, events):
        with io.open(V.EVENTS, "w", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

    def ml_event(self, i, proba, blocked, sym, when="2026-09-10"):
        t = datetime.fromisoformat(when + "T00:00:00+00:00") + timedelta(hours=i)
        e = {"ts": t.isoformat().replace("+00:00", "Z"), "sym": sym, "price": 1.0,
             "tf": "15m", "ml_proba": proba, "is_bull_day": False}
        if blocked:
            e.update(event="blocked", reason_code="ml_zone")
        else:
            e.update(event="blocked", reason_code="mode_range_quality")
        return e


class TestDirectionComesFromTheLoggedValues(Synthetic):
    def test_raising_a_floor_is_a_tighten_whatever_happened_on_the_day(self):
        # The first version used the day's decision: rows blocked under an OLD,
        # higher floor looked "admitted" by 0.30 and raising the floor was graded
        # as a relax -- and accepted. Here some rows were blocked on the day at
        # proba 0.20 (an old floor), which today's 0.15 already admits.
        ev = [self.ml_event(i, 0.20, True, "OLD%d" % i) for i in range(40)]
        ev += [self.ml_event(100 + i, 0.25, False, "MID%d" % i) for i in range(40)]
        ev += [self.ml_event(200 + i, 0.50, False, "HI%d" % i) for i in range(40)]
        self.write(ev)
        for i in range(40):
            self.peak_of["OLD%d" % i] = 1.0
            self.peak_of["MID%d" % i] = 1.0
            self.peak_of["HI%d" % i] = 1.0
        r = V.validate({"config_key": "ML_GENERAL_HARD_BLOCK_MIN", "diff": {"to": 0.30}},
                       cfg_module=cfg())
        self.assertEqual(r["direction"], "tighten")

    def test_rows_before_the_scale_change_are_ignored_for_ml(self):
        ev = [self.ml_event(i, 0.12, True, "A%d" % i, when="2026-08-01") for i in range(80)]
        self.write(ev)
        for i in range(80):
            self.peak_of["A%d" % i] = 9.0
        r = V.validate({"config_key": "ML_GENERAL_HARD_BLOCK_MIN", "diff": {"to": 0.10}},
                       cfg_module=cfg())
        self.assertEqual(r["verdict"], "needs_data")   # nothing after 2026-09-07


class TestTheDecisionRule(Synthetic):
    def build(self, band_peak, ctrl_peak, n=80):
        ev = [self.ml_event(i, 0.12, True, "B%d" % i) for i in range(n)]
        ev += [self.ml_event(200 + i, 0.40, False, "C%d" % i) for i in range(n)]
        self.write(ev)
        for i in range(n):
            self.peak_of["B%d" % i] = band_peak
            self.peak_of["C%d" % i] = ctrl_peak

    def test_a_worse_band_is_rejected(self):
        # the control must carry a >=5% tail, or the tail ratio is 1.0 and the
        # rule -- correctly -- calls it needs_review rather than a clear loss
        self.build(0.5, 6.0)
        r = V.validate({"config_key": "ML_GENERAL_HARD_BLOCK_MIN", "diff": {"to": 0.10}},
                       cfg_module=cfg())
        self.assertEqual((r["direction"], r["verdict"]), ("relax", "reject"))

    def test_a_better_band_is_accepted_not_applied(self):
        self.build(6.0, 1.0)
        r = V.validate({"config_key": "ML_GENERAL_HARD_BLOCK_MIN", "diff": {"to": 0.10}},
                       cfg_module=cfg())
        self.assertEqual(r["verdict"], "accept")

    def test_thin_data_is_needs_data_never_pending_manual(self):
        self.build(6.0, 1.0, n=10)
        r = V.validate({"config_key": "ML_GENERAL_HARD_BLOCK_MIN", "diff": {"to": 0.10}},
                       cfg_module=cfg())
        self.assertEqual(r["verdict"], "needs_data")


class TestNothingParks(unittest.TestCase):
    def test_unknown_key_is_rejected(self):
        r = V.validate({"config_key": "GATE_ENTRY_SCORE_THRESHOLD", "diff": {"to": 1}},
                       cfg_module=cfg())
        self.assertEqual(r["verdict"], "reject")
        self.assertIn("does not exist", r["reason"])

    def test_a_real_key_with_no_replay_is_rejected_as_unvalidatable(self):
        r = V.validate({"config_key": "COOLDOWN_BARS", "diff": {"to": 10}},
                       cfg_module=cfg())
        self.assertEqual(r["verdict"], "reject")
        self.assertIn("unvalidatable", r["reason"])

    def test_dispatch_never_returns_pending_manual_for_an_unregistered_rule(self):
        # the exact shape of the hypothesis that waited from 08-16 to 09-18
        out = PV.dispatch({"rule": "some_name_l2_invented", "config_key": "NOPE_KEY",
                           "diff": {"from": 1, "to": 2}})
        self.assertNotEqual(out.get("verdict"), "pending_manual_validation")

    def test_one_direction_specs_refuse_the_other(self):
        r = V.validate({"config_key": "TREND_1H_CHOP_ADX_MIN", "diff": {"to": 28.0}},
                       cfg_module=cfg())
        self.assertEqual(r["verdict"], "reject")
        self.assertIn("only be replayed as", r["reason"])


class TestMatchedControl(unittest.TestCase):
    def test_gate_specific_specs_compare_against_their_own_mode(self):
        # The first run compared trend-only chop rejects with every 1h mode that
        # passed, and graded relaxing ADX 25 -> 20 as ACCEPT.
        for key in ("TREND_1H_CHOP_ADX_MIN", "TREND_15M_QUALITY_FORECAST_MIN"):
            self.assertEqual(V.REPLAY_SPECS[key].control_mode, "trend", key)

    def test_ml_uses_every_row_because_the_gate_judges_every_mode(self):
        self.assertEqual(V.REPLAY_SPECS["ML_GENERAL_HARD_BLOCK_MIN"].control_mode, "")
        self.assertTrue(V.REPLAY_SPECS["ML_GENERAL_HARD_BLOCK_MIN"].valid_since)


class TestGuardsMatchTheLiveCode(unittest.TestCase):
    def test_trend_quality_predicate(self):
        e = {"reason": "trend quality guard: weak 15m trend "
                       "(forecast 0.000 < 0.250, vol 1.50, ADX 22.0, slope 0.400)"}
        self.assertTrue(V._tq_blocks(e, vars(cfg())))                    # ADX < 24
        self.assertFalse(V._tq_blocks(e, vars(cfg(TREND_15M_QUALITY_ALT_ADX_MIN=20.0))))
        self.assertIsNone(V._tq_blocks({"reason": "price edge 5% > 4%"}, vars(cfg())))

    def test_chop_uses_the_bull_day_thresholds(self):
        e = {"adx": 23.0, "slope_pct": 1.1, "vol_x": 1.25, "is_bull_day": True}
        self.assertFalse(V._chop_blocks(e, vars(cfg())))
        e["is_bull_day"] = False
        self.assertTrue(V._chop_blocks(e, vars(cfg())))


class TestTheLoopCanSeeItself(unittest.TestCase):
    def test_l2_is_told_which_keys_l3_can_test(self):
        src = (HERE / "pipeline_hypothesis.py").read_text(encoding="utf-8")
        self.assertIn('"validatable_config_keys": _validatable_with_values()', src)
        self.assertIn("Propose ONLY config_keys listed in validatable_config_keys", src)

    def test_failed_llm_calls_are_not_reported_as_calls(self):
        import bot_health_report as R
        calls = [{"ts": "2026-08-23T03:00:00Z", "purpose": "weekly_generation", "error": None},
                 {"ts": "2026-09-06T03:00:00Z", "purpose": "weekly_generation",
                  "error": "HTTP 400 | Your credit balance is too low to access the Anthropic API"}]
        prev = R.PL.iter_jsonl
        R.PL.iter_jsonl = lambda p: iter(calls)
        try:
            out = R._render_agent_block()
        finally:
            R.PL.iter_jsonl = prev
        self.assertIn("последний УСПЕШНЫЙ вызов 2026-08-23", out)
        self.assertIn("агент НЕ работает", out)
        self.assertIn("кредиты", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
