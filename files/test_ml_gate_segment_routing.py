"""Guards for the 2026-08-20 ml_zone blackout fix.

That day the gate admitted ZERO of 4486 candidates while the market rose. XRP
(+19% after being blocked), ORDI (+18.8%) and ENA (+17.7%) were all rejected,
and 51% of the 83 blocked coins went on to rise more than 3%.

The cause was NOT the nightly retrain and NOT the threshold. It was per-segment
routing: the deployed `trend|bull` segment scored that day's candidates at a
median 0.0713 where the global model gave 0.2509 -- 6% above the floor against
69%. Segment models train on subsets as small as 160 rows and are kept whenever
they beat the baseline on their own validation, a bar a small sample clears by
luck.

Each test below pins one piece of that story so it cannot quietly come back.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config  # noqa: E402
import ml_signal_model as M  # noqa: E402

MSM_SRC = (HERE / "ml_signal_model.py").read_text(encoding="utf-8")
CFG_SRC = (HERE / "config.py").read_text(encoding="utf-8")


class TestRoutingIsSwitchable(unittest.TestCase):
    def test_the_flag_exists_and_is_off(self):
        self.assertFalse(getattr(config, "ML_SIGNAL_SEGMENT_ROUTING_ENABLED", True))

    def test_the_scoring_path_actually_consults_it(self):
        # A flag the predictor ignores is the failure this whole file exists for.
        self.assertIn("if seg_map and _segment_routing_enabled():", MSM_SRC)

    def test_it_fails_closed_to_the_global_model(self):
        # If config cannot be imported, fall back to the GLOBAL model rather than
        # to routing -- the global model is the conservative side here, because
        # routing is what produced a total blackout.
        self.assertIn("except Exception:\n        return False", MSM_SRC)

    def test_the_rollback_is_written_down(self):
        self.assertIn("Rollback = True.", CFG_SRC)


class TestRoutingActuallyChangesScoring(unittest.TestCase):
    """Behavioural, not textual: build a payload whose segment model disagrees
    hard with the global one and check the flag decides which is used."""

    def setUp(self):
        names = M.safe_feature_names()
        n = len(names)
        self.payload = {
            "model_name": "logistic",
            "feature_names": names,
            "scaler_mean": [0.0] * n,
            "scaler_scale": [1.0] * n,
            "threshold": 0.5,
            "positive_ret_threshold": 0.0,
            # global: strongly positive bias -> high proba
            "model": {"type": "logistic", "weights": [0.0] * n, "bias": 3.0},
            "segment_model_payloads": {
                "trend|bull": {
                    "model_name": "logistic",
                    "feature_names": names,
                    "scaler_mean": [0.0] * n,
                    "scaler_scale": [1.0] * n,
                    "threshold": 0.5,
                    "positive_ret_threshold": 0.0,
                    # segment: strongly negative -> the 08-20 pathology, in miniature
                    "model": {"type": "logistic", "weights": [0.0] * n, "bias": -3.0},
                }
            },
        }
        self.rec = {"signal_type": "trend", "is_bull_day": True}
        self._prev = config.ML_SIGNAL_SEGMENT_ROUTING_ENABLED

    def tearDown(self):
        config.ML_SIGNAL_SEGMENT_ROUTING_ENABLED = self._prev

    def test_routing_off_uses_the_global_model(self):
        config.ML_SIGNAL_SEGMENT_ROUTING_ENABLED = False
        self.assertGreater(M.predict_proba_from_payload(self.payload, self.rec), 0.9)

    def test_routing_on_uses_the_segment_model(self):
        config.ML_SIGNAL_SEGMENT_ROUTING_ENABLED = True
        self.assertLess(M.predict_proba_from_payload(self.payload, self.rec), 0.1)

    def test_an_unrouted_segment_still_falls_back_to_global(self):
        config.ML_SIGNAL_SEGMENT_ROUTING_ENABLED = True
        other = {"signal_type": "retest", "is_bull_day": False}
        self.assertGreater(M.predict_proba_from_payload(self.payload, other), 0.9)


class TestFloorsWereLoweredTogether(unittest.TestCase):
    """The backtest tested ONE floor. Leaving the bull/non-bull gap in place
    would smuggle a second, untested change in alongside this one."""

    def test_both_floors_are_ten_percent(self):
        self.assertAlmostEqual(config.ML_GENERAL_HARD_BLOCK_MIN, 0.10, places=6)
        self.assertAlmostEqual(config.ML_GENERAL_HARD_BLOCK_BULL_DAY_MIN, 0.10, places=6)

    def test_the_upper_cap_still_lets_confident_signals_through(self):
        # 1.01 means "no upper bound". A cap below 1.0 once blocked exactly the
        # high-confidence signals the bot exists to emit (CLAUDE.md section 7).
        self.assertGreaterEqual(config.ML_GENERAL_HARD_BLOCK_MAX, 1.0)

    def test_the_evidence_travels_with_the_number(self):
        for token in ("XRP", "4486", "top-40"):
            self.assertIn(token, CFG_SRC,
                          "the config note lost the evidence for this change")


class TestTheLIVEPathIsTheOneSwitchedOff(unittest.TestCase):
    """The first attempt at this fix set the wrong flag and nothing changed.

    monitor.py::_select_ml_payload chooses the segment model and hands the
    already-chosen payload to predict_proba_from_payload, so the guard inside
    that function never sees a `segment_model_payloads` key and never fires.
    Live median proba after that edit: 0.0445, against 0.0569 before -- no move.
    ML_GENERAL_USE_SEGMENT_WHEN_AVAILABLE is the switch the bot actually reads.
    """

    def test_the_live_switch_is_off(self):
        self.assertFalse(config.ML_GENERAL_USE_SEGMENT_WHEN_AVAILABLE)

    def test_the_selector_consults_that_switch(self):
        mon = (HERE / "monitor.py").read_text(encoding="utf-8")
        self.assertIn('getattr(config, "ML_GENERAL_USE_SEGMENT_WHEN_AVAILABLE", True)', mon)

    def test_the_selector_falls_back_to_the_general_payload(self):
        mon = (HERE / "monitor.py").read_text(encoding="utf-8")
        sel = mon.split("def _select_ml_payload")[1].split(chr(10) + "def ")[0]
        self.assertIn("return _load_ml_general_payload()", sel)

    def test_both_switches_agree(self):
        # Two layers, one intent. Leaving them in disagreement is how the first
        # attempt looked correct in tests and did nothing in production.
        self.assertEqual(bool(config.ML_GENERAL_USE_SEGMENT_WHEN_AVAILABLE),
                         bool(config.ML_SIGNAL_SEGMENT_ROUTING_ENABLED))


if __name__ == "__main__":
    unittest.main(verbosity=2)
