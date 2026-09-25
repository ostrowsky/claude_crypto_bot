"""Guards for two refuted exit/entry hypotheses of 2026-09-25 (QNT).

Both verdicts say "change nothing", which nobody re-checks, so the counting is
pinned: the replay must lift EVERY lateness cap strategy.py reads, and the
partial-exit arithmetic must stay linear in f.

Spec: docs/specs/features/lateness-caps-and-rsi-partial-exit-spec.md
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _backtest_lateness_caps as L  # noqa: E402

STRATEGY = (HERE / "strategy.py").read_text(encoding="utf-8")
LATE = (HERE / "_backtest_lateness_caps.py").read_text(encoding="utf-8")
PARTIAL = (HERE / "_backtest_rsi_exit_partial.py").read_text(encoding="utf-8")


class TestEveryLatenessCapIsLifted(unittest.TestCase):
    def test_all_range_caps_strategy_reads_are_in_the_replay(self):
        # A cap missed here would keep blocking in the "no caps" run and make
        # LATE look smaller than it is -- a verdict that could not be trusted.
        keys = set(re.findall(r'getattr\(config,\s*"([A-Z_0-9]*RANGE_MAX)"', STRATEGY))
        keys |= set(re.findall(r"config\.(DAILY_RANGE_MAX)", STRATEGY))
        keys |= {"_effective_range_max"}
        # BULL_DAY_RANGE_MAX is not a cap strategy.py compares against: analyze_coin
        # copies it into _effective_range_max on a bull day, and the replay sets
        # _effective_range_max itself (CAP10 = that value, NOCAP = lifted).
        covered_via_effective = {"BULL_DAY_RANGE_MAX"}
        missing = sorted(k for k in keys if k not in L.CAP_KEYS
                         and k not in covered_via_effective and not k.startswith("TREND_15M"))
        self.assertEqual(missing, [], "lateness caps not lifted in the NOCAP run")

    def test_the_bull_day_band_is_separated_from_late(self):
        self.assertIn('"CAP10"', LATE)
        self.assertIn("BULL_DAY_RANGE_MAX", LATE)

    def test_the_five_live_rules_are_evaluated(self):
        for fn in ("check_entry_conditions", "check_trend_surge_conditions",
                   "check_impulse_conditions", "check_alignment_conditions",
                   "check_ema_cross_conditions"):
            self.assertIn(fn, LATE)


class TestTheVerdictsTravelWithTheFiles(unittest.TestCase):
    def test_lateness_verdict(self):
        self.assertIn("VERDICT 2026-09-25: REFUTED as a lever", LATE)
        for tok in ("120 542", "2 744", "819 (98.8%)", "9.87x", "vol_x 1.01"):
            self.assertIn(tok, LATE)

    def test_partial_exit_verdict(self):
        self.assertIn("VERDICT 2026-09-25: NOT SUPPORTED", PARTIAL)
        for tok in ("+0.307%", "[-0.38, +1.09]", "3 of 7"):
            self.assertIn(tok, PARTIAL)


class TestPartialExitIsLinear(unittest.TestCase):
    def test_delta_is_f_times_trail_minus_actual(self):
        # the property the verdict rests on: f resizes the bet, never flips it
        pairs = [(5.0, 9.0), (6.0, 2.0), (3.0, 3.5)]
        for f in (0.25, 0.5, 0.75):
            for a, t in pairs:
                blended = (1 - f) * a + f * t
                self.assertAlmostEqual(blended - a, f * (t - a), places=12)
        self.assertIn("p = [(1 - f) * a + f * c for a, c, _, _ in rb]", PARTIAL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
