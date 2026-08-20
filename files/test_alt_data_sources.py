"""Guards for the alternative-data work: funding, and the reproduced 4h rule.

Each test here corresponds to a way one of these numbers could look right and be
wrong. Two of them already did today -- a random split turned funding's
start-vs-middle AUC into 0.846 when the honest figure is 0.606, and an unmatched
symbol set turned a population change into a fake +13pp of recall.

Spec: docs/specs/features/trend-start-detector-spec.md
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _backtest_4h_leader_watch as L        # noqa: E402
import _backtest_funding_start_vs_middle as FU  # noqa: E402

FU_SRC = (HERE / "_backtest_funding_start_vs_middle.py").read_text(encoding="utf-8")
L_SRC = (HERE / "_backtest_4h_leader_watch.py").read_text(encoding="utf-8")
BF_SRC = (HERE / "_backfill_funding_419d.py").read_text(encoding="utf-8")
TD_SRC = (HERE / "_backtest_trend_start_detector.py").read_text(encoding="utf-8")


def bar(i, close, high=None, low=None, vol=100.0, open_=None):
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
    return (ts, open_ if open_ is not None else close,
            high if high is not None else close,
            low if low is not None else close, close, vol)


class TestFundingCannotSeeItsOwnSettlement(unittest.TestCase):
    """Funding posts every 8h. A bar must use the last settlement STRICTLY
    before it -- taking the one that settles later in the same interval would
    hand the model the outcome of the hours it is being asked about."""

    def setUp(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        FU._FUND["TESTUSDT"] = (
            [base + timedelta(hours=8 * k) for k in range(12)],
            [0.0001 * k for k in range(12)],
        )

    def tearDown(self):
        FU._FUND.pop("TESTUSDT", None)

    def test_a_bar_takes_the_previous_settlement_not_the_next(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        # 60h in: settlements at 0,8,...,56 precede it; 56h is index 7.
        f = FU.feats_at("TESTUSDT", base + timedelta(hours=60))
        self.assertAlmostEqual(f["funding"], 0.0001 * 7 * 10000.0, places=6)

    def test_a_bar_exactly_on_a_settlement_uses_the_one_before_it(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        f = FU.feats_at("TESTUSDT", base + timedelta(hours=64))
        self.assertAlmostEqual(f["funding"], 0.0001 * 7 * 10000.0, places=6)

    def test_too_little_history_returns_none_rather_than_a_zero(self):
        # Without this, early rows would read as "no positioning yet" from
        # missing data and look exactly like the hypothesis being tested.
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.assertIsNone(FU.feats_at("TESTUSDT", base + timedelta(hours=8)))

    def test_a_symbol_with_no_funding_file_is_none_not_a_default(self):
        FU._FUND["NOSUCHUSDT"] = ([], [])
        try:
            self.assertIsNone(FU.feats_at("NOSUCHUSDT", datetime.now(timezone.utc)))
        finally:
            FU._FUND.pop("NOSUCHUSDT", None)


class TestFundingSplitIsByTime(unittest.TestCase):
    """This is the test that would have caught today's 0.846.

    Funding is constant across an 8h window, so a random split puts adjacent
    hours of one trend on both sides of the cut and the model memorises
    (symbol, funding value) -> label. The honest figure is 0.606.
    """

    def test_the_split_is_a_day_boundary_not_a_shuffle(self):
        self.assertIn('rows.sort(key=lambda r: r["_day"])', FU_SRC)
        self.assertIn('tr = [r for r in rows if r["_day"] < cut_day]', FU_SRC)
        self.assertIn('te = [r for r in rows if r["_day"] >= cut_day]', FU_SRC)

    def test_no_random_shuffle_survives_before_the_split(self):
        self.assertNotIn("random.Random(0).shuffle(rows)", FU_SRC)

    def test_every_row_carries_the_day_it_belongs_to(self):
        self.assertIn('f["_day"] = bars[i][0].strftime("%Y-%m-%d")', FU_SRC)

    def test_non_trend_bars_are_reported_beside_the_two_classes(self):
        # Without them a separation could be "trends differ from everything"
        # rather than "trend beginnings differ". Non-trend funding landed on the
        # MIDDLE, which is what made the result interesting.
        self.assertIn("NON-TREND", FU_SRC)


class TestFundingBackfillIsSafe(unittest.TestCase):
    def test_the_write_is_atomic(self):
        # A half-written CSV that still parses would silently shorten a symbol's
        # history and change which trends it can be scored against.
        self.assertIn('tmp = out.with_suffix(".csv.part")', BF_SRC)
        self.assertIn("tmp.replace(out)", BF_SRC)

    def test_it_is_resumable_on_both_depth_and_freshness(self):
        self.assertIn("def already_ok(", BF_SRC)
        self.assertIn("oldest_needed_ms", BF_SRC)
        self.assertIn("age_h < 48", BF_SRC)


class TestBothArmsShareOneUniverse(unittest.TestCase):
    """8 of 99 symbols have no funding. Dropping them silently removed 21 trends
    from the denominator and read as +13pp of recall."""

    def test_the_flag_exists_and_applies_without_using_the_features(self):
        self.assertIn('"--funding-universe"', TD_SRC)
        self.assertIn("if args.funding_universe or args.funding:", TD_SRC)

    def test_it_reports_what_it_dropped(self):
        self.assertIn("universe restricted to the %d symbols that have funding", TD_SRC)


class TestFourHLeaderRuleReproduction(unittest.TestCase):
    def test_the_4h_score_uses_the_last_CLOSED_bar_before_now(self):
        # Mirrors i = len(c) - 2 in the original; using the forming bar would
        # let the score see the hour it is scoring.
        self.assertIn("keys[ki + 1] < ts", L_SRC)
        self.assertIn("if keys and keys[ki] < ts", L_SRC)

    def test_the_strength_gate_is_a_conjunction_as_in_the_original(self):
        self.assertIn("strength = today >= 10.0 and vol_x >= 3.0", L_SRC)

    def test_the_strength_only_variant_exists(self):
        # The full rule must be read against its own population, not against
        # random bars: it selects coins already up 10% on 3x volume, so a high
        # hit rate is expected and proves nothing alone.
        self.assertIn('if variant == "strength":', L_SRC)
        self.assertIn("STRENGTH ONLY", L_SRC)

    def test_a_random_baseline_is_reported(self):
        self.assertIn("RANDOM BAR", L_SRC)

    def test_4h_aggregation_lands_on_the_exchange_grid(self):
        bars = [bar(i, 100 + i) for i in range(24)]
        b4 = L.to_4h(bars)
        self.assertTrue(all(x[0].hour % 4 == 0 for x in b4))
        self.assertEqual(len(b4), 6)

    def test_4h_aggregation_keeps_high_low_and_last_close(self):
        bars = [bar(0, 100, high=110, low=90), bar(1, 105, high=120, low=95),
                bar(2, 102, high=104, low=80), bar(3, 108, high=109, low=101)]
        b4 = L.to_4h(bars)
        self.assertEqual(len(b4), 1)
        _, o, h, l, c, v = b4[0]
        self.assertEqual(h, 120)
        self.assertEqual(l, 80)
        self.assertEqual(c, 108)
        self.assertEqual(v, 400.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
