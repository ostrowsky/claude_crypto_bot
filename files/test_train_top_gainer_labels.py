"""Exit gates for training `top_gainer_model` on immutable labels (TH-03).

The property that matters: a training row's label is decided by the close of a
finished UTC day, from exchange data — and a row the store cannot label is
dropped rather than quietly called a non-winner.

Spec: docs/specs/features/top-gainer-immutable-training-labels-spec.md
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import immutable_labels as IL  # noqa: E402
import label_store as LS  # noqa: E402


class _Store(LS.LabelStore):
    def __init__(self, records):
        self._records = records

    def records(self):
        return list(self._records)


def rec(symbol, day, eod, *, complete=True):
    return {"symbol": symbol, "utc_day": day, "eod_return_pct": eod,
            "complete": complete, "qualifies_move5": eod >= 5.0}


def day_rows(day, n=30, top=30.0):
    """Descending returns S00..S{n-1}: S00 = +top%, each next 1pp lower."""
    return [rec(f"S{i:02d}", day, top - i) for i in range(n)]


class TestTierLabels(unittest.TestCase):
    def setUp(self):
        self.store = _Store(day_rows("2026-05-01"))
        self.days = ["2026-05-01"] * 30
        self.syms = [f"S{i:02d}" for i in range(30)]

    def test_rank_and_floor_must_both_hold(self):
        # S00..S09 are ranks 1-10; returns are +30..+21, all above the floor.
        keep, lab, _ = IL.tier_labels(self.days, self.syms, tiers=(10,),
                                      floor=5.0, store=self.store)
        self.assertEqual(len(keep), 30)
        self.assertEqual(sum(lab["top10"]), 10)

    def test_floor_binds_before_the_rank(self):
        # returns +9..-20: only +9,+8,+7,+6,+5 clear the floor (inclusive),
        # though ten rows qualify by rank.
        store = _Store(day_rows("2026-05-01", top=9.0))
        _, lab, _ = IL.tier_labels(self.days, self.syms, tiers=(10,),
                                   floor=5.0, store=store)
        self.assertEqual(sum(lab["top10"]), 5)

    def test_a_rank_qualifying_row_below_the_floor_is_a_negative(self):
        store = _Store(day_rows("2026-05-01", top=9.0))
        _, lab, _ = IL.tier_labels(self.days, self.syms, tiers=(10,),
                                   floor=5.0, store=store)
        self.assertEqual(lab["top10"][self.syms.index("S05")], 0.0)  # +4%
        self.assertEqual(lab["top10"][self.syms.index("S00")], 1.0)  # +9%

    def test_unknown_pair_is_dropped_not_labelled_zero(self):
        # A missing label counted as a negative would teach the model that every
        # symbol outside the store failed to move.
        days = self.days + ["2026-05-01"]
        syms = self.syms + ["NOTINSTORE"]
        keep, lab, stats = IL.tier_labels(days, syms, tiers=(10,), floor=5.0,
                                          store=self.store)
        self.assertNotIn(len(syms) - 1, keep)
        self.assertEqual(len(keep), 30)
        self.assertEqual(stats["dropped_unlabelled"], 1)
        self.assertEqual(len(lab["top10"]), 30)

    def test_a_thin_day_contributes_no_positives(self):
        store = _Store(day_rows("2026-05-01", n=30) + day_rows("2026-05-02", n=4))
        days = self.days + ["2026-05-02"] * 4
        syms = self.syms + [f"S{i:02d}" for i in range(4)]
        keep, lab, _ = IL.tier_labels(days, syms, tiers=(10,), floor=5.0,
                                      store=store)
        thin = [i for i, k in enumerate(keep) if days[k] == "2026-05-02"]
        self.assertTrue(thin, "thin-day rows are still labelled, just negative")
        self.assertEqual(sum(lab["top10"][i] for i in thin), 0.0)

    def test_stats_report_base_rate_beside_the_count(self):
        # TH-01: a positives count without its base rate is not readable.
        _, _, stats = IL.tier_labels(self.days, self.syms, tiers=(10, 20),
                                     floor=5.0, store=self.store)
        for tier in ("top10", "top20"):
            self.assertIn(tier, stats["base_rate"])
            self.assertGreater(stats["base_rate"][tier], 0.0)
        self.assertEqual(stats["n_labelled"], 30)


class TestTrainerWiring(unittest.TestCase):
    def setUp(self):
        self.src = (HERE / "train_top_gainer.py").read_text(encoding="utf-8")

    def test_flag_defaults_to_current_behaviour(self):
        import config
        self.assertFalse(getattr(config, "TRAIN_IMMUTABLE_LABELS_ENABLED", None),
                         "this model feeds the ranker hard veto")
        # The floor defaulted to +5% only because the store held the watchlist
        # alone, where a pure rank put top50 at a 52.6% base rate. Over the
        # global universe the rank is discriminative again and 0.0 reproduces
        # the ORIGINAL label — top-N of all USDT pairs. Both remain supported.
        self.assertEqual(getattr(config, "TRAIN_IMMUTABLE_LABEL_MIN_PCT", None), 0.0)

    def test_label_timing_is_not_hardcoded_to_the_leaky_value(self):
        # Three call sites used to state the leaky provenance unconditionally;
        # if the labels change and the string does not, the artifact lies.
        self.assertIn("immutable_later_eod_close", self.src)
        self.assertNotIn('"label_timing": "same_snapshot_current_24h_leaderboard"',
                         self.src)

    def test_scope_names_both_defects_independently(self):
        # Fixing the split must not let the label defect hide behind it.
        self.assertIn("_evaluation_scope", self.src)

    def test_report_carries_base_rate_and_trained_row_count(self):
        for key in ("label_base_rate", "n_records_labelled"):
            self.assertIn(key, self.src)


if __name__ == "__main__":
    unittest.main()
