"""Exit gates for immutable North-Star labels (TH-03).

The property that matters: a winner is decided by the close of a finished UTC
day, from exchange data, and never by the snapshot that produced the features.

Spec: docs/specs/features/north-star-immutable-labels-spec.md
"""
from __future__ import annotations

import ast
import shutil
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import immutable_labels as IL  # noqa: E402
import label_store as LS  # noqa: E402


class _Store(LS.LabelStore):
    """A store backed by supplied records, so tests never touch the real one."""

    def __init__(self, records):
        self._records = records

    def records(self):
        return list(self._records)


def rec(symbol, day, eod, *, complete=True):
    return {"symbol": symbol, "utc_day": day, "eod_return_pct": eod,
            "complete": complete, "qualifies_move5": eod >= 5.0}


class TestWinnerSelection(unittest.TestCase):
    def _day(self, day, n=30):
        # descending returns: S0 best, S29 worst
        return [rec(f"S{i:02d}", day, 30.0 - i) for i in range(n)]

    def test_top_n_winners_on_a_full_day(self):
        store = _Store(self._day("2026-05-01"))
        winners, _ = IL.winners_by_day(top_n=20, store=store)
        self.assertEqual(len(winners), 20)
        self.assertIn(("2026-05-01", "S00"), winners)
        self.assertNotIn(("2026-05-01", "S29"), winners)

    def test_a_thin_day_yields_no_winners_rather_than_a_short_list(self):
        # Ranking six symbols and calling the top three "the day's top-20"
        # would manufacture winners out of missing data.
        store = _Store(self._day("2026-05-01", n=30) + self._day("2026-05-02", n=6))
        winners, _ = IL.winners_by_day(top_n=20, store=store)
        self.assertEqual({d for d, _ in winners}, {"2026-05-01"})

    def test_incomplete_records_never_win(self):
        rows = self._day("2026-05-01")
        rows.append(rec("SXX", "2026-05-01", 99.0, complete=False))
        winners, _ = IL.winners_by_day(top_n=20, store=_Store(rows))
        self.assertNotIn(("2026-05-01", "SXX"), winners)

    def test_watchlist_restricts_the_universe(self):
        store = _Store(self._day("2026-05-01"))
        winners, _ = IL.winners_by_day(top_n=5, store=store,
                                       watchlist={f"S{i:02d}" for i in range(10)})
        self.assertTrue(all(s in {f"S{i:02d}" for i in range(10)}
                            for _, s in winners))


class TestMissingIsNotZero(unittest.TestCase):
    def test_unknown_pair_returns_none(self):
        store = _Store([rec("AUSDT", "2026-05-01", 10.0)])
        cache: dict = {}
        self.assertIsNone(IL.label_for("ZZZUSDT", "2026-05-01",
                                       cache=cache, store=store))

    def test_known_non_winner_returns_zero_not_none(self):
        rows = [rec(f"S{i:02d}", "2026-05-01", 30.0 - i) for i in range(30)]
        cache: dict = {}
        self.assertEqual(IL.label_for("S29", "2026-05-01", top_n=20,
                                      cache=cache, store=_Store(rows)), 0)
        self.assertEqual(IL.label_for("S00", "2026-05-01", top_n=20,
                                      cache=cache, store=_Store(rows)), 1)


class TestProvenance(unittest.TestCase):
    def test_module_reads_no_bot_dataset(self):
        tree = ast.parse((HERE / "immutable_labels.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for banned in ("top_gainer_dataset", "critic_dataset", "ml_dataset"):
                    self.assertNotIn(banned, node.value)
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        self.assertFalse(names & {"config", "monitor", "strategy", "offline_rl"})

    def test_summary_names_its_provenance(self):
        self.assertEqual(IL.summary.__doc__ is None, True)   # plain helper
        # provenance is a value, not prose: it must survive into the payload
        src = (HERE / "immutable_labels.py").read_text(encoding="utf-8")
        self.assertIn('"immutable_later_eod_klines"', src)


class TestFlagDefaults(unittest.TestCase):
    def test_metric_flag_on_model_flag_off(self):
        import config
        self.assertTrue(getattr(config, "NS_IMMUTABLE_LABELS_ENABLED", None),
                        "the metric change is safe: it alters no behaviour")
        self.assertFalse(getattr(config, "TRAIN_IMMUTABLE_LABELS_ENABLED", None),
                         "relabelling the model changes live gating indirectly")


class TestPublishedSideBySide(unittest.TestCase):
    """The immutable value is published BESIDE the old one, and marked
    non-comparable — the two use different denominators, so a reader who
    differences them measures the denominator, not the bot."""

    def setUp(self):
        self.src = (HERE / "_compute_early_capture.py").read_text(encoding="utf-8")

    def test_primary_metric_keys_are_not_overwritten(self):
        # Substituting the loader would make a change of provenance look like a
        # change in performance across the historical series.
        for key in ('"early_capture": res_top20', '"label_provenance": "rolling_24h_same_snapshot"'):
            self.assertIn(key, self.src)

    def test_immutable_value_is_emitted_with_its_provenance(self):
        for key in ("immutable_early_capture", "immutable_label_provenance",
                    "immutable_denominator"):
            self.assertIn(key, self.src)

    def test_non_comparability_is_machine_readable_not_only_prose(self):
        self.assertIn('metric["immutable_comparable_to_primary"] = False', self.src)


if __name__ == "__main__":
    unittest.main()
