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


class TestRankBeforeFilter(unittest.TestCase):
    """Which top-20 is being asked for. Ranking inside the watchlist answers an
    easier question and mints exactly N winners a day whatever the market did."""

    def _mixed_day(self):
        # 10 off-watchlist coins beat every watchlist coin.
        rows = [rec(f"X{i:02d}", "2026-05-01", 50.0 - i) for i in range(10)]
        rows += [rec(f"W{i:02d}", "2026-05-01", 20.0 - i) for i in range(20)]
        return _Store(rows), {f"W{i:02d}" for i in range(20)}

    def test_global_rank_then_intersect_yields_fewer_winners(self):
        store, wl = self._mixed_day()
        winners, _ = IL.winners_by_day(top_n=20, watchlist=wl, store=store,
                                       rank_before_filter=True)
        # 10 of the global top-20 are off-watchlist, so only 10 survive.
        self.assertEqual(len(winners), 10)
        self.assertTrue(all(s.startswith("W") for _, s in winners))

    def test_ranking_inside_the_watchlist_always_mints_n(self):
        store, wl = self._mixed_day()
        winners, _ = IL.winners_by_day(top_n=20, watchlist=wl, store=store,
                                       rank_before_filter=False)
        self.assertEqual(len(winners), 20)

    def test_a_day_with_no_watchlist_coin_in_the_global_top_n_has_no_winners(self):
        # The honest answer, not a reason to reach further down the list.
        rows = [rec(f"X{i:02d}", "2026-05-01", 50.0 - i) for i in range(25)]
        rows += [rec("W00", "2026-05-01", -5.0)]
        winners, _ = IL.winners_by_day(top_n=20, watchlist={"W00"},
                                       store=_Store(rows),
                                       rank_before_filter=True)
        self.assertEqual(winners, set())


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
        # Held False until the global label universe removed the tier collapse;
        # flipped by the operator 2026-08-17. Both remain rollback switches.
        self.assertIsInstance(
            getattr(config, "TRAIN_IMMUTABLE_LABELS_ENABLED", None), bool)


class TestPublishedSideBySide(unittest.TestCase):
    """The immutable value is published BESIDE the old one, never instead of
    it, and its comparability is a machine-readable field rather than prose."""

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

    def test_comparability_is_machine_readable_and_names_its_denominator(self):
        # This asserted `= False` while the store held only watchlist symbols
        # and the two values genuinely answered different questions. Since the
        # store went global both rank the SAME denominator, so the flag flipped.
        # The invariant that outlived it: comparability is never left to prose,
        # and the denominator it rests on is stated in the artifact.
        self.assertIn('metric["immutable_comparable_to_primary"]', self.src)
        self.assertIn('global_top20_intersect_watchlist_from_label_store', self.src)
        self.assertIn('rank_before_filter=True', self.src)


if __name__ == "__main__":
    unittest.main()
