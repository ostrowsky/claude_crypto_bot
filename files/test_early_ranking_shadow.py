"""Exit gates for the early-ranking shadow path (goals 1 and 2).

The properties that matter: it cannot touch a decision, it cannot invent a list
it did not have data for, and it cannot report a ratio without saying how thin
it is.

Spec: docs/specs/features/early-ranking-shadow-spec.md
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import early_ranking_shadow as ERS  # noqa: E402


class TestIsolation(unittest.TestCase):
    """A shadow that can reach a gate is not a shadow."""

    def test_imports_nothing_that_decides(self):
        tree = ast.parse((HERE / "early_ranking_shadow.py").read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        self.assertFalse(names & {"monitor", "strategy", "bot", "contextual_bandit",
                                  "trend_scout_rules", "rotation"})

    def test_defines_no_gate_or_block(self):
        src = (HERE / "early_ranking_shadow.py").read_text(encoding="utf-8")
        for banned in ("log_blocked", "should_enter", "BlockRule", "reason_code"):
            self.assertNotIn(banned, src)


class TestListConstruction(unittest.TestCase):
    def _rows(self):
        return [("A", 0.9), ("B", 0.5), ("C", 0.7), ("D", 0.1)]

    def test_capped_at_k_and_sorted_desc(self):
        out = ERS.build_list(self._rows(), k=2)
        self.assertEqual([r["symbol"] for r in out], ["A", "C"])
        self.assertGreater(out[0]["proba"], out[1]["proba"])

    def test_k_larger_than_universe_is_not_padded(self):
        out = ERS.build_list(self._rows(), k=99)
        self.assertEqual(len(out), 4)

    def test_no_snapshot_writes_nothing_rather_than_an_empty_list(self):
        # An empty list would later read as "the model named nobody", which is a
        # claim about the model, not about the missing snapshot (TH-05).
        self.assertIsNone(ERS.build_list([], k=5))


class TestScoringHonesty(unittest.TestCase):
    def _lists(self, days, universe=100):
        return [{"utc_day": d, "universe": universe,
                 "picks": [{"symbol": "W"}, {"symbol": "X"}]}
                for d in days]

    def test_unlabelled_day_is_skipped_and_counted(self):
        res = ERS.score(self._lists(["2026-05-01", "2026-05-02"]),
                        winners={("2026-05-01", "W")},
                        label_days={"2026-05-01"})
        self.assertEqual(res["days_scored"], 1)
        self.assertEqual(res["days_without_labels"], 1)

    def test_every_ratio_carries_its_n(self):
        res = ERS.score(self._lists(["2026-05-01"]),
                        winners={("2026-05-01", "W")},
                        label_days={"2026-05-01"})
        for key in ("n_picks", "n_winners_available", "days_scored"):
            self.assertIn(key, res)

    def test_too_few_days_refuses_a_verdict(self):
        res = ERS.score(self._lists(["2026-05-01"]),
                        winners={("2026-05-01", "W")},
                        label_days={"2026-05-01"})
        self.assertEqual(res["verdict"], "too early to judge")
        self.assertLess(res["days_scored"], ERS.MIN_DAYS_TO_JUDGE)

    def test_enough_days_produces_a_control_band(self):
        days = [f"2026-05-{i:02d}" for i in range(1, 26)]
        winners = {(d, "W") for d in days}
        res = ERS.score(self._lists(days), winners=winners, label_days=set(days))
        self.assertNotEqual(res["verdict"], "too early to judge")
        self.assertIn("control_band", res)
        self.assertLess(res["control_band"][0], res["control_band"][1])

    def test_a_missing_universe_refuses_the_control_rather_than_flattering_it(self):
        # Treating an unknown universe as zero gives that day zero chance of a
        # random hit, dragging the band to [0, 0] — after which ANY coverage
        # reads as "above control". A flattering control is worse than none.
        days = [f"2026-05-{i:02d}" for i in range(1, 26)]
        lists = [{"utc_day": d, "picks": [{"symbol": "W"}]} for d in days]
        res = ERS.score(lists, winners={(d, "W") for d in days},
                        label_days=set(days))
        self.assertIsNone(res["control_band"])
        self.assertEqual(res["verdict"], "control unavailable")
        self.assertEqual(res["days_without_universe"], 25)


class TestTheTwoDefectsTheFirstLiveRunHad(unittest.TestCase):
    """Both were silent: the list looked plausible and was wrong."""

    def setUp(self):
        self.src = (HERE / "early_ranking_shadow.py").read_text(encoding="utf-8")

    def test_snapshots_are_keyed_by_symbol_not_appended(self):
        # The dataset carries TWO 00 UTC snapshots per day, so appending put
        # every coin in twice: universe read 210 for a 105-coin watchlist and a
        # "top-10" was really a top-5.
        self.assertIn("by_day[dt.strftime(\"%Y-%m-%d\")][sym]", self.src)
        self.assertIn("defaultdict(dict)", self.src)

    def test_it_refuses_to_write_a_list_the_heuristic_produced(self):
        # TopGainerModel() with no path never calls load(), so predictions came
        # from the heuristic fallback — not from the model whose early-hour AUC
        # is the whole reason this path exists.
        self.assertIn("model_path=str(model_file)", self.src)
        self.assertIn('if not blob.get("tier_models")', self.src)
        self.assertIn("refusing to", self.src)

    def test_provenance_fields_are_written(self):
        for key in ("model_evaluation_scope", "model_label_timing"):
            self.assertIn(key, self.src)


if __name__ == "__main__":
    unittest.main()
