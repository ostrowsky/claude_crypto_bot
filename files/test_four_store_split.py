"""Exit gates for the four-store split.

The point of this suite is one assertion: **appending an approved record to
research memory must not change live gating.** Everything else supports it.

That path was real. `.runtime/pipeline/decisions/decisions.jsonl` was both the
loop's memory and its execution channel, two gating constants were live from it,
and the most recent approved record in it was written by an LLM. Hardening made
the channel visible; only this split severs it.

Spec: docs/specs/features/four-store-split-spec.md
"""
from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import control_plane_stores as ST  # noqa: E402
import release_overrides as REL  # noqa: E402


class TestChannelIsSevered(unittest.TestCase):
    """The one that matters."""

    def setUp(self):
        self.root = HERE.parent / ".runtime" / "four_store_test"
        shutil.rmtree(self.root, ignore_errors=True)
        self.stores = ST.ControlPlaneStores(self.root)
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_research_ledger_entry_cannot_become_an_override(self):
        # Exactly the shape that used to execute: approved, real config key,
        # concrete value.
        self.stores.research.append({
            "decision_id": "d-evil-1", "stage": "approved",
            "config_key": "ENTRY_SCORE_MIN_15M",
            "diff": {"from": 40.0, "to": 1.0},
            "approval_reason": "written by the research path",
        })
        applied = REL.materialise(self.stores, dry_run=True)
        self.assertNotIn("ENTRY_SCORE_MIN_15M", applied,
                         "research memory must never reach the override store")

    def test_only_signed_or_legacy_entries_reach_the_store(self):
        self.stores.approvals.append({
            "approval_id": "a-unsigned", "config_key": "ENTRY_SCORE_MIN_15M",
            "value": 1.0, "signature": None, "source": "signed_approval",
        })
        applied = REL.materialise(self.stores, dry_run=True)
        self.assertNotIn("ENTRY_SCORE_MIN_15M", applied)

    def test_bad_signature_is_refused(self):
        key = b"operator-secret"
        rec = {"approval_id": "a-bad", "config_key": "ENTRY_SCORE_MIN_15M",
               "value": 1.0, "source": "signed_approval"}
        rec["signature"] = "deadbeef" * 8
        self.stores.approvals.append(rec)
        applied = REL.materialise(self.stores, key=key, dry_run=True)
        self.assertNotIn("ENTRY_SCORE_MIN_15M", applied)

    def test_good_signature_is_accepted(self):
        key = b"operator-secret"
        rec = {"approval_id": "a-good", "config_key": "ENTRY_SCORE_MIN_15M",
               "value": 33.0, "source": "signed_approval"}
        rec["signature"] = REL.sign(rec, key)
        self.stores.approvals.append(rec)
        applied = REL.materialise(self.stores, key=key, dry_run=True)
        self.assertEqual(applied["ENTRY_SCORE_MIN_15M"]["value"], 33.0)

    def test_protected_keys_are_refused_even_when_signed(self):
        key = b"operator-secret"
        for protected in ("AUTO_APPLY_OVERRIDES_ENABLED", "DEFAULT_WATCHLIST"):
            rec = {"approval_id": f"a-{protected}", "config_key": protected,
                   "value": True, "source": "signed_approval"}
            rec["signature"] = REL.sign(rec, key)
            self.stores.approvals.append(rec)
        applied = REL.materialise(self.stores, key=key, dry_run=True)
        self.assertNotIn("AUTO_APPLY_OVERRIDES_ENABLED", applied)
        self.assertNotIn("DEFAULT_WATCHLIST", applied)

    def test_legacy_entries_carry_provenance_and_a_review_date(self):
        self.stores.approvals.append({
            "approval_id": "a-legacy", "config_key": "ENTRY_SCORE_MIN_15M",
            "value": 35.0, "signature": None, "source": "legacy_decisions_jsonl",
            "provenance": ["d-2026-06-01-x"], "review_by": "2026-09-14",
        })
        applied = REL.materialise(self.stores, dry_run=True)
        entry = applied["ENTRY_SCORE_MIN_15M"]
        self.assertEqual(entry["source"], "legacy_decisions_jsonl")
        self.assertTrue(entry["provenance"])
        self.assertTrue(entry["review_by"])

    def test_unsigned_legacy_is_reported_as_debt(self):
        self.stores.approvals.append({
            "approval_id": "a-legacy2", "config_key": "ENTRY_SCORE_MIN_15M",
            "value": 35.0, "signature": None, "source": "legacy_decisions_jsonl",
            "provenance": ["d-x"], "review_by": "2026-09-14",
        })
        report = REL.debt_report(self.stores)
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["config_key"], "ENTRY_SCORE_MIN_15M")


class TestConfigReader(unittest.TestCase):
    @staticmethod
    def _live_string_constants(path: Path) -> list[str]:
        """String literals the code actually evaluates — docstrings excluded.

        The first version of this test forbade the substring anywhere in the
        file, which failed on the docstring explaining that the file is no
        longer read. What matters is whether the path is *used*, not whether it
        is *mentioned*.
        """
        import ast
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc is not None:
                    docstrings.add(doc)
        return [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and n.value not in docstrings]

    def test_config_no_longer_references_the_decisions_log(self):
        for module in ("config.py", "_config_runtime_overrides.py"):
            for literal in self._live_string_constants(HERE / module):
                self.assertNotIn(
                    "decisions.jsonl", literal,
                    f"{module} still evaluates a path to the executable "
                    f"decisions log: {literal!r}")

    def test_reader_consumes_the_release_store(self):
        reader = (HERE / "_config_runtime_overrides.py").read_text(encoding="utf-8")
        self.assertIn("runtime_overrides.json", reader)

    def test_protected_keys_still_refused_at_apply_time(self):
        import _config_runtime_overrides as RO
        self.assertIn("AUTO_APPLY_OVERRIDES_ENABLED", RO._NEVER_OVERRIDABLE)


class TestMigrationPreservesBehaviour(unittest.TestCase):
    """Live gating must be identical before and after. This is the constraint
    that makes the whole change safe to ship."""

    def test_effective_values_survive_migration(self):
        store = HERE.parent / ".runtime" / "release" / "runtime_overrides.json"
        self.assertTrue(store.exists(), "migration must have materialised the store")
        data = json.loads(store.read_text(encoding="utf-8"))
        applied = data["overrides"]
        # The two constants that were live before the split.
        self.assertEqual(applied["ENTRY_SCORE_MIN_15M"]["value"], 35.0)
        self.assertTrue(applied["IMPULSE_SPEED_15M_HIGH_MOMENTUM_BYPASS_ENABLED"]["value"])
        for entry in applied.values():
            self.assertIn(entry["source"],
                          ("legacy_decisions_jsonl", "signed_approval"))

    def test_live_config_matches_the_store(self):
        import config
        store = HERE.parent / ".runtime" / "release" / "runtime_overrides.json"
        applied = json.loads(store.read_text(encoding="utf-8"))["overrides"]
        for key, entry in applied.items():
            self.assertEqual(getattr(config, key), entry["value"],
                             f"{key} in config does not match the release store")

    def test_migration_is_idempotent(self):
        first = REL.migrate_legacy(dry_run=True)
        second = REL.migrate_legacy(dry_run=True)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
