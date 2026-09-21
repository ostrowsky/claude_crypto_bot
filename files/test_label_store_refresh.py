"""Guards for the 2026-09-21 repair of the immutable label store.

The store was built once (2026-08-17) and ended 2026-08-16. Nothing extended it,
and two consumers degraded without an error: top_gainer training kept the same
106 507 rows for 34 nights, and the North Star's primary value fell back to the
leaky rolling-24h label from 2026-08-30. These tests pin the fix so the store
cannot silently freeze again.

Spec: docs/specs/features/label-store-refresh-spec.md
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import artifact_freshness as AF  # noqa: E402

DL = (HERE / "daily_learning.py").read_text(encoding="utf-8")
NS = (HERE / "_compute_early_capture.py").read_text(encoding="utf-8")
CB = (HERE / "contextual_bandit.py").read_text(encoding="utf-8")


class TestTheNightlyCycleExtendsTheStore(unittest.TestCase):
    def test_the_refresh_runs_before_anything_trains_on_the_labels(self):
        cycle = DL[DL.index("async def run_daily_cycle") if "async def run_daily_cycle" in DL
                   else DL.index("=== Daily Learning Cycle START ==="):]
        self.assertLess(cycle.index("refresh_label_store()"),
                        cycle.index("resolve_and_train("),
                        "labels must be extended before the training step reads them")

    def test_it_appends_a_bounded_window_not_the_whole_history(self):
        self.assertIn('"build_global_labels.py"), "--days", "30"', DL)

    def test_a_failed_refresh_does_not_stop_the_cycle(self):
        import daily_learning as D
        real = D.subprocess if hasattr(D, "subprocess") else None
        import subprocess as sp
        orig = sp.run

        def boom(*a, **k):
            return types.SimpleNamespace(returncode=2, stdout="", stderr="binance down")
        sp.run = boom
        try:
            out = D.refresh_label_store()
        finally:
            sp.run = orig
        self.assertEqual(out["status"], "error")

    def test_a_successful_refresh_reports_ok(self):
        import daily_learning as D
        import subprocess as sp
        orig = sp.run
        sp.run = lambda *a, **k: types.SimpleNamespace(
            returncode=0, stdout="universe: 750\nwritten=1500\n", stderr="")
        try:
            out = D.refresh_label_store()
        finally:
            sp.run = orig
        self.assertEqual(out["status"], "ok")
        self.assertIn("written=1500", out["tail"][-1])


class TestAStaleStoreIsVisible(unittest.TestCase):
    def test_the_label_store_is_declared_with_a_daily_interval(self):
        art = {a.name: a for a in AF.ARTIFACTS}["label_store"]
        self.assertEqual(art.path, ".runtime/labels/move_events_v1.jsonl")
        self.assertLessEqual(art.max_age_h, 36)

    def test_a_store_that_stops_growing_is_reported_stale(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".runtime" / "labels"
            p.mkdir(parents=True)
            f = p / "move_events_v1.jsonl"
            f.write_text("{}\n")
            old = time.time() - 40 * 24 * 3600            # the real gap: 2026-08-17
            os.utime(f, (old, old))
            row = {r["name"]: r for r in AF.check(root=Path(d))}["label_store"]
        self.assertEqual(row["status"], "stale")


class TestThePrimaryMetricCannotSubstituteSilently(unittest.TestCase):
    def test_degradation_is_recorded_in_the_artifact(self):
        for token in ('"primary_degraded": degraded',
                      '"primary_degraded_reason"',
                      '"immutable_store_last_day"'):
            self.assertIn(token, NS)

    def test_degradation_is_shouted_on_the_console(self):
        self.assertIn("PRIMARY METRIC DEGRADED", NS)
        self.assertIn("LEAKY rolling-24h label", NS)

    def test_degraded_only_when_the_honest_metric_was_enabled(self):
        # if NS_IMMUTABLE_LABELS_ENABLED is off the fallback is the intended
        # state, not a failure -- shouting then would train people to ignore it
        self.assertIn('getattr(config, "NS_IMMUTABLE_LABELS_ENABLED", False)) '
                      'and res_imm is None', NS)

    def test_the_empty_window_case_now_has_a_reason(self):
        # the 2026-08-30..09-21 case: enabled, no exception, no winner-days
        self.assertIn("no winner-days with immutable labels inside the", NS)


class TestThePendingBufferIsGone(unittest.TestCase):
    def test_the_nightly_cycle_no_longer_calls_the_resolver(self):
        self.assertNotIn("resolve_pending_decisions", DL)
        self.assertNotIn("Pending decisions resolved", DL)

    def test_the_bandit_module_no_longer_keeps_the_buffer(self):
        for name in ("resolve_pending_decisions", "_store_pending_decision",
                     "_load_pending_decisions", "_clear_pending_decisions",
                     "PENDING_FILE"):
            self.assertNotIn(name, CB, name)

    def test_the_entry_bandit_is_rebuilt_from_scratch_each_night(self):
        # the premise for removing the resolver: nothing it wrote would survive
        import config
        self.assertTrue(config.BANDIT_REBUILD_ON_TRAIN)

    def test_the_module_still_imports(self):
        import contextual_bandit  # noqa: F401


if __name__ == "__main__":
    unittest.main(verbosity=2)
