"""metrics_daily must always get a row, and a stale row must never read as today.

2026-10-12..10-19: pipeline_run.py killed report_metrics_daily.py at 600 s every
morning (nine scripts x 180 s in sequence, row written only after the last), so
.runtime/metrics_daily.jsonl stopped at 10-11, and bot_health_report printed the
10-11 North Star as today's with status "verified".

Run:
    pyembed\\python.exe -X utf8 files\\test_metrics_daily_staleness.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import report_metrics_daily as R
import pipeline_run as PR
import bot_health_report as H

IMM = "immutable_later_eod_klines"


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _fake_scripts(td: Path, names):
    for n in names:
        (td / n).write_text("", encoding="utf-8")
    return [(n, []) for n in names]


class MetricsRowAlwaysWrittenTests(unittest.TestCase):
    def test_slow_scripts_still_produce_a_row_within_the_budget(self):
        # Reproduces 10-12..10-19: every script slow; the old loop needed more
        # than the caller's timeout before it wrote anything.
        clock = FakeClock()

        def runner(args, timeout):
            clock.t += min(200.0, timeout)
            if timeout < 200.0:
                raise subprocess.TimeoutExpired(args, timeout)
            return 0, 'METRIC_JSON:{"metric": "M_%s"}' % Path(args[1]).stem

        with tempfile.TemporaryDirectory() as td:
            scripts = _fake_scripts(Path(td), [f"s{i}.py" for i in range(9)])
            row = R.build_row(scripts, budget_sec=700, per_script_timeout=300,
                              runner=runner, clock=clock, files_dir=Path(td))
        self.assertLessEqual(row["elapsed_sec"], 700)
        for name, _ in scripts:
            self.assertIn(name, row, "every script must have an entry")
        self.assertEqual(row["s0.py"]["metric"], "M_s0")
        self.assertEqual(row["s2.py"]["metric"], "M_s2")
        self.assertEqual(row["s3.py"]["error"], "timeout")      # cut at the budget edge
        self.assertEqual(row["s8.py"]["error"], "skipped_budget")

    def test_north_star_runs_first(self):
        self.assertEqual(R.SCRIPTS[0][0], "_compute_early_capture.py")

    def test_budget_ends_before_the_orchestrator_kills_the_step(self):
        # room to write the row and for interpreter start-up
        self.assertLessEqual(R.DEFAULT_BUDGET_SEC + 120, PR.L0_METRICS_TIMEOUT_SEC)
        self.assertLessEqual(R.PER_SCRIPT_TIMEOUT_SEC, R.DEFAULT_BUDGET_SEC)

    def test_row_is_appended(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "m.jsonl"
            R.append_row({"ts": "t1"}, out)
            R.append_row({"ts": "t2"}, out)
            rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["ts"] for r in rows], ["t1", "t2"])

    def test_orchestrator_prints_a_timeout(self):
        with mock.patch.object(PR.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired("x", 5)), \
             mock.patch("builtins.print") as p:
            res = PR.run_step("L0 metrics", ["x"], timeout=5)
        self.assertEqual(res["error"], "timeout")
        printed = " ".join(str(c.args[0]) for c in p.call_args_list)
        self.assertIn("TIMEOUT", printed)


def _ns(**kw):
    d = {"metric": "NS_EarlyCapture_top20_v2", "early_capture": 0.204, "n": 30,
         "days_window": 14, "days_full": 14, "label_provenance": IMM}
    d.update(kw)
    return d


def _collect(rows, now):
    import pipeline_lib as PL
    with mock.patch.object(PL, "iter_jsonl", lambda *_a, **_k: iter(rows)):
        return H.collect_metrics_daily_latest(now=now)


NOW = datetime(2026, 10, 20, 7, 30, tzinfo=timezone.utc)


def _ts(hours_ago):
    return (NOW - timedelta(hours=hours_ago)).isoformat()


class StaleGuardTests(unittest.TestCase):
    def test_fresh_row_is_verified(self):
        md = _collect([{"ts": _ts(1), "_compute_early_capture.py": _ns()}], NOW)
        self.assertFalse(md["stale"])
        sc = H.build_canonical_scorecard(md)
        self.assertEqual(sc["north_star"]["status"], "verified")

    def test_nine_day_old_row_is_stale_not_verified(self):
        # the 10-20 report: latest row 2026-10-11T07:00
        md = _collect([{"ts": "2026-10-11T07:00:04+00:00",
                        "_compute_early_capture.py": _ns(),
                        "_backtest_signal_precision.py": {
                            "metric": "D1_D2_precision_msgrate", "precision_pct": 13.6,
                            "label_provenance": IMM}}], NOW)
        self.assertTrue(md["stale"])
        self.assertGreater(md["age_hours"], 36)
        sc = H.build_canonical_scorecard(md)
        self.assertEqual(sc["north_star"]["status"], "stale")
        self.assertEqual(sc["north_star"]["as_of"], "2026-10-11T07:00:04+00:00")
        self.assertEqual(sc["signal_precision"]["status"], "stale")
        for key in ("message_rate",):
            self.assertEqual(sc[key]["status"], "stale", key)

    def test_unreadable_timestamp_is_stale(self):
        md = _collect([{"ts": "x", "_compute_early_capture.py": _ns()}], NOW)
        self.assertTrue(md["stale"])

    def test_a_metric_missing_from_today_falls_back_marked_stale(self):
        # A partial row (North Star timed out) must not blank the value, nor
        # pass yesterday-but-one's value off as today's.
        rows = [
            {"ts": _ts(72), "_compute_early_capture.py": _ns(early_capture=0.19)},
            {"ts": _ts(1), "_compute_early_capture.py": {"error": "timeout"},
             "_backtest_whipsaw_rate.py": {"metric": "Q2_whipsaw_rate", "overall_pct": 3.0}},
        ]
        md = _collect(rows, NOW)
        self.assertFalse(md["stale"])
        sc = H.build_canonical_scorecard(md)
        self.assertEqual(sc["north_star"]["value"], 0.19)
        self.assertEqual(sc["north_star"]["status"], "stale")
        self.assertEqual(sc["whipsaw"]["status"], "measured")

    def test_stale_north_star_asks_to_restore_metrics_not_labels(self):
        md = _collect([{"ts": _ts(200), "_compute_early_capture.py": _ns()}], NOW)
        sc = H.build_canonical_scorecard(md)
        steps = H.derive_next_steps(sc, {"evaluation_scope": "out_of_sample_time_holdout"},
                                    {}, NOW.date())
        ids = {s["id"] for s in steps}
        self.assertIn("restore_metrics_daily", ids)
        self.assertNotIn("restore_eod_ground_truth", ids)

    def test_stale_progress_verdict_says_so(self):
        emoji, head, trend = H._progress_verdict(ground_truth_verified=True,
                                                 stale_as_of="2026-10-11")
        self.assertIn("УСТАРЕЛ", head)
        self.assertIn("2026-10-11", trend)

    def test_telegram_names_the_date_of_a_stale_value(self):
        md = _collect([{"ts": "2026-10-11T07:00:04+00:00",
                        "_compute_early_capture.py": _ns()}], NOW)
        report = {
            "target_date": "2026-10-20",
            "deployment_health": {"available": False},
            "metrics_daily_latest": md,
            "canonical_scorecard": H.build_canonical_scorecard(md),
            "mode_curtail": {"available": False},
            "red_flags": [],
        }
        with mock.patch.object(H, "_action_needed_count", return_value=0), \
             mock.patch.object(H, "_past_decisions_resume", return_value=[]):
            text = H.render_telegram(report)
        self.assertIn("2026-10-11", text)
        self.assertIn("устарел", text.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
