"""Comparable-window trends and target-spec Q5 for the learning-progress analyzer.

Spec: docs/specs/features/learning-progress-comparable-1118-spec.md
(TH-01, TH-02, TH-04, TH-05, TH-10). Synthetic rows only.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyze_learning_progress as A


V2 = "NS_EarlyCapture_top20_v2"
IMM = "immutable_later_eod_klines"
DEN = "global_top20_intersect_watchlist_from_label_store"
OOS = "out_of_sample_time_holdout"
MODEL_SCOPE = "day_grouped_holdout_immutable_later_eod_label"
MODEL_TIMING = "immutable_later_eod_close"


def _d(s: str, plus: int = 0) -> str:
    return (date.fromisoformat(s) + timedelta(days=plus)).isoformat()


def ns_row(day, value, n, *, hhmm="07:00", metric=V2, window=14, prov=IMM,
           denom=DEN, lead="clock_hour", store_last="auto", degraded=False,
           **extra):
    ec = {
        "metric": metric, "days_window": window, "label_provenance": prov,
        "denominator": denom, "lead_definition": lead,
        "primary_degraded": degraded, "early_capture": value, "n": n,
        "days_full": 13, "days_down_or_partial": 1,
        "decomp_coverage": 0.7143, "decomp_capture_mean": 0.3613,
        "decomp_time_lead_mean": 0.5333,
        "move_lead_metric": "NS_EarlyCapture_top20_v3",
        "move_lead_early_capture": 0.0, "move_lead_n": 12,
        "move_lead_winners_without_deadline": 2,
    }
    if store_last == "auto":
        ec["immutable_store_last_day"] = _d(day, -1)
    elif store_last is not None:
        ec["immutable_store_last_day"] = store_last
    ec.update(extra)
    return {"ts": f"{day}T{hhmm}:00+00:00", "_compute_early_capture.py": ec}


def lp_row(day, *, scope=OOS, label="forward_top10_min3", recall=0.1,
           ucb=0.05, auc=0.85, model_scope=MODEL_SCOPE,
           model_timing=MODEL_TIMING, n_pos=None, span=None, **extra):
    row = {
        "ts": f"{day}T00:50:00+00:00",
        "bandit_evaluation_scope": scope, "bandit_label": label,
        "bandit_recall_top20": recall, "bandit_ucb_separation": ucb,
        "bandit_action_rate": 0.012, "bandit_top20_base_rate": 0.0225,
        "bandit_recall_lift": 11.1, "bandit_precision": 0.25,
        "bandit_total_updates": 36670, "bandit_n_signal": 25000,
        "model_auc_top20": auc, "model_evaluation_scope": model_scope,
        "model_label_timing": model_timing,
    }
    if n_pos is not None:
        row["bandit_n_positive"] = n_pos
    if span is not None:
        row["bandit_eval_first_day"], row["bandit_eval_last_day"] = span
    row.update(extra)
    return row


HEALTH_VERIFIED = {"north_star": {"metric": "EarlyCapture@top20", "value": 0.12,
                                  "status": "verified", "baseline_7d": None}}


def run_analyze(*, lp=(), md=(), health=HEALTH_VERIFIED, docs=None):
    docs = docs or {}
    with mock.patch.object(A, "load_learning_progress", return_value=list(lp)), \
         mock.patch.object(A, "load_metrics_daily", return_value=list(md)), \
         mock.patch.object(A, "load_latest_health", return_value=health), \
         mock.patch.object(A, "load_decisions", return_value=[]), \
         mock.patch.object(A, "load_attribution_meta", return_value=None), \
         mock.patch.object(A, "_find_doc", side_effect=lambda repo, name: docs.get(name)), \
         mock.patch.object(A, "_git_last_commit_date", return_value=None):
        return A.analyze(Path("."), docs.get("auto-improvement-loop-spec.md"))


# ── D1: North Star points ─────────────────────────────────────────────────────

class NorthStarPointsTests(unittest.TestCase):
    def test_legacy_label_row_is_not_a_point(self):
        rows = [ns_row("2026-04-29", 0.0743, 66, metric="NS_EarlyCapture_top20",
                       prov="rolling_24h_same_snapshot", window=None,
                       store_last=None)]
        rows += [ns_row(_d("2026-09-22", 14 * i), 0.1, 30) for i in range(5)]
        pts, excluded = A.ns_points(rows)
        self.assertTrue(pts)
        self.assertNotIn("2026-04-29", [p["date"] for p in pts])
        self.assertTrue(all(p["value"] == 0.1 for p in pts))
        self.assertEqual(sum(excluded.values()), 1)

    def test_rows_that_cannot_show_label_coverage_are_excluded(self):
        rows = [
            ns_row("2026-08-25", 0.10, 7, store_last=None),          # field absent
            ns_row("2026-09-10", 0.10, 20, store_last="2026-08-16"),  # frozen store
            ns_row("2026-09-30", 0.10, 30, degraded=True),
            ns_row("2026-10-20", 0.12, 30),
        ]
        pts, excluded = A.ns_points(rows)
        self.assertEqual([p["date"] for p in pts], ["2026-10-20"])
        self.assertEqual(sum(excluded.values()), 3)
        self.assertTrue(any("store" in reason for reason in excluded))

    def test_other_window_or_lead_definition_is_not_comparable(self):
        rows = [ns_row("2026-10-01", 0.2, 30, window=30),
                ns_row("2026-10-10", 0.2, 30, lead="move_relative"),
                ns_row("2026-10-20", 0.1, 30)]
        pts, excluded = A.ns_points(rows)
        self.assertEqual([p["date"] for p in pts], ["2026-10-20"])
        self.assertEqual(sum(excluded.values()), 2)

    def test_one_point_per_date_and_disjoint_chain(self):
        rows = [ns_row(_d("2026-10-01", i), 0.10, 20) for i in range(31)]
        rows.append(ns_row("2026-10-31", 0.99, 25, hhmm="11:58"))  # later rerun wins
        pts, _ = A.ns_points(rows)
        self.assertEqual(len(pts), 31)
        chain = A.non_overlapping_chain(pts)
        self.assertEqual([p["date"] for p in chain],
                         ["2026-10-03", "2026-10-17", "2026-10-31"])
        self.assertEqual(chain[-1]["value"], 0.99)
        self.assertEqual(chain[-1]["n"], 25)
        for older, newer in zip(chain, chain[1:]):
            self.assertLess(older["span"][1], newer["span"][0])


# ── D1: verdict on disjoint windows ───────────────────────────────────────────

def _chain(values_n, step=14):
    pts = [ns_row(_d("2026-09-22", step * i), v, n) for i, (v, n) in enumerate(values_n)]
    return A.non_overlapping_chain(A.ns_points(pts)[0])


class CompareHalvesTests(unittest.TestCase):
    def test_single_window_is_insufficient(self):
        verdict, _ = A.compare_halves(_chain([(0.1, 50)]))
        self.assertEqual(verdict, "INSUFFICIENT")

    def test_small_pooled_n_is_insufficient(self):
        verdict, info = A.compare_halves(_chain([(0.07, 10), (0.12, 14)]))
        self.assertEqual(verdict, "INSUFFICIENT")
        self.assertIn("n", json.dumps(info))

    def test_difference_inside_noise_is_inconclusive(self):
        verdict, info = A.compare_halves(
            _chain([(0.10, 30), (0.11, 30), (0.12, 30), (0.13, 30)]))
        self.assertEqual(verdict, "INCONCLUSIVE")
        self.assertEqual(info["early"]["n"], 60)
        self.assertEqual(info["late"]["n"], 60)
        self.assertAlmostEqual(info["early"]["value"], 0.105, places=4)
        self.assertAlmostEqual(info["late"]["value"], 0.125, places=4)
        self.assertLess(info["ci95"][0], 0)
        self.assertGreater(info["ci95"][1], 0)

    def test_large_clear_difference_is_decided(self):
        up, _ = A.compare_halves(_chain([(0.05, 400), (0.20, 400)]))
        down, _ = A.compare_halves(_chain([(0.20, 400), (0.05, 400)]))
        self.assertEqual(up, "IMPROVING")
        self.assertEqual(down, "DEGRADING")

    def test_printed_delta_is_late_minus_early(self):
        _, info = A.compare_halves(_chain([(0.05, 400), (0.07, 400), (0.20, 400)]))
        # odd chain: the middle window is left out, halves stay disjoint
        self.assertEqual(info["early"]["windows"], 1)
        self.assertEqual(info["late"]["windows"], 1)
        self.assertAlmostEqual(info["delta"],
                               round(info["late"]["value"] - info["early"]["value"], 4))

    def test_missing_n_is_insufficient(self):
        chain = _chain([(0.1, 30), (0.2, 30)])
        chain[0] = dict(chain[0], n=None)
        verdict, _ = A.compare_halves(chain)
        self.assertEqual(verdict, "INSUFFICIENT")

    def test_flat_is_never_a_verdict(self):
        verdict, _ = A.compare_halves(_chain([(0.1, 400), (0.1, 400)]))
        self.assertEqual(verdict, "INCONCLUSIVE")


class NorthStarAnalyzeTests(unittest.TestCase):
    def test_trend_ignores_the_april_legacy_point(self):
        md = [ns_row("2026-04-29", 0.0743, 66, metric="NS_EarlyCapture_top20",
                     prov="rolling_24h_same_snapshot", window=None, store_last=None)]
        md += [ns_row(_d("2026-09-22", 14 * i), v, n)
               for i, (v, n) in enumerate([(0.08, 40), (0.13, 60), (0.19, 50),
                                           (0.08, 18), (0.12, 14)])]
        got = run_analyze(md=md)
        verdict, info = got["north_star"]["trend"]
        self.assertNotEqual(verdict, "IMPROVING")
        self.assertEqual(info["early"]["from"], "2026-09-22")
        text = A.render(got)
        self.assertNotIn("2026-04-29", text)
        self.assertIn("n=14", text)

    def test_provisional_ground_truth_stays_unknown(self):
        md = [ns_row(_d("2026-09-22", 14 * i), 0.1, 400) for i in range(4)]
        health = {"north_star": {"status": "provisional", "value": 0.1}}
        got = run_analyze(md=md, health=health)
        self.assertEqual(got["north_star"]["trend"][0], "UNKNOWN")

    def test_q1_prints_decomposition_v3_and_downtime(self):
        md = [ns_row("2026-11-18", 0.1242, 14, days_full=11, days_down_or_partial=3)]
        text = A.render(run_analyze(md=md))
        self.assertIn("coverage=0.714", text)
        self.assertIn("capture=0.361", text)
        self.assertIn("lead=0.533", text)
        self.assertIn("NS_EarlyCapture_top20_v3", text)
        self.assertIn("move_lead_n=12", text)
        self.assertIn("days_down_or_partial=3", text)
        self.assertIn("11/14", text)


# ── D1/D2: learning_progress series ───────────────────────────────────────────

class BanditSeriesTests(unittest.TestCase):
    def test_recall_before_the_epoch_or_in_sample_is_not_used(self):
        lp = [lp_row("2026-08-14", recall=0.7541),
              lp_row("2026-10-01", scope="in_sample_post_fit", recall=0.99)]
        lp += [lp_row(_d("2026-09-22", i), recall=0.13) for i in range(0, 58)]
        got = run_analyze(lp=lp)
        verdict, info = got["ml_history"]["recall_at_20"]
        self.assertEqual(verdict, "INSUFFICIENT")      # no n logged on these rows
        text = A.render(got)
        self.assertNotIn("0.7541", text)
        self.assertNotIn("0.99", text)

    def test_recall_with_logged_n_and_spans_gets_a_verdict(self):
        lp = []
        for i in range(8):
            last = _d("2026-11-19", 7 * i)
            first = _d(last, -6)
            lp.append(lp_row(_d(last, 1), recall=0.1 if i < 4 else 0.6,
                             n_pos=100, span=(first, last)))
        got = run_analyze(lp=lp)
        verdict, info = got["ml_history"]["recall_at_20"]
        self.assertEqual(verdict, "IMPROVING")
        self.assertEqual(info["early"]["n"], 400)
        text = A.render(got)
        self.assertIn("lift", text)
        self.assertIn("base", text)

    def test_ucb_line_agrees_with_itself_and_starts_at_the_epoch(self):
        lp = [lp_row("2026-04-07", scope=None, ucb=0.0473)]
        lp += [lp_row(_d("2026-09-22", i), ucb=u)
               for i, u in enumerate([0.06, 0.20, 0.20, 0.03, 0.03, 0.1046])]
        got = run_analyze(lp=lp)
        verdict, info = got["ml_history"]["ucb_separation"]
        self.assertEqual(verdict, "DIAGNOSTIC")
        self.assertEqual(info["first"], 0.06)
        self.assertEqual(info["first_date"], "2026-09-22")
        self.assertEqual(info["last"], 0.1046)
        self.assertAlmostEqual(info["last_minus_first"], round(0.1046 - 0.06, 4))
        self.assertNotIn("0.0473", A.render(got))

    def test_auc_on_immutable_close_timing_is_read(self):
        lp = [lp_row("2026-09-01", auc=0.898)]          # frozen-store epoch
        lp += [lp_row(_d("2026-09-22", i), auc=a)
               for i, a in enumerate([0.8475, 0.85, 0.8603])]
        verdict, info = run_analyze(lp=lp)["ml_history"]["auc_top20"]
        self.assertEqual(verdict, "DIAGNOSTIC")
        self.assertEqual(info["first"], 0.8475)
        self.assertEqual(info["last"], 0.8603)

    def test_n_signal_is_a_counter_not_a_trend(self):
        lp = [lp_row(_d("2026-09-22", i)) for i in range(3)]
        got = run_analyze(lp=lp)
        self.assertNotIn("n_signal", got["ml_history"])
        self.assertEqual(got["ml_history"]["counters"]["bandit_n_signal"], [25000, 25000])


# ── D3: roadmap from the target specs ─────────────────────────────────────────

TARGET_SPEC = """# Continuous-improvement agent

- **Status:** design (no implementation)
- **Created:** 2026-08-13 · **Revised:** 2026-08-14 after external review

## 13. Rollout

| Phase | Builds | Exit gate |
|---|---|---|
| **-1** | **Walking skeleton — named, not abstract.** `x.py` does a thing | gate |
| **0a** | Repair measurement: immutable labels, day-grouped splits | gate |
| 1 | Deterministic weekly report on `MoveEvent` | gate |

## 14. Open questions
"""

TRADER_ROADMAP = """# Trader roadmap

## 7. Спеки этапов (порядок написания = приоритет)

| # | спека | этап | статус |
|---|---|---|---|
| 1 | `daily-revenue-metric-1010-spec.md` — DailyRevenue v1 | раздел 1, D | спека |
| 12 | `carry-funding-money-1021-spec.md` — FC | E | **проверка 2026: ACCEPTED** — +0.0024 %/день |

## 8. Источники
"""

LEGACY_SPEC = """# Auto-Improvement Loop

- **Status:** **LEGACY / AS-BUILT** (2026-08-14)
- **Last updated:** 2026-08-13 (Truth Harness shipped)

## 5. Roadmap — prioritised

| ID | Item |
|---|---|
| RM-4 | Runtime config-override mechanism |
| RM-21 | Scan-coverage gap |

## 6. Update protocol
"""


class RoadmapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.docs = {}
        for name, text in [("continuous-improvement-agent-spec.md", TARGET_SPEC),
                           ("autonomous-trading-roadmap-1010-spec.md", TRADER_ROADMAP),
                           ("auto-improvement-loop-spec.md", LEGACY_SPEC)]:
            p = root / name
            p.write_text(text, encoding="utf-8")
            self.docs[name] = p

    def tearDown(self):
        self.tmp.cleanup()

    def test_target_spec_status_and_phases(self):
        t = A.parse_target_spec(self.docs["continuous-improvement-agent-spec.md"])
        self.assertEqual(t["status"], "design (no implementation)")
        self.assertEqual(t["revised"], "2026-08-14")
        self.assertEqual([p[0] for p in t["phases"]], ["-1", "0a", "1"])
        self.assertTrue(t["phases"][0][1].startswith("Walking skeleton"))

    def test_trader_roadmap_rows(self):
        r = A.parse_trader_roadmap(self.docs["autonomous-trading-roadmap-1010-spec.md"])
        self.assertEqual([row["spec"] for row in r["rows"]],
                         ["daily-revenue-metric-1010-spec.md",
                          "carry-funding-money-1021-spec.md"])
        self.assertEqual(r["rows"][0]["verdict"], "спека")
        self.assertEqual(r["rows"][1]["verdict"], "проверка 2026: ACCEPTED")
        self.assertEqual(r["rows"][1]["stage"], "E")

    def test_q5_reports_target_and_calls_legacy_stale_and_q6_drops_rm_items(self):
        got = run_analyze(docs=self.docs)
        text = A.render(got)
        q5 = text.split("[Q5]")[1].split("[Q6]")[0]
        q6 = text.split("[Q6]")[1]
        self.assertIn("design (no implementation)", q5)
        self.assertIn("Walking skeleton", q5)
        self.assertIn("not computed", q5)
        self.assertIn("carry-funding-money-1021-spec.md", q5)
        self.assertIn("LEGACY", q5)
        self.assertIn("not the plan", q5)
        self.assertNotIn("RM-21", q5)
        self.assertNotIn("RM-", q6)


# ── §3: the missing n is logged from now on ───────────────────────────────────

class SaveProgressTests(unittest.TestCase):
    def test_eval_positives_and_span_are_logged(self):
        import daily_learning as DL
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "learning_progress.jsonl"
            with mock.patch.object(DL, "RUNTIME_DIR", Path(tmp)), \
                 mock.patch.object(DL, "PROGRESS_FILE", path):
                DL.save_progress(
                    {}, {"bandit_accuracy": {
                        "evaluation_scope": OOS, "overall_recall_top20": 0.1333,
                        "total_positive": 15,
                        "daily": [{"day": "2026-11-17"}, {"day": "2026-11-11"},
                                  {"day": "2026-11-14"}]}}, {})
            row = json.loads(path.read_text(encoding="utf-8").strip())
        self.assertEqual(row["bandit_n_positive"], 15)
        self.assertEqual(row["bandit_eval_first_day"], "2026-11-11")
        self.assertEqual(row["bandit_eval_last_day"], "2026-11-17")


# ── D4: pipeline blockers (spec §6) ───────────────────────────────────────────

PEND = "pending_manual_validation"


def dec(did, ts, stage, key, hyp, verdict=PEND):
    d = {"decision_id": did, "ts": ts, "stage": stage, "config_key": key, "validation_verdict": verdict}
    if hyp is not None:
        d["hypothesis_id"] = hyp
    return d


MAY = [
    dec("31ae5d", "2026-05-11T11:28:00Z", "approved", "ENTRY_SCORE_FLOOR_GLOBAL", "h-floor"),
    dec("b4196d", "2026-05-11T11:28:05Z", "rolled_back", "ENTRY_SCORE_FLOOR_GLOBAL", None, verdict=None),
    dec("ce9fdf", "2026-05-11T21:35:40Z", "rejected", "WATCHLIST_MATCH_SCORE_TOLERANCE", "h-watch"),
    dec("f0df87", "2026-05-11T21:35:41Z", "rejected", "GATE_RANKER_HARD_VETO_THRESHOLD", "h-veto"),
    dec("4d6cb5", "2026-05-11T21:37:14Z", "approved", "MODE_IMPULSE_SPEED_ENABLED", "h-imp"),
    dec("a3a4c0", "2026-05-28T13:27:54Z", "deferred", "ENTRY_SCORE_FLOOR_GLOBAL", "h-floor", verdict="reject"),
    dec("f37987", "2026-05-28T13:27:55Z", "deferred", "MODE_IMPULSE_SPEED_ENABLED", "h-imp"),
    dec("dec001", "2026-05-07T12:00:00Z", "approved", "DECOUPLING_SHADOW_ENABLED", "h-dec", verdict="backtest_positive"),
]
KNOWN = {"DECOUPLING_SHADOW_ENABLED", "ENTRY_SCORE_MIN_15M", "CORR_MAX_PER_CLUSTER"}


class PipelineBlockerTests(unittest.TestCase):
    def test_may_records_are_not_blockers(self):
        c = A.classify_pending(MAY, KNOWN)
        self.assertEqual(c["blockers"], 0)
        self.assertEqual(c["raw"], 5)
        self.assertEqual(c["not_blocking"], {"superseded": 2, "closed_without_verdict": 2, "unknown_key": 1})
        self.assertTrue(c["keys_verified"])
        why = {d["decision_id"]: d["reason"] for d in c["detail"]}
        self.assertEqual(why, {"31ae5d": "superseded", "4d6cb5": "superseded", "ce9fdf": "closed_without_verdict",
                               "f0df87": "closed_without_verdict", "f37987": "unknown_key"})

    def test_latest_open_record_with_known_key_blocks(self):
        rows = MAY + [dec("n1", "2026-11-01T00:00:00Z", "approved", "CORR_MAX_PER_CLUSTER", "h-corr")]
        c = A.classify_pending(rows, KNOWN)
        self.assertEqual(c["blockers"], 1)
        self.assertIn("n1", [d["decision_id"] for d in c["detail"] if d["reason"] == "blocker"])

    def test_later_record_supersedes(self):
        rows = [dec("x1", "2026-11-01T00:00:00Z", "approved", "CORR_MAX_PER_CLUSTER", "h-corr"),
                dec("x2", "2026-11-02T00:00:00Z", "approved", "CORR_MAX_PER_CLUSTER", "h-corr", verdict="accept")]
        c = A.classify_pending(rows, KNOWN)
        self.assertEqual(c["blockers"], 0)
        self.assertEqual(c["not_blocking"], {"superseded": 1})

    def test_unreadable_config_never_lowers_the_count(self):
        c = A.classify_pending(MAY, set())
        self.assertFalse(c["keys_verified"])
        self.assertEqual(c["blockers"], 1)                  # f37987: latest, deferred, key unverifiable
        self.assertNotIn("unknown_key", c["not_blocking"])

    def test_thread_falls_back_to_config_key(self):
        rows = [dec("k1", "2026-11-01T00:00:00Z", "approved", "CORR_MAX_PER_CLUSTER", None),
                dec("k2", "2026-11-03T00:00:00Z", "rejected", "CORR_MAX_PER_CLUSTER", None, verdict="reject")]
        c = A.classify_pending(rows, KNOWN)
        self.assertEqual(c["blockers"], 0)
        self.assertEqual(c["not_blocking"], {"superseded": 1})

    def _render(self, decisions):
        with mock.patch.object(A, "load_learning_progress", return_value=[]), \
             mock.patch.object(A, "load_metrics_daily", return_value=[]), \
             mock.patch.object(A, "load_latest_health", return_value=HEALTH_VERIFIED), \
             mock.patch.object(A, "load_decisions", return_value=list(decisions)), \
             mock.patch.object(A, "load_config_keys", return_value=set(KNOWN)), \
             mock.patch.object(A, "load_attribution_meta", return_value=None), \
             mock.patch.object(A, "_find_doc", return_value=None), \
             mock.patch.object(A, "_git_last_commit_date", return_value=None):
            a = A.analyze(Path("."), None)
        return a, A.render(a)

    def test_render_without_blockers(self):
        a, txt = self._render(MAY)
        self.assertEqual(a["pipeline"]["pending_manual_validation"], 0)
        self.assertEqual(a["pipeline"]["pending_verdict_records"], 5)
        self.assertNotIn("pipeline blocker", txt)
        self.assertNotIn("Unblock pipeline", txt)
        self.assertIn("none blocking", txt)

    def test_render_with_a_blocker(self):
        rows = MAY + [dec("n1", "2026-11-01T00:00:00Z", "approved", "CORR_MAX_PER_CLUSTER", "h-corr")]
        a, txt = self._render(rows)
        self.assertEqual(a["pipeline"]["pending_manual_validation"], 1)
        self.assertIn("pipeline blocker", txt)
        self.assertIn("Unblock pipeline", txt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
