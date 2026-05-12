"""L7 — Auto-rollback Monitor.

Daily cron. Читает decisions.jsonl, для каждого approved decision возраста
[outcome_check_after_days .. outcome_window_days]:
  - находит daily health reports в окне до и после approval_ts
  - считает actual delta для метрики из expected_delta
  - сравнивает с expected min/max range
  - emit rollback recommendation в .runtime/pipeline/monitor/

Также считает meta-metrics pipeline:
  - hit_rate = N decisions with verdict=hit / total decisions older than 7d
  - false_acceptance_rate = N hypothetical rollbacks / total accepted

Usage:
    pyembed\\python.exe files\\pipeline_monitor.py
    pyembed\\python.exe files\\pipeline_monitor.py --check-after-days 7 --print
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median

import pipeline_lib as PL

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MONITOR_DIR = PL.PIPELINE / "monitor"
MONITOR_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CHECK_AFTER_DAYS = 7
DEFAULT_WINDOW_BEFORE = 7   # days of history reports to take as "before" baseline
DEFAULT_WINDOW_AFTER = 7    # days of history to take as "after"


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def _parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None


def _load_health_in_window(start: datetime, end: datetime) -> list[dict]:
    """Read all health-{date}.json files with target_date in [start, end]."""
    out = []
    if not PL.HEALTH.exists():
        return out
    for p in sorted(PL.HEALTH.glob("health-*.json")):
        h = PL.read_json(p)
        if not h:
            continue
        try:
            d = datetime.fromisoformat(h["target_date"]).replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            continue
        if start <= d <= end:
            out.append(h)
    return out


def _extract_metric(report: dict, metric_name: str) -> float | None:
    """Look for a metric path in a daily health report.
       Known mappings:
         watchlist_top_early_capture_pct -> deployment_health.watchlist_top_early_capture_pct
         watchlist_top_bought_pct        -> deployment_health.watchlist_top_bought_pct
         false_positive_rate             -> deployment_health.false_positive_rate
         recall_at_20                    -> training_health.recall_at_20
         total_realized_pnl_pct          -> exit_quality.modes[*].total_realized_pnl_pct (NB: requires mode)
    """
    # Common metric→path mappings; extend as new red_flag types are added.
    dh = report.get("deployment_health") or {}
    th = report.get("training_health") or {}
    if metric_name in dh:
        return dh.get(metric_name)
    if metric_name in th:
        return th.get(metric_name)
    if metric_name == "training_to_live_gap":
        return (report.get("training_to_live_gap") or {}).get("value")
    return None


def _parse_expected_delta(expected: dict, metric_name: str) -> tuple[float, float] | None:
    """Parse strings like '+0.10..+0.20' or '-0.05..-0.10' to (min, max).
       If a single bound is given ('+0.10..'), the other is +/-inf."""
    if not expected or metric_name not in expected:
        return None
    s = str(expected[metric_name]).strip()
    m = re.match(r"([+\-]?\d+\.\d+)\s*\.\.\s*([+\-]?\d+\.\d+)?", s)
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2)) if m.group(2) else float("inf")
        return min(lo, hi), max(lo, hi)
    try:
        v = float(s)
        return v, v
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Outcome check
# ---------------------------------------------------------------------------


def evaluate_decision(decision: dict,
                      window_before: int = DEFAULT_WINDOW_BEFORE,
                      window_after: int = DEFAULT_WINDOW_AFTER) -> dict:
    ts = _parse_iso(decision.get("ts", ""))
    if ts is None:
        return {"verdict": "skip", "reason": "no parseable ts"}
    expected = decision.get("expected") or {}
    metric_keys = list(expected.keys())
    if not metric_keys:
        return {"verdict": "skip", "reason": "no expected_delta to verify"}

    before_window = _load_health_in_window(ts - timedelta(days=window_before), ts)
    after_window  = _load_health_in_window(ts, ts + timedelta(days=window_after))

    if len(before_window) < 2 or len(after_window) < 2:
        return {
            "verdict": "needs_data",
            "reason": f"insufficient health reports (before={len(before_window)}, after={len(after_window)})",
            "min_required": "2 each side",
        }

    per_metric = {}
    overall_hit = True
    for metric in metric_keys:
        before_vals = [_extract_metric(r, metric) for r in before_window]
        after_vals  = [_extract_metric(r, metric) for r in after_window]
        before_vals = [v for v in before_vals if v is not None]
        after_vals  = [v for v in after_vals  if v is not None]
        if not before_vals or not after_vals:
            per_metric[metric] = {"status": "unknown", "reason": "no values found in reports"}
            overall_hit = False
            continue
        before_median = median(before_vals)
        after_median  = median(after_vals)
        actual_delta  = round(after_median - before_median, 6)

        expected_range = _parse_expected_delta(expected, metric)
        in_range = False
        if expected_range is not None:
            lo, hi = expected_range
            in_range = lo <= actual_delta <= hi
        per_metric[metric] = {
            "before_median": before_median,
            "after_median":  after_median,
            "actual_delta":  actual_delta,
            "expected_range": expected_range,
            "in_range": in_range,
        }
        if not in_range:
            overall_hit = False

    return {
        "verdict": "hit" if overall_hit else "miss",
        "per_metric": per_metric,
        "n_before_reports": len(before_window),
        "n_after_reports":  len(after_window),
    }


# ---------------------------------------------------------------------------
# Pipeline metameetrics
# ---------------------------------------------------------------------------


def compute_pipeline_metameetrics(decisions: list[dict], outcomes: dict[str, dict]) -> dict:
    """a / (a+b) hit rate, ignoring needs_data/skip."""
    a = b = needs_data = skipped = 0
    for d in decisions:
        if d.get("stage") != "approved":
            continue
        v = outcomes.get(d["decision_id"], {}).get("verdict")
        if v == "hit": a += 1
        elif v == "miss": b += 1
        elif v == "needs_data": needs_data += 1
        else: skipped += 1
    total = a + b
    return {
        "approved_total":      a + b + needs_data,
        "evaluated_total":     total,
        "hits":               a,
        "misses":             b,
        "needs_data":         needs_data,
        "skipped":            skipped,
        "hit_rate":           round(a / total, 4) if total > 0 else None,
        "interpretation":     "hit_rate < 0.60 за квартал → pipeline хуже монетки, требует пересмотра",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-after-days", type=int, default=DEFAULT_CHECK_AFTER_DAYS,
                    help="minimum age of decision (days) before checking outcome")
    ap.add_argument("--window-before", type=int, default=DEFAULT_WINDOW_BEFORE)
    ap.add_argument("--window-after", type=int, default=DEFAULT_WINDOW_AFTER)
    ap.add_argument("--print", dest="do_print", action="store_true")
    args = ap.parse_args()

    decisions = list(PL.iter_jsonl(PL.DECISIONS_LOG))
    if not decisions:
        print("[L7] no decisions in log")
        return

    now = datetime.now(timezone.utc)
    cutoff_max_age = now - timedelta(days=args.check_after_days)
    rollback_recs = []
    outcomes = {}

    for d in decisions:
        if d.get("stage") != "approved":
            continue
        ts = _parse_iso(d.get("ts", ""))
        if ts is None or ts > cutoff_max_age:
            continue
        result = evaluate_decision(d, args.window_before, args.window_after)
        outcomes[d["decision_id"]] = result
        if result["verdict"] == "miss":
            rollback_recs.append({
                "decision_id": d["decision_id"],
                "hypothesis_id": d.get("hypothesis_id"),
                "rule": d.get("rule"),
                "config_key": d.get("config_key"),
                "diff_applied": d.get("diff"),
                "outcome": result,
                "recommended_action": f"pyembed\\python.exe files\\pipeline_approve.py --rollback {d['decision_id']} --reason 'expected_delta not met'",
            })

    meta = compute_pipeline_metameetrics(decisions, outcomes)

    report = {
        "report_id": f"monitor-{now.strftime('%Y-%m-%dT%H%M%SZ')}",
        "generated_at": PL.utc_now_iso(),
        "check_after_days": args.check_after_days,
        "window_before": args.window_before,
        "window_after": args.window_after,
        "n_decisions_total": len(decisions),
        "n_approved_evaluated": len([d for d in decisions if d.get("stage") == "approved" and _parse_iso(d.get("ts","")) and _parse_iso(d["ts"]) <= cutoff_max_age]),
        "rollback_recommendations": rollback_recs,
        "outcomes": outcomes,
        "pipeline_metameetrics": meta,
    }

    out_path = MONITOR_DIR / f"{report['report_id']}.json"
    PL.write_json(out_path, report)
    print(f"[L7] wrote {out_path}")

    rec_log = MONITOR_DIR / "rollback_recommendations.jsonl"
    for rec in rollback_recs:
        PL.append_jsonl(rec_log, {**rec, "_report_id": report["report_id"]})
    if rollback_recs:
        print(f"[L7] appended {len(rollback_recs)} rollback recommendation(s) to {rec_log}")

    print(f"[L7] hit_rate={meta['hit_rate']} (hits={meta['hits']} misses={meta['misses']} needs_data={meta['needs_data']})")

    if args.do_print:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
