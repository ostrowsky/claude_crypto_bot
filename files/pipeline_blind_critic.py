"""L5 — Blind Regression Critic.

Принимает decision_id и выносит независимый вердикт об эффекте на основе
ТОЛЬКО метрик до и после, БЕЗ доступа к hypothesis reasoning / validation /
shadow reports. Это режет confirmation bias L1->L4: если бы критик видел
обоснование, он бы их повторил.

Что критик видит:
  - decision.config_key, decision.diff, decision.ts, decision.expected
  - daily health reports в [ts - 7d, ts] и [ts, ts + 7d]
  - 7-day rolling baseline из top_gainer_critic_history.jsonl

Что НЕ видит:
  - hypothesis_id (только косвенно через rule в decision)
  - validation_report (L3 grading)
  - shadow_report (L4 results)
  - rationale, expected_delta interpretation

Verdict:
  - APPROVE — expected метрика улучшилась AND нет регрессий в других >= 2pp
  - REJECT — expected не улучшилась ИЛИ есть регрессия в чувствительной метрике
  - NEEDS_DATA — недостаточно дней с обеих сторон

Чувствительные метрики (любое падение >= 2pp = critical):
  - watchlist_top_bought_pct (north-star adjacent)
  - watchlist_top_early_capture_pct (north-star)
  - recall_at_20 (training)
  - training_to_live_gap (если выросла — деградация)

Usage:
    pyembed\\python.exe files\\pipeline_blind_critic.py --decision d-2026-05-11T...
    pyembed\\python.exe files\\pipeline_blind_critic.py --all-pending   # все решения без вердикта критика
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

import pipeline_lib as PL
import pipeline_claude_client as CC

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CRITIC_DIR = PL.PIPELINE / "critic"
CRITIC_DIR.mkdir(parents=True, exist_ok=True)

SENSITIVE_METRICS = [
    "watchlist_top_bought_pct",
    "watchlist_top_early_capture_pct",
    "recall_at_20",
]
SENSITIVE_REGRESSION_THRESHOLD = 0.02   # 2pp drop = critical
EXPECTED_IMPROVEMENT_TOLERANCE = 0.005  # within 0.5pp of expected_min still counts


# ---------------------------------------------------------------------------
# Helpers (mirror pipeline_monitor; kept separate to enforce that critic does
# NOT import outcome interpretations)
# ---------------------------------------------------------------------------


def _parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None


def _load_health_in_window(start: datetime, end: datetime) -> list[dict]:
    out = []
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


def _extract(report: dict, metric: str) -> float | None:
    dh = report.get("deployment_health") or {}
    th = report.get("training_health") or {}
    if metric in dh: return dh.get(metric)
    if metric in th: return th.get(metric)
    if metric == "training_to_live_gap":
        return (report.get("training_to_live_gap") or {}).get("value")
    return None


def _median_or_none(vals: list[float | None]) -> float | None:
    vals = [v for v in vals if v is not None]
    return median(vals) if vals else None


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


_CLAUDE_CRITIC_SYSTEM = """You are an L5 BLIND regression critic for an AI-improvement pipeline.

YOU SEE ONLY metrics-before vs metrics-after, plus a minimal description of the
change (config_key, diff, expected metric range). You do NOT see the rationale,
the validation report, or the shadow report.

This is by design: if you saw the reasoning, you would echo it, defeating the
purpose of a blind review.

Your task: decide whether the change ACTUALLY worked, independently.

Definitions:
 - "approve" — the expected metric(s) moved within or above the expected range,
   AND no sensitive metric regressed by >= 2pp.
 - "reject" — expected metric(s) failed to move per the expected range, OR a
   sensitive metric regressed by >= 2pp.
 - "needs_data" — fewer than 2 daily reports on either side.

Sensitive metrics (any drop >= 2pp = critical):
   watchlist_top_bought_pct,
   watchlist_top_early_capture_pct,
   recall_at_20.

Reasoning rules:
 1. Treat small noisy moves (|delta| < 0.5pp) on non-target metrics as non-events.
 2. If expected_range is "+lo..+hi", "approved" requires delta >= (lo - 0.5pp).
 3. Never speculate about WHY the change worked or didn't — that is L7's job.
 4. Be terse. One short reason sentence.
"""

_CLAUDE_CRITIC_SCHEMA = (
    '{"verdict": "approve"|"reject"|"needs_data", "reason": str, '
    '"key_metric_movements": [{"metric": str, "before": number, "after": number, "delta": number}]}'
)


def _build_blind_snapshot(decision: dict, before: list[dict], after: list[dict]) -> dict:
    """Extract ONLY numerical metrics — no rationale, no hypothesis text."""
    tracked = SENSITIVE_METRICS + list((decision.get("expected") or {}).keys())
    tracked = list(dict.fromkeys(tracked))   # dedup preserve order

    def snapshot(reports: list[dict]) -> list[dict]:
        out = []
        for r in reports:
            row = {"target_date": r.get("target_date")}
            for m in tracked:
                row[m] = _extract(r, m)
            out.append(row)
        return out

    return {
        # Minimal, BLIND info: what changed and what target was, no rationale.
        "change": {
            "config_key":     decision.get("config_key"),
            "diff":           decision.get("diff"),
            "ts":             decision.get("ts"),
            "expected_range": decision.get("expected"),
        },
        "tracked_metrics":   tracked,
        "before_window":     snapshot(before),
        "after_window":      snapshot(after),
    }


def claude_critique(decision: dict, before: list[dict], after: list[dict]) -> dict | None:
    """Second, independent blind verdict from Claude. Returns None if disabled."""
    if not CC.is_enabled():
        return None
    snapshot = _build_blind_snapshot(decision, before, after)
    res = CC.call_claude_json(
        _CLAUDE_CRITIC_SYSTEM,
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        schema_hint=_CLAUDE_CRITIC_SCHEMA,
        max_tokens=600,
        layer="L5",
        purpose="blind_critique",
    )
    if not res:
        return None
    verdict = res.get("verdict", "").lower()
    if verdict not in ("approve", "reject", "needs_data"):
        return None
    return {
        "verdict":           verdict,
        "reason":            res.get("reason", ""),
        "key_metric_moves":  res.get("key_metric_movements", []),
        "blind_inputs_only": True,
        "model":             res.get("__claude_meta__", {}).get("model"),
        "cost_usd":          res.get("__claude_meta__", {}).get("cost_usd"),
    }


def critique(decision: dict, window_days: int = 7) -> dict:
    ts = _parse_iso(decision.get("ts", ""))
    if ts is None:
        return {"verdict": "needs_data", "reason": "no parseable ts"}

    before = _load_health_in_window(ts - timedelta(days=window_days), ts)
    after  = _load_health_in_window(ts, ts + timedelta(days=window_days))

    if len(before) < 2 or len(after) < 2:
        return {
            "verdict": "needs_data",
            "reason": f"min 2 reports each side required (before={len(before)}, after={len(after)})",
        }

    # Check expected metric(s)
    expected = decision.get("expected") or {}
    expected_metrics = list(expected.keys())
    expected_pass = []
    for em in expected_metrics:
        b = _median_or_none([_extract(r, em) for r in before])
        a = _median_or_none([_extract(r, em) for r in after])
        if b is None or a is None:
            expected_pass.append({"metric": em, "status": "unknown", "before": b, "after": a})
            continue
        delta = a - b
        # parse expected range (lo..hi)
        s = str(expected[em])
        try:
            parts = s.split("..")
            lo = float(parts[0])
            improved = delta >= lo - EXPECTED_IMPROVEMENT_TOLERANCE
        except (ValueError, IndexError):
            improved = delta > 0
        expected_pass.append({
            "metric": em, "before": b, "after": a, "delta": round(delta, 6),
            "improved_per_expected": improved,
        })

    # Check sensitive metrics for regression
    regressions = []
    for sm in SENSITIVE_METRICS:
        b = _median_or_none([_extract(r, sm) for r in before])
        a = _median_or_none([_extract(r, sm) for r in after])
        if b is None or a is None:
            continue
        delta = a - b
        if delta < -SENSITIVE_REGRESSION_THRESHOLD:
            regressions.append({"metric": sm, "before": b, "after": a, "delta": round(delta, 6)})

    expected_ok = expected_pass and all(p.get("improved_per_expected") for p in expected_pass if p.get("status") != "unknown")

    if regressions:
        verdict = "reject"
        reason = f"regression in {len(regressions)} sensitive metric(s): " + ", ".join(r["metric"] for r in regressions)
    elif expected_ok:
        verdict = "approve"
        reason = "expected metric(s) improved per range, no sensitive regression"
    else:
        verdict = "reject"
        reason = "expected metric(s) did not improve per range"

    rule_result = {
        "verdict": verdict,
        "reason": reason,
        "n_before": len(before),
        "n_after":  len(after),
        "expected_metrics_check": expected_pass,
        "sensitive_regressions":  regressions,
        "blind_inputs_only": True,
    }

    # Hybrid: ask Claude for an independent verdict on the SAME blind data.
    # Both critics see only numbers — no hypothesis_id, no rationale, no L3/L4.
    claude_result = claude_critique(decision, before, after)
    if claude_result is not None:
        rule_result["rule_verdict"] = rule_result["verdict"]
        rule_result["claude_verdict"] = claude_result["verdict"]
        rule_result["claude_reason"]  = claude_result["reason"]
        rule_result["claude_key_moves"] = claude_result["key_metric_moves"]
        rule_result["claude_cost_usd"] = claude_result["cost_usd"]
        rule_result["agreement"] = (rule_result["verdict"] == claude_result["verdict"])
        # When critics disagree, downgrade to needs_review so a human looks.
        # This is the whole point of running both: catch what either alone missed.
        if not rule_result["agreement"]:
            rule_result["verdict"] = "needs_review"
            rule_result["reason"]  = (
                f"critics disagree — rule={rule_result['rule_verdict']} "
                f"({rule_result['reason']}); claude={claude_result['verdict']} "
                f"({claude_result['reason']})"
            )
    else:
        rule_result["claude_verdict"] = None
        rule_result["agreement"]      = None

    return rule_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_decision(decision_id: str) -> dict | None:
    for rec in PL.iter_jsonl(PL.DECISIONS_LOG):
        if rec.get("decision_id") == decision_id:
            return rec
    return None


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--decision", help="decision_id to critique")
    g.add_argument("--all-pending", action="store_true",
                   help="critique all approved decisions without existing critic verdict")
    ap.add_argument("--window-days", type=int, default=7)
    ap.add_argument("--print", dest="do_print", action="store_true")
    args = ap.parse_args()

    decisions = []
    if args.decision:
        d = _load_decision(args.decision)
        if not d:
            print(f"ERROR: decision not found: {args.decision}")
            sys.exit(1)
        decisions = [d]
    else:
        # Find approved decisions without existing critic record
        existing = set()
        critic_log = CRITIC_DIR / "blind_critic_verdicts.jsonl"
        if critic_log.exists():
            for rec in PL.iter_jsonl(critic_log):
                existing.add(rec.get("decision_id"))
        for rec in PL.iter_jsonl(PL.DECISIONS_LOG):
            if rec.get("stage") == "approved" and rec.get("decision_id") not in existing:
                decisions.append(rec)

    results = []
    for d in decisions:
        critic_input = {
            "decision_id": d["decision_id"],
            "ts": d["ts"],
            "config_key": d.get("config_key"),
            "diff": d.get("diff"),
            "expected": d.get("expected"),
        }
        verdict = critique(d, args.window_days)
        result = {
            "critic_run_id": f"critic-{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')}-{d['decision_id'][-6:]}",
            "generated_at": PL.utc_now_iso(),
            "decision_id": d["decision_id"],
            "critic_inputs": critic_input,
            "result": verdict,
        }
        results.append(result)

    critic_log = CRITIC_DIR / "blind_critic_verdicts.jsonl"
    for r in results:
        PL.append_jsonl(critic_log, r)

    if args.do_print:
        print(json.dumps(results, indent=2, ensure_ascii=False))

    for r in results:
        v = r["result"]
        print(f"[L5] {r['decision_id']}: verdict={v['verdict']} reason={v['reason']}")


if __name__ == "__main__":
    main()
