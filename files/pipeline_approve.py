"""L6 — Human Approval CLI.

Показывает оператору hypothesis + validation_report + shadow_report,
запрашивает y/n. При одобрении:
  - дописывает decisions.jsonl и already_tried.jsonl (memory)
  - печатает diff для config.py (НЕ применяет автоматически — человек применяет руками)
  - печатает рекомендованное commit message

При reject — фиксирует причину отказа в already_tried (чтобы L2 не предлагал то же снова).

При rollback — отдельная команда `--rollback <decision_id>` с обновлением decisions.jsonl.

Usage:
    pyembed\\python.exe files\\pipeline_approve.py --hypothesis h-2026-05-11-foo
    pyembed\\python.exe files\\pipeline_approve.py --hypothesis ... --approve  # non-interactive
    pyembed\\python.exe files\\pipeline_approve.py --hypothesis ... --reject "reason here"
    pyembed\\python.exe files\\pipeline_approve.py --rollback d-2026-05-12-001
    pyembed\\python.exe files\\pipeline_approve.py --list-pending
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import uuid

# Force UTF-8 stdout on Windows so arrows/emoji don't break cp1251 console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from datetime import datetime, timezone
from pathlib import Path

import pipeline_lib as PL
import pipeline_claude_client as CC
import pipeline_baseline as PB

ADVISORY_LOG = PL.PIPELINE / "approve_advisory.jsonl"


_CLAUDE_ADVISOR_SYSTEM = """You are the L6 ADVISOR for an AI-improvement pipeline.

Unlike the L5 critic (which is blind), you see EVERYTHING about this hypothesis:
the rationale, validation_report from L3, shadow_report from L4 if present,
the safety_checks output, and the decision history.

Your job: give the human operator a single recommendation — approve, reject, or
needs_review — with a one-paragraph justification covering:
  1. Whether L3 + L4 evidence supports the expected delta.
  2. Whether the risk section identifies a plausible failure mode the operator
     should watch for after applying the change.
  3. Whether the rollback procedure is clear and reversible.
  4. Whether the change targets a persistent (not flaky) red flag.

Be honest. If the hypothesis looks weak (small days_red, no shadow yet, expected
range overlapping with noise), say so. If the change is reasonable but the
operator should monitor a specific metric for the first 24h, say which one.

You are advice, not authority — the human and the safety_checks are the gate.
"""

_CLAUDE_ADVISOR_SCHEMA = (
    '{"recommendation": "approve"|"reject"|"needs_review", '
    '"confidence": "low"|"medium"|"high", '
    '"justification": str, '
    '"watch_after_apply": [str]}'
)


def claude_advise(hyp: dict, issues: list[str]) -> dict | None:
    """Get Claude's recommendation. Returns None if disabled.

    Always logged to ADVISORY_LOG, even in non-interactive mode."""
    if not CC.is_enabled():
        return None

    payload = {
        "hypothesis": {
            "hypothesis_id":   hyp.get("hypothesis_id"),
            "rule":            hyp.get("rule"),
            "config_key":      hyp.get("config_key"),
            "diff":            hyp.get("diff"),
            "rationale":       hyp.get("rationale"),
            "expected_delta":  hyp.get("expected_delta"),
            "risk":            hyp.get("risk"),
            "rollback":        hyp.get("rollback"),
            "severity":        hyp.get("severity"),
            "source_flag":     hyp.get("source_flag"),
            "persistence":     hyp.get("persistence"),
            "generator":       hyp.get("generator", "rule"),
        },
        "validation_report": (hyp.get("validation_report") or {}).get("result"),
        "shadow_report":     hyp.get("shadow_report"),
        "safety_checks":     issues,
    }
    res = CC.call_claude_json(
        _CLAUDE_ADVISOR_SYSTEM,
        json.dumps(payload, ensure_ascii=False, indent=2),
        schema_hint=_CLAUDE_ADVISOR_SCHEMA,
        max_tokens=800,
        layer="L6",
        purpose="approval_advice",
    )
    if not res:
        return None
    rec = (res.get("recommendation") or "").lower()
    if rec not in ("approve", "reject", "needs_review"):
        return None
    advice = {
        "ts":                PL.utc_now_iso(),
        "hypothesis_id":     hyp.get("hypothesis_id"),
        "recommendation":    rec,
        "confidence":        res.get("confidence", "unknown"),
        "justification":     res.get("justification", ""),
        "watch_after_apply": res.get("watch_after_apply", []),
        "model":             res.get("__claude_meta__", {}).get("model"),
        "cost_usd":          res.get("__claude_meta__", {}).get("cost_usd"),
    }
    PL.append_jsonl(ADVISORY_LOG, advice)
    return advice


def show_advice(advice: dict) -> None:
    print("")
    print("--- CLAUDE ADVISOR (L6) ---")
    rec = advice["recommendation"].upper()
    print(f"  recommendation: {rec}   (confidence: {advice['confidence']})")
    print(f"  justification:  {advice['justification']}")
    if advice.get("watch_after_apply"):
        print("  watch after apply:")
        for w in advice["watch_after_apply"]:
            print(f"    - {w}")
    print(f"  model: {advice.get('model')}   cost: ${advice.get('cost_usd')}")


def fmt_diff(diff: dict) -> str:
    if not diff:
        return "(no diff)"
    return f"  {diff.get('from')}  ->  {diff.get('to')}"


def show_hypothesis(hyp: dict) -> None:
    p = print
    p("=" * 70)
    p(f"Hypothesis: {hyp['hypothesis_id']}")
    p(f"Rule:       {hyp.get('rule')}")
    p(f"Status:     {hyp.get('status')}")
    p(f"Severity:   {hyp.get('severity')}")
    p(f"Source RF:  {hyp.get('source_flag')}")
    p("")
    p(f"Config key: {hyp.get('config_key')}")
    p(f"Diff:       {fmt_diff(hyp.get('diff', {}))}")
    p("")
    p(f"Rationale:  {hyp.get('rationale')}")
    p(f"Expected:   {hyp.get('expected_delta')}")
    p(f"Risk:       {hyp.get('risk')}")
    p(f"Rollback:   {hyp.get('rollback')}")
    p(f"Validation: {hyp.get('validation_required')}")
    p("")
    if hyp.get("persistence"):
        per = hyp["persistence"]
        p(f"Persistence: red for {per['days_red']} days")
    p("")
    vr = (hyp.get("validation_report") or {}).get("result", {})
    if vr:
        p("--- VALIDATION REPORT ---")
        p(f"  verdict: {vr.get('verdict')}")
        p(f"  reason:  {vr.get('reason')}")
        if vr.get("manual_steps"):
            p("  manual_steps:")
            for s in vr["manual_steps"]:
                p(f"    - {s}")
    else:
        p("--- VALIDATION REPORT ---  (none — run L3 first)")
    p("")
    sr = hyp.get("shadow_report")
    if sr:
        p("--- SHADOW REPORT ---")
        p(f"  verdict: {json.dumps(sr.get('verdict'), ensure_ascii=False)}")
        p(f"  n_events: {sr.get('summary', {}).get('n_events')}")
    else:
        p("--- SHADOW REPORT ---  (none — run L4 first)")
    p("=" * 70)


def safety_checks(hyp: dict) -> list[str]:
    """Return list of blocking issues."""
    issues = []
    if hyp.get("status") in ("rejected", "rolled_back"):
        issues.append(f"hypothesis already in terminal state: {hyp['status']}")
    if hyp.get("config_key") in set(PL.load_do_not_touch().get("config_keys_locked", [])):
        issues.append(f"config_key '{hyp['config_key']}' is in do_not_touch.config_keys_locked")
    if not hyp.get("validation_report"):
        issues.append("no validation_report — run L3 first")
    vr = (hyp.get("validation_report") or {}).get("result", {})
    if vr.get("verdict") == "reject":
        issues.append(f"L3 verdict=reject: {vr.get('reason')}")
    return issues


def record_decision(decision_id: str, hyp: dict, stage: str, extra: dict | None = None) -> None:
    rec = {
        "decision_id": decision_id,
        "ts": PL.utc_now_iso(),
        "hypothesis_id": hyp["hypothesis_id"],
        "stage": stage,                    # approved | rejected | rolled_back
        "rule": hyp.get("rule"),
        "config_key": hyp.get("config_key"),
        "diff": hyp.get("diff"),
        "expected": hyp.get("expected_delta"),
        "validation_verdict": (hyp.get("validation_report") or {}).get("result", {}).get("verdict"),
        "shadow_verdict": hyp.get("shadow_report", {}).get("verdict") if hyp.get("shadow_report") else None,
    }
    if extra:
        rec.update(extra)
    PL.append_jsonl(PL.DECISIONS_LOG, rec)

    # Also dedup memory for L2
    tried = {
        "ts": rec["ts"],
        "rule": hyp.get("rule"),
        "config_key": hyp.get("config_key"),
        "stage": stage,
        "decision_id": decision_id,
    }
    PL.append_jsonl(PL.ALREADY_TRIED, tried)


def emit_apply_instructions(hyp: dict) -> None:
    key = hyp.get("config_key")
    d = hyp.get("diff", {})
    print("")
    print("=" * 70)
    print("APPLY MANUALLY")
    print("=" * 70)
    print(f"1. Edit files/config.py:")
    print(f"     {key} = {d.get('to')}        # was {d.get('from')}")
    print("")
    print(f"2. Restart bot:  restart_bot.bat")
    print("")
    print(f"3. Suggested commit message:")
    print(f"     pipeline: apply {hyp['hypothesis_id']} ({key} {d.get('from')}->{d.get('to')})")
    print("")
    print(f"4. Schedule rollback check in 7 days:")
    print(f"     pyembed\\python.exe files\\pipeline_approve.py --rollback-check {hyp['hypothesis_id']}")


def cmd_list_pending() -> None:
    print(f"Pending hypotheses in {PL.HYPOTHESES}:")
    print("")
    print(f"{'ID':<55} {'STATUS':<22} {'SEVERITY':<10} RULE")
    print("-" * 110)
    for p in sorted(PL.HYPOTHESES.glob("h-*.json")):
        h = PL.read_json(p) or {}
        print(f"{h.get('hypothesis_id',''):<55} {h.get('status',''):<22} {h.get('severity',''):<10} {h.get('rule','')}")


def cmd_approve(hyp_path: Path, non_interactive: bool, force: bool, reason: str | None) -> None:
    """Approve a hypothesis.

    non_interactive: skip y/n prompt (but still respect safety_checks)
    force:           override safety_checks (separate, much riskier flag)
    """
    hyp = PL.read_json(hyp_path)
    if not hyp:
        print(f"ERROR: {hyp_path} not readable")
        sys.exit(1)

    show_hypothesis(hyp)

    issues = safety_checks(hyp)
    if issues:
        print("")
        print("SAFETY CHECKS FAILED:")
        for i in issues:
            print(f"  - {i}")
        if not force:
            print("")
            print("Use --force to override (not recommended). --approve alone is NOT enough.")
            sys.exit(2)
        print("--force set: proceeding despite issues.")

    # Claude advisory — always recorded for audit, shown only when interactive.
    advice = claude_advise(hyp, issues)
    if advice and not non_interactive:
        show_advice(advice)

    if not non_interactive:
        try:
            answer = input("\nApprove this change? [y/N]: ").strip().lower()
        except EOFError:
            answer = "n"
        if answer != "y":
            print("Aborted (not approved).")
            sys.exit(0)

    decision_id = f"d-{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    extra = {"approval_reason": reason}
    if advice:
        extra["claude_advice"] = {
            "recommendation": advice["recommendation"],
            "confidence":     advice["confidence"],
            "model":          advice.get("model"),
        }
    record_decision(decision_id, hyp, "approved", extra)

    # Pin a baseline snapshot of all tracked metrics BEFORE this change goes
    # live. Without this, pipeline_attribution can't honestly tell apart the
    # change's effect from market drift. Failure here is non-fatal: we log
    # but proceed so the operator still gets the apply instructions.
    try:
        bp = PB.pin_baseline(decision_id, datetime.now(timezone.utc))
        print(f"[baseline] pinned: {bp.name}")
    except Exception as e:
        print(f"[baseline] WARNING: pin failed: {e!r} — attribution will be unavailable")

    hyp["status"] = "approved"
    hyp["decision_id"] = decision_id
    PL.write_json(hyp_path, hyp)
    print(f"\nApproved: decision_id={decision_id}")
    emit_apply_instructions(hyp)


def cmd_reject(hyp_path: Path, reason: str) -> None:
    hyp = PL.read_json(hyp_path)
    if not hyp:
        print(f"ERROR: {hyp_path} not readable")
        sys.exit(1)
    decision_id = f"d-{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    record_decision(decision_id, hyp, "rejected", {"rejection_reason": reason})
    hyp["status"] = "rejected"
    hyp["decision_id"] = decision_id
    PL.write_json(hyp_path, hyp)
    print(f"Rejected: decision_id={decision_id} reason={reason!r}")


def cmd_rollback(decision_id: str, reason: str) -> None:
    # Find the original approved decision
    original = None
    for rec in PL.iter_jsonl(PL.DECISIONS_LOG):
        if rec.get("decision_id") == decision_id and rec.get("stage") == "approved":
            original = rec
            break
    if not original:
        print(f"ERROR: no approved decision found for {decision_id}")
        sys.exit(1)
    diff = original.get("diff", {})
    rollback_diff = {"from": diff.get("to"), "to": diff.get("from")}
    new_decision_id = f"d-{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    rec = {
        "decision_id": new_decision_id,
        "ts": PL.utc_now_iso(),
        "stage": "rolled_back",
        "rolling_back": decision_id,
        "rule": original.get("rule"),
        "config_key": original.get("config_key"),
        "diff": rollback_diff,
        "reason": reason,
    }
    PL.append_jsonl(PL.DECISIONS_LOG, rec)
    PL.append_jsonl(PL.ALREADY_TRIED, {
        "ts": rec["ts"], "rule": original.get("rule"),
        "config_key": original.get("config_key"),
        "stage": "rolled_back", "decision_id": new_decision_id,
    })
    print(f"Rollback recorded: {new_decision_id}")
    print(f"\nMANUAL STEP — revert files/config.py:")
    print(f"  {original.get('config_key')} = {rollback_diff['to']}    # was {rollback_diff['from']}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--hypothesis", help="approve/reject this hypothesis_id")
    g.add_argument("--hypothesis-file", help="path to hypothesis JSON")
    g.add_argument("--rollback", help="decision_id to roll back")
    g.add_argument("--list-pending", action="store_true")
    ap.add_argument("--approve", action="store_true", help="non-interactive approve")
    ap.add_argument("--reject", help="reject with reason")
    ap.add_argument("--force", action="store_true", help="override safety checks")
    ap.add_argument("--reason", help="reason for the decision (free text)")
    args = ap.parse_args()

    if args.list_pending:
        cmd_list_pending()
        return

    if args.rollback:
        cmd_rollback(args.rollback, args.reason or "unspecified")
        return

    if args.hypothesis_file:
        hp = Path(args.hypothesis_file)
    else:
        hp = PL.HYPOTHESES / f"{args.hypothesis}.json"

    if args.reject:
        cmd_reject(hp, args.reject)
        return

    cmd_approve(hp, non_interactive=args.approve, force=args.force, reason=args.reason)


if __name__ == "__main__":
    main()
