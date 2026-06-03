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

import re

import pipeline_lib as PL
import pipeline_claude_client as CC
import pipeline_baseline as PB

ADVISORY_LOG = PL.PIPELINE / "approve_advisory.jsonl"
CONFIG_PY = PL.FILES_DIR / "config.py"

# Rule/gate -> the REAL config.py constant(s) it controls. L2 (and Claude)
# frequently emit a synthetic `config_key` like
# GATE_LATE_IMPULSE_ROTATION_THRESHOLD that does NOT exist in config.py,
# producing an un-appliable instruction. This registry maps a hypothesis
# to the constants an operator must actually edit. Extend as new rule
# families gain validators.
RULE_CONFIG_MAP: dict[str, list[str]] = {
    "relax_gate_late_impulse_rotation": [
        "IMPULSE_SPEED_ROTATION_GUARD_15M_RSI_MIN",
        "IMPULSE_SPEED_ROTATION_GUARD_15M_RANGE_MIN",
        "IMPULSE_SPEED_ROTATION_GUARD_1H_RSI_MIN",
        "IMPULSE_SPEED_ROTATION_GUARD_1H_RANGE_MIN",
    ],
    "entry_score_floor_relax": ["ENTRY_SCORE_MIN_15M", "ENTRY_SCORE_MIN_1H"],
    "impulse_guard_high_momentum_bypass": [
        "IMPULSE_SPEED_15M_HIGH_MOMENTUM_BYPASS_ENABLED",
        "IMPULSE_SPEED_1H_HIGH_MOMENTUM_BYPASS_ENABLED",
    ],
    "disable_mode_impulse_speed": ["MODE_IMPULSE_SPEED_ENABLED"],
}

# Known gate reason_codes as logged in critic_dataset.jsonl (decision.reason_code).
# Used to recompute a gate's blocked-bucket edge on RECENT vs REFERENCE windows so
# the L6 advisor isn't fooled by a stale long-window average (markets are
# non-stationary — an edge measured over 60d can already be decaying). Longest
# tokens first so e.g. 'late_impulse_rotation' wins over 'impulse'.
_GATE_REASON_CODES = [
    "late_impulse_rotation", "ranker_hard_veto", "clone_signal_guard",
    "correlation_guard", "open_cluster_cap", "fast_reversal_risk",
    "ml_proba_zone", "trend_1h_chop", "trend_quality", "impulse_guard",
    "entry_score", "breakout", "alignment", "mtf",
]
_RECENT_DAYS = 30
_REFERENCE_DAYS = 60
_CRITIC_DATASET = PL.FILES_DIR / "critic_dataset.jsonl"


def _gate_recency_split(hyp: dict) -> dict | None:
    """Recompute the gate's blocked-bucket forward return (ret_5) on a RECENT
    (<=30d) vs REFERENCE (30..60d) window from critic_dataset.jsonl.

    Defensive: returns None on any problem — must never block an approval flow.
    The point is to expose edge DECAY to the advisor: if the recent window shows
    a much weaker (or flipped) edge than the reference window, a fixed-threshold
    relaxation calibrated on the long window is over-optimistic."""
    try:
        if not _CRITIC_DATASET.exists():
            return None
        hay = f"{hyp.get('rule','')} {hyp.get('config_key','')}".lower()
        reason = next((rc for rc in _GATE_REASON_CODES if rc in hay), None)
        if not reason:
            return None
        now = datetime.now(timezone.utc)

        def _f(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        recent: list[float] = []
        reference: list[float] = []
        with io.open(_CRITIC_DATASET, encoding="utf-8", errors="replace") as f:
            for ln in f:
                if reason not in ln:
                    continue
                try:
                    e = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if str((e.get("decision", {}) or {}).get("reason_code", "")) != reason:
                    continue
                r5 = _f((e.get("labels", {}) or {}).get("ret_5"))
                if r5 is None:
                    continue
                try:
                    age = (now - datetime.fromisoformat(
                        str(e.get("ts_signal", "")).replace("Z", "+00:00"))).days
                except (ValueError, TypeError):
                    continue
                if age < 0:
                    continue
                if age <= _RECENT_DAYS:
                    recent.append(r5)
                elif age <= _REFERENCE_DAYS:
                    reference.append(r5)

        def _agg(xs):
            if not xs:
                return None
            n = len(xs)
            return {"n": n, "avg_r5_pct": round(sum(xs) / n, 4),
                    "win_pct": round(sum(1 for x in xs if x > 0) / n * 100, 1)}

        rec, ref = _agg(recent), _agg(reference)
        if rec is None and ref is None:
            return None
        decay = None
        if rec and ref and ref["avg_r5_pct"] != 0:
            ratio = rec["avg_r5_pct"] / ref["avg_r5_pct"]
            # edge decayed if recent is <50% of reference, or flipped sign
            decay = bool(ratio < 0.5 or (ref["avg_r5_pct"] > 0 and rec["avg_r5_pct"] <= 0))
        return {
            "reason_code": reason,
            "metric": "blocked-bucket forward ret_5 (%)",
            "recent_window_days": _RECENT_DAYS,
            "reference_window_days": f"{_RECENT_DAYS}..{_REFERENCE_DAYS}",
            "recent": rec,
            "reference": ref,
            "edge_decayed": decay,
            "note": ("Recent edge is materially weaker/flipped vs reference — a "
                     "fixed-threshold change calibrated on the long window is "
                     "likely over-optimistic; prefer needs_review."
                     if decay else
                     "Recent edge roughly tracks reference."),
        }
    except Exception:
        return None


def _config_constants() -> dict[str, str]:
    """Top-level CONST names -> their literal RHS, parsed from config.py
    without importing it (config.py has import side effects). Matches
    `NAME: type = value` and `NAME = value`."""
    out: dict[str, str] = {}
    if not CONFIG_PY.exists():
        return out
    pat = re.compile(r"^([A-Z][A-Z0-9_]*)\s*(?::[^=]+)?=\s*(.+?)\s*(?:#.*)?$")
    for line in CONFIG_PY.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.match(line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def resolve_config_keys(hyp: dict) -> dict:
    """Map a hypothesis to REAL config.py constants.

    Returns {"ok": bool, "keys": [...], "missing": [...], "reason": str}.
    ok=False means the operator literally cannot apply this as written —
    L2 emitted a placeholder. This is the L2-mapping guard.
    """
    consts = _config_constants()
    rule = hyp.get("rule", "")
    declared = hyp.get("config_key")

    # 1) explicit registry mapping for the rule. If the hypothesis ALSO
    #    declares a config_key that is in the mapped set, narrow to it
    #    (rule families can cover several tf variants; the hypothesis
    #    targets one).
    if rule in RULE_CONFIG_MAP:
        mapped = RULE_CONFIG_MAP[rule]
        if declared and declared in mapped:
            keys = [declared]
        else:
            keys = mapped
        missing = [k for k in keys if k not in consts]
        return {
            "ok": not missing,
            "keys": keys,
            "missing": missing,
            "reason": ("declared config_key matches mapped — narrowed" if declared in mapped
                       else ("all mapped constants exist" if not missing
                             else f"registry constants missing from config.py: {missing}")),
        }
    # 2) the declared config_key is itself a real constant
    if declared and declared in consts:
        return {"ok": True, "keys": [declared], "missing": [],
                "reason": "declared config_key exists in config.py"}
    # 3) unresolvable -> L2 emitted a synthetic key
    return {
        "ok": False,
        "keys": [],
        "missing": [declared] if declared else [],
        "reason": (f"config_key '{declared}' does not exist in config.py and "
                   f"rule '{rule}' has no entry in RULE_CONFIG_MAP — L2 emitted "
                   f"a placeholder; needs a real-constant mapping before apply"),
    }


def _is_concrete_value(v) -> bool:
    """A diff target is appliable only if it's a real Python literal,
    not a directive like '+10% looser' or 'current'."""
    if isinstance(v, (int, float, bool)):
        return True
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("current", "") or "%" in s or " " in s or "looser" in s or "tighter" in s:
            return False
        # bare numeric / true / false / quoted string
        return bool(re.fullmatch(r"-?\d+(\.\d+)?|true|false|'[^']*'|\"[^\"]*\"", s))
    return False


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
  5. Whether the gate's edge is STILL THERE on recent data. Markets are
     non-stationary: an edge measured over a long reference window can already
     be decaying. The `gate_recency_split` field recomputes the gate's
     blocked-bucket forward return on a recent (<=30d) vs reference (30..60d)
     window. If `edge_decayed` is true (recent edge much weaker than reference,
     or flipped sign), the validation_report's headline average is STALE —
     treat it as over-optimistic, cap confidence at "medium" at most, and
     prefer "needs_review" over "approve". Weight recent evidence over the
     long-window average. If recent `n` is small, say the edge is unproven now.

Be honest. If the hypothesis looks weak (small days_red, no shadow yet, expected
range overlapping with noise, or a decayed recent edge), say so. If the change is
reasonable but the operator should monitor a specific metric for the first 24h,
say which one.

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
        "gate_recency_split": _gate_recency_split(hyp),
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
    # The hyp-file status doesn't track defer/rollback (those live in
    # decisions.jsonl). Without this, a deferred hold is not sticky — it
    # could be silently re-approved, re-polluting attribution.
    hid = hyp.get("hypothesis_id")
    for rec in PL.iter_jsonl(PL.DECISIONS_LOG):
        if rec.get("hypothesis_id") == hid and rec.get("stage") in ("deferred", "rolled_back"):
            issues.append(
                f"hypothesis was {rec.get('stage')} in decisions.jsonl "
                f"({rec.get('decision_id')}): {rec.get('reason') or 'no reason'} "
                f"— use --force only if intentionally re-opening")
    if hyp.get("config_key") in set(PL.load_do_not_touch().get("config_keys_locked", [])):
        issues.append(f"config_key '{hyp['config_key']}' is in do_not_touch.config_keys_locked")
    if not hyp.get("validation_report"):
        issues.append("no validation_report — run L3 first")
    vr = (hyp.get("validation_report") or {}).get("result", {})
    if vr.get("verdict") == "reject":
        issues.append(f"L3 verdict=reject: {vr.get('reason')}")
    # L2-mapping guard: refuse to approve a hypothesis whose config_key
    # cannot be resolved to a real config.py constant — otherwise the
    # operator gets an un-appliable instruction (the
    # GATE_LATE_IMPULSE_ROTATION_THRESHOLD class of bug).
    rk = resolve_config_keys(hyp)
    if not rk["ok"]:
        issues.append(f"unmappable config_key: {rk['reason']}")
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
    d = hyp.get("diff", {})
    rk = resolve_config_keys(hyp)
    consts = _config_constants()
    print("")
    print("=" * 70)
    print("APPLY MANUALLY")
    print("=" * 70)

    if not rk["ok"]:
        print("⚠️  CANNOT EMIT A SAFE APPLY INSTRUCTION")
        print(f"   {rk['reason']}")
        print("   This hypothesis must NOT be pasted into config.py as-is.")
        print("   Fix the L2 mapping (RULE_CONFIG_MAP) or reject the hypothesis.")
        return

    keys = rk["keys"]
    to_val = d.get("to")
    concrete = _is_concrete_value(to_val)
    print("1. Edit files/config.py — real constant(s) for this rule:")
    for k in keys:
        cur = consts.get(k, "<unknown>")
        if concrete and len(keys) == 1:
            print(f"     {k} = {to_val}        # was {cur}")
        else:
            # Directive diff ('+10% looser') or multi-constant gate:
            # show current values + the directive, do NOT fabricate a
            # paste-ready line the operator might apply wrongly.
            print(f"     {k}    # current={cur}   apply: {to_val!r} (directive — compute per-constant)")
    if not concrete:
        print("   NOTE: diff is a DIRECTIVE, not a literal. Compute each "
              "constant's new value explicitly and record it.")
    print("")
    print(f"2. Restart bot:  restart_bot.bat")
    print("")
    print(f"3. Suggested commit message:")
    print(f"     pipeline: apply {hyp['hypothesis_id']} ({','.join(keys)} := {to_val})")
    print("")
    print(f"4. Schedule rollback check in 7 days:")
    print(f"     pyembed\\python.exe files\\pipeline_approve.py --rollback-check {hyp['hypothesis_id']}")


def cmd_defer(hyp_path: Path, reason: str) -> None:
    """Hold an approved hypothesis WITHOUT applying it. Records a
    `deferred` decision so memory is honest and L7 attribution skips it
    (it would otherwise measure a change that was never applied)."""
    hyp = PL.read_json(hyp_path)
    if not hyp:
        print(f"ERROR: cannot read {hyp_path}")
        return
    hid = hyp.get("hypothesis_id")
    prior = None
    for rec in PL.iter_jsonl(PL.DECISIONS_LOG):
        if rec.get("hypothesis_id") == hid and rec.get("stage") == "approved":
            prior = rec
    decision_id = f"d-{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    record_decision(decision_id, hyp, "deferred", extra={
        "reason": reason,
        "defers": prior.get("decision_id") if prior else None,
        "applied": False,
    })
    print(f"Deferred: {hid}")
    print(f"  reason: {reason}")
    if prior:
        print(f"  supersedes approved decision {prior.get('decision_id')} "
              f"(was NOT applied — attribution will skip it)")
    print(f"  decision_id={decision_id}")


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

    # RM-4: auto-apply via runtime override + auto-restart.
    # Concrete diff (literal numeric/bool) -> the override layer in config.py
    # picks it up at next import; trigger a bot restart so the running
    # process actually sees the new value. Directive diffs ("+10% looser")
    # cannot be auto-applied; fall back to printing manual instructions.
    d = hyp.get("diff", {}) or {}
    to_val = d.get("to")
    auto_ok = isinstance(to_val, (int, float, bool)) and not isinstance(to_val, str)
    if auto_ok and not _maybe_auto_restart(hyp, to_val):
        emit_apply_instructions(hyp)
    elif not auto_ok:
        emit_apply_instructions(hyp)


def _maybe_auto_restart(hyp: dict, to_val) -> bool:
    """Auto-apply path: trigger restart_bot.bat so the override takes effect.
    Returns True if auto-apply was attempted (success or printed instructions).
    Stays opt-out-safe via PIPELINE_AUTO_APPLY env var."""
    import os, subprocess, sys
    if os.environ.get("PIPELINE_AUTO_APPLY", "1") == "0":
        print("[auto-apply] disabled via PIPELINE_AUTO_APPLY=0 — manual steps follow")
        return False
    rk = resolve_config_keys(hyp)
    keys = rk.get("keys") or []
    print("\n" + "=" * 70)
    print("AUTO-APPLY (RM-4 runtime override + restart)")
    print("=" * 70)
    print(f"  override: {', '.join(keys)} -> {to_val}")
    print(f"  source  : .runtime/pipeline/decisions/decisions.jsonl")
    print(f"  audit   : .runtime/config_overrides_applied.json")
    bat = PL.REPO_ROOT / "restart_bot.bat"
    if not bat.exists():
        print(f"[auto-apply] restart_bot.bat missing at {bat} — manual restart needed")
        return False
    print(f"  restart : spawning {bat.name} (detached)")
    try:
        # Detach so this approve command returns immediately
        flags = 0x00000008  # DETACHED_PROCESS on Windows
        subprocess.Popen(
            ["cmd", "/c", "start", "", "/min", str(bat), "--run"],
            cwd=str(PL.REPO_ROOT),
            creationflags=flags if sys.platform == "win32" else 0,
            close_fds=True,
        )
        print("[auto-apply] restart launched — bot will load the new override")
    except OSError as e:
        print(f"[auto-apply] restart spawn failed: {e} — run restart_bot.bat manually")
        return False
    print("=" * 70)
    return True


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
    g.add_argument("--defer", help="hypothesis_id to HOLD (approved but not applied)")
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

    if args.defer:
        dp = PL.HYPOTHESES / f"{args.defer}.json"
        cmd_defer(dp, args.reason or "held: awaiting L4 shadow before apply")
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
