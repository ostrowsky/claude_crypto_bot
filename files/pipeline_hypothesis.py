"""L2 — Weekly Hypothesis Generator.

Читает последние N daily health reports, агрегирует устойчивые red_flags
(красные >= MIN_DAYS из 7), исключает уже опробованное и do_not_touch, генерирует
ранжированный список ≤3 гипотез с явным diff/expected/rollback.

V1 — rule-based, без LLM. Каждое правило знает свой red_flag.id и умеет
предложить конкретный config diff. Новые правила добавляются как функции в RULES.

Usage:
    pyembed\\python.exe files\\pipeline_hypothesis.py
    pyembed\\python.exe files\\pipeline_hypothesis.py --window-days 7 --max 3
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pipeline_lib as PL
import pipeline_claude_client as CC

EVAL_PER_MODE_DIR = PL.REPO_ROOT / "evaluation_output" / "per_mode"


# ---------------------------------------------------------------------------
# Incident evidence — supplements red flags with concrete recent events
# ---------------------------------------------------------------------------
#
# Red flags say "this metric is bad". Incidents (premature exits, missed
# sustained trends, losing trades) say "and HERE'S exactly what went wrong
# in the last 24h". The pipeline already generates hypotheses from red flags;
# incidents are used here to:
#   1. Boost a hypothesis's effective severity when supporting evidence is
#      strong (yellow → red, red → critical).
#   2. Attach incident counts to the hypothesis JSON so the Telegram review
#      block can render "📌 Подкреплено: 2 premature, 5 losers за 24ч".
#
# We do NOT generate new hypothesis types from incidents — those would need
# new L3 validators / L4 sim handlers, and the constraint we established
# earlier (universal validators or nothing) means we can't add rule types
# we can't validate.


def _load_per_mode_reports() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not EVAL_PER_MODE_DIR.exists():
        return out
    for sub in EVAL_PER_MODE_DIR.iterdir():
        if not sub.is_dir():
            continue
        rj = sub / "report.json"
        if not rj.exists():
            continue
        data = PL.read_json(rj)
        if isinstance(data, dict):
            out[sub.name] = data
    return out


def _mode_incident_counts() -> dict[str, dict]:
    """Return {mode: {premature, losers, missed}} aggregated across modes.

    Filters mirror pipeline_notify.build_incidents_block — same definitions
    so the operator never sees a discrepancy between "incident block" and
    "incident-supported hypothesis"."""
    reports = _load_per_mode_reports()
    out: dict[str, dict] = {}
    for mode, r in reports.items():
        prem = los = 0
        for v in (r.get("trade_verdicts") or []):
            bars = v.get("sell_lateness_bars")
            pnl  = v.get("captured_pnl_pct") or 0.0
            cap  = v.get("capture_ratio") or 1.0
            if bars is not None and bars < -3 and pnl > 0 and cap < 0.30:
                prem += 1
            if pnl < -1.5:
                los += 1
        miss = sum(1 for t in (r.get("missed_opportunities") or [])
                   if (t.get("gain_pct") or 0) >= 5.0)
        out[mode] = {"premature": prem, "losers": los, "missed": miss}
    return out


def _attach_incident_evidence(hypotheses: list[dict],
                              counts: dict[str, dict]) -> None:
    """Walk hypotheses, attach incident counts when the hypothesis targets
    a recognisable mode. Also bumps `severity` when supporting evidence is
    substantial (≥3 incidents in any category)."""
    sev_bump = {"yellow": "red", "red": "critical"}
    for h in hypotheses:
        mode = _hypothesis_mode(h)
        if not mode:
            continue
        c = counts.get(mode)
        if not c:
            continue
        h["incident_evidence"] = {"mode": mode, **c}
        total = c["premature"] + c["losers"] + c["missed"]
        if total >= 3 and h.get("severity") in sev_bump:
            h["severity_original"] = h["severity"]
            h["severity"] = sev_bump[h["severity"]]
            h["severity_reason"] = f"bumped by {total} incidents in mode {mode}"


def _hypothesis_mode(h: dict) -> str | None:
    """Best-effort extraction of which entry mode this hypothesis targets."""
    rule = h.get("rule", "")
    for prefix in ("disable_mode_", "tighten_proba_", "loosen_trail_k_",
                   "lower_entry_threshold_"):
        if rule.startswith(prefix):
            return rule[len(prefix):]
    # Inspect config_key as fallback (e.g. ML_PROBA_MIN_IMPULSE_SPEED)
    key = (h.get("config_key") or "").upper()
    for mode in ("IMPULSE_SPEED", "ALIGNMENT", "TREND", "IMPULSE",
                 "BREAKOUT", "RETEST", "STRONG_TREND"):
        if f"_{mode}" in key or key.endswith(mode):
            return mode.lower()
    return None

# Минимум дней (из последних N), когда red_flag должен быть красным, чтобы стать кандидатом
MIN_PERSISTENT_DAYS = 4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_recent_health(window_days: int, until: date) -> list[dict]:
    out = []
    for offset in range(window_days):
        d = until - timedelta(days=offset)
        p = PL.HEALTH / f"health-{d.isoformat()}.json"
        if p.exists():
            data = PL.read_json(p)
            if data:
                out.append(data)
    return out


def aggregate_red_flags(reports: list[dict]) -> dict[str, dict]:
    """Group red_flags by id. Track:
       - days_red: how many days the flag was red/critical
       - latest: most recent occurrence (with evidence)
       - severity_max: worst severity observed
    """
    agg: dict[str, dict] = {}
    for rpt in reports:
        for rf in rpt.get("red_flags", []):
            rid = rf["id"]
            if rid not in agg:
                agg[rid] = {
                    "id": rid,
                    "days_red": 0,
                    "severity_max": rf["severity"],
                    "latest": rf,
                    "metric": rf["metric"],
                    "first_seen": rpt["target_date"],
                    "values": [],
                }
            agg[rid]["days_red"] += 1
            agg[rid]["values"].append({"date": rpt["target_date"], "value": rf["value"]})
            sev_order = {"yellow": 1, "red": 2, "critical": 3}
            if sev_order.get(rf["severity"], 0) > sev_order.get(agg[rid]["severity_max"], 0):
                agg[rid]["severity_max"] = rf["severity"]
            # keep latest by date
            if rpt["target_date"] >= agg[rid]["latest"].get("_seen_at", "0000-00-00"):
                agg[rid]["latest"] = rf
                agg[rid]["latest"]["_seen_at"] = rpt["target_date"]
    return agg


# ---------------------------------------------------------------------------
# Rules — one per known red_flag id
# Each rule takes the aggregated flag and returns a proposed hypothesis dict, or None.
# ---------------------------------------------------------------------------

DNT = PL.load_do_not_touch()
DNT_KEYS = set(DNT.get("config_keys_locked", []))
DNT_GATES = {g["name"] for g in DNT.get("gates", [])}


def rule_entry_score_floor(agg_flag: dict, all_aggs: dict) -> dict | None:
    """If RF_early_capture persistent AND missed gainers blocked by entry_score floor → propose lower floor."""
    if agg_flag["days_red"] < MIN_PERSISTENT_DAYS:
        return None
    latest = agg_flag["latest"]
    missed = latest.get("evidence", {}).get("missed_top_gainers", [])
    blocked_by_score = [m for m in missed if (m.get("reason") or "").startswith("entry score")]
    if not blocked_by_score:
        return None
    # Find the worst-case score blocked
    scores = []
    for m in blocked_by_score:
        reason = m.get("reason", "")
        # "entry score 29.87 < floor 40.00"
        try:
            parts = reason.split()
            score = float(parts[2])
            floor = float(parts[5])
            scores.append({"symbol": m["symbol"], "score": score, "floor": floor})
        except (IndexError, ValueError):
            continue
    if not scores:
        return None
    target_floor = min(s["score"] for s in scores) - 1.0  # ниже самого низкого blocked - запас
    current_floor = scores[0]["floor"]
    if target_floor >= current_floor:
        return None
    return {
        "rule": "entry_score_floor_relax",
        "config_key": "ENTRY_SCORE_FLOOR_GLOBAL",  # may need per-mode; placeholder
        "diff": {"from": current_floor, "to": round(target_floor, 1)},
        "rationale": f"{len(scores)} top-gainers за окно блокированы entry_score floor (мин score={min(s['score'] for s in scores):.1f}, floor={current_floor})",
        "expected_delta": {"watchlist_top_early_capture_pct": "+0.10..+0.20"},
        "risk": "может поднять false_positive_rate; проверить на L3 backtest",
        "rollback": f"восстановить {current_floor}",
        "validation_required": ["backtest_60d_pareto_sweep", "shadow_7d"],
        "blocked_symbols": [s["symbol"] for s in scores],
    }


def rule_losing_mode_disable(agg_flag: dict, all_aggs: dict) -> dict | None:
    """If a mode is persistently losing → propose disabling or tightening proba."""
    if agg_flag["days_red"] < MIN_PERSISTENT_DAYS:
        return None
    latest = agg_flag["latest"]
    mode = latest.get("evidence", {}).get("mode")
    if not mode:
        return None
    fpr = latest["evidence"].get("fpr")
    return {
        "rule": f"tighten_proba_{mode}",
        "config_key": f"ML_PROBA_MIN_{mode.upper()}",
        "diff": {"from": "current", "to": "current + 0.05"},
        "rationale": f"Mode {mode} убыточен {agg_flag['days_red']}/{len(agg_flag['values'])} дней, FPR={fpr}",
        "expected_delta": {"total_realized_pnl_pct": "+2..+5pp", "false_positive_rate_per_mode": "-0.05..-0.10"},
        "risk": "может срезать early entries; следить за early_capture",
        "rollback": "вернуть proba_min",
        "validation_required": ["backtest_60d", "shadow_7d"],
    }


def rule_overblock_gate(agg_flag: dict, all_aggs: dict) -> dict | None:
    """If gate over-blocks persistently AND not in do_not_touch → propose threshold relax."""
    if agg_flag["days_red"] < MIN_PERSISTENT_DAYS:
        return None
    gate = agg_flag["id"].replace("RF_overblock_", "")
    if gate in DNT_GATES:
        return None  # protected
    latest = agg_flag["latest"]
    ev = latest.get("evidence", {})
    return {
        "rule": f"relax_gate_{gate}",
        "config_key": f"GATE_{gate.upper()}_THRESHOLD",
        "diff": {"from": "current", "to": "+10% looser"},
        "rationale": f"Gate {gate} блокирует prof events: n={ev.get('n')}, miss={ev.get('miss_pct'):.3f}, Sharpe×√n={ev.get('sharpe')}",
        "expected_delta": {"watchlist_top_bought_pct": "+0.05..+0.15"},
        "risk": "проверить что win% не падает ниже take_baseline",
        "rollback": "вернуть threshold",
        "validation_required": ["backtest_60d_pareto_sweep"],
    }


RULES: dict[str, Callable[[dict, dict], dict | None]] = {
    "RF_early_capture":  rule_entry_score_floor,
    # RF_losing_mode_* matches by prefix below
    # RF_overblock_*   matches by prefix below
}


def apply_rules(aggs: dict[str, dict]) -> list[dict]:
    hypotheses = []
    for rid, agg in aggs.items():
        rule_fn = None
        if rid in RULES:
            rule_fn = RULES[rid]
        elif rid.startswith("RF_losing_mode_"):
            rule_fn = rule_losing_mode_disable
        elif rid.startswith("RF_overblock_"):
            rule_fn = rule_overblock_gate
        if rule_fn:
            h = rule_fn(agg, aggs)
            if h:
                h["source_flag"] = rid
                h["persistence"] = {"days_red": agg["days_red"], "values": agg["values"]}
                h["severity"] = agg["severity_max"]
                h["generator"] = "rule"
                hypotheses.append(h)
    return hypotheses


# ---------------------------------------------------------------------------
# Claude augmentation (optional hybrid layer)
# ---------------------------------------------------------------------------

_CLAUDE_SYSTEM = """You are an L2 hypothesis-generator for a multi-layer pipeline that improves an AI agent.
Your role is narrow: given persistent red flags from L1 and the memory of what was
already tried + what is locked, propose ADDITIONAL hypotheses that the rule-based
generator might have missed.

Rules you MUST follow:
 1. Never propose a config_key that appears in do_not_touch.config_keys_locked.
 2. Never propose a rule+config_key already in already_tried within the last 30 days.
 3. Each hypothesis must include a concrete numerical diff (from -> to). Vague
    suggestions ("tune more aggressively") are unacceptable.
 4. Each hypothesis must include a rollback step.
 5. Prefer hypotheses that target the persistent red flags (days_red >= 4).
 6. Be conservative: a smaller, reversible change is better than a sweeping one.
 7. Output AT MOST 3 hypotheses. Fewer is fine if the rules above leave little room.
 8. Do not duplicate hypotheses already proposed by the rule layer (passed in).
"""

_CLAUDE_SCHEMA_HINT = (
    '{"hypotheses": [{"rule": str, "config_key": str, '
    '"diff": {"from": number|str, "to": number|str}, '
    '"rationale": str, "expected_delta": {metric_name: "lo..hi"}, '
    '"risk": str, "rollback": str, '
    '"validation_required": [str], "source_flag": str}]}'
)


def _claude_augment(
    aggs: dict[str, dict],
    rule_hypotheses: list[dict],
    *,
    purpose: str = "weekly_generation",
) -> list[dict]:
    """Ask Claude for additional hypotheses. Returns [] on disabled/error."""
    if not CC.is_enabled():
        return []

    persistent = [
        {
            "id":           rid,
            "metric":       agg.get("metric"),
            "days_red":     agg.get("days_red"),
            "severity":     agg.get("severity_max"),
            "latest_value": agg.get("latest", {}).get("value"),
            "latest_evidence": agg.get("latest", {}).get("evidence"),
        }
        for rid, agg in aggs.items() if agg.get("days_red", 0) >= MIN_PERSISTENT_DAYS
    ]
    tried_recent = [
        {"rule": t.get("rule"), "config_key": t.get("config_key"),
         "stage": t.get("stage"), "ts": t.get("ts")}
        for t in PL.load_already_tried()
    ][-50:]   # cap to last 50 to keep prompt small
    dnt = PL.load_do_not_touch()

    user_payload = {
        "persistent_red_flags": persistent,
        "rule_layer_proposals": [
            {"rule": h["rule"], "config_key": h["config_key"], "diff": h["diff"]}
            for h in rule_hypotheses
        ],
        "already_tried_recent": tried_recent,
        "do_not_touch": {
            "config_keys_locked": dnt.get("config_keys_locked", []),
            "gates":              [g["name"] for g in dnt.get("gates", [])],
        },
    }

    res = CC.call_claude_json(
        _CLAUDE_SYSTEM,
        json.dumps(user_payload, ensure_ascii=False, indent=2),
        schema_hint=_CLAUDE_SCHEMA_HINT,
        max_tokens=2000,
        layer="L2",
        purpose=purpose,
    )
    if not res:
        return []

    raw = res.get("hypotheses") or []
    if not isinstance(raw, list):
        return []

    out = []
    for item in raw[:3]:
        if not isinstance(item, dict):
            continue
        if not item.get("rule") or not item.get("config_key") or not item.get("diff"):
            continue
        # Attach persistence/severity from the matching source_flag if possible
        sf = item.get("source_flag")
        agg = aggs.get(sf) if sf else None
        item["persistence"] = {
            "days_red": agg["days_red"] if agg else 0,
            "values":   agg["values"]   if agg else [],
        }
        item["severity"]  = agg["severity_max"] if agg else "yellow"
        item["generator"] = "claude"
        item.setdefault("expected_delta", {})
        item.setdefault("validation_required", ["backtest_60d_pareto_sweep"])
        out.append(item)
    return out


def dedup_by_key(hypotheses: list[dict]) -> list[dict]:
    """Drop later duplicates with the same (rule, config_key). Earlier = higher prio."""
    seen = set()
    out = []
    for h in hypotheses:
        k = (h.get("rule"), h.get("config_key"))
        if k in seen:
            continue
        seen.add(k)
        out.append(h)
    return out


def filter_already_tried(hypotheses: list[dict]) -> list[dict]:
    """Skip if same rule + config_key was tried in last 30 days."""
    tried = PL.load_already_tried()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    blocked = set()
    for t in tried:
        if t.get("ts", "") < cutoff:
            continue
        blocked.add((t.get("rule"), t.get("config_key")))
    return [h for h in hypotheses if (h.get("rule"), h.get("config_key")) not in blocked]


def filter_locked_keys(hypotheses: list[dict]) -> list[dict]:
    return [h for h in hypotheses if h.get("config_key") not in DNT_KEYS]


def rank(hypotheses: list[dict]) -> list[dict]:
    """Sort by (severity, days_red, abs(expected_delta upper))."""
    sev_order = {"critical": 3, "red": 2, "yellow": 1}
    def key(h):
        return (
            sev_order.get(h.get("severity"), 0),
            h["persistence"]["days_red"],
        )
    return sorted(hypotheses, key=key, reverse=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    global MIN_PERSISTENT_DAYS  # noqa: PLW0603 — runtime override
    ap.add_argument("--window-days", type=int, default=7)
    ap.add_argument("--until", help="YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--max", type=int, default=3, help="max hypotheses to emit")
    ap.add_argument("--min-persistent-days", type=int, default=MIN_PERSISTENT_DAYS,
                    help=f"min days a flag must be red to qualify (default: {MIN_PERSISTENT_DAYS})")
    ap.add_argument("--no-claude", action="store_true",
                    help="skip Claude augmentation even if API key present (use rule layer only)")
    ap.add_argument("--print-summary", action="store_true")
    args = ap.parse_args()

    MIN_PERSISTENT_DAYS = args.min_persistent_days

    until = date.fromisoformat(args.until) if args.until else datetime.now(timezone.utc).date()
    reports = load_recent_health(args.window_days, until)
    if not reports:
        print(f"[L2] no health reports in last {args.window_days} days under {PL.HEALTH}")
        return

    aggs = aggregate_red_flags(reports)
    rule_hyps = apply_rules(aggs)

    claude_hyps: list[dict] = []
    claude_used = False
    if not args.no_claude and CC.is_enabled():
        claude_hyps = _claude_augment(aggs, rule_hyps)
        claude_used = True

    # Merge: rule first (deterministic priority), then Claude additions
    hypotheses = dedup_by_key(rule_hyps + claude_hyps)
    hypotheses = filter_already_tried(hypotheses)
    hypotheses = filter_locked_keys(hypotheses)
    # Attach incident evidence BEFORE ranking so severity bumps take effect
    incident_counts = _mode_incident_counts()
    _attach_incident_evidence(hypotheses, incident_counts)
    hypotheses = rank(hypotheses)[: args.max]

    PL.HYPOTHESES.mkdir(parents=True, exist_ok=True)
    written = []
    for i, h in enumerate(hypotheses, 1):
        hyp_id = f"h-{until.isoformat()}-{h['rule']}"
        h_record = {
            "hypothesis_id": hyp_id,
            "schema_version": 1,
            "generated_at": PL.utc_now_iso(),
            "window_days": args.window_days,
            "rank": i,
            **h,
            "status": "pending_validation",
            "validation_report": None,
            "shadow_report": None,
        }
        out_path = PL.HYPOTHESES / f"{hyp_id}.json"
        PL.write_json(out_path, h_record)
        written.append(out_path)

    summary = {
        "generated_at": PL.utc_now_iso(),
        "until": until.isoformat(),
        "window_days": args.window_days,
        "reports_seen": [r["target_date"] for r in reports],
        "persistent_flags": [{
            "id": rid,
            "days_red": agg["days_red"],
            "severity_max": agg["severity_max"],
            "metric": agg["metric"],
        } for rid, agg in aggs.items() if agg["days_red"] >= MIN_PERSISTENT_DAYS],
        "hypotheses_emitted": len(written),
        "hypothesis_files": [str(p) for p in written],
        "claude_augmentation": {
            "enabled":         claude_used,
            "n_rule":          len(rule_hyps),
            "n_claude":        len(claude_hyps),
            "n_after_dedup":   len(dedup_by_key(rule_hyps + claude_hyps)),
            "skipped_reason":  ("--no-claude" if args.no_claude
                                else (None if CC.is_enabled() else "no_api_key")),
        },
    }
    summary_path = PL.HYPOTHESES / f"_summary-{until.isoformat()}.json"
    PL.write_json(summary_path, summary)
    print(f"[L2] wrote {len(written)} hypotheses + summary at {summary_path}")

    if args.print_summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
