"""L4 — Shadow Deploy / Diff Report.

Унифицированный reader для shadow-инструментации, существующей в bot-коде
(паттерн H5_TRAILING_ONLY_SHADOW, TREND_SURGE_PRECEDENCE_SHADOW и т.п.).

Контракт shadow-события в `bot_events.jsonl`:

    {
      "ts": "...",
      "event": "shadow",
      "feature_flag": "H5_TRAILING_ONLY",        # ID фичи под флагом SHADOW=True
      "hypothesis_id": "h-2026-05-10-..." | null,# опционально, если запущено L2 гипотезой
      "symbol": "...",
      "prod_decision": "take" | "block" | "exit" | "hold" | ...,
      "shadow_decision": "...",                   # что было бы при ENABLED=True
      "ctx": { ... }                              # любой контекст для пост-анализа
    }

Скрипт умеет:
  1. Сагрегировать shadow-события за окно (по feature_flag или по hypothesis_id).
  2. Дать diff: counts, agreement_rate, delta_metric (если в ctx есть pnl/score/etc).
  3. Вынести verdict accept/reject на основе acceptance criteria из hypothesis.

Usage:
    pyembed\\python.exe files\\pipeline_shadow.py --feature H5_TRAILING_ONLY --window-days 7
    pyembed\\python.exe files\\pipeline_shadow.py --hypothesis h-2026-05-10-foo --window-days 7

Если bot-код ещё не пишет shadow для какой-то фичи — output пустой и verdict=insufficient_data.
В этом случае задача оператора (L6 или Phase 2 pipeline) — добавить логирование
в bot-код по контракту выше.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median

import pipeline_lib as PL

BOT_EVENTS     = PL.FILES_DIR / "bot_events.jsonl"
CRITIC_DATASET = PL.FILES_DIR / "critic_dataset.jsonl"

# Default acceptance gate
DEFAULT_MIN_EVENTS = 3
DEFAULT_MIN_DELTA_PCT = 0.0  # shadow median pnl must be >= prod median pnl


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None


# Adapters: translate existing event types into the canonical shadow contract.
# Bot is not modified — we read what is already logged.
ADAPTERS: dict[str, dict] = {
    "ranker_shadow": {
        "feature_flag": "RANKER_SHADOW",
        "extract": lambda ev: {
            "prod_decision":   ev.get("bot_action") or "?",
            "shadow_decision": "take" if ev.get("ranker_take") else "blocked",
            "ctx": {
                "mode":             ev.get("mode"),
                "candidate_score":  ev.get("candidate_score"),
                "score_floor":      ev.get("score_floor"),
                "ranker_proba":     ev.get("ranker_proba"),
                "ranker_threshold": ev.get("ranker_threshold"),
                "reason":           ev.get("reason"),
            },
        },
    },
    "surge_shadow_win": {
        "feature_flag": "TREND_SURGE_PRECEDENCE",
        "extract": lambda ev: {
            "prod_decision":   ev.get("selected_mode") or "?",
            "shadow_decision": ev.get("would_be_mode") or "?",
            "ctx": {
                "rsi": ev.get("rsi"), "adx": ev.get("adx"),
                "vol_x": ev.get("vol_x"), "slope_pct": ev.get("slope_pct"),
                "ml_proba": ev.get("ml_proba"), "is_bull_day": ev.get("is_bull_day"),
            },
        },
    },
    "peak_risk_shadow": {
        "feature_flag": "PEAK_RISK_EXIT",
        "extract": lambda ev: {
            "prod_decision":   "hold",
            "shadow_decision": "exit" if (ev.get("score") or 0) >= 50 else "hold",
            "ctx": {
                "mode": ev.get("mode"),
                "score": ev.get("score"),
                "prod_pnl_pct":   ev.get("pnl_pct"),  # if held, this is current pnl
                "shadow_pnl_pct": ev.get("pnl_pct"),  # if exited here, locked in same pnl
                "rsi": ev.get("rsi"), "bars_elapsed": ev.get("bars_elapsed"),
            },
        },
    },
}


def adapt_to_canonical(ev: dict) -> dict | None:
    """Translate a bot_events.jsonl record into canonical shadow event,
       returning None if event is not a shadow."""
    etype = ev.get("event", "")
    if etype == "shadow":  # already in canonical form (future bot code)
        return ev
    adapter = ADAPTERS.get(etype)
    if not adapter:
        return None
    translated = {
        "ts":             ev.get("ts"),
        "event":          "shadow",
        "feature_flag":   adapter["feature_flag"],
        "hypothesis_id":  ev.get("hypothesis_id"),
        "symbol":         ev.get("sym"),
        "_source_event":  etype,
    }
    translated.update(adapter["extract"](ev))
    return translated


# ---------------------------------------------------------------------------
# Counterfactual simulators
# ---------------------------------------------------------------------------
#
# Unlike `ADAPTERS` above (which translate events the BOT already writes), the
# simulators below synthesise canonical shadow events from HISTORICAL data
# when bot has no live shadow logging for the proposed change.  This unlocks
# L4 for structural hypotheses ("disable mode X", "widen tolerance", etc.) that
# couldn't be tested otherwise without modifying production code.
#
# Each handler takes a hypothesis dict and a window in days and returns a list
# of canonical shadow events. The existing aggregate()/verdict() pipeline does
# the rest, so simulators stay tiny.


def _ts_of_critic_event(ev: dict) -> datetime | None:
    """critic_dataset rows have two candidate timestamps; pick the most
    informative one (bar_ts = when trade actually happened).

    Both fields may be either ISO strings ('2026-04-15T...') or millisecond
    UNIX timestamps (1774346400000). The dataset has historically used both,
    so accept both."""
    for k in ("bar_ts", "ts_signal"):
        v = ev.get(k)
        if v is None:
            continue
        if isinstance(v, (int, float)):
            try:
                return datetime.fromtimestamp(float(v) / 1000.0, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                continue
        parsed = _parse_ts(v)
        if parsed is not None:
            return parsed
    return None


def _to_iso(ts_value) -> str:
    """Coerce a critic-dataset timestamp (string or ms-number) to ISO string."""
    if isinstance(ts_value, (int, float)):
        try:
            dt = datetime.fromtimestamp(float(ts_value) / 1000.0, tz=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except (OverflowError, OSError, ValueError):
            return ""
    return str(ts_value or "")


def sim_disable_mode(hyp: dict, window_days: int) -> list[dict]:
    """Counterfactual: synthesise shadow events for `disable_mode_<X>`.

    Reads `critic_dataset.jsonl`, finds every entry where the bot took a trade
    on signal_type=X, and emits a synthetic shadow event with:
        prod_decision   = "take"    prod_pnl_pct   = labels.ret_5 (realized)
        shadow_decision = "skip"    shadow_pnl_pct = 0.0          (no trade)

    aggregate() then computes delta_median_pnl_pct = shadow - prod. If the mode
    is genuinely losing, the median delta is positive (skipping is better) and
    verdict() returns accept.
    """
    rule = hyp.get("rule", "")
    if not rule.startswith("disable_mode_"):
        return []
    mode = rule[len("disable_mode_"):]
    if not CRITIC_DATASET.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    out: list[dict] = []
    for ev in PL.iter_jsonl(CRITIC_DATASET):
        if ev.get("signal_type") != mode:
            continue
        dec = ev.get("decision") or {}
        if dec.get("action") != "take":
            continue
        ts = _ts_of_critic_event(ev)
        if ts is None or ts < cutoff:
            continue
        labels = ev.get("labels") or {}
        # Prefer trade_exit_pnl (realized close); fall back to ret_5 (bar-level
        # proxy that is always present). Either is a fair signal for whether
        # this mode is net-positive over the window.
        prod_pnl = labels.get("trade_exit_pnl")
        if prod_pnl is None:
            prod_pnl = labels.get("ret_5")
        if prod_pnl is None:
            continue
        try:
            prod_pnl = float(prod_pnl)
        except (TypeError, ValueError):
            continue
        out.append({
            "ts":              _to_iso(ev.get("bar_ts") or ev.get("ts_signal")),
            "event":           "shadow",
            "feature_flag":    f"DISABLE_MODE_{mode.upper()}",
            "hypothesis_id":   hyp.get("hypothesis_id"),
            "symbol":          ev.get("sym"),
            "_source":         "critic_dataset_counterfactual",
            "prod_decision":   "take",
            "shadow_decision": "skip",
            "ctx": {
                "mode":            mode,
                "prod_pnl_pct":    prod_pnl,
                "shadow_pnl_pct":  0.0,
                "trade_taken":     labels.get("trade_taken"),
                "trade_exit_pnl":  labels.get("trade_exit_pnl"),
                "ret_5":           labels.get("ret_5"),
                "label_5":         labels.get("label_5"),
            },
        })
    return out


def sim_widen_watchlist_match_tolerance(hyp: dict, window_days: int) -> list[dict]:
    """Counterfactual proxy for `widen_watchlist_match_tolerance` and related
    hypotheses that propose loosening a score-based block.

    The hypothesis adds tolerance around an existing score floor: events that
    were blocked because their score was just below the threshold get a
    chance. So the proxy is:

        prod_decision   = "skip"  prod_pnl_pct   = 0.0      (blocked = no PnL)
        shadow_decision = "take"  shadow_pnl_pct = labels.ret_5

    UPPER-BOUND CAVEAT: this assumes any blocked-by-score event becomes a
    take. Real tolerance is narrower (only events within N% of the floor),
    so the realised delta is OPTIMISTIC. We mark this in `_caveat` so the
    operator sees it in the shadow report. A tighter sim would need
    `score_floor`/`candidate_score` to be present in critic_dataset rows —
    not always logged on legacy events. Treat verdict=accept as necessary
    but not sufficient; pair with a tight monitoring window after apply.
    """
    rule = hyp.get("rule", "")
    if not rule.startswith("widen_watchlist_match_tolerance"):
        return []
    if not CRITIC_DATASET.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    out: list[dict] = []
    for ev in PL.iter_jsonl(CRITIC_DATASET):
        dec = ev.get("decision") or {}
        if dec.get("action") != "blocked":
            continue
        reason = (dec.get("reason_code") or dec.get("reason") or "").lower()
        if not any(t in reason for t in ("score", "watchlist", "floor")):
            continue
        ts = _ts_of_critic_event(ev)
        if ts is None or ts < cutoff:
            continue
        labels = ev.get("labels") or {}
        ret_5 = labels.get("ret_5")
        if ret_5 is None:
            continue
        try:
            ret_5 = float(ret_5)
        except (TypeError, ValueError):
            continue
        out.append({
            "ts":              _to_iso(ev.get("bar_ts") or ev.get("ts_signal")),
            "event":           "shadow",
            "feature_flag":    "WIDEN_WATCHLIST_MATCH_TOLERANCE",
            "hypothesis_id":   hyp.get("hypothesis_id"),
            "symbol":          ev.get("sym"),
            "_source":         "critic_dataset_counterfactual",
            "_caveat":         "upper-bound: assumes every score-blocked event becomes a take",
            "prod_decision":   "skip",
            "shadow_decision": "take",
            "ctx": {
                "reason_blocked": reason,
                "prod_pnl_pct":   0.0,
                "shadow_pnl_pct": ret_5,
                "ret_5":          ret_5,
                "candidate_score": dec.get("candidate_score"),
                "score_floor":     dec.get("score_floor"),
            },
        })
    return out


# Rule-prefix -> simulator. New simulators are added here as the pipeline
# expands. Match is "rule.startswith(prefix)" — first match wins.
SIM_HANDLERS = {
    "disable_mode_":                  sim_disable_mode,
    "widen_watchlist_match_tolerance": sim_widen_watchlist_match_tolerance,
}


def find_simulator(rule: str):
    for prefix, handler in SIM_HANDLERS.items():
        if rule.startswith(prefix):
            return handler
    return None


def _count_total_takes_in_window(window_days: int) -> int:
    """Total `take` decisions across all modes in the lookback window —
    used by notify to render volume changes as % of overall signal flow.
    Without this denominator the operator sees "−16/day" with no clue
    whether that's 5% or 50% of the bot's output."""
    if not CRITIC_DATASET.exists():
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    n = 0
    for ev in PL.iter_jsonl(CRITIC_DATASET):
        dec = ev.get("decision") or {}
        if dec.get("action") != "take":
            continue
        ts = _ts_of_critic_event(ev)
        if ts is None or ts < cutoff:
            continue
        n += 1
    return n


def _split_events_by_recency(events: list[dict], recent_days: int) -> list[dict]:
    """Return the subset of events whose ts falls in the last `recent_days`."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=recent_days)
    out = []
    for e in events:
        ts = _parse_ts(e.get("ts", ""))
        if ts and ts >= cutoff:
            out.append(e)
    return out


def load_shadow_events(window_days: int,
                       feature_flag: str | None = None,
                       hypothesis_id: str | None = None) -> list[dict]:
    """Stream-read bot_events.jsonl, return canonical shadow events.
       Accepts both native (event:'shadow') and legacy (ranker_shadow, etc.) types."""
    if not BOT_EVENTS.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    out: list[dict] = []
    for ev in PL.iter_jsonl(BOT_EVENTS):
        canon = adapt_to_canonical(ev)
        if canon is None:
            continue
        ts = _parse_ts(canon.get("ts", ""))
        if ts is None or ts < cutoff:
            continue
        if feature_flag and canon.get("feature_flag") != feature_flag:
            continue
        if hypothesis_id and canon.get("hypothesis_id") != hypothesis_id:
            continue
        out.append(canon)
    return out


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def aggregate(events: list[dict]) -> dict:
    """Return summary diff between prod_decision and shadow_decision."""
    if not events:
        return {"available": False, "n_events": 0}

    # Per-feature buckets if not filtered
    by_feature = defaultdict(list)
    for ev in events:
        by_feature[ev.get("feature_flag", "_unknown")].append(ev)

    summaries = {}
    for feat, evs in by_feature.items():
        decisions = [(e.get("prod_decision"), e.get("shadow_decision")) for e in evs]
        agreement = sum(1 for p, s in decisions if p == s) / len(decisions)

        prod_counts   = Counter(p for p, _ in decisions)
        shadow_counts = Counter(s for _, s in decisions)

        # If ctx has numeric pnl_pct or score, compute median diff
        prod_metric, shadow_metric, delta = None, None, None
        prod_vals = [e.get("ctx", {}).get("prod_pnl_pct") for e in evs if isinstance(e.get("ctx", {}).get("prod_pnl_pct"), (int, float))]
        shadow_vals = [e.get("ctx", {}).get("shadow_pnl_pct") for e in evs if isinstance(e.get("ctx", {}).get("shadow_pnl_pct"), (int, float))]
        if prod_vals and shadow_vals and len(prod_vals) == len(shadow_vals):
            prod_metric = round(median(prod_vals), 4)
            shadow_metric = round(median(shadow_vals), 4)
            delta = round(shadow_metric - prod_metric, 4)

        summaries[feat] = {
            "n_events":       len(evs),
            "agreement_rate": round(agreement, 4),
            "prod_decisions":   dict(prod_counts),
            "shadow_decisions": dict(shadow_counts),
            "prod_median_pnl_pct":    prod_metric,
            "shadow_median_pnl_pct":  shadow_metric,
            "delta_median_pnl_pct":   delta,
            "first_event_ts": evs[0].get("ts"),
            "last_event_ts":  evs[-1].get("ts"),
            "sample_symbols": list({e.get("symbol") for e in evs if e.get("symbol")})[:10],
        }
    return {"available": True, "n_events": len(events), "by_feature": summaries}


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def verdict(summary: dict, min_events: int, min_delta_pct: float) -> dict:
    if not summary.get("available"):
        return {"status": "insufficient_data", "reason": "no shadow events written yet — add logging in bot code per the contract in pipeline_shadow.py"}
    out = {}
    for feat, s in summary["by_feature"].items():
        if s["n_events"] < min_events:
            verdict_str = "insufficient_data"
            reason = f"n_events={s['n_events']} < min_events={min_events}"
        elif s["delta_median_pnl_pct"] is None:
            verdict_str = "review_manual"
            reason = "no pnl_pct in ctx — manual review required"
        elif s["delta_median_pnl_pct"] >= min_delta_pct:
            verdict_str = "accept"
            reason = f"delta={s['delta_median_pnl_pct']:+.4f}% >= min={min_delta_pct:+.4f}%"
        else:
            verdict_str = "reject"
            reason = f"delta={s['delta_median_pnl_pct']:+.4f}% < min={min_delta_pct:+.4f}%"
        out[feat] = {"verdict": verdict_str, "reason": reason, "n_events": s["n_events"]}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", help="feature_flag to filter on (e.g. H5_TRAILING_ONLY)")
    ap.add_argument("--hypothesis", help="hypothesis_id to filter on")
    ap.add_argument("--simulate-from-hypothesis", dest="simulate",
                    help="hypothesis_id to counterfactual-simulate (uses SIM_HANDLERS, "
                         "bypasses bot_events). Useful for `disable_mode_*` etc.")
    ap.add_argument("--window-days", type=int, default=7)
    ap.add_argument("--min-events", type=int, default=DEFAULT_MIN_EVENTS)
    ap.add_argument("--min-delta-pct", type=float, default=DEFAULT_MIN_DELTA_PCT)
    ap.add_argument("--out", help="output path for shadow report JSON")
    ap.add_argument("--print", dest="do_print", action="store_true")
    args = ap.parse_args()

    mode = "reader"
    sim_used = None
    if args.simulate:
        hp = PL.HYPOTHESES / f"{args.simulate}.json"
        hyp = PL.read_json(hp) or {}
        if not hyp:
            print(f"[L4] hypothesis not found: {args.simulate} — skip")
            return
        handler = find_simulator(hyp.get("rule", ""))
        if not handler:
            # Non-fatal: orchestrator can call sim for every pending hypothesis
            # without knowing which ones have handlers. Just log and exit clean.
            print(f"[L4] no SIM_HANDLER for rule={hyp.get('rule')!r} "
                  f"(available: {list(SIM_HANDLERS.keys())}) — skip")
            return
        events = handler(hyp, args.window_days)
        mode = "simulate"
        sim_used = handler.__name__
        # When simulating, attach to that specific hypothesis at the end
        args.hypothesis = args.simulate
    else:
        events = load_shadow_events(args.window_days, args.feature, args.hypothesis)

    summary = aggregate(events)
    vrd = verdict(summary, args.min_events, args.min_delta_pct)

    report = {
        "report_id": f"shadow-{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')}",
        "generated_at": PL.utc_now_iso(),
        "window_days": args.window_days,
        "mode": mode,
        "simulator": sim_used,
        "filter": {"feature_flag": args.feature, "hypothesis_id": args.hypothesis},
        "acceptance_criteria": {"min_events": args.min_events, "min_delta_pct": args.min_delta_pct},
        "summary": summary,
        "verdict": vrd,
    }

    # Auto recency check + volume context. Only meaningful when we just ran
    # a counterfactual simulator (`--simulate-from-hypothesis`). For pure
    # readers we skip — there's no robust way to denominate "total signal
    # volume" without the critic dataset.
    if mode == "simulate" and events:
        half_days = max(1, args.window_days // 2)
        recent_events = _split_events_by_recency(events, half_days)
        if recent_events:
            recent_summary = aggregate(recent_events)
            recent_verdict = verdict(recent_summary, args.min_events, args.min_delta_pct)
            report["recency"] = {
                "recent_days":   half_days,
                "summary":       recent_summary,
                "verdict":       recent_verdict,
            }
        report["context"] = {
            "total_takes_in_window":   _count_total_takes_in_window(args.window_days),
            "window_days":             args.window_days,
        }

    out_path = Path(args.out) if args.out else PL.SHADOW_RUNS / f"{report['report_id']}.json"
    PL.write_json(out_path, report)
    print(f"[L4] wrote {out_path}")

    # If hypothesis filter present, also attach report to hypothesis file
    if args.hypothesis:
        hp = PL.HYPOTHESES / f"{args.hypothesis}.json"
        if hp.exists():
            h = PL.read_json(hp) or {}
            h["shadow_report"] = report
            PL.write_json(hp, h)
            print(f"[L4] attached to {hp}")

    if args.do_print:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
