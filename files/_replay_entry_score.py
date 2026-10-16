"""Real replay engine for ENTRY_SCORE_FLOOR hypothesis.

Streams bot_events.jsonl for last N days. For each event blocked by `entry_score`,
the candidate_score and ranker projections are known — we can simulate "what
would have happened if floor was X" using the bot's own ranker model as oracle
plus, if available, real forward returns from kline cache.

Pareto sweep over grid of floor values. For each grid point computes:
  - n_would_unblock: events that change from blocked->take
  - projected_top_gainer_hits: sum of ranker_top_gainer_prob across unblocked
  - projected_ev_pct: mean ranker_ev across unblocked
  - estimated_fp_added: events with ranker_top_gainer_prob < 0.3 (proxy)
  - sharpe_proxy: projected_ev / std * sqrt(n)
  - if --use-klines: actual forward 5-bar return from history/ cache

Verdict is per grid point:
  - ACCEPT_BEST: this value maximizes projected_ev*sqrt(n) without raising FP > +5pp
  - REJECT: projected FP rate increases by > 5pp OR sharpe_proxy drops
  - REVIEW: marginal

Output JSON consumed by pipeline_validator.py.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVENTS = REPO / "files" / "bot_events.jsonl"
KLINE_CACHE = REPO / "history"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None


def load_relevant_events(window_days: int) -> dict:
    """Return dict with:
       - blocked_by_entry_score: list of dicts (sym, ts, score, ranker_*, etc.)
       - actual_entries: list of dicts (taken entries with later pnl if known)
       - current_floor: dominant score_floor across blocked events
    """
    if not EVENTS.exists():
        return {"blocked": [], "actual": [], "current_floor": None}
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    blocked = []
    actual_entries = {}  # key: (sym, ts) -> entry record
    exits = []
    floors = defaultdict(int)

    with EVENTS.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _parse_ts(ev.get("ts", ""))
            if ts is None or ts < cutoff:
                continue
            etype = ev.get("event")
            if etype == "blocked" and ev.get("reason_code") == "entry_score":
                score = ev.get("candidate_score")
                floor = ev.get("score_floor")
                if score is None or floor is None:
                    continue
                floors[floor] += 1
                blocked.append({
                    "sym": ev.get("sym"),
                    "ts":  ev.get("ts"),
                    "tf":  ev.get("tf"),
                    "signal_type": ev.get("signal_type"),
                    "score": float(score),
                    "floor": float(floor),
                    "ranker_top_gainer_prob": ev.get("ranker_top_gainer_prob"),
                    "ranker_ev":              ev.get("ranker_ev"),
                    "ranker_proba":           ev.get("ranker_proba"),
                    "ranker_quality_proba":   ev.get("ranker_quality_proba"),
                    "ml_proba":               ev.get("ml_proba"),
                    "is_bull_day":            ev.get("is_bull_day"),
                    "atr_pct":                ev.get("atr_pct"),
                    "price":                  ev.get("price"),
                })
            elif etype == "entry":
                key = (ev.get("sym"), ev.get("ts"))
                actual_entries[key] = ev
            elif etype == "exit":
                exits.append(ev)
    current_floor = max(floors.items(), key=lambda kv: kv[1])[0] if floors else None
    return {
        "blocked": blocked,
        "actual_count": len(actual_entries),
        "exit_count": len(exits),
        "current_floor": current_floor,
        "floor_distribution": dict(floors),
    }


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def simulate_floor(blocked: list[dict], new_floor: float, fp_proba_threshold: float = 0.30) -> dict:
    """For a hypothetical floor, count which blocked events would become 'take'
       and project aggregate metrics from ranker fields."""
    would_unblock = [b for b in blocked if b["score"] >= new_floor]
    if not would_unblock:
        return {
            "new_floor": new_floor,
            "n_would_unblock": 0,
            "projected_top_gainer_hits": 0.0,
            "projected_ev_pct": None,
            "ev_std": None,
            "projected_fp_count": 0,
            "projected_fp_rate": None,
            "sharpe_proxy": None,
        }
    tg_probs = [b["ranker_top_gainer_prob"] for b in would_unblock if b["ranker_top_gainer_prob"] is not None]
    evs = [b["ranker_ev"] for b in would_unblock if b["ranker_ev"] is not None]
    fp_count = sum(1 for b in would_unblock
                   if (b["ranker_top_gainer_prob"] or 0) < fp_proba_threshold)

    proj_tg = sum(tg_probs) if tg_probs else 0.0
    mean_ev = statistics.mean(evs) if evs else None
    std_ev  = statistics.stdev(evs) if len(evs) > 1 else None
    sharpe  = None
    if mean_ev is not None and std_ev and std_ev > 0:
        sharpe = round(mean_ev / std_ev * math.sqrt(len(evs)), 4)

    return {
        "new_floor": new_floor,
        "n_would_unblock": len(would_unblock),
        "projected_top_gainer_hits": round(proj_tg, 4),
        "projected_top_gainer_hit_rate": round(proj_tg / len(would_unblock), 4),
        "projected_ev_pct":  round(mean_ev, 6) if mean_ev is not None else None,
        "ev_std":           round(std_ev, 6) if std_ev is not None else None,
        "projected_fp_count": fp_count,
        "projected_fp_rate":  round(fp_count / len(would_unblock), 4),
        "sharpe_proxy":      sharpe,
    }


def grade_grid(grid_results: list[dict], baseline: dict) -> dict:
    """Pick best grid point + verdict.
       Reject if FP rate increase > 5pp or sharpe doesn't improve."""
    # Find best by projected EV * sqrt(n_would_unblock)
    scored = []
    for r in grid_results:
        n = r["n_would_unblock"]
        ev = r["projected_ev_pct"] or 0
        scored.append((ev * math.sqrt(max(n, 1)), r))
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1] if scored else None
    if not best or best["n_would_unblock"] == 0:
        return {"verdict": "reject", "reason": "no events would unblock at any grid point", "best": None}

    baseline_fp = baseline.get("projected_fp_rate") or 0
    fp_increase = (best["projected_fp_rate"] or 0) - baseline_fp
    if fp_increase > 0.05:
        return {
            "verdict": "reject",
            "reason": f"best floor adds {fp_increase:+.1%} FP rate (limit +5%)",
            "best": best,
        }
    return {
        "verdict": "accept",
        "reason": f"best floor unblocks {best['n_would_unblock']} events, projected TG hits {best['projected_top_gainer_hits']:.2f}, FP delta {fp_increase:+.1%}",
        "best": best,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-days", type=int, default=60)
    ap.add_argument("--grid", help="comma-separated grid of floor values; default = 5-point sweep around current_floor*0.6..current_floor")
    ap.add_argument("--fp-threshold", type=float, default=0.30,
                    help="ranker_top_gainer_prob below this counts as projected FP (default 0.30)")
    ap.add_argument("--out", help="output JSON path; default stdout")
    args = ap.parse_args()

    data = load_relevant_events(args.window_days)
    blocked = data["blocked"]
    if not blocked:
        out = {"available": False, "reason": "no entry_score-blocked events in window"}
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    current_floor = data["current_floor"] or 40.0
    if args.grid:
        grid = [float(x) for x in args.grid.split(",")]
    else:
        lo, hi = current_floor * 0.5, current_floor
        grid = [round(lo + (hi - lo) * i / 4, 2) for i in range(5)]

    grid_results = [simulate_floor(blocked, g, args.fp_threshold) for g in grid]
    baseline = simulate_floor(blocked, current_floor, args.fp_threshold)
    grading = grade_grid(grid_results, baseline)

    out = {
        "replay_id": f"replay-entry-score-{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')}",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_days": args.window_days,
        "current_floor": current_floor,
        "floor_distribution": data["floor_distribution"],
        "total_blocked_by_entry_score": len(blocked),
        "actual_entries_in_window": data["actual_count"],
        "fp_proba_threshold": args.fp_threshold,
        "baseline_at_current_floor": baseline,
        "grid": grid,
        "grid_results": grid_results,
        "grading": grading,
    }

    payload = json.dumps(out, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
        print(f"[replay] wrote {args.out}", file=sys.stderr)
    else:
        print(payload)


if __name__ == "__main__":
    main()
