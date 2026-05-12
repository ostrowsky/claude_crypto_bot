"""Aggregate structured blocked-candidate events.

Reads bot_events.jsonl, filters event=blocked, groups by reason_code +
gate, and produces:
  1. Top-15 reason_codes by block volume.
  2. Per-gate breakdown: distribution of features at block time.
  3. Per-symbol top blockers (helps diagnose «why no signal on X»).
  4. Would-be-signal estimate: at relaxed threshold, how many blocks would clear?

Spec: docs/specs/features/structured-blocked-logging-spec.md
"""
from __future__ import annotations
import argparse, io, json, sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent


def load_blocked_events(window_hours: float, sym_filter: str | None = None):
    """Stream bot_events.jsonl, return list[dict] of blocked events in window."""
    cut = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    events = []
    with io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8") as f:
        for ln in f:
            if '"blocked"' not in ln:
                continue
            try:
                e = json.loads(ln)
            except Exception:
                continue
            if e.get("event") != "blocked":
                continue
            try:
                dt = datetime.fromisoformat((e.get("ts") or "").replace("Z", "+00:00"))
            except Exception:
                continue
            if dt < cut:
                continue
            if sym_filter and (e.get("sym") or e.get("symbol")) != sym_filter:
                continue
            e["_dt"] = dt
            events.append(e)
    return events


def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs); n = len(xs)
    return xs[min(n - 1, int(n * p))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--sym", type=str, default=None,
                    help="filter to a single symbol (diagnostic)")
    ap.add_argument("--top-n", type=int, default=15)
    args = ap.parse_args()

    events = load_blocked_events(args.hours, sym_filter=args.sym)
    label = f" sym={args.sym}" if args.sym else ""
    print(f"=== Blocked-event breakdown · last {args.hours:.0f}h{label} ===")
    print(f"Total blocks: {len(events)}\n")
    if not events:
        return

    # 1) Top reason_codes
    by_code = Counter()
    by_signal_type = Counter()
    for e in events:
        rc = e.get("reason_code") or "(unstructured)"
        by_code[rc] += 1
        by_signal_type[e.get("signal_type", "?")] += 1

    print(f"Top-{args.top_n} reason_codes:")
    print(f"  {'reason_code':<26} {'n':>5}  {'share':>6}")
    for rc, n in by_code.most_common(args.top_n):
        print(f"  {rc:<26} {n:>5}  {100*n/len(events):>5.1f}%")

    print(f"\nLegacy signal_type (older events without reason_code):")
    for st, n in by_signal_type.most_common(args.top_n):
        print(f"  {st:<26} {n:>5}")

    # 2) Per-gate feature distributions (only structured events)
    structured = [e for e in events if e.get("reason_code")]
    print(f"\n=== Per-gate feature distributions (n_structured={len(structured)}) ===")
    if structured:
        per_gate = defaultdict(list)
        for e in structured:
            per_gate[e["reason_code"]].append(e)
        for code, lst in sorted(per_gate.items(), key=lambda x: -len(x[1])):
            if len(lst) < 3:
                continue
            print(f"\n── {code}  (n={len(lst)}) ──")
            for feat in ("ml_proba", "ranker_top_gainer_prob", "ranker_ev",
                         "candidate_score", "slope_pct", "adx", "vol_x",
                         "rsi", "daily_range"):
                vals = [e.get(feat) for e in lst if e.get(feat) is not None]
                if not vals:
                    continue
                p25 = percentile(vals, 0.25); p50 = percentile(vals, 0.5)
                p75 = percentile(vals, 0.75)
                mn = min(vals); mx = max(vals)
                print(f"  {feat:<24} p25={p25:>7.3f}  p50={p50:>7.3f}  "
                      f"p75={p75:>7.3f}  min={mn:>7.3f}  max={mx:>7.3f}  n={len(vals)}")

    # 3) Per-symbol top blockers
    if not args.sym:
        print(f"\n=== Per-symbol top blocker (top-15 most-blocked syms) ===")
        per_sym = defaultdict(Counter)
        sym_total = Counter()
        for e in events:
            sym = e.get("sym") or e.get("symbol", "?")
            rc = e.get("reason_code") or e.get("signal_type", "?")
            per_sym[sym][rc] += 1
            sym_total[sym] += 1
        for sym, total in sym_total.most_common(15):
            top_rc, top_n = per_sym[sym].most_common(1)[0]
            print(f"  {sym:<14} blocks={total:>4}  main_blocker={top_rc:<22} "
                  f"({top_n}, {100*top_n/total:.0f}%)")

    # 4) Would-be-signal estimates per gate (heuristic)
    print(f"\n=== «Would-be-signal» estimates (heuristic threshold relaxation) ===")
    # ml_zone: would clear if ml_proba >= floor (we extract floor from text)
    ml_zone_blocks = [e for e in events if e.get("reason_code") == "ml_zone"]
    if ml_zone_blocks:
        probas = [e.get("ml_proba") for e in ml_zone_blocks if e.get("ml_proba") is not None]
        if probas:
            for thr in (0.05, 0.10, 0.15, 0.20):
                n_pass = sum(1 for p in probas if p >= thr)
                print(f"  ml_zone: at threshold {thr:.2f}, would pass: "
                      f"{n_pass}/{len(probas)} = {100*n_pass/len(probas):.0f}%")

    # trend_chop: slope_pct distribution
    chop_blocks = [e for e in events if e.get("reason_code") == "trend_chop"]
    if chop_blocks:
        slopes = [e.get("slope_pct") for e in chop_blocks if e.get("slope_pct") is not None]
        if slopes:
            for thr in (0.5, 0.7, 0.9, 1.0):
                n_pass = sum(1 for s in slopes if s >= thr)
                print(f"  trend_chop: at slope_min {thr:.2f}, would pass: "
                      f"{n_pass}/{len(slopes)} = {100*n_pass/len(slopes):.0f}%")

    # entry_score: distance from floor
    es_blocks = [e for e in events if e.get("reason_code") == "entry_score"]
    if es_blocks:
        deltas = [(e.get("score_floor", 0) - e.get("candidate_score", 0))
                  for e in es_blocks
                  if e.get("score_floor") is not None and e.get("candidate_score") is not None]
        if deltas:
            for delta_cap in (1, 3, 5, 10):
                n = sum(1 for d in deltas if d <= delta_cap)
                print(f"  entry_score: within {delta_cap} of floor: "
                      f"{n}/{len(deltas)} = {100*n/len(deltas):.0f}%")


if __name__ == "__main__":
    main()
