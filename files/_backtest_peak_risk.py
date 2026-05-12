"""Aggregate peak_risk_shadow events.

Reports:
  1. Total events with score-bucket distribution.
  2. Per-position max-score reached.
  3. Per-mode breakdown.
  4. Component analysis: which signal (RSI / edge / MACD / profit) dominates?
  5. Acceptance check: ≥20 events in 7d before Phase 2 promotion.

Spec: docs/specs/features/peak-risk-shadow-spec.md
"""
from __future__ import annotations
import argparse, io, json, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7)
    args = ap.parse_args()
    cut = datetime.now(timezone.utc) - timedelta(days=args.days)

    events = []
    with io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8") as f:
        for ln in f:
            if "peak_risk_shadow" not in ln:
                continue
            try:
                e = json.loads(ln)
            except Exception:
                continue
            if e.get("event") != "peak_risk_shadow":
                continue
            try:
                dt = datetime.fromisoformat((e.get("ts") or "").replace("Z", "+00:00"))
            except Exception:
                continue
            if dt < cut:
                continue
            e["_dt"] = dt
            events.append(e)

    print(f"=== peak_risk_shadow events · last {args.days:.1f}d ===")
    print(f"Total: {len(events)}\n")
    if not events:
        print("(no events — restart bot on v2.20+ and wait for profitable open positions)")
        return

    # Bucket distribution
    buckets = Counter()
    for e in events:
        s = float(e.get("score", 0))
        b = int(s // 20) * 20
        buckets[b] += 1
    print("Score bucket distribution:")
    for b in sorted(buckets.keys()):
        bar = "#" * min(60, buckets[b])
        print(f"  {b:>3}-{b+19:>3}  {buckets[b]:>4}  {bar}")

    # Per-mode
    by_mode = Counter(e.get("mode", "?") for e in events)
    print(f"\nBy mode:")
    for m, n in by_mode.most_common():
        print(f"  {m:<14} {n:>4}")

    # Component analysis
    print(f"\nComponent dominance (max component per event):")
    component_dom = Counter()
    for e in events:
        comps = [
            ("rsi", float(e.get("rsi_score") or 0)),
            ("edge", float(e.get("edge_score") or 0)),
            ("macd", float(e.get("macd_score") or 0)),
            ("profit", float(e.get("profit_score") or 0)),
        ]
        comps.sort(key=lambda x: -x[1])
        if comps[0][1] > 0:
            component_dom[comps[0][0]] += 1
    for c, n in component_dom.most_common():
        print(f"  {c:<10} {n:>4}  ({100*n/len(events):.0f}%)")

    # Highest-score events
    events_sorted = sorted(events, key=lambda x: -float(x.get("score", 0)))
    print(f"\nTop-10 highest-score events:")
    print(f"  {'date':<11} {'sym':<10} {'mode':<14} {'score':>5} {'pnl':>7} {'rsi':>5} {'edge%':>6}")
    for e in events_sorted[:10]:
        d = e["_dt"].strftime("%m-%d %H:%M")
        print(f"  {d:<11} {(e.get('sym') or '?'):<10} "
              f"{(e.get('mode') or '?'):<14} {e.get('score', 0):>5.0f} "
              f"{e.get('pnl_pct', 0):>+6.2f}% {e.get('rsi') or 0:>5.0f} "
              f"{e.get('price_edge_pct') or 0:>+5.1f}%")

    # Acceptance
    print()
    threshold = 20
    if len(events) >= threshold:
        print(f"✅ Acceptance MET ({len(events)} ≥ {threshold} events / {args.days:.0f}d)")
        print("   Ready to design Phase 2 (TG alerts + tighter trail).")
    else:
        print(f"⏳ Acceptance pending ({len(events)} / {threshold} events / {args.days:.0f}d)")


if __name__ == "__main__":
    main()
