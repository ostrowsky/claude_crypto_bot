"""Daily breakdown of blocked candidates by reason_code.

Analyzes bot_events.jsonl blocked events to identify over-blocking gates
and opportunity distribution per gate.

Usage:
    pyembed\python.exe files\_backtest_blocked_breakdown.py
    pyembed\python.exe files\_backtest_blocked_breakdown.py --would-be-signal

Output:
    - Top-20 reason_codes by frequency
    - Distribution histograms per gate
    - Optional would-be-signal analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

LOG_FILE = HERE / "bot_events.jsonl"
REASON_CODE_TAXONOMY = {
    "ml_zone": "ML proba outside profitable zone",
    "ranker_hard_veto": "Ranker final_score below veto threshold",
    "ranker_soft_veto": "Ranker EV below soft veto",
    "trend_chop": "Trend/1h chop filter (slope/adx/vol)",
    "trend_quality": "15m trend quality guard (RSI/edge/range)",
    "entry_score": "Score below floor",
    "impulse_guard": "Impulse_speed sub-conditions",
    "mode_range_quality": "Daily_range outside mode-specific bounds",
    "clone_signal_guard": "Similar setup limit",
    "open_cluster_cap": "Cluster open positions cap",
    "correlation_guard": "High correlation with existing pos",
    "mtf": "Multi-timeframe disagreement",
    "late_continuation": "Continuation entry too late",
    "late_impulse_rotation": "Late rotation candidate",
    "cooldown": "Symbol-level cooldown",
    "portfolio": "Portfolio full",
    "time_block": "Time-of-day block",
    "strategy_cap": "Strategy-level cap",
    "enhanced_block": "Phase-2 enhanced filter",
    "ml_filter": "Generic ML filter (legacy)",
    "near_miss": "Near-miss edge case",
}


def load_blocked_events() -> list[dict]:
    """Load all blocked events from bot_events.jsonl."""
    if not LOG_FILE.exists():
        print(f"[!] {LOG_FILE} not found")
        return []

    blocked = []
    try:
        with LOG_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    if event.get("event") == "blocked":
                        blocked.append(event)
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"[!] Error reading {LOG_FILE}: {e}")

    return blocked


def aggregate_by_code(blocked: list[dict]) -> Dict[str, list[dict]]:
    """Group blocked events by reason_code."""
    by_code = defaultdict(list)
    for event in blocked:
        code = event.get("reason_code", "unknown")
        by_code[code].append(event)
    return by_code


def top_gates_summary(by_code: Dict[str, list[dict]], limit: int = 20) -> None:
    """Print top-N reason_codes by frequency."""
    print(f"\n{'='*80}")
    print(f"TOP-{limit} OVER-BLOCKING GATES (by frequency)")
    print(f"{'='*80}\n")

    summary = [
        (code, len(events), REASON_CODE_TAXONOMY.get(code, ""))
        for code, events in by_code.items()
    ]
    summary.sort(key=lambda x: x[1], reverse=True)

    for i, (code, count, desc) in enumerate(summary[:limit], 1):
        print(f"{i:2d}. {code:25s} {count:4d} blocks  —  {desc}")

    total = sum(count for _, count, _ in summary)
    print(f"\nTotal blocked events: {total}")


def distribution_histogram(events: list[dict], field: str, bins: int = 10) -> None:
    """Print histogram of a numeric field."""
    values = []
    for e in events:
        v = e.get(field)
        if v is not None:
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                pass

    if not values:
        print(f"  {field}: no data")
        return

    min_v, max_v = min(values), max(values)
    print(f"  {field}: min={min_v:.4f}, max={max_v:.4f}, mean={sum(values)/len(values):.4f}, n={len(values)}")


def per_gate_analysis(by_code: Dict[str, list[dict]], limit: int = 10) -> None:
    """Print detailed analysis per top gate."""
    print(f"\n{'='*80}")
    print(f"DISTRIBUTION ANALYSIS — TOP-{limit} GATES")
    print(f"{'='*80}\n")

    summary = sorted(
        ((code, events) for code, events in by_code.items()),
        key=lambda x: len(x[1]),
        reverse=True,
    )

    for code, events in summary[:limit]:
        print(f"\n{code.upper()} ({len(events)} events)")
        print(f"  Symbols: {len(set(e.get('sym') for e in events))} unique")
        distribution_histogram(events, "ml_proba", 10)
        distribution_histogram(events, "ranker_final_score", 10)
        distribution_histogram(events, "candidate_score", 10)
        distribution_histogram(events, "slope_pct", 10)
        distribution_histogram(events, "adx", 10)
        distribution_histogram(events, "vol_x", 10)
        distribution_histogram(events, "rsi", 10)


def would_be_signal_analysis(by_code: Dict[str, list[dict]]) -> None:
    """Estimate effect of relaxing each gate."""
    print(f"\n{'='*80}")
    print(f"WOULD-BE-SIGNAL ANALYSIS — relaxation potential")
    print(f"{'='*80}\n")

    for code, events in sorted(
        by_code.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )[:8]:
        high_potential = 0
        for e in events:
            score = e.get("candidate_score")
            floor = e.get("score_floor")
            ml = e.get("ml_proba")

            if score is not None and floor is not None:
                rel_to_floor = (score - floor) / max(abs(floor), 1.0)
                if rel_to_floor > -0.20:
                    high_potential += 1
            elif ml is not None and ml > 0.40:
                high_potential += 1

        pct = 100.0 * high_potential / len(events) if events else 0
        print(f"{code:25s}: {high_potential:3d}/{len(events)} ({pct:5.1f}%) would-be-signals (score ≥-20% floor OR ml>0.40)")


def main():
    ap = argparse.ArgumentParser(
        description="Analyze blocked candidates by reason_code and gate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--would-be-signal",
        action="store_true",
        help="Include would-be-signal analysis (relaxation potential per gate)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit top gates to show (default: 20)",
    )
    ap.add_argument(
        "--since-hours",
        type=int,
        default=72,
        help="Analyze last N hours of events (default: 72)",
    )
    args = ap.parse_args()

    print(f"\n[*] Loading blocked events from {LOG_FILE}...")
    blocked = load_blocked_events()

    if not blocked:
        print("[!] No blocked events found")
        return 1

    if args.since_hours > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=args.since_hours)
        blocked = [
            e for e in blocked
            if datetime.fromisoformat(e.get("ts", "").replace("Z", "+00:00")) > cutoff
        ]
        print(f"[*] Filtered to last {args.since_hours}h: {len(blocked)} events")

    by_code = aggregate_by_code(blocked)
    print(f"[*] Found {len(by_code)} unique reason_codes")

    top_gates_summary(by_code, args.limit)
    per_gate_analysis(by_code, min(args.limit, 10))

    if args.would_be_signal:
        would_be_signal_analysis(by_code)

    print(f"\n{'='*80}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
