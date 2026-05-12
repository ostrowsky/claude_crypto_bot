"""Aggregate surge_shadow_win events.

Reads bot_events.jsonl, filters event=surge_shadow_win, summarises:
  1. Total count + daily breakdown.
  2. Distribution of selected_mode (what won instead of trend_surge).
  3. Per-symbol top instances.
  4. Acceptance check: ≥5 events in 7d → ready to flip
     TREND_SURGE_PRECEDENCE_ENABLED.

Spec: docs/specs/features/trend-surge-precedence-spec.md §7
"""
from __future__ import annotations
import argparse, io, json, sys
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone, timedelta

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
            if "surge_shadow_win" not in ln:
                continue
            try:
                e = json.loads(ln)
            except Exception:
                continue
            if e.get("event") != "surge_shadow_win":
                continue
            try:
                dt = datetime.fromisoformat((e.get("ts") or "").replace("Z", "+00:00"))
            except Exception:
                continue
            if dt < cut:
                continue
            e["_dt"] = dt
            events.append(e)

    print(f"=== surge_shadow_win events · last {args.days:.1f}d ===")
    print(f"Total: {len(events)}\n")
    if not events:
        print("(no events — restart bot on v2.18+ and wait for surge+entry overlap)")
        return

    # By day
    by_day = Counter()
    for e in events:
        by_day[e["_dt"].strftime("%Y-%m-%d")] += 1
    print("By day:")
    for d, n in sorted(by_day.items()):
        print(f"  {d}  {n:>4}")

    # By selected mode (what won instead of surge)
    by_mode = Counter(e.get("selected_mode", "?") for e in events)
    print(f"\nSelected mode (would have been replaced by trend_surge):")
    for m, n in by_mode.most_common():
        print(f"  {m:<14} {n:>4}  ({100*n/len(events):.0f}%)")

    # By symbol
    by_sym = Counter((e.get("sym") or "?") for e in events)
    print(f"\nTop symbols:")
    for s, n in by_sym.most_common(10):
        print(f"  {s:<14} {n:>4}")

    # By tf
    by_tf = Counter(e.get("tf", "?") for e in events)
    print(f"\nBy tf:")
    for t, n in by_tf.most_common():
        print(f"  {t:<6} {n:>4}")

    # Acceptance
    print()
    threshold = 5
    if len(events) >= threshold:
        print(f"✅ Acceptance MET ({len(events)} ≥ {threshold} events / {args.days:.0f}d)")
        print("   Ready to flip TREND_SURGE_PRECEDENCE_ENABLED → True after review.")
    else:
        print(f"⏳ Acceptance pending ({len(events)} / {threshold} events / {args.days:.0f}d)")
        print(f"   Need {threshold - len(events)} more before flipping flag.")


if __name__ == "__main__":
    main()
