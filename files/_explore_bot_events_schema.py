"""One-shot helper: inspect bot_events.jsonl schema without reading it directly.

Streams last 5000 events, collects field frequency stats + samples per event type.
Output: schema summary on stdout.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVENTS = REPO / "files" / "bot_events.jsonl"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    if not EVENTS.exists():
        print(f"NOT FOUND: {EVENTS}")
        return
    field_counts = Counter()
    event_types = Counter()
    decision_actions = Counter()
    reason_codes = Counter()
    sample_entry = None
    sample_blocked = None
    entry_score_blocked_samples = []
    n = 0
    with EVENTS.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            for k in ev.keys():
                field_counts[k] += 1
            event_types[ev.get("event") or ev.get("type") or "?"] += 1
            dec = ev.get("decision") or {}
            decision_actions[dec.get("action") or "?"] += 1
            rc = dec.get("reason_code") or ev.get("reason_code") or "?"
            reason_codes[rc] += 1
            if dec.get("action") == "take" and sample_entry is None:
                sample_entry = ev
            if dec.get("action") == "blocked" and sample_blocked is None:
                sample_blocked = ev
            if rc == "entry_score" and len(entry_score_blocked_samples) < 5:
                entry_score_blocked_samples.append(ev)
    print(f"total_events: {n}")
    print()
    print("field_counts (top 30):")
    for k, c in field_counts.most_common(30):
        print(f"  {k}: {c}")
    print()
    print(f"event_types: {dict(event_types.most_common(20))}")
    print(f"decision.action: {dict(decision_actions.most_common(20))}")
    print(f"reason_codes (top 20): {dict(reason_codes.most_common(20))}")
    print()
    if sample_entry:
        print("--- SAMPLE: take ---")
        print(json.dumps(sample_entry, indent=2, ensure_ascii=False)[:2500])
        print()
    if entry_score_blocked_samples:
        print("--- SAMPLE: blocked by entry_score (first) ---")
        print(json.dumps(entry_score_blocked_samples[0], indent=2, ensure_ascii=False)[:2500])
        print()
        print(f"total entry_score blocked samples collected: {len(entry_score_blocked_samples)}")


if __name__ == "__main__":
    main()
