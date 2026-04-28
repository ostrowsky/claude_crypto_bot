"""P1: silent-miss audit.
For each (date, sym) where bot had ZERO events on a top-20 EOD winner,
dump all related context: was the symbol in the watchlist that day?
Was there ANY scout/scan event? Was it just below score floor?

Output: actionable diagnosis per silent-miss case.
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
NOW = datetime.now(timezone.utc); CUT = NOW - timedelta(days=14)

# Re-derive silent-miss list (from coverage funnel logic)
top20 = set()
with io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        try: e = json.loads(ln)
        except: continue
        ts_ms = e.get("ts");
        if not ts_ms: continue
        dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
        if dt < CUT: continue
        if e.get("label_top20") == 1:
            top20.add((dt.strftime("%Y-%m-%d"), e.get("symbol")))

# Collect ALL events per (date, sym) — including all event types
all_evs = defaultdict(list)  # (d, sym) -> [(dt, event_type, brief)]
event_type_counter = Counter()
with io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        try: e = json.loads(ln)
        except: continue
        ts = e.get("ts","")
        try: dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        except: continue
        if dt < CUT: continue
        sym = e.get("sym") or e.get("symbol") or ""
        if not sym: continue
        d = dt.strftime("%Y-%m-%d")
        ev = e.get("event","")
        event_type_counter[ev] += 1
        # brief description
        reason = ""
        if isinstance(e.get("decision"), dict):
            reason = e["decision"].get("reason_code","") or ""
        if not reason:
            reason = e.get("reason_code") or e.get("reason","") or ""
        all_evs[(d, sym)].append((dt, ev, reason[:80]))

print("=== P1: Silent-miss audit ===\n")
print(f"Event types in last 14d: {dict(event_type_counter.most_common())}\n")

# Find silent-miss list (top-20 with NO events at all)
silent = [(d, sym) for (d, sym) in top20 if (d, sym) not in all_evs]
print(f"Silent-miss top-20 winners (no events at all): n={len(silent)}")
for d, sym in sorted(silent):
    # Check if symbol had ANY events on that day (different sym? case?)
    print(f"\n  {d}  {sym}")
    same_day_for_sym = [v for k, v in all_evs.items() if k[0] == d and k[1] == sym]
    if not same_day_for_sym:
        print(f"    → NO events anywhere for this sym on this day.")
    # Look at adjacent days
    prev_day = (datetime.fromisoformat(d) - timedelta(days=1)).strftime("%Y-%m-%d")
    next_day = (datetime.fromisoformat(d) + timedelta(days=1)).strftime("%Y-%m-%d")
    for adjd, label in [(prev_day, "prev"), (next_day, "next")]:
        adj_evs = all_evs.get((adjd, sym), [])
        if adj_evs:
            print(f"    {label} day ({adjd}): {len(adj_evs)} events")
            # Sample
            for dt, ev, reason in adj_evs[:3]:
                print(f"      {dt.strftime('%H:%M')} {ev:10s} {reason}")
        else:
            print(f"    {label} day ({adjd}): no events")

# Also: which symbols appear in events but never as 'candidate'/'entry'?
# Maybe whole evaluation pipeline is silent for some symbols.
print("\n\n=== Sym presence in event stream (sanity) ===")
syms_in_events = set(s for (_, s) in all_evs.keys())
silent_syms_in_stream = [s for d, s in silent if s in syms_in_events]
print(f"Of {len(silent)} silent-miss syms, {len(silent_syms_in_stream)} appear in event stream on OTHER days")
print(f"  → means: bot DOES scan these coins normally; the silent days are anomalies, not blacklist.")

# Per silent-miss day, count all top-20 events (was that day quiet generally?)
print("\n=== Per silent-miss day total event volume ===")
silent_days = sorted(set(d for d, _ in silent))
for d in silent_days:
    n_evs = sum(len(v) for k, v in all_evs.items() if k[0] == d)
    n_top20_that_day = sum(1 for k in top20 if k[0] == d)
    n_top20_with_events = sum(1 for k in top20 if k[0] == d and k in all_evs)
    print(f"  {d}: total events {n_evs}, top-20 winners that day {n_top20_that_day}, with events {n_top20_with_events}")
