"""H: bot misses some EOD top-20 winners entirely (no entry, no candidate).
For each (date, symbol) in top_gainer_dataset where label_top20=1:
classify what bot did with that symbol on that day.

Buckets:
- entered      : >=1 entry event
- blocked_only : >=1 blocked event, no entry  (record top reason)
- candidate_only: candidate event but no entry/block resolution
- no_event     : nothing (silent miss — worst)
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent

NOW = datetime.now(timezone.utc)
DAYS = 14
CUT = NOW - timedelta(days=DAYS)

# 1) Top-20 symbols per UTC-date from top_gainer_dataset
# (file has multiple snapshots per day per symbol; take the latest snapshot per date)
top20_by_day = defaultdict(set)  # date_str -> set(symbol)
all_resolved_by_day = defaultdict(int)  # date -> total resolved rows
with io.open(ROOT / "files" / "top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        try: e = json.loads(ln)
        except: continue
        ts_ms = e.get("ts")
        if not ts_ms: continue
        try: dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
        except: continue
        if dt < CUT: continue
        d = dt.strftime("%Y-%m-%d")
        all_resolved_by_day[d] += 1
        if e.get("label_top20") == 1:
            top20_by_day[d].add(e.get("symbol"))

# 2) Bot events per (date, sym)
events_by = defaultdict(list)  # (date, sym) -> [(ev, reason)]
with io.open(ROOT / "files" / "bot_events.jsonl", encoding="utf-8") as f:
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
        # decision events use various shapes; try multiple keys for reason
        reason = (e.get("decision") or {}).get("reason_code") if isinstance(e.get("decision"), dict) else None
        if not reason:
            reason = e.get("reason_code") or e.get("reason","") or ""
        events_by[(d, sym)].append((ev, reason))

# 3) Classify each (date, top20_sym)
classes = Counter()
miss_reasons = Counter()  # top blocked reason among misses
no_event_examples = []
blocked_examples = []
day_breakdown = defaultdict(lambda: Counter())

for d, syms in sorted(top20_by_day.items()):
    for sym in syms:
        evs = events_by.get((d, sym), [])
        ev_types = [x[0] for x in evs]
        if "entry" in ev_types:
            cls = "entered"
        elif "blocked" in ev_types:
            cls = "blocked_only"
            # find most-common reason
            reasons = [x[1] for x in evs if x[0] == "blocked" and x[1]]
            if reasons:
                top_r = Counter(reasons).most_common(1)[0][0]
                miss_reasons[top_r] += 1
                if len(blocked_examples) < 8:
                    blocked_examples.append((d, sym, top_r, len(reasons)))
        elif "candidate" in ev_types:
            cls = "candidate_only"
        elif evs:
            cls = "other_event"
        else:
            cls = "no_event"
            if len(no_event_examples) < 8:
                no_event_examples.append((d, sym))
        classes[cls] += 1
        day_breakdown[d][cls] += 1

total = sum(classes.values())
print(f"=== Top-20 coverage funnel, last {DAYS}d ===")
print(f"Days with TG data: {len(top20_by_day)}, total (date,sym) top-20 hits: {total}\n")

print("Bucket                n     %")
for k in ("entered","blocked_only","candidate_only","other_event","no_event"):
    n = classes.get(k, 0)
    print(f"  {k:<18} {n:>4d}  {100*n/max(1,total):>5.1f}%")

print("\nTop block-reasons among 'blocked_only' top-20 winners:")
for r, c in miss_reasons.most_common(15):
    print(f"  {r:<32s} {c:>3d}")

print("\nDay-by-day breakdown:")
print(f"  {'date':<12} {'n_top20':>7} {'entered':>8} {'blocked':>8} {'no_event':>9}")
for d in sorted(day_breakdown.keys()):
    cb = day_breakdown[d]
    n = sum(cb.values())
    print(f"  {d:<12} {n:>7d} {cb.get('entered',0):>8d} {cb.get('blocked_only',0):>8d} {cb.get('no_event',0):>9d}")

if no_event_examples:
    print("\nExamples of NO-EVENT top-20 winners (silent misses):")
    for d, sym in no_event_examples:
        print(f"  {d}  {sym}")

if blocked_examples:
    print("\nExamples of blocked-only top-20 winners:")
    for d, sym, r, n in blocked_examples:
        print(f"  {d}  {sym:<14} top_reason={r}  (n_blocks={n})")

# METRIC_JSON for daily aggregator
metric = {
    "metric": "C1_C2_coverage_funnel",
    "n_top20_winners": total,
    "entered": classes.get("entered", 0),
    "blocked_only": classes.get("blocked_only", 0),
    "no_event": classes.get("no_event", 0),
    "coverage_pct_raw": 100*classes.get("entered", 0)/max(1, total),
    "silent_miss_pct": 100*classes.get("no_event", 0)/max(1, total),
}
print("\nMETRIC_JSON:" + json.dumps(metric))
