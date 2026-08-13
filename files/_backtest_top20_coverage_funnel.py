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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from block_reasons import normalize_block_reason  # noqa: E402

# The metric is meaningless without saying WHICH top-20 it counts. This project
# has already been burned by that: the recall denominator changed in April and
# PROJECT_CONTEXT records the two methodologies as "несопоставимы напрямую".
# Naming it inside the artifact makes a silent redefinition impossible.
DENOMINATOR = "top20_within_watchlist_from_top_gainer_dataset"

NOW = datetime.now(timezone.utc)
DAYS = 14
CUT = NOW - timedelta(days=DAYS)

# Canonical NS denominator is '#(top-20 in watchlist)' (CLAUDE.md s1).
# top_gainer_dataset spans a broader learning universe (~3-4x watchlist), so
# without this filter the funnel counts coins the bot CANNOT trade and reports
# a ~3-4x worse coverage than reality.
try:
    WATCHLIST = set(json.load(io.open(ROOT / "files" / "watchlist.json", encoding="utf-8")))
except Exception:
    WATCHLIST = set()

# 1) Top-20 symbols per UTC-date from top_gainer_dataset (watchlist-filtered)
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
        if WATCHLIST and e.get("symbol") not in WATCHLIST: continue
        d = dt.strftime("%Y-%m-%d")
        all_resolved_by_day[d] += 1
        if e.get("label_top20") == 1:
            top20_by_day[d].add(e.get("symbol"))

# 2) Bot events per (date, sym)
events_by = defaultdict(list)  # (date, sym) -> [(ev, reason)]
# Uptime: a day the bot was down yields only "no_event" and would be read as a
# silent-miss collapse (the 07-23..07-31 outage did exactly that). Days covering
# fewer than MIN_ACTIVE_HOURS are excluded and reported separately.
MIN_ACTIVE_HOURS = 18
active_hours = defaultdict(set)  # date -> {hour, ...}
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
        active_hours[d].add(dt.hour)
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

full_days = {d for d, hh in active_hours.items() if len(hh) >= MIN_ACTIVE_HOURS}
# Count excluded days against the WINDOW, not against the days that happen to
# have a top-20 winner — otherwise this number disagrees with the one
# _compute_early_capture prints (it reported 4 vs 5 here for the same period)
# and a report meant to be trustworthy contradicts itself.
_window_days = sorted({(NOW - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(DAYS)})
skipped_days = sorted(d for d in _window_days if d not in full_days)
for d, syms in sorted(top20_by_day.items()):
    if d not in full_days:
        continue
    for sym in syms:
        evs = events_by.get((d, sym), [])
        ev_types = [x[0] for x in evs]
        if "entry" in ev_types:
            cls = "entered"
        elif "blocked" in ev_types:
            cls = "blocked_only"
            # Group by the normalised code, not the raw sentence: blocked rows
            # carry free text in two languages with 310 distinct templates, so
            # counting them verbatim splits one gate across many rows and hides
            # which gate actually costs winners.
            raw = [x[1] for x in evs if x[0] == "blocked" and x[1]]
            codes = [normalize_block_reason(r) for r in raw]
            if codes:
                top_code = Counter(codes).most_common(1)[0][0]
                miss_reasons[top_code] += 1
                if len(blocked_examples) < 8:
                    example_raw = next((r for r in raw
                                        if normalize_block_reason(r) == top_code), "")
                    blocked_examples.append((d, sym, top_code, len(raw), example_raw))
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

print(f"\nDenominator: {DENOMINATOR}")

# Harm per gate: not "how often did it fire" (that is a volume statistic) but
# "how many top-20 winners did it cost", which is the only figure that ranks
# gates against the North Star.
print("\nHarm per gate — top-20 winners lost, by normalised block reason:")
if miss_reasons:
    for r, c in miss_reasons.most_common(15):
        print(f"  {r:<28s} {c:>3d} winners  {100*c/max(1,total):>5.1f}% of all top-20")
else:
    print("  none — no top-20 winner was blocked in this window")

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
    for d, sym, r, n, raw in blocked_examples:
        print(f"  {d}  {sym:<14} top_reason={r}  (n_blocks={n})")
        if raw:
            print(f"      {raw[:88]}")

print(f"\nuptime: {len(full_days)} full days counted; "
      f"{len(skipped_days)} day(s) excluded as down/partial"
      + (f" ({', '.join(skipped_days)})" if skipped_days else ""))

# METRIC_JSON for daily aggregator
metric = {
    "metric": "C1_C2_coverage_funnel",
    # Which top-20 this counts. Without it the rate is uninterpretable and a
    # later redefinition is undetectable.
    "denominator": DENOMINATOR,
    "days_full": len(full_days),
    "days_excluded_down": len(skipped_days),
    "n_top20_winners": total,
    "entered": classes.get("entered", 0),
    "blocked_only": classes.get("blocked_only", 0),
    # Emitted so the buckets reconcile: entered + blocked_only + no_event alone
    # left 3 of 26 winners unaccounted for, which reads as an arithmetic error
    # in any report that adds them up.
    "candidate_only": classes.get("candidate_only", 0),
    "other_event": classes.get("other_event", 0),
    "no_event": classes.get("no_event", 0),
    "buckets_sum": sum(classes.values()),
    "coverage_pct_raw": 100*classes.get("entered", 0)/max(1, total),
    "silent_miss_pct": 100*classes.get("no_event", 0)/max(1, total),
    # Harm, not volume: how many top-20 winners each gate cost.
    "blocked_reason_harm": {r: c for r, c in miss_reasons.most_common()},
}
print("\nMETRIC_JSON:" + json.dumps(metric))
