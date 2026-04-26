"""Why no ALGO signal? Inspect last 7d of bot events for ALGOUSDT."""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
FILES = Path(__file__).resolve().parent

NOW = datetime.now(timezone.utc)
CUT = NOW - timedelta(days=7)

events_for_algo = []
for ev_file in (FILES / "bot_events.jsonl", Path(__file__).resolve().parent.parent / "bot_events.jsonl"):
    if not ev_file.exists():
        continue
    with io.open(ev_file, encoding="utf-8") as f:
        for ln in f:
            if "ALGO" not in ln:
                continue
            try: e = json.loads(ln)
            except: continue
            sym = e.get("sym") or e.get("symbol") or ""
            if sym != "ALGOUSDT":
                continue
            ts = e.get("ts", "")
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except:
                continue
            if dt < CUT:
                continue
            e["_dt"] = dt
            events_for_algo.append(e)

events_for_algo.sort(key=lambda e: e["_dt"])
print(f"=== ALGOUSDT events last 7d: {len(events_for_algo)} ===\n")

by_event = Counter(e.get("event", "?") for e in events_for_algo)
print("By event type:")
for k, v in by_event.most_common():
    print(f"  {k:20s} {v}")

print("\n=== Last 30 events (chronological) ===")
for e in events_for_algo[-30:]:
    ts   = e["_dt"].strftime("%m-%d %H:%M")
    typ  = e.get("event", "?")
    tf   = e.get("tf", "?")
    mode = e.get("mode", "?")
    reason = (e.get("reason") or e.get("decision", {}).get("reason") or "")[:90]
    code   = e.get("decision", {}).get("reason_code") or e.get("reason_code") or ""
    print(f"  {ts}  {typ:10s} {mode:14s}/{tf:3s}  [{code:25s}]  {reason}")

print("\n=== Block reasons (last 7d) ===")
by_reason = Counter()
by_code = Counter()
for e in events_for_algo:
    if e.get("event") == "blocked":
        r = e.get("reason") or e.get("decision", {}).get("reason") or "?"
        c = e.get("decision", {}).get("reason_code") or e.get("reason_code") or "?"
        by_reason[r[:80]] += 1
        by_code[c] += 1
print("By reason_code:")
for k, v in by_code.most_common():
    print(f"  {k:30s} {v}")
print("\nTop 15 raw reasons:")
for k, v in by_reason.most_common(15):
    print(f"  {v:>3}× {k}")

# Did the bot even SEE candidate ALGO recently?
print("\n=== ALGO candidate evaluations by day ===")
by_day = defaultdict(lambda: {"total":0, "blocked":0, "entry":0})
for e in events_for_algo:
    d = e["_dt"].strftime("%Y-%m-%d")
    by_day[d]["total"] += 1
    typ = e.get("event", "")
    if typ == "blocked":
        by_day[d]["blocked"] += 1
    elif typ == "entry":
        by_day[d]["entry"] += 1
for d, v in sorted(by_day.items()):
    print(f"  {d}: total={v['total']:>4}  blocked={v['blocked']:>4}  entry={v['entry']}")
