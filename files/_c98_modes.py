"""All C98 signal evaluations today by mode + recent strong_trend 1h winners."""
import json
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

today = datetime.now(timezone.utc).date()
p = Path("files/bot_events.jsonl")
if not p.exists():
    p = Path("bot_events.jsonl")

c98_by_mode = Counter()
c98_all = []
strong_trend_1h_entries = []
strong_trend_1h_blocks_today = Counter()

for line in p.open(encoding="utf-8", errors="ignore"):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    ts = r.get("ts", "")
    try:
        t = datetime.fromisoformat(ts.rstrip("Z")).replace(tzinfo=timezone.utc)
    except Exception:
        continue
    if t.date() != today:
        continue
    sym = r.get("sym")
    ev = r.get("event")
    mode = r.get("signal_type") or r.get("mode") or ""
    tf = r.get("tf") or ""

    if sym == "C98USDT":
        c98_by_mode[(ev, tf, mode)] += 1
        c98_all.append(r)

    if tf == "1h" and mode == "strong_trend":
        if ev == "entry":
            strong_trend_1h_entries.append(r)
        elif ev == "blocked":
            rc = r.get("reason_code") or r.get("reason") or ""
            strong_trend_1h_blocks_today[rc[:60]] += 1

print("=== C98USDT all events today by (event, tf, mode) ===")
for k, v in c98_by_mode.most_common():
    print(f"  {v:>3}  {k}")

print("\n=== Latest 5 C98 blocks (any mode/tf) ===")
blocks = [r for r in c98_all if r.get("event") == "blocked"]
for r in blocks[-5:]:
    print(f"  {r.get('ts')}  tf={r.get('tf')}  mode={r.get('signal_type') or r.get('mode')}  "
          f"reason={(r.get('reason_code') or r.get('reason') or '')[:90]}")

print(f"\n=== strong_trend 1h entries across ALL symbols today: {len(strong_trend_1h_entries)} ===")
for r in strong_trend_1h_entries:
    print(f"  {r.get('ts')}  {r.get('sym'):<12} adx={r.get('adx')}  rsi={r.get('rsi')}  "
          f"score={r.get('candidate_score') or r.get('score')}")

print(f"\n=== strong_trend 1h block reasons today: {sum(strong_trend_1h_blocks_today.values())} total ===")
for k, v in strong_trend_1h_blocks_today.most_common(10):
    print(f"  {v:>4}  {k}")
