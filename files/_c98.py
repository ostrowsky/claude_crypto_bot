import json
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

today = datetime.now(timezone.utc).date()
p = Path("files/bot_events.jsonl")
if not p.exists():
    p = Path("bot_events.jsonl")

rows = []
for line in p.open(encoding="utf-8", errors="ignore"):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get("sym") != "C98USDT":
        continue
    ts = r.get("ts", "")
    try:
        t = datetime.fromisoformat(ts.rstrip("Z")).replace(tzinfo=timezone.utc)
    except Exception:
        continue
    if t.date() != today:
        continue
    rows.append((t, r))
rows.sort(key=lambda x: x[0])
rows = [r for _, r in rows]

print(f"C98 events today: {len(rows)}")
c = Counter()
for r in rows:
    c[(r.get("event"), r.get("reason_code") or r.get("reason") or "")] += 1
for k, v in c.most_common(30):
    print(f"  {v:>4}  {k}")

print("\nAll events after 13:00 UTC (post-cooldown):")
for r in rows:
    ts = r.get("ts", "")
    if ts < "2026-04-18T13:00":
        continue
    print(" ", r.get("ts"), r.get("event"), r.get("tf"),
          r.get("signal_type") or r.get("mode"),
          "adx=", r.get("adx"), "rsi=", r.get("rsi"),
          "vol=", r.get("vol_x"),
          "reason=", (r.get("reason_code") or r.get("reason") or "")[:100])
