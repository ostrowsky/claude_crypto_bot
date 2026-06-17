"""Detection-lag diagnostic (objective = earliness, not P&L).

For watchlist top-20 movers the bot EVENTUALLY ENTERED: did we SEE the symbol
earlier the same day and BLOCK it before finally entering? If so, which gate
delayed us and by how long? This isolates the gates that cost earliness on
coins that turned out to be real movers (so relaxing them trades precision for
earlier alerts — acceptable for an alert channel, unlike a sized trade).

Source: bot_events.jsonl entries + blocked, joined to watchlist top-20 days.
ASCII-only.  pyembed\python.exe files\_backtest_detection_lag.py
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
DAYS = 21
CUT = datetime.now(timezone.utc) - timedelta(days=DAYS)
WL = set(json.load(io.open(ROOT/"files"/"watchlist.json", encoding="utf-8")))

# watchlist top-20 winner days
top = set()
for ln in io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8", errors="replace"):
    if '"label_top20"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    ts = e.get("ts")
    if not ts: continue
    dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
    if dt < CUT: continue
    if e.get("symbol") in WL and e.get("label_top20") == 1:
        top.add((dt.strftime("%Y-%m-%d"), e.get("symbol")))

# per (day,sym): chronological events
ev = defaultdict(list)
for ln in io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8", errors="replace"):
    if '"event"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    et = e.get("event", "")
    if et not in ("entry", "blocked"): continue
    sym = e.get("sym") or e.get("symbol") or ""
    try: dt = datetime.fromisoformat((e.get("ts","")).replace("Z","+00:00"))
    except: continue
    if dt < CUT: continue
    key = (dt.strftime("%Y-%m-%d"), sym)
    if key not in top: continue
    dec = e.get("decision", {}) or {}
    rc = dec.get("reason_code") or e.get("reason_code") or ""
    px = e.get("price") or e.get("entry_price") or dec.get("price")
    try: px = float(px)
    except (TypeError, ValueError): px = None
    ev[key].append((dt, et, rc, px))

entered_keys = [k for k, lst in ev.items() if any(x[1] == "entry" for x in lst)]
lags = []
first_block_reason = Counter()
delayed = 0
for k in entered_keys:
    lst = sorted(ev[k])
    entry_dt = min(x[0] for x in lst if x[1] == "entry")
    earlier_blocks = [x for x in lst if x[1] == "blocked" and x[0] < entry_dt]
    if earlier_blocks:
        delayed += 1
        first_b = min(earlier_blocks, key=lambda x: x[0])
        lag_min = (entry_dt - first_b[0]).total_seconds() / 60.0
        lags.append(lag_min)
        first_block_reason[first_b[2] or "(none)"] += 1

print("=" * 64)
print(f"Detection-lag on caught watchlist top-20 ({DAYS}d)")
print("=" * 64)
print(f"caught top-20 (entered): {len(entered_keys)}")
print(f"  of those, had EARLIER blocked signal(s) same day: {delayed} "
      f"({100*delayed/max(1,len(entered_keys)):.0f}%)")
if lags:
    lags.sort()
    med = lags[len(lags)//2]; mean = sum(lags)/len(lags)
    print(f"  lag from first block -> entry: median={med:.0f} min  mean={mean:.0f} min  "
          f"max={max(lags):.0f} min")
    print(f"\nWhich gate FIRST delayed us (count of earliest-block reason):")
    for rc, n in first_block_reason.most_common(10):
        print(f"  {rc or '(none)':<22} {n}")
print("\nRead: a gate appearing here blocked a coin that LATER proved a real")
print("mover and we entered anyway -> it cost earliness. High count + high lag =")
print("prime target to relax for earlier alerts (precision cost acceptable).")

# lag + price-chase per earliest-blocking gate. chase% = how much HIGHER we
# entered than the early-blocked signal price. Positive chase = the early signal
# was a genuinely earlier (cheaper) point and the gate cost us that earliness;
# chase <= 0 = price did NOT run after the early block, so the block was correct
# and the real setup came later (relaxing it would only add noise).
lag_by_reason = defaultdict(list)
chase_by_reason = defaultdict(list)
for k in entered_keys:
    lst = sorted(ev[k], key=lambda x: x[0])
    entries = [x for x in lst if x[1] == "entry"]
    entry_dt = min(x[0] for x in entries)
    entry_px = next((x[3] for x in entries if x[0] == entry_dt and x[3]), None)
    eb = [x for x in lst if x[1] == "blocked" and x[0] < entry_dt]
    if not eb:
        continue
    fb = min(eb, key=lambda x: x[0])
    rc = fb[2] or "(none)"
    lag_by_reason[rc].append((entry_dt - fb[0]).total_seconds()/60.0)
    if entry_px and fb[3]:
        chase_by_reason[rc].append((entry_px - fb[3]) / fb[3] * 100.0)

print("\nlag + price-chase by earliest-blocking gate:")
print(f"  {'gate':<20}{'n':>4}{'med_lag_min':>12}{'med_chase%':>12}{'mean_chase%':>12}")
for rc, ls in sorted(lag_by_reason.items(), key=lambda x: -len(x[1])):
    ls = sorted(ls)
    ch = sorted(chase_by_reason.get(rc, []))
    med_ch = ch[len(ch)//2] if ch else float("nan")
    mean_ch = sum(ch)/len(ch) if ch else float("nan")
    print(f"  {rc:<20}{len(ls):>4}{ls[len(ls)//2]:>12.0f}{med_ch:>12.2f}{mean_ch:>12.2f}")
print("\nchase% > 0 and large => gate cost real earliness (we entered higher).")
print("chase% ~ 0/negative => early block was correct; relaxing adds only noise.")
