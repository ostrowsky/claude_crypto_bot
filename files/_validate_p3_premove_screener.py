"""P3: pre-move screener — would soft heads-up channel improve early signal?
For each top-20 winner-day, find earliest intraday snapshot satisfying:
  feat['tg_return_since_open'] >= 0.03  (i.e. >= +3% intraday)
Compare snapshot_time vs actual entry_time. If snapshot fires earlier — useful.

Decision: only worthwhile if (a) >=70% of winners would get earlier heads-up,
  AND (b) median lead time >=2h, AND (c) false-positive rate (heads-up on
  non-winners) acceptable on watchlist.
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
NOW = datetime.now(timezone.utc); CUT = NOW - timedelta(days=14)

# Snapshots per (date, sym): list of (dt, tg_return_since_open) sorted ASC
snapshots = defaultdict(list)
top20 = set()
all_winrate_baseline = defaultdict(int)  # date -> total snapshots above threshold (any sym)
fp_count = defaultdict(int)  # (d, sym) for non-winners hit by heads-up rule

with io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        try: e = json.loads(ln)
        except: continue
        ts_ms = e.get("ts");
        if not ts_ms: continue
        dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
        if dt < CUT: continue
        sym = e.get("symbol")
        d = dt.strftime("%Y-%m-%d")
        feat = e.get("features") or {}
        tgret = feat.get("tg_return_since_open")
        # tg_return_since_open is decimal (0.03 = +3%); but some rows
        # show very large values — likely a different normalization for
        # extreme movers. Cap to reasonable.
        if tgret is None: continue
        snapshots[(d, sym)].append((dt, float(tgret)))
        if e.get("label_top20") == 1:
            top20.add((d, sym))

# First entry per (date, sym) from bot
first_entry = {}
with io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if '"event":"entry"' not in ln and '"event": "entry"' not in ln: continue
        try: e = json.loads(ln)
        except: continue
        if e.get("event") != "entry": continue
        ts = e.get("ts","")
        try: dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        except: continue
        if dt < CUT: continue
        sym = e.get("sym") or e.get("symbol") or ""
        d = dt.strftime("%Y-%m-%d")
        if not sym: continue
        prev = first_entry.get((d, sym))
        if prev is None or dt < prev:
            first_entry[(d, sym)] = dt

THRESHOLD = 0.03  # +3% since open

print("=== P3: pre-move screener simulation ===\n")
print(f"Trigger rule: tg_return_since_open >= {THRESHOLD*100:.0f}% (decimal {THRESHOLD})\n")

# For top-20 winners with bot entry: lead time
lead_times_h = []
no_screener_hit = 0  # winners where threshold never reached at any snapshot
no_entry_winners = 0
hit_pre_entry = 0

for key in sorted(top20):
    d, sym = key
    snaps = sorted(snapshots.get(key, []))
    if not snaps: continue
    # Earliest snapshot meeting threshold
    earliest_hit = None
    for dt, val in snaps:
        if val >= THRESHOLD:
            earliest_hit = dt; break
    entry_dt = first_entry.get(key)
    if earliest_hit is None:
        no_screener_hit += 1
        continue
    if entry_dt is None:
        no_entry_winners += 1
        # Heads-up would still fire — counts as "rescue"
        continue
    delta_h = (entry_dt - earliest_hit).total_seconds() / 3600
    lead_times_h.append(delta_h)
    if delta_h > 0: hit_pre_entry += 1

n_winners = len(top20)
print(f"Top-20 winners (date,sym) hits in last 14d: {n_winners}")
print(f"  Heads-up would fire on: {sum(1 for k in top20 if any(v >= THRESHOLD for _, v in snapshots.get(k,[])))}/{n_winners}")
print(f"  Heads-up never fired (threshold not met any snapshot): {no_screener_hit}")
print(f"  Of fired heads-ups for ENTERED winners: hit BEFORE entry: {hit_pre_entry}/{len(lead_times_h)}")
if lead_times_h:
    lt = sorted(lead_times_h)
    median = lt[len(lt)//2]
    mean = sum(lt)/len(lt)
    pos = [x for x in lt if x > 0]
    print(f"  Lead-time stats: median={median:+.1f}h, mean={mean:+.1f}h, positive-only median={(sorted(pos)[len(pos)//2] if pos else 0):+.1f}h")

# False-positive rate: how many NON-top-20 (date, sym) pairs would also fire?
print("\n=== False-positive analysis ===")
all_pairs = set(snapshots.keys())
non_winners = all_pairs - top20
fp = sum(1 for k in non_winners if any(v >= THRESHOLD for _, v in snapshots.get(k,[])))
print(f"Non-top-20 (date, sym) pairs: {len(non_winners)}")
print(f"  Would fire heads-up: {fp} ({100*fp/max(1,len(non_winners)):.1f}%)")
print(f"  Daily heads-up volume estimate: {fp/14:.0f} non-winners/day + ~{n_winners/14:.0f} winners/day")

# Try tighter threshold to cut FP
print("\n=== Threshold sweep (sensitivity vs FP) ===")
print(f"  {'thr':<6} {'winner-hit %':>13} {'FP/day':>8} {'lead-time mdn(h)':>17}")
for thr in [0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
    wh = sum(1 for k in top20 if any(v >= thr for _, v in snapshots.get(k,[])))
    fp_cnt = sum(1 for k in non_winners if any(v >= thr for _, v in snapshots.get(k,[])))
    lts = []
    for k in top20:
        snaps = sorted(snapshots.get(k, []))
        eh = next((dt for dt, v in snaps if v >= thr), None)
        ed = first_entry.get(k)
        if eh and ed:
            lts.append((ed - eh).total_seconds()/3600)
    mdn = sorted(lts)[len(lts)//2] if lts else 0
    print(f"  {thr:<6.2f} {100*wh/max(1,n_winners):>12.1f}% {fp_cnt/14:>7.1f} {mdn:>+15.1f}")
