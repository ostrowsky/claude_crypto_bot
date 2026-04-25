"""
Diagnose per-symbol value in watchlist:
  - How many positive labels (top20 hits) each symbol has contributed
  - How recently they've been contributing (last 30d vs historical)
  - Which symbols are "dead weight" (never hit top-20 or only old hits)

Goal: show which coins can be safely removed from watchlist without reducing
the positive training signal. The critical insight is that label_top20 is the
GLOBAL Binance top-20, so removing watchlist coins that never hit it doesn't
reduce positive-label count in training.
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
FILES = Path(__file__).resolve().parent

NOW = datetime.now(timezone.utc)
CUT_30D = (NOW - timedelta(days=30)).strftime("%Y-%m-%d")
CUT_60D = (NOW - timedelta(days=60)).strftime("%Y-%m-%d")

# Per symbol: count of top20 hits total, last 30d, last 60d
per_sym = defaultdict(lambda: {
    "rows": 0, "top20": 0, "top10": 0, "top5": 0,
    "top20_30d": 0, "top20_60d": 0, "last_hit": None,
})

with io.open(FILES / "top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        try: e = json.loads(ln)
        except: continue
        sym = e.get("symbol") or e.get("sym", "")
        if not sym: continue
        ts = e.get("ts") or e.get("date", "")
        if isinstance(ts, (int, float)):
            d = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        elif isinstance(ts, str):
            d = ts[:10]
        else:
            continue
        per_sym[sym]["rows"] += 1
        if e.get("label_top20"):
            per_sym[sym]["top20"] += 1
            if d >= CUT_30D: per_sym[sym]["top20_30d"] += 1
            if d >= CUT_60D: per_sym[sym]["top20_60d"] += 1
            if per_sym[sym]["last_hit"] is None or d > per_sym[sym]["last_hit"]:
                per_sym[sym]["last_hit"] = d
        if e.get("label_top10"): per_sym[sym]["top10"] += 1
        if e.get("label_top5"):  per_sym[sym]["top5"]  += 1

# Load current watchlist
wl = json.loads((FILES / "watchlist.json").read_text(encoding="utf-8"))

print(f"=== Watchlist value distribution ({len(wl)} symbols) ===\n")
print(f"Period: data spans {min(per_sym[s]['last_hit'] or '9999' for s in wl if per_sym[s]['rows'] > 0)} "
      f"to 2026-04-24")

total_top20 = sum(v["top20"] for v in per_sym.values())
total_top20_30d = sum(v["top20_30d"] for v in per_sym.values())
total_rows = sum(v["rows"] for v in per_sym.values())

print(f"Total rows: {total_rows:,}")
print(f"Total top20 hits (global Binance): {total_top20}")
print(f"Top20 hits last 30d: {total_top20_30d}")

# Sort watchlist by top20 contribution (last 60d)
rows = []
for sym in wl:
    v = per_sym[sym]
    rows.append({
        "sym": sym,
        "rows": v["rows"],
        "t20_total": v["top20"],
        "t20_60d": v["top20_60d"],
        "t20_30d": v["top20_30d"],
        "last_hit": v["last_hit"] or "never",
    })

rows.sort(key=lambda r: (-r["t20_60d"], -r["t20_30d"], -r["t20_total"]))

print(f"\n=== Top 30 most valuable watchlist symbols (by top20 hits, last 60d) ===")
print(f"  {'#':>3} {'Symbol':14s}  {'rows':>5s}  {'t20_all':>7s}  {'t20_60d':>7s}  {'t20_30d':>7s}  last_hit")
print("  " + "-" * 72)
for i, r in enumerate(rows[:30], 1):
    print(f"  {i:>3d} {r['sym']:14s}  {r['rows']:>5d}  {r['t20_total']:>7d}  "
          f"{r['t20_60d']:>7d}  {r['t20_30d']:>7d}  {r['last_hit']}")

# Find dead weight: never hit top20, or haven't hit in 60+ days
dead_weight = [r for r in rows if r["t20_60d"] == 0]
never_hit   = [r for r in rows if r["t20_total"] == 0]
print(f"\n=== Dead weight analysis ===")
print(f"  Symbols NEVER hit top20 (199 days): {len(never_hit)}")
if never_hit:
    print(f"  → {', '.join(r['sym'] for r in never_hit[:20])}")

print(f"  Symbols with NO top20 hits in last 60d: {len(dead_weight)}")
if dead_weight:
    dead_syms = [r['sym'] for r in dead_weight]
    print(f"  → {', '.join(dead_syms[:30])}")
    if len(dead_syms) > 30:
        print(f"  ... and {len(dead_syms)-30} more")

# Cumulative coverage: if we keep top N most valuable symbols
print(f"\n=== Cumulative coverage: if we KEEP top N symbols ===")
total_60d = sum(r["t20_60d"] for r in rows)
total_30d = sum(r["t20_30d"] for r in rows)
cum_60d = 0; cum_30d = 0
for i, r in enumerate(rows, 1):
    cum_60d += r["t20_60d"]
    cum_30d += r["t20_30d"]
    if i in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 105):
        pct_60d = cum_60d / total_60d * 100 if total_60d else 0
        pct_30d = cum_30d / total_30d * 100 if total_30d else 0
        print(f"  Keep top {i:>3d}: covers {cum_60d:>3d}/{total_60d} hits 60d ({pct_60d:.0f}%), "
              f"{cum_30d:>3d}/{total_30d} hits 30d ({pct_30d:.0f}%)")

# Training-data impact
print(f"\n=== Training data size impact ===")
for keep_n in [30, 40, 50, 60, 70, 80]:
    kept_syms = set(r["sym"] for r in rows[:keep_n])
    kept_rows = sum(per_sym[s]["rows"] for s in kept_syms)
    kept_top20 = sum(per_sym[s]["top20"] for s in kept_syms)
    pct_rows = kept_rows / total_rows * 100
    pct_top20 = kept_top20 / total_top20 * 100
    print(f"  Keep top {keep_n}: {kept_rows:>6,} rows ({pct_rows:.0f}% of data),  "
          f"{kept_top20:>5,} positive labels ({pct_top20:.0f}%)")

# Alternative: by frequency threshold (remove symbols with <N hits in 60d)
print(f"\n=== Alternative: remove symbols with < N top20 hits in last 60d ===")
for min_hits in [0, 1, 2, 3, 5]:
    kept = [r for r in rows if r["t20_60d"] >= min_hits]
    removed = len(rows) - len(kept)
    kept_top20_60d = sum(r["t20_60d"] for r in kept)
    print(f"  min_hits>={min_hits}: keep {len(kept):>3d}, remove {removed:>3d}, "
          f"preserved {kept_top20_60d}/{total_60d} ({kept_top20_60d/total_60d*100:.0f}%) of 60d hits")
