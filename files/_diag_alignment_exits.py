"""
Diagnostic: alignment-mode exit patterns.

Questions:
  1. Distribution of bars_held for alignment exits
  2. P&L by bars_held bucket (fast ≤3 vs slow >3)
  3. Exit reason breakdown
  4. Relation to trail_k / stop type
  5. "Near-top" entry pattern: how often do quick exits follow entries where
     price barely moved up before reversing?
  6. Comparison: last 14 days vs all history
"""
from __future__ import annotations
import json, io, sys, math
from pathlib import Path
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding="utf-8")
FILES = Path(__file__).resolve().parent

# ── Load paired entry+exit for alignment ─────────────────────────────────
entries, exits_ = [], []
with io.open(FILES / "bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        try: e = json.loads(ln)
        except: continue
        et = e.get("event")
        if et == "entry":   entries.append(e)
        elif et == "exit":  exits_.append(e)

entries.sort(key=lambda e: e.get("ts",""))
exits_.sort(key=lambda e: e.get("ts",""))

ent_q = defaultdict(list)
for e in entries:
    ent_q[e.get("sym","")].append(e)

trades = []
for x in exits_:
    sym = x.get("sym","")
    q = ent_q.get(sym, [])
    if not q: continue
    ent = q.pop(0)
    pnl = x.get("pnl_pct")
    if pnl is None: continue
    bars = x.get("bars_held") or 0
    trades.append({
        "sym": sym,
        "tf": ent.get("tf",""),
        "mode": ent.get("mode",""),
        "pnl": float(pnl),
        "bars": int(bars),
        "exit_reason": x.get("exit_reason") or x.get("reason",""),
        "entry_ts": ent.get("ts",""),
        "trail_k": ent.get("trail_k"),
        "entry_price": float(ent.get("price") or 0),
        "exit_price": float(x.get("price") or 0),
    })

aln = [t for t in trades if t["mode"] == "alignment"]
aln_15m = [t for t in aln if t["tf"] == "15m"]
aln_1h  = [t for t in aln if t["tf"] == "1h"]

CUTOFF_14D = "2026-04-06"

def agg(lst, label):
    if not lst:
        print(f"  {label:55s} n=0"); return
    n = len(lst); pnls = [t["pnl"] for t in lst]
    wins = sum(1 for p in pnls if p > 0)
    avg = sum(pnls)/n
    sd = (sum((p-avg)**2 for p in pnls)/n)**0.5 if n>1 else 0
    sh = avg/sd*math.sqrt(n) if sd>0 else 0
    print(f"  {label:55s} n={n:>4d}  win={wins/n*100:>5.1f}%  "
          f"avg={avg:+.3f}%  sharpe={sh:+.2f}")

print("=" * 70)
print("ALIGNMENT EXIT DIAGNOSTIC")
print("=" * 70)
print(f"\nTotal alignment trades: {len(aln)}  (15m={len(aln_15m)}  1h={len(aln_1h)})")

# ── 1. bars_held distribution ─────────────────────────────────────────────
print(f"\n=== 1. BARS HELD distribution (15m alignment, all history) ===")
buckets = [(1,1),(2,2),(3,3),(4,5),(6,10),(11,20),(21,36),(37,999)]
for lo, hi in buckets:
    sub = [t for t in aln_15m if lo <= t["bars"] <= hi]
    agg(sub, f"  bars [{lo:>2d}–{hi:>3d}]")

print(f"\n=== 1b. BARS HELD distribution (1h alignment) ===")
for lo, hi in buckets:
    sub = [t for t in aln_1h if lo <= t["bars"] <= hi]
    agg(sub, f"  bars [{lo:>2d}–{hi:>3d}]")

# ── 2. Fast vs slow exits ──────────────────────────────────────────────────
print(f"\n=== 2. FAST (≤3 bars) vs SLOW (>3 bars) — 15m alignment ===")
fast = [t for t in aln_15m if t["bars"] <= 3]
slow = [t for t in aln_15m if t["bars"] > 3]
agg(fast, "fast exits (≤3 bars)")
agg(slow, "slow exits (>3 bars)")
if fast and slow:
    f_avg = sum(t["pnl"] for t in fast)/len(fast)
    s_avg = sum(t["pnl"] for t in slow)/len(slow)
    print(f"  delta slow-fast = {s_avg - f_avg:+.3f}%")

print(f"\n=== 2b. FAST vs SLOW — 1h alignment ===")
fast1h = [t for t in aln_1h if t["bars"] <= 3]
slow1h = [t for t in aln_1h if t["bars"] > 3]
agg(fast1h, "fast exits (≤3 bars)")
agg(slow1h, "slow exits (>3 bars)")

# ── 3. Fast exits: what % of all alignment? ───────────────────────────────
print(f"\n=== 3. Fast exits share (15m alignment) ===")
for period, label in [(CUTOFF_14D, "last 14d"), ("2026-01-01", "all history")]:
    sub = [t for t in aln_15m if t["entry_ts"] >= period]
    n_fast = sum(1 for t in sub if t["bars"] <= 3)
    print(f"  {label:12s}: {n_fast}/{len(sub)} fast exits = "
          f"{n_fast/len(sub)*100:.1f}%" if sub else f"  {label}: no data")

# ── 4. Exit reason breakdown ──────────────────────────────────────────────
print(f"\n=== 4. EXIT REASON breakdown (15m alignment, all) ===")
by_reason = Counter(t["exit_reason"][:60] if t["exit_reason"] else "?" for t in aln_15m)
for reason, cnt in by_reason.most_common(12):
    sub = [t for t in aln_15m if (t["exit_reason"] or "")[:60] == reason]
    fast_n = sum(1 for t in sub if t["bars"] <= 3)
    avg_pnl = sum(t["pnl"] for t in sub)/len(sub)
    print(f"  {cnt:>4d}x  fast={fast_n:>3d}  avg={avg_pnl:+.3f}%  {reason}")

# ── 5. Trail_k distribution for fast vs slow ──────────────────────────────
print(f"\n=== 5. TRAIL_K for fast (≤3) vs slow (>3) — 15m alignment ===")
def trail_stats(lst, label):
    tk = [t["trail_k"] for t in lst if t["trail_k"] is not None]
    if not tk:
        print(f"  {label}: no trail_k data"); return
    avg = sum(tk)/len(tk)
    mn = min(tk); mx = max(tk)
    print(f"  {label:25s} n={len(tk)}  mean={avg:.4f}  min={mn:.4f}  max={mx:.4f}")

trail_stats(fast, "fast (≤3 bars)")
trail_stats(slow, "slow (>3 bars)")

# ── 6. P&L by trail_k bucket ──────────────────────────────────────────────
print(f"\n=== 6. P&L by TRAIL_K bucket (15m alignment) ===")
tk_buckets = [(0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 10.0)]
for lo, hi in tk_buckets:
    sub = [t for t in aln_15m if t["trail_k"] is not None and lo <= t["trail_k"] < hi]
    if sub:
        fast_n = sum(1 for t in sub if t["bars"] <= 3)
        agg(sub, f"  trail_k [{lo:.1f},{hi:.1f})  fast={fast_n}/{len(sub)}")

# ── 7. Recent trend (14d) ──────────────────────────────────────────────────
print(f"\n=== 7. RECENT 14 days (2026-04-06+) — alignment 15m ===")
recent = [t for t in aln_15m if t["entry_ts"] >= CUTOFF_14D]
agg(recent, "all recent 15m alignment")
fast_r = [t for t in recent if t["bars"] <= 3]
slow_r = [t for t in recent if t["bars"] > 3]
agg(fast_r, "  fast (≤3)")
agg(slow_r, "  slow (>3)")
print(f"  fast% = {len(fast_r)/len(recent)*100:.1f}%" if recent else "")

# ── 8. Price move at exit for fast trades ─────────────────────────────────
print(f"\n=== 8. PRICE MOVE profile for fast exits (≤3 bars, 15m aln) ===")
if fast:
    pnls = sorted(t["pnl"] for t in fast)
    n = len(pnls)
    p10 = pnls[int(n*0.10)]
    p25 = pnls[int(n*0.25)]
    p50 = pnls[int(n*0.50)]
    p75 = pnls[int(n*0.75)]
    p90 = pnls[int(n*0.90)]
    wins = sum(1 for p in pnls if p>0)
    print(f"  n={n}  win={wins/n*100:.1f}%")
    print(f"  p10={p10:+.3f}%  p25={p25:+.3f}%  p50={p50:+.3f}%  "
          f"p75={p75:+.3f}%  p90={p90:+.3f}%")
    near_zero = sum(1 for p in pnls if abs(p) <= 0.1)
    print(f"  |pnl| ≤ 0.1% (near-zero): {near_zero}/{n} = {near_zero/n*100:.1f}%")
