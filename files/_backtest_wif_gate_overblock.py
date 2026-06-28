"""WIF/NEAR-class over-block validation (max period, 2026-03-24..now).

Question: do the impulse_speed volatility guard (impulse_guard: daily_range/RSI/
ADX), the clone-'Meme' cap (clone_signal_guard), the late-rotation guard
(late_impulse_rotation) and the ranker veto over-block eventual watchlist top-20
gainers (our North Star: earliest catch of top-20)? WIF was blocked 101x/day by
exactly these and entered only late.

Two lenses (NS objective = coverage/earliness, NOT P&L):
  (1) section-4 ret_5 lens: blocked-bucket avg ret_5 vs actual take baseline.
      Positive & >> take => over-blocking profitable signals.
  (2) NS coverage lens: of watchlist top-20 (day,sym), how many were blocked by
      gate G in >=1 window AND never taken that day = a true coverage MISS
      attributable to G. This is the earliness-relevant number.

Read-only.  pyembed\python.exe files\_backtest_wif_gate_overblock.py
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
WL = set(json.load(io.open(ROOT/"files"/"watchlist.json", encoding="utf-8")))
GATES = ["impulse_guard", "clone_signal_guard", "late_impulse_rotation",
         "ranker_hard_veto", "trend_quality", "entry_score"]


def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def _avg(xs): return sum(xs)/len(xs) if xs else 0.0


def _sd(xs):
    if len(xs) < 2: return 0.0
    m = _avg(xs); return (sum((v-m)**2 for v in xs)/(len(xs)-1))**0.5


# ---- watchlist top-20 (day,sym) ----
top = set()
for ln in io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8", errors="replace"):
    if '"label_top20"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    ts = e.get("ts")
    if not ts or e.get("label_top20") != 1 or e.get("symbol") not in WL: continue
    d = datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime("%Y-%m-%d")
    top.add((d, e.get("symbol")))

# ---- scan critic_dataset ----
buckets = defaultdict(lambda: {"r5": [], "wins": 0})
taken_daysym = set()
blocked_by = defaultdict(set)   # gate -> {(day,sym)}
take_r5 = []

for ln in io.open(ROOT/"files"/"critic_dataset.jsonl", encoding="utf-8", errors="replace"):
    if '"action"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    dec = e.get("decision", {}) or {}; lab = e.get("labels", {}) or {}
    a = dec.get("action"); rc = dec.get("reason_code")
    sym = e.get("sym"); ts = e.get("ts_signal") or e.get("ts")
    d = str(ts)[:10] if ts else None
    r5 = _f(lab.get("ret_5"))
    if a == "take":
        if r5 is not None: take_r5.append(r5)
        if d and sym: taken_daysym.add((d, sym))
    elif a == "blocked" and rc:
        b = buckets[rc]
        if r5 is not None:
            b["r5"].append(r5); b["wins"] += 1 if r5 > 0 else 0
        if d and sym and rc in GATES:
            blocked_by[rc].add((d, sym))

# ---- report ----
tk = _avg(take_r5); tkw = sum(1 for r in take_r5 if r > 0)/max(1, len(take_r5))*100
print("=" * 78)
print(f"Gate over-block validation  (max period; take baseline n={len(take_r5)} "
      f"avg_r5={tk:+.3f}% win={tkw:.0f}%)")
print(f"watchlist top-20 (day,sym) total in window: {len(top)}")
print("=" * 78)
print(f"{'gate':<22}{'n':>6}{'avg_r5%':>9}{'win%':>6}{'sharpe':>8}   "
      f"{'NS: top20 blocked':>16}{'  of-which MISSED (never taken)':>0}")
for g in GATES:
    b = buckets.get(g, {"r5": [], "wins": 0})
    n = len(b["r5"]); avg = _avg(b["r5"]); sd = _sd(b["r5"])
    win = b["wins"]/max(1, n)*100
    sharpe = (avg/sd*(n**0.5)) if sd > 0 else 0.0
    t20 = {k for k in blocked_by[g] if k in top}
    missed = {k for k in t20 if k not in taken_daysym}
    over = "OVER-BLOCK" if (avg > 0 and avg > tk + 0.05) else ("ok" if avg < tk else "neutral")
    print(f"{g:<22}{n:>6}{avg:>+9.3f}{win:>6.0f}{sharpe:>8.2f}   "
          f"top20_blk={len(t20):>3}  MISSED={len(missed):>3}   [{over}]")

# total unique top-20 missed where ANY of the impulse/clone gates was a blocker
focus = {"impulse_guard", "clone_signal_guard", "late_impulse_rotation"}
focus_missed = set()
for g in focus:
    focus_missed |= {k for k in blocked_by[g] if k in top and k not in taken_daysym}
print("-" * 78)
print(f"UNIQUE watchlist top-20 MISSED with impulse/clone/late as a blocker: "
      f"{len(focus_missed)} / {len(top)} ({100*len(focus_missed)/max(1,len(top)):.0f}% of top-20)")
print("examples:", sorted(focus_missed)[:12])
print("\nNOTE: 'blocked' counts windows where the gate fired; MISSED = top-20 the")
print("bot NEVER took that day (true coverage loss). A high MISSED with positive")
print("avg_r5 = the gate is the §4 over-block pattern -> relax candidate (post-BT).")
