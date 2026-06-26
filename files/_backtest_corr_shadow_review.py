"""Review the correlation-guard SHADOW data before deciding to enforce
(CORR_GUARD_SHADOW=False). Run after ~2-3 days of shadow accumulation.

The enforce decision is a COVERAGE-vs-RISK trade: capping correlated clusters
reduces synchronized stop-outs (good) but blocks the 3rd+ correlated entry,
which could have been a top-20 winner (coverage loss). This quantifies it from
the shadow would-blocks:

  - rate: would-blocks/day (how many entries enforcement would cap)
  - of would-blocked: how many were watchlist top-20 (coverage we'd LOSE)
  - avg ret_5 / win% of would-blocked (were they net-negative = correct to cut,
    or winners = enforcing hurts)

ENFORCE if would-blocked are mostly net-negative / few top-20 (we cut losing
correlated stack). HOLD SHADOW / loosen if would-blocked include many top-20
winners. ASCII-only.  pyembed\python.exe files\_backtest_corr_shadow_review.py
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
WL = set(json.load(io.open(ROOT/"files"/"watchlist.json", encoding="utf-8")))


def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


# watchlist top-20 (day,sym)
top = set()
for ln in io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8", errors="replace"):
    if '"label_top20"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    ts = e.get("ts")
    if not ts: continue
    d = datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime("%Y-%m-%d")
    if e.get("symbol") in WL and e.get("label_top20") == 1:
        top.add((d, e.get("symbol")))

by_day = Counter()
r5s = []
n_top = 0
n_resolved = 0
n = 0
for ln in io.open(ROOT/"files"/"critic_dataset.jsonl", encoding="utf-8", errors="replace"):
    if "correlation_guard_shadow" not in ln: continue
    try: e = json.loads(ln)
    except: continue
    if str((e.get("decision", {}) or {}).get("reason_code", "")) != "correlation_guard_shadow":
        continue
    n += 1
    d = str(e.get("ts_signal", ""))[:10]; sym = e.get("sym")
    by_day[d] += 1
    if (d, sym) in top:
        n_top += 1
    r5 = _f((e.get("labels", {}) or {}).get("ret_5"))
    if r5 is not None:
        r5s.append(r5); n_resolved += 1

print("=" * 64)
print("Correlation-guard SHADOW review (would-block analysis)")
print("=" * 64)
print(f"shadow would-blocks logged: {n}")
if n == 0:
    print("\nNo shadow data yet — let it accumulate ~2-3 days, then re-run.")
    sys.exit(0)
days = sorted(by_day)
print(f"  span {days[0]}..{days[-1]}  (~{n/max(1,len(days)):.1f}/day)")
print(f"  of those, watchlist top-20 (coverage we'd LOSE by enforcing): {n_top} "
      f"({100*n_top/n:.0f}%)")
if r5s:
    avg = sum(r5s)/len(r5s); win = sum(1 for r in r5s if r > 0)/len(r5s)*100
    print(f"  resolved {n_resolved}: avg ret_5={avg:+.3f}%  win={win:.0f}%")
print("\nVERDICT GUIDE:")
print("  ENFORCE (flip CORR_GUARD_SHADOW=False) if would-blocked are mostly")
print("  net-negative (avg ret_5 < 0, low top-20 share) — we cut a losing")
print("  correlated stack. HOLD/loosen if many top-20 winners would be lost.")
