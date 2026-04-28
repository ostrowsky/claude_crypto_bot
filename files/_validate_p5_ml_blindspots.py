"""P5: ML signal-model blind-spots.
For each top-20 winner blocked by 'ML proba X outside zone',
extract X. How many winner-days have ml_proba < 0.10 (extreme blind spot)?
"""
from __future__ import annotations
import json, io, sys, re
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
NOW = datetime.now(timezone.utc); CUT = NOW - timedelta(days=30)

# Top-20 winners
top20 = set()
with io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        try: e = json.loads(ln)
        except: continue
        ts_ms = e.get("ts");
        if not ts_ms: continue
        dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
        if dt < CUT: continue
        if e.get("label_top20") == 1:
            top20.add((dt.strftime("%Y-%m-%d"), e.get("symbol")))

# All blocks with ML proba — group by (date, sym)
ml_proba_pat = re.compile(r"ML proba ([\d.]+) outside")
sym_proba = defaultdict(list)  # (d, sym) -> [proba, ...]

with io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if "ML proba" not in ln: continue
        try: e = json.loads(ln)
        except: continue
        if e.get("event") != "blocked": continue
        ts = e.get("ts","")
        try: dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        except: continue
        if dt < CUT: continue
        sym = e.get("sym") or e.get("symbol") or ""
        if not sym: continue
        d = dt.strftime("%Y-%m-%d")
        reason = ""
        if isinstance(e.get("decision"), dict):
            reason = e["decision"].get("reason_code","") or ""
        if not reason:
            reason = e.get("reason_code") or e.get("reason","") or ""
        m = ml_proba_pat.search(reason)
        if m:
            sym_proba[(d, sym)].append(float(m.group(1)))

print("=== P5: ML signal-model blind-spots (last 30d) ===\n")

# 1) Top-20 winners with low ml_proba blocks
hits = []
for key in top20:
    probas = sym_proba.get(key, [])
    if not probas: continue
    minp = min(probas)
    hits.append((key[0], key[1], minp, max(probas), len(probas)))

hits.sort(key=lambda x: x[2])
print(f"Top-20 winners with ML-proba block events: {len(hits)}")
print(f"  {'date':<11} {'sym':<12} {'min_proba':>10} {'max_proba':>10} {'n_blocks':>9}")
for d, s, mn, mx, n in hits[:20]:
    print(f"  {d:<11} {s:<12} {mn:>9.3f} {mx:>9.3f} {n:>8d}")

# 2) Severity buckets
extreme = sum(1 for h in hits if h[2] < 0.10)
moderate = sum(1 for h in hits if 0.10 <= h[2] < 0.28)
mild     = sum(1 for h in hits if h[2] >= 0.28)
print(f"\nSeverity:")
print(f"  extreme blind-spot (proba < 0.10): {extreme}")
print(f"  moderate (0.10-0.28):              {moderate}")
print(f"  near-threshold (>= 0.28):          {mild}")

# 3) Affected unique symbols
syms = Counter(h[1] for h in hits)
print(f"\nUnique winners with ML-block: {len(syms)}")
print(f"Top recurring blind-spot syms:")
for s, c in syms.most_common(10):
    print(f"  {s:<12} {c} winner-days")

# 4) Total volume of low-proba blocks (any sym, not just winners)
low_proba_total = sum(1 for k, ps in sym_proba.items() for p in ps if p < 0.10)
print(f"\nTotal ML-proba blocks in 30d: {sum(len(v) for v in sym_proba.values())}")
print(f"  with proba < 0.10:           {low_proba_total} ({100*low_proba_total/max(1,sum(len(v) for v in sym_proba.values())):.1f}%)")
