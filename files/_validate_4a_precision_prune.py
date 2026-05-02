"""4A: precision-prune. Sweep filter combinations on bot's actual entries
to find a subset with significantly higher precision (top-20 hit rate)
without losing too much recall.

Filters tested:
  ranker_ev > 0
  ml_proba > 0.40
  ranker_top_gainer_prob > 0.30
  combined (all three)
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
NOW = datetime.now(timezone.utc); CUT = NOW - timedelta(days=30)

# top-20
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

ents = []
with io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if '"event"' not in ln: continue
        try: e = json.loads(ln)
        except: continue
        if e.get("event") != "entry": continue
        ts = e.get("ts","")
        try: dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        except: continue
        if dt < CUT: continue
        sym = e.get("sym") or e.get("symbol") or ""
        if not sym: continue
        d = dt.strftime("%Y-%m-%d")
        ents.append({
            "d": d, "sym": sym,
            "ml_proba": e.get("entry_ml_proba") or e.get("ml_proba"),
            "ev": e.get("ranker_ev"),
            "tg_prob": e.get("ranker_top_gainer_prob"),
            "is_top20": (d, sym) in top20,
        })

n = len(ents)
n_t = sum(1 for e in ents if e["is_top20"])
days = len(set(e["d"] for e in ents))
base_prec = 100*n_t/max(1, n)

print(f"=== 4A: precision-prune sweep ===\n")
print(f"Baseline: {n} entries, {n_t} on top-20, precision={base_prec:.1f}%, "
      f"{n/max(1,days):.1f}/day\n")

filters = [
    ("ranker_ev > 0",              lambda e: (e["ev"] or -999) > 0),
    ("ranker_ev > -0.20",          lambda e: (e["ev"] or -999) > -0.20),
    ("ml_proba > 0.40",            lambda e: (e["ml_proba"] or 0) > 0.40),
    ("ml_proba > 0.50",            lambda e: (e["ml_proba"] or 0) > 0.50),
    ("tg_prob > 0.30",             lambda e: (e["tg_prob"] or 0) > 0.30),
    ("tg_prob > 0.50",             lambda e: (e["tg_prob"] or 0) > 0.50),
    ("ev>-0.20 AND ml_proba>0.40", lambda e: (e["ev"] or -999) > -0.20 and (e["ml_proba"] or 0) > 0.40),
    ("ev>0 AND tg_prob>0.30",      lambda e: (e["ev"] or -999) > 0 and (e["tg_prob"] or 0) > 0.30),
    ("ev>-0.30 AND tg_prob>0.20",  lambda e: (e["ev"] or -999) > -0.30 and (e["tg_prob"] or 0) > 0.20),
]

print(f"  {'filter':<32} {'kept':>6} {'/day':>6} {'top20':>6} {'prec':>7} {'recall':>7}")
print(f"  {'-'*32} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*7}")
results = {}
for name, fn in filters:
    kept = [e for e in ents if fn(e)]
    if not kept: continue
    nk = len(kept); nkt = sum(1 for e in kept if e["is_top20"])
    prec = 100*nkt/nk
    recall = 100*nkt/max(1, n_t)
    results[name] = {"kept": nk, "top20": nkt, "precision_pct": prec,
                     "recall_pct": recall, "per_day": nk/max(1, days)}
    print(f"  {name:<32} {nk:>6d} {nk/max(1,days):>5.1f} {nkt:>6d} {prec:>6.1f}% {recall:>6.1f}%")

# Pareto-best: minimize per_day, maximize precision, recall ≥ 70%
print(f"\nViable Pareto candidates (precision >= 25%, recall >= 70%):")
for name, r in results.items():
    if r["precision_pct"] >= 25 and r["recall_pct"] >= 70:
        print(f"  {name}: prec={r['precision_pct']:.1f}%, recall={r['recall_pct']:.1f}%, "
              f"{r['per_day']:.1f}/day")

metric = {
    "metric": "4A_precision_prune",
    "baseline": {"n": n, "top20": n_t, "precision_pct": base_prec, "per_day": n/max(1,days)},
    "filters": results,
}
print("\nMETRIC_JSON:" + json.dumps(metric))
