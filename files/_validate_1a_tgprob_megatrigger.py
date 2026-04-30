"""1A: top_gainer_prob as mega-trigger.
For all entry events with ranker_top_gainer_prob, sort by proba.
At thresholds 0.30/0.50/0.70/0.85, compute:
  - precision = (top-20 winners) / (entries above threshold)
  - recall    = (top-20 caught) / (total top-20 winners that bot saw at all)
  - n_per_day if used as standalone trigger
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
NOW = datetime.now(timezone.utc); CUT = NOW - timedelta(days=30)

# top-20 winners
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

# Bot entries with ranker_top_gainer_prob
ents = []
field_present = 0; field_absent = 0
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
        # try multiple keys
        proba = e.get("ranker_top_gainer_prob") or e.get("top_gainer_prob")
        if proba is None and isinstance(e.get("ranker"), dict):
            proba = e["ranker"].get("top_gainer_prob")
        if proba is None: field_absent += 1; continue
        field_present += 1
        ents.append({"d": d, "sym": sym, "proba": float(proba),
                     "is_top20": (d, sym) in top20})

print(f"=== 1A: top_gainer_prob mega-trigger ===")
print(f"Entries with proba field: {field_present} / w/o: {field_absent}")
print(f"Total entries: {len(ents)}, on top-20: {sum(1 for e in ents if e['is_top20'])}\n")

# Total top-20 in window (lower bound for recall)
n_top20_window = len(top20)

# Days
days = set(e["d"] for e in ents)
n_days = len(days)

print(f"  {'thr':<6} {'n>=thr':>8} {'top20_in':>9} {'precision':>10} {'recall_v':>9} {'/day':>7}")
print(f"  ----- -------- --------- ---------- --------- -------")
results = {}
for thr in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    above = [e for e in ents if e["proba"] >= thr]
    if not above: continue
    n = len(above)
    n_t = sum(1 for e in above if e["is_top20"])
    prec = 100*n_t/n
    # recall = top-20 winner-days that have at least one entry above threshold
    syms_above = {(e["d"], e["sym"]) for e in above if e["is_top20"]}
    rec = 100*len(syms_above)/max(1, n_top20_window)
    print(f"  {thr:<6.2f} {n:>8d} {n_t:>9d} {prec:>9.1f}% {rec:>8.1f}% {n/n_days:>6.1f}")
    results[thr] = {"n": n, "top20": n_t, "precision_pct": prec,
                    "recall_pct": rec, "per_day": n/n_days}

# Acceptance for 1A: precision >= 60% at proba >= 0.7
target_thr = 0.70
target_prec = 60.0
viable = (results.get(target_thr, {}).get("precision_pct", 0) >= target_prec)
print(f"\n1A acceptance (precision >= {target_prec}% @ proba >= {target_thr}): "
      f"{'PASS' if viable else 'FAIL'}")

metric = {
    "metric": "1A_tgprob_megatrigger",
    "thresholds": results,
    "acceptance_pass": viable,
}
print("\nMETRIC_JSON:" + json.dumps(metric))
