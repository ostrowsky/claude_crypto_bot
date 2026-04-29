"""North-star: EarlyCapture@top20 = coverage * capture_ratio * time_lead_score
Aggregator across the 14 d window. Per top-20 winner-day:
  coverage_flag    = 1 if entered, else 0
  capture_ratio    = clamp(realized_pnl / eod_return_pct, 0, 1)
  time_lead_score  = 1 - (entry_hour_UTC / 24)   (early UTC = higher)
EarlyCapture = mean(coverage * capture * time_lead) across all winner-days.
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
NOW = datetime.now(timezone.utc); CUT = NOW - timedelta(days=14)

# Top-20 + eod_return
top20 = set(); eod_ret = {}
with io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        try: e = json.loads(ln)
        except: continue
        ts_ms = e.get("ts");
        if not ts_ms: continue
        dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
        if dt < CUT: continue
        sym = e.get("symbol"); d = dt.strftime("%Y-%m-%d")
        if e.get("label_top20") == 1: top20.add((d, sym))
        eod_ret[(d, sym)] = e.get("eod_return_pct")

# First entry + paired trades (for pnl)
first_entry = {}; pnl_pairs = {}; entries = {}
with io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if '"event"' not in ln: continue
        try: e = json.loads(ln)
        except: continue
        ev = e.get("event","")
        if ev not in ("entry","exit"): continue
        ts = e.get("ts","")
        try: dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        except: continue
        if dt < CUT: continue
        sym = e.get("sym") or e.get("symbol") or ""
        if not sym: continue
        d = dt.strftime("%Y-%m-%d")
        if ev == "entry":
            ep = float(e.get("price") or e.get("entry_price") or 0)
            entries[sym] = (dt, d, ep)
            prev = first_entry.get((d, sym))
            if prev is None or dt < prev[0]:
                first_entry[(d, sym)] = (dt, ep)
        else:
            ent = entries.pop(sym, None)
            if not ent: continue
            ex_p = float(e.get("exit_price") or e.get("price") or 0)
            if ent[2] <= 0 or ex_p <= 0: continue
            pnl = (ex_p - ent[2]) / ent[2] * 100
            pnl_pairs[(ent[1], sym)] = pnl

# Compute per-winner score
ec_scores = []
breakdown = []
for key in top20:
    d, sym = key
    ent = first_entry.get(key)
    coverage = 1.0 if ent else 0.0
    if ent:
        edt, ep = ent
        time_lead = 1.0 - (edt.hour / 24.0)
        pnl = pnl_pairs.get(key)
        eod = eod_ret.get(key)
        if pnl is not None and eod is not None:
            eod_p = float(eod)
            if abs(eod_p) <= 5: eod_p *= 100
            if abs(eod_p) >= 1.0:
                cap = max(0.0, min(1.0, pnl / eod_p))
            else:
                cap = 0.0
        else:
            cap = 0.0  # no exit yet — assume zero capture
    else:
        time_lead = 0.0; cap = 0.0
    score = coverage * cap * time_lead
    ec_scores.append(score)
    breakdown.append({"d": d, "sym": sym, "cov": coverage, "cap": cap,
                      "lead": time_lead, "score": score})

n = len(ec_scores)
mean_ec = sum(ec_scores)/max(1, n)

print("=== NORTH-STAR: EarlyCapture@top20 (14d) ===\n")
print(f"Top-20 winner-days: {n}")
print(f"EarlyCapture@top20 = {mean_ec:.3f}\n")

# Decompose
mean_cov = sum(b["cov"] for b in breakdown)/max(1,n)
mean_cap = sum(b["cap"] for b in breakdown if b["cov"] > 0) / max(1, sum(1 for b in breakdown if b["cov"] > 0))
mean_lead = sum(b["lead"] for b in breakdown if b["cov"] > 0) / max(1, sum(1 for b in breakdown if b["cov"] > 0))
print(f"  coverage     mean = {mean_cov:.3f}")
print(f"  capture (entered only) mean = {mean_cap:.3f}")
print(f"  time_lead (entered only) mean = {mean_lead:.3f}")

# Top contributors / worst
breakdown.sort(key=lambda x: -x["score"])
print(f"\nTop-5 winners (highest EC):")
for b in breakdown[:5]:
    print(f"  {b['d']} {b['sym']:<10} score={b['score']:.3f}  (cov={b['cov']:.0f}, cap={b['cap']:.2f}, lead={b['lead']:.2f})")
print(f"\nBottom-5 (zero-score, biggest opportunity):")
for b in [x for x in breakdown if x["score"] == 0][:5]:
    print(f"  {b['d']} {b['sym']:<10} cov={b['cov']:.0f}, cap={b['cap']:.2f}, lead={b['lead']:.2f}")

metric = {
    "metric": "NS_EarlyCapture_top20",
    "n": n,
    "early_capture": mean_ec,
    "decomp_coverage": mean_cov,
    "decomp_capture_mean": mean_cap,
    "decomp_time_lead_mean": mean_lead,
}
print("\nMETRIC_JSON:" + json.dumps(metric))
