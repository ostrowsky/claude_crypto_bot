"""
Diagnose per-mode precision over multiple time windows.
Goal: find which modes/tf combos are consistently bad and what
features discriminate TP vs FP within each mode.
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
FILES = Path(__file__).resolve().parent

NOW = datetime.now(timezone.utc)

# ── Load top_gainer_dataset labels ──────────────────────────────────────────
top20_by_date: dict[str, set] = defaultdict(set)
with io.open(FILES / "top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        try: e = json.loads(ln)
        except: continue
        l20 = e.get("label_top20", 0)
        if not l20: continue
        ts = e.get("ts") or e.get("date", "")
        if isinstance(ts, (int, float)):
            d = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        elif isinstance(ts, str):
            d = ts[:10]
        else:
            continue
        sym = e.get("symbol") or e.get("sym", "")
        if sym:
            top20_by_date[d].add(sym)

# ── Load bot entries ─────────────────────────────────────────────────────────
entries = []
with io.open(FILES / "bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        try: e = json.loads(ln)
        except: continue
        if e.get("event") != "entry": continue
        ts = e.get("ts", "")
        if not ts: continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except:
            continue
        e["_dt"] = dt
        e["_date"] = dt.strftime("%Y-%m-%d")
        entries.append(e)

def is_tp(e):
    sym  = e.get("sym", "")
    date = e["_date"]
    winners = top20_by_date.get(date, set())
    return sym in winners

def precision_table(subset, label):
    by_mode = defaultdict(lambda: {"n": 0, "tp": 0})
    for e in subset:
        mode = e.get("mode", "?")
        tf   = e.get("tf", "?")
        key  = f"{mode}/{tf}"
        by_mode[key]["n"] += 1
        if is_tp(e):
            by_mode[key]["tp"] += 1
    # Sort by n desc
    rows = sorted(by_mode.items(), key=lambda x: -x[1]["n"])
    total_n  = sum(v["n"] for v in by_mode.values())
    total_tp = sum(v["tp"] for v in by_mode.values())
    prec = total_tp / total_n * 100 if total_n else 0
    print(f"\n=== {label}  (n={total_n}  TP={total_tp}  precision={prec:.1f}%) ===")
    print(f"  {'Mode/TF':30s}  {'n':>5}  {'TP':>4}  {'Prec':>6}")
    print("  " + "-" * 55)
    for key, v in rows:
        n, tp = v["n"], v["tp"]
        p = tp / n * 100 if n else 0
        flag = "  ❌" if (n >= 10 and p < 5) else ("  ✅" if p >= 20 else "")
        print(f"  {key:30s}  {n:>5}  {tp:>4}  {p:>5.1f}%{flag}")

cutoffs = {
    "last 7d":  NOW - timedelta(days=7),
    "last 14d": NOW - timedelta(days=14),
    "last 30d": NOW - timedelta(days=30),
    "last 60d": NOW - timedelta(days=60),
}
for label, cutoff in cutoffs.items():
    subset = [e for e in entries if e["_dt"] >= cutoff]
    precision_table(subset, label)

# ── Deep dive: alignment 15m TP vs FP features ───────────────────────────────
print("\n\n=== Deep dive: alignment/15m  (last 30d) — TP vs FP indicators ===")
cutoff30 = NOW - timedelta(days=30)
al15 = [e for e in entries if e.get("mode") == "alignment" and e.get("tf") == "15m" and e["_dt"] >= cutoff30]
tp_list = [e for e in al15 if is_tp(e)]
fp_list = [e for e in al15 if not is_tp(e)]
print(f"  alignment/15m n={len(al15)}  TP={len(tp_list)}  FP={len(fp_list)}")

def stat(lst, key):
    vals = [float(v) for e in lst for v in [e.get(key)] if v is not None]
    if not vals: return "N/A"
    vals.sort()
    n = len(vals)
    return f"avg={sum(vals)/n:+.3f}  p25={vals[n//4]:+.3f}  p50={vals[n//2]:+.3f}  p75={vals[3*n//4]:+.3f}"

for key in ["adx", "rsi", "slope_pct", "vol_x", "ml_proba", "daily_range", "macd_hist"]:
    tp_s = stat(tp_list, key)
    fp_s = stat(fp_list, key)
    print(f"  {key:12s}  TP: {tp_s}")
    print(f"  {' ':12s}  FP: {fp_s}")
    print()

# ── Deep dive: trend/1h ──────────────────────────────────────────────────────
print("\n=== Deep dive: trend/1h  (last 30d) ===")
tr1h = [e for e in entries if e.get("mode") == "trend" and e.get("tf") == "1h" and e["_dt"] >= cutoff30]
tp1h = [e for e in tr1h if is_tp(e)]
fp1h = [e for e in tr1h if not is_tp(e)]
print(f"  trend/1h n={len(tr1h)}  TP={len(tp1h)}  FP={len(fp1h)}")
for key in ["adx", "rsi", "slope_pct", "vol_x", "ml_proba", "daily_range"]:
    tp_s = stat(tp1h, key)
    fp_s = stat(fp1h, key)
    print(f"  {key:12s}  TP: {tp_s}")
    print(f"  {' ':12s}  FP: {fp_s}")
    print()

# ── impulse_speed/15m for contrast ──────────────────────────────────────────
print("\n=== Contrast: impulse_speed/15m  (last 30d) ===")
is15 = [e for e in entries if e.get("mode") == "impulse_speed" and e.get("tf") == "15m" and e["_dt"] >= cutoff30]
tp_is = [e for e in is15 if is_tp(e)]
fp_is = [e for e in is15 if not is_tp(e)]
print(f"  impulse_speed/15m n={len(is15)}  TP={len(tp_is)}  FP={len(fp_is)}")
for key in ["adx", "rsi", "slope_pct", "vol_x", "ml_proba", "daily_range"]:
    tp_s = stat(tp_is, key)
    fp_s = stat(fp_is, key)
    print(f"  {key:12s}  TP: {tp_s}")
    print(f"  {' ':12s}  FP: {fp_s}")
    print()

# ── ml_proba distribution by mode ───────────────────────────────────────────
print("\n=== ml_proba distribution per mode/tf (last 14d) ===")
cutoff14 = NOW - timedelta(days=14)
by_mode14 = defaultdict(list)
for e in entries:
    if e["_dt"] < cutoff14: continue
    key = f"{e.get('mode','?')}/{e.get('tf','?')}"
    ml = e.get("ml_proba")
    if ml is not None:
        by_mode14[key].append((float(ml), is_tp(e)))

for key, items in sorted(by_mode14.items(), key=lambda x: -len(x[1])):
    probas = [p for p, _ in items]
    tp_probas = [p for p, tp in items if tp]
    fp_probas = [p for p, tp in items if not tp]
    n = len(probas)
    avg_all = sum(probas)/n
    avg_tp = sum(tp_probas)/len(tp_probas) if tp_probas else 0
    avg_fp = sum(fp_probas)/len(fp_probas) if fp_probas else 0
    print(f"  {key:30s}  n={n:>4}  avg_proba={avg_all:.3f}  "
          f"TP_proba={avg_tp:.3f}({len(tp_probas)})  FP_proba={avg_fp:.3f}({len(fp_probas)})")
