"""
Backtest: find optimal slope_pct_min / daily_range_min gates per mode.

Two hypotheses:
  H1 - slope_pct_min: TP have significantly higher slope (alignment +55%, impulse_speed +71%)
  H2 - daily_range_min: TP happen on more volatile days (+39% for alignment, +92% for impulse_speed)

Sweep over 60-day window to avoid overfitting to recent quiet period.
Report: precision gain vs recall cost per threshold, Pareto-optimal combos.
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
FILES = Path(__file__).resolve().parent

NOW = datetime.now(timezone.utc)
CUTOFF = NOW - timedelta(days=60)

# ── Load top-20 labels ───────────────────────────────────────────────────────
top20_by_date: dict[str, set] = defaultdict(set)
with io.open(FILES / "top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        try: e = json.loads(ln)
        except: continue
        if not e.get("label_top20"): continue
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

# ── Load entries ─────────────────────────────────────────────────────────────
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
        if dt < CUTOFF: continue
        e["_dt"] = dt
        e["_date"] = dt.strftime("%Y-%m-%d")
        e["_tp"] = e.get("sym", "") in top20_by_date.get(e["_date"], set())
        entries.append(e)

def safe_float(v, default=None):
    try: return float(v)
    except: return default

# Attach float fields
for e in entries:
    e["_slope"] = safe_float(e.get("slope_pct"))
    e["_drange"] = safe_float(e.get("daily_range"))
    e["_adx"]   = safe_float(e.get("adx"))
    e["_mlp"]   = safe_float(e.get("ml_proba"))

# ── Helper: metrics for a subset ─────────────────────────────────────────────
def metrics(subset):
    n = len(subset)
    if n == 0: return 0, 0, 0, 0
    tp = sum(1 for e in subset if e["_tp"])
    prec = tp / n * 100
    return n, tp, prec, 0

# ── Target modes for tightening ──────────────────────────────────────────────
TARGET_MODES = {
    "alignment/15m",
    "alignment/1h",
    "trend/15m",
    "trend/1h",
    "impulse_speed/1h",
    "breakout/15m",
    "retest/1h",
    "strong_trend/15m",
    "strong_trend/1h",
}
KEEP_MODES = {
    "impulse_speed/15m",
    "impulse/15m",
    "impulse/1h",
}

def mode_key(e):
    return f"{e.get('mode','?')}/{e.get('tf','?')}"

# Separate entries by mode category
target_ent  = [e for e in entries if mode_key(e) in TARGET_MODES]
keep_ent    = [e for e in entries if mode_key(e) in KEEP_MODES]
all_n       = len(entries)
all_tp      = sum(1 for e in entries if e["_tp"])

print(f"60-day window: {len(entries)} entries, {all_tp} TP, precision={all_tp/all_n*100:.1f}%")
print(f"  Target modes: {len(target_ent)} entries")
print(f"  Keep modes:   {len(keep_ent)} entries")

# ── H1: Sweep slope_pct_min per mode ────────────────────────────────────────
print("\n\n=== H1: slope_pct_min sweep (per mode, 60d) ===")
print(f"{'Mode/TF':28s}  {'baseline':>20s}  | thres  {'n':>5}  {'prec':>6}  {'rec_vs_all':>10}  note")
print("─" * 95)

SLOPE_THRESHOLDS = [0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

for mk in sorted(TARGET_MODES):
    sub = [e for e in target_ent if mode_key(e) == mk]
    if len(sub) < 10: continue
    base_n  = len(sub)
    base_tp = sum(1 for e in sub if e["_tp"])
    base_pr = base_tp / base_n * 100

    best = None
    for thr in SLOPE_THRESHOLDS:
        fil = [e for e in sub if e.get("_slope") is not None and e["_slope"] >= thr]
        if not fil: continue
        n  = len(fil)
        tp = sum(1 for e in fil if e["_tp"])
        pr = tp / n * 100
        bl = tp / all_tp * 100 if all_tp else 0
        if thr == 0.0:
            print(f"  {mk:28s}  baseline n={base_n:>4d} prec={base_pr:>5.1f}%  | {thr:>5.2f}  {n:>5}  {pr:>5.1f}%  {bl:>8.1f}%  --")
        else:
            gain = pr - base_pr
            cost_n = base_n - n
            note = f"+{gain:.1f}% prec, -{cost_n} entries"
            print(f"  {'':28s}  {' ':20s}  | {thr:>5.2f}  {n:>5}  {pr:>5.1f}%  {bl:>8.1f}%  {note}")

# ── H2: Sweep daily_range_min ────────────────────────────────────────────────
print("\n\n=== H2: daily_range_min sweep (per mode, 60d) ===")
print(f"{'Mode/TF':28s}  {'baseline':>20s}  | thres  {'n':>5}  {'prec':>6}  {'rec_vs_all':>10}  note")
print("─" * 95)

RANGE_THRESHOLDS = [0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]

for mk in sorted(TARGET_MODES):
    sub = [e for e in target_ent if mode_key(e) == mk]
    if len(sub) < 10: continue
    base_n  = len(sub)
    base_tp = sum(1 for e in sub if e["_tp"])
    base_pr = base_tp / base_n * 100

    for thr in RANGE_THRESHOLDS:
        fil = [e for e in sub if e.get("_drange") is not None and e["_drange"] >= thr]
        if not fil: continue
        n  = len(fil)
        tp = sum(1 for e in fil if e["_tp"])
        pr = tp / n * 100
        bl = tp / all_tp * 100 if all_tp else 0
        if thr == 0.0:
            print(f"  {mk:28s}  baseline n={base_n:>4d} prec={base_pr:>5.1f}%  | {thr:>4.1f}  {n:>5}  {pr:>5.1f}%  {bl:>8.1f}%  --")
        else:
            gain = pr - base_pr
            cost_n = base_n - n
            note = f"+{gain:.1f}% prec, -{cost_n} entries"
            print(f"  {'':28s}  {' ':20s}  | {thr:>4.1f}  {n:>5}  {pr:>5.1f}%  {bl:>8.1f}%  {note}")

# ── H3: Combined slope + range (best combo) ──────────────────────────────────
print("\n\n=== H3: Combined sweep slope_pct >= X AND daily_range >= Y (all target modes) ===")
print(f"  {'slope':>7}  {'range':>7}  {'n':>5}  {'TP':>4}  {'prec':>6}  {'rec%':>6}  delta_prec")
print("  " + "-" * 65)

# Find best combo for alignment/15m (largest problem mode)
base_all_prec = all_tp / all_n * 100
sub_al15 = [e for e in entries if mode_key(e) == "alignment/15m"]
sub_other_good = [e for e in entries if mode_key(e) not in TARGET_MODES]

for slope_thr in [0.0, 0.30, 0.40, 0.50]:
    for range_thr in [0.0, 4.0, 5.0, 6.0]:
        # Apply filter to all target modes
        def passes(e):
            mk = mode_key(e)
            if mk not in TARGET_MODES: return True
            sl = e.get("_slope")
            dr = e.get("_drange")
            if sl is not None and sl < slope_thr: return False
            if dr is not None and dr < range_thr: return False
            return True

        filtered = [e for e in entries if passes(e)]
        if not filtered: continue
        n  = len(filtered)
        tp = sum(1 for e in filtered if e["_tp"])
        prec = tp / n * 100
        rec  = tp / all_tp * 100 if all_tp else 0
        delta = prec - base_all_prec
        print(f"  slope>={slope_thr:.2f}  range>={range_thr:.1f}  "
              f"n={n:>5}  TP={tp:>4}  prec={prec:>5.1f}%  rec={rec:>5.1f}%  "
              f"delta={delta:+.1f}%")

# ── Specific concern: impulse_speed/1h (declining) ───────────────────────────
print("\n\n=== impulse_speed/1h: is it worth keeping? ===")
is1h = [e for e in entries if mode_key(e) == "impulse_speed/1h"]
if is1h:
    # By month
    from collections import Counter
    by_month = defaultdict(lambda: [0, 0])
    for e in is1h:
        m = e["_date"][:7]
        by_month[m][1] += 1
        if e["_tp"]: by_month[m][0] += 1
    for m, (tp, n) in sorted(by_month.items()):
        print(f"  {m}: n={n:>3}  TP={tp:>3}  prec={tp/n*100:.1f}%")
    print()
    # Range/slope stats
    tp_is1h = [e for e in is1h if e["_tp"]]
    fp_is1h = [e for e in is1h if not e["_tp"]]
    for key in ["_slope", "_drange", "_adx"]:
        tp_v = [e[key] for e in tp_is1h if e[key] is not None]
        fp_v = [e[key] for e in fp_is1h if e[key] is not None]
        avg_tp = sum(tp_v)/len(tp_v) if tp_v else 0
        avg_fp = sum(fp_v)/len(fp_v) if fp_v else 0
        print(f"  {key:12s}  TP avg={avg_tp:.3f}  FP avg={avg_fp:.3f}")

# ── Final summary: recommended thresholds ───────────────────────────────────
print("\n\n=== Summary: mode-specific observations ===")
for mk in sorted(TARGET_MODES | KEEP_MODES):
    sub = [e for e in entries if mode_key(e) == mk]
    if not sub: continue
    n = len(sub); tp = sum(1 for e in sub if e["_tp"])
    pr = tp / n * 100

    # Last 14d
    cut14 = NOW - timedelta(days=14)
    sub14 = [e for e in sub if e["_dt"] >= cut14]
    n14 = len(sub14); tp14 = sum(1 for e in sub14 if e["_tp"])
    pr14 = tp14/n14*100 if n14 else 0

    print(f"  {mk:28s}  60d: n={n:>3} prec={pr:>5.1f}%  14d: n={n14:>3} prec={pr14:>5.1f}%")
