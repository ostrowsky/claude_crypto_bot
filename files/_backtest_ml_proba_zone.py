"""
Backtest: optimal ML_GENERAL_HARD_BLOCK_MIN threshold for ml_proba_zone gate.

Data source: critic_dataset.jsonl
Goal: find the best floor threshold (currently 0.28) to avoid blocking top-gainer candidates.
"""

import json
import sys
from collections import defaultdict
import math

DATASET = "files/critic_dataset.jsonl" if len(sys.argv) < 2 else sys.argv[1]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
rows_take = []          # action == 'take'
rows_blocked_zone = []  # action == 'blocked', reason_code == 'ml_proba_zone'
rows_blocked_other = [] # action == 'blocked', other reason_codes

skipped_no_proba = 0
skipped_no_ret = 0
total = 0

print(f"Loading {DATASET} ...", flush=True)

with open(DATASET, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        total += 1
        d = row.get("decision", {})
        lbl = row.get("labels", {})

        proba = d.get("ml_proba")
        ret5 = lbl.get("ret_5")
        action = d.get("action", "")
        reason_code = d.get("reason_code", "")

        if proba is None:
            skipped_no_proba += 1
            continue
        if ret5 is None:
            skipped_no_ret += 1
            continue

        rec = {
            "proba": float(proba),
            "ret5": float(ret5),
            "win": 1 if float(ret5) > 0 else 0,
            "action": action,
            "reason_code": reason_code,
            "tf": row.get("tf", "?"),
            "signal_type": row.get("signal_type", "?"),
            "sym": row.get("sym", "?"),
        }

        if action == "take":
            rows_take.append(rec)
        elif action == "blocked" and reason_code == "ml_proba_zone":
            rows_blocked_zone.append(rec)
        else:
            rows_blocked_other.append(rec)

print(f"Total rows: {total:,}")
print(f"  Skipped (no ml_proba): {skipped_no_proba:,}")
print(f"  Skipped (no ret_5):    {skipped_no_ret:,}")
print(f"  rows_take:             {len(rows_take):,}")
print(f"  rows_blocked_zone:     {len(rows_blocked_zone):,}")
print(f"  rows_blocked_other:    {len(rows_blocked_other):,}")
print()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def stats(rows):
    if not rows:
        return dict(n=0, avg_r5=None, win_pct=None, sharpe=None)
    n = len(rows)
    r5 = [r["ret5"] for r in rows]
    avg = sum(r5) / n
    win = sum(r["win"] for r in rows) / n * 100
    if n > 1:
        variance = sum((x - avg) ** 2 for x in r5) / (n - 1)
        sd = math.sqrt(variance)
        sharpe = (avg / sd * math.sqrt(n)) if sd > 1e-9 else None
    else:
        sharpe = None
    return dict(n=n, avg_r5=avg, win_pct=win, sharpe=sharpe)


def fmt(s):
    if s["n"] == 0:
        return f"{'n=0':>8}"
    sh = f"{s['sharpe']:.2f}" if s['sharpe'] is not None else "  n/a"
    return (f"n={s['n']:>5}  avg_r5={s['avg_r5']:+.3f}%  "
            f"win%={s['win_pct']:>5.1f}  sharpe={sh:>6}")


# ---------------------------------------------------------------------------
# Section 1: Overall baseline (take decisions)
# ---------------------------------------------------------------------------
print("=" * 75)
print("SECTION 1 — BASELINE: 'take' decisions")
print("=" * 75)
s = stats(rows_take)
print(f"  All takes:          {fmt(s)}")
takes_low = [r for r in rows_take if r["proba"] < 0.28]
takes_high = [r for r in rows_take if r["proba"] >= 0.28]
print(f"  take ml_proba<0.28: {fmt(stats(takes_low))}")
print(f"  take ml_proba>=0.28:{fmt(stats(takes_high))}")
print()


# ---------------------------------------------------------------------------
# Section 2: Blocked by ml_proba_zone — bucket breakdown
# ---------------------------------------------------------------------------
print("=" * 75)
print("SECTION 2 — ml_proba_zone BLOCKED rows: bucketed by ml_proba")
print("=" * 75)

BUCKETS = [
    (0.00, 0.10),
    (0.10, 0.15),
    (0.15, 0.20),
    (0.20, 0.22),
    (0.22, 0.25),
    (0.25, 0.28),
    (0.28, 0.35),
    (0.35, 0.50),
    (0.50, 0.55),
    (0.55, 1.01),  # "zone" (should pass if inside)
]

header = f"  {'Bucket':<18} {'n':>6}  {'avg_r5':>8}  {'win%':>6}  {'sharpe':>7}"
print(header)
print("  " + "-" * 56)
for lo, hi in BUCKETS:
    bucket_rows = [r for r in rows_blocked_zone if lo <= r["proba"] < hi]
    s = stats(bucket_rows)
    if s["n"] == 0:
        print(f"  [{lo:.2f}-{hi:.2f}){'':>6}  {'(empty)':>6}")
        continue
    sh = f"{s['sharpe']:+.2f}" if s["sharpe"] is not None else "   n/a"
    print(f"  [{lo:.2f}-{hi:.2f}){'':>6}  "
          f"{s['n']:>6}  {s['avg_r5']:>+8.3f}%  {s['win_pct']:>5.1f}%  {sh:>7}")
print()


# ---------------------------------------------------------------------------
# Section 3: Threshold sweep — what happens if we lower the floor?
# ---------------------------------------------------------------------------
print("=" * 75)
print("SECTION 3 — THRESHOLD SWEEP for ML_GENERAL_HARD_BLOCK_MIN")
print("  (rows BELOW threshold are blocked; rows ABOVE threshold pass through)")
print("=" * 75)

THRESHOLDS = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35]

# Baseline: current threshold 0.28
baseline_blocked = [r for r in rows_blocked_zone if r["proba"] < 0.28]
baseline_pass = [r for r in rows_blocked_zone if r["proba"] >= 0.28]

print(f"\n  Baseline threshold = 0.28")
print(f"    blocked (proba<0.28):    {fmt(stats(baseline_blocked))}")
print(f"    would-pass (proba>=0.28):{fmt(stats(baseline_pass))}")
print()

print(f"  {'Threshold':<12} {'n_blocked':>10} {'avg_r5_blk':>12} {'n_released':>11} "
      f"{'avg_r5_rel':>12} {'delta_vs_baseline':>18}")
print("  " + "-" * 80)

for thr in THRESHOLDS:
    newly_blocked = [r for r in rows_blocked_zone if r["proba"] < thr]
    released = [r for r in rows_blocked_zone if r["proba"] >= thr]  # would now pass
    sb = stats(newly_blocked)
    sr = stats(released)
    delta = ""
    if sr["avg_r5"] is not None and stats(baseline_pass)["avg_r5"] is not None:
        d = sr["avg_r5"] - stats(baseline_pass)["avg_r5"]
        delta = f"{d:+.3f}%"
    tag = "  <-- CURRENT" if thr == 0.28 else ""
    print(f"  thr={thr:<6.2f}  "
          f"blocked: n={sb['n']:>5} avg={sb['avg_r5']:>+7.3f}%  "
          f"released: n={sr['n']:>5} avg={sr['avg_r5']:>+7.3f}%  "
          f"delta={delta:>8}{tag}")
print()


# ---------------------------------------------------------------------------
# Section 4: Breakdown by (tf, signal_type) for low-proba zone-blocked rows
# ---------------------------------------------------------------------------
print("=" * 75)
print("SECTION 4 — BREAKDOWN BY (tf, signal_type) for proba < 0.35 zone-blocked")
print("=" * 75)

focus = [r for r in rows_blocked_zone if r["proba"] < 0.35]
combos = defaultdict(list)
for r in focus:
    combos[(r["tf"], r["signal_type"])].append(r)

sorted_combos = sorted(combos.items(), key=lambda x: -len(x[1]))
print(f"\n  {'tf':<6} {'signal_type':<20} {'n':>6}  {'avg_r5':>8}  {'win%':>6}  {'sharpe':>7}")
print("  " + "-" * 60)
for (tf, st), rows in sorted_combos[:20]:
    s = stats(rows)
    sh = f"{s['sharpe']:+.2f}" if s["sharpe"] is not None else "   n/a"
    print(f"  {tf:<6} {st:<20} {s['n']:>6}  {s['avg_r5']:>+8.3f}%  "
          f"{s['win_pct']:>5.1f}%  {sh:>7}")
print()


# ---------------------------------------------------------------------------
# Section 5: 'take' decisions with low ml_proba (slipped through)
# ---------------------------------------------------------------------------
print("=" * 75)
print("SECTION 5 — 'take' decisions by ml_proba range (slippage analysis)")
print("=" * 75)

TAKE_BUCKETS = [
    (0.00, 0.15),
    (0.15, 0.20),
    (0.20, 0.25),
    (0.25, 0.28),
    (0.28, 0.35),
    (0.35, 0.50),
    (0.50, 0.65),
    (0.65, 0.80),
    (0.80, 1.01),
]

print(f"  {'Range':<18} {'n':>6}  {'avg_r5':>8}  {'win%':>6}  {'sharpe':>7}")
print("  " + "-" * 56)
for lo, hi in TAKE_BUCKETS:
    bucket_rows = [r for r in rows_take if lo <= r["proba"] < hi]
    s = stats(bucket_rows)
    if s["n"] == 0:
        print(f"  [{lo:.2f}-{hi:.2f}){'':>6}  {'(empty)':>6}")
        continue
    sh = f"{s['sharpe']:+.2f}" if s["sharpe"] is not None else "   n/a"
    tag = "  <-- current floor" if hi == 0.28 else ""
    print(f"  [{lo:.2f}-{hi:.2f}){'':>6}  "
          f"{s['n']:>6}  {s['avg_r5']:>+8.3f}%  {s['win_pct']:>5.1f}%  {sh:>7}{tag}")
print()


# ---------------------------------------------------------------------------
# Section 6: Pareto summary — recommended threshold
# ---------------------------------------------------------------------------
print("=" * 75)
print("SECTION 6 — PARETO SUMMARY & RECOMMENDATION")
print("=" * 75)

print("\n  Pareto table: for each candidate threshold, how many extra signals")
print("  would be released vs baseline (0.28), and their expected quality:\n")

print(f"  {'thr':<6} {'extra_n':>8} {'extra_avg_r5':>14} {'extra_win%':>11} "
      f"{'extra_sharpe':>13} {'verdict':>20}")
print("  " + "-" * 80)

baseline_blocked_set = set(id(r) for r in rows_blocked_zone if r["proba"] < 0.28)
baseline_stats = stats(rows_take)  # compare against take baseline

for thr in THRESHOLDS:
    # rows that would be released (between thr and 0.28 = no longer blocked)
    if thr >= 0.28:
        extra = [r for r in rows_blocked_zone if 0.28 <= r["proba"] < thr]
        direction = "tighter"
    else:
        extra = [r for r in rows_blocked_zone if thr <= r["proba"] < 0.28]
        direction = "looser"

    s = stats(extra)
    if s["n"] == 0:
        verdict = "no change" if thr == 0.28 else "no extra signals"
        print(f"  {thr:<6.2f} {'0':>8} {'—':>14} {'—':>11} {'—':>13} {verdict:>20}")
        continue

    sh = f"{s['sharpe']:+.2f}" if s["sharpe"] is not None else "n/a"

    if thr == 0.28:
        verdict = "CURRENT"
    elif direction == "looser":
        if s["avg_r5"] is not None and s["avg_r5"] > baseline_stats["avg_r5"]:
            verdict = "RECOMMEND (above baseline)"
        elif s["avg_r5"] is not None and s["avg_r5"] > -0.05:
            verdict = "consider (neutral)"
        else:
            verdict = "avoid (below baseline)"
    else:
        verdict = "tighter (loses signals)"

    print(f"  {thr:<6.2f} {s['n']:>8} {s['avg_r5']:>+14.3f}% {s['win_pct']:>10.1f}% "
          f"{sh:>13} {verdict:>20}")

print()

# Find best threshold based on avg_r5 of released signals vs take baseline
take_avg = stats(rows_take)["avg_r5"] or 0.0
print(f"  'take' baseline avg_r5: {take_avg:+.3f}%")
print()

best_thr = None
best_score = -999
for thr in [t for t in THRESHOLDS if t < 0.28]:
    extra = [r for r in rows_blocked_zone if thr <= r["proba"] < 0.28]
    s = stats(extra)
    if s["n"] >= 5 and s["avg_r5"] is not None and s["avg_r5"] > best_score:
        best_score = s["avg_r5"]
        best_thr = thr

if best_thr is not None:
    extra = [r for r in rows_blocked_zone if best_thr <= r["proba"] < 0.28]
    s = stats(extra)
    print(f"  BEST lower threshold by avg_r5: {best_thr}")
    print(f"    Extra signals released: n={s['n']}, avg_r5={s['avg_r5']:+.3f}%, "
          f"win%={s['win_pct']:.1f}%")
    if s["avg_r5"] > take_avg:
        print(f"    These signals OUTPERFORM the take baseline ({take_avg:+.3f}%) => safe to lower")
    else:
        print(f"    These signals are BELOW take baseline ({take_avg:+.3f}%) => lower with caution")
else:
    print("  No threshold produced enough extra signals with positive expected value.")

print()
print("Done.")
