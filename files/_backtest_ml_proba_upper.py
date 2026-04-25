"""
Backtest: investigate ml_proba upper-bound over-blocking and proba/ret_5 correlation.

Key question: are high-confidence signals (ml_proba > 0.50) being blocked somewhere,
and does higher proba actually predict better outcomes?

Data source: critic_dataset.jsonl
"""

import json
import sys
import math
from collections import defaultdict

DATASET = "files/critic_dataset.jsonl" if len(sys.argv) < 2 else sys.argv[1]

# ---------------------------------------------------------------------------
# Load ALL rows with ml_proba and ret_5
# ---------------------------------------------------------------------------
all_rows = []
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

        all_rows.append({
            "proba": float(proba),
            "ret5": float(ret5),
            "win": 1 if float(ret5) > 0 else 0,
            "action": action,
            "reason_code": reason_code,
            "tf": row.get("tf", "?"),
            "signal_type": row.get("signal_type", d.get("signal_type", "?")),
            "sym": row.get("sym", "?"),
        })

rows_take    = [r for r in all_rows if r["action"] == "take"]
rows_blocked = [r for r in all_rows if r["action"] == "blocked"]

print(f"Total rows in file:       {total:,}")
print(f"  Skipped (no ml_proba):  {skipped_no_proba:,}")
print(f"  Skipped (no ret_5):     {skipped_no_ret:,}")
print(f"  Usable rows total:      {len(all_rows):,}")
print(f"    take:                 {len(rows_take):,}")
print(f"    blocked:              {len(rows_blocked):,}")
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


def fmt(s, label=""):
    if s["n"] == 0:
        return f"{'n=0':>8}"
    sh = f"{s['sharpe']:+.2f}" if s["sharpe"] is not None else "   n/a"
    tag = f"  {label}" if label else ""
    return (f"n={s['n']:>5}  avg_r5={s['avg_r5']:+.3f}%  "
            f"win%={s['win_pct']:>5.1f}  sharpe={sh:>6}{tag}")


BUCKETS = [
    (0.00, 0.10),
    (0.10, 0.15),
    (0.15, 0.20),
    (0.20, 0.25),
    (0.25, 0.30),
    (0.30, 0.35),
    (0.35, 0.40),
    (0.40, 0.45),
    (0.45, 0.50),
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 2.00),
]


# ---------------------------------------------------------------------------
# Section 1: ALL rows (take + blocked) bucketed by ml_proba
# ---------------------------------------------------------------------------
print("=" * 85)
print("SECTION 1 -- ALL rows (take + blocked), bucketed by ml_proba")
print("  Shows: n total, action split, avg_ret5, win%")
print("=" * 85)

hdr = (f"  {'Bucket':<14} {'n_total':>8}  {'n_take':>7}  {'n_blocked':>9}  "
       f"{'avg_r5':>8}  {'win%':>6}  {'sharpe':>7}")
print(hdr)
print("  " + "-" * 70)

for lo, hi in BUCKETS:
    bucket = [r for r in all_rows if lo <= r["proba"] < hi]
    if not bucket:
        print(f"  [{lo:.2f}-{hi if hi < 2 else '0.65+'}){'':<4}  (empty)")
        continue
    s = stats(bucket)
    n_take    = sum(1 for r in bucket if r["action"] == "take")
    n_blocked = sum(1 for r in bucket if r["action"] == "blocked")
    sh = f"{s['sharpe']:+.2f}" if s["sharpe"] is not None else "   n/a"
    hi_str = f"{hi:.2f}" if hi < 2 else "0.65+"
    print(f"  [{lo:.2f}-{hi_str}){'':>4}  "
          f"{s['n']:>8}  {n_take:>7}  {n_blocked:>9}  "
          f"{s['avg_r5']:>+8.3f}%  {s['win_pct']:>5.1f}%  {sh:>7}")
print()


# ---------------------------------------------------------------------------
# Section 2: BLOCKED rows ONLY — same buckets, with top reason_codes
# ---------------------------------------------------------------------------
print("=" * 85)
print("SECTION 2 -- BLOCKED rows ONLY, bucketed by ml_proba + top reason_codes")
print("=" * 85)

hdr2 = (f"  {'Bucket':<14} {'n':>6}  {'avg_r5':>8}  {'win%':>6}  {'sharpe':>7}  "
        f"{'top_reason_code (n)'}")
print(hdr2)
print("  " + "-" * 80)

for lo, hi in BUCKETS:
    bucket = [r for r in rows_blocked if lo <= r["proba"] < hi]
    if not bucket:
        hi_str = f"{hi:.2f}" if hi < 2 else "0.65+"
        print(f"  [{lo:.2f}-{hi_str}){'':>4}  (empty)")
        continue
    s = stats(bucket)
    # Count reason_codes
    rc_counts = defaultdict(int)
    for r in bucket:
        rc_counts[r["reason_code"]] += 1
    top_rc = sorted(rc_counts.items(), key=lambda x: -x[1])[:3]
    rc_str = "  |  ".join(f"{rc}({n})" for rc, n in top_rc)
    sh = f"{s['sharpe']:+.2f}" if s["sharpe"] is not None else "   n/a"
    hi_str = f"{hi:.2f}" if hi < 2 else "0.65+"
    print(f"  [{lo:.2f}-{hi_str}){'':>4}  "
          f"{s['n']:>6}  {s['avg_r5']:>+8.3f}%  {s['win_pct']:>5.1f}%  {sh:>7}  {rc_str}")
print()


# ---------------------------------------------------------------------------
# Section 3: High-proba blocked (>0.50) — detailed reason_code breakdown
# ---------------------------------------------------------------------------
print("=" * 85)
print("SECTION 3 -- Blocked rows with ml_proba > 0.50: reason_code breakdown")
print("=" * 85)

high_proba_blocked = [r for r in rows_blocked if r["proba"] > 0.50]
print(f"  Total blocked with proba > 0.50: {len(high_proba_blocked):,}")
print()

rc_groups = defaultdict(list)
for r in high_proba_blocked:
    rc_groups[r["reason_code"]].append(r)

sorted_rc = sorted(rc_groups.items(), key=lambda x: -len(x[1]))
hdr3 = f"  {'reason_code':<30} {'n':>6}  {'avg_r5':>8}  {'win%':>6}  {'sharpe':>7}"
print(hdr3)
print("  " + "-" * 65)
for rc, rrows in sorted_rc:
    s = stats(rrows)
    sh = f"{s['sharpe']:+.2f}" if s["sharpe"] is not None else "   n/a"
    print(f"  {rc:<30} {s['n']:>6}  {s['avg_r5']:>+8.3f}%  {s['win_pct']:>5.1f}%  {sh:>7}")
print()

# Fine-grained buckets for high-proba blocked
print("  High-proba blocked -- sub-buckets (0.50+):")
HIGH_BUCKETS = [
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 0.70),
    (0.70, 0.80),
    (0.80, 2.00),
]
hdr4 = f"  {'Bucket':<14} {'n':>6}  {'avg_r5':>8}  {'win%':>6}  {'top_reason_code'}"
print(hdr4)
print("  " + "-" * 65)
for lo, hi in HIGH_BUCKETS:
    bucket = [r for r in rows_blocked if lo <= r["proba"] < hi]
    if not bucket:
        hi_str = f"{hi:.2f}" if hi < 2 else "0.80+"
        print(f"  [{lo:.2f}-{hi_str})  (empty)")
        continue
    s = stats(bucket)
    rc_counts = defaultdict(int)
    for r in bucket:
        rc_counts[r["reason_code"]] += 1
    top_rc = sorted(rc_counts.items(), key=lambda x: -x[1])[:2]
    rc_str = "  |  ".join(f"{rc}({n})" for rc, n in top_rc)
    sh = f"{s['sharpe']:+.2f}" if s["sharpe"] is not None else "   n/a"
    hi_str = f"{hi:.2f}" if hi < 2 else "0.80+"
    print(f"  [{lo:.2f}-{hi_str})  "
          f"{s['n']:>6}  {s['avg_r5']:>+8.3f}%  {s['win_pct']:>5.1f}%  {rc_str}")
print()


# ---------------------------------------------------------------------------
# Section 4: ml_proba_zone gate specifically — confirm upper vs lower blocking
# ---------------------------------------------------------------------------
print("=" * 85)
print("SECTION 4 -- ml_proba_zone gate: upper vs lower block analysis")
print("  (ML_GENERAL_HARD_BLOCK_MAX=1.01 = no upper cap, MIN=0.28)")
print("=" * 85)

zone_blocked = [r for r in rows_blocked if r["reason_code"] == "ml_proba_zone"]
print(f"  Total blocked by ml_proba_zone:  {len(zone_blocked):,}")
print(f"  proba range in zone_blocked: "
      f"min={min((r['proba'] for r in zone_blocked), default=0):.4f}  "
      f"max={max((r['proba'] for r in zone_blocked), default=0):.4f}")
print()

# Upper block: proba > MAX (should be empty if MAX=1.01)
upper_blocked = [r for r in zone_blocked if r["proba"] > 1.00]
lower_blocked = [r for r in zone_blocked if r["proba"] < 0.28]
mid_blocked   = [r for r in zone_blocked if 0.28 <= r["proba"] <= 1.00]

print(f"  Upper-blocked (proba > 1.00):  n={len(upper_blocked)}")
print(f"  Lower-blocked (proba < 0.28):  n={len(lower_blocked)}")
print(f"  Mid-zone-blocked (0.28-1.00):  n={len(mid_blocked)}  *** unexpected if MAX=1.01")
if mid_blocked:
    s = stats(mid_blocked)
    print(f"    -> avg_r5={s['avg_r5']:+.3f}%, win%={s['win_pct']:.1f}%  -- these passed MIN but still blocked")
    probas = sorted(r["proba"] for r in mid_blocked)
    print(f"    -> proba dist: min={probas[0]:.4f} p25={probas[len(probas)//4]:.4f} "
          f"p50={probas[len(probas)//2]:.4f} p75={probas[3*len(probas)//4]:.4f} max={probas[-1]:.4f}")
print()

# Bucket the zone_blocked rows (to show where the [0.50-0.55) bucket sits)
print("  ml_proba_zone blocked, by bucket:")
hdr5 = f"  {'Bucket':<14} {'n':>6}  {'avg_r5':>8}  {'win%':>6}  note"
print(hdr5)
print("  " + "-" * 60)
for lo, hi in BUCKETS:
    bucket = [r for r in zone_blocked if lo <= r["proba"] < hi]
    if not bucket:
        continue
    s = stats(bucket)
    hi_str = f"{hi:.2f}" if hi < 2 else "0.65+"
    note = ""
    if lo >= 0.28:
        note = "  *** INSIDE zone (proba >= MIN=0.28) - anomalous block"
    elif hi <= 0.28:
        note = "  blocked correctly (proba < MIN)"
    else:
        note = "  straddles MIN threshold"
    print(f"  [{lo:.2f}-{hi_str})  "
          f"{s['n']:>6}  {s['avg_r5']:>+8.3f}%  {s['win_pct']:>5.1f}%{note}")
print()


# ---------------------------------------------------------------------------
# Section 5: Correlation between ml_proba and ret_5 (all rows)
# ---------------------------------------------------------------------------
print("=" * 85)
print("SECTION 5 -- Correlation: ml_proba vs ret_5")
print("=" * 85)

probas = [r["proba"] for r in all_rows]
rets   = [r["ret5"]  for r in all_rows]
n = len(probas)

mean_p = sum(probas) / n
mean_r = sum(rets) / n
cov = sum((probas[i] - mean_p) * (rets[i] - mean_r) for i in range(n)) / (n - 1)
sd_p = math.sqrt(sum((x - mean_p) ** 2 for x in probas) / (n - 1))
sd_r = math.sqrt(sum((x - mean_r) ** 2 for x in rets) / (n - 1))
pearson = cov / (sd_p * sd_r) if (sd_p > 1e-9 and sd_r > 1e-9) else 0.0

print(f"  Pearson r(proba, ret_5) over all {n:,} rows: {pearson:+.4f}")
print(f"  (positive = higher proba -> better outcome; 0 = no signal)")
print()

# Separate for take vs blocked
def pearson_r(rows_sub):
    if len(rows_sub) < 3:
        return None
    ps = [r["proba"] for r in rows_sub]
    rs = [r["ret5"]  for r in rows_sub]
    m_p = sum(ps) / len(ps)
    m_r = sum(rs) / len(rs)
    c = sum((ps[i] - m_p) * (rs[i] - m_r) for i in range(len(ps))) / (len(ps) - 1)
    sp = math.sqrt(sum((x - m_p) ** 2 for x in ps) / (len(ps) - 1))
    sr = math.sqrt(sum((x - m_r) ** 2 for x in rs) / (len(rs) - 1))
    return c / (sp * sr) if (sp > 1e-9 and sr > 1e-9) else 0.0

r_take    = pearson_r(rows_take)
r_blocked = pearson_r(rows_blocked)
print(f"  Pearson r for 'take' rows only:    {r_take:+.4f}" if r_take is not None else "  n/a (take)")
print(f"  Pearson r for 'blocked' rows only: {r_blocked:+.4f}" if r_blocked is not None else "  n/a (blocked)")
print()

# Decile table: does proba actually rank outcomes?
sorted_all = sorted(all_rows, key=lambda r: r["proba"])
decile_size = max(1, len(sorted_all) // 10)
print("  Decile table (all rows, sorted by proba low->high):")
print(f"  {'Decile':>8}  {'proba_range':>18}  {'n':>6}  {'avg_r5':>8}  {'win%':>6}")
print("  " + "-" * 55)
for d in range(10):
    start = d * decile_size
    end = (d + 1) * decile_size if d < 9 else len(sorted_all)
    slice_ = sorted_all[start:end]
    if not slice_:
        continue
    s = stats(slice_)
    plo = min(r["proba"] for r in slice_)
    phi = max(r["proba"] for r in slice_)
    print(f"  D{d+1:02d}      [{plo:.3f}-{phi:.3f}]  "
          f"{s['n']:>6}  {s['avg_r5']:>+8.3f}%  {s['win_pct']:>5.1f}%")
print()


# ---------------------------------------------------------------------------
# Section 6: Key question — is [0.50-0.55) finding real or artifact?
# ---------------------------------------------------------------------------
print("=" * 85)
print("SECTION 6 -- Key question: [0.50-0.55) finding real or artifact?")
print("=" * 85)

# Find all rows (any action) in [0.50, 0.55)
bucket_50_55 = [r for r in all_rows if 0.50 <= r["proba"] < 0.55]
take_50_55    = [r for r in rows_take    if 0.50 <= r["proba"] < 0.55]
blocked_50_55 = [r for r in rows_blocked if 0.50 <= r["proba"] < 0.55]

print(f"  All rows  [0.50-0.55): {fmt(stats(bucket_50_55))}")
print(f"  Take      [0.50-0.55): {fmt(stats(take_50_55))}")
print(f"  Blocked   [0.50-0.55): {fmt(stats(blocked_50_55))}")
print()

# Breakdown of blocked 0.50-0.55 by reason_code
if blocked_50_55:
    rc_counts = defaultdict(int)
    for r in blocked_50_55:
        rc_counts[r["reason_code"]] += 1
    top_rc = sorted(rc_counts.items(), key=lambda x: -x[1])
    print(f"  Blocked [0.50-0.55) by reason_code:")
    for rc, cnt in top_rc:
        sub = [r for r in blocked_50_55 if r["reason_code"] == rc]
        s = stats(sub)
        print(f"    {rc:<30} n={cnt:>5}  avg_r5={s['avg_r5']:>+.3f}%  win%={s['win_pct']:.1f}%")
    print()

# Compare adjacent buckets
print("  Adjacent bucket comparison (all rows):")
compare_buckets = [
    (0.40, 0.45),
    (0.45, 0.50),
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 2.00),
]
print(f"  {'Bucket':<14} {'n_all':>7}  {'avg_r5_all':>11}  {'n_take':>7}  {'avg_r5_take':>12}  {'n_blocked':>10}  {'avg_r5_blk':>11}")
print("  " + "-" * 80)
for lo, hi in compare_buckets:
    all_b    = [r for r in all_rows    if lo <= r["proba"] < hi]
    take_b   = [r for r in rows_take   if lo <= r["proba"] < hi]
    block_b  = [r for r in rows_blocked if lo <= r["proba"] < hi]
    sa = stats(all_b)
    st = stats(take_b)
    sb = stats(block_b)
    hi_str = f"{hi:.2f}" if hi < 2 else "0.65+"
    all_avg  = f"{sa['avg_r5']:>+.3f}%" if sa["n"] else "     -"
    take_avg = f"{st['avg_r5']:>+.3f}%" if st["n"] else "     -"
    blk_avg  = f"{sb['avg_r5']:>+.3f}%" if sb["n"] else "     -"
    print(f"  [{lo:.2f}-{hi_str})  "
          f"{sa['n']:>7}  {all_avg:>11}  {st['n']:>7}  {take_avg:>12}  "
          f"{sb['n']:>10}  {blk_avg:>11}")
print()


# ---------------------------------------------------------------------------
# Section 7: Summary verdict
# ---------------------------------------------------------------------------
print("=" * 85)
print("SECTION 7 -- SUMMARY & VERDICT")
print("=" * 85)
print()

# High proba (>0.50) overview
high_take    = [r for r in rows_take    if r["proba"] >= 0.50]
high_blocked = [r for r in rows_blocked if r["proba"] >= 0.50]
all_take_s = stats(rows_take)
high_take_s = stats(high_take)
high_blocked_s = stats(high_blocked)

print(f"  Baseline take (all proba):        {fmt(all_take_s)}")
print(f"  Take with proba >= 0.50:          {fmt(high_take_s)}")
print(f"  Blocked with proba >= 0.50:       {fmt(high_blocked_s)}")
print()

# Is there an upper-cap blocking problem?
if len(high_blocked) > 50 and high_blocked_s["avg_r5"] is not None:
    if high_blocked_s["avg_r5"] > (all_take_s["avg_r5"] or 0.0):
        print("  VERDICT: HIGH-PROBA OVER-BLOCKING DETECTED")
        print(f"    {len(high_blocked)} signals with proba>=0.50 are blocked, "
              f"avg_r5={high_blocked_s['avg_r5']:+.3f}% OUTPERFORMS take baseline.")
        print("    Investigate: which gate blocks them? (see Section 3)")
    else:
        print("  VERDICT: No systematic high-proba over-blocking.")
        print(f"    Blocked high-proba avg_r5={high_blocked_s['avg_r5']:+.3f}% "
              f"vs take baseline {all_take_s['avg_r5']:+.3f}%")
elif len(high_blocked) == 0:
    print("  VERDICT: No blocked rows with proba >= 0.50. Upper cap not an issue.")
else:
    print(f"  VERDICT: Small sample ({len(high_blocked)} rows) — inconclusive.")

print()

# Proba-outcome correlation verdict
print(f"  Pearson r(proba, ret_5): {pearson:+.4f}")
if pearson > 0.05:
    print("  -> Moderate positive correlation: higher proba predicts better return.")
    print("     Blocking high-proba signals is counter-productive.")
elif pearson > 0.01:
    print("  -> Weak positive correlation: proba has some predictive value.")
elif pearson > -0.01:
    print("  -> Near-zero correlation: proba does NOT predict ret_5 well.")
else:
    print("  -> Negative correlation: higher proba predicts WORSE outcome (!)")

print()
print("Done.")
