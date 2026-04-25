"""
_backtest_impulse_guard.py
==========================
Deep-dive on whether impulse_guard is over-blocking profitable signals.

Data:
  - critic_dataset.jsonl  (decision.reason_code, decision.action, labels.ret_5, f.*)
  - bot_events.jsonl      (today's impulse_guard blocks with reason text)

Run:
    pyembed\\python.exe files\\_backtest_impulse_guard.py
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent
FILES = Path(__file__).parent
CRITIC_DS = FILES / "critic_dataset.jsonl"
BOT_EVENTS = REPO / "bot_events.jsonl"

TODAY_STR = "2026-04-19"  # adjust if needed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _avg(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _sd(xs):
    if len(xs) < 2:
        return 0.0
    m = _avg(xs)
    return (sum((v - m) ** 2 for v in xs) / (len(xs) - 1)) ** 0.5


def _sharpe(xs):
    if len(xs) < 5:
        return float("nan")
    a = _avg(xs)
    s = _sd(xs)
    return a / s * math.sqrt(len(xs)) if s > 0 else 0.0


def _stats(xs, label=""):
    n = len(xs)
    if n == 0:
        return {"label": label, "n": 0, "avg_r5": float("nan"),
                "median_r5": float("nan"), "win_pct": float("nan"),
                "sd_r5": float("nan"), "sharpe": float("nan")}
    wins = sum(1 for v in xs if v > 0)
    return {
        "label": label,
        "n": n,
        "avg_r5": _avg(xs),
        "median_r5": statistics.median(xs),
        "win_pct": 100.0 * wins / n,
        "sd_r5": _sd(xs),
        "sharpe": _sharpe(xs),
    }


def print_stats(s, indent="  "):
    nan_str = lambda v: f"{v:+.3f}" if math.isfinite(v) else "  n/a"
    print(f"{indent}{s['label']:<40} n={s['n']:>5}  "
          f"avg_r5={nan_str(s['avg_r5'])}%  "
          f"win%={s['win_pct']:>5.1f}  "
          f"median={nan_str(s['median_r5'])}%  "
          f"Sharpe*sqrtN={s['sharpe']:>+6.2f}")


# ---------------------------------------------------------------------------
# Step 1+2: Load critic_dataset
# ---------------------------------------------------------------------------
print("=" * 70)
print("LOADING critic_dataset.jsonl ...")
print("=" * 70)

impulse_rows = []   # blocked by impulse_guard
take_rows = []      # action == take

total_lines = 0
skipped = 0

with CRITIC_DS.open("r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        total_lines += 1
        try:
            r = json.loads(line)
        except Exception:
            skipped += 1
            continue

        dec = r.get("decision") or {}
        lab = r.get("labels") or {}
        action = dec.get("action")
        rc = dec.get("reason_code") or ""
        ret5 = lab.get("ret_5")
        label5 = lab.get("label_5")

        if ret5 is None:
            continue

        ret5 = float(ret5)
        tf = r.get("tf", "")
        sig_type = r.get("signal_type", "") or ""
        reason_text = dec.get("reason", "") or ""

        f = r.get("f") or {}
        adx = f.get("adx")
        rsi = f.get("rsi")
        vol_x = f.get("vol_x")
        slope = f.get("slope")
        macd_hist_norm = f.get("macd_hist_norm")
        upper_wick_pct = f.get("upper_wick_pct")
        body_pct = f.get("body_pct")

        row = {
            "ret5": ret5,
            "label5": label5,
            "tf": tf,
            "sig_type": sig_type,
            "reason_text": reason_text,
            "adx": float(adx) if adx is not None else None,
            "rsi": float(rsi) if rsi is not None else None,
            "vol_x": float(vol_x) if vol_x is not None else None,
            "slope": float(slope) if slope is not None else None,
            "macd_hist_norm": float(macd_hist_norm) if macd_hist_norm is not None else None,
            "upper_wick_pct": float(upper_wick_pct) if upper_wick_pct is not None else None,
            "body_pct": float(body_pct) if body_pct is not None else None,
        }

        if action == "take":
            take_rows.append(row)
        elif action == "blocked" and "impulse_guard" in rc:
            impulse_rows.append(row)

print(f"  Total lines: {total_lines:,}   parse-errors: {skipped}")
print(f"  take rows (with ret_5): {len(take_rows):,}")
print(f"  impulse_guard blocked rows (with ret_5): {len(impulse_rows):,}")


# ---------------------------------------------------------------------------
# Step 3: Overall comparison
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 3 — OVERALL COMPARISON")
print("=" * 70)

take_r5 = [r["ret5"] for r in take_rows]
imp_r5 = [r["ret5"] for r in impulse_rows]

take_s = _stats(take_r5, "take (baseline)")
imp_s = _stats(imp_r5, "impulse_guard BLOCKED")

print_stats(take_s)
print_stats(imp_s)

miss = imp_s["avg_r5"] - take_s["avg_r5"]
print(f"\n  >>> avg miss (blocked - take) = {miss:+.3f}%  "
      f"({'OVER-BLOCKING' if miss > 0.05 else 'within margin'})")


# ---------------------------------------------------------------------------
# Step 4: Breakdown by (tf, signal_type)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 4 — IMPULSE_GUARD BLOCKED: breakdown by (tf, signal_type)")
print("=" * 70)

by_tf_sig = defaultdict(list)
for r in impulse_rows:
    key = f"{r['tf']} / {r['sig_type']}"
    by_tf_sig[key].append(r["ret5"])

# Sort by count desc
sorted_keys = sorted(by_tf_sig.keys(), key=lambda k: -len(by_tf_sig[k]))
for key in sorted_keys:
    s = _stats(by_tf_sig[key], key)
    print_stats(s)

# tf-only breakdown
print()
print("  [TF only]")
by_tf = defaultdict(list)
for r in impulse_rows:
    by_tf[r["tf"] or "?"].append(r["ret5"])
for tf, xs in sorted(by_tf.items(), key=lambda kv: -len(kv[1])):
    s = _stats(xs, f"tf={tf}")
    print_stats(s)


# ---------------------------------------------------------------------------
# Step 5: Feature distributions
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 5 — FEATURE DISTRIBUTIONS (impulse_guard blocked vs take)")
print("=" * 70)

def _feat_summary(rows, feat_key):
    vals = [r[feat_key] for r in rows if r[feat_key] is not None]
    if not vals:
        return "n/a"
    return (f"n={len(vals)}  mean={_avg(vals):.2f}  "
            f"p25={sorted(vals)[len(vals)//4]:.2f}  "
            f"p50={sorted(vals)[len(vals)//2]:.2f}  "
            f"p75={sorted(vals)[3*len(vals)//4]:.2f}")

features = ["adx", "rsi", "vol_x", "slope", "macd_hist_norm", "upper_wick_pct", "body_pct"]

print(f"  {'feature':<22}  {'impulse_guard BLOCKED':<58}  {'take (baseline)'}")
for feat in features:
    b_str = _feat_summary(impulse_rows, feat)
    t_str = _feat_summary(take_rows, feat)
    print(f"  {feat:<22}  BLOCKED: {b_str}")
    print(f"  {'':22}  TAKE:    {t_str}")
    print()


# ---------------------------------------------------------------------------
# Step 6+7: ADX bucket analysis
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 6+7 — ADX BUCKETS for impulse_guard blocked")
print("=" * 70)

adx_buckets = [
    ("<15",    lambda a: a < 15),
    ("15-20",  lambda a: 15 <= a < 20),
    ("20-25",  lambda a: 20 <= a < 25),
    ("25-30",  lambda a: 25 <= a < 30),
    ("30-40",  lambda a: 30 <= a < 40),
    ("40+",    lambda a: a >= 40),
]

print(f"  {'ADX bucket':<12}  {'n':>5}  {'avg_r5%':>8}  {'win%':>6}  {'Sharpe*sqrtN':>13}  interpretation")
for label, pred in adx_buckets:
    subset = [r["ret5"] for r in impulse_rows
              if r["adx"] is not None and pred(r["adx"])]
    if not subset:
        print(f"  {label:<12}  {'0':>5}  {'n/a':>8}  {'n/a':>6}  {'n/a':>13}")
        continue
    a = _avg(subset)
    w = 100.0 * sum(1 for v in subset if v > 0) / len(subset)
    sh = _sharpe(subset)
    note = ""
    if a > 0.1 and len(subset) >= 10:
        note = "<-- OVER-BLOCKING (good signals wrongly blocked)"
    elif a < -0.1 and len(subset) >= 10:
        note = "<-- correctly blocked (bad signals)"
    print(f"  {label:<12}  {len(subset):>5}  {a:>+8.3f}  {w:>5.1f}%  {sh:>+13.2f}  {note}")

# Same for take rows
print()
print("  [TAKE baseline by ADX bucket]")
print(f"  {'ADX bucket':<12}  {'n':>5}  {'avg_r5%':>8}  {'win%':>6}")
for label, pred in adx_buckets:
    subset = [r["ret5"] for r in take_rows
              if r["adx"] is not None and pred(r["adx"])]
    if not subset:
        print(f"  {label:<12}  {'0':>5}  {'n/a':>8}  {'n/a':>6}")
        continue
    a = _avg(subset)
    w = 100.0 * sum(1 for v in subset if v > 0) / len(subset)
    print(f"  {label:<12}  {len(subset):>5}  {a:>+8.3f}  {w:>5.1f}%")


# ---------------------------------------------------------------------------
# Step 7b: Reason text breakdown (what sub-check triggered?)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 7b — REASON TEXT BREAKDOWN (what sub-check inside impulse_guard?)")
print("=" * 70)

reason_buckets = defaultdict(list)
for r in impulse_rows:
    rt = r["reason_text"].lower()
    if "adx" in rt and ("weak" in rt or "< " in rt):
        key = "ADX below floor (weak impulse)"
    elif "rsi" in rt and ">" in rt:
        key = "RSI too high (overbought)"
    elif "ext" in rt and "atr" in rt:
        key = "ext_ATR too high (price extended above EMA20)"
    elif "daily_range" in rt or "range" in rt:
        key = "daily_range too high (late entry)"
    elif "macd" in rt:
        key = "MACD fade (momentum weakening)"
    elif "price_edge" in rt or "edge" in rt:
        key = "price_edge too high"
    elif "late" in rt:
        key = "late entry (general)"
    else:
        key = f"other: {r['reason_text'][:60]}"
    reason_buckets[key].append(r["ret5"])

print(f"  {'sub-check':<52}  {'n':>5}  {'avg_r5%':>8}  {'win%':>6}  {'Sharpe*sqrtN':>13}")
for key, xs in sorted(reason_buckets.items(), key=lambda kv: -len(kv[1])):
    a = _avg(xs)
    w = 100.0 * sum(1 for v in xs if v > 0) / len(xs)
    sh = _sharpe(xs)
    print(f"  {key:<52}  {len(xs):>5}  {a:>+8.3f}  {w:>5.1f}%  {sh:>+13.2f}")


# ---------------------------------------------------------------------------
# Step 8: Relaxation sweep — ADX floor thresholds
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 8 — RELAXATION SWEEP: What if ADX floor was lower?")
print("=" * 70)
print("  (Simulates: unblock rows where ADX >= new_floor. Measures what we'd gain.)")
print()

# ADX-blocked rows
adx_blocked = [r for r in impulse_rows
               if r["adx"] is not None
               and r["reason_text"] and "adx" in r["reason_text"].lower()]
print(f"  Rows blocked by ADX check: {len(adx_blocked)}")

sweep_floors = [8, 10, 12, 14, 15, 16, 17, 18, 20, 22, 25]
print(f"\n  {'ADX_floor':<12}  {'unblocked_n':>12}  {'unblocked_avg_r5%':>18}  "
      f"{'unblocked_win%':>15}  {'still_blocked_n':>16}  verdict")

for floor in sweep_floors:
    unblocked = [r for r in adx_blocked if r["adx"] >= floor]
    still_blocked = [r for r in adx_blocked if r["adx"] < floor]
    if not unblocked:
        print(f"  {floor:<12}  {'0':>12}  {'n/a':>18}  {'n/a':>15}  {'n/a':>16}")
        continue
    ua = _avg([r["ret5"] for r in unblocked])
    uw = 100.0 * sum(1 for r in unblocked if r["ret5"] > 0) / len(unblocked)
    sb_r5 = [r["ret5"] for r in still_blocked]
    sa = _avg(sb_r5) if sb_r5 else float("nan")
    verdict = ""
    if ua > take_s["avg_r5"] + 0.05:
        verdict = "GAIN vs take"
    elif ua < take_s["avg_r5"] - 0.05:
        verdict = "worse than take"
    print(f"  {floor:<12}  {len(unblocked):>12}  {ua:>+18.3f}  {uw:>14.1f}%  "
          f"{len(still_blocked):>16}  {verdict}")

# RSI sweep
print()
print("  Rows blocked by RSI check:")
rsi_blocked = [r for r in impulse_rows
               if r["rsi"] is not None
               and r["reason_text"] and "rsi" in r["reason_text"].lower()]
print(f"  n={len(rsi_blocked)}")

if rsi_blocked:
    rsi_r5 = [r["ret5"] for r in rsi_blocked]
    ra = _avg(rsi_r5)
    rw = 100.0 * sum(1 for v in rsi_r5 if v > 0) / len(rsi_r5)
    rsh = _sharpe(rsi_r5)
    print(f"  avg_r5={ra:+.3f}%  win%={rw:.1f}%  Sharpe*sqrtN={rsh:+.2f}")
    # RSI bucket sweep
    rsi_floors = [68, 70, 72, 74, 76, 78, 80, 82, 85]
    print(f"\n  {'RSI_max':<10}  {'unblocked_n':>12}  {'unblocked_avg_r5%':>18}  {'unblocked_win%':>15}")
    for rmax in rsi_floors:
        unblocked = [r for r in rsi_blocked if r["rsi"] is not None and r["rsi"] <= rmax]
        if not unblocked:
            print(f"  {rmax:<10}  {'0':>12}  {'n/a':>18}  {'n/a':>15}")
            continue
        ua = _avg([r["ret5"] for r in unblocked])
        uw = 100.0 * sum(1 for r in unblocked if r["ret5"] > 0) / len(unblocked)
        print(f"  {rmax:<10}  {len(unblocked):>12}  {ua:>+18.3f}  {uw:>14.1f}%")


# ---------------------------------------------------------------------------
# Step 8b: EXT_ATR relaxation sweep
# ---------------------------------------------------------------------------
print()
print("  Rows blocked by ext_ATR / daily_range check:")
ext_blocked = [r for r in impulse_rows
               if r["reason_text"] and
               ("ext" in r["reason_text"].lower() or "range" in r["reason_text"].lower()
                or "atr" in r["reason_text"].lower())]
print(f"  n={len(ext_blocked)}")
if ext_blocked:
    ea = _avg([r["ret5"] for r in ext_blocked])
    ew = 100.0 * sum(1 for r in ext_blocked if r["ret5"] > 0) / len(ext_blocked)
    esh = _sharpe([r["ret5"] for r in ext_blocked])
    print(f"  avg_r5={ea:+.3f}%  win%={ew:.1f}%  Sharpe*sqrtN={esh:+.2f}")


# ---------------------------------------------------------------------------
# Load bot_events.jsonl — today's impulse_guard blocks
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print(f"BOT_EVENTS.JSONL — impulse_guard blocks on {TODAY_STR}")
print("=" * 70)

today_blocks = []
today_all_blocks = 0
today_impulse_blocks = 0

if BOT_EVENTS.exists():
    with BOT_EVENTS.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue

            # Filter to today
            ts = ev.get("ts") or ev.get("timestamp") or ev.get("bar_ts")
            if ts:
                try:
                    if isinstance(ts, (int, float)) and ts > 1e10:
                        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                    elif isinstance(ts, (int, float)):
                        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    else:
                        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    if dt.strftime("%Y-%m-%d") != TODAY_STR:
                        continue
                except Exception:
                    continue
            else:
                continue

            action = ev.get("action") or ev.get("decision", {}).get("action", "")
            rc = (ev.get("reason_code") or
                  ev.get("decision", {}).get("reason_code", "") or "")
            reason = (ev.get("reason") or
                      ev.get("decision", {}).get("reason", "") or "")

            if action == "blocked":
                today_all_blocks += 1
                if "impulse_guard" in rc:
                    today_impulse_blocks += 1
                    today_blocks.append({
                        "sym": ev.get("sym", "?"),
                        "tf": ev.get("tf", "?"),
                        "sig_type": ev.get("signal_type", "?"),
                        "reason": reason,
                        "ts": dt.strftime("%H:%M:%S"),
                    })

    print(f"  Total blocked events today: {today_all_blocks}")
    print(f"  impulse_guard blocked today: {today_impulse_blocks}")
    if today_blocks:
        print()
        print(f"  Sample impulse_guard blocks today (up to 25):")
        print(f"  {'time':>8}  {'sym':<14}  {'tf':<5}  {'sig_type':<15}  reason")
        for ev in today_blocks[:25]:
            reason_short = ev["reason"][:80] if ev["reason"] else "?"
            print(f"  {ev['ts']:>8}  {ev['sym']:<14}  {ev['tf']:<5}  "
                  f"{ev['sig_type']:<15}  {reason_short}")
        if len(today_blocks) > 25:
            print(f"  ... and {len(today_blocks) - 25} more")

        # Reason sub-type distribution for today
        print()
        print("  Reason sub-type counts (today):")
        today_reason_counts = defaultdict(int)
        for ev in today_blocks:
            rt = ev["reason"].lower()
            if "adx" in rt:
                today_reason_counts["ADX below floor"] += 1
            elif "rsi" in rt:
                today_reason_counts["RSI too high"] += 1
            elif "ext" in rt and "atr" in rt:
                today_reason_counts["ext_ATR too high"] += 1
            elif "range" in rt:
                today_reason_counts["daily_range too high"] += 1
            elif "macd" in rt:
                today_reason_counts["MACD fade"] += 1
            elif "edge" in rt:
                today_reason_counts["price_edge too high"] += 1
            else:
                today_reason_counts[f"other: {ev['reason'][:50]}"] += 1
        for key, cnt in sorted(today_reason_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {key:<45}  n={cnt}")
else:
    print("  bot_events.jsonl not found at expected path.")


# ---------------------------------------------------------------------------
# Summary verdict
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("SUMMARY / VERDICT")
print("=" * 70)

take_avg = take_s["avg_r5"]
imp_avg = imp_s["avg_r5"]
miss_pct = imp_avg - take_avg

print(f"  take baseline:            avg_r5={take_avg:+.3f}%  win%={take_s['win_pct']:.1f}%  n={take_s['n']}")
print(f"  impulse_guard blocked:    avg_r5={imp_avg:+.3f}%  win%={imp_s['win_pct']:.1f}%  n={imp_s['n']}")
print(f"  miss delta:               {miss_pct:+.3f}%")
print()

if miss_pct > 0.3:
    print("  VERDICT: SEVERE over-blocking — impulse_guard is blocking signals that "
          "significantly outperform actual entries. Relaxation strongly recommended.")
elif miss_pct > 0.05:
    print("  VERDICT: MODERATE over-blocking — impulse_guard blocks positive-return signals "
          "above take baseline. Consider targeted relaxation.")
else:
    print("  VERDICT: NO SIGNIFICANT over-blocking — impulse_guard is performing correctly "
          "or close to the take baseline.")

# Top ADX bucket that is most over-blocked
best_bucket = None
best_miss = 0.0
for label, pred in adx_buckets:
    subset_r5 = [r["ret5"] for r in impulse_rows
                 if r["adx"] is not None and pred(r["adx"])]
    if len(subset_r5) >= 10:
        ba = _avg(subset_r5)
        m = ba - take_avg
        if m > best_miss:
            best_miss = m
            best_bucket = label

if best_bucket:
    print(f"\n  Worst ADX bucket for over-blocking: ADX {best_bucket} "
          f"(miss vs take = {best_miss:+.3f}%)")
    print(f"  -> These high-ADX signals are being wrongly rejected.")

print()
print("Done.")
