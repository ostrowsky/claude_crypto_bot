"""
Diagnose + backtest ML veto for 1h impulse / impulse_speed.

Current logic (monitor.py L1807-1810, generic 1h path):
  veto if (final <= -1.50 AND TG <= 0.25)

Proposed 1h impulse-specific multi-feature gate (mirror of 15m impulse):
  veto if (final <= V_FINAL AND EV <= V_EV AND Q <= V_Q AND TG <= V_TG AND CAP <= V_CAP)

Steps:
  1. Load ranker payload (ml_candidate_ranker.json).
  2. Stream critic_dataset.jsonl, for each 'take' decision on 1h impulse/impulse_speed:
     compute ranker components (final, ev, q, tg, cap).
  3. Bucket by ret_5 outcome (winner vs loser).
  4. Sweep thresholds; report block rate, blocked_avg_ret, kept_avg_ret.
  5. Compare current gate vs 7 candidate composite gates.

BNT reference: final=-0.33 EV=-0.38 Q=0.56 TG=0.23 CAP=0.06, ret ~ -0.6%.
"""
from __future__ import annotations
import json, io, sys, math
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
FILES = Path(__file__).resolve().parent

import ml_candidate_ranker
import ml_signal_model  # noqa

print("Loading ranker payload...")
payload = json.loads((FILES / "ml_candidate_ranker.json").read_text(encoding="utf-8"))
print(f"  payload_version={payload.get('payload_version')}  features={len(payload.get('feature_names', []))}")

# Sanity: does payload have top_gainer + capture?
has_tg = "top_gainer_model" in payload
has_cap = "capture_model" in payload
print(f"  has top_gainer_model={has_tg}  has capture_model={has_cap}")

print("\nStreaming critic_dataset.jsonl and scoring 1h impulse/impulse_speed 'take' rows...")
rows_scored = []
n_total = n_take = n_1h_imp = 0
with io.open(FILES / "critic_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        n_total += 1
        try: rec = json.loads(ln)
        except: continue
        dec = rec.get("decision") or {}
        if dec.get("action") != "take": continue
        n_take += 1
        if rec.get("tf") != "1h": continue
        if rec.get("signal_type") not in ("impulse", "impulse_speed"): continue
        n_1h_imp += 1
        labels = rec.get("labels") or {}
        ret5 = labels.get("ret_5")
        if ret5 is None: continue
        try:
            comps = ml_candidate_ranker.predict_components_from_candidate_payload(payload, rec)
        except Exception as e:
            continue
        rows_scored.append({
            "sym": rec.get("sym"), "sig": rec.get("signal_type"),
            "ret5": float(ret5),
            "exit_pnl": labels.get("trade_exit_pnl"),
            "final": float(comps.get("final_score", 0)),
            "ev": float(comps.get("ev_raw", 0)),
            "q": float(comps.get("quality_proba", 0)),
            "tg": float(comps.get("top_gainer_prob", 0)),
            "cap": float(comps.get("capture_ratio_pred", 0)),
        })

print(f"  total rows: {n_total}  |  take: {n_take}  |  1h impulse*: {n_1h_imp}  |  scored: {len(rows_scored)}")

# ── Distribution summary ──
print("\n=== Ranker components on 1h impulse/impulse_speed (scored 'take') ===")
def pct(vals, p):
    s = sorted(vals); i = int(len(s)*p)
    return s[min(i, len(s)-1)]
for key in ["final", "ev", "q", "tg", "cap"]:
    vals = [r[key] for r in rows_scored]
    print(f"  {key:>6s}:  min={min(vals):+.3f}  p10={pct(vals,0.1):+.3f}  p25={pct(vals,0.25):+.3f}  p50={pct(vals,0.5):+.3f}  p75={pct(vals,0.75):+.3f}  max={max(vals):+.3f}  mean={sum(vals)/len(vals):+.3f}")

# ── Outcome distribution ──
def agg(lst, name):
    if not lst:
        print(f"  {name:55s} n=0"); return None
    n = len(lst); rets = [r["ret5"] for r in lst]
    wins = sum(1 for v in rets if v>0); tot=sum(rets); avg=tot/n
    sd = (sum((v-avg)**2 for v in rets)/n)**0.5 if n>1 else 0
    sh = (avg/sd*math.sqrt(n)) if sd>0 else 0
    print(f"  {name:55s} n={n:>4d}  win={wins/n*100:>5.1f}%  avg_ret5={avg:+6.3f}%  sum={tot:+7.1f}%  sharpe={sh:+.2f}")
    return {"n":n,"avg":avg,"win":wins/n*100,"sum":tot,"sharpe":sh}

print("\n=== BASELINE (all 1h impulse/impulse_speed 'take') ===")
base = agg(rows_scored, "baseline")

# ── Current gate (generic 1h) ──
print("\n=== CURRENT gate: veto if (final<=-1.50 AND tg<=0.25) ===")
cur_blocked = [r for r in rows_scored if r["final"]<=-1.50 and r["tg"]<=0.25]
cur_kept    = [r for r in rows_scored if not (r["final"]<=-1.50 and r["tg"]<=0.25)]
agg(cur_blocked, "blocked by CURRENT")
agg(cur_kept,    "KEPT by CURRENT")

# ── Gate variants: 5-feature AND — catches BNT (final=-0.33, ev=-0.38, q=0.56, tg=0.23, cap=0.06) ──
print("\n=== PROPOSED 5-feature AND gates (mirror of 15m impulse veto) ===")
variants = [
    # (name, final_max, ev_max, q_max, tg_max, cap_max)
    ("V1 strict",      -0.80, -0.40, 0.55, 0.25, 0.08),
    ("V2 moderate",     0.00, -0.20, 0.60, 0.28, 0.10),
    ("V3 catches_BNT", -0.20, -0.30, 0.58, 0.25, 0.08),
    ("V4 ev-focused",   0.50, -0.30, 0.70, 0.30, 0.15),
    ("V5 very-strict", -1.00, -0.50, 0.50, 0.22, 0.06),
    ("V6 q-focused",    0.30, -0.10, 0.55, 0.30, 0.12),
    ("V7 multi-cond",  -0.30, -0.25, 0.60, 0.28, 0.10),
]
for name, fm, em, qm, tgm, capm in variants:
    blocked = [r for r in rows_scored
               if r["final"]<=fm and r["ev"]<=em and r["q"]<=qm and r["tg"]<=tgm and r["cap"]<=capm]
    kept    = [r for r in rows_scored if r not in blocked]
    # Use set for speed on larger sets:
    blocked_set = {id(r) for r in blocked}
    kept = [r for r in rows_scored if id(r) not in blocked_set]
    print(f"\n  --- {name:16s} final<={fm:+.2f} ev<={em:+.2f} q<={qm:.2f} tg<={tgm:.2f} cap<={capm:.2f} ---")
    bk = agg(blocked, "BLOCKED")
    kp = agg(kept,    "KEPT")
    if base and bk and kp:
        d_kept = kp["avg"] - base["avg"]
        block_rate = len(blocked) / len(rows_scored) * 100
        print(f"      delta_kept_avg={d_kept:+.4f}%   block_rate={block_rate:.1f}%")
        # BNT-like check
        bnt_like = [r for r in blocked if r["final"]>=-0.50 and r["final"]<=0.0 and r["ev"]<=-0.3 and r["q"]<=0.65 and r["tg"]<=0.30]
        if bnt_like:
            print(f"      'BNT-like' (weak-not-catastrophic) catches: {len(bnt_like)}/{len(blocked)} blocked")
            bnt_rets = [r["ret5"] for r in bnt_like]
            print(f"      their avg_ret5: {sum(bnt_rets)/len(bnt_rets):+.3f}%")

# ── What do the BNT-like trades look like in data? ──
print("\n=== 'BNT-like' profile census (anywhere in 1h impulse* scored, by current/proposed gate result) ===")
bnt_like_all = [r for r in rows_scored
                if -0.50 <= r["final"] <= 0.0
                and r["ev"] <= -0.25
                and 0.40 <= r["q"] <= 0.65
                and r["tg"] <= 0.30]
print(f"  profile matches: {len(bnt_like_all)}")
agg(bnt_like_all, "BNT-like profile outcomes")

# Show 5 worst / 5 best BNT-likes
if bnt_like_all:
    sl = sorted(bnt_like_all, key=lambda r: r["ret5"])
    print("  worst 5 of profile (ret5):")
    for r in sl[:5]:
        print(f"    {r['sym']:12s} {r['sig']:>15s}  final={r['final']:+.2f} ev={r['ev']:+.2f} q={r['q']:.2f} tg={r['tg']:.2f} cap={r['cap']:.2f}  ret5={r['ret5']:+.2f}%")
    print("  best 5 of profile:")
    for r in sl[-5:]:
        print(f"    {r['sym']:12s} {r['sig']:>15s}  final={r['final']:+.2f} ev={r['ev']:+.2f} q={r['q']:.2f} tg={r['tg']:.2f} cap={r['cap']:.2f}  ret5={r['ret5']:+.2f}%")

# ── Also check ranker quality on 1h impulse broadly: does it discriminate at all? ──
print("\n=== Ranker discriminative power on this segment (quartile analysis) ===")
for metric in ["final", "ev", "q", "tg"]:
    vals = [(r[metric], r["ret5"]) for r in rows_scored]
    vals.sort()
    n = len(vals); q = n // 4
    if q == 0: continue
    q1 = vals[:q]; q4 = vals[-q:]
    q1_ret = sum(v[1] for v in q1)/len(q1)
    q4_ret = sum(v[1] for v in q4)/len(q4)
    spread = q4_ret - q1_ret
    print(f"  {metric:>6s}:  Q1 mean ret5={q1_ret:+.3f}%   Q4 mean ret5={q4_ret:+.3f}%   spread(Q4-Q1)={spread:+.3f}%")
