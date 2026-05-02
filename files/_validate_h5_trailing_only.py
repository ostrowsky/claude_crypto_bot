"""H5: trailing-only after break-even.

Hypothesis: when pos.pnl >= +0.5%, suppress soft EMA20-weakness exits
(2 closes below EMA20, slope<0, adx_weak, below_ema20+weakness).
Keep ATR-trail, RSI overbought, and time max_hold as-is.

Backtest 30 d:
  1) Count paired exits by reason class.
  2) Of those, how many would be SUPPRESSED by H5 (reason is soft + pnl >= 0.5%)?
  3) For each suppressed candidate, look at potential (max(eod_return, tg_4h, ...)).
     If potential >> realised, suppression is value.
  4) For top-20 only (since EarlyCapture target).
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
NOW = datetime.now(timezone.utc); CUT = NOW - timedelta(days=30)


def _normalize_pct(v):
    if v is None: return None
    try: v = float(v)
    except: return None
    return v if abs(v) > 5 else v * 100


# Top-20 + potential
top20 = set(); potential = {}
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
        feat = e.get("features") or {}
        cands = []
        eod = _normalize_pct(e.get("eod_return_pct"))
        if eod is not None: cands.append(eod)
        for k in ("tg_return_4h", "tg_return_since_open", "tg_return_1h"):
            v = _normalize_pct(feat.get(k))
            if v is not None: cands.append(v)
        if cands:
            cur = potential.get((d, sym))
            new = max(cands)
            potential[(d, sym)] = max(cur, new) if cur is not None else new


def is_soft_ema_reason(reason: str) -> bool:
    """Soft EMA-pattern-based exits that H5 would suppress."""
    if not reason: return False
    r = reason.lower()
    soft_markers = [
        "2 закрытия подряд ниже ema20",
        "ниже ema20",
        "ema20 разворачивается",
        "adx ослабевает",
    ]
    if any(m in r for m in soft_markers):
        # but NOT WEAK (already has own guard)
        if r.startswith("⚠️") or "weak" in r:
            return False
        return True
    return False


# Pair exits, classify
open_t = {}; pairs = []
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
        if ev == "entry":
            open_t[sym] = {"d": dt.strftime("%Y-%m-%d"),
                           "price": float(e.get("price") or e.get("entry_price") or 0),
                           "mode": e.get("mode","?"), "tf": e.get("tf","?")}
        else:
            ent = open_t.pop(sym, None)
            if not ent: continue
            ex_p = float(e.get("exit_price") or e.get("price") or 0)
            if ent["price"] <= 0 or ex_p <= 0: continue
            pnl = (ex_p - ent["price"]) / ent["price"] * 100
            reason = (e.get("reason") or "")
            pairs.append({"d": ent["d"], "sym": sym, "mode": ent["mode"], "tf": ent["tf"],
                          "pnl": pnl, "reason": reason,
                          "is_top20": (ent["d"], sym) in top20,
                          "potential": potential.get((ent["d"], sym)),
                          "soft_ema": is_soft_ema_reason(reason)})

print(f"=== H5 trailing-only-after-break-even validation (30 d) ===\n")
print(f"Total paired exits: {len(pairs)}")

# H5 suppression criteria: pnl >= +0.5% AND reason is soft EMA-pattern
H5_BREAK_EVEN_PCT = 0.5
suppressed = [r for r in pairs if r["soft_ema"] and r["pnl"] >= H5_BREAK_EVEN_PCT]
print(f"H5-eligible (soft EMA exit AND pnl>=+0.5%): {len(suppressed)}")
print(f"  on top-20: {sum(1 for r in suppressed if r['is_top20'])}")
print(f"  on non-winners: {sum(1 for r in suppressed if not r['is_top20'])}")

# For each suppressed: realized vs potential
print(f"\n--- Suppressed exits (top-20 only): would they have made more if kept? ---")
top20_supp = [r for r in suppressed if r["is_top20"] and r["potential"] is not None]
print(f"With potential data: {len(top20_supp)}")
total_realized = sum(r["pnl"] for r in top20_supp)
total_potential = sum(r["potential"] for r in top20_supp if r["potential"]>0)
print(f"  total realized pnl on suppressed: {total_realized:+.2f}%")
print(f"  total potential pnl              : {total_potential:+.2f}%")
print(f"  ratio realized/potential         : {total_realized/total_potential if total_potential>0 else 0:.4f}")
print(f"  delta (potential − realized)     : {total_potential - total_realized:+.2f}% ← potential left on table")

# Sample
print(f"\nSample top-20 suppressed exits (first 12):")
print(f"  {'date':<11} {'sym':<10} {'mode/tf':<22} {'pnl':>7} {'pot':>8}  reason")
for r in top20_supp[:12]:
    p = r["potential"] or 0
    print(f"  {r['d']:<11} {r['sym']:<10} {r['mode']+'/'+r['tf']:<22} "
          f"{r['pnl']:>+6.2f}% {p:>+7.1f}%  {r['reason'][:70]}")

# Counter: how many soft exits are AT loss (negative pnl)? Those are NOT suppressed
# (correctly — protect against further loss)
soft_at_loss = [r for r in pairs if r["soft_ema"] and r["pnl"] < H5_BREAK_EVEN_PCT]
print(f"\n--- Counter: soft EMA exits where pnl < +0.5% (NOT suppressed by H5) ---")
print(f"n={len(soft_at_loss)}  avg_pnl={sum(r['pnl'] for r in soft_at_loss)/max(1,len(soft_at_loss)):+.2f}%")
print(f"  → these are protective exits, H5 keeps them ON")

# Risk: how many suppressed exits had next-bar reversal that ATR-trail wouldn't catch fast enough?
# Without intraday data we can't test. Document as risk.
print(f"\n--- Reason-class distribution ---")
classes = defaultdict(int)
for r in pairs:
    rl = r["reason"].lower()
    if "atr" in rl or "трейл" in rl: classes["atr_trail"] += 1
    elif "2 закрытия" in rl: classes["2_closes_below_ema20"] += 1
    elif "ниже ema20" in rl: classes["below_ema20"] += 1
    elif "ema20 разворач" in rl: classes["ema20_reversing"] += 1
    elif "adx ослаб" in rl: classes["adx_weak"] += 1
    elif "rsi перек" in rl: classes["rsi_overbought"] += 1
    elif "rsi" in rl: classes["rsi_other"] += 1
    elif rl.startswith("⚠️") or "weak" in rl: classes["weak"] += 1
    elif "macd" in rl: classes["macd"] += 1
    elif "лимит" in rl or "max_hold" in rl or "время" in rl: classes["time_max_hold"] += 1
    else: classes["other"] += 1
for k, v in sorted(classes.items(), key=lambda x: -x[1]):
    print(f"  {k:<28} {v}")

# Acceptance check
metric = {
    "metric": "H5_validation",
    "n_total_pairs": len(pairs),
    "n_h5_eligible": len(suppressed),
    "n_h5_eligible_top20": sum(1 for r in suppressed if r["is_top20"]),
    "top20_total_realized_pct": total_realized,
    "top20_total_potential_pct": total_potential,
    "top20_left_on_table_pct": total_potential - total_realized if total_potential > 0 else 0,
    "soft_at_loss_count": len(soft_at_loss),
    "soft_at_loss_avg_pnl": sum(r["pnl"] for r in soft_at_loss)/max(1, len(soft_at_loss)),
}
print("\nMETRIC_JSON:" + json.dumps(metric))
