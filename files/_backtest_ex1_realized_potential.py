"""EX1: realized-to-potential capture metric.

For each paired (entry, exit) on top-20 winners (last 30 d):
  realized_pct  = (exit_price - entry_price) / entry_price * 100
  potential_pct = max(eod_return_pct, tg_return_4h, tg_return_since_open)
                  # best-available proxy of intraday high without klines
  EX1 = clamp(realized_pct / potential_pct, -0.5, 1.5)  if potential_pct > 0

Spec: docs/specs/features/ex1-realized-potential-spec.md
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
    """eod_return_pct in dataset is sometimes decimal, sometimes %.
    Heuristic: if abs > 5 -> already in %; else multiply by 100.
    """
    if v is None: return None
    try: v = float(v)
    except: return None
    return v if abs(v) > 5 else v * 100


# 1) Top-20 + features (potential proxies) per (date, sym)
top20 = set()
potential_proxy = {}  # (d, sym) -> max(eod, tg_4h, tg_since_open)
with io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        try: e = json.loads(ln)
        except: continue
        ts_ms = e.get("ts");
        if not ts_ms: continue
        dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
        if dt < CUT: continue
        sym = e.get("symbol"); d = dt.strftime("%Y-%m-%d")
        if e.get("label_top20") == 1:
            top20.add((d, sym))
        # Aggregate potential per (date, sym): take MAX across all snapshots
        feat = e.get("features") or {}
        candidates = []
        eod = _normalize_pct(e.get("eod_return_pct"))
        if eod is not None: candidates.append(eod)
        for k in ("tg_return_4h", "tg_return_since_open", "tg_return_1h"):
            v = feat.get(k)
            if v is not None:
                v_pct = _normalize_pct(v)
                if v_pct is not None: candidates.append(v_pct)
        if candidates:
            cur_max = potential_proxy.get((d, sym))
            new_max = max(candidates)
            potential_proxy[(d, sym)] = max(cur_max, new_max) if cur_max is not None else new_max

# 2) Pair entries with exits
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
            open_t[sym] = {
                "d": dt.strftime("%Y-%m-%d"),
                "price": float(e.get("price") or e.get("entry_price") or 0),
                "mode": e.get("mode","?"),
                "tf": e.get("tf","?"),
                "entry_dt": dt,
            }
        else:
            ent = open_t.pop(sym, None)
            if not ent: continue
            ex_p = float(e.get("exit_price") or e.get("price") or 0)
            if ent["price"] <= 0 or ex_p <= 0: continue
            pnl_pct = (ex_p - ent["price"]) / ent["price"] * 100
            reason = (e.get("reason") or "")
            # Classify exit reason coarsely
            r_lower = reason.lower()
            if "atr" in r_lower or "трейл" in r_lower or "trail" in r_lower:
                exit_class = "atr_trail"
            elif "max_hold" in r_lower or "время" in r_lower or "лимит" in r_lower:
                exit_class = "time_max_hold"
            elif "ema20" in r_lower or "ema 20" in r_lower:
                exit_class = "ema20_weakness"
            elif "rsi" in r_lower:
                exit_class = "rsi"
            elif "macd" in r_lower:
                exit_class = "macd"
            else:
                exit_class = "other"
            pairs.append({
                "d": ent["d"], "sym": sym, "mode": ent["mode"], "tf": ent["tf"],
                "pnl": pnl_pct, "is_top20": (ent["d"], sym) in top20,
                "potential": potential_proxy.get((ent["d"], sym)),
                "exit_class": exit_class,
                "exit_reason": reason[:60],
            })

print(f"=== EX1: realized-to-potential capture (30 d) ===\n")
print(f"Total paired trades: {len(pairs)}")
print(f"  on top-20 winners: {sum(1 for r in pairs if r['is_top20'])}")
print(f"  with potential data: {sum(1 for r in pairs if r['potential'] is not None)}\n")


def compute_ex1(rows):
    out = []
    for r in rows:
        p = r["potential"]
        if p is None or p <= 0: continue
        ex1 = r["pnl"] / p
        ex1 = max(-0.5, min(1.5, ex1))
        out.append(ex1)
    return out


def stats(values, label):
    if not values:
        print(f"  {label:<32} n=0"); return None
    vs = sorted(values); n = len(vs)
    median = vs[n//2]; mean = sum(vs)/n
    p25 = vs[n//4]; p75 = vs[3*n//4]
    pos = sum(1 for x in vs if x >= 0.5)
    print(f"  {label:<32} n={n:>4}  median={median:+.3f}  mean={mean:+.3f}  "
          f"p25={p25:+.3f}  p75={p75:+.3f}  ex1>=0.5: {100*pos/n:.0f}%")
    return {"n": n, "median": median, "mean": mean, "p25": p25, "p75": p75,
            "share_ex1_ge_05": 100*pos/n}


# ── Overall ──────────────────────────────────────────────────────────
print("Overall (all paired, with potential data):")
top20_pairs = [r for r in pairs if r["is_top20"] and r["potential"] is not None]
non_pairs = [r for r in pairs if not r["is_top20"] and r["potential"] is not None]
s_top20 = stats(compute_ex1(top20_pairs), "top-20 only")
s_non   = stats(compute_ex1(non_pairs),   "non-winners")

# ── Per mode/tf ──────────────────────────────────────────────────────
print(f"\nPer mode/tf (top-20 only):")
groups = defaultdict(list)
for r in top20_pairs:
    groups[f"{r['mode']}/{r['tf']}"].append(r)
for k, rows in sorted(groups.items(), key=lambda x: -len(x[1])):
    if len(rows) < 3: continue
    stats(compute_ex1(rows), k)

# ── Per exit-class ───────────────────────────────────────────────────
print(f"\nPer exit-class (top-20 only):")
exit_groups = defaultdict(list)
for r in top20_pairs:
    exit_groups[r["exit_class"]].append(r)
for k, rows in sorted(exit_groups.items(), key=lambda x: -len(x[1])):
    if len(rows) < 2: continue
    stats(compute_ex1(rows), k)

# ── Worst examples (lowest EX1 on top-20 — biggest left-on-table) ──
print(f"\nWorst 10 cases (top-20, lowest EX1 — most money left on table):")
worst = []
for r in top20_pairs:
    p = r["potential"]
    if p is None or p <= 0: continue
    ex1 = r["pnl"] / p
    ex1 = max(-0.5, min(1.5, ex1))
    worst.append((ex1, r))
worst.sort(key=lambda x: x[0])
print(f"  {'date':<11} {'sym':<10} {'mode/tf':<22} {'pnl':>7} {'pot':>7} {'ex1':>6}  exit_reason")
for ex1, r in worst[:10]:
    print(f"  {r['d']:<11} {r['sym']:<10} {r['mode']+'/'+r['tf']:<22} "
          f"{r['pnl']:>+6.2f}% {r['potential']:>+6.1f}% {ex1:>+5.2f}  {r['exit_reason']}")

metric = {
    "metric": "EX1_realized_potential",
    "top20": s_top20,
    "non_winners": s_non,
}
print("\nMETRIC_JSON:" + json.dumps(metric))
