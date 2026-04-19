"""
Backtest: what if we add a TG-proba floor to 15m impulse/impulse_speed entries?

Approach:
  1. Load top_gainer_dataset.jsonl — per-day snapshots with tg_* features
     and EOD label. This is what the TG model was trained on.
  2. Load top_gainer_model.json and score each snapshot -> tg20_proba.
  3. For each historical trade (entry+exit pair from bot_events.jsonl),
     look up tg20_proba for (symbol, date_of_entry).
  4. Sweep thresholds; for each, report: kept/filtered counts, win%,
     avg pnl, total pnl. Compare to baseline (no filter).

Daily granularity is a limitation — intra-day TG proba changes aren't
captured. But it's representative of how a TG floor at the bar level
would behave on average.

Run: pyembed\\python.exe files\\_backtest_tg_floor.py
"""
from __future__ import annotations
import json, io, sys, base64, tempfile, os, math
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")

FILES = Path(__file__).resolve().parent
ROOT = FILES.parent

# ── Load TG model ───────────────────────────────────────────────────────────
print("Loading top_gainer_model...")
tg_model_blob = json.load(io.open(FILES / "top_gainer_model.json", encoding="utf-8"))
TG_FEATURE_NAMES = tg_model_blob["feature_names"]
TG_TIERS = tg_model_blob["tier_models"]

from catboost import CatBoostClassifier
tier_models = {}
for tier, meta in TG_TIERS.items():
    mj = meta.get("model_json")
    if mj is None: continue
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")
    tmp.write(mj); tmp.close()
    cb = CatBoostClassifier()
    cb.load_model(tmp.name, format="json"); os.unlink(tmp.name)
    tier_models[tier] = cb
print(f"  loaded tiers: {list(tier_models.keys())}")

def score_tg(features: dict) -> float:
    """Return tg20_proba for the given feature dict (missing -> 0.0)."""
    x = [float(features.get(name, 0.0) or 0.0) for name in TG_FEATURE_NAMES]
    import numpy as np
    X = np.array([x], dtype=np.float32)
    p = tier_models["top20"].predict_proba(X)[0, -1]
    return float(p)

# ── Load TG snapshots keyed by (symbol, date) ──────────────────────────────
print("Loading top_gainer_dataset.jsonl...")
snap_by_sym_date: dict = {}
n_snaps = 0
with io.open(FILES / "top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        try: e = json.loads(ln)
        except: continue
        sym = e.get("symbol")
        ts = e.get("ts")
        if isinstance(ts, (int, float)):
            # epoch seconds or ms
            ts_s = ts / 1000 if ts > 1e12 else ts
            date = datetime.fromtimestamp(ts_s, tz=timezone.utc).strftime("%Y-%m-%d")
        elif isinstance(ts, str) and ts:
            date = ts[:10]
        else:
            date = None
        if not sym or not date: continue
        feat = e.get("features") or {}
        snap_by_sym_date[(sym, date)] = feat
        n_snaps += 1
print(f"  loaded {n_snaps} snapshots, {len(snap_by_sym_date)} unique (sym,date)")

# ── Pair entries+exits from bot_events ──────────────────────────────────────
print("Pairing trades from bot_events.jsonl...")
entries = []
exits_ = []
with io.open(FILES / "bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        try: e = json.loads(ln)
        except: continue
        et = e.get("event")
        if et == "entry": entries.append(e)
        elif et == "exit": exits_.append(e)
entries.sort(key=lambda e: e.get("ts", ""))
exits_.sort(key=lambda e: e.get("ts", ""))

ent_queue = defaultdict(list)
for e in entries: ent_queue[e.get("sym", "")].append(e)

trades = []
for x in exits_:
    sym = x.get("sym", "")
    q = ent_queue.get(sym, [])
    if not q: continue
    ent = q.pop(0)
    pnl = x.get("pnl_pct")
    if pnl is None: continue
    trades.append({
        "sym": sym,
        "mode": ent.get("mode"),
        "tf": ent.get("tf"),
        "pnl": float(pnl),
        "bars": x.get("bars_held"),
        "entry_ts": ent.get("ts", ""),
        "entry_date": ent.get("ts", "")[:10],
        "is_bull_day_proxy": None,  # filled below if we can find it
    })
print(f"  paired {len(trades)} trades")

# ── Attach TG proba to each trade ───────────────────────────────────────────
print("Scoring TG proba for each trade...")
scored = 0
missing = 0
for t in trades:
    feat = snap_by_sym_date.get((t["sym"], t["entry_date"]))
    if feat is None:
        # try previous day (snapshot often taken at UTC 00:00-ish)
        try:
            from datetime import date, timedelta
            d = date.fromisoformat(t["entry_date"])
            prev = (d - timedelta(days=1)).isoformat()
            feat = snap_by_sym_date.get((t["sym"], prev))
        except: pass
    if feat is None:
        missing += 1
        t["tg20"] = None
        continue
    t["tg20"] = score_tg(feat)
    scored += 1
print(f"  scored: {scored}, missing snapshot: {missing}")

# ── Analysis ────────────────────────────────────────────────────────────────
def summarize(subset, name):
    if not subset:
        print(f"  {name:40s} n=0")
        return None
    n = len(subset)
    pnls = [t["pnl"] for t in subset]
    wins = sum(1 for p in pnls if p > 0)
    total = sum(pnls)
    avg = total / n
    med = sorted(pnls)[n // 2]
    wr = wins / n * 100
    # Sharpe proxy
    mean = avg
    sd = (sum((p - mean) ** 2 for p in pnls) / n) ** 0.5 if n > 1 else 0
    sharpe_proxy = (mean / sd * math.sqrt(n)) if sd > 0 else 0
    print(f"  {name:40s} n={n:>4d}  win={wr:>5.1f}%  avg={avg:+.3f}%  med={med:+.2f}%  sum={total:+7.1f}%  sharpe={sharpe_proxy:+.2f}")
    return {"n": n, "win": wr, "avg": avg, "sum": total, "sharpe": sharpe_proxy}

# Filter: only trades we could score
scored_trades = [t for t in trades if t.get("tg20") is not None]
print(f"\n=== OVERALL (scored only, n={len(scored_trades)}) ===")
summarize(scored_trades, "baseline: all scored trades")

# Only 15m impulse / impulse_speed (the target of proposed gate)
target_modes = {"impulse", "impulse_speed"}
target_trades = [t for t in scored_trades if t["tf"] == "15m" and t["mode"] in target_modes]
print(f"\n=== TARGET: 15m impulse+impulse_speed (n={len(target_trades)}) ===")
summarize(target_trades, "baseline (no TG filter)")

print("\n  TG floor sweep — KEPT subset (entries where tg20 >= threshold):")
for thr in [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    kept = [t for t in target_trades if t["tg20"] >= thr]
    blocked = [t for t in target_trades if t["tg20"] < thr]
    k_stats = summarize(kept, f"thr={thr:.3f} KEPT")
    b_stats = summarize(blocked, f"          BLOCKED")
    if k_stats and b_stats and k_stats["n"] > 0:
        delta_avg = k_stats["avg"] - (sum(t["pnl"] for t in target_trades) / len(target_trades))
        print(f"          -> delta_avg vs baseline: {delta_avg:+.3f}%  (block {len(blocked)}/{len(target_trades)})")

# Also: what about other modes on 15m? (retest, alignment, trend, breakout)
print("\n=== CONTROL: 15m other modes (alignment/trend/retest) ===")
other_trades = [t for t in scored_trades if t["tf"] == "15m" and t["mode"] not in target_modes]
summarize(other_trades, "baseline")
for thr in [0.10, 0.20, 0.30]:
    kept = [t for t in other_trades if t["tg20"] >= thr]
    summarize(kept, f"thr={thr:.2f} kept")

# Per-mode breakdown at chosen threshold (0.10)
print("\n=== PER-MODE at thr=0.10 ===")
by_mode = defaultdict(list)
for t in scored_trades:
    if t["tf"] != "15m": continue
    by_mode[t["mode"]].append(t)
for m, lst in sorted(by_mode.items(), key=lambda kv: -len(kv[1])):
    kept = [t for t in lst if t["tg20"] >= 0.10]
    blocked = [t for t in lst if t["tg20"] < 0.10]
    print(f"  {m}: n={len(lst)}")
    summarize(kept, f"    KEPT (tg>=.10)")
    summarize(blocked, f"    BLOCKED (tg<.10)")
