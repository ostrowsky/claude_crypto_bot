"""Step 5.1: train a REGULARIZED CatBoost proba_fast_reversal model and report an
HONEST train/holdout gap. The stdlib logistic (train_fast_reversal.py) gets val
AUC only 0.531 (signal is nonlinear); unregularized CatBoost overfits (train 0.97
-> holdout 0.63). Question: does the ~0.60 holdout survive regularization (=>
trustworthy, wire as bandit context/reward) or collapse (=> overfit, reward-only)?

Sweeps a small regularization grid, temporal 70/30 split (no lookahead), saves the
best-by-holdout model to fast_reversal_catboost.cbm + a report.

  pyembed\python.exe files\_train_fast_reversal_catboost.py
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier, Pool

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/"files"/"fast_reversal_catboost.cbm"
REPORT = ROOT/"files"/"fast_reversal_catboost_report.json"
FEATS = ["close_vs_ema20", "close_vs_ema50", "close_vs_ema200", "ema20_vs_ema50",
         "ema50_vs_ema200", "slope", "rsi", "adx", "vol_x", "macd_hist_norm",
         "atr_pct", "daily_range", "body_pct", "upper_wick_pct", "lower_wick_pct",
         "btc_vs_ema50", "btc_momentum_4h", "market_vol_24h"]


def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0
    ranks = avg[inv]; npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0: return float("nan")
    return (ranks[y == 1].sum() - npos*(npos+1)/2.0) / (npos*nneg)


rows = []
for ln in io.open(ROOT/"files"/"critic_dataset.jsonl", encoding="utf-8", errors="replace"):
    if '"take"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    if (e.get("decision", {}) or {}).get("action") != "take": continue
    lab = e.get("labels", {}) or {}
    fr = lab.get("label_fast_reversal")
    if fr is None: continue
    f = e.get("f", {}) or {}
    x = [_f(f.get(k)) for k in FEATS]
    if any(v is None for v in x): continue
    rows.append((str(e.get("ts_signal", "")), x + [str(e.get("tf", "?")),
                 str(e.get("signal_type", "?"))], int(fr)))

rows.sort(key=lambda r: r[0])
n = len(rows); cut = int(n*0.70)
cat_idx = [len(FEATS), len(FEATS)+1]
Xtr = [r[1] for r in rows[:cut]]; ytr = [r[2] for r in rows[:cut]]
Xho = [r[1] for r in rows[cut:]]; yho = [r[2] for r in rows[cut:]]
ptr = Pool(Xtr, ytr, cat_features=cat_idx); pho = Pool(Xho, yho, cat_features=cat_idx)
print(f"rows={n} train={cut} holdout={n-cut}  pos: train={sum(ytr)/len(ytr):.1%} "
      f"holdout={sum(yho)/len(yho):.1%}")
print("=" * 60)

grid = [dict(depth=d, iterations=it, learning_rate=lr, l2_leaf_reg=l2)
        for d, it, lr, l2 in [(2, 120, 0.05, 10), (3, 150, 0.04, 12),
                              (3, 200, 0.03, 20), (4, 120, 0.03, 30)]]
best = None
for g in grid:
    m = CatBoostClassifier(loss_function="Logloss", verbose=False, random_seed=42,
                           auto_class_weights="Balanced", **g)
    m.fit(ptr)
    a_tr = auc(ytr, m.predict_proba(ptr)[:, 1]); a_ho = auc(yho, m.predict_proba(pho)[:, 1])
    gap = a_tr - a_ho
    print(f"depth={g['depth']} it={g['iterations']:>3} lr={g['learning_rate']} "
          f"l2={g['l2_leaf_reg']:>2}  train={a_tr:.3f} HOLDOUT={a_ho:.3f} gap={gap:.3f}")
    if best is None or a_ho > best[1]:
        best = (g, a_ho, a_tr, m)
g, a_ho, a_tr, m = best
print("=" * 60)
print(f"BEST holdout AUC={a_ho:.3f} (train={a_tr:.3f}, gap={a_tr-a_ho:.3f})  params={g}")
m.save_model(str(OUT))
json.dump({"holdout_auc": a_ho, "train_auc": a_tr, "params": g, "features": FEATS,
           "n": n, "pos_rate": sum(r[2] for r in rows)/n},
          io.open(REPORT, "w", encoding="utf-8"), indent=2)
print(f"saved -> {OUT.name}, {REPORT.name}")
verdict = ("TRUSTWORTHY (gap small, holdout >=0.58) -> wire as bandit context + reward"
           if (a_ho >= 0.58 and a_tr - a_ho <= 0.12)
           else "WEAK/overfit -> wire REWARD-only, no model gate")
print("VERDICT:", verdict)
