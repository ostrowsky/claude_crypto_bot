"""Step 4: OOS learnability of the two new learning labels (false-entry /
false-early-exit). Before wiring them into the bandit reward (step 5), check
whether they are PREDICTABLE at entry time from available features. Prior
findings on this bot warn that entry-time winner/loser separation is often
OOS AUC ~0.50 (non-stationary) -> a model can't avoid them, but the reward can
still reflect the true cost. This quantifies which case we're in.

Method: take records carrying each label; 18 entry-time features from `f` + mode;
TEMPORAL split (first 70% train / last 30% holdout, no shuffle = no lookahead);
CatBoost classifier; report train vs holdout AUC. holdout AUC >= 0.55 => gateable
(wire as model/guard); ~0.50 => reward-honesty only (penalize outcome, don't gate).

Read-only.  pyembed\python.exe files\_backtest_label_learnability.py
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier, Pool


def roc_auc_score(y, s):
    y = np.asarray(y); s = np.asarray(s)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float); ranks[order] = np.arange(1, len(s)+1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0: return float("nan")
    return (ranks[y == 1].sum() - npos*(npos+1)/2.0) / (npos*nneg)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
FEATS = ["close_vs_ema20", "close_vs_ema50", "close_vs_ema200", "ema20_vs_ema50",
         "ema50_vs_ema200", "slope", "rsi", "adx", "vol_x", "macd_hist_norm",
         "atr_pct", "daily_range", "body_pct", "upper_wick_pct", "lower_wick_pct",
         "btc_vs_ema50", "btc_momentum_4h", "market_vol_24h"]


def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


rows = []
for ln in io.open(ROOT/"files"/"critic_dataset.jsonl", encoding="utf-8", errors="replace"):
    if '"take"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    if (e.get("decision", {}) or {}).get("action") != "take": continue
    lab = e.get("labels", {}) or {}
    fr = lab.get("label_fast_reversal"); pe = lab.get("label_premature_exit")
    if fr is None and pe is None: continue
    f = e.get("f", {}) or {}
    x = [_f(f.get(k)) for k in FEATS]
    if any(v is None for v in x): continue
    rows.append((str(e.get("ts_signal", "")), x, str(e.get("tf", "?")),
                 str(e.get("signal_type", "?")), fr, pe))

rows.sort(key=lambda r: r[0])
print(f"labeled take rows usable: {len(rows)}")


def evaluate(label_name, idx):
    data = [(r[1] + [r[2], r[3]], r[idx]) for r in rows if r[idx] is not None]
    n = len(data)
    if n < 200:
        print(f"\n[{label_name}] too few ({n}) — skip"); return
    cut = int(n * 0.70)
    Xtr = [d[0] for d in data[:cut]]; ytr = [int(d[1]) for d in data[:cut]]
    Xho = [d[0] for d in data[cut:]]; yho = [int(d[1]) for d in data[cut:]]
    cat_idx = [len(FEATS), len(FEATS) + 1]   # tf, mode
    ptr = Pool(Xtr, ytr, cat_features=cat_idx); pho = Pool(Xho, yho, cat_features=cat_idx)
    m = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
                           loss_function="Logloss", verbose=False,
                           random_seed=42, auto_class_weights="Balanced")
    m.fit(ptr)
    def auc(P, y):
        if len(set(y)) < 2: return float("nan")
        return roc_auc_score(y, m.predict_proba(P)[:, 1])
    base_tr = sum(ytr)/len(ytr); base_ho = sum(yho)/len(yho)
    print(f"\n[{label_name}]  n={n}  train={cut} holdout={n-cut}")
    print(f"  positive rate: train={base_tr:.1%}  holdout={base_ho:.1%}")
    print(f"  AUC: train={auc(ptr,ytr):.3f}   HOLDOUT={auc(pho,yho):.3f}")
    imp = sorted(zip(FEATS+["tf","mode"], m.get_feature_importance(ptr)),
                 key=lambda t: -t[1])[:6]
    print("  top features:", ", ".join(f"{k}={v:.0f}" for k, v in imp))


evaluate("label_fast_reversal (false entry)", 4)
evaluate("label_premature_exit (false early exit)", 5)
print("\nVERDICT: holdout AUC >= 0.55 => gateable (build guard/model + reward).")
print("~0.50 => not predictable at entry; wire as REWARD-only (honest cost), do")
print("NOT add a hard guard (it would over-block, per the §7 non-stationarity rule).")
