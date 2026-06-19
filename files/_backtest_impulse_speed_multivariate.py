"""Final check: is there ANY learnable entry-time signal for impulse_speed?

Three single/linear analyses failed to separate winners from losers. This
throws the FULL feature set at a proper multivariate logistic regression
(numpy GD, L2) with a temporal split, and measures OUT-OF-SAMPLE AUC. If even
this is ~0.5, no entry-time edge exists and the lever is regime/mode-level. If
OOS AUC is meaningfully >0.55 AND a score gate keeps big-mover recall, there is
a real multivariate gate worth building.

Targets: win = ret_5 > 0 (primary). Also reports the admitted subset's realized
pnl and big-mover (ret_10>=5%) recall at the median-score gate.

ASCII-only. Run from repo root:
    pyembed\python.exe files\_backtest_impulse_speed_multivariate.py
"""
from __future__ import annotations
import json, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATASET = "critic_dataset.jsonl"
TEST_FRAC = 0.30
WINNER_RET = 5.0
FEATS = [
    "close_vs_ema20", "close_vs_ema50", "close_vs_ema200",
    "ema20_vs_ema50", "ema50_vs_ema200", "slope", "rsi", "adx", "vol_x",
    "macd_hist_norm", "atr_pct", "daily_range", "body_pct",
    "upper_wick_pct", "lower_wick_pct", "btc_vs_ema50", "btc_momentum_4h",
    "market_vol_24h",
]
DEC_FEATS = ["candidate_score", "ml_proba", "forecast_return_pct",
             "today_change_pct", "mtf_soft_penalty"]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    rows = []
    for ln in open(DATASET, encoding="utf-8", errors="replace"):
        if "impulse_speed" not in ln:
            continue
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if e.get("signal_type") != "impulse_speed":
            continue
        dec = e.get("decision", {}) or {}
        if str(dec.get("action", "")) != "take":
            continue
        lab = e.get("labels", {}) or {}
        r5 = _f(lab.get("ret_5"))
        if r5 is None:
            continue
        f = e.get("f", {}) or {}
        x = [_f(f.get(k)) for k in FEATS] + [_f(dec.get(k)) for k in DEC_FEATS]
        if any(v is None for v in x):
            continue
        rows.append({"day": str(e.get("ts_signal", ""))[:10], "x": x, "r5": r5,
                     "ret10": _f(lab.get("ret_10")),
                     "pnl": _f(lab.get("trade_exit_pnl"))})
    rows.sort(key=lambda r: r["day"])
    days = sorted({r["day"] for r in rows})
    n_test = max(1, int(round(len(days) * TEST_FRAC)))
    test_days = set(days[-n_test:])
    tr = [r for r in rows if r["day"] not in test_days]
    te = [r for r in rows if r["day"] in test_days]

    names = FEATS + DEC_FEATS
    Xtr = np.array([r["x"] for r in tr], float)
    ytr = np.array([1.0 if r["r5"] > 0 else 0.0 for r in tr])
    Xte = np.array([r["x"] for r in te], float)
    yte = np.array([1.0 if r["r5"] > 0 else 0.0 for r in te])

    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
    Xtr_s = (Xtr - mu) / sd
    Xte_s = (Xte - mu) / sd
    Xtr_s = np.hstack([Xtr_s, np.ones((len(Xtr_s), 1))])
    Xte_s = np.hstack([Xte_s, np.ones((len(Xte_s), 1))])

    # logistic regression, GD + L2
    w = np.zeros(Xtr_s.shape[1]); lr = 0.1; lam = 1.0
    for _ in range(3000):
        p = 1.0 / (1.0 + np.exp(-Xtr_s @ w))
        g = Xtr_s.T @ (p - ytr) / len(ytr) + lam * np.r_[w[:-1], 0.0] / len(ytr)
        w -= lr * g

    def auc(scores, labels):
        pos = scores[labels == 1]; neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        order = np.argsort(scores)
        ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(scores)+1)
        return (ranks[labels == 1].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

    s_tr = Xtr_s @ w; s_te = Xte_s @ w
    print("=" * 70)
    print("impulse_speed multivariate logistic — learnable entry signal?")
    print("=" * 70)
    print(f"n={len(rows)}  span {days[0]}..{days[-1]}  train={len(tr)} test={len(te)}")
    print(f"features: {len(names)}  (all f.* + decision numerics)")
    print(f"\nAUC train={auc(s_tr, ytr):.3f}   AUC test(OOS)={auc(s_te, yte):.3f}"
          f"   (0.5 = no signal)")

    # gate at median train score: keep top half by score
    thr = np.median(s_tr)
    keep = s_te >= thr
    nk = int(keep.sum())
    if nk:
        ar5 = np.mean([te[i]["r5"] for i in range(len(te)) if keep[i]])
        pn = [te[i]["pnl"] for i in range(len(te)) if keep[i] and te[i]["pnl"] is not None]
        apnl = np.mean(pn) if pn else float("nan")
        wk = np.mean([1.0 if te[i]["r5"] > 0 else 0.0 for i in range(len(te)) if keep[i]])*100
    big = [i for i in range(len(te)) if (te[i]["ret10"] or 0) >= WINNER_RET]
    kept_big = [i for i in big if keep[i]]
    rec = (len(kept_big)/len(big)*100) if big else float("nan")
    allr5 = np.mean([r["r5"] for r in te])
    print(f"\nGate = keep top-half by model score (OOS):")
    print(f"  ALL test avg_ret5={allr5:+.3f}")
    print(f"  KEEP n={nk} avg_ret5={ar5:+.3f} avg_pnl={apnl:+.3f} win={wk:.0f}%")
    print(f"  big movers(ret_10>={WINNER_RET}%): {len(big)} kept {len(kept_big)} -> recall {rec:.0f}%")

    # top feature weights
    order = np.argsort(-np.abs(w[:-1]))
    print("\nTop weighted features (standardized):")
    for i in order[:8]:
        print(f"  {names[i]:<18} w={w[i]:+.3f}")
    print("\nVERDICT: OOS AUC<=~0.53 => no learnable entry edge (regime/mode lever).")
    print("OOS AUC>0.55 with big-mover recall kept => a real multivariate gate.")


if __name__ == "__main__":
    main()
