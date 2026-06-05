"""A/B: does adding an EXTENSION feature (+ impulse_speed interaction) to the
entry-bandit context help it skip late/extended impulse_speed losers WITHOUT a
hard gate (which over-blocks winners)?

RM-22 philosophy: don't hard-gate a non-stationary edge — let the regime-aware
bandit learn the conditional. The static extension gate failed (over-blocked
~70% of big movers). The bandit can combine extension with other features and
the impulse_speed flag, so it may isolate the losing-extension cases.

Method (critic_dataset, full history, temporal split, read-only):
  baseline ctx  = the live 20-dim bandit context (replicated here)
  treatment ctx = baseline + [ext_norm, impulse_speed * ext_norm]   (22-dim)
    ext = close_vs_ema50 (primary extension signal from the discriminator search)
  Train a LinUCB 2-arm entry bandit on TRAIN days (universal both-arm scheme,
  winner = ret_5 > 0). On TEST, for impulse_speed entries compare:
    - AUC(enter_pref, win)              -> ranking quality
    - admitted set (enter_pref>0): n, avg ret_5, avg realized pnl, win%
  Also report OVERALL (all modes) admitted set to confirm no collateral damage.

Deploy only if treatment raises impulse_speed AUC AND admitted avg (ret_5/pnl)
without lowering the overall admitted average.

ASCII-only. Run from repo root:
    pyembed\python.exe files\_backtest_bandit_extension_feature.py
"""
from __future__ import annotations
import json, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")
from contextual_bandit import LinUCBBandit

DATASET = "critic_dataset.jsonl"
TEST_FRAC = 0.30
ALPHA = 1.5


def _f(v, d=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def base_ctx(f, dec, is_bull, mode, tf):
    slope = _f(f.get("slope"), 0.0); adx = _f(f.get("adx"), 20.0)
    rsi = _f(f.get("rsi"), 50.0); vol_x = _f(f.get("vol_x"), 1.0)
    ml = _f(dec.get("ml_proba"), 0.5); btc = _f(f.get("btc_vs_ema50"), 0.0)
    dr = _f(f.get("daily_range"), 3.0); macd = _f(f.get("macd_hist_norm"), 0.0)
    is_imp = mode in ("impulse", "impulse_speed")
    return [
        np.clip(slope/0.5, -2, 2), np.clip(adx/50.0, 0, 1),
        np.clip(rsi/100.0, 0, 1), np.clip(vol_x/3.0, 0, 2),
        np.clip(ml, 0, 1), np.clip(btc/5.0, -2, 2),
        np.clip(dr/10.0, 0, 2),
        1.0 if macd > 0 else (-1.0 if macd < 0 else 0.0),
        1.0 if is_bull else 0.0,
        1.0 if is_bull else 0.0,   # regime_bull proxy
        0.0,                        # regime_bear
        1.0 if tf == "1h" else 0.0,
        1.0 if mode in ("trend", "strong_trend") else 0.0,
        1.0 if mode == "retest" else 0.0,
        1.0 if mode == "breakout" else 0.0,
        1.0 if is_imp else 0.0,
        1.0 if mode == "alignment" else 0.0,
        1.0,                        # bias
        0.0, 0.0,                   # regime interaction (inert)
    ]


def main():
    rows = []
    for ln in open(DATASET, encoding="utf-8", errors="replace"):
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            continue
        dec = e.get("decision", {}) or {}
        if str(dec.get("action", "")) not in ("take", "blocked"):
            continue
        lab = e.get("labels", {}) or {}
        r5 = _f(lab.get("ret_5"))
        if r5 is None:
            continue
        f = e.get("f", {}) or {}
        mode = e.get("signal_type", "trend"); tf = e.get("tf", "15m")
        ext = _f(f.get("close_vs_ema50"), 0.0)
        rows.append({
            "day": str(e.get("ts_signal", ""))[:10],
            "r5": r5, "pnl": _f(lab.get("trade_exit_pnl")),
            "mode": mode,
            "base": base_ctx(f, dec, bool(e.get("is_bull_day", False)), mode, tf),
            "ext_norm": float(np.clip(ext/10.0, -2, 2)),
            "is_imp_speed": mode == "impulse_speed",
        })
    rows.sort(key=lambda r: r["day"])
    days = sorted({r["day"] for r in rows})
    n_test = max(1, int(round(len(days) * TEST_FRAC)))
    test_days = set(days[-n_test:])
    train = [r for r in rows if r["day"] not in test_days]
    test = [r for r in rows if r["day"] in test_days]

    print("=" * 74)
    print("Bandit extension-feature A/B (entry bandit, impulse_speed focus)")
    print("=" * 74)
    print(f"records n={len(rows)}  span {days[0]}..{days[-1]}  "
          f"train={len(train)} test={len(test)}  "
          f"impulse_speed test={sum(1 for r in test if r['is_imp_speed'])}")

    def vec(r, treat):
        v = list(r["base"])
        if treat:
            imp_s = 1.0 if r["is_imp_speed"] else 0.0
            v += [r["ext_norm"], imp_s * r["ext_norm"]]
        return np.array(v, dtype=np.float64)

    def train_bandit(treat):
        d = 22 if treat else 20
        b = LinUCBBandit(2, d, alpha=ALPHA)
        for r in train:
            x = vec(r, treat)
            win = r["r5"] > 0
            if win:
                b.update(x, 1, 1.0); b.update(x, 0, -0.8)
            else:
                b.update(x, 1, -0.12); b.update(x, 0, 0.10)
        return b

    def enter_pref(b, x):
        out = []
        for a in (0, 1):
            A_inv = np.linalg.solve(b.A[a], np.eye(b.d))
            out.append(float((A_inv @ b.b[a]) @ x))
        return out[1] - out[0]

    def auc(scores, labels):
        pos = [s for s, l in zip(scores, labels) if l]
        neg = [s for s, l in zip(scores, labels) if not l]
        if not pos or not neg:
            return float("nan")
        c = sum((s > t) + 0.5 * (s == t) for s in pos for t in neg)
        return c / (len(pos) * len(neg))

    def evalset(b, treat, subset):
        prefs = [enter_pref(b, vec(r, treat)) for r in subset]
        labs = [r["r5"] > 0 for r in subset]
        a = auc(prefs, labs)
        adm = [r for r, p in zip(subset, prefs) if p > 0]
        n = len(adm)
        ar5 = sum(r["r5"] for r in adm)/n if n else float("nan")
        pnls = [r["pnl"] for r in adm if r["pnl"] is not None]
        apnl = sum(pnls)/len(pnls) if pnls else float("nan")
        win = sum(1 for r in adm if r["r5"] > 0)/n*100 if n else float("nan")
        return a, n, ar5, apnl, win

    imp_test = [r for r in test if r["is_imp_speed"]]
    for name, treat in (("BASELINE (20-dim)", False), ("TREATMENT +ext (22-dim)", True)):
        b = train_bandit(treat)
        ai, ni, r5i, pnli, wi = evalset(b, treat, imp_test)
        ao, no, r5o, pnlo, wo = evalset(b, treat, test)
        print(f"\n{name}")
        print(f"  impulse_speed: AUC={ai:.3f}  admitted n={ni}  "
              f"avg_ret5={r5i:+.3f}  avg_pnl={pnli:+.3f}  win={wi:.0f}%")
        print(f"  ALL modes:     AUC={ao:.3f}  admitted n={no}  "
              f"avg_ret5={r5o:+.3f}  avg_pnl={pnlo:+.3f}  win={wo:.0f}%")
    print("\nVERDICT: deploy only if TREATMENT raises impulse_speed AUC AND its")
    print("admitted avg_ret5/avg_pnl, while ALL-modes admitted avg does not drop.")


if __name__ == "__main__":
    main()
