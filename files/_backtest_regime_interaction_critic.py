"""RM-22 Step B backtest (critic_dataset variant).

The top_gainer_dataset source derives is_bull_day from BTC sign, which makes
the bull_day x BTC-trend interaction degenerate (bull implies btc>0). The LIVE
bandit instead receives is_bull_day from config._bull_day_active, INDEPENDENT
of btc_vs_ema50 sign — so the conjunction is real (Step A found a populated
bull_day/btc_dn cell here). This backtest therefore evaluates Step B on
critic_dataset.jsonl, where both flags are logged independently.

Label: per-day "top gainer" defined like offline_rl source-2 — rank the day's
records by labels.ret_10 desc, top-20 or ret_10>=3% are positives.

Offline policy evaluation: train two fresh LinUCB entry bandits on the train
split (baseline = interaction cols zeroed; treatment = active), evaluate ENTER
preference ranking of positives on held-out later days. Read-only.

Run from repo root:
    pyembed\python.exe files\_backtest_regime_interaction_critic.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config as _cfg
from contextual_bandit import (
    LinUCBBandit, extract_context, N_ENTRY_ARMS, N_FEATURES,
    _REGIME_INTERACTION_START,
)

DATASET = os.environ.get("CRITIC_DATASET", "critic_dataset.jsonl")
TEST_FRAC = 0.25
TOP_N_PER_DAY = 20
TOP_RET_PCT = 3.0


def _f(v, d=0.0):
    try:
        return float(v) if v is not None else d
    except (TypeError, ValueError):
        return d


def _auc(scores, labels):
    pos_n = sum(labels)
    neg_n = len(labels) - pos_n
    if pos_n == 0 or neg_n == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda k: scores[k])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(ranks[k] for k in range(len(scores)) if labels[k] == 1)
    return (sum_pos - pos_n * (pos_n + 1) / 2.0) / (pos_n * neg_n)


def _cell(is_bull, btc):
    return f"{'bull' if is_bull else 'flat'}/{'btcUp' if btc >= 0 else 'btcDn'}"


def _enter_pref(b, x):
    out = []
    for a in range(N_ENTRY_ARMS):
        A_inv = np.linalg.solve(b.A[a], np.eye(b.d))
        out.append(float((A_inv @ b.b[a]) @ x))
    return out[1] - out[0]


def main():
    if not os.path.exists(DATASET):
        print("dataset not found:", DATASET)
        return

    by_day = defaultdict(list)
    for line in open(DATASET, "r", encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        day = str(rec.get("ts_signal", ""))[:10]
        ret10 = rec.get("labels", {}).get("ret_10")
        if not day or ret10 is None:
            continue
        by_day[day].append(rec)

    days = sorted(by_day.keys())
    if len(days) < 8:
        print("not enough labelled days:", len(days))
        return
    n_test = max(1, int(round(len(days) * TEST_FRAC)))
    train_days = days[:-n_test]
    test_days = days[-n_test:]

    print("=" * 72)
    print("RM-22 Step B backtest (critic_dataset, independent bull-day flag)")
    print("=" * 72)
    print(f"days={len(days)} train={len(train_days)} ({days[0]}..{days[-n_test-1]}) "
          f"test={len(test_days)} ({days[-n_test]}..{days[-1]})")

    _cfg.BANDIT_REGIME_INTERACTION_ENABLED = True

    def build(day_list):
        out = []
        for day in day_list:
            recs = by_day[day]
            scored = sorted(recs, key=lambda r: _f(r.get("labels", {}).get("ret_10")),
                            reverse=True)
            top_ids = set()
            for rank, r in enumerate(scored):
                if rank < TOP_N_PER_DAY or _f(r.get("labels", {}).get("ret_10")) >= TOP_RET_PCT:
                    top_ids.add(id(r))
            for r in recs:
                f = r.get("f", {}) or {}
                is_bull = bool(r.get("is_bull_day", False))
                btc = _f(f.get("btc_vs_ema50"))
                state = {
                    "slope_pct": _f(f.get("slope")),
                    "adx": _f(f.get("adx"), 20.0),
                    "rsi": _f(f.get("rsi"), 50.0),
                    "vol_x": _f(f.get("vol_x"), 1.0),
                    "ml_proba": _f((r.get("decision", {}) or {}).get("ml_proba"), 0.5),
                    "daily_range": _f(f.get("daily_range"), 3.0),
                    "macd_hist": _f(f.get("macd_hist_norm")),
                }
                x = extract_context(
                    state, mode=r.get("signal_type", "trend"), tf=r.get("tf", "15m"),
                    is_bull_day=is_bull,
                    market_regime="bull" if is_bull else "neutral",
                    btc_vs_ema50=btc,
                )
                out.append((np.asarray(x, np.float64), int(id(r) in top_ids), is_bull, btc))
        return out

    train = build(train_days)
    test = build(test_days)
    print(f"train n={len(train)} (top={sum(t[1] for t in train)})  "
          f"test n={len(test)} (top={sum(t[1] for t in test)})")
    # confirm the interaction cells are actually populated now
    cnt = defaultdict(int)
    for _, _, ib, bt in train + test:
        cnt[_cell(ib, bt)] += 1
    print("regime-cell coverage:", dict(cnt))

    def zero_int(x):
        x2 = x.copy(); x2[_REGIME_INTERACTION_START:] = 0.0
        return x2

    def train_bandit(use_int):
        b = LinUCBBandit(N_ENTRY_ARMS, N_FEATURES, alpha=getattr(_cfg, "BANDIT_ALPHA", 1.5))
        for x, y, _, _ in train:
            xx = x if use_int else zero_int(x)
            if y == 1:
                b.update(xx, 1, 1.0); b.update(xx, 0, -0.8)
            else:
                b.update(xx, 0, 0.10); b.update(xx, 1, -0.12)
        return b

    base = train_bandit(False)
    treat = train_bandit(True)

    def ev(b, use_int):
        s, l, c = [], [], []
        for x, y, ib, bt in test:
            xx = x if use_int else zero_int(x)
            s.append(_enter_pref(b, xx)); l.append(y); c.append(_cell(ib, bt))
        return s, l, c

    bs, bl, bc = ev(base, False)
    ts, tl, tc = ev(treat, True)
    ab, at = _auc(bs, bl), _auc(ts, tl)
    print()
    print("-- held-out ENTER-ranking AUC (top gainer = positive) --")
    print(f"  BASELINE  : AUC={ab:.4f}")
    print(f"  TREATMENT : AUC={at:.4f}")
    print(f"  delta AUC = {at-ab:+.4f}")
    print()
    print("-- per-regime test AUC --")
    print(f"{'cell':<12}{'n':>6}{'top':>6}{'AUC_base':>11}{'AUC_treat':>11}{'delta':>9}")
    for cell in sorted(set(tc)):
        idx = [k for k in range(len(tc)) if tc[k] == cell]
        labs = [tl[k] for k in idx]
        if len(idx) < 10 or sum(labs) in (0, len(labs)):
            print(f"{cell:<12}{len(idx):>6}{sum(labs):>6}   (too few/degenerate)")
            continue
        a_b = _auc([bs[k] for k in idx], labs)
        a_t = _auc([ts[k] for k in idx], labs)
        print(f"{cell:<12}{len(idx):>6}{sum(labs):>6}{a_b:>11.4f}{a_t:>11.4f}{(a_t-a_b):>+9.4f}")
    print()
    POS = 0.005
    d = at - ab
    if d >= POS:
        print(f"VERDICT: POSITIVE (delta {d:+.4f} >= +{POS}) -> DEPLOY Step B")
    elif d <= -POS:
        print(f"VERDICT: NEGATIVE (delta {d:+.4f}) -> do NOT deploy")
    else:
        print(f"VERDICT: NEUTRAL (delta {d:+.4f}); inspect per-regime cells")


if __name__ == "__main__":
    main()
