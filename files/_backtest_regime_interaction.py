"""RM-22 Step B backtest: do the regime-interaction features actually make
the entry bandit rank true top-20 winners better, on held-out days?

Gate before deploy (operator rule 2026-06-01): ship Step B ONLY if this
backtest shows a positive, out-of-sample effect.

Method (offline policy evaluation, no live state touched):
  1. Load top_gainer_dataset.jsonl (ALL watchlist coins x daily snapshots,
     each carries label_top20 + features) — the same primary source
     offline_rl.train_entry_bandit uses.
  2. Temporal split by calendar day: earliest (1 - TEST_FRAC) days = train,
     latest TEST_FRAC days = test. No look-ahead.
  3. Build universal samples exactly like offline_rl (both-arm rewards).
  4. Train TWO fresh LinUCB entry bandits on the train samples:
       - BASELINE: regime-interaction columns zeroed (== old d=18 behaviour)
       - TREATMENT: regime-interaction columns active
  5. On the test days, score each (day,symbol) by the bandit's ENTER
     preference = theta_enter . x  -  theta_skip . x, and measure how well
     that ranks the actual top-20 winners (AUC, plus recall/precision at the
     enter_pref>0 operating point). Also break AUC out by regime cell, since
     the whole point is the bull-day conjunction.

ASCII-only output for cp1251. Read-only. Run from repo root:
    pyembed\python.exe files\_backtest_regime_interaction.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config as _cfg
from contextual_bandit import (
    LinUCBBandit, extract_context, N_ENTRY_ARMS, N_FEATURES,
    _REGIME_INTERACTION_START,
)
import offline_rl
from offline_rl import _load_top_gainer_dataset, _tg_features_to_context

# Allow pointing at the live dataset while running updated worktree code.
import os
from pathlib import Path
_ds = os.environ.get("TG_DATASET")
if _ds:
    offline_rl.TOP_GAINER_DATASET_FILE = Path(_ds)

TEST_FRAC = 0.25  # fraction of most-recent days held out


def _auc(scores, labels):
    """Mann-Whitney AUC. scores: list[float], labels: list[0/1]."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda k: scores[k])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank for ties
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_pos = sum(ranks[k] for k in range(len(scores)) if labels[k] == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _regime_cell(is_bull, btc):
    day = "bull" if is_bull else "flat"
    d = "btcUp" if btc >= 0 else "btcDn"
    return f"{day}/{d}"


def _enter_pref(bandit, x):
    """theta_enter . x  -  theta_skip . x (mean reward gap, ENTER over SKIP)."""
    prefs = []
    for a in range(N_ENTRY_ARMS):
        A_inv = np.linalg.solve(bandit.A[a], np.eye(bandit.d))
        theta = A_inv @ bandit.b[a]
        prefs.append(float(theta @ x))
    return prefs[1] - prefs[0]  # ENTER(1) - SKIP(0)


def main():
    recs = _load_top_gainer_dataset(max_records=200000)
    if not recs:
        print("top_gainer_dataset empty/not found")
        return

    # earliest snapshot per (day,symbol), same as offline_rl default
    by_day_sym = defaultdict(lambda: defaultdict(list))
    for r in recs:
        ts = r.get("ts", 0)
        if not ts:
            continue
        day = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
        by_day_sym[day][r.get("symbol", "")].append(r)

    days = sorted(by_day_sym.keys())
    if len(days) < 6:
        print("not enough days:", len(days))
        return
    n_test = max(1, int(round(len(days) * TEST_FRAC)))
    train_days = set(days[:-n_test])
    test_days = set(days[-n_test:])
    print("=" * 72)
    print("RM-22 Step B backtest: regime-interaction features in entry bandit")
    print("=" * 72)
    print(f"days total={len(days)}  train={len(train_days)} "
          f"({days[0]}..{days[-n_test-1]})  test={len(test_days)} "
          f"({days[-n_test]}..{days[-1]})")

    # Build (x_full, is_top20, is_bull, btc) per (day,sym)
    def build(day_set):
        out = []
        for day in day_set:
            for sym, rs in by_day_sym[day].items():
                rec = min(rs, key=lambda r: r.get("ts", 0))
                feats = rec.get("features", {})
                is_top20 = int(bool(rec.get("label_top20", 0)))
                state, btc = _tg_features_to_context(feats)
                is_bull = btc > 0.3
                x = extract_context(
                    state, mode="trend", tf="15m",
                    is_bull_day=is_bull,
                    market_regime="bull" if is_bull else "neutral",
                    btc_vs_ema50=btc,
                )
                out.append((np.asarray(x, dtype=np.float64), is_top20, is_bull, btc))
        return out

    # ensure treatment features are actually on while we build contexts
    _cfg.BANDIT_REGIME_INTERACTION_ENABLED = True
    train = build(train_days)
    test = build(test_days)
    print(f"train samples={len(train)} (top20={sum(t[1] for t in train)})  "
          f"test samples={len(test)} (top20={sum(t[1] for t in test)})")

    def zero_interaction(x):
        x2 = x.copy()
        x2[_REGIME_INTERACTION_START:] = 0.0
        return x2

    # Train two bandits on identical samples; baseline just sees zeroed cols.
    def train_bandit(use_interaction):
        b = LinUCBBandit(n_arms=N_ENTRY_ARMS, n_features=N_FEATURES,
                         alpha=getattr(_cfg, "BANDIT_ALPHA", 1.5))
        for x, y, _, _ in train:
            xx = x if use_interaction else zero_interaction(x)
            if y == 1:
                b.update(xx, 1, 1.0)
                b.update(xx, 0, -0.8)
            else:
                b.update(xx, 0, 0.10)
                b.update(xx, 1, -0.12)
        return b

    base = train_bandit(False)
    treat = train_bandit(True)

    # Evaluate on held-out test days
    def evaluate(b, use_interaction):
        scores, labels, cells = [], [], []
        for x, y, is_bull, btc in test:
            xx = x if use_interaction else zero_interaction(x)
            scores.append(_enter_pref(b, xx))
            labels.append(y)
            cells.append(_regime_cell(is_bull, btc))
        return scores, labels, cells

    bs, bl, bc = evaluate(base, False)
    ts, tl, tc = evaluate(treat, True)

    auc_base = _auc(bs, bl)
    auc_treat = _auc(ts, tl)

    # operating point: ENTER if enter_pref > 0
    def rec_prec(scores, labels):
        tp = sum(1 for s, y in zip(scores, labels) if s > 0 and y == 1)
        pred_pos = sum(1 for s in scores if s > 0)
        pos = sum(labels)
        recall = tp / pos if pos else float("nan")
        prec = tp / pred_pos if pred_pos else float("nan")
        return recall, prec, pred_pos

    rb, pb, ppb = rec_prec(bs, bl)
    rt, pt, ppt = rec_prec(ts, tl)

    print()
    print("-- held-out ENTER-ranking quality (higher AUC = ranks top20 better) --")
    print(f"  BASELINE  (no interaction): AUC={auc_base:.4f}  "
          f"recall@enter>0={rb:.3f}  precision={pb:.3f}  n_enter={ppb}")
    print(f"  TREATMENT (interaction)   : AUC={auc_treat:.4f}  "
          f"recall@enter>0={rt:.3f}  precision={pt:.3f}  n_enter={ppt}")
    d_auc = auc_treat - auc_base
    print(f"  delta AUC = {d_auc:+.4f}")
    print()

    # per-regime AUC (the conjunction is the whole point)
    print("-- per-regime test AUC --")
    print(f"{'cell':<12}{'n':>6}{'top20':>7}{'AUC_base':>11}{'AUC_treat':>11}{'delta':>9}")
    for cell in sorted(set(tc)):
        idx = [k for k in range(len(tc)) if tc[k] == cell]
        if len(idx) < 10:
            continue
        labs = [tl[k] for k in idx]
        if sum(labs) == 0 or sum(labs) == len(labs):
            continue
        ab = _auc([bs[k] for k in idx], labs)
        at = _auc([ts[k] for k in idx], labs)
        print(f"{cell:<12}{len(idx):>6}{sum(labs):>7}{ab:>11.4f}{at:>11.4f}"
              f"{(at-ab):>+9.4f}")
    print()

    POS = 0.005  # require >=0.5pp AUC gain to call it a win
    if d_auc >= POS:
        print(f"VERDICT: POSITIVE (delta AUC {d_auc:+.4f} >= +{POS}). Step B")
        print("improves out-of-sample ENTER ranking of top-20 winners -> DEPLOY.")
    elif d_auc <= -POS:
        print(f"VERDICT: NEGATIVE (delta AUC {d_auc:+.4f}). Do NOT deploy; keep")
        print("BANDIT_REGIME_INTERACTION_ENABLED=False.")
    else:
        print(f"VERDICT: NEUTRAL (delta AUC {d_auc:+.4f}, |.|<{POS}). No harm but")
        print("no measured gain at the global level; check per-regime cells above.")


if __name__ == "__main__":
    main()
