"""RM-22 Step C backtest: turn the entry_score hard-veto into a SOFT gate the
entry bandit decides, and measure the North-Star-aligned effect out of sample.

Decision gate (operator rule 2026-06-01): wire + deploy Step C ONLY if this
shows a positive effect — more top-gainer coverage WITHOUT adding net-losing
entries, and better selectivity than naive "relax everything".

Method (critic_dataset.jsonl, temporal split, read-only):
  1. Split days: earlier (1-TEST_FRAC) = train, latest TEST_FRAC = test.
  2. Train a fresh LinUCB entry bandit on train days (universal both-arm
     scheme, identical to offline_rl). Interaction features per current
     config flag.
  3. Per-day "top gainer" positive = rank by labels.ret_10 desc, top-20 or
     ret_10>=3% (same as offline_rl source-2).
  4. On TEST days compare three policies and their realised ret_5:
       HARD   : entries = records the bot actually took (action=="take")
       RELAX  : HARD + every entry_score-blocked record (no selectivity)
       SOFT   : HARD + entry_score-blocked records the bandit would ENTER
                (enter_pref>0) — the Step C proposal
     Report n, avg ret_5, win%, and top-gainer COVERAGE
     (# day-top records entered / # day-top records total) for each.

ASCII-only. Run from repo root:
    pyembed\python.exe files\_backtest_soft_gate.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config as _cfg
from contextual_bandit import LinUCBBandit, extract_context, N_ENTRY_ARMS, N_FEATURES

DATASET = os.environ.get("CRITIC_DATASET", "critic_dataset.jsonl")
TEST_FRAC = 0.25
TOP_N_PER_DAY = 20
TOP_RET_PCT = 3.0
GATE = "entry_score"


def _f(v, d=0.0):
    try:
        return float(v) if v is not None else d
    except (TypeError, ValueError):
        return d


def _stats(rets):
    n = len(rets)
    if n == 0:
        return {"n": 0, "avg": float("nan"), "win": float("nan"), "sharpe": float("nan")}
    avg = sum(rets) / n
    win = sum(1 for r in rets if r > 0) / n * 100.0
    if n > 1:
        sd = math.sqrt(sum((r - avg) ** 2 for r in rets) / (n - 1))
    else:
        sd = 0.0
    sh = avg / sd * math.sqrt(n) if sd > 1e-9 else 0.0
    return {"n": n, "avg": avg, "win": win, "sharpe": sh}


def _state_from(r):
    f = r.get("f", {}) or {}
    return {
        "slope_pct": _f(f.get("slope")),
        "adx": _f(f.get("adx"), 20.0),
        "rsi": _f(f.get("rsi"), 50.0),
        "vol_x": _f(f.get("vol_x"), 1.0),
        "ml_proba": _f((r.get("decision", {}) or {}).get("ml_proba"), 0.5),
        "daily_range": _f(f.get("daily_range"), 3.0),
        "macd_hist": _f(f.get("macd_hist_norm")),
    }


def _ctx(r):
    f = r.get("f", {}) or {}
    is_bull = bool(r.get("is_bull_day", False))
    btc = _f(f.get("btc_vs_ema50"))
    return extract_context(
        _state_from(r), mode=r.get("signal_type", "trend"), tf=r.get("tf", "15m"),
        is_bull_day=is_bull, market_regime="bull" if is_bull else "neutral",
        btc_vs_ema50=btc,
    )


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
        if day and rec.get("labels", {}).get("ret_10") is not None:
            by_day[day].append(rec)

    days = sorted(by_day.keys())
    if len(days) < 8:
        print("not enough days:", len(days)); return
    n_test = max(1, int(round(len(days) * TEST_FRAC)))
    train_days, test_days = days[:-n_test], days[-n_test:]

    print("=" * 72)
    print("RM-22 Step C backtest: entry_score hard-veto -> bandit soft gate")
    print("=" * 72)
    print(f"days={len(days)} train={len(train_days)} test={len(test_days)} "
          f"({test_days[0]}..{test_days[-1]})  interaction_flag="
          f"{getattr(_cfg,'BANDIT_REGIME_INTERACTION_ENABLED',False)}")

    # mark per-day top-gainers
    def top_ids(recs):
        sc = sorted(recs, key=lambda r: _f(r.get("labels", {}).get("ret_10")), reverse=True)
        ids = set()
        for rank, r in enumerate(sc):
            if rank < TOP_N_PER_DAY or _f(r.get("labels", {}).get("ret_10")) >= TOP_RET_PCT:
                ids.add(id(r))
        return ids

    # train bandit on train days (universal scheme)
    b = LinUCBBandit(N_ENTRY_ARMS, N_FEATURES, alpha=getattr(_cfg, "BANDIT_ALPHA", 1.5))
    for day in train_days:
        recs = by_day[day]
        tids = top_ids(recs)
        for r in recs:
            x = _ctx(r)
            if id(r) in tids:
                b.update(x, 1, 1.0); b.update(x, 0, -0.8)
            else:
                b.update(x, 0, 0.10); b.update(x, 1, -0.12)

    # evaluate on test
    hard_r, relax_r, soft_r = [], [], []
    soft_added_r = []
    blocked_added_r = []  # relax-only additions (all blocked)
    top_total = 0
    top_hard = 0
    top_relax = 0
    top_soft = 0
    soft_added_n = 0
    for day in test_days:
        recs = by_day[day]
        tids = top_ids(recs)
        top_total += len(tids)
        for r in recs:
            dec = r.get("decision", {}) or {}
            action = str(dec.get("action", ""))
            reason = str(dec.get("reason_code", ""))
            ret5 = _f(r.get("labels", {}).get("ret_5"))
            is_top = id(r) in tids
            took = (action == "take")
            blocked_es = (action == "blocked" and reason == GATE)

            if took:
                hard_r.append(ret5); relax_r.append(ret5); soft_r.append(ret5)
                if is_top:
                    top_hard += 1; top_relax += 1; top_soft += 1
            elif blocked_es:
                # RELAX: enter all blocked
                relax_r.append(ret5); blocked_added_r.append(ret5)
                if is_top:
                    top_relax += 1
                # SOFT: bandit decides
                if _enter_pref(b, _ctx(r)) > 0:
                    soft_r.append(ret5); soft_added_r.append(ret5); soft_added_n += 1
                    if is_top:
                        top_soft += 1

    def cov(x):
        return x / top_total * 100.0 if top_total else float("nan")

    h, rl, sf = _stats(hard_r), _stats(relax_r), _stats(soft_r)
    print()
    print("-- entries on test days: realised ret_5 + top-gainer coverage --")
    print(f"{'policy':<8}{'n':>7}{'avg_r5':>9}{'win%':>8}{'sharpe':>9}"
          f"{'top_cov%':>10}")
    print(f"{'HARD':<8}{h['n']:>7}{h['avg']:>9.3f}{h['win']:>8.1f}{h['sharpe']:>9.2f}"
          f"{cov(top_hard):>10.1f}")
    print(f"{'RELAX':<8}{rl['n']:>7}{rl['avg']:>9.3f}{rl['win']:>8.1f}{rl['sharpe']:>9.2f}"
          f"{cov(top_relax):>10.1f}")
    print(f"{'SOFT':<8}{sf['n']:>7}{sf['avg']:>9.3f}{sf['win']:>8.1f}{sf['sharpe']:>9.2f}"
          f"{cov(top_soft):>10.1f}")
    print()
    sa = _stats(soft_added_r)
    ba = _stats(blocked_added_r)
    print(f"entry_score-blocked pool on test: {ba['n']} (avg_r5={ba['avg']:.3f}, "
          f"win%={ba['win']:.1f})")
    print(f"  SOFT admitted {sa['n']} of them (avg_r5={sa['avg']:.3f}, "
          f"win%={sa['win']:.1f}, sharpe={sa['sharpe']:.2f})")
    print(f"  top-gainer coverage: HARD {cov(top_hard):.1f}% -> SOFT {cov(top_soft):.1f}% "
          f"(+{cov(top_soft)-cov(top_hard):.1f}pp), RELAX {cov(top_relax):.1f}%")
    print()

    # verdict: SOFT must raise coverage AND not add net-losing entries AND
    # be more selective than RELAX (higher avg_r5 on its additions).
    cov_gain = cov(top_soft) - cov(top_hard)
    added_ok = (sa["n"] == 0) or (sa["avg"] >= 0.0)
    more_selective = (sa["n"] == 0) or (ba["n"] == 0) or (sa["avg"] >= ba["avg"])
    print("VERDICT:")
    if cov_gain >= 1.0 and added_ok and more_selective:
        print(f"  POSITIVE — SOFT lifts top coverage +{cov_gain:.1f}pp, its added")
        print(f"  entries are net non-negative (avg_r5={sa['avg']:.3f}) and more")
        print(f"  selective than RELAX. Proceed to wire Step C behind a flag.")
    else:
        why = []
        if cov_gain < 1.0:
            why.append(f"coverage gain only +{cov_gain:.1f}pp")
        if not added_ok:
            why.append(f"added entries net-negative (avg_r5={sa['avg']:.3f})")
        if not more_selective:
            why.append("no better than relax-all")
        print(f"  NOT POSITIVE — {'; '.join(why)}. Do not wire/deploy yet.")


if __name__ == "__main__":
    main()
