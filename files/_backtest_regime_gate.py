"""RM-22 Shadow Analysis (Step A): is the optimal entry_score threshold
regime-dependent?

The user's thesis: static gate thresholds can never track a non-stationary
market; the bot should pick the threshold from the market regime. Before
re-architecting the pipeline (turning the entry_score hard-veto into a soft,
regime-conditioned feature) we validate the premise on data we already
collect.

We use `critic_dataset.jsonl`. Every blocked `entry_score` event carries the
candidate_score, the static score_floor, the market regime proxies
(is_bull_day, btc_vs_ema50, market_vol_24h) and, filled at EOD, the realised
forward return `labels.ret_5`. If the static floor were correct, blocked
candidates should have NEGATIVE forward returns in EVERY regime. If instead
the blocked-bucket return flips sign across regimes, a single static threshold
is provably wrong and a regime-conditional gate is justified.

Output is ASCII-only for cp1251 consoles. Read-only; writes nothing.

Run from repo root:
    pyembed\python.exe files\_backtest_regime_gate.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "critic_dataset.jsonl")

GATE = sys.argv[1] if len(sys.argv) > 1 else "entry_score"


def _f(v, default=0.0):
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _regime_key(rec):
    """Coarse market-regime bucket from already-logged proxies."""
    is_bull = bool(rec.get("is_bull_day", False))
    f = rec.get("f", {}) or {}
    btc = _f(f.get("btc_vs_ema50"))
    btc_dir = "btc_up" if btc >= 0 else "btc_dn"
    day = "bull_day" if is_bull else "flat_day"
    return f"{day}/{btc_dir}"


def _stats(rets):
    n = len(rets)
    if n == 0:
        return None
    mean = sum(rets) / n
    if n > 1:
        var = sum((r - mean) ** 2 for r in rets) / (n - 1)
        sd = math.sqrt(var)
    else:
        sd = 0.0
    win = sum(1 for r in rets if r > 0) / n * 100.0
    # Sharpe-proxy used elsewhere in the repo: avg / sd * sqrt(n)
    sharpe = (mean / sd * math.sqrt(n)) if sd > 1e-9 else 0.0
    return {"n": n, "avg": mean, "sd": sd, "win": win, "sharpe": sharpe}


def main():
    if not os.path.exists(DATASET):
        print("dataset not found:", DATASET)
        return

    # take baseline (what entered) per regime, plus blocked-by-GATE per regime
    take_by_regime = defaultdict(list)
    blocked_by_regime = defaultdict(list)
    # also bucket blocked events by score-deficit band, per regime, to see
    # where a regime-aware floor would sit
    deficit_by_regime = defaultdict(lambda: defaultdict(list))

    total = 0
    blocked_gate = 0
    for line in open(DATASET, "r", encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += 1
        dec = rec.get("decision", {}) or {}
        labels = rec.get("labels", {}) or {}
        ret5 = labels.get("ret_5")
        if ret5 is None:
            continue
        ret5 = _f(ret5)
        action = str(dec.get("action", ""))
        regime = _regime_key(rec)

        if action == "take":
            take_by_regime[regime].append(ret5)
            continue

        if action == "blocked" and str(dec.get("reason_code", "")) == GATE:
            blocked_gate += 1
            blocked_by_regime[regime].append(ret5)
            cand = _f(dec.get("candidate_score"))
            floor = _f(dec.get("score_floor"))
            deficit = floor - cand  # how far below the static floor
            if deficit <= 2:
                band = "0-2 below"
            elif deficit <= 5:
                band = "2-5 below"
            elif deficit <= 10:
                band = "5-10 below"
            else:
                band = ">10 below"
            deficit_by_regime[regime][band].append(ret5)

    print("=" * 72)
    print(f"RM-22 regime-gate shadow analysis  |  gate = {GATE}")
    print("=" * 72)
    print(f"records scanned (with ret_5): contributing")
    print(f"blocked by '{GATE}' with realised ret_5: {blocked_gate}")
    print()

    regimes = sorted(set(list(take_by_regime) + list(blocked_by_regime)))

    print("-- per-regime: TAKE baseline vs BLOCKED-by-gate (forward ret_5 %) --")
    print(f"{'regime':<18}{'take_n':>7}{'take_avg':>10}{'blk_n':>7}"
          f"{'blk_avg':>10}{'blk_win%':>10}{'blk_shrp':>10}  verdict")
    flip_detected = False
    blk_avgs = {}
    for rg in regimes:
        ts = _stats(take_by_regime.get(rg, []))
        bs = _stats(blocked_by_regime.get(rg, []))
        take_avg = ts["avg"] if ts else float("nan")
        if bs is None:
            continue
        blk_avgs[rg] = bs["avg"]
        # over-blocking in this regime if blocked candidates were profitable
        # AND beat what we actually took
        over = bs["avg"] > 0 and bs["avg"] > take_avg
        verdict = "OVER-BLOCK" if over else "ok"
        if over:
            flip_detected = True
        print(f"{rg:<18}{(ts['n'] if ts else 0):>7}{take_avg:>10.3f}"
              f"{bs['n']:>7}{bs['avg']:>10.3f}{bs['win']:>10.1f}"
              f"{bs['sharpe']:>10.2f}  {verdict}")

    print()
    if len(blk_avgs) >= 2:
        lo = min(blk_avgs.values())
        hi = max(blk_avgs.values())
        spread = hi - lo
        print(f"blocked-bucket avg_r5 spread across regimes: {spread:.3f} pp "
              f"(min {lo:.3f} / max {hi:.3f})")
        if spread > 0.20:
            print(">> Spread is material: the SAME static floor produces very")
            print("   different forward returns depending on regime. A single")
            print("   threshold cannot be right in all of them -> regime-")
            print("   conditional gating (RM-22) is justified by data.")
        else:
            print(">> Spread is small: regime does not strongly move the gate's")
            print("   value here; static floor is defensible for this gate.")
    print()

    print("-- blocked-by-gate broken down by score-deficit band, per regime --")
    print("   (shows where a regime-aware floor would actually sit)")
    for rg in regimes:
        bands = deficit_by_regime.get(rg)
        if not bands:
            continue
        print(f"  [{rg}]")
        for band in ["0-2 below", "2-5 below", "5-10 below", ">10 below"]:
            st = _stats(bands.get(band, []))
            if not st:
                continue
            tag = "would-keep-blocked" if st["avg"] <= 0 else "PROFITABLE->relax"
            print(f"    {band:<12} n={st['n']:>5}  avg_r5={st['avg']:>8.3f}"
                  f"  win%={st['win']:>5.1f}   {tag}")
    print()
    if flip_detected:
        print("CONCLUSION: at least one regime shows the gate blocking")
        print("profitable candidates that beat live entries. The static gate is")
        print("leaving money on the table in that regime. Next RM-22 step: turn")
        print("entry_score into a soft, regime-conditioned signal for the entry")
        print("bandit instead of a hard veto (shadow-log first, then backtest).")
    else:
        print("CONCLUSION: no regime shows clear over-blocking on this gate.")


if __name__ == "__main__":
    main()
