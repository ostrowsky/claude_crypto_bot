"""RM-3 step 5/5: 60-day FAST_REVERSAL_PROBA_MAX threshold sweep.

Replays critic_dataset `take` decisions over a window, scores each with
the trained fast_reversal_model, and sweeps the guard threshold to find
the Pareto point: maximum fast-reversal reduction subject to NOT
regressing the winner-recall.

Definitions (outcome-based, computable on real history — the spec's
trail-stop label needs entry_context.atr_pct/trail_k which were never
logged to production, 0/17437 records):
  - fast_reversal (BAD, want to block): labels.ret_3 <= --fr-threshold (%)
  - winner        (GOOD, must keep)   : labels.ret_5 >= --win-threshold (%)

A guard blocking a take with proba > T:
  - correctly avoids a fast-reversal  -> good
  - wrongly drops a winner            -> recall loss (the spec's hard gate)

Verdict: a threshold is ACCEPTABLE only if winner-recall retained
>= --min-recall (default 0.95 = lose <=5% of winners, per
anti-fast-reversal-spec.md secondary metric).

Usage:
    pyembed\\python.exe files\\_backtest_fast_reversal_threshold.py
    pyembed\\python.exe files\\_backtest_fast_reversal_threshold.py --window-days 60
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import fast_reversal_model
import train_fast_reversal as tfr

CRITIC = HERE / "critic_dataset.jsonl"


def _parse_ts(rec: dict) -> Optional[datetime]:
    ts = rec.get("ts_signal")
    if ts:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass
    bt = rec.get("bar_ts")
    if bt:
        try:
            return datetime.fromtimestamp(int(bt) / 1000, tz=timezone.utc)
        except (ValueError, TypeError, OSError):
            pass
    return None


def load_takes(window_days: int) -> List[dict]:
    cut = datetime.now(timezone.utc) - timedelta(days=window_days)
    out = []
    with CRITIC.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (rec.get("decision") or {}).get("action") != "take":
                continue
            lab = rec.get("labels") or {}
            if lab.get("ret_3") is None or lab.get("ret_5") is None:
                continue
            t = _parse_ts(rec)
            if t is None or t < cut:
                continue
            out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="RM-3 fast-reversal threshold sweep")
    ap.add_argument("--window-days", type=int, default=60)
    ap.add_argument("--fr-threshold", type=float, default=-0.3,
                    help="ret_3 %% at/below = fast-reversal (default -0.3)")
    ap.add_argument("--win-threshold", type=float, default=1.0,
                    help="ret_5 %% at/above = winner to protect (default 1.0)")
    ap.add_argument("--min-recall", type=float, default=0.95,
                    help="min winner-recall to call a threshold acceptable")
    ap.add_argument("--model", type=Path, default=HERE / "fast_reversal_model.json")
    args = ap.parse_args()

    if not args.model.exists():
        print(f"[!] model {args.model} not found — run train_fast_reversal.py first")
        return 1
    # point the inference module at the requested model
    fast_reversal_model.MODEL_FILE = args.model
    fast_reversal_model._MODEL_CACHE = None
    fast_reversal_model._MODEL_LOADED_MTIME = 0.0
    if not fast_reversal_model.is_available():
        print("[!] model failed to load")
        return 1

    takes = load_takes(args.window_days)
    print(f"[*] window={args.window_days}d  takes with ret_3&ret_5: {len(takes)}")
    if len(takes) < 100:
        print("[!] too few takes for a meaningful sweep")
        return 1

    fr_thr = args.fr_threshold
    win_thr = args.win_threshold
    scored = []
    n_fr = n_win = 0
    for rec in takes:
        feats = {n: v for n, v in zip(tfr.FEATURE_NAMES, tfr.extract_features(rec))}
        proba = fast_reversal_model.predict_proba(feats)
        if proba is None:
            continue
        lab = rec["labels"]
        r3 = float(lab["ret_3"])
        r5 = float(lab["ret_5"])
        is_fr = r3 <= fr_thr
        is_win = r5 >= win_thr
        n_fr += is_fr
        n_win += is_win
        scored.append((proba, is_fr, is_win, r5))

    base_fr_rate = 100.0 * n_fr / len(scored)
    print(f"[*] baseline: fast-reversal {n_fr}/{len(scored)} ({base_fr_rate:.1f}%), "
          f"winners {n_win} ({100*n_win/len(scored):.1f}%)")
    print()
    print(f"  {'T':>5} {'blocked':>8} {'FR_avoid':>9} {'win_lost':>9} "
          f"{'FR_rate':>8} {'win_recall':>10} {'net_r5':>9}  verdict")
    print("  " + "-" * 78)

    best = None
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        blocked = [s for s in scored if s[0] > t]
        kept = [s for s in scored if s[0] <= t]
        fr_avoided = sum(1 for s in blocked if s[1])
        win_lost = sum(1 for s in blocked if s[2])
        kept_fr = sum(1 for s in kept if s[1])
        new_fr_rate = 100.0 * kept_fr / len(kept) if kept else 0.0
        win_recall = (n_win - win_lost) / n_win if n_win else 1.0
        # net effect on protected-winner ret_5 we would forgo by blocking
        net_r5 = -sum(s[3] for s in blocked if s[2]) + sum(
            -s[3] for s in blocked if (not s[2] and s[3] < 0))
        ok = win_recall >= args.min_recall and new_fr_rate < base_fr_rate
        verdict = "OK" if ok else "reject"
        print(f"  {t:>5.2f} {len(blocked):>8d} {fr_avoided:>9d} {win_lost:>9d} "
              f"{new_fr_rate:>7.1f}% {win_recall:>9.1%} {net_r5:>+8.1f}  {verdict}")
        if ok:
            # prefer the threshold maximising FR avoided while OK
            if best is None or fr_avoided > best[1]:
                best = (t, fr_avoided, win_recall, new_fr_rate)

    print()
    if best:
        t, fra, wr, nfr = best
        print(f"[+] Pareto point: FAST_REVERSAL_PROBA_MAX = {t:.2f}")
        print(f"    avoids {fra} fast-reversals, winner-recall {wr:.1%} "
              f"(>= {args.min_recall:.0%}), FR-rate {base_fr_rate:.1f}% -> {nfr:.1f}%")
        print(f"[verdict] guard CAN be staged at T={t:.2f} (still SHADOW until "
              f"7d live confirms; flip FAST_REVERSAL_GUARD_ENABLED only after)")
    else:
        print("[verdict] NO acceptable threshold — model too weak "
              f"(val AUC ~0.55) to gate without losing >{1-args.min_recall:.0%} "
              f"of winners. Keep FAST_REVERSAL_GUARD_ENABLED=False; use "
              f"proba only as a bandit-context feature (RM-3 step 4), not a hard guard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
