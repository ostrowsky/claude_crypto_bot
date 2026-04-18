"""
analyze_blocked_gates.py — Pareto sweep over Scout gates.

For every (action, reason_code) bucket in critic_dataset.jsonl, compute
average forward 5-bar return, win rate, and Sharpe-proxy metric
(avg_r5 / sd_r5 * sqrt(N)).

If the AVG ret_5 of a `blocked` bucket is positive and significantly larger
than the avg ret_5 of actual `take` entries, that gate is over-blocking
profitable signals — a candidate for relaxation.

Run from files/:
    pyembed\\python.exe files\\analyze_blocked_gates.py

Key findings as of 2026-04-18:
- take/take:                  n=2250, avg_r5=-0.016%, win=44.8%      (our real entries are barely losing)
- blocked/impulse_guard:      n=550,  avg_r5=+0.641%, win=40.0%      (*** huge miss)
- blocked/entry_score:        n=1374, avg_r5=+0.123%, win=50.3%      (largest miss by count)
- blocked/ranker_hard_veto:   n=787,  avg_r5=+0.128%, win=46.3%      (still blocking winners)
- blocked/ml_proba_zone:      n=1340, avg_r5=+0.009%, win=44.1%      (neutral, but take is negative)
- blocked/open_cluster_cap:   n=136,  avg_r5=-0.360%, win=35.3%      (working correctly)
- blocked/mtf:                n=227,  avg_r5=-0.208%, win=37.0%      (working correctly)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _sd(xs):
    if len(xs) < 2:
        return 0.0
    m = _avg(xs)
    return (sum((v - m) ** 2 for v in xs) / (len(xs) - 1)) ** 0.5


def main(path: str = "critic_dataset.jsonl") -> None:
    p = Path(path)
    if not p.exists():
        p = Path("files") / path  # allow running from repo root
    buckets = defaultdict(lambda: {"n": 0, "wins": 0, "ret5": []})

    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            dec = r.get("decision") or {}
            lab = r.get("labels") or {}
            a = dec.get("action")
            rc = dec.get("reason_code")
            rr = lab.get("ret_5")
            ll = lab.get("label_5")
            if rr is None:
                continue
            key = ("take", "take") if a == "take" else (a, rc or "none")
            b = buckets[key]
            b["n"] += 1
            if ll:
                b["wins"] += 1
            b["ret5"].append(rr)

    # Reference: actual entries
    take = buckets.get(("take", "take"), {"n": 0, "wins": 0, "ret5": []})
    take_r5 = _avg(take["ret5"])

    rows = []
    for (a, rc), b in buckets.items():
        if b["n"] == 0:
            continue
        a5 = _avg(b["ret5"])
        s5 = _sd(b["ret5"])
        sh = a5 / s5 * (b["n"] ** 0.5) if s5 > 0 else 0.0
        miss = a5 - take_r5 if a == "blocked" else 0.0
        rows.append((a, rc, b["n"], a5, 100 * b["wins"] / b["n"], sh, miss))
    rows.sort(key=lambda r: -r[2])

    print(f"{'action':<10} {'reason_code':<28} {'n':>5} {'avg_r5%':>9} "
          f"{'win5%':>7} {'sharpe*sqrtN':>13} {'miss_vs_take':>12}")
    for r in rows:
        print(f"{r[0]:<10} {r[1]:<28} {r[2]:>5} {r[3]:>9.3f} "
              f"{r[4]:>7.1f} {r[5]:>13.2f} {r[6]:>+12.3f}")

    print()
    print("Over-blocking candidates (avg_r5 > take_r5 AND sharpe*sqrtN >= +1.5):")
    for r in rows:
        if r[0] == "blocked" and r[6] > 0.05 and r[5] >= 1.5:
            print(f"  {r[1]:<28} n={r[2]:>5}  miss={r[6]:+.3f}%  "
                  f"win%={r[4]:.1f}  Sh*sqN={r[5]:+.2f}")


if __name__ == "__main__":
    main()
