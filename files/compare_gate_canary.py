"""
compare_gate_canary.py — A/B canary comparison for Scout-gate relaxation.

Compares the LAST 7 days of `take` outcomes against the prior 7 days, after a
gate-config change. Run after at least 7 days post-deploy.

Usage (from repo root):
    pyembed\\python.exe files\\compare_gate_canary.py
    pyembed\\python.exe files\\compare_gate_canary.py --window 5

Outputs:
- avg_ret_5 / win-rate / Sharpe-proxy for each window
- per-reason_code breakdown of how blocked-volume shifted
- explicit verdict: REGRESSION / FLAT / IMPROVEMENT
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

CRITIC = Path(__file__).resolve().parent / "critic_dataset.jsonl"


def _avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _sd(xs):
    if len(xs) < 2:
        return 0.0
    m = _avg(xs)
    return (sum((v - m) ** 2 for v in xs) / (len(xs) - 1)) ** 0.5


def _parse_ts(raw):
    if not raw:
        return None
    s = str(raw).rstrip("Z")
    try:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=7,
                    help="Days per window (before/after).")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    boundary = now - timedelta(days=args.window)
    far = now - timedelta(days=2 * args.window)

    rows_after = []
    rows_before = []
    blocks_after = Counter()
    blocks_before = Counter()

    with CRITIC.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            ts = _parse_ts(r.get("ts_signal"))
            if not ts or ts < far:
                continue
            dec = r.get("decision") or {}
            lab = r.get("labels") or {}
            a = dec.get("action")
            rc = dec.get("reason_code") or ""
            rr = lab.get("ret_5")
            ll = lab.get("label_5")
            bucket_after = ts >= boundary

            if a == "blocked":
                if bucket_after:
                    blocks_after[rc] += 1
                else:
                    blocks_before[rc] += 1
                continue
            if a != "take" or rr is None:
                continue
            row = {"ret5": rr, "win": bool(ll)}
            (rows_after if bucket_after else rows_before).append(row)

    def stats(label, arr):
        if not arr:
            print(f"{label:<30} n=0")
            return None
        rets = [x["ret5"] for x in arr]
        a = _avg(rets)
        s = _sd(rets)
        sh = a / s * (len(arr) ** 0.5) if s > 0 else 0.0
        wr = 100 * sum(x["win"] for x in arr) / len(arr)
        print(f"{label:<30} n={len(arr):<4}  avg_r5={a:+.3f}%  "
              f"win={wr:5.1f}%  Sh*sqN={sh:+.2f}")
        return a, sh

    print(f"Window: {args.window} days each (boundary={boundary.date()})")
    print()
    bs = stats("BEFORE (takes)", rows_before)
    af = stats("AFTER  (takes)", rows_after)

    print()
    print("Blocked volume shift (top reasons):")
    all_keys = set(blocks_before) | set(blocks_after)
    rows = sorted(all_keys, key=lambda k: -(blocks_before[k] + blocks_after[k]))
    for k in rows[:10]:
        b = blocks_before[k]
        a = blocks_after[k]
        delta = a - b
        sign = "+" if delta >= 0 else ""
        print(f"  {k:<28} before={b:<5} after={a:<5} delta={sign}{delta}")

    if bs and af:
        print()
        d_avg = af[0] - bs[0]
        d_sh = af[1] - bs[1]
        if af[0] > bs[0] + 0.05 and af[1] > bs[1]:
            verdict = "IMPROVEMENT"
        elif af[0] < bs[0] - 0.05 or af[1] < bs[1] - 1.0:
            verdict = "REGRESSION (consider rollback)"
        else:
            verdict = "FLAT (need more data)"
        print(f"Verdict: {verdict}   d_avg_r5={d_avg:+.3f}%  d_Sharpe={d_sh:+.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
