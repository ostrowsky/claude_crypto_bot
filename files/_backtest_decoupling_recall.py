"""Translate the decoupling edge into a concrete answer:
of REAL watchlist daily top-gainers, how many carried the
'high-vol + decoupled' flag at DAY-START (trailing data only, no lookahead)?

This is the recall/precision of the flag as a top-gainer detector — the
direct "bot could have caught N of M" number the operator asked for.

Definitions (no lookahead — flag uses only bars strictly before the day):
  - market basket = mean hourly return across all watchlist coins
  - trailing_corr = coin's 7d (168h) corr to basket, ending at day-start
  - vol_pctile    = cross-sectional rank of coin's trailing 7d realized vol
                    among all watchlist coins that day
  - FLAG = (vol_pctile >= VOLQ) AND (trailing_corr <= CORRMAX)
  - top-gainer day = coin in that day's top-K by daily close-to-close return
    (also reported: absolute daily return >= BIGDAY)

Period: max available 365d × 1h watchlist klines, train/holdout split.
Research-only.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
HISTORY_DIR = ROOT / "history"
WATCHLIST_FILE = ROOT / "files" / "watchlist.json"


def load_aligned(syms, suffix="_1h_365d", min_bars=4000):
    closes = {}
    for s in syms:
        p = HISTORY_DIR / f"{s}{suffix}.csv"
        if not p.exists():
            continue
        m = {}
        try:
            with io.open(p, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    try:
                        m[datetime.fromisoformat(row["ts"])] = float(row["close"])
                    except Exception:
                        continue
        except Exception:
            continue
        if len(m) >= min_bars:
            closes[s] = m
    if not closes:
        return [], {}, []
    ts = set(next(iter(closes.values())).keys())
    for m in closes.values():
        ts &= set(m.keys())
    return sorted(ts), closes, sorted(closes.keys())


def corr(a, b):
    n = min(len(a), len(b))
    if n < 20:
        return 0.0
    a, b = a[:n], b[:n]
    ma, mb = sum(a)/n, sum(b)/n
    da = [x-ma for x in a]; db = [x-mb for x in b]
    num = sum(x*y for x, y in zip(da, db))
    na = math.sqrt(sum(x*x for x in da)); nb = math.sqrt(sum(x*x for x in db))
    return num/(na*nb) if na and nb else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=int, default=168)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--bigday", type=float, default=0.10)
    ap.add_argument("--volq", type=float, default=0.66, help="vol percentile floor for flag")
    ap.add_argument("--corrmax", type=float, default=0.60, help="trailing corr ceiling for flag")
    args = ap.parse_args()

    wl = sorted(set(json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))))
    ts, closes, kept = load_aligned(wl)
    if not kept:
        print("[ERR] no klines"); return
    # hourly returns + basket
    px = {s: [closes[s][t] for t in ts] for s in kept}
    R = {s: [math.log(px[s][i]/px[s][i-1]) if px[s][i-1] > 0 and px[s][i] > 0 else 0.0
             for i in range(1, len(ts))] for s in kept}
    rts = ts[1:]  # timestamps for returns
    n = len(rts)
    basket = [sum(R[s][i] for s in kept)/len(kept) for i in range(n)]

    # group return-bar indices by UTC date
    days = defaultdict(list)
    for i, t in enumerate(rts):
        days[t.date()].append(i)
    daylist = sorted(d for d in days if len(days[d]) >= 20)  # near-full days
    print(f"[aligned] {len(kept)} watchlist coins, {n} bars, {len(daylist)} full days "
          f"({daylist[0]} → {daylist[-1]})")
    print(f"[flag] vol_pctile≥{args.volq} AND trailing_corr≤{args.corrmax}; "
          f"top-gainer = top-{args.topk}/day OR daily≥{args.bigday*100:.0f}%")

    half_day = daylist[len(daylist)//2]

    def daily_ret(s, idxs):
        return sum(R[s][i] for i in idxs)

    # for stats
    stats = {"train": defaultdict(int), "holdout": defaultdict(int)}
    # also count flagged coin-days total (for precision)
    for di, day in enumerate(daylist):
        idxs = days[day]
        start = idxs[0]
        if start < args.win:
            continue
        split = "train" if day < half_day else "holdout"
        # compute per-coin trailing corr + vol (ending at day start) and daily ret
        rows = []
        for s in kept:
            wnd = R[s][start-args.win:start]
            bwnd = basket[start-args.win:start]
            c = corr(wnd, bwnd)
            vol = math.sqrt(sum(x*x for x in wnd)/len(wnd))
            dr = daily_ret(s, idxs)
            rows.append([s, c, vol, dr])
        # cross-sectional vol percentile this day
        vols = sorted(r[2] for r in rows)
        def vpct(v):
            # fraction of coins with vol <= v
            lo = 0
            for x in vols:
                if x <= v: lo += 1
            return lo/len(vols)
        # top-K by daily return
        ranked = sorted(rows, key=lambda r: -r[3])
        topset = set(id(r) for r in ranked[:args.topk])
        for r in rows:
            s, c, vol, dr = r
            flagged = (vpct(vol) >= args.volq) and (c <= args.corrmax)
            is_top = id(r) in topset or dr >= args.bigday
            st = stats[split]
            if flagged:
                st["flagged"] += 1
                if is_top: st["flagged_top"] += 1
            if is_top:
                st["top"] += 1
                if flagged: st["top_flagged"] += 1
            st["total"] += 1

    print(f"\n{'='*70}")
    print("FLAG AS TOP-GAINER DETECTOR (no lookahead)")
    print(f"{'='*70}")
    for split in ("train", "holdout"):
        s = stats[split]
        top = s["top"]; flagged = s["flagged"]
        recall = s["top_flagged"]/top*100 if top else 0
        precision = s["flagged_top"]/flagged*100 if flagged else 0
        base = top/s["total"]*100 if s["total"] else 0
        lift = precision/base if base else 0
        print(f"\n  {split.upper()}  (coin-days={s['total']})")
        print(f"    real top-gainer days:          {top}")
        print(f"    flagged coin-days:             {flagged}")
        print(f"    top-gainers that were flagged: {s['top_flagged']}  "
              f"→ RECALL = {recall:.1f}% (bot could pre-flag {s['top_flagged']}/{top})")
        print(f"    flags that became top-gainers: {s['flagged_top']}  "
              f"→ PRECISION = {precision:.1f}%")
        print(f"    base rate (any coin-day top):  {base:.1f}%  → flag LIFT = x{lift:.2f}")

    print("\n[note] recall = of real top-gainers, how many the flag catches early.")
    print("       precision x base = how much better than random the flag is.")
    print("       research-only, no production change.")


if __name__ == "__main__":
    main()
