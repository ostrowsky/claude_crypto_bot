"""H-DECOUPLE validation: are decoupled coins (low trailing correlation to
the market) more likely to make a forward idiosyncratic big move?

Premise: the bot is tuned to beta (moves with the market). Coins that
detach from the market and run on their own narrative are exactly the
top-gainers it under-captures. If low trailing-corr predicts a forward
big move BEYOND what volatility alone explains, the signal is real and
real-time gateable.

Data: 365d × 1h klines, full universe. Train/holdout split.

For each coin at each hour t (sampled every STEP bars):
  - trailing corr to market basket over prior WIN bars   (the signal, known at t)
  - trailing realized vol over prior WIN bars            (the control)
  - forward FWD-bar return                               (the outcome)
Bucket by trailing-corr decile; report forward big-move rate and mean
forward return. Then control for vol: within vol terciles, does low-corr
still predict big moves?
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
HISTORY_DIR = ROOT / "history"
UNIVERSE_FILE = ROOT / ".runtime" / "binance_universe.json"
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
    ap.add_argument("--win", type=int, default=168, help="trailing window bars (7d)")
    ap.add_argument("--fwd", type=int, default=24, help="forward horizon bars (24h)")
    ap.add_argument("--step", type=int, default=12, help="sample every N bars")
    ap.add_argument("--big-move", type=float, default=0.10, help="forward big-move threshold")
    ap.add_argument("--watchlist-only", action="store_true")
    args = ap.parse_args()

    universe = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))["symbols"]
    wl = set(json.loads(WATCHLIST_FILE.read_text(encoding="utf-8")))
    pool = sorted(wl) if args.watchlist_only else universe
    ts, closes, kept = load_aligned(pool)
    if not kept:
        print("[ERR] no 365d klines"); return
    # market basket = mean return across ALL aligned coins
    px = {s: [closes[s][t] for t in ts] for s in kept}
    R = {s: [math.log(px[s][i]/px[s][i-1]) if px[s][i-1] > 0 and px[s][i] > 0 else 0.0
             for i in range(1, len(ts))] for s in kept}
    n = len(ts) - 1
    basket = [sum(R[s][i] for s in kept)/len(kept) for i in range(n)]
    print(f"[aligned] {len(kept)} coins, {n} return bars ({ts[0].date()} → {ts[-1].date()})  "
          f"{'WATCHLIST-only' if args.watchlist_only else 'UNIVERSE'}")
    print(f"[params] win={args.win}h fwd={args.fwd}h step={args.step} big_move≥{args.big_move*100:.0f}%")

    half = n // 2
    samples = []  # (t, sym, trailing_corr, trailing_vol, fwd_ret, is_train)
    for s in kept:
        r = R[s]
        for t in range(args.win, n - args.fwd, args.step):
            wnd = r[t-args.win:t]
            bwnd = basket[t-args.win:t]
            c = corr(wnd, bwnd)
            vol = math.sqrt(sum(x*x for x in wnd)/len(wnd))
            # forward cumulative return
            fwd = sum(r[t:t+args.fwd])
            samples.append((t, s, c, vol, fwd, t < half))
    print(f"[samples] {len(samples)}")

    def report(rows, label):
        if not rows:
            print(f"  {label}: (empty)"); return
        rows = sorted(rows, key=lambda x: x[2])  # by trailing corr
        nq = len(rows)
        print(f"\n  {label}  (n={nq})  — buckets by trailing corr (low=decoupled)")
        print(f"    {'bucket':<10}{'corr_rng':>16}{'mean_fwd%':>11}{'bigmove%':>10}{'mean_vol':>10}")
        for q in range(5):
            lo = q*nq//5; hi = (q+1)*nq//5
            seg = rows[lo:hi]
            cr = f"{seg[0][2]:+.2f}..{seg[-1][2]:+.2f}"
            mf = sum(x[4] for x in seg)/len(seg)*100
            bm = sum(1 for x in seg if x[4] >= args.big_move)/len(seg)*100
            mv = sum(x[3] for x in seg)/len(seg)
            tag = " DECOUPLED" if q == 0 else (" COUPLED" if q == 4 else "")
            print(f"    Q{q+1:<9}{cr:>16}{mf:>11.3f}{bm:>10.2f}{mv:>10.4f}{tag}")

    report([x for x in samples if x[5]], "TRAIN")
    report([x for x in samples if not x[5]], "HOLDOUT")

    # ---- vol-controlled: within vol terciles, low-corr vs high-corr big-move rate ----
    print(f"\n{'='*70}\nVOL-CONTROLLED (is decoupling predictive BEYOND volatility?)\n{'='*70}")
    for split_name, subset in (("TRAIN", [x for x in samples if x[5]]),
                               ("HOLDOUT", [x for x in samples if not x[5]])):
        byvol = sorted(subset, key=lambda x: x[3])
        m = len(byvol)
        print(f"\n  {split_name}")
        print(f"    {'vol tercile':<14}{'lowcorr_big%':>14}{'highcorr_big%':>15}{'lift(pp)':>10}")
        for vt in range(3):
            seg = byvol[vt*m//3:(vt+1)*m//3]
            seg = sorted(seg, key=lambda x: x[2])
            k = len(seg)//3
            low = seg[:k]; high = seg[-k:]
            lb = sum(1 for x in low if x[4] >= args.big_move)/len(low)*100
            hb = sum(1 for x in high if x[4] >= args.big_move)/len(high)*100
            print(f"    {['low','mid','high'][vt]:<14}{lb:>14.2f}{hb:>15.2f}{lb-hb:>10.2f}")

    print("\n[note] research-only. If DECOUPLED (Q1) big-move% > COUPLED (Q5) in BOTH")
    print("       train+holdout AND the vol-controlled lift stays positive, the bot's")
    print("       beta-tuning is leaving idiosyncratic top-movers on the table.")


if __name__ == "__main__":
    main()
