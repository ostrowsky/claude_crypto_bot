"""Does BTC (and ETH) movement LEAD alt movements at 15m? (90d)

1h bars were too coarse — crypto BTC→alt propagation lives in minutes.
This tests at 15m:
  A) lagged cross-corr: corr(BTC[t-k], alt_basket[t]), k=1..8 bars (15m..2h)
  B) event study: after a BIG BTC bar (|ret| in top decile), what is the
     alt-basket forward return over next 1/2/4 bars? Split by BTC up/down.
  C) per-coin BTC-responsiveness ranking (which alts react most, and lag).
All with train/holdout split (first vs second half).

Research-only — answers "is a BTC surge an early indicator of alt moves".
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
WATCHLIST_FILE = ROOT / "files" / "watchlist.json"


def load_aligned(syms, suffix="_15m_90d", min_bars=5000):
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
    ts = sorted(ts)
    return ts, closes, sorted(closes.keys())


def rets_of(ts, closes, s):
    px = [closes[s][t] for t in ts]
    return [math.log(px[i] / px[i-1]) if px[i-1] > 0 and px[i] > 0 else 0.0
            for i in range(1, len(px))]


def pearson(a, b):
    n = min(len(a), len(b))
    if n < 30:
        return 0.0
    a, b = a[:n], b[:n]
    ma, mb = sum(a)/n, sum(b)/n
    da = [x-ma for x in a]; db = [x-mb for x in b]
    num = sum(x*y for x, y in zip(da, db))
    na = math.sqrt(sum(x*x for x in da)); nb = math.sqrt(sum(x*x for x in db))
    return num/(na*nb) if na and nb else 0.0


def lagcorr(lead, follow, k):
    """corr(lead[t-k], follow[t]); k>0 => lead precedes follow."""
    if k == 0:
        return pearson(lead, follow)
    return pearson(lead[:-k], follow[k:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lags", type=int, default=8)
    args = ap.parse_args()

    wl = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
    ts, closes, kept = load_aligned(wl + ["BTCUSDT", "ETHUSDT"])
    if not kept or "BTCUSDT" not in kept:
        print("[ERR] need 15m klines incl BTCUSDT — run _backfill_watchlist_15m.py")
        return
    print(f"[aligned] {len(kept)} coins, {len(ts)} bars 15m "
          f"({ts[0].date()} → {ts[-1].date()})")

    R = {s: rets_of(ts, closes, s) for s in kept}
    n = len(R["BTCUSDT"])
    alts = [s for s in kept if s not in ("BTCUSDT", "ETHUSDT")]
    basket = [sum(R[s][i] for s in alts)/len(alts) for i in range(n)]
    half = n // 2

    def split_lagcorr(lead, follow, k):
        full = lagcorr(lead, follow, k)
        tr = lagcorr(lead[:half], follow[:half], k)
        ho = lagcorr(lead[half:], follow[half:], k)
        return full, tr, ho

    print(f"\n{'='*72}\nA) LAGGED CROSS-CORR  (lead[t-k] vs follow[t]); train|holdout\n{'='*72}")
    for lead in ("BTCUSDT", "ETHUSDT"):
        print(f"\n  {lead} → ALT-BASKET")
        print(f"    k=0 (contemp): {pearson(R[lead], basket):+.3f}")
        for k in range(1, args.lags + 1):
            f, tr, ho = split_lagcorr(R[lead], basket, k)
            rev = lagcorr(basket, R[lead], k)
            mark = "  <= net+" if (f - rev) > 0.02 and tr > 0 and ho > 0 else ""
            print(f"    +{k}bar({k*15}m): fwd={f:+.3f} rev={rev:+.3f} net={f-rev:+.3f}  (tr={tr:+.3f} ho={ho:+.3f}){mark}")

    # B) event study
    print(f"\n{'='*72}\nB) EVENT STUDY — alt-basket forward return after a BIG BTC bar\n{'='*72}")
    btc = R["BTCUSDT"]
    absr = sorted(abs(x) for x in btc)
    thr = absr[int(len(absr) * 0.90)]  # top-decile abs move
    print(f"  BTC big-bar threshold |ret| ≥ {thr*100:.2f}% (top decile), n bars={n}")
    for horizon in (1, 2, 4):
        for sign, lab in ((1, "UP"), (-1, "DN")):
            fwds = []
            for i in range(n - horizon):
                if sign * btc[i] >= thr:  # big move in this direction
                    fwds.append(sum(basket[i+1:i+1+horizon]))
            if fwds:
                avg = sum(fwds)/len(fwds)
                pos = sum(1 for x in fwds if x > 0)/len(fwds)
                print(f"  BTC {lab} big-bar → basket next {horizon}bar ({horizon*15}m): "
                      f"avg={avg*100:+.3f}%  pos={pos*100:.0f}%  n={len(fwds)}")
    # contemporaneous (same-bar) for reference
    same_up = [basket[i] for i in range(n) if btc[i] >= thr]
    print(f"  [ref] same-bar basket when BTC UP big: avg={sum(same_up)/len(same_up)*100:+.3f}% "
          f"(propagation already mostly within the 15m bar if next-bar ≈ 0)")

    # C) per-coin BTC responsiveness (best forward lag)
    print(f"\n{'='*72}\nC) PER-COIN: best BTC→coin forward lag (stable in train+holdout)\n{'='*72}")
    rows = []
    for s in alts:
        best = None
        for k in range(1, args.lags + 1):
            f, tr, ho = split_lagcorr(R["BTCUSDT"], R[s], k)
            rev = lagcorr(R[s], R["BTCUSDT"], k)
            net = f - rev
            if tr > 0 and ho > 0 and (best is None or net > best[1]):
                best = (k, net, f, rev)
        if best and best[1] > 0.02:
            rows.append((best[1], s, best[0], best[2], best[3]))
    rows.sort(reverse=True)
    if not rows:
        print("  (no coin shows stable BTC-lead net-asymmetry > 0.02 — propagation is contemporaneous)")
    for net, s, k, f, rev in rows[:25]:
        print(f"  BTC → {s:<12} lag={k}bar({k*15}m) fwd={f:+.3f} rev={rev:+.3f} net={net:+.3f}")

    print("\n[note] research-only. If next-bar net-asymmetry ≈ 0 while contemp corr is")
    print("       high, BTC and alts move TOGETHER within 15m — no exploitable lead.")


if __name__ == "__main__":
    main()
