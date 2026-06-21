"""Lead-lag dependencies between SECTOR clusters (15m, 90d).

Data-driven Union-Find collapses to one mega-cluster (single-linkage
chaining), so we define interpretable narrative sectors and test all
ordered pairs: does sector A's move precede sector B's?

  - mean RAW 15m log-return series per sector (raw = keeps shared drift
    that carries the lead-lag);
  - lagged cross-corr A[t-k] vs B[t], k=1..8 bars (15m..2h);
  - net asymmetry = lead(A->B) - lead(B->A);
  - train/holdout stability gate (first vs second half);
  - also a contemporaneous corr matrix and a "fastest/slowest sector"
    ranking (which sector the market's move reaches first).

Research-only. No production change.
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

# Narrative sectors. Only coins with 15m klines are used; missing are skipped.
SECTORS = {
    "L1_major":   ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT",
                   "DOTUSDT", "TRXUSDT", "NEARUSDT", "ATOMUSDT", "APTUSDT", "SUIUSDT",
                   "SEIUSDT", "ICPUSDT", "ALGOUSDT", "HBARUSDT", "EGLDUSDT", "XLMUSDT",
                   "XTZUSDT", "ETCUSDT", "XRPUSDT", "LTCUSDT", "BCHUSDT"],
    "L2_scaling": ["ARBUSDT", "OPUSDT", "STRKUSDT", "METISUSDT", "POLUSDT", "ZROUSDT"],
    "defi":       ["AAVEUSDT", "UNIUSDT", "CRVUSDT", "SUSHIUSDT", "COMPUSDT", "YFIUSDT",
                   "SNXUSDT", "MKRUSDT", "1INCHUSDT", "LDOUSDT", "DYDXUSDT", "CAKEUSDT",
                   "LQTYUSDT", "CVXUSDT", "RUNEUSDT", "INJUSDT"],
    "meme":       ["DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "FLOKIUSDT", "WIFUSDT", "BONKUSDT",
                   "MEMEUSDT", "1000SATSUSDT", "ORDIUSDT", "1MBABYDOGEUSDT"],
    "gaming":     ["AXSUSDT", "SANDUSDT", "MANAUSDT", "GALAUSDT", "ILVUSDT", "PYRUSDT",
                   "PIXELUSDT", "RONINUSDT", "XAIUSDT", "GMTUSDT", "APEUSDT"],
    "ai":         ["FETUSDT", "RNDRUSDT", "RENDERUSDT", "TAOUSDT", "WLDUSDT", "ARKMUSDT",
                   "GRTUSDT"],
    "ton_eco":    ["TONUSDT", "DOGSUSDT", "NOTUSDT"],
    "sol_eco":    ["SOLUSDT", "RAYUSDT", "BNSOLUSDT"],
    "oldgen":     ["ZILUSDT", "ZRXUSDT", "BATUSDT", "QNTUSDT", "GLMUSDT", "BNTUSDT",
                   "AUDIOUSDT", "CHZUSDT", "JASMYUSDT", "CELOUSDT", "CELRUSDT"],
}


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
    return sorted(ts), closes, sorted(closes.keys())


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
    if k == 0:
        return pearson(lead, follow)
    return pearson(lead[:-k], follow[k:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lags", type=int, default=8)
    ap.add_argument("--lead-min", type=float, default=0.015)
    args = ap.parse_args()

    allsyms = sorted({s for v in SECTORS.values() for s in v})
    ts, closes, kept = load_aligned(allsyms)
    if not kept:
        print("[ERR] need 15m klines — run _backfill_watchlist_15m.py")
        return
    keptset = set(kept)
    R = {s: rets_of(ts, closes, s) for s in kept}
    n = min(len(v) for v in R.values())
    print(f"[aligned] {len(kept)} coins, {n} bars 15m ({ts[0].date()} → {ts[-1].date()})")

    # sector mean raw return series
    series = {}
    members = {}
    for name, syms in SECTORS.items():
        ms = [s for s in syms if s in keptset]
        if len(ms) < 2:
            continue
        members[name] = ms
        series[name] = [sum(R[s][i] for s in ms)/len(ms) for i in range(n)]
    names = list(series.keys())
    for nm in names:
        print(f"  {nm:<10} n={len(members[nm]):2d}  {', '.join(m.replace('USDT','') for m in members[nm][:10])}")

    half = n // 2

    # contemporaneous corr
    print(f"\n[contemporaneous corr matrix k=0]")
    print("            " + "  ".join(f"{nm[:6]:>6}" for nm in names))
    for a in names:
        row = "  ".join(f"{pearson(series[a], series[b]):+.2f}" for b in names)
        print(f"  {a:<9} {row}")

    # all ordered pairs directed lead
    print(f"\n{'='*78}")
    print("DIRECTED SECTOR LEAD DEPENDENCIES (A leads B; stable train+holdout)")
    print(f"{'='*78}")
    deps = []
    for a in names:
        for b in names:
            if a == b:
                continue
            fwd = [(k, lagcorr(series[a], series[b], k)) for k in range(1, args.lags+1)]
            bk, bf = max(fwd, key=lambda x: x[1])
            rev = max(lagcorr(series[b], series[a], k) for k in range(1, args.lags+1))
            net = bf - rev
            ntr = lagcorr(series[a][:half], series[b][:half], bk) - \
                  max(lagcorr(series[b][:half], series[a][:half], k) for k in range(1, args.lags+1))
            nho = lagcorr(series[a][half:], series[b][half:], bk) - \
                  max(lagcorr(series[b][half:], series[a][half:], k) for k in range(1, args.lags+1))
            if net >= args.lead_min and ntr > 0 and nho > 0:
                deps.append((net, a, b, bk, bf, rev, ntr, nho))
    deps.sort(reverse=True)
    if not deps:
        print("  (no stable directional lead dependencies above threshold)")
    for net, a, b, k, bf, rev, ntr, nho in deps:
        print(f"  {a:<10} → {b:<10} lag={k}bar({k*15}m) fwd={bf:+.3f} rev={rev:+.3f} "
              f"net={net:+.3f}  (tr={ntr:+.3f} ho={nho:+.3f})")

    # "speed" ranking: average forward-lead a sector has over all others
    print(f"\n[sector leadership score] mean net-asymmetry vs all other sectors (>0 => tends to lead)")
    score = {}
    for a in names:
        vals = []
        for b in names:
            if a == b:
                continue
            bf = max(lagcorr(series[a], series[b], k) for k in range(1, args.lags+1))
            rev = max(lagcorr(series[b], series[a], k) for k in range(1, args.lags+1))
            vals.append(bf - rev)
        score[a] = sum(vals)/len(vals)
    for a in sorted(names, key=lambda x: -score[x]):
        print(f"  {a:<10} {score[a]:+.4f}")

    print("\n[note] research-only. Contemporaneous corr ~0.8 across sectors (one")
    print("       market); lead signal is the small net-asymmetry, gated on holdout.")


if __name__ == "__main__":
    main()
