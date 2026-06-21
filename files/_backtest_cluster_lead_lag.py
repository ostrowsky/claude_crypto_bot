"""Cluster-level lead-lag dependency analysis (max period 365d × 1h).

Hypothesis under test: movements of one cluster DRIVE/PRECEDE movements
of other clusters — e.g. the BTC cluster leads alt clusters.

Method:
  1. Load 365d × 1h klines for full universe.
  2. Cluster coins into sectors (Union-Find on residualized correlation,
     so clusters are real sectors not one BTC blob).
  3. For each cluster, build the RAW mean log-return series across members
     (raw, NOT residualized — we WANT the shared market drift, that's the
     lead-lag carrier).
  4. For every ordered pair (A,B), compute lagged cross-correlation
     corr(A[t-k], B[t]) for k = 1..LAGS hours.
  5. lead_score(A->B) = max_k>0 corr(A[t-k],B[t]).
     Net asymmetry = lead(A->B) - lead(B->A). Positive => A leads B.
  6. Split train/holdout — only report dependencies stable in BOTH halves.

This is research-only: no production wiring. The output answers
"which cluster's move is an early indicator of which other cluster's move".
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
UNIVERSE_FILE = ROOT / ".runtime" / "binance_universe.json"
WATCHLIST_FILE = ROOT / "files" / "watchlist.json"


def load_universe():
    return json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))["symbols"]


def load_watchlist():
    return set(json.loads(WATCHLIST_FILE.read_text(encoding="utf-8")))


def load_klines_aligned(syms, suffix="_1h_365d", min_bars=4000):
    closes = {}
    for sym in syms:
        p = HISTORY_DIR / f"{sym}{suffix}.csv"
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
            closes[sym] = m
    if not closes:
        return [], {}, []
    ts = set(next(iter(closes.values())).keys())
    for m in closes.values():
        ts &= set(m.keys())
    ts = sorted(ts)
    if len(ts) < 1000:
        return [], {}, []
    return ts, closes, sorted(closes.keys())


def log_returns(ts, closes, syms):
    out = {}
    for s in syms:
        m = closes[s]
        px = [m[t] for t in ts]
        r = []
        for i in range(1, len(px)):
            a, b = px[i - 1], px[i]
            r.append(math.log(b / a) if a > 0 and b > 0 else 0.0)
        out[s] = r
    return out


def pearson(a, b):
    n = min(len(a), len(b))
    if n < 30:
        return 0.0
    a = a[:n]; b = b[:n]
    ma = sum(a) / n; mb = sum(b) / n
    da = [x - ma for x in a]; db = [x - mb for x in b]
    num = sum(x * y for x, y in zip(da, db))
    na = math.sqrt(sum(x * x for x in da)); nb = math.sqrt(sum(x * x for x in db))
    return num / (na * nb) if na and nb else 0.0


def residualize(rets, syms, anchor):
    if anchor not in rets:
        return rets
    ar = rets[anchor]; n = len(ar)
    am = sum(ar) / n
    av = sum((x - am) ** 2 for x in ar) / n
    if av == 0:
        return rets
    out = {anchor: ar}
    for s in syms:
        if s == anchor:
            continue
        r = rets[s]; rm = sum(r) / len(r)
        cov = sum((x - rm) * (y - am) for x, y in zip(r, ar)) / n
        beta = cov / av
        out[s] = [x - beta * y for x, y in zip(r, ar)]
    return out


class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


def build_clusters(rets, syms, thr, raw=False):
    """Cluster coins. residualized=sectors; raw=natural co-movement groups
    (BTC forms a majors cluster — needed to test BTC-cluster-leads-others)."""
    if raw:
        res = rets
    else:
        res = residualize(rets, syms, "BTCUSDT")
        if "ETHUSDT" in res:
            res = residualize(res, syms, "ETHUSDT")
    n = len(syms)
    uf = UF(n)
    for i in range(n):
        for j in range(i + 1, n):
            if pearson(res[syms[i]], res[syms[j]]) >= thr:
                uf.union(i, j)
    g = defaultdict(list)
    for i in range(n):
        g[uf.find(i)].append(syms[i])
    return sorted([sorted(v) for v in g.values()], key=lambda c: -len(c))


def mean_series(rets, members):
    n = min(len(rets[m]) for m in members)
    return [sum(rets[m][i] for m in members) / len(members) for i in range(n)]


def lagged_corr(a, b, k):
    """corr(a[t-k], b[t]) — a leads b by k bars if positive/large."""
    if k <= 0:
        return pearson(a, b)
    return pearson(a[:-k], b[k:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster-threshold", type=float, default=0.5)
    ap.add_argument("--min-cluster-size", type=int, default=3)
    ap.add_argument("--lags", type=int, default=6)
    ap.add_argument("--lead-min", type=float, default=0.05,
                    help="min net-asymmetry to report a lead dependency")
    ap.add_argument("--suffix", default="_1h_365d")
    ap.add_argument("--raw", action="store_true",
                    help="cluster on raw (not residualized) corr — BTC forms a majors cluster")
    args = ap.parse_args()

    universe = load_universe()
    watchlist = load_watchlist()
    ts, closes, kept = load_klines_aligned(universe, suffix=args.suffix)
    if not kept:
        print("[ERR] no 365d klines aligned — run _backfill_universe_365d.py first")
        return
    print(f"[aligned] {len(kept)}/{len(universe)} coins, {len(ts)} bars "
          f"({ts[0].date()} → {ts[-1].date()})")

    rets = log_returns(ts, closes, kept)

    # ---- cluster (full period) ----
    clusters = [c for c in build_clusters(rets, kept, args.cluster_threshold, raw=args.raw)
                if len(c) >= args.min_cluster_size]
    print(f"[clusters @ {args.cluster_threshold}] {len(clusters)} sectors of size ≥ {args.min_cluster_size}")

    # name each cluster by its largest-cap / most-recognizable member + watchlist tag
    def cname(c, i):
        wl = [s for s in c if s in watchlist]
        anchor = (wl or c)[0]
        return f"C{i}:{anchor.replace('USDT','')}(n={len(c)},wl={len(wl)})"

    names = [cname(c, i) for i, c in enumerate(clusters, 1)]
    for nm, c in zip(names, clusters):
        wl = [s.replace("USDT", "") for s in c if s in watchlist]
        ext = [s.replace("USDT", "") for s in c if s not in watchlist]
        print(f"  {nm}")
        print(f"      wl: {', '.join(wl[:12])}{' …' if len(wl)>12 else ''}")
        if ext:
            print(f"      ext: {', '.join(ext[:10])}{' …' if len(ext)>10 else ''}")

    # ---- cluster mean RAW return series ----
    series = [mean_series(rets, c) for c in clusters]
    n = min(len(s) for s in series)
    series = [s[:n] for s in series]
    half = n // 2
    tr = [s[:half] for s in series]
    ho = [s[half:] for s in series]

    print(f"\n[lead-lag] full n={n} bars  train={half}  holdout={n-half}  lags=1..{args.lags}h")

    # contemporaneous corr matrix (how coupled at all)
    print("\n[contemporaneous corr] (k=0)")
    for i in range(len(clusters)):
        row = "  ".join(f"{pearson(series[i], series[j]):+.2f}" for j in range(len(clusters)))
        print(f"  {names[i]:<28} {row}")

    # ---- directed lead dependencies ----
    print(f"\n{'='*78}")
    print("DIRECTED LEAD DEPENDENCIES  (A leads B; reported if stable in train+holdout)")
    print(f"{'='*78}")
    deps = []
    K = len(clusters)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            # best forward lead A(i)->B(j) over full
            fwd = [(k, lagged_corr(series[i], series[j], k)) for k in range(1, args.lags + 1)]
            best_k, best_fwd = max(fwd, key=lambda x: x[1])
            rev = max(lagged_corr(series[j], series[i], k) for k in range(1, args.lags + 1))
            net = best_fwd - rev
            # stability: same direction positive net in both halves at best_k
            net_tr = lagged_corr(tr[i], tr[j], best_k) - max(lagged_corr(tr[j], tr[i], k) for k in range(1, args.lags + 1))
            net_ho = lagged_corr(ho[i], ho[j], best_k) - max(lagged_corr(ho[j], ho[i], k) for k in range(1, args.lags + 1))
            if net >= args.lead_min and net_tr > 0 and net_ho > 0:
                deps.append((net, i, j, best_k, best_fwd, rev, net_tr, net_ho))

    deps.sort(reverse=True)
    if not deps:
        print("  (no stable directional lead dependencies above threshold)")
    for net, i, j, k, fwd, rev, ntr, nho in deps:
        print(f"  {names[i]:<26} → {names[j]:<26}  lag={k}h  "
              f"fwd={fwd:+.3f} rev={rev:+.3f} net={net:+.3f}  (tr={ntr:+.3f} ho={nho:+.3f})")

    # ---- BTC cluster specific summary ----
    btc_idx = next((i for i, c in enumerate(clusters) if "BTCUSDT" in c), None)
    if btc_idx is not None:
        print(f"\n[BTC-cluster ({names[btc_idx]}) lead over each other cluster]")
        for j in range(K):
            if j == btc_idx:
                continue
            fwd = [(k, lagged_corr(series[btc_idx], series[j], k)) for k in range(1, args.lags + 1)]
            bk, bf = max(fwd, key=lambda x: x[1])
            rev = max(lagged_corr(series[j], series[btc_idx], k) for k in range(1, args.lags + 1))
            tag = "LEADS" if bf - rev >= args.lead_min else ("symmetric" if abs(bf - rev) < args.lead_min else "LAGS")
            print(f"  vs {names[j]:<26} bestlag={bk}h fwd={bf:+.3f} rev={rev:+.3f} → {tag}")

    print("\n[note] research-only. No production change. corr on hourly returns is")
    print("       weak by nature; net-asymmetry + train/holdout stability is the signal.")


if __name__ == "__main__":
    main()
