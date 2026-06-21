"""Universe-wide correlation clustering with watchlist tagging.

Discovers clusters across all 427 Binance USDT-spot pairs. For each
cluster containing watchlist coins, identifies external (non-watchlist)
members that can serve as LEADING INDICATORS — watching their movement
predicts our watchlist coin's behavior.

Reads cached 1h × 30d klines from history/<sym>_1h.csv.

Spec rationale:
  Our internal cluster work found that residualized intra-correlation
  identifies real sector groups. By expanding the universe from 69
  watchlist coins to 427 Binance USDT pairs, we discover MORE coins
  per cluster — many of which we don't currently scan, but which
  pre-pump and signal incoming watchlist moves.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
HISTORY_DIR = ROOT / "history"
UNIVERSE_FILE = ROOT / ".runtime" / "binance_universe.json"
WATCHLIST_FILE = ROOT / "files" / "watchlist.json"


def load_universe() -> list[str]:
    return json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))["symbols"]


def load_watchlist() -> set[str]:
    return set(json.loads(WATCHLIST_FILE.read_text(encoding="utf-8")))


def load_klines_aligned(syms: list[str], tf: str = "1h"):
    """Load klines for each sym, return aligned closes."""
    closes_by_sym: dict[str, dict[datetime, float]] = {}
    for sym in syms:
        p = HISTORY_DIR / f"{sym}_{tf}.csv"
        if not p.exists():
            continue
        m: dict[datetime, float] = {}
        try:
            with io.open(p, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    try:
                        dt = datetime.fromisoformat(row["ts"])
                        m[dt] = float(row["close"])
                    except Exception:
                        continue
        except Exception:
            continue
        if len(m) >= 600:  # >~25d of 1h bars
            closes_by_sym[sym] = m

    if not closes_by_sym:
        return [], {}, []

    timestamps = set(next(iter(closes_by_sym.values())).keys())
    for m in closes_by_sym.values():
        timestamps &= set(m.keys())
    timestamps_sorted = sorted(timestamps)

    if len(timestamps_sorted) < 200:
        return [], {}, []
    return timestamps_sorted, closes_by_sym, sorted(closes_by_sym.keys())


def log_returns(timestamps: list[datetime], closes_by_sym: dict, syms: list[str]) -> dict[str, list[float]]:
    out = {}
    for sym in syms:
        m = closes_by_sym[sym]
        prices = [m[t] for t in timestamps]
        rets = []
        for i in range(1, len(prices)):
            p0, p1 = prices[i - 1], prices[i]
            if p0 > 0 and p1 > 0:
                rets.append(math.log(p1 / p0))
            else:
                rets.append(0.0)
        out[sym] = rets
    return out


def pearson(a, b):
    n = len(a)
    if n < 10 or len(b) != n: return 0.0
    ma = sum(a) / n; mb = sum(b) / n
    da = [x - ma for x in a]; db = [x - mb for x in b]
    num = sum(x*y for x, y in zip(da, db))
    da_n = math.sqrt(sum(x*x for x in da))
    db_n = math.sqrt(sum(x*x for x in db))
    if da_n == 0 or db_n == 0: return 0.0
    return num / (da_n * db_n)


def residualize_vs(rets: dict, syms: list[str], anchor: str) -> dict:
    if anchor not in rets:
        return rets
    anchor_r = rets[anchor]
    n = len(anchor_r)
    a_mean = sum(anchor_r) / n
    a_var = sum((x - a_mean) ** 2 for x in anchor_r) / n
    if a_var == 0: return rets
    out = {anchor: anchor_r}
    for sym in syms:
        if sym == anchor: continue
        r = rets[sym]
        r_mean = sum(r) / max(1, len(r))
        cov = sum((x - r_mean) * (y - a_mean) for x, y in zip(r, anchor_r)) / max(1, n)
        beta = cov / a_var
        out[sym] = [x - beta * y for x, y in zip(r, anchor_r)]
    return out


def corr_matrix(rets: dict, syms: list[str]):
    n = len(syms)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        mat[i][i] = 1.0
        for j in range(i + 1, n):
            c = pearson(rets[syms[i]], rets[syms[j]])
            mat[i][j] = mat[j][i] = c
    return mat


class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


def cluster_at(syms, mat, thr):
    n = len(syms)
    uf = UF(n)
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i][j] >= thr: uf.union(i, j)
    groups = defaultdict(list)
    for i in range(n): groups[uf.find(i)].append(i)
    return sorted([sorted(syms[i] for i in g) for g in groups.values()], key=lambda g: -len(g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-cluster-size", type=int, default=3)
    ap.add_argument("--residualize", choices=["btc", "btc_eth", "none"], default="btc")
    ap.add_argument("--save", type=Path, default=ROOT / ".runtime" / "universe_clusters.json")
    args = ap.parse_args()

    universe = load_universe()
    watchlist = load_watchlist()
    print(f"[universe] {len(universe)} coins, {len(watchlist)} watchlist")

    timestamps, closes, kept = load_klines_aligned(universe, tf="1h")
    if not kept:
        print("[ERR] no klines aligned"); return

    print(f"[aligned] {len(kept)} / {len(universe)} coins, {len(timestamps)} hourly bars")
    in_wl = sum(1 for s in kept if s in watchlist)
    print(f"[aligned] watchlist members covered: {in_wl} / {len(watchlist)}")

    rets = log_returns(timestamps, closes, kept)

    if args.residualize == "btc" and "BTCUSDT" in rets:
        print(f"[resid] vs BTC only")
        rets = residualize_vs(rets, kept, "BTCUSDT")
    elif args.residualize == "btc_eth" and "BTCUSDT" in rets and "ETHUSDT" in rets:
        print(f"[resid] vs BTC then ETH")
        rets = residualize_vs(rets, kept, "BTCUSDT")
        rets = residualize_vs(rets, kept, "ETHUSDT")
    else:
        print(f"[resid] none (raw correlation)")

    print(f"[matrix] computing {len(kept)}×{len(kept)} ({len(kept)*(len(kept)-1)//2} pairs)…")
    mat = corr_matrix(rets, kept)

    multi = [c for c in cluster_at(kept, mat, args.threshold) if len(c) >= args.min_cluster_size]
    print(f"\n[clusters @ corr ≥ {args.threshold}] {len(multi)} clusters of size ≥ {args.min_cluster_size}")
    print(f"  total coins in clusters: {sum(len(c) for c in multi)}")
    wl_covered_in_clusters = sum(1 for c in multi for s in c if s in watchlist)
    print(f"  watchlist coins in clusters: {wl_covered_in_clusters}")

    print(f"\n{'='*78}")
    print(f"CLUSTER REPORT (cluster_id · n_total / n_watchlist · n_external)")
    print(f"{'='*78}")

    saved = []
    for ci, cluster in enumerate(multi, 1):
        wl_members = [s for s in cluster if s in watchlist]
        ext_members = [s for s in cluster if s not in watchlist]
        if not wl_members:
            continue  # skip clusters that have NO watchlist coin (no leading-indicator use)
        # Intra-corr
        idx = {s: kept.index(s) for s in cluster}
        intra = []
        for a in range(len(cluster)):
            for b in range(a + 1, len(cluster)):
                intra.append(mat[idx[cluster[a]]][idx[cluster[b]]])
        avg_intra = sum(intra) / max(1, len(intra))

        # Volatility (mean abs log-ret × 100 = pct)
        vol = {s: float(sum(abs(r) for r in rets[s]) / len(rets[s]) * 100.0) for s in cluster}

        print(f"\n── Cluster #{ci}  (n={len(cluster)}, watchlist={len(wl_members)}, external={len(ext_members)}, intra-corr={avg_intra:+.3f}) ──")
        if wl_members:
            print(f"  📋 IN WATCHLIST  ({len(wl_members)}):")
            for s in sorted(wl_members, key=lambda x: -vol[x])[:15]:
                print(f"     {s:<14}  vol={vol[s]:.3f}%")
        if ext_members:
            print(f"  🔭 EXTERNAL leading-indicator candidates ({len(ext_members)}):")
            for s in sorted(ext_members, key=lambda x: -vol[x])[:15]:
                print(f"     {s:<14}  vol={vol[s]:.3f}%")
            if len(ext_members) > 15:
                print(f"     … and {len(ext_members) - 15} more")

        saved.append({
            "cluster_id": ci,
            "size": len(cluster),
            "watchlist_members": sorted(wl_members),
            "external_members": sorted(ext_members),
            "avg_intra_corr": round(avg_intra, 4),
            "volatility": {k: round(v, 4) for k, v in vol.items()},
        })

    # Also: clusters with 0 watchlist coins (pure external) — show top-3 biggest
    pure_ext = [c for c in multi if not any(s in watchlist for s in c)]
    if pure_ext:
        print(f"\n{'─'*78}")
        print(f"PURE-EXTERNAL clusters (no watchlist members) — top 3 by size:")
        for c in pure_ext[:3]:
            print(f"  n={len(c)}  members: {', '.join(c[:8])}{' …' if len(c) > 8 else ''}")
        print(f"  total pure-external clusters: {len(pure_ext)}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tf": "1h",
            "days": 30,
            "threshold": args.threshold,
            "residualize": args.residualize,
            "n_universe": len(universe),
            "n_aligned": len(kept),
            "n_watchlist": len(watchlist),
            "clusters_with_watchlist": saved,
            "n_clusters_pure_external": len(pure_ext),
        }, indent=2), encoding="utf-8")
        print(f"\n[save] wrote {args.save}")


if __name__ == "__main__":
    main()
