"""Discover stable correlation clusters across the watchlist.

Reads cached klines from history/<sym>_15m.csv, builds a pairwise
correlation matrix on log-returns, runs Union-Find at multiple
thresholds, and outputs human-readable clusters + JSON.

Spec: docs/specs/features/correlation-clusters-spec.md

Usage:
  pyembed/python.exe files/_backtest_correlation_clusters.py
  pyembed/python.exe files/_backtest_correlation_clusters.py --days 30
  pyembed/python.exe files/_backtest_correlation_clusters.py --threshold 0.75
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
HISTORY_DIR = ROOT / "history"
WATCHLIST_FILE = ROOT / "files" / "watchlist.json"


def load_watchlist() -> list[str]:
    return json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))


def load_klines_aligned(syms: list[str], days: int, tf: str = "15m"):
    """Load klines for each sym, align on common timestamp grid.

    Returns (timestamps_sorted, closes_by_sym, kept_syms).
    Drops syms whose coverage < 90% of the requested window.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    bar_min = {"1m":1,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}.get(tf, 15)
    expected_bars = int(days * 24 * 60 / bar_min)

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
                        if dt < cutoff:
                            continue
                        m[dt] = float(row["close"])
                    except Exception:
                        continue
        except Exception:
            continue
        if len(m) >= 0.9 * expected_bars:
            closes_by_sym[sym] = m

    if not closes_by_sym:
        return [], {}, []

    # Intersect timestamps across kept syms
    timestamps = set(next(iter(closes_by_sym.values())).keys())
    for m in closes_by_sym.values():
        timestamps &= set(m.keys())
    timestamps_sorted = sorted(timestamps)

    if len(timestamps_sorted) < 100:
        return [], {}, []

    return timestamps_sorted, closes_by_sym, sorted(closes_by_sym.keys())


def compute_log_returns(timestamps: list[datetime], closes_by_sym: dict[str, dict],
                       syms: list[str]) -> dict[str, list[float]]:
    """Return dict sym -> log-returns list aligned to timestamps[1:]."""
    out = {}
    for sym in syms:
        m = closes_by_sym[sym]
        prices = [m[t] for t in timestamps]
        rets = []
        for i in range(1, len(prices)):
            p_prev = prices[i - 1]; p_cur = prices[i]
            if p_prev > 0 and p_cur > 0:
                rets.append(math.log(p_cur / p_prev))
            else:
                rets.append(0.0)
        out[sym] = rets
    return out


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 10 or len(b) != n:
        return 0.0
    mean_a = sum(a) / n; mean_b = sum(b) / n
    da = [x - mean_a for x in a]; db = [x - mean_b for x in b]
    num = sum(x * y for x, y in zip(da, db))
    den_a = math.sqrt(sum(x * x for x in da))
    den_b = math.sqrt(sum(x * x for x in db))
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)


def correlation_matrix(rets: dict[str, list[float]], syms: list[str]):
    """Pairwise Pearson on log-returns."""
    n = len(syms)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        mat[i][i] = 1.0
        for j in range(i + 1, n):
            c = pearson(rets[syms[i]], rets[syms[j]])
            mat[i][j] = mat[j][i] = c
    return mat


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra = self.find(a); rb = self.find(b)
        if ra != rb:
            self.p[ra] = rb


def cluster_at_threshold(syms: list[str], mat: list[list[float]], threshold: float):
    n = len(syms)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i][j] >= threshold:
                uf.union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    # Sort groups: largest first
    clusters = sorted(groups.values(), key=lambda g: -len(g))
    return [sorted([syms[i] for i in g]) for g in clusters]


def intra_cluster_corr(cluster_syms: list[str], all_syms: list[str],
                      mat: list[list[float]]) -> float:
    if len(cluster_syms) < 2:
        return 1.0
    idx = {s: i for i, s in enumerate(all_syms)}
    ids = [idx[s] for s in cluster_syms]
    vals = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            vals.append(mat[ids[a]][ids[b]])
    return sum(vals) / max(1, len(vals))


def find_lead(cluster_syms: list[str], rets: dict[str, list[float]],
             max_lag: int = 3) -> tuple[str, dict]:
    """For each pair in cluster, find optimal lag. Aggregate to find leader."""
    lead_score = defaultdict(int)
    pair_info = {}
    for i, a in enumerate(cluster_syms):
        for b in cluster_syms[i + 1:]:
            best_corr = -1.0; best_k = 0
            for k in range(-max_lag, max_lag + 1):
                if k >= 0:
                    aa = rets[a][:len(rets[a]) - k]
                    bb = rets[b][k:]
                else:
                    aa = rets[a][-k:]
                    bb = rets[b][:len(rets[b]) + k]
                if len(aa) < 30:
                    continue
                c = pearson(aa, bb)
                if c > best_corr:
                    best_corr = c; best_k = k
            pair_info[(a, b)] = (best_k, best_corr)
            if best_k > 0:   lead_score[a] += 1
            elif best_k < 0: lead_score[b] += 1
    if not lead_score:
        return cluster_syms[0], {"all_simultaneous": True}
    leader = max(lead_score.items(), key=lambda x: x[1])[0]
    return leader, {"score": dict(lead_score), "pairs": {f"{a}-{b}": v for (a, b), v in pair_info.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--tf", default="15m")
    ap.add_argument("--thresholds", type=str, default="0.5,0.6,0.7,0.8")
    ap.add_argument("--report-threshold", type=float, default=0.7,
                    help="Primary threshold to deep-analyze (lead-lag etc.)")
    ap.add_argument("--save", type=Path, default=None,
                    help="Optional path to save clusters as JSON")
    ap.add_argument("--residualize-btc", action="store_true",
                    help="Subtract BTC beta from each coin's returns before "
                         "clustering (finds sector groups orthogonal to BTC).")
    args = ap.parse_args()

    syms = load_watchlist()
    print(f"[clusters] watchlist size: {len(syms)}, window: {args.days}d × {args.tf}")

    timestamps, closes, kept = load_klines_aligned(syms, args.days, args.tf)
    if not kept:
        print("[clusters] insufficient data — run files/_backfill_klines_history.py first")
        return

    print(f"[clusters] coins with klines coverage ≥ 90%: {len(kept)} / {len(syms)}")
    print(f"[clusters] aligned bars: {len(timestamps)}")

    rets = compute_log_returns(timestamps, closes, kept)

    # Optional: residualize log-returns against BTC beta to find sector clusters
    # orthogonal to the dominant BTC-beta factor.
    if args.residualize_btc and "BTCUSDT" in kept:
        print(f"[clusters] residualizing log-returns vs BTCUSDT (sector mode)")
        btc_rets = rets["BTCUSDT"]
        n_btc = len(btc_rets)
        btc_mean = sum(btc_rets) / max(1, n_btc)
        btc_var = sum((x - btc_mean) ** 2 for x in btc_rets) / max(1, n_btc)
        if btc_var > 0:
            new_rets = {}
            for sym in kept:
                if sym == "BTCUSDT":
                    new_rets[sym] = btc_rets
                    continue
                r = rets[sym]
                r_mean = sum(r) / max(1, len(r))
                cov = sum((x - r_mean) * (y - btc_mean) for x, y in zip(r, btc_rets)) / max(1, n_btc)
                beta = cov / btc_var
                resid = [x - beta * y for x, y in zip(r, btc_rets)]
                new_rets[sym] = resid
            rets = new_rets

    print(f"[clusters] computing pairwise correlation matrix ({len(kept)} × {len(kept)})…")
    mat = correlation_matrix(rets, kept)

    # BTC correlation profile (everything is BTC-beta to some degree)
    if "BTCUSDT" in kept:
        btc_i = kept.index("BTCUSDT")
        btc_corrs = [(kept[j], mat[btc_i][j]) for j in range(len(kept)) if j != btc_i]
        btc_corrs.sort(key=lambda x: -x[1])
        print(f"\n[BTC beta] top-10 most correlated with BTCUSDT:")
        for s, c in btc_corrs[:10]:
            print(f"  {s:<14} corr={c:+.3f}")
        print(f"[BTC beta] bottom-5 least correlated with BTCUSDT:")
        for s, c in btc_corrs[-5:]:
            print(f"  {s:<14} corr={c:+.3f}")

    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
    all_clusters_by_thr: dict[float, list[list[str]]] = {}
    for thr in thresholds:
        clusters = cluster_at_threshold(kept, mat, thr)
        # Keep only multi-member clusters
        multi = [c for c in clusters if len(c) >= 2]
        all_clusters_by_thr[thr] = multi
        print(f"\n[threshold {thr:.2f}] multi-member clusters: {len(multi)}, "
              f"largest: {max((len(c) for c in multi), default=0)}, "
              f"covered coins: {sum(len(c) for c in multi)}")

    # Deep analysis at report threshold
    print(f"\n{'='*78}")
    print(f"DEEP ANALYSIS @ corr ≥ {args.report_threshold}")
    print(f"{'='*78}")
    multi = all_clusters_by_thr.get(args.report_threshold, [])
    if not multi:
        print("(no multi-member clusters at this threshold)")
        return

    summary_for_save: list[dict] = []
    for ci, cluster in enumerate(multi, 1):
        if len(cluster) < 2: continue
        intra = intra_cluster_corr(cluster, kept, mat)
        leader, leadinfo = find_lead(cluster, rets, max_lag=3)
        vol_profile = {s: float(sum(abs(r) for r in rets[s]) / len(rets[s]) * 100.0)
                       for s in cluster}
        max_vol = max(vol_profile.values())
        print(f"\n── Cluster #{ci} (n={len(cluster)}, intra-corr={intra:+.3f}) ──")
        print(f"  Members:  {', '.join(cluster)}")
        print(f"  Leader:   {leader}  ({leadinfo.get('score',{}).get(leader,0)} lead-wins)")
        if len(cluster) <= 8:
            print(f"  Volatility (avg |log-ret| ×100 = pct):")
            for s in sorted(cluster, key=lambda s: -vol_profile[s]):
                marker = " ◄ most volatile" if vol_profile[s] == max_vol else ""
                print(f"    {s:<14} {vol_profile[s]:.4f}%{marker}")
        summary_for_save.append({
            "cluster_id": ci,
            "members": cluster,
            "intra_corr": round(intra, 4),
            "leader": leader,
            "lead_score": leadinfo.get("score", {}),
            "volatility": {k: round(v, 5) for k, v in vol_profile.items()},
        })

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "days": args.days,
            "tf": args.tf,
            "threshold": args.report_threshold,
            "n_coins": len(kept),
            "clusters": summary_for_save,
        }, indent=2), encoding="utf-8")
        print(f"\n[save] wrote {args.save}")


if __name__ == "__main__":
    main()
