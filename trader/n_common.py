"""What N1-N5 share (n-family-experiment-designs-1105-spec.md §1, n3-1106-spec.md §1).

  ci / cluster_ci   block bootstrap with the design's B = 20 000, ci_level and ci_method on every interval
  required_n        §1.3, sample size at a preset effect (never the observed one)
  Volumes           §1.5, point-in-time rank by quote volume over [d-30, d-1]
  fetch_months      archive months of one prefix into one npz, resumable; symbols may be non-ASCII
  fetch_usdt_1d     1d bars of every USDT perp of the archive, nothing after 2025-12
"""
from __future__ import annotations

import json
import math
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import archive_bh as A            # noqa: E402
import trader_config as TC        # noqa: E402

BOOT = 20_000
SEED = 29
MIN_N = 10
MIN_TAIL = 20
LAST_MONTH = "2025-12"
STORE_1D = TC.RUNTIME / "usdt_1d"


def _tail_interval(means: np.ndarray, m: float, alpha: float):
    s = np.sort(means)
    b = len(s)
    if int(alpha / 2 * b) >= MIN_TAIL:
        return float(s[int(alpha / 2 * b)]), float(s[min(b - 1, int((1 - alpha / 2) * b))]), "block_bootstrap_percentile"
    z = NormalDist().inv_cdf(1 - alpha / 2)
    se = float(means.std(ddof=1))
    return m - z * se, m + z * se, "normal_block_se"


def _block_sums(x: np.ndarray, b: int) -> np.ndarray:
    c = np.r_[0.0, np.cumsum(x)]
    return c[b:] - c[:-b]


def _boot_sums(x: np.ndarray, block: int, boot: int, rng_starts: np.ndarray, k: int, r: int) -> np.ndarray:
    full = _block_sums(x, block)
    out = full[rng_starts[:, :k - 1]].sum(axis=1) if k > 1 else np.zeros(boot)
    if r:
        part = _block_sums(x, r)
        out = out + part[np.minimum(rng_starts[:, k - 1], len(part) - 1)]
    return out


def _starts(n: int, block: int, boot: int, seed: int):
    k = n // block + (1 if n % block else 0)
    r = n - (k - 1) * block
    return np.random.default_rng(seed).integers(0, n - block + 1, size=(boot, k)), k, r


def ci(x, alpha: float = 0.05, block: int = 20, boot: int = BOOT, seed: int = SEED) -> dict:
    a = np.asarray(x, dtype=float)
    n = len(a)
    m = float(a.mean()) if n else float("nan")
    base = {"mean": m, "ci_level": round(1 - alpha, 6), "block": block, "boot": boot, "n": n}
    if n < MIN_N:
        return {**base, "lo": None, "hi": None, "ci_method": "too_few"}
    b = max(1, min(block, n))
    st, k, r = _starts(n, b, boot, seed)
    means = _boot_sums(a, b, boot, st, k, r) / n
    lo, hi, method = _tail_interval(means, m, alpha)
    return {**base, "lo": lo, "hi": hi, "ci_method": method}


def cluster_ci(values, clusters, alpha: float = 0.05, block: int = 1, boot: int = BOOT, seed: int = SEED) -> dict:
    """Mean per event; whole clusters are resampled in their order of appearance."""
    vals = np.asarray(values, dtype=float)
    order, idx = {}, []
    for c in clusters:
        idx.append(order.setdefault(c, len(order)))
    idx = np.asarray(idx, dtype=np.int64)
    n = len(order)
    m = float(vals.mean()) if len(vals) else float("nan")
    base = {"mean": m, "ci_level": round(1 - alpha, 6), "block": block, "boot": boot, "n": int(len(vals)), "n_clusters": n}
    if n < MIN_N:
        return {**base, "lo": None, "hi": None, "ci_method": "too_few"}
    s = np.bincount(idx, weights=vals, minlength=n)
    c = np.bincount(idx, minlength=n).astype(float)
    b = max(1, min(block, n))
    st, k, r = _starts(n, b, boot, seed)
    means = _boot_sums(s, b, boot, st, k, r) / _boot_sums(c, b, boot, st, k, r)
    lo, hi, method = _tail_interval(means, m, alpha)
    return {**base, "lo": lo, "hi": hi, "ci_method": method}


def required_n(sigma: float, delta: float, alpha: float, power: float = 0.80, m: float = 1.0, rho: float = 0.0):
    nd = NormalDist()
    n = ((nd.inv_cdf(1 - alpha / 2) + nd.inv_cdf(power)) * sigma / abs(delta)) ** 2
    return n, n * (1 + (m - 1) * rho)


class Volumes:
    """Quote volume by symbol and UTC day; days with zero or no volume count as no data."""

    def __init__(self, by_symbol: dict, first_day: dict):
        days = sorted({d for x in by_symbol.values() for d in x})
        self.d0 = date.fromisoformat(days[0]) if days else date(2000, 1, 1)
        self.n_days = (date.fromisoformat(days[-1]) - self.d0).days + 1 if days else 0
        self.syms = sorted(by_symbol)
        self.first = first_day
        self.q = np.zeros((self.n_days, len(self.syms)))
        for j, s in enumerate(self.syms):
            for d, v in by_symbol[s].items():
                if v and v > 0:
                    self.q[(date.fromisoformat(d) - self.d0).days, j] = v
        self.first_ix = np.array([(date.fromisoformat(first_day[s]) - self.d0).days if s in first_day else 10 ** 9 for s in self.syms])

    def ranks(self, day: str, exclude=()) -> dict:
        i = (date.fromisoformat(day) - self.d0).days
        lo, hi = max(0, i - 30), min(self.n_days, max(0, i))
        w = self.q[lo:hi]
        if not len(w):
            return {}
        days_with = (w > 0).sum(axis=0)
        total = w.sum(axis=0)
        ok = [j for j, s in enumerate(self.syms) if days_with[j] >= 25 and self.first_ix[j] <= i - 30 and s not in exclude]
        ok.sort(key=lambda j: (-total[j], self.syms[j]))
        return {self.syms[j]: r + 1 for r, j in enumerate(ok)}

    def top(self, day: str, k: int, exclude=()) -> list:
        r = self.ranks(day, exclude)
        return [s for s, v in sorted(r.items(), key=lambda x: x[1]) if v <= k]


def quote_path(sym: str) -> str:
    return urllib.parse.quote(sym)


def month_days(month: str) -> tuple:
    y, m = int(month[:4]), int(month[5:7])
    first = date(y, m, 1)
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    return first.isoformat(), (nxt - timedelta(days=1)).isoformat()


def months_between(first: str, last: str) -> list:
    y, m = int(first[:4]), int(first[5:7])
    out = []
    while f"{y:04d}-{m:02d}" <= last:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch_months(prefix: str, out: Path, parser, keys, ex, first: str = "2019-01", last: str = LAST_MONTH) -> dict:
    """Months in [first, last] of `prefix` (an unquoted archive path) into one npz; a months file makes it resumable."""
    out.parent.mkdir(parents=True, exist_ok=True)
    done_path = out.with_name(out.name + ".months.json")
    done = json.loads(done_path.read_text(encoding="utf-8")) if done_path.exists() else {}
    ks = A.list_all(urllib.parse.quote(prefix), delimiter=False)
    months = A.months_in_range(ks, first, last)
    by_month = {}
    for k in ks:
        for m in months:
            if k.endswith(f"-{m}.zip"):
                by_month[m] = k
    todo = [m for m in months if m not in done]

    def one(m):
        raw = A.http_get(f"https://{A.FILES}/{urllib.parse.quote(by_month[m])}")
        return m, parser(A._zip_text(raw)) if raw else None

    parts = list(ex.map(one, todo))
    old = None
    if out.exists():
        with np.load(out) as z:
            old = {k: z[k] for k in z.files}
    arrays = ([old] if old else []) + [p for _, p in parts if p is not None]
    if arrays:
        A._save(out, arrays, keys)
    done.update({m: p is not None for m, p in parts})
    done_path.write_text(json.dumps(done), encoding="utf-8")
    return {"months": len(months), "new": len(todo), "missing": [m for m, p in parts if p is None]}


def usdt_symbols() -> list:
    items = A.list_all("data/futures/um/monthly/klines/", delimiter=True)
    return sorted(s for s in (i.rstrip("/").split("/")[-1] for i in items) if s.endswith("USDT"))


def fetch_usdt_1d(log=print, workers: int = 12) -> dict:
    from n1_usdc import parse_daily
    tot = {"symbols": 0, "months": 0, "new": 0, "missing": 0, "empty": 0}
    with ThreadPoolExecutor(workers) as ex, ThreadPoolExecutor(4) as outer:
        def one(s):
            return s, fetch_months(f"data/futures/um/monthly/klines/{s}/1d/", STORE_1D / f"{s}.npz", parse_daily, ("qv", "count"), ex)
        for s, r in outer.map(one, usdt_symbols()):
            tot["months"] += r["months"]
            tot["new"] += r["new"]
            tot["missing"] += len(r["missing"])
            tot["empty"] += not r["months"]
            tot["symbols"] += 1
            if tot["symbols"] % 100 == 0:
                log(ascii(tot))
    log(json.dumps(tot))
    return tot


def load_volumes(last_day: str = "2025-12-31") -> Volumes:
    from datetime import datetime, timezone
    by, first = {}, {}
    for p in STORE_1D.glob("*.npz"):
        with np.load(p) as z:
            t, q = z["t"], z["qv"]
        days = [datetime.fromtimestamp(int(x) / 1000, timezone.utc).strftime("%Y-%m-%d") for x in t]
        if not days:
            continue
        s = p.stem
        first[s] = days[0]
        by[s] = {d: float(v) for d, v in zip(days, q) if d <= last_day and math.isfinite(float(v))}
    return Volumes(by, first)


def pegged() -> set:
    """USDT perps of pegged bases: outside every rank of N1-N5 (n3-1106-spec.md §1)."""
    import trading_universe as TU
    bases = set(TU.STABLE_BASES) | set(A.OLD_PEGGED)
    return {b + "USDT" for b in bases}
