"""DIR -- pieces of the 4h / 1d direction-forecast test (direction-forecast-dir-1029-spec.md §3-§5).

  pair_features     per-pair features at every bar index from bars <= that index (no look-ahead)
  label             entry -> exit return; exit missing: delisted -> -0.5 (xsmom.DELIST_HAIRCUT), gap -> None
  block_first_close close of the 15m bar that opens a 4h block (the entry / exit fill of the 4h horizon)
  select            top floor(n / 10) by predicted probability, ties by name
  turnover_costs    c/2 when a pair enters the selection, c/2 when it leaves
  auc / auc_ci      Mann-Whitney AUC; CI by a moving-block bootstrap over decision days (binned scores)
  folds             purged walk-forward, 4 folds, 2-day embargo (purged_cv)
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DELIST_R = -0.5
N_FOLDS = 4
EMBARGO_DAYS = 2.0
BLOCK_DAYS = 5
N_BINS = 2000
FEATURES = ("r1", "r2", "r3", "r6", "r12", "r24", "vol24", "z1", "vq", "pos24")


def _lag_log(c, k):
    out = np.full(len(c), np.nan)
    if len(c) > k:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[k:] = np.log(c[k:] / c[:-k])
    return out


def _window(x, w):
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full((len(x), w), np.nan)
    if len(x) >= w:
        out[w - 1:] = sliding_window_view(x, w)
    return out


def pair_features(c, qv) -> dict:
    c = np.asarray(c, dtype=float)
    qv = np.asarray(qv, dtype=float)
    f = {f"r{k}": _lag_log(c, k) for k in (1, 2, 3, 6, 12, 24)}
    W = _window(f["r1"], 24)
    with np.errstate(invalid="ignore", divide="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vol = np.std(W, axis=1, ddof=1)
            vol[np.isnan(W).sum(axis=1) > 0] = np.nan
            f["vol24"] = vol
            f["z1"] = f["r1"] / vol
            Q = _window(qv, 24)
            f["vq"] = np.log(qv / np.nanmedian(Q, axis=1))
            C = _window(c, 24)
            lo, hi = np.nanmin(C, axis=1), np.nanmax(C, axis=1)
            f["pos24"] = np.where(hi > lo, (c - lo) / (hi - lo), np.nan)
    for k in f:
        f[k][~np.isfinite(f[k])] = np.nan
    return f


def label(entry, exit_, delisted: bool):
    if exit_ is None or not (isinstance(exit_, (int, float)) and math.isfinite(exit_)):
        return DELIST_R if delisted else None
    if entry is None or not math.isfinite(entry) or entry <= 0:
        return None
    return float(exit_) / float(entry) - 1


def block_first_close(t15, c15, block_start_ms: int):
    t15 = np.asarray(t15, dtype=np.int64)
    i = int(np.searchsorted(t15, block_start_ms))
    if i < len(t15) and int(t15[i]) == int(block_start_ms):
        v = float(np.asarray(c15)[i])
        return v if math.isfinite(v) else None
    return None


def select(scores: dict) -> list:
    k = len(scores) // 10
    return [s for s, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def turnover_costs(selections, c: float) -> list:
    out = []
    n = len(selections)
    for t, sel in enumerate(selections):
        prev = set(selections[t - 1]) if t > 0 else set()
        nxt = set(selections[t + 1]) if t + 1 < n else set()
        out.append({s: (c / 2 if s not in prev else 0.0) + (c / 2 if s not in nxt else 0.0) for s in sel})
    return out


def auc(y, s) -> float:
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(ss):                                     # average ranks over ties
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _auc_hist(pos, neg) -> float:
    n1, n0 = pos.sum(), neg.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    below = np.cumsum(neg) - neg                             # negatives in lower bins
    return float((pos * (below + neg / 2)).sum() / (n1 * n0))


def auc_ci(y, s, days, alpha: float = 0.05, seed: int = 29, boot: int = 2000):
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    days = np.asarray(days)
    point = auc(y, s)
    edges = np.unique(np.quantile(s, np.linspace(0, 1, N_BINS + 1)[1:-1]))
    b = np.searchsorted(edges, s, side="right")
    nb = len(edges) + 1
    ud, di = np.unique(days, return_inverse=True)
    D = len(ud)
    pos = np.zeros((D, nb))
    neg = np.zeros((D, nb))
    np.add.at(pos, (di[y], b[y]), 1)
    np.add.at(neg, (di[~y], b[~y]), 1)
    blk = max(1, min(BLOCK_DAYS, D))
    starts = D - blk + 1
    nblocks = int(math.ceil(D / blk))
    cp, cn = np.cumsum(np.vstack([np.zeros(nb), pos]), axis=0), np.cumsum(np.vstack([np.zeros(nb), neg]), axis=0)
    rnd = random.Random(seed)
    vals = []
    for _ in range(boot):
        st = [rnd.randrange(starts) for _ in range(nblocks)]
        P = sum(cp[a + blk] - cp[a] for a in st)
        N = sum(cn[a + blk] - cn[a] for a in st)
        vals.append(_auc_hist(P, N))
    vals = np.sort(np.asarray(vals))
    return point, float(vals[int(alpha / 2 * len(vals))]), float(vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))])


def folds(rows) -> list:
    import purged_cv as PC
    return PC.walk_forward(rows, n_folds=N_FOLDS, embargo_days=EMBARGO_DAYS)
