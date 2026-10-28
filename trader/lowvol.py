"""N8 -- low-volatility chooser for xsmom.rotate (lowvol-alts-n8-1028-spec.md §2).

  vol_scores   sample sd of the daily log returns of the N days before d (N returns, all days complete)
  pick         the K lowest scores, ties by name
Days are 'YYYY-MM-DD' strings.
"""
from __future__ import annotations

import math

import numpy as np


def vol_scores(panel, day: str, syms, N: int) -> dict:
    r = panel.row(day)
    lo = r - 1 - N                         # closes of rows lo .. r-1 give N returns
    out = {}
    if lo < 0 or r - 1 >= panel.n:
        return out
    for s in syms:
        j = panel.col.get(s)
        if j is None:
            continue
        c = panel.close[lo:r, j]
        ok = panel.complete[lo:r, j]
        if len(c) != N + 1 or not np.all(ok) or not np.all(np.isfinite(c)) or np.any(c <= 0):
            continue
        sd = float(np.std(np.diff(np.log(c)), ddof=1))
        if math.isfinite(sd):
            out[s] = sd
    return out


def pick(scores: dict, K: int) -> list:
    return [s for s, _ in sorted(scores.items(), key=lambda kv: (kv[1], kv[0]))[:K]]


def lowvol_chooser(panel, K: int, N: int):
    def chooser(ctx) -> list:
        return pick(vol_scores(panel, ctx.day, ctx.universe, N), K)
    return chooser
