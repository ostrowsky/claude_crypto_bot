"""FC-W -- passive funding carry on the N most liquid pairs (fc-wide-carry-fcw-1104-spec.md §2-§3).

  chooser(N)   first N of the eligible universe (carry.eligible: spot liquidity order); a held pair is kept while it
               is in the first 2N; free slots are filled in liquidity order
  candidates   FCW_N4 / FCW_N6 / FCW_N10: carry.CarryConfig(K = N, L = 1, margin stop 50%)
  map_perp     archive USD-M perp -> spot pair (multiplier prefix 1000 / 1000000)
  ratio_ok     median perp / mult / spot within 1 +- tol (the same check as experiment_fc.build_market)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import carry as C                 # noqa: E402

NS = (4, 6, 10)
MULT_PREFIXES = (("1000000", 1_000_000), ("1000", 1000))


def chooser(N: int):
    def choose(ctx) -> list:
        uni = list(ctx.universe)
        top2 = set(uni[:2 * N])
        keep = [s for s in uni if s in ctx.held and s in top2][:N]
        out = list(keep)
        for s in uni:
            if len(out) >= N:
                break
            if s not in out:
                out.append(s)
        return out
    return choose


def candidates() -> dict:
    return {f"FCW_N{n}": C.CarryConfig(W=3, K=n, theta_in=0.0) for n in NS}


def map_perp(perp: str, spot_syms):
    if not perp.endswith("USDT") or "_" in perp:
        return None
    if perp in spot_syms:
        return perp, 1
    for pre, m in MULT_PREFIXES:
        if perp.startswith(pre) and perp[len(pre):] in spot_syms:
            return perp[len(pre):], m
    return None


def ratio_ok(spot_close, perp_close, mult, tol) -> bool:
    s = np.asarray(spot_close, dtype=float)
    p = np.asarray(perp_close, dtype=float) / float(mult)
    ok = np.isfinite(s) & np.isfinite(p) & (s > 0)
    if not ok.any():
        return False
    return abs(float(np.median(p[ok] / s[ok])) - 1) <= tol
