"""Decoupling signal — shadow feature (2026-05-07).

Validated hypothesis (docs/reports/2026-05-07-decoupling-validation.md):
a watchlist coin that is BOTH high-volatility AND decoupled from the
market (low trailing correlation to the watchlist basket) is ~2x more
likely to make a forward idiosyncratic big move (a top-gainer rocket)
than a coin moving with the market.

Backtest summary (365d, 1h, train/holdout, no lookahead):
  - universe vol-controlled lift (high-vol tercile): +1.85pp / +3.14pp
  - strict-rocket flag (top-5/day or daily>=12%): precision ~10-13% vs
    base ~5.5% → LIFT x1.87 (holdout) / x2.25 (train), recall 11-22%.

This module computes the per-coin decoupling score live. It is wired in
SHADOW MODE ONLY: scores are logged on entry events for later bandit
shadow-replay validation; they do NOT change any entry/exit decision
until DECOUPLING_GATE_ENABLED is implemented & shadow-validated.

The flag is a PRIORITY signal, not a detector — ~8 misses per hit — so
the design intent is a bandit/ranker context feature, never a hard gate.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

try:
    import config  # type: ignore
except Exception:  # pragma: no cover
    config = None  # type: ignore


def _pearson(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n < 10:
        return 0.0
    ax, bx = a[-n:], b[-n:]
    ma, mb = sum(ax) / n, sum(bx) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in ax) / n)
    sb = math.sqrt(sum((x - mb) ** 2 for x in bx) / n)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    cov = sum((ax[i] - ma) * (bx[i] - mb) for i in range(n)) / n
    return max(-1.0, min(1.0, cov / (sa * sb)))


def _realized_vol(rets: List[float]) -> float:
    if not rets:
        return 0.0
    return math.sqrt(sum(x * x for x in rets) / len(rets))


def scores_from_rets(
    rets_map: Dict[str, List[float]],
    *,
    vol_q: float = 0.66,
    corr_max: float = 0.60,
) -> Dict[str, dict]:
    """Pure, testable core. Given {sym: log_returns}, return per-sym:
    {trailing_corr, vol, vol_pctile, decoupling_score, flag}.

    basket = equal-weight mean return across all provided syms.
    decoupling_score = (1 - normalized_corr) * vol_pctile  ∈ [0,1]-ish,
      higher = more "high-vol AND decoupled".
    flag = vol_pctile >= vol_q AND trailing_corr <= corr_max.
    """
    syms = [s for s, r in rets_map.items() if r and len(r) >= 10]
    if len(syms) < 5:
        return {}
    n = min(len(rets_map[s]) for s in syms)
    basket = [sum(rets_map[s][-n:][i] for s in syms) / len(syms) for i in range(n)]

    vols = {s: _realized_vol(rets_map[s][-n:]) for s in syms}
    sorted_vols = sorted(vols.values())
    m = len(sorted_vols)

    def vpct(v: float) -> float:
        lo = 0
        for x in sorted_vols:
            if x <= v:
                lo += 1
        return lo / m

    out: Dict[str, dict] = {}
    for s in syms:
        c = _pearson(rets_map[s][-n:], basket)
        vp = vpct(vols[s])
        # normalized corr to [0,1] where 0 = perfectly coupled, 1 = fully decoupled
        decoup = max(0.0, min(1.0, (1.0 - c) / 2.0))
        score = round(decoup * vp, 4)
        flag = bool(vp >= vol_q and c <= corr_max)
        out[s] = {
            "trailing_corr": round(c, 4),
            "vol": round(vols[s], 6),
            "vol_pctile": round(vp, 4),
            "decoupling_score": score,
            "flag": flag,
        }
    return out


async def compute_scores(
    session,
    syms: List[str],
    *,
    interval: Optional[str] = None,
    limit: Optional[int] = None,
    vol_q: Optional[float] = None,
    corr_max: Optional[float] = None,
) -> Dict[str, dict]:
    """Live entry point. Fetches closes (reusing corr_guard's fetcher) and
    returns the per-sym score dict. Fail-open: returns {} on any error."""
    try:
        import correlation_guard as cg
    except Exception:
        return {}
    g = (lambda k, d: float(getattr(config, k, d)) if config else d)
    gi = (lambda k, d: int(getattr(config, k, d)) if config else d)
    interval = interval or (str(getattr(config, "DECOUPLING_TF", "1h")) if config else "1h")
    limit = limit or gi("DECOUPLING_WINDOW_BARS", 168)
    vol_q = vol_q if vol_q is not None else g("DECOUPLING_VOL_PCTILE_MIN", 0.66)
    corr_max = corr_max if corr_max is not None else g("DECOUPLING_CORR_MAX", 0.60)

    rets_map: Dict[str, List[float]] = {}
    import asyncio
    tasks = {s: asyncio.create_task(cg._fetch_closes_async(session, s, interval, limit))
             for s in syms}
    for s, t in tasks.items():
        try:
            closes = await t
        except Exception:
            closes = None
        if closes and len(closes) >= 10:
            rets_map[s] = cg._log_returns(closes)
    return scores_from_rets(rets_map, vol_q=vol_q, corr_max=corr_max)


if __name__ == "__main__":
    # self-test on synthetic data: one decoupled high-vol coin should flag
    import random
    random.seed(0)
    n = 200
    market = [random.gauss(0, 0.01) for _ in range(n)]
    rmap = {}
    # coupled low-vol coins
    for i in range(8):
        rmap[f"COUP{i}"] = [market[t] + random.gauss(0, 0.002) for t in range(n)]
    # one decoupled high-vol coin
    rmap["DECOUP"] = [random.gauss(0, 0.03) for _ in range(n)]
    res = scores_from_rets(rmap, vol_q=0.66, corr_max=0.60)
    d = res["DECOUP"]
    print("DECOUP:", d)
    print("COUP0 :", res["COUP0"])
    assert d["flag"] is True, "decoupled high-vol coin should be flagged"
    assert res["COUP0"]["flag"] is False, "coupled coin should NOT be flagged"
    assert d["decoupling_score"] > res["COUP0"]["decoupling_score"]
    print("self-test PASSED")
