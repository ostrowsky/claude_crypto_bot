"""RB -- periodic rebalancing to fixed weights vs holding the initial weights (rebalancing-rb-1029-spec.md §2).

  simulate(px, weights, period, fm)  one mark per day at that day's price BEFORE trading; day 0 buys the weights,
                                     then every `period` days sells what is above target first, then buys what is
                                     below; orders < min_order USDT are skipped; "USDT" in weights is plain cash;
                                     period None = buy-and-hold of the initial weights
  fees = execution cost vs the mid price (taker fee + half-spread), turnover = traded notional at the mid
"""
from __future__ import annotations

import math

import numpy as np


def simulate(px: dict, weights: dict, period, fm, start: float = 1_000_000.0, min_order: float = 10.0) -> dict:
    syms = [s for s in weights if s != "USDT"]
    n = len(next(iter(px.values())))
    cash, qty = float(start), {s: 0.0 for s in syms}
    last = {s: None for s in syms}
    equity = np.zeros(n)
    trade_days, fees, turnover = [], 0.0, 0.0
    for d in range(n):
        ok = True
        for s in syms:
            v = float(px[s][d])
            if math.isfinite(v) and v > 0:
                last[s] = v
            else:
                ok = False
        equity[d] = cash + sum(qty[s] * (last[s] or 0.0) for s in syms)
        due = d == 0 or (period is not None and d % int(period) == 0)
        if not due or not ok:
            continue
        V = equity[d]
        traded = False
        for s in syms:                                              # sells first
            excess = qty[s] * last[s] - weights[s] * V
            if excess >= min_order and d > 0:
                q = excess / last[s]
                proceeds = q * fm.sell(last[s], excess, None) * (1 - fm.taker_fee)
                fees += excess - proceeds
                turnover += excess
                qty[s] -= q
                cash += proceeds
                traded = True
        for s in syms:                                              # then buys, limited by cash
            need = min(weights[s] * V - qty[s] * last[s], cash)
            if need >= min_order:
                q = need / fm.buy(last[s], need, None) * (1 - fm.taker_fee)
                fees += need - q * last[s]
                turnover += need
                qty[s] += q
                cash -= need
                traded = True
        if traded:
            trade_days.append(d)
    final = cash + sum(qty[s] * (last[s] or 0.0) for s in syms)
    return {"equity": equity, "final": final, "trade_days": trade_days, "fees": fees, "turnover": turnover,
            "cash": cash, "qty": dict(qty)}
