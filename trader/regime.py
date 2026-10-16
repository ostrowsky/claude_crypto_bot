"""Market regime flags per 15m bar (x4-family-money-1015-spec.md §3.1).

btc_down    -- BTCUSDT close below its EMA200 on 15m (~ EMA50 on 1h, the bot's btc_vs_ema50 idea);
breadth_low -- fewer than half of the grid pairs close above their own EMA200 on 15m.

Causal: the flag at bar q uses closes up to q only. A pair's EMA is valid after EMA_BARS finite closes;
a NaN close leaves its EMA as is and the pair out of the share at that bar. With fewer than MIN_PAIRS
valid pairs, or without a valid BTC, the flag is False: no data -> no block.
"""
from __future__ import annotations

import numpy as np

EMA_BARS = 200
MIN_PAIRS = 20
BTC = "BTCUSDT"


def flags(grid, ema_bars: int = EMA_BARS, min_pairs: int = MIN_PAIRS, btc: str = BTC) -> dict:
    C = np.asarray(grid.C, dtype=float)
    n, s = C.shape
    a = 2.0 / (ema_bars + 1)
    ema = np.zeros(s)
    cnt = np.zeros(s, dtype=np.int64)
    jb = grid.col.get(btc) if hasattr(grid, "col") else None
    btc_down = np.zeros(n, dtype=bool)
    breadth_low = np.zeros(n, dtype=bool)
    for q in range(n):
        row = C[q]
        m = np.isfinite(row)
        first = m & (cnt == 0)
        ema[first] = row[first]
        upd = m & ~first
        ema[upd] += a * (row[upd] - ema[upd])
        cnt[m] += 1
        valid = m & (cnt >= ema_bars)
        nv = int(valid.sum())
        if nv >= min_pairs:
            breadth_low[q] = (row[valid] > ema[valid]).sum() / nv < 0.5
        if jb is not None and valid[jb]:
            btc_down[q] = row[jb] < ema[jb]
    return {"btc_down": btc_down, "breadth_low": breadth_low}


def blocks(name: str, f) -> bool:
    """R1 btc_down | R2 breadth_low | R3 both | R4 either. f = flags of one bar or None (no data -> False)."""
    if name not in ("R1", "R2", "R3", "R4"):
        raise ValueError(name)
    if not f:
        return False
    b, w = bool(f.get("btc_down")), bool(f.get("breadth_low"))
    return {"R1": b, "R2": w, "R3": b and w, "R4": b or w}[name]
