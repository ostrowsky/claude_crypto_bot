"""Registry of learned exit policies and the position path they read (trader-learning-loop-f-1020-spec.md F2).

A TradingPolicy is a frozen, hashed dataclass, so it cannot hold a fitted model: Exits(kind="learned",
learned_id=...) names an entry of REGISTRY instead. Replay only -- the executor does not use this kind.
"""
from __future__ import annotations

import math

import numpy as np

ROUNDTRIP = 2 * (0.001 + 0.0008)          # taker fee + 8 bps per side (fill model level 1), both sides
REGISTRY: dict = {}


def register(agent, name: str) -> str:
    REGISTRY[name] = agent
    return name


def get(learned_id):
    return REGISTRY[learned_id]


def path_from_grid(grid, cand: dict, x4_plan=None, max_bars: int = 96):
    """Closes from the entry bar ie over max_bars bars, with the X4 exit offset; None when a bar before the X4
    exit has no close (a gap the learned exit could not have seen through)."""
    import trading_policy as TP
    from dataclasses import replace
    j, ie = cand.get("j"), cand.get("ie")
    if j is None or ie is None:
        return None
    if x4_plan is None:
        x4p = replace(TP.CHAMPION, exits=TP.Exits(kind="p1_trail", floor_override=0.08, max_bars=max_bars))
        x4_plan = TP.exit_plan(x4p, grid, cand, floor_of=lambda m: 0.0)
        if x4_plan is None:
            return None
    off = int(x4_plan[0]) - int(ie)
    n = min(max_bars + 1, grid.N - ie)
    close = np.array([grid.C[ie + i, j] for i in range(n)], dtype=float)
    if off < 1 or off >= n or not np.all(np.isfinite(close[: off + 1])):
        return None
    ts = cand.get("ts")
    key = f"{cand.get('sym')}|{cand.get('tf')}|{ts.isoformat() if hasattr(ts, 'isoformat') else ts}"
    return {"key": key, "close": close, "x4_exit": off, "cost": ROUNDTRIP}


_ = math
