"""RV -- all capital in BTC or ETH, chosen daily by their relative N-day return (btc-eth-rotation-rv-1028-spec.md §2).

  holdings   0 = BTC, 1 = ETH for decision rows r0..r1-1, from closes of rows < r (ties / no data -> BTC)
  simulate   one position; a switch sells everything and buys the other at first_close; mark at 00:00 by close d-1;
             window edge: last mark, then sold
  shift      circular shift of the holding sequence (random control)
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

import daily_revenue as DR        # noqa: E402
import xsmom as XS                # noqa: E402

PAIR = ("BTCUSDT", "ETHUSDT")
START_EQUITY = 1_000_000.0


def candidates() -> dict:
    return {f"RV_{k}_{n}": {"kind": k, "N": n} for k in ("mom", "rev") for n in (14, 28, 56)}


def _ret(c, r, N):
    a, b = r - 1, r - 1 - N
    if b < 0 or a >= len(c) or not (math.isfinite(c[a]) and math.isfinite(c[b])) or c[b] <= 0:
        return None
    return c[a] / c[b] - 1


def holdings(panel, cand: dict, r0: int, r1: int) -> np.ndarray:
    cb, ce = panel.close[:, panel.col[PAIR[0]]], panel.close[:, panel.col[PAIR[1]]]
    out = np.zeros(r1 - r0, dtype=int)
    for i, r in enumerate(range(r0, r1)):
        a, b = _ret(cb, r, cand["N"]), _ret(ce, r, cand["N"])
        if a is None or b is None or a == b:
            continue
        eth_stronger = b > a
        out[i] = int(eth_stronger if cand["kind"] == "mom" else not eth_stronger)
    return out


def shift(h, seed: int, min_shift: int = 30) -> np.ndarray:
    h = np.asarray(h)
    n = len(h)
    lo = min(min_shift, max(1, n // 4))
    return np.roll(h, random.Random(seed).randint(lo, max(lo, n - lo)))


def simulate(panel, h, fm, day_from: str, day_to: str, start_equity: float = START_EQUITY) -> dict:
    W = panel.cut(day_to)
    r0, r1 = W.row(day_from), W.row(day_to)
    cash, sym, qty, basis = float(start_equity), None, 0.0, 0.0
    marks, pnls, last_px = {}, [], None
    n_sw = 0

    def sell(raw):
        nonlocal cash, sym, qty, basis
        px = fm.sell(raw, qty * raw, None)
        proceeds = qty * px * (1 - fm.taker_fee)
        pnls.append(proceeds - basis)
        cash += proceeds
        sym, qty, basis = None, 0.0, 0.0

    for i, r in enumerate(range(max(r0, 0), r1)):
        day = XS.day_str(W.d0 + r)
        if sym is not None and r - 1 >= 0:
            c = W.close[r - 1, W.col[sym]]
            if math.isfinite(c):
                last_px = float(c)
        marks[day] = cash + (qty * last_px if sym is not None else 0.0)
        want = PAIR[int(h[r - r0])]
        if want == sym:
            continue
        jw = W.col[want]
        if not (W.has_first[r, jw] and math.isfinite(W.first_close[r, jw])):
            continue
        if sym is not None:
            j = W.col[sym]
            if not (W.has_first[r, j] and math.isfinite(W.first_close[r, j])):
                continue
            sell(float(W.first_close[r, j]))
            n_sw += 1
        raw = float(W.first_close[r, jw])
        fill = fm.buy(raw, cash, None)
        sym, qty, basis, last_px = want, cash / fill * (1 - fm.taker_fee), cash, raw
        cash = 0.0
    if r1 > max(r0, 0):
        if sym is not None:
            c = W.close[r1 - 1, W.col[sym]]
            if math.isfinite(c):
                last_px = float(c)
        marks[day_to] = cash + (qty * last_px if sym is not None else 0.0)
        if sym is not None:
            sell(last_px)
    days = sorted(marks)
    series = [(a, (marks[b] / marks[a] - 1) * 100) for a, b in zip(days, days[1:]) if marks[a] > 0]
    return {"series": series, "agg": DR.aggregate(series), "final_equity": cash, "n_switches": n_sw,
            "eth_share": float(np.mean(h)) if len(h) else 0.0,
            "conservation_ok": abs((cash - start_equity) - sum(pnls)) < 1e-6 * max(1.0, start_equity) and sym is None}


def run(panel, cand: dict, fm, day_from: str, day_to: str, start_equity: float = START_EQUITY) -> dict:
    W = panel.cut(day_to)
    h = holdings(W, cand, W.row(day_from), W.row(day_to))
    out = simulate(panel, h, fm, day_from, day_to, start_equity)
    out["holdings"] = h
    return out
