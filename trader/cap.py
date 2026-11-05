"""CAP -- market-wide capitulation basket (capitulation-basket-cap-1104-spec.md §3-§4).

  breadth   share of the day's universe whose log return over W bars is <= -Y, on every 15m grid bar; NaN below
            min_pairs pairs with both bars
  events    share >= X; the next event not before the hold of the previous one has ended
  basket    equal-weight buy of the universe at the close of bar i + 1, exit hold_bars later (lq.trade: costs, -50% on
            a delisting inside the hold)
  mean_ci   mean event return, iid bootstrap over events (they do not overlap); no interval below 10 events
  book_marked  daily R of the event book marked at every 00:00 UTC (protocol-fixes-1105-spec.md P5)
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import lq                         # noqa: E402

BAR = lq.BAR
DAY = lq.DAY
MIN_PAIRS = 50
MIN_CI_N = 10


def breadth(grid, pairs: dict, uni: dict, W: int, Y: float, min_pairs: int = MIN_PAIRS) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.int64)
    n = len(grid)
    hit, cnt = np.zeros(n), np.zeros(n)
    for sym, (t, c) in pairs.items():
        t = np.asarray(t, dtype=np.int64)
        c = np.asarray(c, dtype=float)
        if len(t) <= W:
            continue
        back = t - W * BAR
        j = np.searchsorted(t, back)
        jc = np.minimum(j, len(t) - 1)
        ok = (j < len(t)) & (t[jc] == back)
        lr = np.full(len(t), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            lr[ok] = np.log(c[ok] / c[jc[ok]])
        g = np.searchsorted(grid, t)
        gc = np.minimum(g, n - 1)
        on_grid = (g < n) & (grid[gc] == t)
        ud, inv = np.unique(t // DAY, return_inverse=True)
        member = np.array([sym in uni.get(int(d), ()) for d in ud], dtype=bool)
        m = np.isfinite(lr) & on_grid & member[inv]
        np.add.at(cnt, g[m], 1.0)
        np.add.at(hit, g[m & (lr <= -Y)], 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt >= min_pairs, hit / np.maximum(cnt, 1), np.nan)


def events(share, X: float, hold_bars: int) -> list:
    share = np.asarray(share, dtype=float)
    out, nxt = [], -1
    with np.errstate(invalid="ignore"):
        idx = np.flatnonzero(share >= X)
    for i in idx:
        if i >= nxt:
            out.append(int(i))
            nxt = int(i) + 1 + int(hold_bars)
    return out


def basket(grid, i: int, hold_bars: int, pairs: dict, syms, fm):
    grid = np.asarray(grid, dtype=np.int64)
    if i + 1 >= len(grid):
        return None
    te = int(grid[i + 1])
    rets, marks = [], {}
    for s in syms:
        p = pairs.get(s)
        if p is None:
            continue
        t, c = np.asarray(p[0], dtype=np.int64), np.asarray(p[1], dtype=float)
        e = int(np.searchsorted(t, te))
        if e >= len(t) or int(t[e]) != te:
            continue
        x = lq.trade(t, c, e - 1, hold_bars, fm, delisted=int(t[-1]) < te + hold_bars * BAR)
        if x is not None:
            rets.append(x["ret_pct"])
            for d, v in (x.get("marks") or {}).items():
                marks.setdefault(d, []).append(v)
    if not rets:
        return None
    return {"ret_pct": float(np.mean(rets)), "n_pairs": len(rets), "entry_close_ms": te + BAR,
            "exit_close_ms": te + BAR + hold_bars * BAR,
            "marks": {d: float(np.mean(v)) for d, v in marks.items()}}       # basket value / cost at 00:00 UTC


def mean_ci(rets, alpha: float = 0.05, seed: int = 29, boot: int = 4000) -> dict:
    v = [float(x) for x in rets]
    n = len(v)
    if not n:
        return {"n": 0}
    if n < MIN_CI_N:                                     # protocol-fixes-1105-spec.md P7
        return {"n": n, "mean_pct": round(sum(v) / n, 4), "lo95": None, "hi95": None, "ci_level": 1 - alpha,
                "ci_method": "too_few"}
    rnd = random.Random(seed)
    bs = sorted(sum(rnd.choices(v, k=n)) / n for _ in range(boot))
    return {"n": n, "mean_pct": round(sum(v) / n, 4), "lo95": round(bs[int(alpha / 2 * boot)], 4),
            "hi95": round(bs[min(boot - 1, int((1 - alpha / 2) * boot))], 4), "ci_level": 1 - alpha,
            "ci_method": "iid_bootstrap_percentile"}


def book_marked(events, days) -> list:
    """[(day, R %)]: every event holds the whole capital and is marked at each 00:00 UTC by its "marks" (value per
    unit of cost); overlapping events share the day equally (protocol-fixes-1105-spec.md P5)."""
    from datetime import datetime, timezone

    def dms(day):
        return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    out = []
    for day in days:
        a, b = dms(day), dms(day) + DAY
        rs = []
        for e in events:
            en, ex = int(e["entry_close_ms"]), int(e["exit_close_ms"])
            if ex <= a or en >= b:
                continue
            mk = {int(k): float(v) for k, v in (e.get("marks") or {}).items()}
            v0 = 1.0 if en >= a else mk.get(a)
            v1 = (1 + float(e["ret_pct"]) / 100) if ex <= b else mk.get(b)
            if v0 and v1 is not None:
                rs.append((v1 / v0 - 1) * 100)
        out.append((day, float(np.mean(rs)) if rs else 0.0))
    return out
