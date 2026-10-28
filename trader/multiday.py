"""MD row U -- event entries on the spot universe held H days (multiday-spot-md-1028-spec.md §2).

  events     BRK (close above the prior 20-day closing max), DIP (3-day pullback <= -8% above SMA50),
             VOL (quote volume >= 3x the prior 20-day median on an up day); rows < d only
  simulate   K slots of equity/K at the 00:00 mark; entries at first_close of d by strength into free slots,
             never into a held coin; exit at first_close of entry + H (or the next day with a 00:00 bar);
             delisting at a 50% haircut; window edge: last mark, then everything sold
  random     random_chooser(moves, seed): the same number of new entries on the same days, random buyable coins
Days are 'YYYY-MM-DD' strings.
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

BRK_DAYS = 20
DIP_SMA = 50
DIP_DAYS = 3
DIP_MIN = -0.08
VOL_DAYS = 20
VOL_MULT = 3.0
DELIST_HAIRCUT = 0.5
MIN_TRADE_USDT = 10.0


def day_gap(a: str, b: str) -> int:
    return XS.day_num(b) - XS.day_num(a)


def _fin(x) -> bool:
    return bool(np.all(np.isfinite(x)))


def events(panel, day: str, kind: str, syms) -> dict:
    r = panel.row(day)
    a = r - 1
    out = {}
    if a < 1 or a >= panel.n:
        return out
    for s in syms:
        j = panel.col.get(s)
        if j is None or not panel.complete[a, j]:
            continue
        c = panel.close[:, j]
        if kind == "BRK":
            lo = a - BRK_DAYS
            if lo < 0:
                continue
            win = c[lo:a]
            if not _fin(win) or not math.isfinite(c[a]):
                continue
            m = float(np.max(win))
            if c[a] > m > 0:
                out[s] = float(c[a] / m - 1)
        elif kind == "DIP":
            lo = r - DIP_SMA
            if lo < 0 or a - DIP_DAYS < 0:
                continue
            win = c[lo:r]
            if not _fin(win):
                continue
            ret = float(c[a] / c[a - DIP_DAYS] - 1)
            if c[a] > float(np.mean(win)) and ret <= DIP_MIN:
                out[s] = -ret
        elif kind == "VOL":
            lo = a - VOL_DAYS
            if lo < 0:
                continue
            q = np.where(panel.complete[lo:a, j], panel.qv[lo:a, j], np.nan)
            if np.sum(np.isfinite(q)) < VOL_DAYS // 2:
                continue
            med = float(np.nanmedian(q))
            qa = float(panel.qv[a, j])
            if med > 0 and math.isfinite(qa) and qa >= VOL_MULT * med and c[a] > c[a - 1]:
                out[s] = qa / med
        else:
            raise ValueError(kind)
    return out


def strength_chooser(ctx_day, ranked, n_free, held, buyable):
    return [s for s, _ in ranked if s not in held and s in buyable][:n_free]


def random_chooser(moves: dict, seed: int):
    rnd = random.Random(seed)

    def chooser(ctx_day, ranked, n_free, held, buyable, universe=None):
        n = min(moves.get(ctx_day, 0), n_free)
        pool = sorted(s for s in (universe or []) if s not in held and s in buyable)
        return rnd.sample(pool, min(n, len(pool)))
    chooser.is_random = True
    return chooser


def simulate(panel, cand: dict, fm, day_from: str, day_to: str, ucfg, start_equity: float = 1_000_000.0,
             chooser=None) -> dict:
    chooser = chooser or strength_chooser
    W = panel.cut(day_to)
    r0, r1 = W.row(day_from), W.row(day_to)
    H, K, kind = int(cand["H"]), int(cand["K"]), cand["kind"]
    cash = float(start_equity)
    pos = {}
    trades, marks, moves, last_px = [], {}, {}, {}
    last_bar = W.last_bar_rows()
    max_open = 0

    def equity():
        return cash + sum(p["qty"] * last_px.get(s, p["entry_raw"]) for s, p in pos.items())

    def close(s, raw, day, reason, fee=True):
        nonlocal cash
        p = pos.pop(s)
        if fee:
            px = fm.sell(raw, p["qty"] * raw, None)
            proceeds = p["qty"] * px * (1 - fm.taker_fee)
        else:
            proceeds = p["qty"] * raw
        cash += proceeds
        pnl = proceeds - p["cost"]
        trades.append({"sym": s, "entry_day": p["entry_day"], "exit_day": day, "cost": p["cost"], "pnl": pnl,
                       "ret_pct": pnl / p["cost"] * 100, "reason": reason})

    for r in range(max(r0, 0), r1):
        day = XS.day_str(W.d0 + r)
        if r - 1 >= 0:
            for s in pos:
                c = W.close[r - 1, W.col[s]]
                if math.isfinite(c):
                    last_px[s] = float(c)
        eq = equity()
        marks[day] = eq
        for s in [s for s in pos if last_bar[W.col[s]] < r]:
            close(s, last_px.get(s, pos[s]["entry_raw"]) * (1 - DELIST_HAIRCUT), day, "delisted", fee=False)
        for s in [s for s, p in pos.items() if r - p["entry_row"] >= H]:
            j = W.col[s]
            if W.has_first[r, j] and math.isfinite(W.first_close[r, j]):
                close(s, float(W.first_close[r, j]), day, "hold")
        uni = XS.universe(W, day, ucfg)
        buyable = frozenset(s for s in uni if W.has_first[r, W.col[s]] and math.isfinite(W.first_close[r, W.col[s]]))
        ev = {} if getattr(chooser, "is_random", False) else events(W, day, kind, uni)
        ranked = sorted(ev.items(), key=lambda kv: (-kv[1], kv[0]))
        n_free = K - len(pos)
        want = []
        if n_free > 0:
            if getattr(chooser, "is_random", False):
                want = chooser(day, ranked, n_free, set(pos), buyable, universe=uni)
            else:
                want = chooser(day, ranked, n_free, set(pos), buyable)
        slot = eq / K
        n_buy = 0
        for s in want:
            notional = min(slot, cash)
            if notional < MIN_TRADE_USDT:
                break
            raw = float(W.first_close[r, W.col[s]])
            fill = fm.buy(raw, notional, None)
            pos[s] = {"qty": notional / fill * (1 - fm.taker_fee), "cost": notional, "entry_day": day, "entry_row": r,
                      "entry_raw": raw}
            last_px[s] = raw
            cash -= notional
            n_buy += 1
        moves[day] = n_buy
        max_open = max(max_open, len(pos))
    if r1 > max(r0, 0):
        for s in pos:
            c = W.close[r1 - 1, W.col[s]]
            if math.isfinite(c):
                last_px[s] = float(c)
        marks[day_to] = equity()
    for s in list(pos):
        close(s, last_px.get(s, pos[s]["entry_raw"]), day_to, "flush")
    days = sorted(marks)
    series = [(a, (marks[b] / marks[a] - 1) * 100) for a, b in zip(days, days[1:]) if marks[a] > 0]
    pnl_sum = sum(t["pnl"] for t in trades)
    return {"series": series, "agg": DR.aggregate(series), "trades": trades, "moves": moves, "max_open": max_open,
            "final_equity": cash,
            "conservation_ok": abs((cash - start_equity) - pnl_sum) < 1e-6 * max(1.0, start_equity) and not pos}
