"""TF-ENS -- Donchian trend ensemble on the 20 most liquid spot pairs (tf-ensemble-tfens-1118-spec.md §3).

Nothing deployed: reads the daily panel built from trader/runtime/history (experiment_xsmom.py panel).

  channel_state  per lookback n: long when the close is above the max of the n prior closes; trailing stop at the channel
                 midpoint, only rising; flat when a close falls below the previous day's stop
  ensemble       share of the 9 lookbacks long
  weights        w = e * min(0.25 / sigma_90d, 2.0) / 20, sum <= 1 (spot, no borrowing); universe re-selected on the
                 first UTC day of each month; signals from closes of days < d only
  simulate       rebalance at the first 15m close of d when |w - w_now| >= band * max(w, w_now) (full entries and exits
                 always); mark at 00:00 by the close of d-1; delisted -> last price * (1 - 0.5), no fee; window edge: sold

Days are 'YYYY-MM-DD' strings in every public function.
"""
from __future__ import annotations

import math
import random
import sys
import warnings
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import xsmom as XS                # noqa: E402

LOOKBACKS = (5, 10, 20, 30, 60, 90, 150, 250, 360)
N_U = 20
VOL_TARGET = 0.25
VOL_DAYS = 90
VOL_MIN = 60
LEV_MAX = 2.0
AGE_DAYS = 365
LIQ_DAYS = 30
LIQ_MIN_DAYS = 20
MIN_TRADE_USDT = 10.0
START_EQUITY = 1_000_000.0
DELIST_HAIRCUT = 0.5
SHIFT_MIN = 30


# ------------------------------------------------------------------ signal
def channel_state(x, n: int, with_stop: bool = False):
    x = np.asarray(x, dtype=float)
    T = len(x)
    st = np.zeros(T)
    stop = np.full(T, np.nan)
    prior_max = np.full(T, np.nan)        # max of x[t-n .. t-1], NaN unless all finite
    mid = np.full(T, np.nan)              # midpoint of x[t-n+1 .. t] (finite values)
    if T > n:
        win = sliding_window_view(x, n)
        fin = np.isfinite(win).all(axis=1)
        mx = np.where(fin, np.max(np.where(np.isfinite(win), win, -np.inf), axis=1), np.nan)
        prior_max[n:] = mx[:T - n]
    if T >= n:
        win = sliding_window_view(x, n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mid[n - 1:] = (np.nanmax(win, axis=1) + np.nanmin(win, axis=1)) / 2
    long, s = False, math.nan
    for t in range(T):
        xt = x[t]
        if math.isfinite(xt):
            if long:
                if xt < s:
                    long, s = False, math.nan
                elif math.isfinite(mid[t]):
                    s = max(s, mid[t])
            elif math.isfinite(prior_max[t]) and xt > prior_max[t]:
                long, s = True, mid[t]
        st[t] = 1.0 if long else 0.0
        stop[t] = s if long else math.nan
    return (st, stop) if with_stop else st


def ensemble(x) -> np.ndarray:
    return np.mean([channel_state(x, n) for n in LOOKBACKS], axis=0)


def vol_mult(x, t: int) -> float:
    x = np.asarray(x, dtype=float)
    seg = x[max(0, t - VOL_DAYS):t + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(np.log(seg))
    r = r[np.isfinite(r)]
    if len(r) < VOL_MIN:
        return 0.0
    sig = float(np.std(r, ddof=1)) * math.sqrt(365)
    return LEV_MAX if sig <= 0 else min(VOL_TARGET / sig, LEV_MAX)


def normalize(w: dict) -> dict:
    s = sum(w.values())
    return {k: v / s for k, v in w.items()} if s > 1 else dict(w)


def need_trade(w_target: float, w_now: float, band: float) -> bool:
    if w_target <= 0 and w_now <= 0:
        return False
    if w_now <= 0 < w_target or w_target <= 0 < w_now:
        return True
    return abs(w_target - w_now) >= band * max(w_target, w_now)


# ------------------------------------------------------------------ universe
_CFG = XS.Config(L=1, H=1, K=N_U, n_liq=N_U, liq_days=LIQ_DAYS, liq_min_days=LIQ_MIN_DAYS, min_age_days=AGE_DAYS)


def month_start(day: str) -> str:
    return day[:8] + "01"


def universe(panel, day: str) -> list:
    return XS.universe(panel, month_start(day), _CFG)


def shifts(syms, n_rows: int, seed: int) -> dict:
    rng = random.Random(seed)
    return {s: rng.randint(SHIFT_MIN, n_rows - SHIFT_MIN) for s in sorted(syms)}


class Signals:
    def __init__(self, panel):
        self.panel = panel
        self._ens, self._vol = {}, {}

    def _e(self, s):
        if s not in self._ens:
            self._ens[s] = ensemble(self.panel.close[:, self.panel.col[s]])
        return self._ens[s]

    def _v(self, s, t):
        key = (s, t)
        if key not in self._vol:
            self._vol[key] = vol_mult(self.panel.close[:, self.panel.col[s]], t)
        return self._vol[key]

    def weights(self, day: str, mode: str = "tf", shift: dict | None = None, window=None) -> dict:
        p = self.panel
        t = p.row(day) - 1                       # the last closed day before d
        if t < 0 or t >= p.n:
            return {}
        w = {}
        for s in universe(p, day):
            if mode == "always":
                e = 1.0
            else:
                idx = t
                if shift is not None and window is not None and s in shift:
                    a, b = p.row(window[0]) - 1, p.row(window[1]) - 1
                    L = b - a
                    if L > 0 and a <= t < b:
                        idx = a + (t - a - shift[s]) % L
                e = float(self._e(s)[idx])
            v = self._v(s, t)
            if e > 0 and v > 0:
                w[s] = e * v / N_U
        return normalize(w)


# ------------------------------------------------------------------ portfolio
def simulate(panel, weight_fn, fm, day_from: str, day_to: str, band: float, start_equity: float = START_EQUITY) -> dict:
    W = panel.cut(day_to)
    r0, r1 = max(W.row(day_from), 0), W.row(day_to)
    last_bar = W.last_bar_rows()
    cash, qty, last_px = start_equity, {}, {}
    trades, marks = [], []
    n_rebal, turnover = 0, 0.0

    def refresh(r):
        if 0 <= r < W.n:
            for s in qty:
                c = W.close[r, W.col[s]]
                if math.isfinite(c):
                    last_px[s] = float(c)

    def sell(s, q, raw, day, reason):
        nonlocal cash, turnover
        fill = fm.sell(raw, q * raw, None)
        cash += q * fill * (1 - fm.taker_fee)
        turnover += q * raw
        qty[s] -= q
        if qty[s] <= 1e-12:
            del qty[s]
        trades.append({"day": day, "sym": s, "side": "SELL", "notional": q * raw, "reason": reason})

    for r in range(r0, r1):
        day = XS.day_str(W.d0 + r)
        refresh(r - 1)
        marks.append((day, cash + sum(q * last_px.get(s, 0.0) for s, q in qty.items())))
        for s in list(qty):                                   # delisted: no bar to the end of the window
            if last_bar[W.col[s]] < r:
                cash += qty[s] * last_px.get(s, 0.0) * (1 - DELIST_HAIRCUT)
                trades.append({"day": day, "sym": s, "side": "SELL", "notional": qty[s] * last_px.get(s, 0.0),
                               "reason": "delisted"})
                del qty[s]
        tg = weight_fn(day) or {}
        px = {}
        for s in set(qty) | set(tg):
            j = W.col.get(s)
            if j is not None and W.has_first[r, j] and math.isfinite(W.first_close[r, j]):
                px[s] = float(W.first_close[r, j])
        eq = cash + sum(q * px.get(s, last_px.get(s, 0.0)) for s, q in qty.items())
        if eq <= 0:
            continue
        traded = False
        for s in sorted(qty):
            if s not in px:
                continue
            wn, wt = qty[s] * px[s] / eq, tg.get(s, 0.0)
            if wt < wn and need_trade(wt, wn, band):
                q = qty[s] if wt <= 0 else (wn - wt) * eq / px[s]
                if wt > 0 and q * px[s] < MIN_TRADE_USDT:
                    continue
                sell(s, q, px[s], day, "rebalance" if wt > 0 else "exit")
                traded = True
        for s in sorted(tg):
            wt = tg[s]
            if wt <= 0 or s not in px:
                continue
            wn = qty.get(s, 0.0) * px[s] / eq
            if wt > wn and need_trade(wt, wn, band):
                notional = min((wt - wn) * eq, cash)
                if notional < MIN_TRADE_USDT:
                    continue
                fill = fm.buy(px[s], notional, None)
                qty[s] = qty.get(s, 0.0) + notional / fill * (1 - fm.taker_fee)
                cash -= notional
                turnover += notional
                last_px.setdefault(s, px[s])
                trades.append({"day": day, "sym": s, "side": "BUY", "notional": notional, "reason": "rebalance"})
                traded = True
        n_rebal += traded
    refresh(r1 - 1)
    end_day = XS.day_str(W.d0 + r1)
    for s in sorted(qty):
        sell(s, qty[s], last_px.get(s, 0.0), end_day, "window_end")
    marks.append((end_day, cash))
    series = [(marks[i][0], (marks[i + 1][1] / marks[i][1] - 1) * 100) for i in range(len(marks) - 1) if marks[i][1] > 0]
    n_days = max(1, len(series))
    avg_eq = float(np.mean([m for _, m in marks])) if marks else start_equity
    return {"series": series, "trades": trades, "n_rebal": n_rebal, "end_equity": cash,
            "turnover_per_day": turnover / n_days / avg_eq if avg_eq else None, "cash_ok": cash >= -1e-6}
