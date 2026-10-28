"""TM -- BTC/ETH spot timing, "in the market or in USDT" (btc-eth-timing-tm-1028-spec.md §3-§5).

Nothing deployed: reads the daily panel built from trader/runtime/history and the perp funding store.

  target       the share w in [0, 1] of a sleeve for decision row r, from rows < r and funding paid before 00:00 of r
               T_ret / T_sma (trend), V (volatility-managed), F (funding overheat); no data -> w = 1   (spec §4)
  simulate     two sleeves, 50% each; mark at 00:00 by the close of d-1; rebalance at first_close of d when
               |w - w_now| > band; window edge: last mark, then everything is sold                      (spec §3)
  controls     ce_targets (constant mean share), shift_targets (circular shift per sleeve)             (spec §5)

Days are 'YYYY-MM-DD' strings in every public function.
"""
from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import daily_revenue as DR        # noqa: E402
import xsmom as XS                # noqa: E402

DAY_MS = XS.DAY_MS
SYMS = ("BTCUSDT", "ETHUSDT")
BAND = 0.10
START_EQUITY = 1_000_000.0
V_MIN_HISTORY = 60
F_MIN_HISTORY = 90
SHIFT_MIN = 30


@dataclass(frozen=True)
class Cand:
    name: str
    kind: str                 # "T_ret" | "T_sma" | "V" | "F"
    N: int = 0
    k: float = 1.0
    W: int = 0
    q: float = 0.0


def candidates() -> dict:
    out = {}
    for N in (7, 14, 28, 56, 112):
        out[f"T_ret_{N}"] = Cand(f"T_ret_{N}", "T_ret", N=N)
    for N in (7, 14, 28, 56, 112):
        out[f"T_sma_{N}"] = Cand(f"T_sma_{N}", "T_sma", N=N)
    for N in (7, 14, 30):
        for k in (0.75, 1.0):
            out[f"V_{N}_{k:g}"] = Cand(f"V_{N}_{k:g}", "V", N=N, k=k)
    for W in (3, 7):
        for q in (0.80, 0.90):
            out[f"F_{W}_{q:g}"] = Cand(f"F_{W}_{q:g}", "F", W=W, q=q)
    return out


# ------------------------------------------------------------------ signals
def funding_signal(funding: dict, sym: str, day_n: int, W: int):
    """Mean funding rate of payments in [00:00 of day_n - W days, 00:00 of day_n); None if none."""
    f = (funding or {}).get(sym)
    if f is None or len(f["t"]) == 0:
        return None
    t = np.asarray(f["t"], dtype=np.int64)
    hi = day_n * DAY_MS
    a, b = np.searchsorted(t, hi - W * DAY_MS, "left"), np.searchsorted(t, hi, "left")
    if b <= a:
        return None
    return float(np.mean(np.asarray(f["rate"], dtype=float)[a:b]))


def _series(panel, funding, cand: Cand, sym: str) -> np.ndarray:
    """Target share for every decision row 0..n (row n = the day after the panel), from rows strictly before."""
    n = panel.n
    out = np.ones(n + 1)
    j = panel.col.get(sym)
    if j is None:
        return out
    c = panel.close[:, j]
    if cand.kind == "T_ret":
        for r in range(n + 1):
            a, b = r - 1, r - 1 - cand.N
            if b >= 0 and math.isfinite(c[a]) and math.isfinite(c[b]) and c[b] > 0:
                out[r] = 1.0 if c[a] / c[b] - 1 > 0 else 0.0
    elif cand.kind == "T_sma":
        for r in range(n + 1):
            lo = r - cand.N
            if lo >= 0:
                win = c[lo:r]
                if np.all(np.isfinite(win)):
                    out[r] = 1.0 if c[r - 1] > win.mean() else 0.0
    elif cand.kind == "V":
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.r_[np.nan, np.log(c[1:] / c[:-1])]
        sig = np.full(n + 1, np.nan)                       # sig[r] from lr[r - N .. r - 1]
        for r in range(cand.N + 1, n + 1):
            w = lr[r - cand.N:r]
            if np.all(np.isfinite(w)):
                sig[r] = float(np.std(w, ddof=1))
        hist = []
        for r in range(n + 1):
            if len(hist) >= V_MIN_HISTORY and math.isfinite(sig[r]) and sig[r] > 0:
                out[r] = min(1.0, cand.k * float(np.median(hist)) / sig[r])
            if math.isfinite(sig[r]):
                hist.append(sig[r])
    elif cand.kind == "F":
        hist = []
        for r in range(n + 1):
            f = funding_signal(funding, sym, panel.d0 + r, cand.W)
            if f is not None and len(hist) >= F_MIN_HISTORY:
                out[r] = 0.0 if f > float(np.quantile(hist, cand.q)) else 1.0
            if f is not None:
                hist.append(f)
    else:
        raise ValueError(cand.kind)
    return out


def target(panel, funding, cand: Cand, sym: str, r: int) -> float:
    return float(_series(panel, funding, cand, sym)[r])


def target_paths(panel, funding, cand: Cand, day_from: str, day_to: str, syms=SYMS) -> dict:
    W = panel.cut(day_to)
    r0, r1 = W.row(day_from), W.row(day_to)
    return {s: _series(W, funding, cand, s)[r0:r1].copy() for s in syms}


# ------------------------------------------------------------------ controls
def ce_targets(targets: dict) -> dict:
    return {s: np.full(len(w), float(np.mean(w))) for s, w in targets.items()}


def shift_targets(targets: dict, seed: int, min_shift: int = SHIFT_MIN) -> dict:
    rnd = random.Random(seed)
    out = {}
    for s in sorted(targets):
        w = np.asarray(targets[s], dtype=float)
        n = len(w)
        lo = min(min_shift, max(1, n // 4))
        out[s] = np.roll(w, rnd.randint(lo, max(lo, n - lo)))
    return out


# ------------------------------------------------------------------ money
def simulate(panel, targets: dict, fm, day_from: str, day_to: str, band: float = BAND,
             start_equity: float = START_EQUITY, shares: dict | None = None) -> dict:
    W = panel.cut(day_to)
    r0, r1 = W.row(day_from), W.row(day_to)
    syms = sorted(targets)
    shares = shares or {s: 1.0 / len(syms) for s in syms}
    st = {s: {"cash": start_equity * shares[s], "qty": 0.0, "basis": 0.0} for s in syms}
    last_px, marks, trades = {}, {}, []
    turnover = 0.0

    def refresh(r):
        if 0 <= r < W.n:
            for s in syms:
                c = W.close[r, W.col[s]]
                if math.isfinite(c):
                    last_px[s] = float(c)

    def equity():
        return sum(v["cash"] + v["qty"] * last_px.get(s, 0.0) for s, v in st.items())

    def sell(s, qty, raw, day, reason):
        nonlocal turnover
        v = st[s]
        qty = min(qty, v["qty"])
        if qty <= 0:
            return
        px = fm.sell(raw, qty * raw, None)
        proceeds = qty * px * (1 - fm.taker_fee)
        avg = v["basis"] / v["qty"]
        pnl = proceeds - qty * avg
        v["basis"] -= qty * avg
        v["qty"] -= qty
        if v["qty"] <= 1e-15:
            v["qty"], v["basis"] = 0.0, 0.0
        v["cash"] += proceeds
        turnover += qty * raw
        trades.append({"day": day, "sym": s, "side": "SELL", "notional": qty * raw, "pnl": pnl, "reason": reason})

    def buy(s, notional, raw, day):
        nonlocal turnover
        v = st[s]
        notional = min(notional, v["cash"])
        if notional <= 0:
            return
        fill = fm.buy(raw, notional, None)
        v["qty"] += notional / fill * (1 - fm.taker_fee)
        v["basis"] += notional
        v["cash"] -= notional
        turnover += notional
        trades.append({"day": day, "sym": s, "side": "BUY", "notional": notional, "pnl": 0.0, "reason": "rebalance"})

    for r in range(max(r0, 0), r1):
        day = XS.day_str(W.d0 + r)
        refresh(r - 1)
        marks[day] = equity()
        for s in syms:
            j = W.col[s]
            if not W.has_first[r, j] or not math.isfinite(W.first_close[r, j]):
                continue
            raw = float(W.first_close[r, j])
            v = st[s]
            E = v["cash"] + v["qty"] * raw
            if E <= 0:
                continue
            w_now = v["qty"] * raw / E
            w_t = float(targets[s][r - r0])
            if abs(w_t - w_now) <= band + 1e-12:
                continue
            delta = w_t * E - v["qty"] * raw
            if delta > 0:
                buy(s, delta, raw, day)
            elif w_t <= 0:
                sell(s, v["qty"], raw, day, "rebalance")
            else:
                sell(s, -delta / raw, raw, day, "rebalance")
            last_px.setdefault(s, raw)
    refresh(r1 - 1)
    if r1 > max(r0, 0):
        marks[day_to] = equity()
    for s in syms:
        if st[s]["qty"] > 0:
            sell(s, st[s]["qty"], last_px[s], day_to, "flush")
    days = sorted(marks)
    series = [(a, (marks[b] / marks[a] - 1) * 100) for a, b in zip(days, days[1:]) if marks[a] > 0]
    final = sum(v["cash"] for v in st.values())
    pnl_sum = sum(t["pnl"] for t in trades if t["side"] == "SELL")
    n_days = max(1, r1 - max(r0, 0))
    avg_eq = (sum(marks.values()) / len(marks)) if marks else start_equity
    return {"series": series, "agg": DR.aggregate(series), "trades": trades, "final_equity": final,
            "conservation_ok": abs((final - start_equity) - pnl_sum) < 1e-6 * max(1.0, start_equity)
            and all(v["qty"] == 0 for v in st.values()),
            "n_rebal": sum(1 for t in trades if t["reason"] == "rebalance"),
            "exposure_share": float(np.mean([np.mean(w) for w in targets.values()])) if targets else 0.0,
            "turnover_per_day": turnover / avg_eq / n_days if avg_eq > 0 else 0.0}


def run(panel, funding, cand: Cand, fm, day_from: str, day_to: str, start_equity: float = START_EQUITY,
        band: float = BAND) -> dict:
    tg = target_paths(panel, funding, cand, day_from, day_to)
    out = simulate(panel, tg, fm, day_from, day_to, band=band, start_equity=start_equity)
    out["targets"] = tg
    return out
