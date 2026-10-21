"""XSMOM -- daily cross-sectional momentum on the whole spot universe, in money.

xsmom-daily-money-1021-spec.md. Nothing deployed: reads the phase-B history (trader/runtime/history).

  daily bars   UTC days from 15m: close = last 15m close, first_close = close of the 00:00 bar (the fill
               reference), qv = sum close*volume, complete = >= COMPLETE_MIN_BARS bars        (spec §3)
  universe     day d sees full days < d only: no stablecoins (not in the store), no leveraged tokens,
               complete days d-1 and d-1-L, >= LIQ_MIN_DAYS complete days of the last LIQ_DAYS,
               top N_LIQ by median quote volume, ties by name                                (spec §4.1)
  rotation     H = 1: one tranche, daily; H = 7: seven tranches of 1/7, tranche i on (d - start) % 7 == i,
               all tranches on the first day; minimal turnover: only dropped names are sold, freed cash is
               split equally among new names                                                 (spec §4.3)
  money        fill_model at first_close with the fee convention of execution_replay.run; equity marked at
               00:00 UTC by the close of d-1; R_d by daily_revenue.aggregate                  (spec §4.4)
  controls     matched_random (same sells/buys per step, random names), ewu_chooser (the whole universe)

Days are 'YYYY-MM-DD' strings in every public function.
"""
from __future__ import annotations

import math
import random
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import daily_revenue as DR        # noqa: E402

DAY_MS = 86_400_000
BAR_MS = 15 * 60 * 1000
COMPLETE_MIN_BARS = 90
LIQ_DAYS = 30
LIQ_MIN_DAYS = 20
N_LIQ = 100
FILTER_DAYS = 28
FILTER_MIN_PAIRS = 20
DELIST_HAIRCUT = 0.5
MIN_TRADE_USDT = 10.0
START_EQUITY = 10_000.0
_EPOCH = datetime(1970, 1, 1)


def day_num(day: str) -> int:
    return (datetime.strptime(day, "%Y-%m-%d") - _EPOCH).days


def day_str(n) -> str:
    return (_EPOCH + timedelta(days=int(n))).strftime("%Y-%m-%d")


# ------------------------------------------------------------------ daily bars
def daily_from_15m(arr) -> dict:
    t = np.asarray(arr["t"], dtype=np.int64)
    keys = ("day", "close", "first_close", "qv", "n_bars", "complete", "has_first")
    if len(t) == 0:
        return {k: np.zeros(0) for k in keys}
    order = np.argsort(t, kind="stable")
    t = t[order]
    c = np.asarray(arr["c"], dtype=float)[order]
    v = np.asarray(arr["v"], dtype=float)[order]
    day = t // DAY_MS
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    ends = np.r_[starts[1:], len(t)]
    has_first = (t[starts] % DAY_MS) // BAR_MS == 0
    n_bars = ends - starts
    return {"day": day[starts], "close": c[ends - 1], "first_close": np.where(has_first, c[starts], np.nan),
            "qv": np.add.reduceat(c * v, starts), "n_bars": n_bars, "complete": n_bars >= COMPLETE_MIN_BARS,
            "has_first": has_first}


# ------------------------------------------------------------------ panel
class Panel:
    """Daily arrays [days x syms] on a contiguous UTC-day axis; NaN / False where a pair has no bars."""

    def __init__(self, days, syms, close, first_close, qv, complete, has_first, has_bar, _cache=None):
        self.days = np.asarray(days, dtype=np.int64)
        self.syms = list(syms)
        self.close, self.first_close, self.qv = close, first_close, qv
        self.complete, self.has_first, self.has_bar = complete, has_first, has_bar
        self.col = {s: j for j, s in enumerate(self.syms)}
        self.d0 = int(self.days[0]) if len(self.days) else 0
        self.n = len(self.days)
        self._cache = {} if _cache is None else _cache
        self._lev = None

    @classmethod
    def from_daily(cls, daily: dict) -> "Panel":
        syms = sorted(s for s, d in daily.items() if len(d["day"]))
        if not syms:
            z = np.zeros((0, 0))
            return cls([], [], z, z, z, z.astype(bool), z.astype(bool), z.astype(bool))
        lo = min(int(daily[s]["day"][0]) for s in syms)
        hi = max(int(daily[s]["day"][-1]) for s in syms)
        days = np.arange(lo, hi + 1, dtype=np.int64)
        D, S = len(days), len(syms)
        close, first, qv = (np.full((D, S), np.nan) for _ in range(3))
        complete, has_first, has_bar = (np.zeros((D, S), dtype=bool) for _ in range(3))
        for j, s in enumerate(syms):
            d = daily[s]
            r = np.asarray(d["day"], dtype=np.int64) - lo
            close[r, j] = d["close"]
            first[r, j] = d["first_close"]
            qv[r, j] = d["qv"]
            complete[r, j] = d["complete"]
            has_first[r, j] = d["has_first"]
            has_bar[r, j] = True
        return cls(days, syms, close, first, qv, complete, has_first, has_bar)

    def row(self, day: str) -> int:
        return day_num(day) - self.d0

    def cut(self, day_to: str) -> "Panel":
        """Rows of days < day_to only (views; the cache is shared because a row's universe reads only rows before it)."""
        k = max(0, min(self.n, self.row(day_to)))
        return Panel(self.days[:k], self.syms, self.close[:k], self.first_close[:k], self.qv[:k], self.complete[:k],
                     self.has_first[:k], self.has_bar[:k], _cache=self._cache)

    def leveraged(self) -> np.ndarray:
        if self._lev is None:
            import trading_universe as TU
            bases = {s[:-4] for s in self.syms if s.endswith("USDT")}
            self._lev = np.array([s.endswith("USDT") and TU.is_leveraged(s[:-4], bases) for s in self.syms], dtype=bool)
        return self._lev

    def last_bar_rows(self) -> np.ndarray:
        """Per pair, the last row with any bar (-1 if none)."""
        if not self.n:
            return np.full(len(self.syms), -1)
        rev = self.has_bar[::-1]
        any_ = rev.any(axis=0)
        return np.where(any_, self.n - 1 - rev.argmax(axis=0), -1)


def _liq(p: Panel, r: int, cfg) -> np.ndarray:
    """Median quote volume over complete days of rows [r - liq_days, r); NaN when fewer than liq_min_days."""
    key = ("liq", r if r <= p.n else ("beyond", r, p.n), cfg.liq_days, cfg.liq_min_days)
    if key in p._cache:
        return p._cache[key]
    lo, hi = max(0, r - cfg.liq_days), min(r, p.n)
    S = len(p.syms)
    if hi <= lo:
        out = np.full(S, np.nan)
    else:
        q = np.where(p.complete[lo:hi], p.qv[lo:hi], np.nan)
        cnt = p.complete[lo:hi].sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(q, axis=0)
        out = np.where(cnt >= cfg.liq_min_days, med, np.nan)
    p._cache[key] = out
    return out


# ------------------------------------------------------------------ config, universe, signal
@dataclass(frozen=True)
class Config:
    L: int
    H: int
    K: int
    market_filter: bool = False
    n_liq: int = N_LIQ
    liq_days: int = LIQ_DAYS
    liq_min_days: int = LIQ_MIN_DAYS
    filter_days: int = FILTER_DAYS
    filter_min_pairs: int = FILTER_MIN_PAIRS
    delist_haircut: float = DELIST_HAIRCUT
    min_trade_usdt: float = MIN_TRADE_USDT

    @property
    def name(self) -> str:
        return f"{'MF' if self.market_filter else 'M'}_L{self.L}_H{self.H}_K{self.K}"


def universe(p: Panel, day: str, cfg) -> list:
    r = p.row(day)
    key = ("uni", r if r <= p.n else ("beyond", r, p.n), cfg.L, cfg.n_liq, cfg.liq_days, cfg.liq_min_days)
    if key in p._cache:
        return list(p._cache[key])
    out = []
    a, b = r - 1, r - 1 - cfg.L
    if b >= 0 and a < p.n:
        ok = p.complete[a] & p.complete[b] & ~p.leveraged()
        liq = _liq(p, r, cfg)
        ok &= np.isfinite(liq)
        idx = np.flatnonzero(ok)
        if len(idx):
            order = idx[np.lexsort((idx, -liq[idx]))]     # liquidity desc, ties by name (syms are sorted)
            out = [p.syms[j] for j in order[:cfg.n_liq]]
    p._cache[key] = tuple(out)
    return out


def scores(p: Panel, day: str, L: int, syms) -> dict:
    r = p.row(day)
    a, b = r - 1, r - 1 - L
    out = {}
    if b < 0 or a >= p.n:
        return out
    for s in syms:
        j = p.col.get(s)
        if j is None:
            continue
        c1, c0 = p.close[a, j], p.close[b, j]
        if math.isfinite(c1) and math.isfinite(c0) and c0 > 0:
            out[s] = float(c1 / c0 - 1)
    return out


def top_k(sc: dict, K: int) -> list:
    return [s for s, _ in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))[:K]]


def market_on(p: Panel, day: str, syms, cfg) -> bool:
    """Mean filter_days return of the universe > 0; fewer than filter_min_pairs pairs with data -> True (no data, no block)."""
    r = p.row(day)
    a, b = r - 1, r - 1 - cfg.filter_days
    rets = []
    if b >= 0 and a < p.n:
        for s in syms:
            j = p.col.get(s)
            if j is None:
                continue
            c1, c0 = p.close[a, j], p.close[b, j]
            if math.isfinite(c1) and math.isfinite(c0) and c0 > 0:
                rets.append(c1 / c0 - 1)
    if len(rets) < cfg.filter_min_pairs:
        return True
    return sum(rets) / len(rets) > 0


# ------------------------------------------------------------------ choosers
@dataclass(frozen=True)
class Ctx:
    day: str
    tranche: int
    held: frozenset
    universe: list
    scores: dict
    cfg: Config
    buyable: frozenset | None = None      # universe names with a 00:00 bar today (None = not known)


def momentum_chooser(ctx: Ctx) -> list:
    return top_k(ctx.scores, ctx.cfg.K)


def ewu_chooser(ctx: Ctx) -> list:
    return list(ctx.universe)


def matched_random(moves: dict, seed: int):
    """Random names with the candidate's turnover: at (tranche, day) sell n_sell random held names and buy n_buy
    random universe names not held (spec §6). Deterministic by seed."""
    rnd = random.Random(seed)

    def chooser(ctx: Ctx) -> list:
        n_sell, n_buy = moves.get((ctx.tranche, ctx.day), (0, 0))
        held = sorted(ctx.held)
        sell = set(rnd.sample(held, min(n_sell, len(held))))
        pool = [s for s in ctx.universe if s not in ctx.held and (ctx.buyable is None or s in ctx.buyable)]
        pool.sort()
        buy = rnd.sample(pool, min(n_buy, len(pool)))
        return [s for s in held if s not in sell] + buy
    return chooser


# ------------------------------------------------------------------ rotation
def rotate(panel: Panel, cfg: Config, fm, day_from: str, day_to: str, chooser=None,
           start_equity: float = START_EQUITY) -> dict:
    chooser = chooser or momentum_chooser
    W = panel.cut(day_to)                       # nothing at or after day_to is visible
    r0, r1 = W.row(day_from), W.row(day_to)
    n_tr = 1 if cfg.H <= 1 else int(cfg.H)
    cash = [start_equity / n_tr] * n_tr
    hold = [dict() for _ in range(n_tr)]        # sym -> position
    last_px, trades, moves, marks = {}, [], {}, {}
    last_bar = W.last_bar_rows()
    turnover = 0.0
    on_days = n_days = 0
    was_on = True

    def held_syms():
        return {s for h in hold for s in h}

    def refresh_px(r):
        if 0 <= r < W.n:
            for s in held_syms():
                c = W.close[r, W.col[s]]
                if math.isfinite(c):
                    last_px[s] = float(c)

    def equity():
        return sum(cash) + sum(p["qty"] * last_px.get(s, p["entry_raw"]) for h in hold for s, p in h.items())

    def close_pos(tr, s, raw, day, reason, fee=True):
        nonlocal turnover
        p = hold[tr].pop(s)
        if fee:
            notional = p["qty"] * raw
            px = fm.sell(raw, notional, None)
            proceeds = p["qty"] * px * (1 - fm.taker_fee)
            turnover += notional
        else:
            proceeds = p["qty"] * raw
        cash[tr] += proceeds
        pnl = proceeds - p["cost"]
        trades.append({"sym": s, "tranche": tr, "entry_day": p["entry_day"], "exit_day": day, "cost": p["cost"],
                       "pnl": pnl, "ret_pct": pnl / p["cost"] * 100, "reason": reason,
                       "entry_px_raw": p["entry_raw"], "exit_px_raw": raw})

    for r in range(max(r0, 0), r1):
        day = day_str(W.d0 + r)
        refresh_px(r - 1)
        marks[day] = equity()
        # delisting: no bar from row r to the end of the window (spec §4.6)
        for tr in range(n_tr):
            for s in [s for s in hold[tr] if last_bar[W.col[s]] < r]:
                close_pos(tr, s, last_px.get(s, hold[tr][s]["entry_raw"]) * (1 - cfg.delist_haircut), day,
                          "delisted", fee=False)
        uni = universe(W, day, cfg)
        on = market_on(W, day, uni, cfg) if cfg.market_filter else True
        n_days += 1
        on_days += bool(on)
        due = []
        if not on:
            for tr in range(n_tr):
                for p in hold[tr].values():
                    p["pending"] = p["pending"] or "filter"
        elif r == r0 or not was_on:
            due = list(range(n_tr))
        else:
            due = [(r - r0) % n_tr]
        was_on = on
        desired = {}
        if due:
            sc = scores(W, day, cfg.L, uni)
            buyable = frozenset(s for s in uni if W.has_first[r, W.col[s]])
            for tr in due:
                want = list(chooser(Ctx(day, tr, frozenset(hold[tr]), uni, sc, cfg, buyable)))
                desired[tr] = want
                n_sell = 0
                for s, p in hold[tr].items():
                    if s in want:
                        p["pending"] = None
                    else:
                        p["pending"] = p["pending"] or "rotate"
                        n_sell += 1
                moves[(tr, day)] = [n_sell, 0]
        # sells that can execute today (a pair without its 00:00 bar waits for the next one)
        for tr in range(n_tr):
            for s in [s for s, p in hold[tr].items() if p["pending"]]:
                j = W.col[s]
                if W.has_first[r, j]:
                    close_pos(tr, s, float(W.first_close[r, j]), day, hold[tr][s]["pending"])
        # buys of the due tranches: the tranche's cash split equally among the new names
        for tr in due:
            new = [s for s in desired[tr] if s not in hold[tr] and s in W.col and W.has_first[r, W.col[s]]]
            if new:
                notional = cash[tr] / len(new)
                if notional >= cfg.min_trade_usdt:
                    for s in new:
                        raw = float(W.first_close[r, W.col[s]])
                        fill = fm.buy(raw, notional, None)
                        qty = notional / fill * (1 - fm.taker_fee)
                        cash[tr] -= notional
                        turnover += notional
                        hold[tr][s] = {"qty": qty, "cost": notional, "entry_day": day, "entry_raw": raw, "pending": None}
                        last_px[s] = raw
                        moves[(tr, day)][1] += 1
    # window edge: last mark at 00:00 of day_to, then sell everything at that close (spec §4.7)
    end_day = day_to
    refresh_px(r1 - 1)
    if r1 > max(r0, 0):
        marks[end_day] = equity()
    for tr in range(n_tr):
        for s in list(hold[tr]):
            close_pos(tr, s, last_px.get(s, hold[tr][s]["entry_raw"]), end_day, "flush")
    days = sorted(marks)
    series = [(a, (marks[b] / marks[a] - 1) * 100) for a, b in zip(days, days[1:]) if marks[a] > 0]
    final = sum(cash)
    pnl_sum = sum(t["pnl"] for t in trades)
    avg_eq = (sum(marks.values()) / len(marks)) if marks else start_equity
    return {"series": series, "agg": DR.aggregate(series), "trades": trades, "final_equity": final,
            "conservation_ok": abs((final - start_equity) - pnl_sum) < 1e-6 * max(1.0, start_equity) and not any(hold),
            "moves": {k: tuple(v) for k, v in moves.items()},
            "turnover_per_day": (turnover / avg_eq / n_days) if (n_days and avg_eq > 0) else 0.0,
            "exposure_share": (on_days / n_days) if n_days else 0.0}


def random_control(panel: Panel, cfg: Config, fm, day_from: str, day_to: str, moves: dict, n: int = 50,
                   start_equity: float = START_EQUITY) -> list:
    """Mean daily R of n matched-turnover random rotations (seeds 0..n-1)."""
    return [rotate(panel, cfg, fm, day_from, day_to, chooser=matched_random(moves, seed),
                   start_equity=start_equity)["agg"].get("mean_pct") for seed in range(n)]


def to_e1_candidates(trades) -> list:
    """Rotation trades -> execution_replay candidates (exits.kind = "real"): entry at the close of the 00:00 bar
    (ts 00:16), exit likewise, a window-edge sale at the close of the previous day (ts 00:01). Delisted
    positions have no exchange exit and are not converted."""
    out = []
    for t in trades:
        if t["reason"] == "delisted":
            continue
        e0 = datetime.strptime(t["entry_day"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        x0 = datetime.strptime(t["exit_day"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        out.append({"kind": "entry", "sym": t["sym"], "tf": "15m", "ts": e0 + timedelta(minutes=16),
                    "price": t["entry_px_raw"],
                    "exit_ts": x0 + timedelta(minutes=1 if t["reason"] == "flush" else 16),
                    "exit_price": t["exit_px_raw"], "real_pnl": t["ret_pct"], "mode": "xsmom", "trail_k": None})
    out.sort(key=lambda c: (c["ts"], c["sym"]))
    return out
