"""FC -- funding carry: spot long + USD-M perpetual short of the same coin, in money (carry-funding-money-1021-spec.md).

  position   slot capital C, leverage L: notional N = C / (1 + 1/L); spot leg N, isolated perp margin N / L
  entry/exit close of the first 15m bar of the day on both legs, fees and slippage on each leg (fee convention of
             execution_replay.run on the spot leg; perp fee on the perp fill)
  funding    the short receives qty x price x rate for every event t with entry < t <= exit; price = perp close of the
             day of (t - 1 ms); credited to the wallet, not to the position margin
  00:00 UTC  liquidations of day d-1 (perp high >= entry x (1 + 1/L - MMR): margin lost, spot sold at the close of d-1,
             no funding of that day) -> funding -> equity mark -> margin stop (short equity < 50% of its margin) ->
             delisting -> signal exits and entries at the first 15m close of day d
  choosers   funding_chooser (trailing annualised funding, hysteresis theta_in / theta_out = theta_in / 2, at most K),
             passive_chooser (BTCUSDT + ETHUSDT); random control = xsmom.matched_random
Days are 'YYYY-MM-DD' strings in every public function.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import daily_revenue as DR        # noqa: E402
import xsmom as XS                # noqa: E402

DAY_MS = 86_400_000
FILL_OFFSET_MS = 15 * 60 * 1000          # fills at the close of the first 15m bar
MMR = 0.01
MARGIN_STOP = 0.5
SPOT_TAKER = 0.001
PERP_TAKER = 0.0005                      # Binance VIP0 USD-M taker; to be confirmed on the demo (F0, spec §10)
SLIP = {0: 0.0, 1: 0.0008, 2: 0.0020}
START_EQUITY = 1_000_000.0
MIN_TRADE_USDT = 10.0
PASSIVE_PAIRS = ("BTCUSDT", "ETHUSDT")


@dataclass(frozen=True)
class Fees:
    spot_taker: float
    perp_taker: float
    slip: float


def fees(level: int = 1, spot_taker: float = SPOT_TAKER, perp_taker: float = PERP_TAKER) -> Fees:
    return Fees(spot_taker, perp_taker, SLIP[int(level)])


def day_ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def annualized_funding(t, rate, decision_ms: int, days: int):
    """Sum of funding events in [decision - days, decision) x 365 / days; None without events."""
    t = np.asarray(t, dtype=np.int64)
    lo = int(np.searchsorted(t, decision_ms - days * DAY_MS, side="left"))
    hi = int(np.searchsorted(t, decision_ms, side="left"))
    if hi <= lo:
        return None
    return float(np.asarray(rate, dtype=float)[lo:hi].sum() * 365.0 / days)


def liq_price(entry: float, lev: float, mmr: float = MMR) -> float:
    return entry * (1 + 1.0 / lev - mmr)


# ------------------------------------------------------------------ perp panel
def daily_perp(arr) -> dict:
    d = XS.daily_from_15m(arr)
    t = np.asarray(arr["t"], dtype=np.int64)
    if not len(t):
        return {**d, "high": np.zeros(0)}
    order = np.argsort(t, kind="stable")
    day = t[order] // DAY_MS
    h = np.asarray(arr["h"], dtype=float)[order]
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    return {**d, "high": np.maximum.reduceat(h, starts)}


class PerpPanel(XS.Panel):
    """xsmom.Panel plus the daily high; prices per spot base unit (divided by the contract multiplier)."""

    def __init__(self, days, syms, close, first_close, qv, complete, has_first, has_bar, high, _cache=None):
        super().__init__(days, syms, close, first_close, qv, complete, has_first, has_bar, _cache=_cache)
        self.high = high

    @classmethod
    def from_daily(cls, daily: dict, mult: dict | None = None) -> "PerpPanel":
        mult = mult or {}
        base = XS.Panel.from_daily(daily)
        high = np.full(base.close.shape, np.nan)
        for s, d in daily.items():
            if s in base.col and len(d["day"]):
                high[np.asarray(d["day"], dtype=np.int64) - base.d0, base.col[s]] = d["high"]
        m = np.array([float(mult.get(s, 1)) for s in base.syms]) if base.syms else np.zeros(0)
        div = (lambda a: a / m) if len(m) else (lambda a: a)
        return cls(base.days, base.syms, div(base.close), div(base.first_close), base.qv, base.complete, base.has_first,
                   base.has_bar, div(high))

    def cut(self, day_to: str) -> "PerpPanel":
        k = max(0, min(self.n, self.row(day_to)))
        return PerpPanel(self.days[:k], self.syms, self.close[:k], self.first_close[:k], self.qv[:k], self.complete[:k],
                         self.has_first[:k], self.has_bar[:k], self.high[:k], _cache=self._cache)


@dataclass
class Market:
    spot: XS.Panel
    perp: PerpPanel
    funding: dict = field(default_factory=dict)      # spot symbol -> {"t": int64 ms (sorted), "rate": float}

    def cut(self, day_to: str) -> "Market":
        ms = day_ms(day_to)
        f = {}
        for s, v in self.funding.items():
            # an event at exactly 00:00 of day_to closes a period of the previous day: it is known at the edge
            k = int(np.searchsorted(np.asarray(v["t"], dtype=np.int64), ms, side="right"))
            f[s] = {"t": np.asarray(v["t"])[:k], "rate": np.asarray(v["rate"])[:k]}
        return Market(self.spot.cut(day_to), self.perp.cut(day_to), f)


# ------------------------------------------------------------------ config, eligibility, signal, choosers
@dataclass(frozen=True)
class CarryConfig:
    W: int
    K: int
    theta_in: float
    lev: float = 1.0
    passive: bool = False
    n_liq: int = XS.N_LIQ
    liq_days: int = XS.LIQ_DAYS
    liq_min_days: int = XS.LIQ_MIN_DAYS
    min_age_days: int = 0
    margin_stop: float = MARGIN_STOP
    mmr: float = MMR
    min_trade_usdt: float = MIN_TRADE_USDT

    @property
    def theta_out(self) -> float:
        return self.theta_in / 2

    @property
    def name(self) -> str:
        return "FC_PASSIVE" if self.passive else f"FC_W{self.W}_K{self.K}_T{round(self.theta_in * 100)}"

    def spot_cfg(self) -> XS.Config:
        return XS.Config(L=1, H=1, K=self.K, n_liq=self.n_liq, liq_days=self.liq_days, liq_min_days=self.liq_min_days,
                         min_age_days=self.min_age_days)


def _perp_ok(m: Market, sym: str, day: str) -> bool:
    j = m.perp.col.get(sym)
    if j is None:
        return False
    a = m.perp.row(day) - 1
    return 0 <= a < m.perp.n and bool(m.perp.complete[a, j])


def eligible(m: Market, day: str, cfg: CarryConfig) -> list:
    dm = day_ms(day)
    out = []
    for s in XS.universe(m.spot, day, cfg.spot_cfg()):
        f = m.funding.get(s)
        if f is None or not _perp_ok(m, s, day):
            continue
        if annualized_funding(f["t"], f["rate"], dm, cfg.W) is None:
            continue
        out.append(s)
    return out


def signal(m: Market, day: str, cfg: CarryConfig, syms) -> dict:
    dm = day_ms(day)
    out = {}
    for s in syms:
        f = m.funding.get(s)
        if f is None:
            continue
        v = annualized_funding(f["t"], f["rate"], dm, cfg.W)
        if v is not None:
            out[s] = v
    return out


def funding_chooser(ctx) -> list:
    cfg, sc = ctx.cfg, ctx.scores
    keep = [s for s in sorted(ctx.held) if sc.get(s) is not None and sc[s] >= cfg.theta_out]
    free = max(0, cfg.K - len(keep))
    cand = sorted((-v, s) for s, v in sc.items()
                  if s not in ctx.held and v >= cfg.theta_in and (ctx.buyable is None or s in ctx.buyable))
    return keep + [s for _, s in cand[:free]]


def passive_chooser(ctx) -> list:
    return list(PASSIVE_PAIRS)


def default_chooser(cfg: CarryConfig):
    return passive_chooser if cfg.passive else funding_chooser


# ------------------------------------------------------------------ simulation
def simulate(market: Market, cfg: CarryConfig, fees: Fees, day_from: str, day_to: str, chooser=None,
             start_equity: float = START_EQUITY) -> dict:
    chooser = chooser or default_chooser(cfg)
    m = market.cut(day_to)                     # nothing at or after day_to is visible
    S, P = m.spot, m.perp
    r0, r1 = S.row(day_from), S.row(day_to)
    s_last, p_last = S.last_bar_rows(), P.last_bar_rows()
    cash = float(start_equity)
    pos = {}                                   # sym -> position
    trades, moves, marks = [], {}, {}
    funding_income, liquidations, margin_stops, turnover = 0.0, 0, 0, 0.0
    held_days = n_days = 0

    def px(panel, arr, r, sym):
        j = panel.col.get(sym)
        if j is None or not (0 <= r < panel.n):
            return None
        v = arr[r, j]
        return float(v) if math.isfinite(v) else None

    def spot_close(sym, r):
        return px(S, S.close, r, sym)

    def perp_close_day(sym, d):                # d = day number
        return px(P, P.close, d - P.d0, sym)

    def close(sym, s_raw, p_raw, day, reason, lost_margin=False):
        nonlocal cash, turnover
        p = pos.pop(sym)
        q = p["q"]
        s_px = s_raw * (1 - fees.slip)
        proceeds = q * s_px * (1 - fees.spot_taker)
        turnover += q * s_raw
        if lost_margin:
            perp_back, fee = 0.0, 0.0
        else:
            fill = p_raw * (1 + fees.slip)
            fee = q * fill * fees.perp_taker
            perp_back = p["margin"] + q * (p["perp_entry"] - fill) - fee
            turnover += q * p_raw
        cash += proceeds + perp_back
        pnl = proceeds - p["N"] + (perp_back - p["margin"]) - p["fee_in"] + p["funding"]
        trades.append({"sym": sym, "entry_day": p["entry_day"], "exit_day": day, "reason": reason, "N": p["N"],
                       "funding": p["funding"], "pnl": pnl, "ret_pct": pnl / (p["N"] + p["margin"]) * 100})

    def accrue(sym, upto_ms, pay_until_ms=None):
        """Funding events (entry, upto] for sym; events after pay_until_ms are skipped (liquidation day)."""
        nonlocal cash, funding_income
        p = pos[sym]
        f = m.funding.get(sym)
        if f is None:
            return
        t, rate = f["t"], f["rate"]
        i = p["fi"]
        while i < len(t) and t[i] <= upto_ms:
            if pay_until_ms is None or t[i] <= pay_until_ms:
                d = int((int(t[i]) - 1) // DAY_MS)
                price = perp_close_day(sym, d)
                if price is None:
                    price = p["p_last"]
                amt = p["q"] * price * float(rate[i])
                cash += amt
                p["funding"] += amt
                funding_income += amt
            i += 1
        p["fi"] = i

    for r in range(max(r0, 0), r1):
        day = XS.day_str(S.d0 + r)
        dms = day_ms(day)
        rp = P.row(day)
        blocked = set()
        # 1) liquidations during day d-1 (no funding of that day)
        for sym in list(pos):
            p = pos[sym]
            hi = px(P, P.high, rp - 1, sym)
            if hi is not None and hi >= p["liq"]:
                accrue(sym, dms, pay_until_ms=dms - DAY_MS)
                sc = spot_close(sym, r - 1) or p["s_last"]
                close(sym, sc, None, XS.day_str(S.d0 + r - 1), "liquidation", lost_margin=True)
                liquidations += 1
                blocked.add(sym)
        # 2) funding up to 00:00 of day d, last known prices
        for sym in list(pos):
            accrue(sym, dms)
            p = pos[sym]
            sc, pc = spot_close(sym, r - 1), px(P, P.close, rp - 1, sym)
            p["s_last"] = sc if sc is not None else p["s_last"]
            p["p_last"] = pc if pc is not None else p["p_last"]
        # 3) equity mark
        eq = cash + sum(p["q"] * p["s_last"] + p["margin"] + p["q"] * (p["perp_entry"] - p["p_last"]) for p in pos.values())
        marks[day] = eq
        n_days += 1
        held_days += bool(pos)

        def first(sym):
            a, b = px(S, S.first_close, r, sym), px(P, P.first_close, rp, sym)
            return a, b

        # 4) margin stop (short equity under margin_stop x margin)
        for sym in list(pos):
            p = pos[sym]
            if p["margin"] + p["q"] * (p["perp_entry"] - p["p_last"]) < cfg.margin_stop * p["margin"]:
                a, b = first(sym)
                close(sym, a if a is not None else p["s_last"], b if b is not None else p["p_last"], day, "margin_stop")
                margin_stops += 1
                blocked.add(sym)
        # 5) delisting of either leg inside the window
        for sym in list(pos):
            js, jp = S.col.get(sym), P.col.get(sym)
            if (js is None or s_last[js] < r) or (jp is None or p_last[jp] < rp):
                p = pos[sym]
                close(sym, p["s_last"], p["p_last"], day, "delisted")
                blocked.add(sym)
        # 6) signal exits and entries
        uni = eligible(m, day, cfg)
        sc = signal(m, day, cfg, uni)
        buyable = frozenset(s for s in uni if None not in first(s))
        want = list(chooser(XS.Ctx(day, 0, frozenset(pos), uni, sc, cfg, buyable)))
        n_sell = 0
        for sym in [s for s in pos if s not in want]:
            a, b = first(sym)
            if a is None or b is None:
                continue                                    # no 00:00 bar on a leg: try again tomorrow
            close(sym, a, b, day, "rotate")
            n_sell += 1
        new = [s for s in want if s not in pos and s not in blocked and None not in first(s)]
        n_buy = 0
        if new:
            cap = min(cash / len(new), eq / max(1, cfg.K))
            if cap >= cfg.min_trade_usdt:
                for sym in new:
                    a, b = first(sym)
                    N = cap / (1 + 1.0 / cfg.lev)
                    fill_s = a * (1 + fees.slip)
                    q = N / fill_s * (1 - fees.spot_taker)
                    perp_entry = b * (1 - fees.slip)
                    fee_in = q * perp_entry * fees.perp_taker
                    margin = N / cfg.lev
                    cash -= N + margin + fee_in
                    turnover += N + q * b
                    t = m.funding.get(sym, {"t": np.zeros(0, dtype=np.int64)})["t"]
                    entry_ms = dms + FILL_OFFSET_MS
                    pos[sym] = {"q": q, "N": N, "margin": margin, "perp_entry": perp_entry, "fee_in": fee_in,
                                "liq": liq_price(perp_entry, cfg.lev, cfg.mmr), "entry_day": day, "funding": 0.0,
                                "fi": int(np.searchsorted(np.asarray(t, dtype=np.int64), entry_ms, side="right")),
                                "s_last": a, "p_last": b}
                    n_buy += 1
        moves[(0, day)] = (n_sell, n_buy)
    # window edge: liquidations of the last day, funding to 00:00 of day_to, mark, then sell at the last closes
    if r1 > max(r0, 0):
        end_ms = day_ms(day_to)
        rp = P.row(day_to)
        for sym in list(pos):
            p = pos[sym]
            hi = px(P, P.high, rp - 1, sym)
            if hi is not None and hi >= p["liq"]:
                accrue(sym, end_ms, pay_until_ms=end_ms - DAY_MS)
                sc = spot_close(sym, r1 - 1) or p["s_last"]
                close(sym, sc, None, XS.day_str(S.d0 + r1 - 1), "liquidation", lost_margin=True)
                liquidations += 1
        for sym in list(pos):
            accrue(sym, end_ms)
            p = pos[sym]
            sc, pc = spot_close(sym, r1 - 1), px(P, P.close, rp - 1, sym)
            p["s_last"] = sc if sc is not None else p["s_last"]
            p["p_last"] = pc if pc is not None else p["p_last"]
        marks[day_to] = cash + sum(p["q"] * p["s_last"] + p["margin"] + p["q"] * (p["perp_entry"] - p["p_last"])
                                   for p in pos.values())
        for sym in list(pos):
            p = pos[sym]
            close(sym, p["s_last"], p["p_last"], day_to, "flush")
    days = sorted(marks)
    series = [(a, (marks[b] / marks[a] - 1) * 100) for a, b in zip(days, days[1:]) if marks[a] > 0]
    pnl_sum = sum(t["pnl"] for t in trades)
    avg_eq = (sum(marks.values()) / len(marks)) if marks else start_equity
    return {"series": series, "agg": DR.aggregate(series), "trades": trades, "final_equity": cash,
            "conservation_ok": abs((cash - start_equity) - pnl_sum) < 1e-6 * max(1.0, start_equity) and not pos,
            "moves": moves, "funding_income": funding_income, "liquidations": liquidations, "margin_stops": margin_stops,
            "turnover_per_day": (turnover / avg_eq / n_days) if (n_days and avg_eq > 0) else 0.0,
            "exposure_share": (held_days / n_days) if n_days else 0.0}


def random_control(market: Market, cfg: CarryConfig, fees: Fees, day_from: str, day_to: str, moves: dict, n: int = 50,
                   start_equity: float = START_EQUITY) -> list:
    return [simulate(market, cfg, fees, day_from, day_to, chooser=XS.matched_random(moves, seed),
                     start_equity=start_equity)["agg"].get("mean_pct") for seed in range(n)]
