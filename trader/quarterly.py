"""QB -- quarterly basis carry: spot long + USD-M quarterly short of the same coin held to expiry
(quarterly-basis-qb-1029-spec.md §3).

  calendar  expiry = 08:00 UTC on the last Friday of Mar / Jun / Sep / Dec; symbol PAIR_YYMMDD
  decision  daily at the close of the 00:00-01:00 UTC hour; nearest contract with >= 14 days to expiry
  position  N = pair capital / 2 (L = 1 isolated): spot N, margin N; exit at the close of the 07:00-08:00 hour on expiry
  risk      liquidation when a futures 1h high reaches entry x (1 + 1/L - MMR) (margin lost); margin stop at the 00:00
            mark when the short's equity < 50% of its margin
  costs     carry.Fees on both legs (spot taker + slip, futures taker + slip)
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import daily_revenue as DR        # noqa: E402

H = 3_600_000
D = 86_400_000
MIN_DAYS = 14
LEV = 1.0
MMR = 0.01
MARGIN_STOP = 0.5
PAIRS = ("BTCUSDT", "ETHUSDT")


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def expiry_of(year: int, q: int) -> int:
    month = 3 * q
    last = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
    while last.weekday() != 4:                     # Friday
        last -= timedelta(days=1)
    return _ms(last.replace(hour=8))


def expiries(year_from: int = 2023, year_to: int = 2026) -> list:
    return [expiry_of(y, q) for y in range(year_from, year_to + 1) for q in (1, 2, 3, 4)]


def symbol_of(pair: str, expiry_ms: int) -> str:
    return f"{pair}_{datetime.fromtimestamp(expiry_ms / 1000, timezone.utc):%y%m%d}"


def parse_expiry(symbol: str) -> int:
    d = datetime.strptime(symbol.split("_")[1], "%y%m%d").replace(hour=8, tzinfo=timezone.utc)
    return _ms(d)


def pick_contract(decision_ms: int, exps, min_days: int = MIN_DAYS, max_expiry_ms: int | None = None):
    ok = [e for e in sorted(exps) if e - decision_ms >= min_days * D and (max_expiry_ms is None or e < max_expiry_ms)]
    return ok[0] if ok else None


def basis_ann(F: float, S: float, decision_ms: int, expiry_ms: int) -> float:
    days = (expiry_ms - decision_ms) / D
    return (F / S - 1) * 365.0 / days if days > 0 and S > 0 else float("nan")


def _idx(arr, t: int):
    tt = np.asarray(arr["t"], dtype=np.int64)
    i = int(np.searchsorted(tt, t))
    return i if i < len(tt) and int(tt[i]) == t else None


def _close_at(arr, t):
    i = _idx(arr, t)
    if i is None:
        return None
    v = float(arr["c"][i])
    return v if math.isfinite(v) and v > 0 else None


def simulate(spot: dict, fut: dict, theta: float, fees, day_from: str, day_to: str, pairs=PAIRS,
             start_equity: float = 1_000_000.0, expiries=None, max_expiry_ms: int | None = None) -> dict:
    exps = sorted(expiries) if expiries is not None else sorted({parse_expiry(s) for s in fut})
    d0 = _ms(datetime.strptime(day_from, "%Y-%m-%d").replace(tzinfo=timezone.utc))
    d1 = _ms(datetime.strptime(day_to, "%Y-%m-%d").replace(tzinfo=timezone.utc))
    share = 1.0 / len(pairs)
    cash = float(start_equity)
    pos, trades, marks = {}, [], {}
    liq_n = stop_n = 0

    def close(pair, s_px, f_px, t, reason, lost=False):
        nonlocal cash
        p = pos.pop(pair)
        q = p["q"]
        proceeds = q * s_px * (1 - fees.slip) * (1 - fees.spot_taker)
        if lost:
            back = 0.0
        else:
            fill = f_px * (1 + fees.slip)
            back = p["margin"] + q * (p["f_entry"] - fill) - q * fill * fees.perp_taker
        cash += proceeds + back
        pnl = proceeds - p["N"] + back - p["margin"] - p["fee_in"]
        trades.append({"pair": pair, "symbol": p["symbol"], "entry_ms": p["entry_ms"], "exit_ms": t, "reason": reason,
                       "N": p["N"], "basis_ann": p["basis_ann"], "pnl": pnl, "ret_pct": pnl / (p["N"] + p["margin"]) * 100})

    def advance(pair, upto_ms):
        """Liquidation or expiry of `pair` between its last check and upto_ms (both legs' 1h bars)."""
        nonlocal liq_n
        p = pos[pair]
        fa, sa = fut[p["symbol"]], spot[pair]
        t = p["checked"] + H
        exp_bar = p["expiry"] - H
        while t <= min(upto_ms - H, exp_bar):
            i = _idx(fa, t)
            if i is not None:
                hi = float(fa["h"][i])
                if math.isfinite(hi) and hi >= p["liq"] and t != exp_bar:
                    s = _close_at(sa, t) or p["s_last"]
                    close(pair, s, None, t + H, "liquidation", lost=True)
                    liq_n += 1
                    return
                c = float(fa["c"][i])
                if math.isfinite(c):
                    p["f_last"] = c
            s = _close_at(sa, t)
            if s is not None:
                p["s_last"] = s
            if t == exp_bar:
                close(pair, p["s_last"], p["f_last"], t + H, "expiry")
                return
            p["checked"] = t
            t += H

    def equity():
        return cash + sum(p["q"] * p["s_last"] + p["margin"] + p["q"] * (p["f_entry"] - p["f_last"]) for p in pos.values())

    day = d0
    while day <= d1:
        for pair in list(pos):
            advance(pair, day)
        eq = equity()
        marks[datetime.fromtimestamp(day / 1000, timezone.utc).strftime("%Y-%m-%d")] = eq
        if day == d1:
            break
        for pair in list(pos):                     # margin stop at the mark, closed at the decision bar
            p = pos[pair]
            if p["margin"] + p["q"] * (p["f_entry"] - p["f_last"]) < MARGIN_STOP * p["margin"]:
                s = _close_at(spot[pair], day) or p["s_last"]
                f = _close_at(fut[p["symbol"]], day) or p["f_last"]
                close(pair, s, f, day + H, "margin_stop")
                stop_n += 1
        for pair in pairs:                         # decision at the close of the 00:00 hour
            if pair in pos or pair not in spot:
                continue
            dec = day + H
            e = pick_contract(dec, exps, max_expiry_ms=max_expiry_ms)
            if e is None:
                continue
            sym = symbol_of(pair, e)
            if sym not in fut:
                continue
            S, F = _close_at(spot[pair], day), _close_at(fut[sym], day)
            if S is None or F is None:
                continue
            b = basis_ann(F, S, dec, e)
            if not (math.isfinite(b) and b >= theta):
                continue
            cap = min(eq * share, cash)
            N = cap / (1 + 1.0 / LEV)
            fill_s = S * (1 + fees.slip)
            q = N / fill_s * (1 - fees.spot_taker)
            f_entry = F * (1 - fees.slip)
            fee_in = q * f_entry * fees.perp_taker
            margin = N / LEV
            if N + margin + fee_in > cash + 1e-9:
                continue
            cash -= N + margin + fee_in
            pos[pair] = {"symbol": sym, "q": q, "N": N, "margin": margin, "f_entry": f_entry, "fee_in": fee_in,
                         "liq": f_entry * (1 + 1.0 / LEV - MMR), "expiry": e, "entry_ms": dec, "checked": day,
                         "s_last": S, "f_last": F, "basis_ann": b}
        day += D
    for pair in list(pos):                         # window edge: closed at the last known prices
        p = pos[pair]
        close(pair, p["s_last"], p["f_last"], d1, "flush")
    days = sorted(marks)
    series = [(a, (marks[b] / marks[a] - 1) * 100) for a, b in zip(days, days[1:]) if marks[a] > 0]
    pnl = sum(t["pnl"] for t in trades)
    return {"series": series, "agg": DR.aggregate(series), "trades": trades, "final_equity": cash,
            "liquidations": liq_n, "margin_stops": stop_n,
            "conservation_ok": abs((cash - start_equity) - pnl) < 1e-6 * max(1.0, start_equity) and not pos}
