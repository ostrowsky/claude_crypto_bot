"""PX -- market-neutral pairs trading: long spot of the cheap leg, short USD-M perp of the rich leg
(pairs-trading-px-1104-spec.md §3).

  spread_z / ret_corr   z of the log-ratio spread over the last F days; correlation of the last F daily log returns
  select / legs         pairs by |z| desc, one coin in one pair; the cheap leg is bought, the rich one shorted
  exit_reason           converged |z| <= 0.5, stop |z| >= 4, time 20 days, delisted
  open_pos / close_pos  both-leg costs (carry.Fees), isolated perp margin N at L = 1, funding of the short leg
  simulate              daily book on carry.Market: decisions on closes to d - 1, fills at the first 15m close of d;
                        plan = [(entry_day, exit_day)] -> random control (random pair of the day's pool, random side)
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import carry as C                 # noqa: E402
import daily_revenue as DR        # noqa: E402
import xsmom as XS                # noqa: E402

DAY_MS = C.DAY_MS
POOL = 30
CORR_MIN = 0.80
CORR_LAG = 5
Z_OUT, Z_STOP, HMAX = 0.5, 4.0, 20
SLOTS = 10
LEV, MMR = 1.0, C.MMR
START_EQUITY = 1_000_000.0


def spread_z(la, lb, F: int):
    la, lb = np.asarray(la, dtype=float), np.asarray(lb, dtype=float)
    if len(la) < F or len(lb) < F or F < 2:
        return None
    w = la[-F:] - lb[-F:]
    if not np.all(np.isfinite(w)):
        return None
    sd = float(np.std(w, ddof=1))
    if sd <= 0:
        return None
    return float((w[-1] - w.mean()) / sd)


def ret_corr(la, lb, F: int):
    la, lb = np.asarray(la, dtype=float), np.asarray(lb, dtype=float)
    if len(la) < F + 1 or len(lb) < F + 1:
        return None
    ra, rb = np.diff(la[-(F + 1):]), np.diff(lb[-(F + 1):])
    if not (np.all(np.isfinite(ra)) and np.all(np.isfinite(rb))) or ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def select(cands, free: int, busy) -> list:
    busy = set(busy)
    out = []
    for c in sorted(cands, key=lambda c: (-abs(c["z"]), c["a"], c["b"])):
        if len(out) >= free:
            break
        if c["a"] in busy or c["b"] in busy:
            continue
        out.append(c)
        busy |= {c["a"], c["b"]}
    return out


def legs(a: str, b: str, z: float):
    """z > 0: log(a / b) is high -> a is rich: buy b spot, short a perp."""
    return (b, a) if z > 0 else (a, b)


def exit_reason(z, held_days: int, delisted: bool):
    if delisted:
        return "delisted"
    if z is not None and abs(z) <= Z_OUT:
        return "converged"
    if z is not None and abs(z) >= Z_STOP:
        return "stop"
    if held_days >= HMAX:
        return "time"
    return None


def open_pos(long_sym, short_sym, s_px, p_px, N, fees) -> dict:
    fill_s = float(s_px) * (1 + fees.slip)
    perp_entry = float(p_px) * (1 - fees.slip)
    return {"long": long_sym, "short": short_sym, "N": float(N), "margin": float(N),
            "q_s": float(N) / fill_s * (1 - fees.spot_taker), "perp_entry": perp_entry, "q_p": float(N) / perp_entry,
            "fees_in": float(N) * fees.perp_taker, "funding": 0.0}


def close_pos(p, s_px, p_px, fees, lost_margin: bool = False) -> dict:
    spot = p["q_s"] * float(s_px) * (1 - fees.slip) * (1 - fees.spot_taker)
    if lost_margin:
        perp_pnl, fee_out, margin_back = -p["margin"], 0.0, 0.0
    else:
        perp_exit = float(p_px) * (1 + fees.slip)
        perp_pnl, fee_out, margin_back = p["q_p"] * (p["perp_entry"] - perp_exit), p["q_p"] * perp_exit * fees.perp_taker, p["margin"]
    pnl = spot - p["N"] + perp_pnl - p["fees_in"] - fee_out + p["funding"]
    cash_in = spot + margin_back + (0.0 if lost_margin else perp_pnl) - fee_out
    return {"pnl": pnl, "cash_in": cash_in, "spot_pnl": spot - p["N"], "perp_pnl": perp_pnl, "funding": p["funding"],
            "fees": p["fees_in"] + fee_out}


# ------------------------------------------------------------------ book
def simulate(market, cand: dict, fees, day_from: str, day_to: str, plan=None, seed=None,
             start_equity: float = START_EQUITY) -> dict:
    F, zin = int(cand["F"]), float(cand["z_in"])
    m = market.cut(day_to)
    S, P = m.spot, m.perp
    ccfg = C.CarryConfig(W=3, K=SLOTS, theta_in=0.0)
    r0, r1 = max(S.row(day_from), 0), min(S.row(day_to), S.n)
    with np.errstate(divide="ignore", invalid="ignore"):
        LS = np.log(S.close)
    s_last_row, p_last_row = S.last_bar_rows(), P.last_bar_rows()
    rnd = random.Random(seed)
    plan_in = {}
    for e, x in (plan or []):
        plan_in.setdefault(e, []).append(x)
    cash = float(start_equity)
    pos, trades, marks = {}, [], {}
    liquidations = 0

    def val(panel, arr, r, sym):
        j = panel.col.get(sym)
        if j is None or r < 0 or r >= panel.n:
            return None
        v = arr[r, j]
        return float(v) if np.isfinite(v) and v > 0 else None

    def z_of(a, b, r):
        ja, jb = S.col[a], S.col[b]
        return spread_z(LS[max(0, r - F):r, ja], LS[max(0, r - F):r, jb], F)

    def accrue(p, upto_ms):
        nonlocal cash
        f = m.funding.get(p["short"])
        if f is None:
            return
        t, rate = f["t"], f["rate"]
        i = p["fi"]
        while i < len(t) and t[i] <= upto_ms:
            price = val(P, P.close, P.row(XS.day_str(int((int(t[i]) - 1) // DAY_MS))), p["short"]) or p["p_last"]
            amt = p["q_p"] * price * float(rate[i])
            cash += amt
            p["funding"] += amt
            i += 1
        p["fi"] = i

    def finish(k, day, s_px, p_px, reason, lost=False):
        nonlocal cash
        p = pos.pop(k)
        c = close_pos(p, s_px, p_px, fees, lost_margin=lost)
        cash += c["cash_in"]
        trades.append({"long": p["long"], "short": p["short"], "a": p["a"], "b": p["b"], "entry_day": p["entry_day"],
                       "exit_day": day, "reason": reason, "z_entry": p["z"], "N": p["N"], "pnl": c["pnl"],
                       "ret_pct": c["pnl"] / (2 * p["N"]) * 100, "spot_pnl": c["spot_pnl"], "perp_pnl": c["perp_pnl"],
                       "funding": c["funding"], "fees": c["fees"],
                       "gross_price_pct": ((s_px / p["s_in"] - 1) + (-1.0 if lost else 1 - p_px / p["p_in"])) / 2 * 100,
                       "funding_pct": c["funding"] / (2 * p["N"]) * 100})

    for r in range(r0, r1):
        day = XS.day_str(S.d0 + r)
        rp, dms = P.row(day), C.day_ms(day)
        # 1) liquidation of the short leg during day d-1
        for k in list(pos):
            p = pos[k]
            h = val(P, P.high, rp - 1, p["short"])
            if h is not None and h >= p["liq"]:
                accrue(p, dms - DAY_MS)
                finish(k, XS.day_str(S.d0 + r - 1), val(S, S.close, r - 1, p["long"]) or p["s_last"], None,
                       "liquidation", lost=True)
                liquidations += 1
        # 2) funding to 00:00 of d, last prices, equity mark
        for p in pos.values():
            accrue(p, dms)
            p["s_last"] = val(S, S.close, r - 1, p["long"]) or p["s_last"]
            p["p_last"] = val(P, P.close, rp - 1, p["short"]) or p["p_last"]
        marks[day] = cash + sum(p["q_s"] * p["s_last"] + p["margin"] + p["q_p"] * (p["perp_entry"] - p["p_last"])
                                for p in pos.values())
        eq = marks[day]
        # 3) exits at the first 15m close of d
        for k in list(pos):
            p = pos[k]
            js, jp = S.col[p["long"]], P.col[p["short"]]
            gone = s_last_row[js] < r or p_last_row[jp] < rp
            if gone:
                finish(k, day, p["s_last"], p["p_last"], "delisted")
                continue
            if plan is not None:
                reason = "time" if day >= p["plan_exit"] else None
            else:
                reason = exit_reason(z_of(p["a"], p["b"], r), r - p["r_entry"], False)
            if reason:
                a, b = val(S, S.first_close, r, p["long"]), val(P, P.first_close, rp, p["short"])
                if a is None or b is None:
                    continue                                   # no 00:00 bar on a leg: try again tomorrow
                finish(k, day, a, b, reason)
        # 4) entries
        free = SLOTS - len(pos)
        if free <= 0:
            continue
        busy = {s for p in pos.values() for s in (p["long"], p["short"])}
        pool = [s for s in C.eligible(m, day, ccfg)[:POOL] if s in S.col and s in P.col]
        chosen = []
        if plan is not None:
            for x in plan_in.get(day, []):
                avail = [s for s in pool if s not in busy]
                if len(avail) < 2 or len(chosen) >= free:
                    continue
                a, b = sorted(rnd.sample(avail, 2))
                chosen.append({"a": a, "b": b, "z": rnd.choice((-1.0, 1.0)), "plan_exit": x})
                busy |= {a, b}
        else:
            cands = []
            for i, a in enumerate(pool):
                for b in pool[i + 1:]:
                    a_, b_ = sorted((a, b))
                    ja, jb = S.col[a_], S.col[b_]
                    lo_c = r - CORR_LAG - F - 1
                    if lo_c < 0:
                        continue
                    c = ret_corr(LS[lo_c:r - CORR_LAG, ja], LS[lo_c:r - CORR_LAG, jb], F)
                    if c is None or c < CORR_MIN:
                        continue
                    z = z_of(a_, b_, r)
                    if z is not None and abs(z) >= zin:
                        cands.append({"a": a_, "b": b_, "z": z})
            chosen = select(cands, free, busy)
        for c in chosen:
            long_s, short_s = legs(c["a"], c["b"], c["z"])
            sp, pp = val(S, S.first_close, r, long_s), val(P, P.first_close, rp, short_s)
            if sp is None or pp is None:
                continue
            cap = min(cash, eq / SLOTS)
            N = cap / 2
            if N < C.MIN_TRADE_USDT:
                continue
            p = open_pos(long_s, short_s, sp, pp, N, fees)
            cash -= p["N"] + p["margin"] + p["fees_in"]
            f = m.funding.get(short_s)
            fi = int(np.searchsorted(np.asarray(f["t"], dtype=np.int64), dms + C.FILL_OFFSET_MS, side="right")) if f else 0
            p.update(a=c["a"], b=c["b"], z=c["z"], entry_day=day, r_entry=r, liq=p["perp_entry"] * (1 + 1 / LEV - MMR),
                     fi=fi, s_last=sp, p_last=pp, s_in=sp, p_in=pp, plan_exit=c.get("plan_exit"))
            pos[(long_s, short_s)] = p
    end = XS.day_str(S.d0 + r1)
    for k in list(pos):                                        # window edge: closed at the last known prices
        p = pos[k]
        accrue(p, C.day_ms(end))
        finish(k, end, p["s_last"], p["p_last"], "window_end")
    marks[end] = cash
    days = sorted(marks)
    series = [(a, (marks[b] / marks[a] - 1) * 100) for a, b in zip(days, days[1:]) if marks[a] > 0]
    pnl = sum(t["pnl"] for t in trades)
    return {"series": series, "agg": DR.aggregate(series), "trades": trades, "final_equity": cash,
            "liquidations": liquidations,
            "conservation_ok": abs((cash - start_equity) - pnl) < 1e-6 * max(1.0, start_equity)}
