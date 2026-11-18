"""FC-STK -- the FC_PASSIVE carry with a staking token on the long leg (fc-stk-staking-leg-1110-spec.md).

  hold       {coin: instrument}: the long leg of the coin's pair is the instrument (WBETHUSDT for ETHUSDT); the short is
             the coin's USD-M perp with quantity q_perp = q_hold * rho_d
  rho_d      median of instrument close / coin close over the RATIO_DAYS days before d (only days < d); none known ->
             the first 15m closes of d
  re-hedge   once a day at the signals: if q_hold * rho_d >= q_perp * (1 + BAND) the short is raised to q_hold * rho_d at
             the perp's first 15m close; only upward; margin (dq * fill / L) and fee from the wallet, else the day is
             skipped and counted; the short's entry price becomes the weighted average
  the rest   carry.simulate unchanged: funding on the short's quantity, liquidation, margin stop, delisting, marks, flush;
             the held leg pays the spot fee and slip_hold, everything else carry's fees
  expected   per series day (a, b): sum over held legs of (leg value / equity at mark a) x (close ratio at b / at a - 1)

  (from trader/) ../pyembed/python.exe -u fcstk.py run
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import carry as C                 # noqa: E402
import daily_revenue as DR        # noqa: E402
import trader_config as TC        # noqa: E402
import xsmom as XS                # noqa: E402

SPEC = "fc-stk-staking-leg-1110-spec.md"
HOLD = {"ETHUSDT": "WBETHUSDT"}
BAND = 0.005
RATIO_DAYS = 7
SLIP_HOLD = 0.0020
WINDOW = ("2024-01-01", "2026-01-01")
REFERENCE = ("2023-08-01", "2026-01-01")
BLOCKS = (20, 60)
SLIP_SENSITIVITY = (0.0008, 0.0020, 0.0050)
GAP_LEVELS = (0.01, 0.05, 0.20)
RESIDUAL_MAX = 0.01
DD_TOL_PP = 0.10
EXPLAINED = (0.75, 1.35)
YEARS = ("2024", "2025")
OUT = TC.RUNTIME / "experiments" / "FCSTK_TRAIN.json"


def ratio_at(S: XS.Panel, inst: str, coin: str, r: int, ratio_days: int = RATIO_DAYS):
    """rho for row r from rows r - ratio_days .. r - 1; None if no day has both closes."""
    ji, jc = S.col.get(inst), S.col.get(coin)
    if ji is None or jc is None:
        return None
    lo = max(0, r - ratio_days)
    vals = []
    for k in range(lo, min(r, S.n)):
        a, b = S.close[k, ji], S.close[k, jc]
        if math.isfinite(a) and math.isfinite(b) and b > 0:
            vals.append(a / b)
    return float(np.median(vals)) if vals else None


def simulate(market: C.Market, cfg: C.CarryConfig, fees: C.Fees, day_from: str, day_to: str, hold=None, chooser=None,
             slip_hold: float = SLIP_HOLD, band: float = BAND, ratio_days: int = RATIO_DAYS,
             start_equity: float = C.START_EQUITY) -> dict:
    hold = HOLD if hold is None else dict(hold)
    chooser = chooser or C.default_chooser(cfg)
    m = market.cut(day_to)
    S, P = m.spot, m.perp
    r0, r1 = S.row(day_from), S.row(day_to)
    s_last, p_last = S.last_bar_rows(), P.last_bar_rows()
    cash = float(start_equity)
    pos = {}
    trades, moves, marks, legs = [], {}, {}, {}
    funding_income, liquidations, margin_stops, turnover = 0.0, 0, 0, 0.0
    held_days = n_days = 0
    rehedges = rehedge_skipped = 0
    max_resid = 0.0

    def inst(sym):
        return hold.get(sym, sym)

    def slip_of(sym):
        return slip_hold if sym in hold else fees.slip

    def px(panel, arr, r, sym):
        j = panel.col.get(sym)
        if j is None or not (0 <= r < panel.n):
            return None
        v = arr[r, j]
        return float(v) if math.isfinite(v) else None

    def spot_close(sym, r):
        return px(S, S.close, r, inst(sym))

    def perp_close_day(sym, d):
        return px(P, P.close, d - P.d0, sym)

    def close(sym, s_raw, p_raw, day, reason, lost_margin=False):
        nonlocal cash, turnover
        p = pos.pop(sym)
        q, qp = p["q"], p["qp"]
        s_px = s_raw * (1 - slip_of(sym))
        proceeds = q * s_px * (1 - fees.spot_taker)
        turnover += q * s_raw
        if lost_margin:
            perp_back = 0.0
        else:
            fill = p_raw * (1 + fees.slip)
            fee = qp * fill * fees.perp_taker
            perp_back = p["margin"] + qp * (p["perp_entry"] - fill) - fee
            turnover += qp * p_raw
        cash += proceeds + perp_back
        pnl = proceeds - p["N"] + (perp_back - p["margin"]) - p["fee_in"] + p["funding"]
        trades.append({"sym": sym, "entry_day": p["entry_day"], "exit_day": day, "reason": reason, "N": p["N"],
                       "funding": p["funding"], "pnl": pnl, "ret_pct": pnl / (p["N"] + p["margin"]) * 100})

    def accrue(sym, upto_ms, pay_until_ms=None):
        nonlocal cash, funding_income
        p = pos[sym]
        f = m.funding.get(sym)
        if f is None:
            return
        t, rate = f["t"], f["rate"]
        i = p["fi"]
        while i < len(t) and t[i] <= upto_ms:
            if pay_until_ms is None or t[i] <= pay_until_ms:
                d = int((int(t[i]) - 1) // C.DAY_MS)
                price = perp_close_day(sym, d)
                if price is None:
                    price = p["p_last"]
                amt = p["qp"] * price * float(rate[i])
                cash += amt
                p["funding"] += amt
                funding_income += amt
            i += 1
        p["fi"] = i

    def equity():
        return cash + sum(p["q"] * p["s_last"] + p["margin"] + p["qp"] * (p["perp_entry"] - p["p_last"])
                          for p in pos.values())

    def record_legs(day, eq):
        legs[day] = (eq, {s: p["q"] * p["s_last"] for s, p in pos.items() if s in hold})

    for r in range(max(r0, 0), r1):
        day = XS.day_str(S.d0 + r)
        dms = C.day_ms(day)
        rp = P.row(day)
        blocked = set()
        # 1) liquidations during day d-1
        for sym in list(pos):
            p = pos[sym]
            hi = px(P, P.high, rp - 1, sym)
            if hi is not None and hi >= p["liq"]:
                accrue(sym, dms, pay_until_ms=dms - C.DAY_MS)
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
        eq = equity()
        marks[day] = eq
        record_legs(day, eq)
        n_days += 1
        held_days += bool(pos)

        def first(sym):
            return px(S, S.first_close, r, inst(sym)), px(P, P.first_close, rp, sym)

        # 4) margin stop
        for sym in list(pos):
            p = pos[sym]
            if p["margin"] + p["qp"] * (p["perp_entry"] - p["p_last"]) < cfg.margin_stop * p["margin"]:
                a, b = first(sym)
                close(sym, a if a is not None else p["s_last"], b if b is not None else p["p_last"], day, "margin_stop")
                margin_stops += 1
                blocked.add(sym)
        # 5) delisting of either leg
        for sym in list(pos):
            js, jp = S.col.get(inst(sym)), P.col.get(sym)
            if (js is None or s_last[js] < r) or (jp is None or p_last[jp] < rp):
                p = pos[sym]
                close(sym, p["s_last"], p["p_last"], day, "delisted")
                blocked.add(sym)
        # 6) signal exits, re-hedge, entries
        uni = C.eligible(m, day, cfg)
        sc = C.signal(m, day, cfg, uni)
        buyable = frozenset(s for s in uni if None not in first(s))
        want = list(chooser(XS.Ctx(day, 0, frozenset(pos), uni, sc, cfg, buyable)))
        n_sell = 0
        for sym in [s for s in pos if s not in want]:
            a, b = first(sym)
            if a is None or b is None:
                continue
            close(sym, a, b, day, "rotate")
            n_sell += 1
        for sym in [s for s in pos if s in hold]:
            p = pos[sym]
            rho = ratio_at(S, hold[sym], sym, r, ratio_days)
            if rho is None:
                continue
            target = p["q"] * rho
            if target >= p["qp"] * (1 + band):
                b = first(sym)[1]
                if b is not None:
                    dq = target - p["qp"]
                    fill = b * (1 - fees.slip)
                    dm = dq * fill / cfg.lev
                    fee = dq * fill * fees.perp_taker
                    if cash >= dm + fee:
                        cash -= dm + fee
                        turnover += dq * b
                        p["perp_entry"] = (p["qp"] * p["perp_entry"] + dq * fill) / target
                        p["qp"] = target
                        p["margin"] += dm
                        p["fee_in"] += fee
                        p["liq"] = C.liq_price(p["perp_entry"], cfg.lev, cfg.mmr)
                        rehedges += 1
                    else:
                        rehedge_skipped += 1
            max_resid = max(max_resid, abs(p["q"] * rho - p["qp"]) / p["qp"])
        new = [s for s in want if s not in pos and s not in blocked and None not in first(s)]
        n_buy = 0
        if new:
            cap = min(cash / len(new), eq / max(1, cfg.K))
            if cap >= cfg.min_trade_usdt:
                for sym in new:
                    a, b = first(sym)
                    N = cap / (1 + 1.0 / cfg.lev)
                    fill_s = a * (1 + slip_of(sym))
                    q = N / fill_s * (1 - fees.spot_taker)
                    if sym in hold:
                        rho = ratio_at(S, hold[sym], sym, r, ratio_days)
                        qp = q * (rho if rho is not None else a / b)
                    else:
                        qp = q
                    perp_entry = b * (1 - fees.slip)
                    fee_in = qp * perp_entry * fees.perp_taker
                    margin = N / cfg.lev
                    cash -= N + margin + fee_in
                    turnover += N + qp * b
                    t = m.funding.get(sym, {"t": np.zeros(0, dtype=np.int64)})["t"]
                    pos[sym] = {"q": q, "qp": qp, "N": N, "margin": margin, "perp_entry": perp_entry, "fee_in": fee_in,
                                "liq": C.liq_price(perp_entry, cfg.lev, cfg.mmr), "entry_day": day, "funding": 0.0,
                                "fi": int(np.searchsorted(np.asarray(t, dtype=np.int64), dms + C.FILL_OFFSET_MS,
                                                          side="right")),
                                "s_last": a, "p_last": b}
                    n_buy += 1
        moves[(0, day)] = (n_sell, n_buy)
    # window edge, as carry.simulate
    if r1 > max(r0, 0):
        end_ms = C.day_ms(day_to)
        rp = P.row(day_to)
        for sym in list(pos):
            p = pos[sym]
            hi = px(P, P.high, rp - 1, sym)
            if hi is not None and hi >= p["liq"]:
                accrue(sym, end_ms, pay_until_ms=end_ms - C.DAY_MS)
                sc = spot_close(sym, r1 - 1) or p["s_last"]
                close(sym, sc, None, XS.day_str(S.d0 + r1 - 1), "liquidation", lost_margin=True)
                liquidations += 1
        for sym in list(pos):
            accrue(sym, end_ms)
            p = pos[sym]
            sc, pc = spot_close(sym, r1 - 1), px(P, P.close, rp - 1, sym)
            p["s_last"] = sc if sc is not None else p["s_last"]
            p["p_last"] = pc if pc is not None else p["p_last"]
        marks[day_to] = equity()
        record_legs(day_to, marks[day_to])
        for sym in list(pos):
            p = pos[sym]
            close(sym, p["s_last"], p["p_last"], day_to, "flush")
    days = sorted(marks)
    series = [(a, (marks[b] / marks[a] - 1) * 100) for a, b in zip(days, days[1:]) if marks[a] > 0]

    def close_ratio(sym, mark_day):            # the closes a mark uses: day before the mark
        k = S.row(mark_day) - 1
        a, b = px(S, S.close, k, hold[sym]), px(S, S.close, k, sym)
        return a / b if (a is not None and b) else None

    expected = []
    for a, b in zip(days, days[1:]):
        if marks[a] <= 0:
            continue
        eq_a, vals = legs[a]
        v = 0.0
        for sym, val in vals.items():
            ra, rb = close_ratio(sym, a), close_ratio(sym, b)
            if ra and rb:
                v += val / eq_a * (rb / ra - 1)
        expected.append((a, v * 100))
    pnl_sum = sum(t["pnl"] for t in trades)
    avg_eq = (sum(marks.values()) / len(marks)) if marks else start_equity
    return {"series": series, "agg": DR.aggregate(series), "trades": trades, "final_equity": cash,
            "conservation_ok": abs((cash - start_equity) - pnl_sum) < 1e-6 * max(1.0, start_equity) and not pos,
            "moves": moves, "funding_income": funding_income, "liquidations": liquidations, "margin_stops": margin_stops,
            "turnover_per_day": (turnover / avg_eq / n_days) if (n_days and avg_eq > 0) else 0.0,
            "exposure_share": (held_days / n_days) if n_days else 0.0,
            "rehedges": rehedges, "rehedge_skipped": rehedge_skipped, "max_residual_delta": max_resid,
            "expected": expected}


def judge(e: dict) -> dict:
    """Spec §5. `explained` failing is an ANOMALY: no verdict until the engine and data are examined."""
    failed = []
    p = e.get("paired") or {}
    los = [(p.get(str(b)) or {}).get("lo95") for b in BLOCKS]
    if any(lo is None or lo <= 0 for lo in los):
        failed.append("ci")
    py = e.get("paired_by_year") or {}
    if any(py.get(y) is None or py[y] <= 0 for y in YEARS):
        failed.append("years")
    if e["margin_stops"] > e["fc_margin_stops"] or e["liquidations"] > e["fc_liquidations"]:
        failed.append("stops")
    if e["max_dd_pct"] < e["fc_max_dd_pct"] - DD_TOL_PP:
        failed.append("drawdown")
    if e["max_residual_delta"] > RESIDUAL_MAX:
        failed.append("residual_delta")
    x = e.get("explained_share")
    if x is None or not (EXPLAINED[0] <= x <= EXPLAINED[1]):
        failed.append("explained")
    return {"passes": not failed, "anomaly": "explained" in failed, "failed": failed}


def intraday_gaps(hold15, coin15, ratio_by_day: dict, levels=GAP_LEVELS) -> dict:
    """15m bars where the instrument's low is below rho_d x the coin's close of the same bar by at least each level."""
    tc = np.asarray(coin15["t"], dtype=np.int64)
    cc = np.asarray(coin15["c"], dtype=float)
    idx = {int(t): i for i, t in enumerate(tc)}
    n = {lv: 0 for lv in levels}
    bars, worst, worst_t = 0, 0.0, None
    for t, lo in zip(np.asarray(hold15["t"], dtype=np.int64), np.asarray(hold15["l"], dtype=float)):
        rho = ratio_by_day.get(XS.day_str(int(t) // C.DAY_MS))
        i = idx.get(int(t))
        if rho is None or i is None or not (math.isfinite(lo) and math.isfinite(cc[i]) and cc[i] > 0):
            continue
        bars += 1
        g = lo / (rho * cc[i]) - 1
        for lv in levels:
            if g <= -lv:
                n[lv] += 1
        if worst_t is None or g < worst:
            worst, worst_t = g, int(t)
    return {"n_bars": n, "worst": worst, "worst_t": worst_t, "bars": bars}


# ------------------------------------------------------------------ the run (spec §5)
def _year_means(series) -> dict:
    by = {}
    for d, v in series:
        by.setdefault(d[:4], []).append(v)
    return {y: float(np.mean(v)) for y, v in sorted(by.items())}


def evaluate(market, lo: str, hi: str, slip_hold: float = SLIP_HOLD) -> dict:
    import experiment_fc as EF
    cfg, fees = EF.passive_config(), C.fees(1)
    fc = C.simulate(market, cfg, fees, lo, hi, start_equity=C.START_EQUITY)
    st = simulate(market, cfg, fees, lo, hi, slip_hold=slip_hold, start_equity=C.START_EQUITY)
    df, ds = dict(fc["series"]), dict(st["series"])
    diff = [(d, ds[d] - df[d]) for d in ds if d in df]
    exp = dict(st["expected"])
    sum_diff = sum(v for _, v in diff)
    sum_exp = sum(exp.get(d, 0.0) for d, _ in diff)
    reasons = {}
    for t in st["trades"]:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    return {"window": [lo, hi], "slip_hold": slip_hold,
            "fc": {k: fc[k] for k in ("agg", "margin_stops", "liquidations", "funding_income", "conservation_ok")},
            "fcstk": {k: st[k] for k in ("agg", "margin_stops", "liquidations", "funding_income", "conservation_ok",
                                         "rehedges", "rehedge_skipped", "max_residual_delta")},
            "fcstk_exit_reasons": reasons,
            "paired": {str(b): DR.paired(st["series"], fc["series"], block=b) for b in BLOCKS},
            "paired_by_year": _year_means(diff), "r_by_year": _year_means(st["series"]), "fc_by_year": _year_means(fc["series"]),
            "sum_paired_pct": sum_diff, "sum_expected_pct": sum_exp,
            "explained_share": (sum_exp / sum_diff) if sum_diff else None}


def run() -> dict:
    import experiment_fc as EF
    import history_store as HS
    import provenance as PV
    PV.require_clean()
    t0 = time.time()
    market = EF.load_market()
    for s in list(HOLD) + list(HOLD.values()):
        if s not in market.spot.col:
            raise RuntimeError(f"FC-STK: {s} is not in the spot panel")
    main = evaluate(market, *WINDOW)
    e = {"paired": {b: {"lo95": v["lo95"]} for b, v in main["paired"].items()}, "paired_by_year": main["paired_by_year"],
         "margin_stops": main["fcstk"]["margin_stops"], "fc_margin_stops": main["fc"]["margin_stops"],
         "liquidations": main["fcstk"]["liquidations"], "fc_liquidations": main["fc"]["liquidations"],
         "max_dd_pct": main["fcstk"]["agg"]["max_dd_pct"], "fc_max_dd_pct": main["fc"]["agg"]["max_dd_pct"],
         "max_residual_delta": main["fcstk"]["max_residual_delta"], "explained_share": main["explained_share"]}
    verdict = judge(e)
    sens = {str(s): {k: v for k, v in evaluate(market, *WINDOW, slip_hold=s).items()
                     if k in ("paired", "paired_by_year", "explained_share")} for s in SLIP_SENSITIVITY if s != SLIP_HOLD}
    ref = {k: v for k, v in evaluate(market, *REFERENCE).items() if k in ("fc", "fcstk", "paired", "paired_by_year")}
    S = market.spot
    ratio_by_day = {}
    for r in range(S.row(WINDOW[0]), S.row(WINDOW[1])):
        rho = ratio_at(S, HOLD["ETHUSDT"], "ETHUSDT", r)
        if rho is not None:
            ratio_by_day[XS.day_str(S.d0 + r)] = rho
    gaps = intraday_gaps(HS.load(HS.path_of(HOLD["ETHUSDT"])), HS.load(HS.path_of("ETHUSDT")), ratio_by_day)
    out = {"spec": SPEC, "stage": "train", "window": WINDOW, "main": main, "criterion_input": e, "judge": verdict,
           "slip_sensitivity": sens, "reference": ref, "intraday_gaps": {**gaps, "n_bars": {str(k): v for k, v in gaps["n_bars"].items()}},
           "verdict": ("ANOMALY: examine the engine and data before any verdict" if verdict["anomaly"] else
                       "propose a single 2026 look to the operator" if verdict["passes"] else "no gain shown: STOP"),
           "provenance": PV.block(), "minutes": round((time.time() - t0) / 60, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("criterion_input", "judge", "verdict")}, indent=1, default=str))
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["run"]:
        run()
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
