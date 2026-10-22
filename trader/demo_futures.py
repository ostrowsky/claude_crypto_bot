"""F0 -- readiness of the demo account for USD-M futures (carry-funding-money-1021-spec.md §10,
fc-passive-next-1022-spec.md §B).

readiness(results) turns the observations of a smoke run into GO / NO_GO. It makes no request itself.
Items 1-9 of the spec are required; a key that was not checked counts as not verified. Items 10-11 only warn:
funding credit needs a held position across a funding time, and demo funding is computed from the demo book
(measured 2026-10-21: ETH 0.0000206 on demo vs 0.0000650 live), so demo carry money is not live carry money.

run_f0(go=...) makes the observations. Without go it returns the plan and sends nothing. With go it reads the
demo futures account, sets one-way mode (only if hedge), ISOLATED margin and leverage 1 on BTCUSDT and ETHUSDT,
and sends one POST /fapi/v1/order/test. It never opens a position.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import ROUND_CEILING, ROUND_DOWN, Decimal
from pathlib import Path

import trader_config as TC

MIN_FUTURES_BALANCE = 100.0
FUNDING_GAP_WARN_BP = 1.0
F0_DIR = TC.RUNTIME / "futures"
PAIRS = ("BTCUSDT", "ETHUSDT")
LOT_NOTIONAL_USDT = 2500.0           # the shadow book's notional per pair: 10 000 / 2 pairs / (1 + 1/L), L = 1
ORDER_TEST_HEADROOM = 1.05           # order/test quantity: minimum notional x 1.05, so a tick of price cannot void it
PERMISSION_CODES = {-2014, -2015, -2008}
NO_CHANGE_CODES = {-4046, -4059}     # "No need to change margin type" / "... position side": already as wanted

REQUIRED = ("account_ok", "futures_permission", "futures_balance_usdt", "one_way_mode", "isolated_ok", "leverage1_ok",
            "commission", "order_test_ok", "lot_match_ok")

PLAN = (
    "F0 plan (nothing is sent without --go):",
    "  GET  demo-fapi /fapi/v1/time, /fapi/v2/account, /fapi/v2/balance, /fapi/v1/positionSide/dual",
    "  POST demo-fapi /fapi/v1/positionSide/dual  one-way mode, ONLY if the account is in hedge mode",
    "  POST demo-fapi /fapi/v1/marginType ISOLATED and /fapi/v1/leverage 1 for " + ", ".join(PAIRS),
    "  GET  demo-fapi /fapi/v1/commissionRate, /fapi/v1/income?incomeType=FUNDING_FEE",
    "  POST demo-fapi /fapi/v1/order/test  MARKET SELL BTCUSDT, smallest valid quantity: validated, NOT executed",
    "  GET  exchangeInfo (demo spot, demo futures); premiumIndex on demo-fapi and on live fapi (public)",
    "  no position is opened; the result goes to trader/runtime/futures/F0_<UTC>.json",
)


def _ok(key, value) -> bool:
    if key == "futures_balance_usdt":
        return isinstance(value, (int, float)) and value >= MIN_FUTURES_BALANCE
    if key == "commission":
        return isinstance(value, dict) and isinstance(value.get("taker"), (int, float))
    return value is True


def readiness(results: dict) -> dict:
    missing = [k for k in REQUIRED if k not in results or not _ok(k, results[k])]
    warnings = []
    if results.get("funding_income_seen") is not True:
        warnings.append("funding_income not verified on the demo (needs a held position across a funding time)")
    gap = results.get("demo_vs_live_funding_gap_bp")
    if gap is None:
        warnings.append("demo_vs_live_funding_gap_bp not measured")
    elif abs(gap) > FUNDING_GAP_WARN_BP:
        warnings.append("demo_vs_live_funding_gap_bp = %.2f bp: demo carry money is not live carry money" % gap)
    return {"status": "GO" if not missing else "NO_GO", "missing": missing, "warnings": warnings}


# ------------------------------------------------------------------ lot arithmetic (pure)
def _filters(sym_info: dict) -> dict:
    return {f.get("filterType"): f for f in (sym_info or {}).get("filters", [])}


def _lot(sym_info: dict, market: bool) -> tuple:
    f = _filters(sym_info)
    lot = (f.get("MARKET_LOT_SIZE") if market else None) or f.get("LOT_SIZE") or {}
    step = Decimal(str(lot.get("stepSize") or "0"))
    if step <= 0:
        step = Decimal(str((f.get("LOT_SIZE") or {}).get("stepSize") or "0"))
    return step, Decimal(str(lot.get("minQty") or "0"))


def _min_notional(sym_info: dict) -> float:
    f = _filters(sym_info)
    for key, field in (("MIN_NOTIONAL", "notional"), ("NOTIONAL", "minNotional"), ("MIN_NOTIONAL", "minNotional")):
        v = (f.get(key) or {}).get(field)
        if v not in (None, ""):
            return float(v)
    return 0.0


def _floor(qty: Decimal, step: Decimal) -> Decimal:
    return (qty / step).to_integral_value(rounding=ROUND_DOWN) * step if step > 0 else qty


def lot_match(spot_info: dict, perp_info: dict, notional: float, price: float) -> tuple:
    """-> (ok, detail). Spot qty on the spot LOT_SIZE step, perp qty on MARKET_LOT_SIZE (else LOT_SIZE); the two
    differ by at most one perp step, both reach their minimum quantity and each leg's notional its minimum."""
    raw = Decimal(str(notional)) / Decimal(str(price))
    s_step, s_min = _lot(spot_info, market=False)
    p_step, p_min = _lot(perp_info, market=True)
    qs, qp = _floor(raw, s_step), _floor(raw, p_step)
    ns, np_ = float(qs) * price, float(qp) * price
    ms, mp = _min_notional(spot_info), _min_notional(perp_info)
    checks = {"qty_spot_min": qs > 0 and qs >= s_min, "qty_perp_min": qp > 0 and qp >= p_min,
              "qty_diff": abs(qs - qp) <= p_step, "notional_spot": ns >= ms, "notional_perp": np_ >= mp}
    detail = {"notional": notional, "price": price, "qty_spot": str(qs), "qty_perp": str(qp), "step_perp": str(p_step),
              "notional_spot": round(ns, 4), "notional_perp": round(np_, 4), "min_notional_spot": ms,
              "min_notional_perp": mp, "checks": checks}
    return all(checks.values()), detail


def order_test_qty(perp_info: dict, mark: float) -> str:
    """Smallest quantity on the step, >= minQty, with notional >= min notional x ORDER_TEST_HEADROOM."""
    step, q_min = _lot(perp_info, market=True)
    need = Decimal(str(_min_notional(perp_info) * ORDER_TEST_HEADROOM / float(mark)))
    q = (need / step).to_integral_value(rounding=ROUND_CEILING) * step if step > 0 else need
    q = max(q, q_min)
    return str(q.quantize(step)) if step > 0 else str(q)


def funding_gap_bp(demo: dict, live: dict):
    common = [s for s in demo if s in live and demo[s] is not None and live[s] is not None]
    if not common:
        return None
    return max(abs(float(demo[s]) - float(live[s])) for s in common) * 10_000


# ------------------------------------------------------------------ the run
def _sym(xinfo: dict, sym: str) -> dict:
    return next((s for s in (xinfo or {}).get("symbols", []) if s.get("symbol") == sym), {})


def run_f0(go: bool = False, fut=None, spot=None, out_dir=None, now=None) -> dict:
    if not go:
        return {"status": "PLAN", "plan": list(PLAN)}
    import exchange_client as X
    fut = fut if fut is not None else X.ExchangeClient(base=X.DEMO_FAPI)
    spot = spot if spot is not None else X.ExchangeClient()
    now = now or datetime.now(timezone.utc)
    res, steps = {}, []

    def step(name, call):
        try:
            out = call()
            steps.append({"step": name, "ok": True})
            return True, out
        except X.ExchangeError as e:
            steps.append({"step": name, "ok": False, "status": e.status, "code": e.code, "msg": str(e.msg)[:200]})
            return False, e
        except Exception as e:  # one failed observation must not hide the others
            steps.append({"step": name, "ok": False, "error": repr(e)[:200]})
            return False, e

    ok, _ = step("time", fut.fut_time)
    res["time_ok"] = ok
    ok, acc = step("account", fut.fut_account)
    signed = ok
    if ok:
        res["account_ok"] = True
        res["futures_permission"] = (acc or {}).get("canTrade") is not False
    else:
        res["account_ok"] = False
        refused = isinstance(acc, X.ExchangeError) and (acc.status == 401 or acc.code in PERMISSION_CODES)
        res["futures_permission"] = False if refused else None

    if signed:
        ok, bal = step("balance", fut.fut_balance)
        if ok:
            usdt = next((b for b in (bal or []) if b.get("asset") == "USDT"), None)
            res["futures_balance_usdt"] = float(usdt.get("availableBalance") or usdt.get("balance") or 0) if usdt else 0.0
        ok, ps = step("position_side", fut.fut_position_side)
        if ok:
            dual = (ps or {}).get("dualSidePosition")
            if dual is True:
                ok2, r = step("set_one_way", lambda: fut.fut_set_position_side(False))
                res["one_way_mode"] = ok2 or (isinstance(r, X.ExchangeError) and r.code in NO_CHANGE_CODES)
            else:
                res["one_way_mode"] = dual is False
        iso, lev = [], []
        for sym in PAIRS:
            ok, r = step("margin_type_" + sym, lambda s=sym: fut.fut_margin_type(s, "ISOLATED"))
            iso.append(ok or (isinstance(r, X.ExchangeError) and r.code in NO_CHANGE_CODES))
            ok, r = step("leverage_" + sym, lambda s=sym: fut.fut_leverage(s, 1))
            lev.append(ok and int(float((r or {}).get("leverage", 0))) == 1)
        res["isolated_ok"], res["leverage1_ok"] = all(iso), all(lev)
        ok, c = step("commission", lambda: fut.fut_commission(PAIRS[0]))
        if ok:
            res["commission"] = {"taker": float(c["takerCommissionRate"]), "maker": float(c["makerCommissionRate"])}

    # public: exchange filters and funding on demo vs live
    _, perp_x = step("futures_exchange_info", lambda: fut.futures_exchange_info(base=X.DEMO_FAPI))
    _, spot_x = step("spot_exchange_info", lambda: spot.exchange_info())
    demo_rate, live_rate, mark = {}, {}, {}
    for sym in PAIRS:
        ok, p = step("premium_demo_" + sym, lambda s=sym: fut.premium_index(s, base=X.DEMO_FAPI))
        if ok:
            demo_rate[sym], mark[sym] = float(p["lastFundingRate"]), float(p["markPrice"])
        ok, p = step("premium_live_" + sym, lambda s=sym: fut.premium_index(s, base=X.LIVE_FAPI))
        if ok:
            live_rate[sym] = float(p["lastFundingRate"])
            mark.setdefault(sym, float(p.get("markPrice") or 0) or None)
    res["demo_funding_rate"], res["live_funding_rate"] = demo_rate, live_rate
    res["demo_vs_live_funding_gap_bp"] = funding_gap_bp(demo_rate, live_rate)
    if isinstance(perp_x, dict) and isinstance(spot_x, dict) and all(mark.get(s) for s in PAIRS):
        lots = {s: lot_match(_sym(spot_x, s), _sym(perp_x, s), LOT_NOTIONAL_USDT, mark[s]) for s in PAIRS}
        res["lot_match_ok"] = all(v[0] for v in lots.values())
        res["lot_match"] = {s: v[1] for s, v in lots.items()}

    if signed:
        if isinstance(perp_x, dict) and mark.get(PAIRS[0]):
            qty = order_test_qty(_sym(perp_x, PAIRS[0]), mark[PAIRS[0]])
            ok, _ = step("order_test", lambda: fut.fut_test_order(PAIRS[0], "SELL", "MARKET", quantity=qty))
            res["order_test_ok"], res["order_test_qty"] = ok, qty
        ok, rows = step("income_funding", lambda: fut.fut_income("FUNDING_FEE", limit=10))
        if ok:
            res["funding_income_seen"] = True if rows else None

    ready = readiness(res)
    out_dir = Path(out_dir) if out_dir is not None else F0_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("F0_%s.json" % now.strftime("%Y%m%dT%H%M%SZ"))
    doc = {"utc": now.isoformat(timespec="seconds"), "spec": "fc-passive-next-1022-spec.md", "results": res,
           "readiness": ready, "steps": steps}
    path.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    return {"status": ready["status"], "results": res, "readiness": ready, "steps": steps, "file": str(path)}
