"""Reconciliation of the executor with the demo account (demo-execution-c-1013-spec.md §5.2).

run() only READS the exchange and returns actions; the executor applies them and writes the alerts.
Sources compared: balances + open orders of the account, the executor's positions, and the day
status of the phase-D metric (RECONCILE_FAIL blocks new entries).

Actions:
  place_stop        open position, coin on the account, no live stop     -> stop from the policy stop
  orphan_balance    coin on the account (not dust) without a position    -> stop 11% below the price
  cancel_order      our order (clientOrderId cb_...) without a position  -> cancel
  alert             a foreign order (placed by hand) without a position  -> alert only, never touched
  close_from_trades position in the state, nothing on the account        -> close from myTrades
  block_entries     the metric's day status is RECONCILE_FAIL
"""
from __future__ import annotations

from decimal import ROUND_DOWN, Decimal

QUOTE = "USDT"
STABLES = frozenset({"USDT", "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDE", "XUSD", "BFUSD", "PYUSD",
                     "RLUSD", "USD1"})
ORPHAN_STOP_PCT = 11.0           # 8% floor of X4 + 3% catastrophe gap, applied to the current price
OUR_PREFIX = "cb_"


def _floor(x, step) -> float:
    s = Decimal(str(step))
    if s <= 0:
        return float(x)
    return float((Decimal(str(x)) / s).to_integral_value(rounding=ROUND_DOWN) * s)


def tick(filters: dict) -> str:
    return filters.get("PRICE_FILTER", {}).get("tickSize", "0")


def step(filters: dict) -> str:
    return filters.get("LOT_SIZE", {}).get("stepSize", "0")


def min_notional(filters: dict) -> float:
    for k in ("NOTIONAL", "MIN_NOTIONAL"):
        if k in filters:
            return float(filters[k].get("minNotional") or 0)
    return 0.0


def catastrophe_stop(policy_stop: float, gap_pct: float, filters: dict) -> float:
    """policy stop x (1 - gap), in Decimal so 1.84 x 0.97 stays 1.7848, rounded down to the tick."""
    v = Decimal(str(policy_stop)) * (1 - Decimal(str(gap_pct)) / 100)
    return _floor(v, tick(filters))


def is_dust(orders, sym: str, qty: float) -> bool:
    if qty <= 0:
        return True
    try:
        px = float(orders.price(sym))
    except Exception:
        return False                       # unknown price: not provably dust, keep it visible
    return qty * px < min_notional(orders.filters(sym))


def run(positions: dict, orders, policy_stop, cfg, day_status: str | None = None) -> list:
    acts = []
    if day_status == "RECONCILE_FAIL":
        acts.append({"action": "block_entries"})
    open_pos = {k: p for k, p in positions.items() if p.get("state") != "CLOSED"}
    held = {}
    for b in orders.balances():
        q = float(b.get("free") or 0) + float(b.get("locked") or 0)
        if q > 0:
            held[b["asset"]] = q
    oo = orders.open_orders()
    live_ids = {o["orderId"] for o in oo}
    pos_stop_ids = {p.get("stop_order_id") for p in open_pos.values() if p.get("stop_order_id") is not None}
    pos_syms = {p["sym"] for p in open_pos.values()}

    for k, p in sorted(open_pos.items()):
        sym = p["sym"]
        base = sym[: -len(QUOTE)]
        q = held.get(base, 0.0)
        live = p.get("stop_order_id") in live_ids
        if live:
            continue
        if is_dust(orders, sym, q):
            acts.append({"action": "close_from_trades", "key": k, "sym": sym})
            continue
        ps = policy_stop(k)
        f = orders.filters(sym)
        if ps is None:
            ps_stop = _floor(float(orders.price(sym)) * (1 - ORPHAN_STOP_PCT / 100), tick(f))
        else:
            ps_stop = catastrophe_stop(ps, cfg.TRADING_CATASTROPHE_GAP_PCT, f)
        acts.append({"action": "place_stop", "key": k, "sym": sym, "stop": ps_stop, "qty": _floor(q, step(f))})

    for o in oo:
        if o["orderId"] in pos_stop_ids:
            continue
        cid = str(o.get("clientOrderId") or "")
        if o.get("symbol") in pos_syms and not cid.startswith(OUR_PREFIX):
            acts.append({"action": "alert", "code": "foreign_order", "sym": o.get("symbol"), "order_id": o["orderId"]})
        elif cid.startswith(OUR_PREFIX):
            acts.append({"action": "cancel_order", "sym": o["symbol"], "order_id": o["orderId"], "client_order_id": cid})
        else:
            acts.append({"action": "alert", "code": "foreign_order", "sym": o.get("symbol"), "order_id": o["orderId"]})

    for asset, q in sorted(held.items()):
        if asset in STABLES or asset == "BNB":
            continue
        sym = asset + QUOTE
        if sym in pos_syms:
            continue
        try:
            px = float(orders.price(sym))
            f = orders.filters(sym)
        except Exception:
            continue
        if q * px < min_notional(f):
            continue
        acts.append({"action": "orphan_balance", "sym": sym, "qty": _floor(q, step(f)),
                     "stop": _floor(px * (1 - ORPHAN_STOP_PCT / 100), tick(f))})
    return acts
