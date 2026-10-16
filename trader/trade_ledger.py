"""Trade ledger of the demo account from exchange fills (demo-ledger-d-1013-spec.md §3).

The exchange is the source of truth: GET /api/v3/myTrades per symbol that moved since the last sync.
One line per fill in trader/runtime/trading/ledger.jsonl, append-only, idempotent (symbol + trade id).
Commission is converted to USDT (trade price for the base asset, a mid price for others such as BNB)
and the price source is written. Fills are linked to the decision that caused them via
orderId -> clientOrderId (audit exchange journal) -> decision_id (audit decisions journal).

  (from trader/) ../pyembed/python.exe trade_ledger.py sync
"""
from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402

LEDGER = TC.RUNTIME / "trading" / "ledger.jsonl"
STATE = TC.RUNTIME / "trading" / "ledger_state.json"
QUOTE = "USDT"
LOOKBACK_DAYS = 7          # daily sync; an OCO leg lives at most 96 x 15m bars
STABLES = frozenset({"USDT", "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDE", "XUSD", "BFUSD", "PYUSD", "RLUSD", "USD1"})


def symbols_to_sync(audit_rows, open_orders, balances, quote: str = QUOTE) -> list:
    syms = set()
    for r in audit_rows or []:
        if str(r.get("path", "")).startswith("/api/v3/order") and (r.get("params") or {}).get("symbol"):
            syms.add(r["params"]["symbol"])
    for o in open_orders or []:
        if o.get("symbol"):
            syms.add(o["symbol"])
    for b in balances or []:
        a = b.get("asset")
        if a and a not in STABLES and float(b.get("free") or 0) + float(b.get("locked") or 0) > 0:
            syms.add(a + quote)
    return sorted(syms)


def order_client_map(audit_rows) -> dict:
    """orderId -> clientOrderId from the exchange journal: order responses are kept in `body`
    (an OCO response lists its legs in `orderReports`, a cancelReplace response its two orders in
    `cancelResponse` / `newOrderResponse`)."""
    out = {}
    for r in audit_rows or []:
        body = r.get("body") if isinstance(r.get("body"), dict) else {}
        legs = [body.get("cancelResponse"), body.get("newOrderResponse")] + list(body.get("orderReports") or [])
        for o in [r, body] + legs:
            if isinstance(o, dict) and o.get("orderId") is not None and o.get("clientOrderId"):
                out[int(o["orderId"])] = o["clientOrderId"]
    return out


def decision_map(decision_rows) -> dict:
    out = {}
    for r in decision_rows or []:
        for cid in r.get("client_order_ids") or []:
            out[cid] = r.get("decision_id")
    return out


def sync(client, symbols, state: dict, ledger_path: Path, order_map: dict, decision_map_: dict, price_at) -> int:
    """Append new fills of `symbols`; returns how many were added. state: {"from_id": {sym: next id}}."""
    state.setdefault("from_id", {})
    added = 0
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(ledger_path, "a", encoding="utf-8") as fh:
        for sym in symbols:
            start = state["from_id"].get(sym)
            rows = client.my_trades(sym, from_id=start, limit=1000)
            base = sym[: -len(QUOTE)] if sym.endswith(QUOTE) else sym
            for t in sorted(rows, key=lambda x: int(x["id"])):
                tid = int(t["id"])
                if start is not None and tid < start:
                    continue
                price, qty, comm = float(t["price"]), float(t["qty"]), float(t["commission"])
                asset = t.get("commissionAsset")
                tdt = datetime.fromtimestamp(int(t["time"]) / 1000, timezone.utc)
                if asset == QUOTE:
                    c_usdt, src = comm, "quote"
                elif asset == base:
                    c_usdt, src = comm * price, "trade_price"
                else:
                    px, src = price_at(asset, tdt)
                    c_usdt = comm * px if px is not None else None
                oid = int(t.get("orderId")) if t.get("orderId") is not None else None
                cid = order_map.get(oid)
                rec = {"trade_id": tid, "order_id": oid, "client_order_id": cid, "decision_id": decision_map_.get(cid),
                       "symbol": sym, "side": "BUY" if t.get("isBuyer") else "SELL", "price": price, "qty": qty,
                       "quote_qty": float(t.get("quoteQty") or price * qty), "commission": comm,
                       "commission_asset": asset, "commission_usdt": c_usdt, "commission_price_source": src,
                       "time": tdt.isoformat(), "is_maker": bool(t.get("isMaker"))}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                added += 1
                state["from_id"][sym] = tid + 1
    return added


def load(path: Path | None = None) -> list:
    path = path or LEDGER
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import faulthandler
    faulthandler.dump_traceback_later(300, exit=True)      # watchdog
    import audit as A
    import audit_report as AR
    import exchange_client as X
    A.set_process("ledger")
    c = X.ExchangeClient()
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    ex_rows = [r for r in AR.load("exchange", LOOKBACK_DAYS) if str(r.get("path", "")).startswith("/api/v3/order")]
    acct = c.account()
    syms = sorted(set(symbols_to_sync(ex_rows, c.open_orders(), acct.get("balances", []))) | set(st.get("from_id", {})))

    def price_at(asset, t):
        try:
            k = c.klines(asset + QUOTE, "1m", 1, base=X.DEMO_REST)
            return float(k[-1][4]), "kline_1m_now"
        except Exception:
            return None, "unavailable"
    n = sync(c, syms, st, LEDGER, order_client_map(ex_rows), decision_map(AR.load("decisions", LOOKBACK_DAYS)), price_at)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st), encoding="utf-8")
    tmp.replace(STATE)
    print(json.dumps({"symbols": syms, "added": n}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
