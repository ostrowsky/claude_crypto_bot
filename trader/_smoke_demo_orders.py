"""C0 -- smoke of real orders on the DEMO account, one pair, ~10 USDT (demo-execution-c-1013-spec.md §6).

Checks on the demo exchange what a fake exchange cannot:
  buy (MARKET quoteOrderQty) -> STOP_LOSS on the free balance -> order/cancelReplace -> cancel ->
  OCO (LIMIT_MAKER above, STOP_LOSS below) -> cancel the OCO -> STOP_LOSS with trailingDelta ->
  cancel -> MARKET SELL of the free balance -> phase-D ledger sync -> zero open orders.
A failed step stops the run and the cleanup cancels every open order of the pair and sells the coin
back. Report: trader/runtime/trading/smoke_orders_<ts>.json. Every request is in the audit journal.

It places orders, so the operator runs it, explicitly:
  (from trader/) ../pyembed/python.exe _smoke_demo_orders.py --go [--symbol BTCUSDT] [--quote 10]
Demo hosts only (exchange_client refuses anything else); it does not depend on TRADING_ENABLED.
"""
from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402
import exchange_client as X       # noqa: E402
import reconcile as RC            # noqa: E402

OUT_DIR = TC.RUNTIME / "trading"
TRAILING_BIPS = 800               # 8%, inside TRAILING_DELTA 10..2000 on all 105 pairs (checked 2026-10-13)


def _mid(client, symbol):
    """One pair per request: bookTicker over all pairs took 4.6-4.8 s on demo (C0 2026-10-13)."""
    r = client.book_ticker(symbol)
    return (float(r["bidPrice"]) + float(r["askPrice"])) / 2


def _free(client, asset):
    for b in client.account().get("balances", []):
        if b.get("asset") == asset:
            return float(b.get("free") or 0)
    return 0.0


def _s(x) -> str:
    return X.Orders._s(x)


def cleanup(client, xinfo, symbol) -> dict:
    """Cancel every open order of the pair (OCO as a list), then sell the free coin back."""
    out = {"cancelled": [], "sold": None, "ok": True}
    f = X.symbol_filters(xinfo, symbol)
    try:
        lists_done = set()
        for o in client.open_orders(symbol):
            lid = o.get("orderListId", -1)
            if lid is not None and lid != -1:
                if lid not in lists_done:
                    client.cancel_order_list(symbol, lid)
                    lists_done.add(lid)
                    out["cancelled"].append({"orderListId": lid})
                continue
            client.cancel_order(symbol, order_id=o["orderId"])
            out["cancelled"].append({"orderId": o["orderId"]})
        base = symbol[:-4]
        q = RC._floor(_free(client, base), RC.step(f))
        if q > 0 and q * _mid(client, symbol) >= RC.min_notional(f):
            client.place_order(symbol, "SELL", "MARKET", quantity=_s(q), newClientOrderId="cb_smoke_clean_x")
            out["sold"] = q
    except Exception as e:
        out.update(ok=False, error=repr(e)[:300])
    return out


def run_smoke(client, xinfo, symbol, quote: float = 10.0, ledger_sync=None) -> dict:
    f = X.symbol_filters(xinfo, symbol)
    base, ts = symbol[:-4], datetime.now(timezone.utc).strftime("%H%M%S")
    rep = {"symbol": symbol, "quote": quote, "steps": [], "ok": True, "failed_step": None}
    st = {}

    def step(name, fn):
        try:
            r = fn()
            rep["steps"].append({"step": name, "ok": True, "result": r})
            return r
        except Exception as e:
            rep["steps"].append({"step": name, "ok": False, "error": repr(e)[:300],
                                 "code": getattr(e, "code", None), "data": getattr(e, "data", None)})
            raise

    def stop_px(k):
        return _s(RC._floor(_mid(client, symbol) * k, RC.tick(f)))

    def qty():
        return _s(RC._floor(_free(client, base), RC.step(f)))

    try:
        st["buy"] = step("buy", lambda: client.place_order(symbol, "BUY", "MARKET", quoteOrderQty=_s(quote),
                                                           newClientOrderId=f"cb_smoke_{ts}_b"))
        st["stop"] = step("stop", lambda: client.place_order(symbol, "SELL", "STOP_LOSS", quantity=qty(),
                                                             stopPrice=stop_px(0.95), newClientOrderId=f"cb_smoke_{ts}_s1"))
        st["repl"] = step("replace", lambda: client.cancel_replace(
            symbol, st["stop"]["orderId"], "STOP_LOSS", quantity=_locked_qty(client, base, f),
            stopPrice=stop_px(0.96), newClientOrderId=f"cb_smoke_{ts}_s2"))
        new_id = (st["repl"].get("newOrderResponse") or {}).get("orderId")
        step("cancel_stop", lambda: client.cancel_order(symbol, order_id=new_id))
        st["oco"] = step("oco", lambda: client.place_oco(
            symbol, "SELL", quantity=qty(), aboveType="LIMIT_MAKER", abovePrice=stop_px(1.05),
            belowType="STOP_LOSS", belowStopPrice=stop_px(0.95), listClientOrderId=f"cb_smoke_{ts}_oco"))
        step("cancel_oco", lambda: client.cancel_order_list(symbol, st["oco"]["orderListId"]))
        st["trail"] = step("trailing_stop", lambda: client.place_order(
            symbol, "SELL", "STOP_LOSS", quantity=qty(), trailingDelta=TRAILING_BIPS, newClientOrderId=f"cb_smoke_{ts}_t"))
        step("cancel_trailing", lambda: client.cancel_order(symbol, order_id=st["trail"]["orderId"]))
        step("sell", lambda: client.place_order(symbol, "SELL", "MARKET", quantity=qty(), newClientOrderId=f"cb_smoke_{ts}_x"))
        step("ledger_sync", lambda: {"added": ledger_sync() if ledger_sync else None})

        def no_open():
            left = client.open_orders(symbol)
            if left:
                raise RuntimeError(f"{len(left)} open orders left")
            return {"open_orders": 0, "dust_qty": _free(client, base)}
        step("no_open_orders", no_open)
    except Exception:
        rep["ok"] = False
        rep["failed_step"] = rep["steps"][-1]["step"] if rep["steps"] else None
        rep["cleanup"] = cleanup(client, xinfo, symbol)
    return rep


def _locked_qty(client, base, f):
    """The stop locks the coin, so the replacement uses the locked quantity."""
    for b in client.account().get("balances", []):
        if b.get("asset") == base:
            return _s(RC._floor(float(b.get("locked") or 0) + float(b.get("free") or 0), RC.step(f)))
    return "0"


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if "--go" not in argv:
        print("C0 places real orders on the DEMO account. Run with --go to confirm.")
        return 2
    symbol = argv[argv.index("--symbol") + 1] if "--symbol" in argv else "BTCUSDT"
    quote = float(argv[argv.index("--quote") + 1]) if "--quote" in argv else 10.0
    import audit as A
    import microstructure_recorder as M
    import trade_ledger as L
    A.set_process("smoke_orders")
    client = X.ExchangeClient(cfg=types.SimpleNamespace(TRADING_ENABLED=True, TRADING_MODE="demo"))
    xinfo = M.cached_exchange_info(client)
    def sync():
        import faulthandler
        before = len(L.load())
        try:
            L.main([])                             # returns an exit code, not the count
            return len(L.load()) - before
        finally:
            faulthandler.cancel_dump_traceback_later()   # trade_ledger.main arms a 300 s watchdog
            A.set_process("smoke_orders")
    rep = run_smoke(client, xinfo, symbol, quote, ledger_sync=sync)
    rep["ts"] = datetime.now(timezone.utc).isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"smoke_orders_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for s in rep["steps"]:
        print(f"{s['step']:16} {'OK' if s['ok'] else 'FAIL ' + str(s.get('error'))}")
    print(("PASS" if rep["ok"] else "FAIL at " + str(rep["failed_step"])) + f" -> {out}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
