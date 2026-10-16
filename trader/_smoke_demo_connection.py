"""Gate-A smoke test of the Binance DEMO connection -- read-only + order/test, no order is placed.

demo-connection-a-1010-spec.md, gate A item 2. Prints results only: never the key,
the secret or a signature. Steps:
  1 host guard        a signed request to api.binance.com must be refused locally
  2 time              server time and clock offset
  3 exchangeInfo      pairs, USDT pairs, rate limits
  4 account           canTrade, permissions, balances (demo money)
  5 commission        real maker/taker rates on BTCUSDT
  6 open orders       count (expected 0)
  7 order/test        20 random universe pairs: MARKET BUY by quoteOrderQty and LIMIT BUY
                      at the minimum valid size, below the bid -- validated, never executed
  8 user data stream  WS API userDataStream.subscribe.signature -> subscribed, then closed
  9 demo vs live      one bookTicker snapshot: |mid divergence| distribution
Exit code 0 = every step OK. Writes trader/runtime/trading/smoke_<ts>.json and one equity snapshot.
Run from files/:  ../pyembed/python.exe _smoke_demo_connection.py
"""
from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import exchange_client as X  # noqa: E402
import trading_universe as U  # noqa: E402
import equity_snapshots as EQ  # noqa: E402

import trader_config as TC  # noqa: E402
OUT = TC.RUNTIME / "trading"


async def ws_subscribe(key: str, secret: str) -> dict:
    import aiohttp
    ts = int(time.time() * 1000)
    params = {"apiKey": key, "timestamp": ts}
    q = "&".join(f"{k}={params[k]}" for k in sorted(params))
    params["signature"] = X.sign(q, secret)
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect("wss://demo-ws-api.binance.com/ws-api/v3", timeout=10) as ws:
            await ws.send_json({"id": "smoke-sub", "method": "userDataStream.subscribe.signature", "params": params})
            msg = await asyncio.wait_for(ws.receive_json(), 10)
            out = {"status": msg.get("status"), "subscriptionId": (msg.get("result") or {}).get("subscriptionId"),
                   "error": msg.get("error")}
            if out["subscriptionId"] is not None:
                await ws.send_json({"id": "smoke-unsub", "method": "userDataStream.unsubscribe",
                                    "params": {"subscriptionId": out["subscriptionId"]}})
                try:
                    await asyncio.wait_for(ws.receive_json(), 5)
                except Exception:
                    pass
            return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import audit
    audit.set_process("smoke")
    res, ok = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}, True

    def step(name, fn):
        nonlocal ok
        try:
            v = fn()
            res[name] = {"ok": True, **(v if isinstance(v, dict) else {"value": v})}
            print(f"[OK]   {name}: {json.dumps(v, ensure_ascii=False, default=str)[:400]}")
            return v
        except Exception as e:
            ok = False
            res[name] = {"ok": False, "error": X.redact(repr(e))[:300]}
            print(f"[FAIL] {name}: {res[name]['error']}")
            return None

    key, secret = X.load_credentials()
    print(f"credentials: key {'present' if key else 'MISSING'} ({len(key)} chars), "
          f"secret {'present' if secret else 'MISSING'} ({len(secret)} chars)")

    def guard():
        try:
            X.ExchangeClient(base=X.LIVE_REST, key="k", secret="s")
        except X.HostNotAllowed:
            return {"mainnet_signed_refused": True}
        raise AssertionError("signed client accepted api.binance.com")
    step("1_host_guard", guard)
    c = X.ExchangeClient()
    step("2_time", lambda: {"server_ms": c.server_time(), "offset_ms": c._sync_time()})
    xi = step("3_exchange_info", lambda: (lambda x: {"symbols": len(x["symbols"]),
                                                      "usdt_trading": sum(1 for s in x["symbols"] if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"),
                                                      "rateLimits": x.get("rateLimits")})(c.exchange_info()))
    acct = step("4_account", lambda: (lambda a: {"canTrade": a.get("canTrade"), "permissions": a.get("permissions"),
                                                 "accountType": a.get("accountType"),
                                                 "balances_nonzero": len(a.get("balances", [])),
                                                 "USDT_free": next((b["free"] for b in a.get("balances", []) if b["asset"] == "USDT"), "0"),
                                                 "assets": [b["asset"] for b in a.get("balances", [])][:30]})(c.account()))
    step("5_commission_BTCUSDT", lambda: (lambda r: {"standard": r.get("standardCommission"), "tax": r.get("taxCommission"),
                                                     "discount": r.get("discount")})(c.commission("BTCUSDT")))
    step("6_open_orders", lambda: {"count": len(c.open_orders())})

    def tests():
        x = c.exchange_info()
        u = U.build(x, hist=lambda s: 10 ** 6)
        books = {r["symbol"]: r for r in c.book_tickers()}
        pool = [p["symbol"] for p in u["pairs"] if p["symbol"] in books and float(books[p["symbol"]]["bidPrice"] or 0) > 0]
        random.seed(10)
        picks = random.sample(pool, min(20, len(pool)))
        results, fails = {}, []
        for sym in picks:
            f = X.symbol_filters(x, sym)
            bid = Decimal(books[sym]["bidPrice"])
            r = {}
            try:
                mn = max(Decimal(f.get("NOTIONAL", f.get("MIN_NOTIONAL", {})).get("minNotional", "5")) * Decimal("1.2"), Decimal("12"))
                t = c.test_order(sym, "BUY", "MARKET", quoteOrderQty=str(mn.quantize(Decimal("0.01"))))
                r["market"] = "ok"
                r["commission"] = (t.get("standardCommissionForOrder") or {})
            except X.ExchangeError as e:
                r["market"] = f"{e.code} {e.msg}"
                fails.append(sym)
            try:
                price = X.round_price(f, bid * Decimal("0.99"))
                qty = X.min_order_qty(f, price)
                local = X.check_order(f, qty, price)
                c.test_order(sym, "BUY", "LIMIT", timeInForce="GTC", quantity=format(qty, "f"), price=format(price, "f"))
                r["limit"] = "ok" + (f" (local check said {local})" if local else "")
            except X.ExchangeError as e:
                r["limit"] = f"{e.code} {e.msg}"
                fails.append(sym)
            results[sym] = r
        if fails:
            raise AssertionError(f"order/test failed on {sorted(set(fails))}: " + json.dumps({s: results[s] for s in set(fails)})[:300])
        return {"pairs_tested": len(picks), "universe_structural": len(pool), "sample": dict(list(results.items())[:3])}
    step("7_order_test", tests)
    step("8_user_data_stream", lambda: (lambda r: r if r.get("status") == 200 else (_ for _ in ()).throw(AssertionError(r)))(
        asyncio.run(ws_subscribe(key, secret))))

    def divergence():
        live = {r["symbol"]: r for r in c.book_tickers(base=X.LIVE_REST)}
        demo = {r["symbol"]: r for r in c.book_tickers()}
        d = []
        for s, r in demo.items():
            if not s.endswith("USDT") or s not in live:
                continue
            a = (float(r["bidPrice"] or 0) + float(r["askPrice"] or 0)) / 2
            b = (float(live[s]["bidPrice"] or 0) + float(live[s]["askPrice"] or 0)) / 2
            if a > 0 and b > 0:
                d.append(abs(a / b - 1) * 1e4)
        d.sort()
        q = lambda p: round(d[min(len(d) - 1, int(p * len(d)))], 2) if d else None  # noqa: E731
        return {"pairs": len(d), "abs_div_bps_p50": q(.5), "p90": q(.9), "p99": q(.99), "max": round(d[-1], 1) if d else None}
    step("9_demo_vs_live", divergence)
    step("10_equity_snapshot", lambda: (lambda r: {k: r[k] for k in ("equity_usdt", "assets", "unpriced_share")})(EQ.snapshot(c)))

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"smoke_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(("\nGATE A SMOKE: ALL OK" if ok else "\nGATE A SMOKE: FAILED") + f"  -> {p}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
