"""Signed client for the Binance SPOT **demo** account -- phase A1 (read + dry-run only).

WHY (2026-10-10, demo-connection-a-1010-spec.md)
The bot gets an execution side for the first time. Before any order exists the
client must guarantee three things that no flag can switch off:

  1. signed requests go to demo hosts only (ALLOWED_HOSTS is a code constant, not
     config; mainnet means a new spec and an edit here, by operator decision);
  2. the key and the signature never reach a log line (redact());
  3. exchange filters come from exchangeInfo, never from constants.

Phase A methods: time, exchangeInfo, account, commission, openOrders, myTrades,
public market data, order/test. place_order / place_oco exist only as the
interface for phase C and refuse while TRADING_ENABLED is off or the mode is not
"demo".

Synchronous on purpose (urllib, no new dependency): phase A callers are
standalone scripts. The live executor of phase C runs it in a worker thread so
the bot's event loop never waits on the exchange (lesson of 2026-08-04).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
import trader_config as TC  # noqa: E402
import audit as A  # noqa: E402

# demo-fapi: USD-M futures of the demo account (carry-funding-money-1021-spec.md §10); still demo only
ALLOWED_HOSTS = frozenset({"demo-api.binance.com", "demo-ws-api.binance.com", "demo-stream.binance.com",
                           "demo-fapi.binance.com"})
# unsigned market data only; fapi / data.binance.vision = live futures history for the FC money test (spec §3)
PUBLIC_HOSTS = frozenset({"api.binance.com", "fapi.binance.com", "data.binance.vision"}) | ALLOWED_HOSTS
DEMO_REST = "https://demo-api.binance.com"
LIVE_REST = "https://api.binance.com"
LIVE_FAPI = "https://fapi.binance.com"
DEMO_FAPI = "https://demo-fapi.binance.com"
KEY_VAR, SECRET_VAR = "BINANCE_DEMO_API_KEY", "BINANCE_DEMO_API_SECRET"
RECV_WINDOW = 15000          # ms; slow links (2-3 s per request seen 2026-10-10) broke 5000
TS_LAG_MS = 500              # stamp slightly behind the server: Binance rejects > serverTime + 1000
TIME_RESYNC_SEC = 600
TIMEOUT_SEC = 10             # per socket operation (silence)
REQUEST_DEADLINE_SEC = 30    # whole request incl. body: a slow trickle once kept exchangeInfo alive > 10 min


class ExchangeError(RuntimeError):
    def __init__(self, status, code, msg, path, data=None):
        super().__init__(f"HTTP {status} code {code}: {msg} ({path})")
        self.status, self.code, self.msg, self.path = status, code, msg, path
        self.data = data          # "data" of an error body, e.g. a partial cancelReplace failure (-2021)


class RequestDeadline(RuntimeError):
    pass


class HostNotAllowed(RuntimeError):
    pass


class TradingDisabled(RuntimeError):
    pass


def redact(text: str, secrets=()) -> str:
    """Remove the key, the secret and any signature from a string before it is logged."""
    out = str(text)
    for s in secrets:
        if s:
            out = out.replace(s, "<redacted>")
    if "signature=" in out:
        head, _, tail = out.partition("signature=")
        rest = tail.split("&", 1)
        out = head + "signature=<redacted>" + ("&" + rest[1] if len(rest) > 1 else "")
    return out


def load_credentials(env_path: Path | None = None):
    """(key, secret) from files/.env -- only the DEMO-named variables are read."""
    try:
        from dotenv import dotenv_values
        vals = dotenv_values(env_path or TC.ENV_FILE)
    except Exception:
        vals = {}
    norm = {str(k).strip(): (str(v).strip().strip('"').strip("'") if v is not None else "") for k, v in vals.items()}
    key = norm.get(KEY_VAR) or os.environ.get(KEY_VAR, "")
    sec = norm.get(SECRET_VAR) or os.environ.get(SECRET_VAR, "")
    return key, sec


def sign(query: str, secret: str) -> str:
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


def _host(url: str) -> str:
    return (urllib.parse.urlparse(url).hostname or "").lower()


class ExchangeClient:
    def __init__(self, base: str = DEMO_REST, key: str | None = None, secret: str | None = None, cfg=None,
                 opener=None, audit=None):
        if _host(base) not in ALLOWED_HOSTS:
            raise HostNotAllowed(f"signed client refuses host {_host(base)!r}; allowed: {sorted(ALLOWED_HOSTS)}")
        if key is None or secret is None:
            k, s = load_credentials()
            key, secret = key or k, secret or s
        self.base, self._key, self._secret = base.rstrip("/"), key or "", secret or ""
        self.cfg = cfg
        self._open = opener or urllib.request.urlopen
        self._offset_ms, self._synced_at = 0, 0.0
        self.used_weight = {}
        self._xinfo, self._xinfo_at = None, 0.0
        # one audit line per request (trader-audit-log-1012-spec.md); None = not logged
        self._audit = audit if audit is not None else A.default_writer("exchange")

    def _log(self, rec, t_start, host, **kw):
        """Never raises: a failed audit write must not break the request (spec §5)."""
        try:
            if self._audit is None:
                return
            body = kw.pop("body", None)
            line = dict(rec, latency_ms=round((time.time() - t_start) * 1000, 1),
                        used_weight_1m=self.used_weight.get(host), **kw)
            if body is not None and A.keep_body(rec["path"]):
                line["body"] = A.summarize_body(rec["path"], body)
                for k in ("clientOrderId", "orderId", "orderListId"):
                    if isinstance(body, dict) and k in body:
                        line[k] = body[k]
            line.setdefault("bytes", None)
            text = json.dumps(line, default=str)
            for sec in (self._key, self._secret):
                if sec and sec in text:
                    line = json.loads(text.replace(sec, "<redacted>"))
                    text = json.dumps(line, default=str)
            self._audit.write(line)
        except Exception:
            pass

    # ------------------------------------------------------------ transport
    def _request(self, method: str, path: str, params: dict | None = None, signed: bool = False,
                 base: str | None = None, _retried: bool = False):
        base = (base or self.base).rstrip("/")
        host = _host(base)
        if signed and host not in ALLOWED_HOSTS:
            raise HostNotAllowed(f"signed request to {host!r} refused")
        if not signed and host not in PUBLIC_HOSTS:
            raise HostNotAllowed(f"request to {host!r} refused")
        params = {k: v for k, v in (params or {}).items() if v is not None}
        headers = {}
        if signed:
            if not (self._key and self._secret):
                raise RuntimeError(f"{KEY_VAR} / {SECRET_VAR} missing in files/.env")
            self._maybe_resync()
            params["timestamp"] = int(time.time() * 1000) + self._offset_ms - TS_LAG_MS
            params.setdefault("recvWindow", RECV_WINDOW)
            q = urllib.parse.urlencode(params, doseq=True)
            q = q + "&signature=" + sign(q, self._secret)
            headers["X-MBX-APIKEY"] = self._key
        else:
            q = urllib.parse.urlencode(params, doseq=True)
        url = base + path + ("?" + q if q and method in ("GET", "DELETE") else "")
        data = q.encode() if method in ("POST", "PUT") and q else None
        if data is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        t_start = time.time()
        rec = {"host": host, "method": method, "path": path, "params": A.redact_params(params), "signed": bool(signed),
               "retry": bool(_retried)}
        try:
            with self._open(req, timeout=TIMEOUT_SEC) as resp:
                self._track(resp.headers, host)
                parts = []
                reader = getattr(resp, "read1", None) or resp.read
                while True:
                    if time.time() - t_start > REQUEST_DEADLINE_SEC:
                        raise RequestDeadline(f"{method} {path}: over {REQUEST_DEADLINE_SEC} s")
                    chunk = reader(65536)
                    if not chunk:
                        break
                    parts.append(chunk)
                raw = b"".join(parts)
                out = json.loads(raw.decode("utf-8")) if raw else {}
                self._log(rec, t_start, host, status=getattr(resp, "status", 200), ok=True, bytes=len(raw), body=out)
                return out
        except urllib.error.HTTPError as e:
            self._track(e.headers or {}, host)
            try:
                j = json.loads(e.read().decode("utf-8"))
            except Exception:
                j = {}
            code, msg, edata = j.get("code"), j.get("msg", ""), j.get("data")
            self._log(rec, t_start, host, status=e.code, ok=False, error_code=code,
                      error_msg=redact(msg, (self._key, self._secret)),
                      **({"error_data": edata} if edata is not None else {}))
            if code == -1021 and signed and not _retried:   # timestamp outside recvWindow: resync, retry once
                self._sync_time(force=True)
                return self._request(method, path, {k: v for k, v in params.items()
                                                    if k not in ("timestamp", "recvWindow")}, signed, base, True)
            raise ExchangeError(e.code, code, redact(msg, (self._key, self._secret)), path, edata) from None
        except RequestDeadline:
            self._log(rec, t_start, host, status=None, ok=False, error_code="deadline")
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            self._log(rec, t_start, host, status=None, ok=False, error_code="network",
                      error_msg=redact(repr(e), (self._key, self._secret)))
            # timeouts, SSL handshake failures, resets: one hiccup must not kill a caller's whole loop
            # (2026-10-11/12: three recorder snapshots lost to an uncaught TimeoutError)
            raise ExchangeError(None, "network", redact(repr(e), (self._key, self._secret)), path) from None

    def _track(self, headers, host: str = ""):
        """used_weight[host] = the 1-minute request weight the exchange reports for this IP on that host.
        It includes every other process on the same IP (the live bot polls api.binance.com)."""
        try:
            for k, v in dict(headers).items():
                if str(k).lower() == "x-mbx-used-weight-1m":
                    self.used_weight[host] = int(v)
        except Exception:
            pass

    def _sync_time(self, force: bool = False):
        t0 = time.time()
        # a futures host has no /api/v3/time (fc-passive-next-1022-spec.md §B.2)
        path = "/fapi/v1/time" if "fapi" in _host(self.base) else "/api/v3/time"
        srv = self._request("GET", path)["serverTime"]
        t1 = time.time()
        self._offset_ms = int(srv - (t0 + t1) / 2 * 1000)
        self._synced_at = t1
        return self._offset_ms

    def _maybe_resync(self):
        if time.time() - self._synced_at > TIME_RESYNC_SEC:
            self._sync_time()

    # ------------------------------------------------------------ public
    def server_time(self) -> int:
        return self._request("GET", "/api/v3/time")["serverTime"]

    def exchange_info(self, max_age_sec: int = 86400) -> dict:
        if self._xinfo is None or time.time() - self._xinfo_at > max_age_sec:
            self._xinfo, self._xinfo_at = self._request("GET", "/api/v3/exchangeInfo"), time.time()
        return self._xinfo

    def tickers_24h(self, base: str | None = None) -> list:
        return self._request("GET", "/api/v3/ticker/24hr", base=base)

    def book_tickers(self, base: str | None = None, symbols: list | None = None) -> list:
        """All pairs (slow on demo: 4.6-4.8 s, 2026-10-13), or one request for `symbols`."""
        p = {"symbols": json.dumps(list(symbols), separators=(",", ":"))} if symbols else None
        return self._request("GET", "/api/v3/ticker/bookTicker", p, base=base)

    def book_ticker(self, symbol: str, base: str | None = None) -> dict:
        return self._request("GET", "/api/v3/ticker/bookTicker", {"symbol": symbol}, base=base)

    def klines(self, symbol: str, interval: str = "15m", limit: int = 500, base: str | None = None) -> list:
        return self._request("GET", "/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit}, base=base)

    def klines_from(self, symbol: str, interval: str, start_ms: int, limit: int = 1000, base: str | None = None) -> list:
        """Klines from startTime (history: entry-price-gap-1013-spec.md)."""
        return self._request("GET", "/api/v3/klines", {"symbol": symbol, "interval": interval, "startTime": int(start_ms),
                                                       "limit": limit}, base=base)

    # ------------------------------------------------------------ USD-M futures, public (FC spec §3)
    def futures_exchange_info(self, base: str = LIVE_FAPI) -> dict:
        return self._request("GET", "/fapi/v1/exchangeInfo", base=base)

    def funding_rates(self, symbol: str, start_ms: int, limit: int = 1000, base: str = LIVE_FAPI) -> list:
        """Funding events from startTime (without startTime the API returns only the latest ones)."""
        return self._request("GET", "/fapi/v1/fundingRate", {"symbol": symbol, "startTime": int(start_ms), "limit": limit},
                             base=base)

    def futures_klines_from(self, symbol: str, interval: str, start_ms: int, limit: int = 1000,
                            base: str = LIVE_FAPI) -> list:
        return self._request("GET", "/fapi/v1/klines", {"symbol": symbol, "interval": interval,
                                                        "startTime": int(start_ms), "limit": limit}, base=base)

    def premium_index(self, symbol: str, base: str = LIVE_FAPI) -> dict:
        return self._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol}, base=base)

    def depth(self, symbol: str, limit: int = 20, base: str | None = None) -> dict:
        return self._request("GET", "/api/v3/depth", {"symbol": symbol, "limit": limit}, base=base)

    # ------------------------------------------------------------ USD-M futures of the demo account, F0
    # fc-passive-next-1022-spec.md §B: settings and checks only. There is deliberately no method for a
    # real futures order; the only order call is POST /fapi/v1/order/test (validated, never executed).
    def _fut(self):
        if "fapi" not in _host(self.base):
            raise HostNotAllowed(f"futures call on a non-futures client ({_host(self.base)!r})")

    def fut_time(self) -> dict:
        self._fut()
        return self._request("GET", "/fapi/v1/time")

    def fut_account(self) -> dict:
        self._fut()
        return self._request("GET", "/fapi/v2/account", signed=True)

    def fut_balance(self) -> list:
        self._fut()
        return self._request("GET", "/fapi/v2/balance", signed=True)

    def fut_position_side(self) -> dict:
        self._fut()
        return self._request("GET", "/fapi/v1/positionSide/dual", signed=True)

    def fut_set_position_side(self, dual: bool) -> dict:
        self._fut()
        return self._request("POST", "/fapi/v1/positionSide/dual", {"dualSidePosition": "true" if dual else "false"},
                             signed=True)

    def fut_margin_type(self, symbol: str, margin_type: str) -> dict:
        self._fut()
        return self._request("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": margin_type}, signed=True)

    def fut_leverage(self, symbol: str, leverage: int) -> dict:
        self._fut()
        return self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": int(leverage)}, signed=True)

    def fut_commission(self, symbol: str) -> dict:
        self._fut()
        return self._request("GET", "/fapi/v1/commissionRate", {"symbol": symbol}, signed=True)

    def fut_test_order(self, symbol: str, side: str, type_: str, **kw) -> dict:
        """POST /fapi/v1/order/test -- validated by the exchange, never executed."""
        self._fut()
        return self._request("POST", "/fapi/v1/order/test", {"symbol": symbol, "side": side, "type": type_, **kw},
                             signed=True)

    def fut_income(self, income_type: str, limit: int = 10) -> list:
        self._fut()
        return self._request("GET", "/fapi/v1/income", {"incomeType": income_type, "limit": limit}, signed=True)

    # ------------------------------------------------------------ signed, read
    def account(self) -> dict:
        return self._request("GET", "/api/v3/account", {"omitZeroBalances": "true"}, signed=True)

    def commission(self, symbol: str) -> dict:
        return self._request("GET", "/api/v3/account/commission", {"symbol": symbol}, signed=True)

    def open_orders(self, symbol: str | None = None) -> list:
        return self._request("GET", "/api/v3/openOrders", {"symbol": symbol}, signed=True)

    def my_trades(self, symbol: str, from_id: int | None = None, limit: int = 500) -> list:
        return self._request("GET", "/api/v3/myTrades", {"symbol": symbol, "fromId": from_id, "limit": limit},
                             signed=True)

    # ------------------------------------------------------------ dry run
    def test_order(self, symbol: str, side: str, type_: str, **kw) -> dict:
        """POST /api/v3/order/test -- validated by the exchange, never executed."""
        p = {"symbol": symbol, "side": side, "type": type_, "computeCommissionRates": "true", **kw}
        return self._request("POST", "/api/v3/order/test", p, signed=True)

    # ------------------------------------------------------------ phase C interface
    def _trading_allowed(self):
        cfg = self.cfg
        if cfg is None:
            cfg = TC
        if not getattr(cfg, "TRADING_ENABLED", False) or getattr(cfg, "TRADING_MODE", "paper") != "demo":
            raise TradingDisabled("TRADING_ENABLED is off or TRADING_MODE != 'demo' (phase C)")

    def place_order(self, symbol: str, side: str | None = None, type_: str | None = None, **kw) -> dict:
        """POST /api/v3/order (demo-execution-c-1013-spec.md §3). Full response with fills."""
        self._trading_allowed()
        p = {"symbol": symbol, "side": side, "type": type_, "newOrderRespType": "FULL", **kw}
        return self._request("POST", "/api/v3/order", p, signed=True)

    def place_oco(self, symbol: str, side: str | None = None, **kw) -> dict:
        """POST /api/v3/orderList/oco: aboveType/belowType legs, for policies with a TP."""
        self._trading_allowed()
        return self._request("POST", "/api/v3/orderList/oco", {"symbol": symbol, "side": side, **kw}, signed=True)

    def cancel_order(self, symbol: str, order_id: int | None = None, orig_client_order_id: str | None = None) -> dict:
        self._trading_allowed()
        return self._request("DELETE", "/api/v3/order", {"symbol": symbol, "orderId": order_id,
                                                         "origClientOrderId": orig_client_order_id}, signed=True)

    def cancel_order_list(self, symbol: str, order_list_id: int) -> dict:
        """DELETE /api/v3/orderList: cancel a whole OCO."""
        self._trading_allowed()
        return self._request("DELETE", "/api/v3/orderList", {"symbol": symbol, "orderListId": order_list_id}, signed=True)

    def cancel_replace(self, symbol: str, cancel_order_id: int, type_: str, side: str = "SELL", **kw) -> dict:
        """Cancel one order and place its replacement in ONE request (STOP_ON_FAILURE)."""
        self._trading_allowed()
        p = {"symbol": symbol, "side": side, "type": type_, "cancelReplaceMode": "STOP_ON_FAILURE",
             "cancelOrderId": cancel_order_id, "newOrderRespType": "FULL", **kw}
        return self._request("POST", "/api/v3/order/cancelReplace", p, signed=True)

    def get_order(self, symbol: str, order_id: int | None = None, orig_client_order_id: str | None = None) -> dict:
        return self._request("GET", "/api/v3/order", {"symbol": symbol, "orderId": order_id,
                                                      "origClientOrderId": orig_client_order_id}, signed=True)


# ---------------------------------------------------------------- filters

def symbol_filters(xinfo: dict, symbol: str) -> dict:
    for s in xinfo.get("symbols", []):
        if s.get("symbol") == symbol:
            return {f["filterType"]: f for f in s.get("filters", [])}
    raise KeyError(symbol)


def _floor_step(value, step) -> Decimal:
    v, st = Decimal(str(value)), Decimal(str(step))
    if st <= 0:
        return v
    return (v / st).to_integral_value(rounding=ROUND_DOWN) * st


def round_qty(filters: dict, qty, market: bool = False) -> Decimal:
    f = filters.get("MARKET_LOT_SIZE") if market and filters.get("MARKET_LOT_SIZE", {}).get("stepSize", "0") not in ("0", "0.00000000") \
        else filters.get("LOT_SIZE", {})
    return _floor_step(qty, f.get("stepSize", "0")).normalize()


def round_price(filters: dict, price) -> Decimal:
    return _floor_step(price, filters.get("PRICE_FILTER", {}).get("tickSize", "0")).normalize()


def check_order(filters: dict, qty, price, market: bool = False) -> list:
    """Names of the filters an order of `qty` at `price` would violate (empty = passes)."""
    bad = []
    q, p = Decimal(str(qty)), Decimal(str(price))
    lot = filters.get("MARKET_LOT_SIZE") if market and "MARKET_LOT_SIZE" in filters else filters.get("LOT_SIZE")
    if lot:
        if q < Decimal(lot["minQty"]) or (Decimal(lot["maxQty"]) > 0 and q > Decimal(lot["maxQty"])):
            bad.append(("MARKET_" if lot is filters.get("MARKET_LOT_SIZE") else "") + "LOT_SIZE")
        elif Decimal(lot.get("stepSize", "0")) > 0 and _floor_step(q, lot["stepSize"]) != q:
            bad.append("LOT_SIZE_STEP")
    pf = filters.get("PRICE_FILTER")
    if pf and not market and Decimal(pf.get("tickSize", "0")) > 0 and _floor_step(p, pf["tickSize"]) != p:
        bad.append("PRICE_FILTER_TICK")
    nf = filters.get("NOTIONAL")
    if nf:
        notional = q * p
        apply_min = (not market) or nf.get("applyMinToMarket", True)
        apply_max = (not market) or nf.get("applyMaxToMarket", False)
        if apply_min and notional < Decimal(nf.get("minNotional", "0")):
            bad.append("NOTIONAL_MIN")
        if apply_max and Decimal(nf.get("maxNotional", "0")) > 0 and notional > Decimal(nf["maxNotional"]):
            bad.append("NOTIONAL_MAX")
    mn = filters.get("MIN_NOTIONAL")
    if mn and (not market or mn.get("applyToMarket", True)) and q * p < Decimal(mn.get("minNotional", "0")):
        bad.append("MIN_NOTIONAL")
    return bad


def min_order_qty(filters: dict, price) -> Decimal:
    """Smallest valid quantity at `price` (lot step and minimum notional respected), with 10% headroom."""
    p = Decimal(str(price))
    lot = filters.get("LOT_SIZE", {})
    step = Decimal(lot.get("stepSize", "0"))
    need = Decimal(lot.get("minQty", "0"))
    for key in ("NOTIONAL", "MIN_NOTIONAL"):
        mn = Decimal(filters.get(key, {}).get("minNotional", "0"))
        if mn > 0 and p > 0:
            need = max(need, mn * Decimal("1.1") / p)
    if step > 0:
        q = _floor_step(need, step)
        if q < need:
            q += step
        return q.normalize()
    return need


# ---------------------------------------------------------------- orders protocol (phase C)

class Orders:
    """The orders protocol of demo-execution-c-1013-spec.md §2.3 on the demo account.

    Quantities and prices are rounded DOWN by the pair's filters here, so the executor never sends
    a value the exchange rejects for its step or tick."""

    def __init__(self, client, xinfo: dict):
        self.c, self.xinfo = client, xinfo
        self._sym = {s["symbol"]: s for s in xinfo.get("symbols", [])}
        self._px = {}             # sym -> (mid, time): one pair per request, cached PRICE_TTL_SEC

    def filters(self, sym: str) -> dict:
        return symbol_filters(self.xinfo, sym)

    def _qp(self, sym: str) -> int:
        return int(self._sym.get(sym, {}).get("quoteAssetPrecision", 8))

    @staticmethod
    def _s(d) -> str:
        return format(Decimal(str(d)).normalize(), "f")

    def balances(self) -> list:
        return list(self.c.account().get("balances", []))

    def free(self, asset: str) -> float:
        for b in self.balances():
            if b.get("asset") == asset:
                return float(b.get("free") or 0)
        return 0.0

    PRICE_TTL_SEC = 5.0

    @staticmethod
    def _mid(r):
        bid, ask = float(r.get("bidPrice") or 0), float(r.get("askPrice") or 0)
        return (bid + ask) / 2 if bid > 0 and ask > 0 else None

    def price(self, sym: str) -> float:
        hit = self._px.get(sym)
        if hit is None or time.time() - hit[1] > self.PRICE_TTL_SEC:
            m = self._mid(self.c.book_ticker(sym))
            if m is None:
                raise KeyError(f"{sym}: empty book")
            self._px[sym] = hit = (m, time.time())
        return hit[0]

    def equity(self) -> float:
        """USDT + every balance priced in ONE bookTicker request over the pairs that exist."""
        bal = self.balances()
        held = {}
        for b in bal:
            q = float(b.get("free") or 0) + float(b.get("locked") or 0)
            if q > 0:
                held[b["asset"]] = held.get(b["asset"], 0.0) + q
        v = held.pop("USDT", 0.0)
        pairs = sorted(a + "USDT" for a in held if a + "USDT" in self._sym)
        px = {}
        if pairs:
            rows = self.c.book_tickers(symbols=pairs)
            for r in rows if isinstance(rows, list) else [rows]:
                m = self._mid(r)
                if m is not None:
                    px[r["symbol"]] = m
                    self._px[r["symbol"]] = (m, time.time())
        for a, q in held.items():
            pair = a + "USDT"
            if pair not in self._sym:
                continue                            # no USDT pair on this exchange: not priceable, as before
            if pair not in px:                      # a known pair without a price row is unknown, not 0 (D1, 10-14)
                raise KeyError(f"equity: no price for {pair}")
            v += q * px[pair]
        return v

    def market_buy(self, sym: str, quote_qty: float, cid: str) -> dict:
        q = _floor_step(quote_qty, Decimal(1).scaleb(-self._qp(sym)))
        return self.c.place_order(sym, "BUY", "MARKET", quoteOrderQty=self._s(q), newClientOrderId=cid)

    def market_sell(self, sym: str, qty: float, cid: str) -> dict:
        q = round_qty(self.filters(sym), qty, market=True)
        return self.c.place_order(sym, "SELL", "MARKET", quantity=self._s(q), newClientOrderId=cid)

    def stop_loss(self, sym: str, qty: float, stop: float, cid: str) -> dict:
        f = self.filters(sym)
        return self.c.place_order(sym, "SELL", "STOP_LOSS", quantity=self._s(round_qty(f, qty)),
                                  stopPrice=self._s(round_price(f, stop)), newClientOrderId=cid)

    def cancel_replace(self, sym: str, cancel_order_id: int, qty: float, stop: float, cid: str) -> dict:
        f = self.filters(sym)
        return self.c.cancel_replace(sym, cancel_order_id, "STOP_LOSS", quantity=self._s(round_qty(f, qty)),
                                     stopPrice=self._s(round_price(f, stop)), newClientOrderId=cid)

    def cancel(self, sym: str, order_id: int) -> dict:
        return self.c.cancel_order(sym, order_id=order_id)

    def get_order(self, sym: str, order_id=None, cid=None) -> dict:
        return self.c.get_order(sym, order_id=order_id, orig_client_order_id=cid)

    def open_orders(self) -> list:
        return self.c.open_orders()
