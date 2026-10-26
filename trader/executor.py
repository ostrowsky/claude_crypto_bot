"""Phase C executor: the E7 book X4 executed on the demo account at a small scale.

demo-execution-c-1013-spec.md. The executor decides nothing: a paper copy of the E7 book
`x4_floor8` (PaperBook, the same ShadowEngine code) makes every decision, and the executor carries
it out on the exchange with a real size = paper size x TRADING_SCALE.

  entry      MARKET BUY quoteOrderQty -> STOP_LOSS on the free balance in the same cycle, placed
             TRADING_CATASTROPHE_GAP_PCT below the policy stop (a catastrophe stop, §3.3)
  trail      the policy stop rises on closed 15m bars -> one order/cancelReplace when the exchange
             stop would move by >= TRADING_TRAIL_UPDATE_MIN_PCT
  exit       the paper book exits (close below the stop, or 96 bars) -> cancel the stop, MARKET SELL
  stop fired the exchange stop filled -> the position is closed, the paper book is NOT adjusted,
             an execution_divergence line is written

Every step goes to the decisions journal (kinds real_entry, real_exit, real_trail, alert, control,
execution_divergence, reconcile) with the clientOrderIds, so the phase-D ledger links every fill to
its decision. Writes only trader/runtime/. No messenger: the operator controls it with files in
runtime/executor/control/ (PAUSE, FLATTEN, RESET_NOTED) through the Claude chat.

TRADING_ENABLED = False (default) runs the C1 paper mode: PaperExchange fills on live demo prices,
no order is sent. TRADING_ENABLED = True and TRADING_MODE = "demo" sends orders to demo-api only.

  (from trader/) ../pyembed/python.exe executor.py run     -- long-running loop (scheduled task)
  (from trader/) ../pyembed/python.exe executor.py status  -- open positions and the last heartbeat
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402
import audit as A                 # noqa: E402
import exchange_client as X       # noqa: E402
import reconcile as RC            # noqa: E402

EXEC_DIR = TC.RUNTIME / "executor"
CONTROL_DIR = EXEC_DIR / "control"
LOCK = EXEC_DIR / "executor.lock"
HEARTBEAT = EXEC_DIR / "heartbeat.json"
BOOK = "exec_x4"
QUOTE = "USDT"
BNB_FEE_RATE = 0.00075        # 0.10% x the demo BNB discount 0.75 (read from the account 2026-10-22)
CLOSED_KEEP = 300


class _Null:
    def write(self, rec, now=None):
        return True


def _iso(t) -> str:
    return t.isoformat() if isinstance(t, datetime) else str(t)


def _dt(v) -> datetime:
    d = v if isinstance(v, datetime) else datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def client_order_id(epoch: str, key: str, leg: str) -> str:
    """cb_<epoch8>_<12 hex of the position key>_<leg>: deterministic (a retry finds its own order),
    unique per position and leg, <= 36 chars as Binance requires. The key is hashed because
    <sym>_<bar_ts> does not fit 36 chars for long symbols (spec §1.3 amended)."""
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    return f"cb_{str(epoch)[:8]}_{h}_{leg}"[:36]


def acquire_lock(path: Path, pid: int, is_alive) -> bool:
    """One executor at a time. A lock whose PID is not a live executor is taken over."""
    path = Path(path)
    try:
        old = json.loads(path.read_text(encoding="utf-8")).get("pid")
    except Exception:
        old = None
    if old is not None and old != pid and is_alive(old):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": pid, "at": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
    return True


def pid_is_executor(pid: int) -> bool:
    """True when `pid` is alive AND its command line runs executor.py (never trust a bare PID)."""
    try:
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').CommandLine"],
                             capture_output=True, text=True, timeout=20).stdout
        return "executor.py" in (out or "")
    except Exception:
        return True                        # unknown: do not start a second one


# ================================================================ paper book (decisions)

class PaperBook:
    """The E7 book X4 as the executor's decision maker (spec §2.2). Same engine, same events, same bars."""

    def __init__(self, policy, fm, bars, floor_of=None, decisions=None, name: str = "x4_floor8"):
        import shadow_portfolio as S
        self.S, self.name = S, name
        self.engine = S.ShadowEngine({name: policy}, fm, floor_of=floor_of, bars=bars,
                                     decisions=decisions if decisions is not None else _Null())
        self._n = 0
        self._last_bar = None

    @property
    def book(self):
        return self.engine.books[self.name]

    def due(self, events, now) -> bool:
        """Live: run the engine only on new events or a newly closed 15m bar (klines cost weight)."""
        last = self.engine.bars.last_closed(now - timedelta(seconds=3))
        return bool(events) or self._last_bar is None or last != self._last_bar

    def process(self, events, now) -> list:
        self.engine.process(events, now)
        self._last_bar = self.engine.bars.last_closed(now - timedelta(seconds=3))
        rows, self._n = self.engine.ledger[self._n:], len(self.engine.ledger)
        acts = []
        for r in rows:
            if r["type"] == "entry":
                acts.append({"type": "entry", "key": f"{r['sym']}|{r['tf']}|{r['ts']}", "sym": r["sym"], "tf": r["tf"],
                             "ts": r["ts"], "price": r["price"], "notional": r["notional"]})
            elif r["type"] == "exit":
                acts.append({"type": "exit", "key": f"{r['sym']}|{r['tf']}|{_iso(r['entry_ts'])}", "sym": r["sym"],
                             "tf": r["tf"], "ts": _iso(r["exit_time"]), "reason": r["reason"],
                             "paper_ret_pct": r.get("ret_pct")})   # the per-trade replay of F1 (10-21)
        return acts

    def trim(self):
        """Live process: drop consumed ledger rows so memory does not grow."""
        del self.engine.ledger[: self._n]
        self._n = 0

    def stop_of(self, key):
        for p in self.book.positions:
            if f"{p['sym']}|{p['tf']}|{_iso(p['entry_ts'])}" == key and p.get("p1") is not None:
                return float(p["p1"].stop)
        return None

    def to_state(self) -> dict:
        return {"books": self.engine.to_state()}

    def load(self, st):
        if st and st.get("books"):
            self.engine.load(st["books"])


# ================================================================ C1 paper exchange

class PaperExchange:
    """The orders protocol on live prices with fill_model costs (C1, spec §6). No request is signed."""

    def __init__(self, price_fn, fm, filters, usdt: float = 5000.0):
        self.price_fn, self.fm, self._f = price_fn, fm, filters
        self.bal, self.locked, self.orders, self.next_id = {QUOTE: float(usdt)}, {}, {}, 1

    def filters(self, sym):
        return self._f(sym) if callable(self._f) else self._f[sym]

    def price(self, sym):
        return float(self.price_fn(sym))

    def free(self, asset):
        return self.bal.get(asset, 0.0)

    def balances(self):
        assets = set(self.bal) | {a for a, v in self.locked.items() if v > 1e-15}
        return [{"asset": a, "free": str(self.bal.get(a, 0.0)), "locked": str(self.locked.get(a, 0.0))}
                for a in sorted(assets)]

    def equity(self):
        v = self.bal.get(QUOTE, 0.0)
        for a in (set(self.bal) | set(self.locked)) - {QUOTE}:
            q = self.bal.get(a, 0.0) + self.locked.get(a, 0.0)
            if q > 0:
                v += q * self.price(a + QUOTE)     # no price -> raise: an unpriced coin is unknown, not 0 (D1, 10-14)
        return v

    def _new(self, sym, side, typ, qty, cid, status, stop=None, quote=0.0, fills=None):
        self.next_id += 1
        o = {"symbol": sym, "orderId": self.next_id, "clientOrderId": cid, "side": side, "type": typ,
             "origQty": str(qty), "executedQty": str(qty if status == "FILLED" else 0), "cummulativeQuoteQty": str(quote),
             "status": status, "stopPrice": str(stop or 0), "fills": fills or []}
        self.orders[o["orderId"]] = o
        return dict(o)

    def market_buy(self, sym, quote_qty, cid):
        px = self.fm.buy(self.price(sym))
        qty = RC._floor(float(quote_qty) / px, RC.step(self.filters(sym)))
        base = sym[: -len(QUOTE)]
        comm = qty * self.fm.taker_fee
        self.bal[base] = self.bal.get(base, 0.0) + qty - comm
        self.bal[QUOTE] -= qty * px
        return self._new(sym, "BUY", "MARKET", qty, cid, "FILLED", quote=qty * px,
                         fills=[{"price": str(px), "qty": str(qty), "commission": str(comm), "commissionAsset": base}])

    def _sell(self, sym, qty, raw_px):
        px = self.fm.sell(raw_px)
        comm = qty * px * self.fm.taker_fee
        self.bal[QUOTE] += qty * px - comm
        return px, comm

    def market_sell(self, sym, qty, cid):
        base = sym[: -len(QUOTE)]
        raw = self.price(sym)                  # price first: a failed request must not take the coins (D2, 10-14)
        if self.bal.get(base, 0.0) + 1e-12 < float(qty):
            raise X.ExchangeError(400, -2010, "Account has insufficient balance", "/api/v3/order")
        self.bal[base] -= float(qty)
        px, comm = self._sell(sym, float(qty), raw)
        return self._new(sym, "SELL", "MARKET", float(qty), cid, "FILLED", quote=float(qty) * px,
                         fills=[{"price": str(px), "qty": str(qty), "commission": str(comm), "commissionAsset": QUOTE}])

    def stop_loss(self, sym, qty, stop, cid):
        base = sym[: -len(QUOTE)]
        if self.bal.get(base, 0.0) + 1e-12 < float(qty):
            raise X.ExchangeError(400, -2010, "Account has insufficient balance", "/api/v3/order")
        self.bal[base] -= float(qty)
        self.locked[base] = self.locked.get(base, 0.0) + float(qty)
        return self._new(sym, "SELL", "STOP_LOSS", float(qty), cid, "NEW", stop=float(stop))

    def _unlock(self, o):
        base = o["symbol"][: -len(QUOTE)]
        q = float(o["origQty"])
        self.locked[base] = self.locked.get(base, 0.0) - q
        self.bal[base] = self.bal.get(base, 0.0) + q

    def cancel(self, sym, order_id):
        o = self.orders.get(order_id)
        if o is None or o["status"] != "NEW":
            raise X.ExchangeError(400, -2011, "Unknown order sent.", "/api/v3/order")
        o["status"] = "CANCELED"
        self._unlock(o)
        return dict(o)

    def cancel_replace(self, sym, cancel_order_id, qty, stop, cid):
        old = self.cancel(sym, cancel_order_id)
        new = self.stop_loss(sym, qty, stop, cid)
        return {"cancelResult": "SUCCESS", "newOrderResult": "SUCCESS", "cancelResponse": old, "newOrderResponse": new}

    def get_order(self, sym, order_id=None, cid=None):
        for o in self.orders.values():
            if o["symbol"] == sym and (o["orderId"] == order_id or (cid is not None and o["clientOrderId"] == cid)):
                return dict(o)
        raise X.ExchangeError(400, -2013, "Order does not exist.", "/api/v3/order")

    def open_orders(self):
        return [dict(o) for o in self.orders.values() if o["status"] == "NEW"]

    def on_prices(self):
        """Fill the stops the live price has crossed (a market stop sells at the price, with costs)."""
        for o in list(self.orders.values()):
            if o["type"] != "STOP_LOSS" or o["status"] != "NEW":
                continue
            try:
                p = self.price(o["symbol"])
            except Exception:
                continue
            if p <= float(o["stopPrice"]):
                base, q = o["symbol"][: -len(QUOTE)], float(o["origQty"])
                self.locked[base] = self.locked.get(base, 0.0) - q
                px, comm = self._sell(o["symbol"], q, p)
                o.update(status="FILLED", executedQty=str(q), cummulativeQuoteQty=str(q * px),
                         fills=[{"price": str(px), "qty": str(q), "commission": str(comm), "commissionAsset": QUOTE}])

    def to_state(self):
        return {"bal": self.bal, "locked": self.locked, "orders": self.orders, "next_id": self.next_id}

    def load(self, st):
        if st:
            self.bal, self.locked = dict(st["bal"]), dict(st["locked"])
            self.orders = {int(k): v for k, v in st["orders"].items()}
            self.next_id = int(st["next_id"])


# ================================================================ executor

class Executor:
    def __init__(self, paper, orders, cfg, decisions=None, state=None):
        self.paper, self.ex, self.cfg = paper, orders, cfg
        self.w = decisions if decisions is not None else _Null()
        self.positions, self.closed = {}, []
        self.realized, self.seq = {}, {}
        self.peak_equity, self.killed, self.block_entries = None, False, False
        self.last_poll = self.last_reconcile = None
        self.day_status = None
        if state:
            self._load(state)

    # ----- journal
    def _log(self, kind, now, **kw):
        rec = {"kind": kind, "book": BOOK, "mode": getattr(self.cfg, "TRADING_MODE", None), **kw}
        if kw.get("key") and "decision_id" not in rec:
            sym, tf, ts = (kw["key"].split("|") + [None, None, None])[:3]
            rec["decision_id"] = A.decision_id(BOOK, sym, tf, ts, kind)
        try:
            self.w.write(rec, now=now)
        except TypeError:
            self.w.write(rec)
        except Exception:
            pass

    def _alert(self, now, code, **kw):
        self._log("alert", now, code=code, **kw)

    # ----- helpers
    def _cid(self, key, leg):
        return client_order_id(self.cfg.TRADING_EPOCH, key, leg)

    def _next_stop_cid(self, key):
        n = self.seq.get(key, 0) + 1
        return n, self._cid(key, f"s{n}")

    @staticmethod
    def _base(sym):
        return sym[: -len(QUOTE)] if sym.endswith(QUOTE) else sym

    def _open(self):
        return {k: p for k, p in self.positions.items() if p["state"] != "CLOSED"}

    def _closed_keys(self):
        return {p["key"] for p in self.closed}

    def _fee_info(self, o, quote, only_bnb=False):
        """(fee in USDT, estimated?) of an order. A USDT commission counts as it is; a BNB commission (the demo
        discount is on, 2026-10-22) is priced at BNBUSDT, or estimated at the discounted rate when that price is
        unknown; a base-asset commission is already in the qty. No fills: the taker estimate as before."""
        fills = o.get("fills") or []
        if not fills:
            return (0.0, False) if only_bnb else (quote * 0.001, True)
        fee, est = 0.0, False
        for f in fills:
            asset, c = f.get("commissionAsset"), float(f.get("commission") or 0)
            if asset == QUOTE and not only_bnb:
                fee += c
            elif asset == "BNB":
                try:
                    fee += c * float(self.ex.price("BNB" + QUOTE))
                except Exception:
                    fee += float(f.get("price") or 0) * float(f.get("qty") or 0) * BNB_FEE_RATE
                    est = True
        return fee, est

    def _fee_usdt(self, o, quote):
        return self._fee_info(o, quote)[0]

    # ----- cycle
    def cycle(self, events, now, control_dir=None):
        paused = self._control(now, control_dir)
        if self.last_reconcile is None or (now - self.last_reconcile).total_seconds() >= self.cfg.TRADING_RECONCILE_SEC:
            self.reconcile(now)
        poll_due = self.last_poll is None or (now - self.last_poll).total_seconds() >= self.cfg.TRADING_POLL_SEC
        if poll_due:
            paused = self._kill_switch(now, control_dir) or paused
        due = getattr(self.paper, "due", None)
        acts = self.paper.process(events, now) if (due is None or due(events, now)) else []
        for a in acts:
            if a["type"] == "entry":
                self._enter(a, now, paused)
            else:
                self._exit(a, now)
        for k in list(self._open()):
            self._sync_stop(k, now)
        if poll_due:
            self._poll(now)
            self.last_poll = now
        return acts

    # ----- control files
    def _control(self, now, d):
        if d is None:
            return self.killed
        d = Path(d)
        if (d / "FLATTEN").exists():
            self.flatten(now, "flatten")
            d.mkdir(parents=True, exist_ok=True)
            (d / "PAUSE").write_text(now.isoformat(), encoding="utf-8")
            (d / "FLATTEN").unlink()
            self._log("control", now, action="flatten")
        if (d / "RESET_NOTED").exists():
            (d / "RESET_NOTED").unlink()
            self._log("control", now, action="reset_noted", day=now.strftime("%Y-%m-%d"))
        paused = (d / "PAUSE").exists()
        if self.killed and not paused:                      # the operator removed PAUSE: resume
            self.killed, self.peak_equity = False, None
            self._log("control", now, action="resume_after_kill_switch")
        return paused

    def _kill_switch(self, now, d) -> bool:
        if self.killed:
            return True
        try:
            eq = float(self.ex.equity())
        except Exception:
            return False
        self.peak_equity = eq if self.peak_equity is None else max(self.peak_equity, eq)
        if self.peak_equity - eq >= self.cfg.TRADING_KILL_SWITCH_USDT:
            self.killed = True
            self._alert(now, "kill_switch", equity=eq, peak=self.peak_equity)
            self.flatten(now, "kill_switch")
            if d is not None:
                Path(d).mkdir(parents=True, exist_ok=True)
                (Path(d) / "PAUSE").write_text(now.isoformat(), encoding="utf-8")
            return True
        return False

    def flatten(self, now, reason):
        for k in list(self._open()):
            self._close_by_policy(self.positions[k], now, reason)

    # ----- entry
    def _enter(self, a, now, paused):
        key, sym = a["key"], a["sym"]
        cfg = self.cfg

        def skip(reason, **kw):
            self._log("real_entry", now, key=key, sym=sym, action="skip", reason=reason, paper_notional=a["notional"], **kw)

        if paused:
            return skip("paused")
        if self.block_entries:
            return skip("reconcile_fail")
        day = now.strftime("%Y-%m-%d")
        if -self.realized.get(day, 0.0) >= cfg.TRADING_DAILY_LOSS_LIMIT_USDT:
            return skip("daily_loss_limit_real", realized=self.realized.get(day, 0.0))
        if any(p["sym"] == sym for p in self._open().values()):
            return skip("asset_held")
        quote = float(a["notional"]) * cfg.TRADING_SCALE
        try:
            f = self.ex.filters(sym)
        except Exception as e:
            return skip("no_filters", error=str(e))
        need = max(RC.min_notional(f) * 1.2, cfg.TRADING_MIN_ORDER_USDT)
        if quote < need:
            return skip("below_min_notional", quote=quote, need=need)
        exposure = sum(p["cost"] for p in self._open().values())
        if exposure + quote > cfg.TRADING_MAX_EXPOSURE_USDT:
            return skip("exposure_cap", quote=quote, exposure=exposure)
        ps = self.paper.stop_of(key)
        if ps is None:
            return skip("no_policy_stop")
        cid = self._cid(key, "b")
        o = self._place(lambda: self.ex.market_buy(sym, quote, cid), sym, cid)
        if isinstance(o, X.ExchangeError):
            return skip("rejected", code=o.code, msg=o.msg, client_order_ids=[cid])
        executed = float(o.get("executedQty") or 0)
        if executed <= 0:
            return skip("not_filled", status=o.get("status"), client_order_ids=[cid])
        cost = float(o.get("cummulativeQuoteQty") or 0)
        fill = cost / executed
        bnb_fee, fee_est = self._fee_info(o, cost, only_bnb=True)   # a BNB-paid buy fee is part of the cost
        cost += bnb_fee
        p = {"key": key, "sym": sym, "tf": a["tf"], "entry_ts": a["ts"], "state": "FILLED", "buy_order_id": o.get("orderId"),
             "cost": cost, "executed_qty": executed, "fill_price": fill, "paper_price": float(a["price"]),
             "paper_notional": float(a["notional"]), "filled_at": now.isoformat(), "protected_at": None,
             "stop_order_id": None, "stop_price": None, "qty": None, "client_order_ids": [cid],
             "buy_status": o.get("status")}
        self.positions[key] = p
        ok = self._protect(p, now, RC.catastrophe_stop(ps, cfg.TRADING_CATASTROPHE_GAP_PCT, f))
        latency = None
        try:
            latency = (now - _dt(a["ts"])).total_seconds()
        except Exception:
            pass
        self._log("real_entry", now, key=key, sym=sym, action="enter", reason="enter", quote=quote, cost=cost,
                  qty=executed, fill_price=fill, paper_price=float(a["price"]),
                  slippage_bps=(fill / float(a["price"]) - 1) * 1e4 if a.get("price") else None,
                  latency_sec=latency, policy_stop=ps, stop_price=p["stop_price"], protected=ok,
                  fee_usdt=bnb_fee, fee_estimated=fee_est,
                  status=o.get("status"), client_order_ids=list(p["client_order_ids"]))

    def _place(self, call, sym, cid):
        """Place once; on a lost reply look the order up by its clientOrderId; place again only if
        the exchange does not know it (invariant 3). Returns the order or the ExchangeError."""
        try:
            return call()
        except X.ExchangeError as e:
            if e.status is not None and e.code not in (-1007,):     # a definite answer: rejected
                return e
            try:
                return self.ex.get_order(sym, cid=cid)
            except X.ExchangeError as e2:
                if e2.code != -2013:
                    return e2
            try:
                return call()
            except X.ExchangeError as e3:
                return e3
        except X.RequestDeadline as e:
            try:
                return self.ex.get_order(sym, cid=cid)
            except X.ExchangeError:
                return X.ExchangeError(None, "deadline", str(e), "")

    # ----- protection
    def _protect(self, p, now, stop) -> bool:
        sym, f = p["sym"], self.ex.filters(p["sym"])
        free = self.ex.free(self._base(sym))
        qty = RC._floor(free, RC.step(f))
        min_qty = float(f.get("LOT_SIZE", {}).get("minQty") or 0)
        if qty <= 0 or qty < min_qty:                        # no coins -> nothing to protect (D3, 10-14)
            self._alert(now, "nothing_to_protect", key=p["key"], sym=sym, free=free)
            self._finish(p, now, "no_balance", 0.0, dust=free, sold=0.0)
            return False
        last = None
        n, cid = self._next_stop_cid(p["key"])
        for _ in range(int(self.cfg.TRADING_PROTECT_RETRIES)):
            o = self._place(lambda: self.ex.stop_loss(sym, qty, stop, cid), sym, cid)
            if not isinstance(o, X.ExchangeError):
                self.seq[p["key"]] = n
                if cid not in p["client_order_ids"]:
                    p["client_order_ids"].append(cid)
                p.update(state="PROTECTED", stop_order_id=o.get("orderId"), stop_price=float(stop), qty=qty,
                         protected_at=now.isoformat())
                return True
            last = o
        if cid not in p["client_order_ids"]:
            p["client_order_ids"].append(cid)
        self._alert(now, "protection_failed", key=p["key"], sym=sym, err_code=getattr(last, "code", None),
                    msg=getattr(last, "msg", None))
        self._sell_and_close(p, now, "protection_failed")
        return False

    def _sync_stop(self, key, now):
        p = self.positions.get(key)
        if p is None or p["state"] != "PROTECTED":
            return
        ps = self.paper.stop_of(key)
        if ps is None:
            return
        f = self.ex.filters(p["sym"])
        want = RC.catastrophe_stop(ps, self.cfg.TRADING_CATASTROPHE_GAP_PCT, f)
        if want < p["stop_price"] * (1 + self.cfg.TRADING_TRAIL_UPDATE_MIN_PCT / 100):
            return
        n, cid = self._next_stop_cid(key)
        try:
            r = self.ex.cancel_replace(p["sym"], p["stop_order_id"], p["qty"], want, cid)
        except X.ExchangeError as e:
            data = getattr(e, "data", None) or {}
            self._alert(now, "trail_replace_failed", key=key, err_code=e.code, msg=e.msg, data=data)
            if data.get("cancelResult") == "SUCCESS":           # old stop gone, new one not placed
                p["client_order_ids"].append(cid)
                self.seq[key] = n
                p["state"] = "FILLED"
                self._protect(p, now, want)
            return
        new = r.get("newOrderResponse") or {}
        self.seq[key] = n
        p["client_order_ids"].append(cid)
        old = p["stop_price"]
        p.update(stop_order_id=new.get("orderId"), stop_price=float(want))
        self._log("real_trail", now, key=key, sym=p["sym"], policy_stop=ps, stop_from=old, stop_to=want,
                  client_order_ids=[cid])

    # ----- exits
    def _exit(self, a, now):
        key = a["key"]
        p = self.positions.get(key)
        if p is None or p["state"] == "CLOSED":
            why = "already_closed" if key in self._closed_keys() else "no_real_position"
            self._log("real_exit", now, key=key, sym=a["sym"], action="skip", reason=why, paper_reason=a.get("reason"),
                      paper_ret_pct=a.get("paper_ret_pct"))
            return
        p["paper_ret_pct"] = a.get("paper_ret_pct")
        self._close_by_policy(p, now, a.get("reason") or "paper_exit")

    def _close_by_policy(self, p, now, reason):
        if p.get("stop_order_id") is not None and p["state"] == "PROTECTED":
            try:
                self.ex.cancel(p["sym"], p["stop_order_id"])
            except X.ExchangeError:
                try:
                    o = self.ex.get_order(p["sym"], order_id=p["stop_order_id"])
                except X.ExchangeError:
                    o = {}
                if o.get("status") == "FILLED":
                    return self._closed_by_exchange(p, o, now)
        self._sell_and_close(p, now, reason)

    def _sell_and_close(self, p, now, reason):
        sym, f = p["sym"], self.ex.filters(p["sym"])
        free = self.ex.free(self._base(sym))
        qty = RC._floor(free, RC.step(f))
        min_qty = float(f.get("LOT_SIZE", {}).get("minQty") or 0)
        proceeds = fee = 0.0
        sold, cid = 0.0, self._cid(p["key"], "x")
        try:
            px = float(self.ex.price(sym))
        except Exception:
            px = p.get("fill_price") or 0.0
        if qty >= min_qty and qty > 0 and qty * px >= RC.min_notional(f):
            o = self._place(lambda: self.ex.market_sell(sym, qty, cid), sym, cid)
            if isinstance(o, X.ExchangeError):
                self._alert(now, "sell_failed", key=p["key"], sym=sym, err_code=o.code, msg=o.msg)
                return False
            sold = float(o.get("executedQty") or qty)
            proceeds = float(o.get("cummulativeQuoteQty") or 0)
            fee = self._fee_usdt(o, proceeds)
            p["client_order_ids"].append(cid)
        self._finish(p, now, reason, proceeds - fee, dust=max(0.0, free - sold), sold=sold)
        return True

    def _closed_by_exchange(self, p, o, now):
        proceeds = float(o.get("cummulativeQuoteQty") or 0)
        fee = self._fee_usdt(o, proceeds)
        self._finish(p, now, "exchange_stop", proceeds - fee, dust=None, sold=float(o.get("executedQty") or 0))
        self._log("execution_divergence", now, key=p["key"], sym=p["sym"], reason="exchange_stop",
                  stop_price=p.get("stop_price"), proceeds=proceeds)

    def _finish(self, p, now, reason, net, dust, sold):
        pnl = net - p["cost"] if net is not None else None   # 0 proceeds is a loss; None = unknown (D4, 10-14)
        p.update(state="CLOSED", exit_reason=reason, closed_at=now.isoformat(), net_proceeds=net, pnl=pnl,
                 dust_qty=dust, sold_qty=sold)
        if pnl is not None:
            day = now.strftime("%Y-%m-%d")
            self.realized[day] = self.realized.get(day, 0.0) + pnl
        self.positions.pop(p["key"], None)
        self.closed.append(p)
        self.closed = self.closed[-CLOSED_KEEP:]
        self._log("real_exit", now, key=p["key"], sym=p["sym"], action="exit", reason=reason, pnl=pnl,
                  ret_pct=(pnl / p["cost"] * 100) if (pnl is not None and p["cost"]) else None,
                  net_proceeds=net, dust_qty=dust, paper_ret_pct=p.get("paper_ret_pct"),
                  client_order_ids=list(p["client_order_ids"]))

    # ----- poll
    def _poll(self, now):
        for k, p in list(self._open().items()):
            if p["state"] != "PROTECTED":
                continue
            try:
                o = self.ex.get_order(p["sym"], order_id=p["stop_order_id"])
            except X.ExchangeError as e:
                if e.code != -2013:
                    continue
                o = {"status": "MISSING"}
            st = o.get("status")
            if st == "FILLED":
                self._closed_by_exchange(p, o, now)
            elif st in ("CANCELED", "EXPIRED", "REJECTED", "EXPIRED_IN_MATCH", "MISSING"):
                self._alert(now, "stop_missing", key=k, sym=p["sym"], status=st)
                p["state"] = "FILLED"
                ps = self.paper.stop_of(k)
                f = self.ex.filters(p["sym"])
                stop = RC.catastrophe_stop(ps, self.cfg.TRADING_CATASTROPHE_GAP_PCT, f) if ps else p["stop_price"]
                self._protect(p, now, stop)

    # ----- reconcile
    def reconcile(self, now):
        self.last_reconcile = now
        try:
            acts = RC.run(self._open(), self.ex, self.paper.stop_of, self.cfg, day_status=self.day_status)
        except Exception as e:
            self._alert(now, "reconcile_error", error=repr(e))
            return []
        self.block_entries = any(a["action"] == "block_entries" for a in acts)
        for a in acts:
            act = a["action"]
            if act == "place_stop":
                p = self.positions[a["key"]]
                self._alert(now, "reconcile_place_stop", key=a["key"], sym=a["sym"])
                p["state"] = "FILLED"
                self._protect(p, now, a["stop"])
            elif act == "orphan_balance":
                self._alert(now, "orphan_balance", sym=a["sym"], qty=a["qty"])
                cid = self._cid("orphan|" + a["sym"] + "|" + now.strftime("%Y%m%d%H%M"), "o")
                o = self._place(lambda: self.ex.stop_loss(a["sym"], a["qty"], a["stop"], cid), a["sym"], cid)
                if isinstance(o, X.ExchangeError):
                    self._alert(now, "orphan_stop_failed", sym=a["sym"], err_code=o.code, msg=o.msg)
            elif act == "cancel_order":
                self._alert(now, "orphan_order", sym=a["sym"], order_id=a["order_id"])
                try:
                    self.ex.cancel(a["sym"], a["order_id"])
                except X.ExchangeError as e:
                    self._alert(now, "cancel_failed", sym=a["sym"], err_code=e.code)
            elif act == "close_from_trades":
                self._alert(now, "position_gone", key=a["key"], sym=a["sym"])
                self._finish(self.positions[a["key"]], now, "gone_from_exchange", None, dust=None, sold=0.0)
            elif act == "alert":
                self._alert(now, a.get("code", "reconcile"), **{k: v for k, v in a.items() if k not in ("action", "code")})
        if acts:
            self._log("reconcile", now, actions=[a["action"] for a in acts])
        return acts

    # ----- state
    def to_state(self) -> dict:
        return {"positions": self.positions, "closed": self.closed[-CLOSED_KEEP:], "realized": self.realized,
                "seq": self.seq, "peak_equity": self.peak_equity, "killed": self.killed,
                "paper": self.paper.to_state()}

    def _load(self, st):
        self.positions = dict(st.get("positions") or {})
        self.closed = list(st.get("closed") or [])
        self.realized = dict(st.get("realized") or {})
        self.seq = dict(st.get("seq") or {})
        self.peak_equity, self.killed = st.get("peak_equity"), bool(st.get("killed"))
        if st.get("paper") is not None:
            self.paper.load(st["paper"])


# ================================================================ live process

def _save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def _day_status_yesterday(now):
    try:
        import daily_revenue as DR
        rows = DR.latest_by_day(TC.RUNTIME / "metrics" / "daily_revenue.jsonl")
        return (rows.get((now - timedelta(days=1)).strftime("%Y-%m-%d")) or {}).get("status")
    except Exception:
        return None


def pair_price_fn(client, ttl: float = 5.0):
    """Mid price of ONE pair per request, cached `ttl` seconds per pair (never the all-pairs ticker)."""
    cache = {}

    def price(sym):
        hit = cache.get(sym)
        if hit is None or time.time() - hit[1] > ttl:
            r = client.book_ticker(sym)
            b, a = float(r.get("bidPrice") or 0), float(r.get("askPrice") or 0)
            if not (b > 0 and a > 0):
                raise KeyError(f"{sym}: empty book")
            cache[sym] = hit = ((b + a) / 2, time.time())
        return hit[0]
    return price


def build(mode: str):
    """Paper book + orders for `mode` ('paper' = C1, 'demo' = C2)."""
    import fill_model as FM
    import microstructure_recorder as M
    import shadow_portfolio as S
    TC.bot_readonly()
    import config as BOT
    import exit_validator as EV
    pub = X.ExchangeClient(key="", secret="")
    bars = S.LiveBars(pub, atr_period=int(getattr(BOT, "ATR_PERIOD", 14)))
    paper = PaperBook(S.X4, FM.model("taker", 1), bars, floor_of=lambda m: EV.floor_of(m, BOT))
    if mode == "demo":
        client = X.ExchangeClient()
        orders = X.Orders(client, M.cached_exchange_info(client))
    else:
        xinfo = M.cached_exchange_info(pub)
        orders = PaperExchange(pair_price_fn(pub), FM.model("taker", 1), lambda s: X.symbol_filters(xinfo, s),
                               usdt=5000.0)
    return paper, orders


def run_forever(max_loops: int | None = None) -> int:
    import faulthandler
    import shadow_portfolio as S
    A.set_process("executor")
    mode = "demo" if (TC.TRADING_ENABLED and TC.TRADING_MODE == "demo") else "paper"
    EXEC_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    if not acquire_lock(LOCK, os.getpid(), pid_is_executor):
        print("another executor is running -- exit")
        return 0
    state_path = EXEC_DIR / f"state_{mode}.json"
    st = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    tail = S.EventTail(TC.BOT_EVENTS, st.get("tail") or {})
    if not st:                                            # start from NOW, never replay history
        tail.st["offset"] = TC.BOT_EVENTS.stat().st_size if TC.BOT_EVENTS.exists() else 0
        st["started_utc"] = datetime.now(timezone.utc).isoformat()
    paper, orders = build(mode)
    if mode == "paper" and st.get("paper_exchange"):
        orders.load(st["paper_exchange"])
    cfg = TC
    exe = Executor(paper, orders, cfg, A.default_writer("decisions") or _Null(), state=st.get("executor"))
    now = datetime.now(timezone.utc)
    for b in paper.engine.books.values():
        if b.cur_day is None:
            b.cur_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            b.day_start_eq.setdefault(b.cur_day.strftime("%Y-%m-%d"), b.cash)
    exe._log("control", now, action="start", mode=mode, pid=os.getpid())
    loops = 0
    while True:
        faulthandler.dump_traceback_later(180, exit=True)     # a hung loop exits; the task restarts it
        now = datetime.now(timezone.utc)
        try:
            exe.day_status = _day_status_yesterday(now)
            if mode == "paper":
                orders.on_prices()
            evs = tail.read_new()
            exe.cycle(evs, now, control_dir=CONTROL_DIR)
            paper.trim()
            st.update(tail=tail.st, executor=exe.to_state(), last_loop_utc=now.isoformat(), mode=mode)
            if mode == "paper":
                st["paper_exchange"] = orders.to_state()
            _save(state_path, st)
            _save(HEARTBEAT, {"ts": now.isoformat(), "pid": os.getpid(), "mode": mode,
                              "open": len(exe._open()), "closed": len(exe.closed)})
        except Exception as e:                                # one bad loop must not kill the process
            exe._alert(now, "loop_error", error=repr(e)[:500])
        loops += 1
        if max_loops is not None and loops >= max_loops:
            faulthandler.cancel_dump_traceback_later()
            return 0
        time.sleep(float(TC.TRADING_LOOP_SEC))


def status() -> dict:
    hb = json.loads(HEARTBEAT.read_text(encoding="utf-8")) if HEARTBEAT.exists() else None
    out = {"heartbeat": hb, "control": sorted(p.name for p in CONTROL_DIR.glob("*")) if CONTROL_DIR.exists() else []}
    for mode in ("paper", "demo"):
        p = EXEC_DIR / f"state_{mode}.json"
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8")).get("executor") or {}
            out[mode] = {"open": list((s.get("positions") or {}).keys()), "closed": len(s.get("closed") or []),
                         "realized": s.get("realized")}
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "status":
        print(json.dumps(status(), ensure_ascii=False, indent=1, default=str))
        return 0
    if argv and argv[0] == "run":
        return run_forever(int(argv[1]) if len(argv) > 1 else None)
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
