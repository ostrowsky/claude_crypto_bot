"""FC-DEMO -- FC_PASSIVE (spot long + USD-M perp short, L = 1 isolated) on the demo account
(fc-demo-execution-1030-spec.md). Demo funding is computed from the demo book: demo money checks MECHANICS only;
the revenue verdict of FC stays with the FC-SHADOW book on live funding.

  FcDemo.step(now, control_dir)
    disabled -> nothing; FLATTEN -> close everything + PAUSE; every cycle: margin stop (short equity < 50% of margin:
    perp reduce-only buy first, then spot sell; re-entry from the next UTC day) and the kill switch (FC equity change
    <= -FC_DEMO_KILL_USDT -> flatten + PAUSE); once a UTC day after 00:15: the day mark, then entries of the pairs
    without a position (spot MARKET BUY quote N -> perp MARKET SELL q floored to the perp step; a failed perp leg
    unwinds the spot leg, alerts one_leg and writes PAUSE)
  PaperVenue  live public prices (spot book, demo-fapi mark), model fees, no order
  DemoVenue   real orders on demo-api / demo-fapi through exchange_client (signed demo hosts only)

  (from trader/) ../pyembed/python.exe fc_demo.py step | status
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402

DIR = TC.RUNTIME / "fc_demo"
CONTROL = DIR / "control"
STATE = DIR / "state.json"
BOOK = "fc_demo"
MARGIN_STOP = 0.5
OPEN_AFTER = timedelta(minutes=15)
SPOT_FEE = 0.001
PERP_FEE = 0.0005


def _day(now) -> str:
    return now.strftime("%Y-%m-%d")


def _floor(x, step):
    return math.floor(x / step + 1e-9) * step


class FcDemo:
    def __init__(self, cfg, venue, log, state=None):
        self.cfg, self.v, self.log = cfg, venue, log
        self.state = state or {"pos": {}, "stopped_day": {}, "realized": 0.0, "funding": 0.0, "last_entry_day": None,
                               "last_mark_day": None, "last_mark_ms": None, "killed": False}

    # ----- control
    def _control(self, now, d):
        if d is None:
            return self.state.get("killed", False)
        d = Path(d)
        if (d / "FLATTEN").exists():
            self.flatten(now, "flatten")
            d.mkdir(parents=True, exist_ok=True)
            (d / "PAUSE").write_text(now.isoformat(), encoding="utf-8")
            (d / "FLATTEN").unlink()
            self.log("fc_control", now, action="flatten")
        paused = (d / "PAUSE").exists()
        if self.state.get("killed") and not paused:
            self.state["killed"] = False
            self.log("fc_control", now, action="resume_after_kill_switch")
        return paused or self.state.get("killed", False)

    def _pause(self, d, now):
        if d is not None:
            Path(d).mkdir(parents=True, exist_ok=True)
            (Path(d) / "PAUSE").write_text(now.isoformat(), encoding="utf-8")

    # ----- money
    def equity_change(self) -> float:
        u = 0.0
        for sym, p in self.state["pos"].items():
            u += p["q_spot"] * self.v.spot_price(sym) - p["spot_cost"] + p["q_perp"] * (p["perp_entry"] - self.v.mark(sym)) \
                - p["fees_in"]
        return self.state["realized"] + self.state["funding"] + u

    # ----- legs
    def _enter(self, sym, now, d):
        N = float(self.cfg.FC_DEMO_NOTIONAL_USDT)
        stamp = now.strftime("%Y%m%d%H%M")
        cs, cp = f"fc_{sym[:3].lower()}_{stamp}_s", f"fc_{sym[:3].lower()}_{stamp}_p"
        try:
            s = self.v.spot_buy_quote(sym, N, cs)
        except Exception as e:                                       # noqa: BLE001
            self.log("fc_alert", now, code="entry_failed", sym=sym, leg="spot", error=str(e))
            return
        q = _floor(float(s["qty"]), self.v.perp_step(sym))
        if q <= 0:
            self._unwind_spot(sym, s["qty"], now, d, "perp_qty_zero")
            return
        try:
            pr = self.v.perp_sell(sym, q, cp)
        except Exception as e:                                       # noqa: BLE001
            err = str(e)
            try:                                                     # the order may have filled anyway (bug 10-30)
                open_q = float(getattr(self.v, "perp_open_qty", lambda _s: 0.0)(sym) or 0)
                if open_q > 0:
                    self.v.perp_buy_reduce(sym, open_q, f"fc_{sym[:3].lower()}_{stamp}_r")
                    err += f"; opened perp {open_q} closed"
            except Exception as e2:                                  # noqa: BLE001
                err += f"; perp check/close failed: {e2}"
            self._unwind_spot(sym, s["qty"], now, d, "perp_rejected", error=err)
            return
        p = {"q_spot": float(s["qty"]), "spot_cost": float(s["cost"]), "q_perp": float(pr["qty"]),
             "perp_entry": float(pr["price"]), "margin": float(pr["qty"]) * float(pr["price"]),
             "fees_in": float(s.get("fee_usdt") or 0) + float(pr.get("fee_usdt") or 0),
             "dust": float(s["qty"]) - float(pr["qty"]), "opened": now.isoformat()}
        self.state["pos"][sym] = p
        self.log("fc_entry", now, sym=sym, client_order_ids=[cs, cp], **{k: v for k, v in p.items() if k != "opened"})

    def _unwind_spot(self, sym, qty, now, d, why, error=None):
        try:
            r = self.v.spot_sell(sym, qty, f"fc_{sym[:3].lower()}_{now.strftime('%Y%m%d%H%M')}_u")
            self.state["realized"] += float(r["proceeds"]) - float(self.cfg.FC_DEMO_NOTIONAL_USDT)
        except Exception as e:                                       # noqa: BLE001
            error = f"{error}; unwind failed: {e}"
        self.log("fc_alert", now, code="one_leg", sym=sym, why=why, error=error)
        self._pause(d, now)

    def close(self, sym, now, reason):
        p = self.state["pos"].get(sym)
        if p is None:
            return
        stamp = now.strftime("%Y%m%d%H%M")
        b = self.v.perp_buy_reduce(sym, p["q_perp"], f"fc_{sym[:3].lower()}_{stamp}_b")
        s = self.v.spot_sell(sym, p["q_spot"], f"fc_{sym[:3].lower()}_{stamp}_x")
        pnl = (float(s["proceeds"]) - p["spot_cost"]) + p["q_perp"] * (p["perp_entry"] - float(b["price"])) \
            - p["fees_in"] - float(b.get("fee_usdt") or 0) - float(s.get("fee_usdt") or 0)
        self.state["realized"] += pnl
        del self.state["pos"][sym]
        self.log("fc_exit", now, sym=sym, reason=reason, pnl=pnl, perp_exit=float(b["price"]), q_spot=p["q_spot"],
                 q_perp=p["q_perp"])

    def flatten(self, now, reason):
        for sym in list(self.state["pos"]):
            self.close(sym, now, reason)

    # ----- the step
    def step(self, now, control_dir=None) -> dict:
        if not getattr(self.cfg, "FC_DEMO_ENABLED", False):
            return {"skipped": "disabled"}
        paused = self._control(now, control_dir)
        day = _day(now)
        for sym, p in list(self.state["pos"].items()):                # margin stop, every cycle
            eq = p["margin"] + p["q_perp"] * (p["perp_entry"] - self.v.mark(sym))
            if eq < MARGIN_STOP * p["margin"]:
                self.close(sym, now, "margin_stop")
                self.state["stopped_day"][sym] = day
        if self.state["pos"] and self.equity_change() <= -float(self.cfg.FC_DEMO_KILL_USDT):
            self.log("fc_alert", now, code="kill_switch", equity_change=self.equity_change())
            self.flatten(now, "kill_switch")
            self.state["killed"] = True
            self._pause(control_dir, now)
            paused = True
        day_open = now >= now.replace(hour=0, minute=0, second=0, microsecond=0) + OPEN_AFTER
        if day_open and self.state.get("last_mark_day") != day:
            self._mark(now)
        if day_open and not paused and self.state.get("last_entry_day") != day:
            for sym in self.cfg.FC_DEMO_PAIRS:
                if sym not in self.state["pos"] and self.state["stopped_day"].get(sym) != day:
                    self._enter(sym, now, control_dir)
            self.state["last_entry_day"] = day
        return {"positions": sorted(self.state["pos"]), "paused": paused}

    def _mark(self, now):
        since = self.state.get("last_mark_ms") or int(now.timestamp() * 1000)
        fund = 0.0
        for sym in self.state["pos"]:
            try:
                fund += float(self.v.funding_since(sym, since))
            except Exception:                                        # noqa: BLE001
                pass
        self.state["funding"] += fund
        spot_value = sum(p["q_spot"] * self.v.spot_price(s) for s, p in self.state["pos"].items())
        perp_pnl = sum(p["q_perp"] * (p["perp_entry"] - self.v.mark(s)) for s, p in self.state["pos"].items())
        margin = sum(p["margin"] for p in self.state["pos"].values())
        delta = {s: p["q_spot"] - p["q_perp"] for s, p in self.state["pos"].items()}
        self.log("fc_mark", now, spot_value=spot_value, perp_pnl=perp_pnl, margin=margin, funding=fund,
                 funding_total=self.state["funding"], hedge_delta=delta, equity_change=self.equity_change(),
                 realized=self.state["realized"], mode=getattr(self.cfg, "FC_DEMO_MODE", None),
                 note="demo funding is not live funding: mechanics only")
        self.state["last_mark_day"], self.state["last_mark_ms"] = _day(now), int(now.timestamp() * 1000)


# ------------------------------------------------------------------ journal readers (daily revenue, report)
def _rows_demo(decisions):
    out = []
    for r in decisions:
        if r.get("book") == BOOK and r.get("mode") == "demo":
            try:
                t = datetime.fromisoformat(str(r["ts"]).replace("Z", "+00:00"))
            except Exception:
                continue
            out.append((t, r))
    return sorted(out, key=lambda x: x[0])


def fc_spot_qty_at(decisions, ts) -> dict:
    """{asset: qty} of the demo FC spot legs at `ts`, from fc_entry / fc_exit rows."""
    q = {}
    for t, r in _rows_demo(decisions):
        if t > ts:
            break
        a = str(r.get("sym", ""))[:-4]
        if r.get("kind") == "fc_entry":
            q[a] = float(r.get("q_spot") or 0)
        elif r.get("kind") == "fc_exit":
            q.pop(a, None)
    return {a: v for a, v in q.items() if v}


def fc_equity_at(decisions, ts):
    """FC equity change (both legs, funding) of the first demo fc_mark at or after `ts` (marks come at ~00:15 UTC)."""
    for t, r in _rows_demo(decisions):
        if t >= ts and r.get("kind") == "fc_mark":
            return float(r.get("equity_change") or 0.0)
    return None


# ------------------------------------------------------------------ venues
class PaperVenue:
    """Live public prices; model fees; no order. Funding is not credited in paper (logged as 0)."""

    def __init__(self):
        import exchange_client as X
        self.X = X
        self.spot = X.ExchangeClient(key="", secret="")             # public calls pass base= (live spot book)
        self.fut = X.ExchangeClient(base=X.DEMO_FAPI, key="", secret="")
        self._steps = None

    def _book(self, sym):
        b = self.spot.book_ticker(sym, base="https://api.binance.com")
        return float(b["bidPrice"]), float(b["askPrice"])

    def spot_price(self, sym):
        bid, ask = self._book(sym)
        return (bid + ask) / 2

    def mark(self, sym):
        return float(self.fut.premium_index(sym, base=self.X.DEMO_FAPI)["markPrice"])

    def perp_step(self, sym):
        if self._steps is None:
            info = self.fut.futures_exchange_info(base=self.X.DEMO_FAPI)
            self._steps = {s["symbol"]: float(next(f["stepSize"] for f in s["filters"] if f["filterType"] == "MARKET_LOT_SIZE"))
                           for s in info.get("symbols", [])}
        return self._steps[sym]

    def spot_buy_quote(self, sym, quote, cid):
        _, ask = self._book(sym)
        q = quote / ask * (1 - SPOT_FEE)
        return {"qty": q, "cost": quote, "fee_usdt": 0.0}           # the fee is inside the qty, as on the exchange

    def spot_sell(self, sym, qty, cid):
        bid, _ = self._book(sym)
        return {"proceeds": qty * bid * (1 - SPOT_FEE), "fee_usdt": 0.0}

    def perp_sell(self, sym, qty, cid):
        m = self.mark(sym)
        return {"qty": qty, "price": m, "fee_usdt": qty * m * PERP_FEE}

    def perp_buy_reduce(self, sym, qty, cid):
        m = self.mark(sym)
        return {"qty": qty, "price": m, "fee_usdt": qty * m * PERP_FEE}

    def funding_since(self, sym, since_ms):
        return 0.0

    def perp_open_qty(self, sym):
        return 0.0


class DemoVenue(PaperVenue):
    """Real orders on demo-api (spot) and demo-fapi (perp); signed requests go to demo hosts only."""

    def __init__(self):
        super().__init__()
        X = self.X
        self.spot_c = X.ExchangeClient()                             # demo-api, keys from files/.env
        self.fut_c = X.ExchangeClient(base=X.DEMO_FAPI)
        self._setup = set()

    def _setup_sym(self, sym):
        if sym in self._setup:
            return
        for call in (lambda: self.fut_c.fut_margin_type(sym, "ISOLATED"), lambda: self.fut_c.fut_leverage(sym, 1)):
            try:
                call()
            except self.X.ExchangeError as e:
                if e.code not in (-4046, -4059):
                    raise
        self._setup.add(sym)

    def spot_buy_quote(self, sym, quote, cid):
        o = self.spot_c.place_order(sym, side="BUY", type_="MARKET", quoteOrderQty=f"{quote:.2f}", newClientOrderId=cid)
        q = float(o.get("executedQty") or 0)
        fee_base = sum(float(f["commission"]) for f in o.get("fills", []) if f.get("commissionAsset") == sym[:-4])
        fee_usdt = sum(float(f["commission"]) * float(f["price"]) for f in o.get("fills", [])
                       if f.get("commissionAsset") == "BNB")
        return {"qty": q - fee_base, "cost": float(o.get("cummulativeQuoteQty") or 0), "fee_usdt": fee_usdt}

    def spot_sell(self, sym, qty, cid):
        info = self.spot_c.exchange_info()
        f = next(s for s in info["symbols"] if s["symbol"] == sym)
        step = float(next(x["stepSize"] for x in f["filters"] if x["filterType"] == "LOT_SIZE"))
        q = _floor(qty, step)
        o = self.spot_c.place_order(sym, side="SELL", type_="MARKET", quantity=f"{q:.8f}".rstrip("0").rstrip("."),
                                    newClientOrderId=cid)
        fee_usdt = sum(float(x["commission"]) for x in o.get("fills", []) if x.get("commissionAsset") == "USDT")
        return {"proceeds": float(o.get("cummulativeQuoteQty") or 0) - fee_usdt, "fee_usdt": 0.0}

    def _perp(self, sym, side, qty, cid, reduce_only):
        self._setup_sym(sym)
        kw = {"quantity": f"{qty:.8f}".rstrip("0").rstrip("."), "newClientOrderId": cid}
        if reduce_only:
            kw["reduceOnly"] = "true"
        o = self.fut_c.fut_order(sym, side, "MARKET", **kw)
        q, px = float(o.get("executedQty") or 0), float(o.get("avgPrice") or 0)
        if q <= 0 and o.get("status") != "FILLED":
            raise RuntimeError(f"perp order not filled: {o.get('status')}")
        est = False
        if px <= 0 or q <= 0:                  # demo RESULT answers FILLED without avgPrice (bug 2026-10-30)
            g = self.fut_c.fut_get_order(sym, o.get("orderId"))
            q = q if q > 0 else float(g.get("executedQty") or 0)
            px = float(g.get("avgPrice") or 0)
            if px <= 0:
                px, est = self.mark(sym), True
        if q <= 0:
            raise RuntimeError(f"perp order without executed quantity: {o.get('status')}")
        return {"qty": q, "price": px, "fee_usdt": q * px * PERP_FEE, "fee_estimated": True, "price_estimated": est}

    def perp_open_qty(self, sym):
        for p in self.fut_c.fut_positions(sym) or []:
            if p.get("symbol") == sym:
                return abs(float(p.get("positionAmt") or 0))
        return 0.0

    def perp_sell(self, sym, qty, cid):
        return self._perp(sym, "SELL", qty, cid, False)

    def perp_buy_reduce(self, sym, qty, cid):
        return self._perp(sym, "BUY", qty, cid, True)

    def funding_since(self, sym, since_ms):
        rows = self.fut_c.fut_income("FUNDING_FEE", limit=100) or []
        return sum(float(r["income"]) for r in rows if r.get("symbol") == sym and int(r.get("time", 0)) > since_ms)


# ------------------------------------------------------------------ one step at a time (scheduler + a manual run)
LOCK = DIR / "step.lock"
LOCK_STALE = timedelta(minutes=10)


def acquire_lock(path: Path, now) -> bool:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            t = datetime.fromisoformat(path.read_text(encoding="utf-8").strip())
        except Exception:
            t = None
        if t is not None and now - t < LOCK_STALE:
            return False
        path.write_text(now.isoformat(), encoding="utf-8")      # stale lock of a crashed step: take it
        return True
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(now.isoformat())
    return True


def release_lock(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


# ------------------------------------------------------------------ CLI
def _logger():
    import audit as A
    A.set_process("fc_demo")
    w = A.default_writer("decisions")

    def log(kind, now, **kw):
        rec = {"kind": kind, "book": BOOK, "mode": getattr(TC, "FC_DEMO_MODE", None), **kw}
        try:
            if w is not None:
                w.write(rec, now=now)
        except Exception:                                            # noqa: BLE001
            pass
        print(json.dumps({"ts": now.isoformat(), **rec}, default=str), flush=True)
    return log


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd not in ("step", "status"):
        print(__doc__)
        return 0
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else None
    if cmd == "status":
        print(json.dumps(st, indent=1))
        return 0
    if not TC.FC_DEMO_ENABLED:
        return 0
    if TC.FC_DEMO_MODE == "demo" and not (TC.TRADING_ENABLED and TC.TRADING_MODE == "demo"):
        print("FC-DEMO: mode demo needs TRADING_ENABLED and TRADING_MODE = 'demo'")
        return 1
    venue = DemoVenue() if TC.FC_DEMO_MODE == "demo" else PaperVenue()
    if st and st.get("mode") != TC.FC_DEMO_MODE and st.get("pos"):
        print(f"FC-DEMO: state was built in mode {st.get('mode')!r}; flatten it before switching")
        return 1
    now = datetime.now(timezone.utc)
    if not acquire_lock(LOCK, now):
        print("FC-DEMO: another step is running; skipped")
        return 0
    try:
        st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else None     # re-read under the lock
        e = FcDemo(TC, venue, _logger(), state=st)
        r = e.step(now, CONTROL)
        DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps({**e.state, "mode": TC.FC_DEMO_MODE}, indent=1, default=str), encoding="utf-8")
        tmp.replace(STATE)
    finally:
        release_lock(LOCK)
    print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
