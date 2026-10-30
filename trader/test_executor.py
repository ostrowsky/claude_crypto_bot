"""Phase C executor -- tests written from demo-execution-c-1013-spec.md §7 BEFORE the code
(operator rule 2026-10-10: spec -> tests -> code)."""
import io
import json
import tempfile
import types
import unittest
import urllib.error
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

import exchange_client as X
import fill_model as FM
import trading_policy as TP
from test_demo_connection_a import FakeResp

HERE = Path(__file__).resolve().parent
NOW = datetime(2026, 10, 14, 1, 0, 5, tzinfo=timezone.utc)
X4 = replace(TP.CHAMPION, name="x4_floor8", exits=TP.Exits(kind="p1_trail", floor_override=0.08))


def CFG(**kw):
    base = dict(TRADING_SCALE=0.02, TRADING_MIN_ORDER_USDT=6.0, TRADING_MAX_EXPOSURE_USDT=250.0,
                TRADING_DAILY_LOSS_LIMIT_USDT=25.0, TRADING_KILL_SWITCH_USDT=60.0, TRADING_CATASTROPHE_GAP_PCT=3.0,
                TRADING_TRAIL_UPDATE_MIN_PCT=0.2, TRADING_MAX_UNPROTECTED_SEC=60, TRADING_POLL_SEC=15,
                TRADING_PROTECT_RETRIES=3, TRADING_RECONCILE_SEC=300, TRADING_EPOCH="t1013a01")
    base.update(kw)
    return types.SimpleNamespace(**base)


def filt(step="0.01", tick="0.0001", min_notional="5", min_qty="0.01"):
    return {"LOT_SIZE": {"filterType": "LOT_SIZE", "minQty": min_qty, "maxQty": "9000000", "stepSize": step},
            "PRICE_FILTER": {"filterType": "PRICE_FILTER", "minPrice": "0.0001", "maxPrice": "100000", "tickSize": tick},
            "NOTIONAL": {"filterType": "NOTIONAL", "minNotional": min_notional, "applyMinToMarket": True,
                         "maxNotional": "9000000", "applyMaxToMarket": False}}


def floor_step(x, step):
    s = Decimal(step)
    return float((Decimal(str(x)) / s).to_integral_value(rounding=ROUND_DOWN) * s)


class W:
    """In-memory audit writer."""

    def __init__(self):
        self.rows = []

    def write(self, rec, now=None):
        self.rows.append(dict(rec))
        return True

    def kind(self, k):
        return [r for r in self.rows if r.get("kind") == k]


class FakeOrders:
    """The orders protocol of spec §2.3 on an in-memory spot account."""

    def __init__(self, prices, filters=None, usdt=5000.0, fee=0.001):
        self.px, self.f, self.fee = dict(prices), dict(filters or {}), fee
        self.bal, self.locked = {"USDT": float(usdt)}, {}
        self.orders, self.next_id, self.calls = {}, 100, []
        self.fail, self.lost_reply, self.partial = {}, set(), None

    # ----- helpers
    def _maybe_fail(self, m):
        q = self.fail.get(m)
        if q:
            e = q.pop(0)
            if e is not None:
                raise e

    def _new(self, sym, side, typ, qty, cid, status, stop=None, quote=0.0):
        self.next_id += 1
        o = {"symbol": sym, "orderId": self.next_id, "clientOrderId": cid, "side": side, "type": typ,
             "origQty": str(qty), "executedQty": str(qty if status in ("FILLED", "EXPIRED") else 0),
             "cummulativeQuoteQty": str(quote), "status": status, "stopPrice": str(stop or 0)}
        self.orders[o["orderId"]] = o
        return dict(o)

    def _step(self, sym):
        return self.filters(sym)["LOT_SIZE"]["stepSize"]

    # ----- protocol
    def filters(self, sym):
        return self.f.get(sym, filt())

    def free(self, asset):
        return self.bal.get(asset, 0.0)

    def price(self, sym):
        return self.px[sym]

    def balances(self):
        assets = set(self.bal) | {a for a, v in self.locked.items() if v}
        return [{"asset": a, "free": str(self.bal.get(a, 0.0)), "locked": str(self.locked.get(a, 0.0))} for a in sorted(assets)]

    def equity(self):
        v = self.bal.get("USDT", 0.0)
        for a in set(self.bal) | set(self.locked):
            if a != "USDT":
                v += (self.bal.get(a, 0.0) + self.locked.get(a, 0.0)) * self.px.get(a + "USDT", 0.0)
        return v

    def market_buy(self, sym, quote_qty, cid):
        self.calls.append(("market_buy", sym, float(quote_qty), cid))
        self._maybe_fail("market_buy")
        px = self.px[sym]
        qty = floor_step(float(quote_qty) / px, self._step(sym))
        status = "FILLED"
        if self.partial:
            qty, status = floor_step(qty * self.partial, self._step(sym)), "EXPIRED"
        base = sym[:-4]
        self.bal[base] = self.bal.get(base, 0.0) + qty * (1 - self.fee)
        self.bal["USDT"] -= qty * px
        o = self._new(sym, "BUY", "MARKET", qty, cid, status, quote=qty * px)
        if "market_buy" in self.lost_reply:
            self.lost_reply.discard("market_buy")
            raise X.ExchangeError(None, "network", "timeout", "/api/v3/order")
        return o

    def market_sell(self, sym, qty, cid):
        self.calls.append(("market_sell", sym, float(qty), cid))
        self._maybe_fail("market_sell")
        base = sym[:-4]
        if self.bal.get(base, 0.0) + 1e-12 < float(qty):
            raise X.ExchangeError(400, -2010, "Account has insufficient balance", "/api/v3/order")
        self.bal[base] -= float(qty)
        px = self.px[sym]
        self.bal["USDT"] += float(qty) * px * (1 - self.fee)
        return self._new(sym, "SELL", "MARKET", float(qty), cid, "FILLED", quote=float(qty) * px)

    def stop_loss(self, sym, qty, stop, cid):
        self.calls.append(("stop_loss", sym, float(qty), float(stop), cid))
        self._maybe_fail("stop_loss")
        base = sym[:-4]
        if self.bal.get(base, 0.0) + 1e-12 < float(qty):
            raise X.ExchangeError(400, -2010, "Account has insufficient balance", "/api/v3/order")
        self.bal[base] -= float(qty)
        self.locked[base] = self.locked.get(base, 0.0) + float(qty)
        return self._new(sym, "SELL", "STOP_LOSS", float(qty), cid, "NEW", stop=float(stop))

    def _unlock(self, o):
        base = o["symbol"][:-4]
        q = float(o["origQty"])
        self.locked[base] = self.locked.get(base, 0.0) - q
        self.bal[base] = self.bal.get(base, 0.0) + q

    def cancel(self, sym, order_id):
        self.calls.append(("cancel", sym, order_id))
        self._maybe_fail("cancel")
        o = self.orders[order_id]
        if o["status"] != "NEW":
            raise X.ExchangeError(400, -2011, "Unknown order sent.", "/api/v3/order")
        o["status"] = "CANCELED"
        self._unlock(o)
        return dict(o)

    def cancel_replace(self, sym, cancel_order_id, qty, stop, cid):
        self.calls.append(("cancel_replace", sym, cancel_order_id, float(qty), float(stop), cid))
        q = self.fail.get("cancel_replace")
        if q and q[0] == "new_failed":
            q.pop(0)
            o = self.orders[cancel_order_id]
            o["status"] = "CANCELED"
            self._unlock(o)
            e = X.ExchangeError(400, -2021, "Order cancel-replace partially failed.", "/api/v3/order/cancelReplace")
            e.data = {"cancelResult": "SUCCESS", "newOrderResult": "FAILURE"}
            raise e
        self._maybe_fail("cancel_replace")
        old = self.orders[cancel_order_id]
        old["status"] = "CANCELED"
        self._unlock(old)
        new = self.stop_loss(sym, qty, stop, cid)
        self.calls.pop()                                  # count the replace once
        return {"cancelResult": "SUCCESS", "newOrderResult": "SUCCESS", "cancelResponse": dict(old),
                "newOrderResponse": new}

    def get_order(self, sym, order_id=None, cid=None):
        self.calls.append(("get_order", sym, order_id, cid))
        for o in self.orders.values():
            if o["symbol"] == sym and (o["orderId"] == order_id or (cid is not None and o["clientOrderId"] == cid)):
                return dict(o)
        raise X.ExchangeError(400, -2013, "Order does not exist.", "/api/v3/order")

    def open_orders(self):
        return [dict(o) for o in self.orders.values() if o["status"] == "NEW"]

    # ----- market moves (tests only)
    def trigger(self, sym, price):
        self.px[sym] = price
        for o in self.orders.values():
            if o["symbol"] == sym and o["type"] == "STOP_LOSS" and o["status"] == "NEW" and price <= float(o["stopPrice"]):
                o["status"], o["executedQty"] = "FILLED", o["origQty"]
                base = sym[:-4]
                q = float(o["origQty"])
                self.locked[base] -= q
                self.bal["USDT"] += q * price * (1 - self.fee)
                o["cummulativeQuoteQty"] = str(q * price)

    def n(self, method):
        return sum(1 for c in self.calls if c[0] == method)


class StubPaper:
    """The PaperBook interface of spec §2.3 with scripted actions."""

    def __init__(self):
        self.queue, self.stops = [], {}

    def process(self, events, now):
        out, self.queue = self.queue, []
        return out

    def stop_of(self, key):
        return self.stops.get(key)

    def to_state(self):
        return {"stops": self.stops}

    def load(self, st):
        self.stops = dict((st or {}).get("stops", {}))

    def enter(self, sym="ABCUSDT", tf="15m", ts="2026-10-14T00:45:00+00:00", price=2.0, notional=1000.0, stop=1.84):
        key = f"{sym}|{tf}|{ts}"
        self.queue.append({"type": "entry", "key": key, "sym": sym, "tf": tf, "ts": ts, "price": price,
                           "notional": notional})
        self.stops[key] = stop
        return key

    def exit(self, key, reason="trail"):
        sym, tf, ts = key.split("|")
        self.queue.append({"type": "exit", "key": key, "sym": sym, "tf": tf, "ts": ts, "reason": reason})


def make(prices=None, filters=None, cfg=None, usdt=5000.0):
    import executor as E
    E.FC_STATE = Path(__file__).with_name("_no_fc_state_in_tests.json")   # never the live runtime/fc_demo state
    paper, ex, w = StubPaper(), FakeOrders(prices or {"ABCUSDT": 2.0}, filters, usdt=usdt), W()
    return E, paper, ex, w, E.Executor(paper, ex, cfg or CFG(), w)


def _iso(v):
    return v if isinstance(v, datetime) else datetime.fromisoformat(v)


def closed(exe, key):
    for p in exe.closed:
        if p["key"] == key:
            return p
    return None


# ======================================================================================== §7.1–7.3

class TestEntryAndProtection(unittest.TestCase):
    def test_buy_scaled_and_stop_in_same_cycle(self):
        E, paper, ex, w, exe = make()
        key = paper.enter(notional=1000.0, stop=1.84)
        exe.cycle([], NOW)
        buys = [c for c in ex.calls if c[0] == "market_buy"]
        self.assertEqual(len(buys), 1)
        self.assertAlmostEqual(buys[0][2], 20.0)                       # 1000 x 0.02
        stops = [c for c in ex.calls if c[0] == "stop_loss"]
        self.assertEqual(len(stops), 1)
        self.assertAlmostEqual(stops[0][2], 9.99)                      # free balance after the 0.1% fee, step 0.01
        self.assertAlmostEqual(stops[0][3], 1.7848)                    # 1.84 x 0.97 rounded down to tick
        p = exe.positions[key]
        self.assertEqual(p["state"], "PROTECTED")
        self.assertLessEqual((_iso(p["protected_at"]) - _iso(p["filled_at"])).total_seconds(), 60)
        rec = [r for r in w.kind("real_entry") if r.get("action") == "enter"]
        self.assertEqual(len(rec), 1)
        self.assertEqual(len(rec[0]["client_order_ids"]), 2)           # buy + stop
        for cid in rec[0]["client_order_ids"]:
            self.assertTrue(cid.startswith("cb_"))
            self.assertLessEqual(len(cid), 36)

    def test_below_min_notional_skipped(self):
        E, paper, ex, w, exe = make()
        paper.enter(notional=250.0)                                    # 5 USDT < max(5 x 1.2, 6)
        exe.cycle([], NOW)
        self.assertEqual(ex.n("market_buy"), 0)
        self.assertEqual(w.kind("real_entry")[-1]["reason"], "below_min_notional")

    def test_protection_fails_three_times_then_market_sell(self):
        E, paper, ex, w, exe = make()
        err = X.ExchangeError(400, -1013, "Filter failure", "/api/v3/order")
        ex.fail["stop_loss"] = [err, err, err]
        key = paper.enter()
        exe.cycle([], NOW)
        self.assertEqual(ex.n("stop_loss"), 3)
        self.assertEqual(ex.n("market_sell"), 1)
        self.assertIsNotNone(closed(exe, key))
        self.assertIn("protection_failed", [a.get("code") for a in w.kind("alert")])

    def test_lost_reply_is_looked_up_not_repeated(self):
        E, paper, ex, w, exe = make()
        ex.lost_reply.add("market_buy")
        key = paper.enter()
        exe.cycle([], NOW)
        self.assertEqual(ex.n("market_buy"), 1)
        self.assertGreaterEqual(ex.n("get_order"), 1)
        self.assertEqual(exe.positions[key]["state"], "PROTECTED")

    def test_partial_fill_protects_actual_balance(self):
        E, paper, ex, w, exe = make()
        ex.partial = 0.5
        key = paper.enter()
        exe.cycle([], NOW)
        stop = [c for c in ex.calls if c[0] == "stop_loss"][0]
        self.assertAlmostEqual(stop[2], 4.99)                          # 5 x 0.999 rounded down
        self.assertEqual(exe.positions[key]["state"], "PROTECTED")

    def test_one_real_position_per_asset(self):
        E, paper, ex, w, exe = make()
        paper.enter(tf="15m")
        paper.enter(tf="1h", ts="2026-10-14T00:00:00+00:00")
        exe.cycle([], NOW)
        self.assertEqual(ex.n("market_buy"), 1)
        self.assertIn("asset_held", [r.get("reason") for r in w.kind("real_entry")])


# ======================================================================================== §7.6–7.9

class TestTrailAndExits(unittest.TestCase):
    def test_trail_update_threshold(self):
        E, paper, ex, w, exe = make()
        key = paper.enter(stop=1.84)
        exe.cycle([], NOW)
        paper.stops[key] = 1.842                                       # +0.11%: no request
        exe.cycle([], NOW + timedelta(minutes=15))
        self.assertEqual(ex.n("cancel_replace"), 0)
        paper.stops[key] = 1.846                                       # +0.33%: one replace
        exe.cycle([], NOW + timedelta(minutes=30))
        self.assertEqual(ex.n("cancel_replace"), 1)
        self.assertAlmostEqual(exe.positions[key]["stop_price"], floor_step(1.846 * 0.97, "0.0001"))
        paper.stops[key] = 1.80                                        # the trail never moves down
        exe.cycle([], NOW + timedelta(minutes=45))
        self.assertEqual(ex.n("cancel_replace"), 1)

    def test_replace_new_leg_failed_places_stop_separately(self):
        E, paper, ex, w, exe = make()
        key = paper.enter(stop=1.84)
        exe.cycle([], NOW)
        ex.fail["cancel_replace"] = ["new_failed"]
        paper.stops[key] = 1.86
        exe.cycle([], NOW + timedelta(minutes=15))
        self.assertEqual(exe.positions[key]["state"], "PROTECTED")
        self.assertEqual(len(ex.open_orders()), 1)
        self.assertEqual(ex.n("stop_loss"), 2)

    def test_policy_exit_cancels_then_sells_and_records_dust(self):
        E, paper, ex, w, exe = make(filters={"ABCUSDT": filt(step="0.1", min_qty="0.1")})
        key = paper.enter()
        exe.cycle([], NOW)
        paper.exit(key, "trail")
        exe.cycle([], NOW + timedelta(minutes=15))
        kinds = [c[0] for c in ex.calls]
        self.assertLess(kinds.index("cancel"), kinds.index("market_sell"))
        sell = [c for c in ex.calls if c[0] == "market_sell"][0]
        self.assertAlmostEqual(sell[2], 9.9)                           # 9.99 held, step 0.1
        c = closed(exe, key)
        self.assertEqual(c["exit_reason"], "trail")
        self.assertAlmostEqual(c["dust_qty"], 0.09, places=6)
        self.assertEqual(ex.open_orders(), [])

    def test_time_exit_is_a_policy_exit(self):
        E, paper, ex, w, exe = make()
        key = paper.enter()
        exe.cycle([], NOW)
        paper.exit(key, "time")
        exe.cycle([], NOW + timedelta(hours=24))
        self.assertEqual(closed(exe, key)["exit_reason"], "time")
        self.assertEqual(ex.n("market_sell"), 1)

    def test_exchange_stop_fired_closes_without_touching_paper(self):
        E, paper, ex, w, exe = make()
        key = paper.enter()
        exe.cycle([], NOW)
        ex.trigger("ABCUSDT", 1.70)
        exe.cycle([], NOW + timedelta(seconds=20))
        c = closed(exe, key)
        self.assertEqual(c["exit_reason"], "exchange_stop")
        self.assertIn(key, paper.stops)                                # the paper book is not adjusted
        self.assertTrue(any(r.get("key") == key for r in w.kind("execution_divergence")))
        paper.exit(key, "trail")                                       # the paper exit comes later
        exe.cycle([], NOW + timedelta(minutes=15))
        self.assertEqual(ex.n("market_sell"), 0)
        self.assertIn("already_closed", [r.get("reason") for r in w.kind("real_exit")])

    def test_stop_cancelled_outside_is_replaced_with_alert(self):
        E, paper, ex, w, exe = make()
        key = paper.enter()
        exe.cycle([], NOW)
        ex.cancel("ABCUSDT", exe.positions[key]["stop_order_id"])
        exe.cycle([], NOW + timedelta(seconds=20))
        self.assertEqual(len(ex.open_orders()), 1)
        self.assertIn("stop_missing", [a.get("code") for a in w.kind("alert")])


# ======================================================================================== §7.11, §7.13

class TestRiskAndControl(unittest.TestCase):
    def test_exposure_cap(self):
        syms = [f"S{i:02d}USDT" for i in range(13)]
        E, paper, ex, w, exe = make(prices={s: 2.0 for s in syms})
        for s in syms:
            paper.enter(sym=s, stop=1.84)
        exe.cycle([], NOW)
        self.assertEqual(ex.n("market_buy"), 12)                       # 12 x 20 = 240; the 13th would be 260
        self.assertIn("exposure_cap", [r.get("reason") for r in w.kind("real_entry")])

    def test_daily_loss_limit_until_utc_midnight(self):
        syms = ["AAUSDT", "BBUSDT", "CCUSDT", "DDUSDT"]
        E, paper, ex, w, exe = make(prices={s: 2.0 for s in syms})
        keys = [paper.enter(sym=s, stop=1.0) for s in syms[:3]]
        exe.cycle([], NOW)
        for s, k in zip(syms[:3], keys):
            ex.px[s] = 1.0                                             # -50% each: ~ -10 USDT x 3
            paper.exit(k, "trail")
        exe.cycle([], NOW + timedelta(minutes=15))
        paper.enter(sym="DDUSDT", ts="2026-10-14T01:30:00+00:00")
        exe.cycle([], NOW + timedelta(minutes=30))
        self.assertEqual(ex.n("market_buy"), 3)
        self.assertIn("daily_loss_limit_real", [r.get("reason") for r in w.kind("real_entry")])
        paper.enter(sym="DDUSDT", ts="2026-10-15T00:15:00+00:00")
        exe.cycle([], datetime(2026, 10, 15, 0, 20, tzinfo=timezone.utc))
        self.assertEqual(ex.n("market_buy"), 4)

    def test_kill_switch_flattens_and_pauses(self):
        E, paper, ex, w, exe = make()
        with tempfile.TemporaryDirectory() as d:
            ctl = Path(d)
            key = paper.enter()
            exe.cycle([], NOW, control_dir=ctl)
            ex.bal["USDT"] -= 70.0                                     # equity falls 70 USDT below its peak
            exe.cycle([], NOW + timedelta(minutes=1), control_dir=ctl)
            self.assertIsNotNone(closed(exe, key))
            self.assertTrue((ctl / "PAUSE").exists())
            self.assertIn("kill_switch", [a.get("code") for a in w.kind("alert")])

    def test_control_files(self):
        E, paper, ex, w, exe = make(prices={"AAUSDT": 2.0, "BBUSDT": 2.0, "CCUSDT": 2.0})
        with tempfile.TemporaryDirectory() as d:
            ctl = Path(d)
            (ctl / "PAUSE").write_text("", encoding="utf-8")
            paper.enter(sym="AAUSDT")
            exe.cycle([], NOW, control_dir=ctl)
            self.assertEqual(ex.n("market_buy"), 0)
            self.assertIn("paused", [r.get("reason") for r in w.kind("real_entry")])
            (ctl / "PAUSE").unlink()
            k1, k2 = paper.enter(sym="BBUSDT"), paper.enter(sym="CCUSDT")
            exe.cycle([], NOW + timedelta(minutes=1), control_dir=ctl)
            self.assertEqual(ex.n("market_buy"), 2)
            (ctl / "FLATTEN").write_text("", encoding="utf-8")
            exe.cycle([], NOW + timedelta(minutes=2), control_dir=ctl)
            self.assertIsNotNone(closed(exe, k1))
            self.assertIsNotNone(closed(exe, k2))
            self.assertEqual(ex.open_orders(), [])
            self.assertFalse((ctl / "FLATTEN").exists())
            self.assertTrue((ctl / "PAUSE").exists())
            (ctl / "RESET_NOTED").write_text("", encoding="utf-8")
            exe.cycle([], NOW + timedelta(minutes=3), control_dir=ctl)
            self.assertFalse((ctl / "RESET_NOTED").exists())
            self.assertIn("reset_noted", [r.get("action") for r in w.kind("control")])


# ======================================================================================== §7.14–7.16

class TestStateLockJournal(unittest.TestCase):
    def test_restart_restores_and_reconciles(self):
        import executor as E
        paper, ex, w = StubPaper(), FakeOrders({"ABCUSDT": 2.0}), W()
        exe = E.Executor(paper, ex, CFG(), w)
        key = paper.enter()
        exe.cycle([], NOW)
        st = json.loads(json.dumps(exe.to_state(), default=str))
        ex.cancel("ABCUSDT", exe.positions[key]["stop_order_id"])     # lost while the executor was down
        exe2 = E.Executor(paper, ex, CFG(), w, state=st)
        self.assertIn(key, exe2.positions)
        exe2.cycle([], NOW + timedelta(minutes=10))
        self.assertEqual(len(ex.open_orders()), 1)
        self.assertEqual(exe2.positions[key]["state"], "PROTECTED")

    def test_single_instance_lock(self):
        import executor as E
        with tempfile.TemporaryDirectory() as d:
            lock = Path(d) / "executor.lock"
            self.assertTrue(E.acquire_lock(lock, 111, is_alive=lambda pid: False))
            self.assertFalse(E.acquire_lock(lock, 222, is_alive=lambda pid: pid == 111))
            self.assertTrue(E.acquire_lock(lock, 333, is_alive=lambda pid: False))    # stale lock taken over

    def test_client_order_ids_deterministic_and_unique(self):
        import executor as E
        a = E.client_order_id("t1013a01", "ABCUSDT|15m|2026-10-14T00:45:00+00:00", "b")
        self.assertEqual(a, E.client_order_id("t1013a01", "ABCUSDT|15m|2026-10-14T00:45:00+00:00", "b"))
        seen = {E.client_order_id("t1013a01", f"{s}|{tf}|2026-10-14T00:45:00+00:00", leg)
                for s in ("ABCUSDT", "VERYLONGNAMEUSDT") for tf in ("15m", "1h") for leg in ("b", "s1", "s2", "x")}
        self.assertEqual(len(seen), 16)
        for c in seen:
            self.assertTrue(c.startswith("cb_"))
            self.assertLessEqual(len(c), 36)


# ======================================================================================== §7.12 parity

class TestPaperParity(unittest.TestCase):
    def test_paper_book_equals_shadow_x4(self):
        import executor as E
        import shadow_portfolio as S
        from test_execution_replay import T0, STEP
        from test_shadow_portfolio import two_sym_grid, grid_events, write_events
        g = two_sym_grid()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ev.jsonl"
            write_events(p, grid_events(g, n_trades=40, seed=11))
            evs = S.EventTail(p, {}).read_new()
            end = T0 + (g.N - 1) * STEP
            eng = S.ShadowEngine({"x4_floor8": X4}, FM.model("taker", 1), floor_of=lambda m: 0.0,
                                 bars=S.GridBars(g), decisions=None)
            eng.process(evs, end)
            pb = E.PaperBook(X4, FM.model("taker", 1), S.GridBars(g), floor_of=lambda m: 0.0)
            acts = pb.process(evs, end)
            ents = sorted(a["key"] for a in acts if a["type"] == "entry")
            ref_ents = sorted(f"{r['sym']}|{r['tf']}|{r['ts']}" for r in eng.ledger if r["type"] == "entry")
            self.assertEqual(ents, ref_ents)
            exits = sorted((a["key"], a["reason"]) for a in acts if a["type"] == "exit")
            ref_ex = sorted((f"{r['sym']}|{r['tf']}|{r['entry_ts']}", r["reason"]) for r in eng.ledger if r["type"] == "exit")
            self.assertEqual(exits, ref_ex)
            self.assertTrue(ents)

    def test_every_paper_entry_is_bought_or_skipped_with_reason(self):
        import executor as E
        import shadow_portfolio as S
        from test_execution_replay import T0, STEP
        from test_shadow_portfolio import two_sym_grid, grid_events, write_events
        g = two_sym_grid()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ev.jsonl"
            write_events(p, grid_events(g, n_trades=20, seed=5))
            evs = S.EventTail(p, {}).read_new()
            pb = E.PaperBook(X4, FM.model("taker", 1), S.GridBars(g), floor_of=lambda m: 0.0)
            ex, w = FakeOrders({"AUSDT": 100.0, "BUSDT": 100.0}), W()
            exe = E.Executor(pb, ex, CFG(), w)
            exe.cycle(evs, T0 + (g.N - 1) * STEP)
            n_paper = len([r for r in pb.engine.ledger if r["type"] == "entry"])
            self.assertEqual(len(w.kind("real_entry")), n_paper)
            for r in w.kind("real_entry"):
                self.assertTrue(r.get("action") == "enter" or r.get("reason"))


# ======================================================================================== client (§2.3)

class TestClientOrders(unittest.TestCase):
    def _client(self, enabled=True, mode="demo", opener=None, audit=None):
        return X.ExchangeClient(key="KEY_TEST_0123", secret="SECRET_TEST_4567", cfg=types.SimpleNamespace(TRADING_ENABLED=enabled, TRADING_MODE=mode),
                                opener=opener or (lambda *a, **k: FakeResp({})), audit=audit)

    def test_writes_gated_reads_not(self):
        for enabled, mode in ((False, "demo"), (True, "paper")):
            c = self._client(enabled, mode)
            for call in (lambda: c.place_order("ABCUSDT", "BUY", "MARKET", quoteOrderQty="20"),
                         lambda: c.place_oco("ABCUSDT", "SELL", quantity="1"),
                         lambda: c.cancel_order("ABCUSDT", order_id=1),
                         lambda: c.cancel_order_list("ABCUSDT", order_list_id=1),
                         lambda: c.cancel_replace("ABCUSDT", cancel_order_id=1, type_="STOP_LOSS", quantity="1",
                                                  stopPrice="1.0")):
                with self.assertRaises(X.TradingDisabled):
                    call()

        def opener(req, timeout=None):
            return FakeResp({"serverTime": 1} if "/time" in req.full_url else {"orderId": 5, "status": "NEW"})
        c = self._client(False, "demo", opener=opener)
        self.assertEqual(c.get_order("ABCUSDT", order_id=5)["orderId"], 5)

    def test_request_shapes(self):
        seen = []

        def opener(req, timeout=None):
            if "/time" in req.full_url:
                return FakeResp({"serverTime": 1})
            seen.append((req.get_method(), req.full_url, (req.data or b"").decode()))
            return FakeResp({"orderId": 7, "clientOrderId": "cb_x", "status": "FILLED"})
        c = self._client(opener=opener)
        c.place_order("ABCUSDT", "BUY", "MARKET", quoteOrderQty="20", newClientOrderId="cb_x")
        m, url, body = seen[-1]
        self.assertEqual(m, "POST")
        self.assertTrue(url.endswith("/api/v3/order"))
        for part in ("type=MARKET", "quoteOrderQty=20", "newClientOrderId=cb_x", "newOrderRespType=FULL"):
            self.assertIn(part, body)
        c.cancel_replace("ABCUSDT", cancel_order_id=7, type_="STOP_LOSS", quantity="9.99", stopPrice="1.7848",
                         newClientOrderId="cb_y")
        m, url, body = seen[-1]
        self.assertTrue(url.endswith("/api/v3/order/cancelReplace"))
        for part in ("cancelReplaceMode=STOP_ON_FAILURE", "cancelOrderId=7", "type=STOP_LOSS", "stopPrice=1.7848",
                     "side=SELL"):
            self.assertIn(part, body)
        c.cancel_order("ABCUSDT", order_id=7)
        self.assertEqual(seen[-1][0], "DELETE")

    def test_error_data_kept_and_audited(self):
        data = {"cancelResult": "SUCCESS", "newOrderResult": "FAILURE"}

        def opener(req, timeout=None):
            if "/time" in req.full_url:
                return FakeResp({"serverTime": 1})
            raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(json.dumps(
                {"code": -2021, "msg": "Order cancel-replace partially failed.", "data": data}).encode()))
        w = W()
        c = self._client(opener=opener, audit=w)
        with self.assertRaises(X.ExchangeError) as cm:
            c.cancel_replace("ABCUSDT", cancel_order_id=7, type_="STOP_LOSS", quantity="1", stopPrice="1")
        self.assertEqual(cm.exception.data, data)
        self.assertEqual(w.rows[-1].get("error_data"), data)

    def test_orders_adapter_rounds_by_filters(self):
        seen = []

        def opener(req, timeout=None):
            if "/time" in req.full_url:
                return FakeResp({"serverTime": 1})
            seen.append((req.data or b"").decode())
            return FakeResp({"orderId": 8, "status": "NEW"})
        xinfo = {"symbols": [{"symbol": "ABCUSDT", "quoteAssetPrecision": 8,
                              "filters": list(filt().values())}]}
        o = X.Orders(self._client(opener=opener), xinfo)
        o.stop_loss("ABCUSDT", 9.987, 1.78489, "cb_s")
        self.assertIn("quantity=9.98", seen[-1])
        self.assertIn("stopPrice=1.7848", seen[-1])
        o.market_buy("ABCUSDT", 20.123456789, "cb_b")
        self.assertIn("quoteOrderQty=20.12345678", seen[-1])


# ======================================================================================== prices per pair (§2.3, after C0)

class TestPricePerPair(unittest.TestCase):
    """C0 2026-10-13: bookTicker over ALL pairs took 4.6-4.8 s; the executor asks only for the pairs it needs."""

    def _opener(self, seen, balances=None):
        import urllib.parse as U

        def opener(req, timeout=None):
            url = req.full_url
            if "/time" in url:
                return FakeResp({"serverTime": 1})
            seen.append(url)
            if "/api/v3/account" in url:
                return FakeResp({"balances": balances or []})
            q = U.parse_qs(U.urlparse(url).query)
            if "symbol" in q:
                return FakeResp({"symbol": q["symbol"][0], "bidPrice": "1.99", "askPrice": "2.01"})
            if "symbols" in q:
                return FakeResp([{"symbol": s, "bidPrice": "9", "askPrice": "11"} for s in json.loads(q["symbols"][0])])
            return FakeResp([{"symbol": "ALLUSDT", "bidPrice": "1", "askPrice": "1"}])
        return opener

    def _all_pairs_requests(self, seen):
        return [u for u in seen if "/ticker/bookTicker" in u and "symbol" not in u]

    def test_client_urls(self):
        seen = []
        c = X.ExchangeClient(key="KEY_TEST_0123", secret="SECRET_TEST_4567", opener=self._opener(seen))
        self.assertEqual(c.book_ticker("ABCUSDT")["symbol"], "ABCUSDT")
        self.assertIn("/api/v3/ticker/bookTicker?symbol=ABCUSDT", seen[-1])
        c.book_tickers(symbols=["AUSDT", "BUSDT"])
        self.assertIn("symbols=%5B%22AUSDT%22%2C%22BUSDT%22%5D", seen[-1])         # ["AUSDT","BUSDT"], no spaces

    def test_orders_price_one_pair_cached(self):
        seen = []
        c = X.ExchangeClient(key="KEY_TEST_0123", secret="SECRET_TEST_4567", opener=self._opener(seen))
        o = X.Orders(c, {"symbols": [{"symbol": "ABCUSDT", "filters": list(filt().values())}]})
        self.assertAlmostEqual(o.price("ABCUSDT"), 2.0)
        self.assertAlmostEqual(o.price("ABCUSDT"), 2.0)
        self.assertEqual(len([u for u in seen if "bookTicker" in u]), 1)
        self.assertEqual(self._all_pairs_requests(seen), [])

    def test_orders_equity_one_request_known_pairs_only(self):
        seen = []
        bal = [{"asset": "USDT", "free": "100", "locked": "0"}, {"asset": "ETH", "free": "1", "locked": "0"},
               {"asset": "BTC", "free": "0", "locked": "2"}, {"asset": "XYZ", "free": "5", "locked": "0"}]
        c = X.ExchangeClient(key="KEY_TEST_0123", secret="SECRET_TEST_4567", opener=self._opener(seen, bal))
        xinfo = {"symbols": [{"symbol": s, "filters": list(filt().values())} for s in ("BTCUSDT", "ETHUSDT")]}
        eq = X.Orders(c, xinfo).equity()
        self.assertAlmostEqual(eq, 100 + 1 * 10 + 2 * 10)
        bt = [u for u in seen if "bookTicker" in u]
        self.assertEqual(len(bt), 1)
        self.assertIn("symbols=%5B%22BTCUSDT%22%2C%22ETHUSDT%22%5D", bt[0])        # XYZ has no pair: not asked
        self.assertEqual(self._all_pairs_requests(seen), [])

    def test_paper_price_fn_per_pair(self):
        import executor as E
        seen = []
        c = X.ExchangeClient(key="", secret="", opener=self._opener(seen))
        f = E.pair_price_fn(c, ttl=5.0)
        self.assertAlmostEqual(f("ABCUSDT"), 2.0)
        self.assertAlmostEqual(f("ABCUSDT"), 2.0)
        self.assertAlmostEqual(f("DEFUSDT"), 2.0)
        self.assertEqual(len([u for u in seen if "bookTicker" in u]), 2)
        self.assertEqual(self._all_pairs_requests(seen), [])


# ======================================================================================== C1 paper exchange

class TestPaperExchange(unittest.TestCase):
    def test_buy_stop_trigger(self):
        import executor as E
        px = {"ABCUSDT": 2.0}
        pe = E.PaperExchange(lambda s: px[s], FM.model("taker", 1), {"ABCUSDT": filt()}, usdt=5000.0)
        b = pe.market_buy("ABCUSDT", 20.0, "cb_b")
        self.assertEqual(b["status"], "FILLED")
        q = pe.free("ABC")
        self.assertGreater(q, 9.9)
        s = pe.stop_loss("ABCUSDT", floor_step(q, "0.01"), 1.7848, "cb_s")
        self.assertEqual(pe.open_orders()[0]["orderId"], s["orderId"])
        px["ABCUSDT"] = 1.70
        pe.on_prices()
        self.assertEqual(pe.get_order("ABCUSDT", order_id=s["orderId"])["status"], "FILLED")
        self.assertEqual(pe.open_orders(), [])


# ======================================================================================== §7 boundary

class TestBoundary(unittest.TestCase):
    def test_runtime_only_no_telegram(self):
        import executor as E
        import trader_config as TC
        for name in ("executor.py", "reconcile.py"):
            src = (HERE / name).read_text(encoding="utf-8")
            self.assertNotIn('"' + '.runtime' + '"', src)
            self.assertNotIn("telegram", src.lower())
        self.assertTrue(str(E.EXEC_DIR).startswith(str(TC.RUNTIME)))

    def test_config_flags_exist_and_trading_off(self):
        """C2 since 2026-10-20 (operator): trading may be on, but only on the demo account."""
        import trader_config as TC
        self.assertIn(TC.TRADING_MODE, ("paper", "demo"))           # mainnet is not a value
        if TC.TRADING_ENABLED:
            self.assertEqual(TC.TRADING_MODE, "demo")
        for k in ("TRADING_SCALE", "TRADING_MIN_ORDER_USDT", "TRADING_MAX_EXPOSURE_USDT", "TRADING_DAILY_LOSS_LIMIT_USDT",
                  "TRADING_KILL_SWITCH_USDT", "TRADING_CATASTROPHE_GAP_PCT", "TRADING_TRAIL_UPDATE_MIN_PCT",
                  "TRADING_MAX_UNPROTECTED_SEC", "TRADING_POLL_SEC", "TRADING_PROTECT_RETRIES", "TRADING_RECONCILE_SEC",
                  "TRADING_LOOP_SEC", "TRADING_EPOCH"):
            self.assertTrue(hasattr(TC, k), k)
        self.assertEqual(TC.TRADING_SCALE, 0.02)
        self.assertEqual(TC.TRADING_CATASTROPHE_GAP_PCT, 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
