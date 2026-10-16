"""E7 -- shadow portfolios on live data: P0 (bot's own exits) vs P4 (X4, P-1 trail with an 8% floor).

e7-shadow-portfolio-1012-spec.md. Paper only: no key, no order, no signed request. Reads the tail of
files/bot_events.jsonl (read-only) and public 15m klines; writes only trader/runtime/shadow/.

The engine mirrors execution_replay.run step by step (spec §5, tested for parity): the same entry
decision (trading_policy.decide_entry), the same fills (fill_model), the same equity marks at 00:00
UTC, and for P4 the P-1 trail fed bar by bar (P1Exit == trading_policy.exit_plan).

  (from trader/) ../pyembed/python.exe shadow_portfolio.py step
"""
from __future__ import annotations

import collections
import io
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402
import fill_model as FM           # noqa: E402
import trading_policy as TP       # noqa: E402
import audit as A                 # noqa: E402

STEP = timedelta(minutes=15)
SHADOW_DIR = TC.RUNTIME / "shadow"
STATE = SHADOW_DIR / "state.json"
LEDGER = SHADOW_DIR / "ledger.jsonl"
T0_LIVE = datetime(2026, 1, 1, tzinfo=timezone.utc)
SEEN_MAX = 50_000
X4 = replace(TP.CHAMPION, name="x4_floor8", exits=TP.Exits(kind="p1_trail", floor_override=0.08))
POLICIES = {"champion": TP.CHAMPION, "x4_floor8": X4}


def _dt(ts):
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _bar(t):
    return t.replace(minute=t.minute // 15 * 15, second=0, microsecond=0)


def closed_bars(bars, now):
    """Bars whose close time is not after `now` (spec §7.7)."""
    return [b for b in bars if b["close_time"] <= now]


# ---------------------------------------------------------------- state

def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def load_state(path: Path):
    """None if absent; ValueError if present but unreadable (never silently restart from zero)."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"shadow state {path} is corrupt: {e!r}") from None


# ---------------------------------------------------------------- event tail

class EventTail:
    """New complete `entry`/`exit` lines since the stored offset; survives rotation without duplicates."""

    def __init__(self, path: Path, state: dict):
        self.path, self.st = Path(path), state
        self.st.setdefault("offset", 0)
        self.st.setdefault("seen", [])

    def read_new(self) -> list:
        seen = set(tuple(k) for k in self.st["seen"])
        size = self.path.stat().st_size if self.path.exists() else 0
        if size < self.st["offset"]:
            self.st["offset"] = 0                         # rotated / truncated: re-read, dedup by key
        with open(self.path, "rb") as fh:
            fh.seek(self.st["offset"])
            data = fh.read()
        end = data.rfind(b"\n")
        if end < 0:
            return []
        chunk, self.st["offset"] = data[: end + 1], self.st["offset"] + end + 1
        out = []
        for raw in chunk.split(b"\n"):
            if b'"entry"' not in raw and b'"exit"' not in raw:
                continue
            try:
                e = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue
            if e.get("event") not in ("entry", "exit"):
                continue
            key = (e["event"], e.get("sym"), e.get("tf"), str(e.get("ts")))
            if key in seen:
                continue
            seen.add(key)
            self.st["seen"].append(list(key))
            out.append(e)
        if len(self.st["seen"]) > SEEN_MAX:
            self.st["seen"] = self.st["seen"][-SEEN_MAX:]
        return out


# ---------------------------------------------------------------- bars

class GridBars:
    """Bars from an execution_replay grid (tests, parity)."""

    def __init__(self, grid):
        self.g = grid

    def idx(self, dt):
        return self.g.idx(dt)

    def close_time(self, ix):
        return self.g.t0 + (ix + 1) * STEP

    def j(self, sym):
        return self.g.col.get(sym)

    def valid_ie(self, sym, ie):
        return self.j(sym) is not None and 100 <= ie < self.g.N

    def last_closed(self, now):
        return min(self.g.N - 1, self.idx(now) - 1)

    def _v(self, arr, sym, ix):
        j = self.j(sym)
        if j is None or not (0 <= ix < self.g.N):
            return float("nan")
        return float(arr[ix, j])

    def close(self, sym, ix):
        return self._v(self.g.C, sym, ix)

    def atr(self, sym, ix):
        return self._v(self.g.ATR, sym, ix)

    def vol(self, sym, ix):
        return self._v(self.g.V, sym, ix)

    def refresh(self, syms, now):
        pass


class LiveBars:
    """Closed 15m bars from public Binance klines; global index from T0_LIVE; ATR by the bot's _atr."""

    def __init__(self, client, atr_period: int = 14, limit: int = 500):
        self.client, self.period, self.limit = client, atr_period, limit
        self.data = {}            # sym -> {ix: (close, atr, vol)}
        self.now = None

    def idx(self, dt):
        return int((dt - T0_LIVE) / STEP)

    def close_time(self, ix):
        return T0_LIVE + (ix + 1) * STEP

    def j(self, sym):
        return 0 if sym in self.data else None

    def valid_ie(self, sym, ie):
        return sym in self.data and ie in self.data[sym]

    def last_closed(self, now):
        return self.idx(now) - 1

    def refresh(self, syms, now):
        import numpy as np
        TC.bot_readonly()
        from indicators import _atr                     # the bot's own ATR (read-only import)
        self.now = now
        for sym in sorted(set(syms)):
            try:
                rows = self.client.klines(sym, "15m", self.limit, base="https://api.binance.com")
            except Exception:
                continue
            rows = [r for r in rows if datetime.fromtimestamp((int(r[6]) + 1) / 1000, timezone.utc) <= now]
            if len(rows) < self.period + 2:
                continue
            h = np.array([float(r[2]) for r in rows])
            l = np.array([float(r[3]) for r in rows])
            c = np.array([float(r[4]) for r in rows])
            v = np.array([float(r[5]) for r in rows])
            a = _atr(h, l, c, self.period)
            self.data[sym] = {self.idx(datetime.fromtimestamp(int(r[0]) / 1000, timezone.utc)): (c[i], a[i], v[i])
                              for i, r in enumerate(rows)}

    def _get(self, sym, ix, k):
        x = self.data.get(sym, {}).get(ix)
        return float(x[k]) if x is not None else float("nan")

    def close(self, sym, ix):
        return self._get(sym, ix, 0)

    def atr(self, sym, ix):
        return self._get(sym, ix, 1)

    def vol(self, sym, ix):
        return self._get(sym, ix, 2)


# ---------------------------------------------------------------- incremental exit

class P1Exit:
    """trading_policy.exit_plan(p1_trail) fed one closed bar at a time (spec §7.2)."""

    def __init__(self, k, floor, c0, a0, ie, max_bars=96, hourly=False, stop=None, done=False):
        self.k, self.floor, self.c0, self.a0, self.ie = float(k), float(floor), float(c0), float(a0), int(ie)
        self.max_bars, self.hourly, self.done = int(max_bars), bool(hourly), bool(done)
        self.stop = float(stop) if stop is not None else self.c0 - max(self.k * self.a0, self.floor * self.c0)
        self.last = self.ie + self.max_bars

    def feed(self, q, cq, aq):
        """-> (exit_bar, price, reason, dist) or None. Bars must come in order, q > ie."""
        if self.done or q <= self.ie or q > self.last:
            return None
        if math.isfinite(cq) and not (self.hourly and (q + 1) % 4):
            if math.isfinite(aq):
                self.stop = max(self.stop, cq - max(self.k * aq, self.floor * cq))
            if cq < self.stop:
                self.done = True
                return q, float(cq), "trail", None
        if q == self.last:
            self.done = True
            return (q, float(cq), "time", None) if math.isfinite(cq) else None
        return None

    def to_dict(self):
        return {"k": self.k, "floor": self.floor, "c0": self.c0, "a0": self.a0, "ie": self.ie,
                "max_bars": self.max_bars, "hourly": self.hourly, "stop": self.stop, "done": self.done}


# ---------------------------------------------------------------- engine

class Book:
    def __init__(self, name, policy, start_equity=10_000.0):
        self.name, self.policy = name, policy
        self.cash, self.positions, self.closed = start_equity, [], []
        self.day_start_eq, self.cur_day = {}, None
        self.cooldown, self.day_count = {}, collections.Counter()


class ShadowEngine:
    def __init__(self, policies: dict, fm, floor_of=None, bars=None, start_equity=10_000.0, decisions=None):
        self.books = {n: Book(n, p, start_equity) for n, p in policies.items()}
        self.fm, self.floor_of, self.bars, self.ledger = fm, floor_of, bars, []
        # one audit line per decision (trader-audit-log-1012-spec.md §2.2); None = not logged
        self.decisions = decisions if decisions is not None else A.default_writer("decisions")

    def _dlog(self, b, kind, sym, tf, ts, **kw):
        if self.decisions is None:
            return
        try:
            tsv = ts.isoformat() if isinstance(ts, datetime) else str(ts)
            self.decisions.write({"kind": kind, "book": b.name, "policy_epoch": b.policy.epoch(),
                                  "decision_id": A.decision_id(b.name, sym, tf, tsv, kind),
                                  "sym": sym, "tf": tf, "event_ts": tsv, **kw})
        except Exception:
            pass

    def _skip(self, b, ev, t, reason, **kw):
        self._dlog(b, "entry_decision", ev.get("sym"), ev.get("tf"), t, action="skip", reason=reason,
                   candidate=self._cand_view(ev), **kw)

    @staticmethod
    def _cand_view(ev):
        return {k: ev.get(k) for k in ("price", "mode", "signal_mode", "trail_k", "ml_proba", "candidate_score",
                                        "ranker_final_score", "event") if ev.get(k) is not None}

    # ----- marks and closes (mirror execution_replay.run: close_due / mark / roll_days)
    def _mark(self, b, at_ix):
        v = b.cash
        for p in b.positions:
            c = self.bars.close(p["sym"], at_ix)
            if math.isfinite(c):
                p["last_px"] = c
            v += p["qty"] * p["last_px"]
        return v

    def _close(self, b, p, exit_time, exit_raw, reason, exit_ix):
        b.positions.remove(p)
        fee = self.fm.maker_fee if reason == "tp" else self.fm.taker_fee
        qv = None
        if exit_ix is not None:
            vv, cc = self.bars.vol(p["sym"], exit_ix), self.bars.close(p["sym"], exit_ix)
            qv = vv * cc if math.isfinite(vv) and math.isfinite(cc) else None
        notional = p["qty"] * exit_raw
        px = exit_raw if reason == "tp" else self.fm.sell(exit_raw, notional, qv)
        proceeds = p["qty"] * px * (1 - fee)
        b.cash += proceeds
        pnl = proceeds - p["cost"]
        t = {"sym": p["sym"], "mode": p["mode"], "tf": p["tf"], "entry_ts": p["entry_ts"], "exit_time": exit_time,
             "reason": reason, "cost": p["cost"], "pnl": pnl, "ret_pct": pnl / p["cost"] * 100,
             "gross_ret_pct": (exit_raw / p["entry_raw"] - 1) * 100, "kind": "entry"}
        b.closed.append(t)
        self._dlog(b, "exit_decision", p["sym"], p["tf"], p["entry_ts"], reason=reason, exit_time=exit_time,
                   exit_bar=exit_ix, price_raw=exit_raw, price_after_costs=px, entry_price=p["entry_raw"],
                   cost=p["cost"], pnl=pnl, ret_pct=t["ret_pct"],
                   hold_bars=(exit_ix - p["ie"]) if (exit_ix is not None and p.get("ie") is not None) else None)
        if exit_ix is not None:
            b.cooldown[p["sym"]] = exit_ix + b.policy.risk.cooldown_bars
        self.ledger.append({"type": "exit", "book": b.name, **{k: (v.isoformat() if isinstance(v, datetime) else v)
                                                              for k, v in t.items()}})

    def _advance(self, b, until):
        """Feed P-1 positions every closed bar with close time <= until; close the exits found."""
        last = self.bars.last_closed(until)
        due = []
        for p in list(b.positions):
            ex = p.get("p1")
            if ex is None:
                continue
            while p["next_q"] <= last and not ex.done:
                q = p["next_q"]
                p["next_q"] += 1
                r = ex.feed(q, self.bars.close(p["sym"], q), self.bars.atr(p["sym"], q))
                if r is not None:
                    due.append((self.bars.close_time(r[0]), p, r))
        for ct, p, r in sorted(due, key=lambda x: x[0]):
            self._close(b, p, ct, r[1], r[2], r[0])

    def _roll(self, b, until):
        if b.cur_day is None:
            return
        while True:
            nxt = b.cur_day + timedelta(days=1)
            if nxt > until:
                break
            self._advance(b, nxt)
            ix = self.bars.idx(nxt) - 1
            b.day_start_eq[nxt.strftime("%Y-%m-%d")] = self._mark(b, ix)
            self._dlog(b, "day_mark", None, None, nxt, day=nxt.strftime("%Y-%m-%d"),
                       equity=b.day_start_eq[nxt.strftime("%Y-%m-%d")], open=len(b.positions), cash=b.cash)
            b.cur_day = nxt
        self._advance(b, until)

    # ----- events
    def _enter(self, b, ev):
        t = _dt(ev["ts"])
        if b.cur_day is None:
            b.cur_day = t.replace(hour=0, minute=0, second=0, microsecond=0)
            b.day_start_eq[b.cur_day.strftime("%Y-%m-%d")] = b.cash
        price = ev.get("price")
        if not price:
            return self._skip(b, ev, t, "no_price")
        sym, tf = ev.get("sym"), ev.get("tf")
        ie = self.bars.idx(_bar(t)) - 1
        ok_ie = self.bars.valid_ie(sym, ie)
        pol = b.policy
        if not ok_ie and pol.exits.kind != "real":
            return self._skip(b, ev, t, "off_grid")
        dkey = t.strftime("%Y-%m-%d")
        eq_now = self._mark(b, ie)
        cand = {"kind": "entry", "sym": sym, "tf": tf, "ts": t, "price": float(price),
                "mode": ev.get("mode") or ev.get("signal_mode"), "trail_k": ev.get("trail_k"),
                **{k: ev.get(k) for k in ("ml_proba", "candidate_score", "ranker_final_score")}}
        state = {"open": len(b.positions), "open_syms": {p["sym"] for p in b.positions}, "day_count": b.day_count[dkey],
                 "day_return_pct": (eq_now / b.day_start_eq.get(dkey, eq_now) - 1) * 100,
                 "cooldown_until": b.cooldown, "bar": ie if ok_ie else 0, "equity": eq_now}
        frac, why = TP.decide_entry_explained(pol, cand, state)
        state_view = {"open": state["open"], "open_syms": sorted(state["open_syms"]), "day_count": state["day_count"],
                      "day_return_pct": state["day_return_pct"], "equity": eq_now,
                      "cooldown_until_sym": b.cooldown.get(sym), "bar": state["bar"]}
        if frac is None:
            return self._skip(b, ev, t, why, state=state_view)
        p1 = None
        if pol.exits.kind == "p1_trail":
            c0, a0 = self.bars.close(sym, ie), self.bars.atr(sym, ie)
            if not (math.isfinite(c0) and math.isfinite(a0)):
                return self._skip(b, ev, t, "no_bars", state=state_view)
            floor = pol.exits.floor_override if pol.exits.floor_override is not None else \
                (self.floor_of(cand["mode"]) if self.floor_of else 0.0)
            p1 = P1Exit(float(cand["trail_k"] or 2.0), floor, c0, a0, ie, pol.exits.max_bars, tf == "1h")
        notional = min(eq_now * frac, b.cash * (1 - pol.sizing.cash_reserve_pct / 100.0))
        if notional <= 1.0:
            return self._skip(b, ev, t, "notional_too_small", state=state_view)
        vv, cc = self.bars.vol(sym, ie), self.bars.close(sym, ie)
        qv = vv * cc if ok_ie and math.isfinite(vv) and math.isfinite(cc) else None
        fill = self.fm.buy(float(price), notional, qv)
        qty = notional / fill * (1 - self.fm.taker_fee)
        b.cash -= notional
        b.positions.append({"sym": sym, "mode": cand["mode"], "tf": tf, "qty": qty, "cost": notional,
                            "entry_raw": float(price), "entry_ts": t, "last_px": fill, "p1": p1, "next_q": ie + 1, "ie": ie})
        self._dlog(b, "entry_decision", sym, tf, t, action="enter", reason="enter", candidate=self._cand_view(ev),
                   state=state_view, frac=frac, notional=notional, fill=fill, qty=qty, fee=self.fm.taker_fee,
                   exit_plan={"kind": pol.exits.kind, "stop": p1.stop if p1 else None,
                              "floor": p1.floor if p1 else None})
        b.day_count[dkey] += 1
        self.ledger.append({"type": "entry", "book": b.name, "sym": sym, "tf": tf, "ts": t.isoformat(),
                            "price": float(price), "notional": notional})

    def _exit_real(self, b, ev):
        if b.policy.exits.kind != "real":
            return
        sym, tf, t = ev.get("sym"), ev.get("tf"), _dt(ev["ts"])
        pos = [p for p in b.positions if p["sym"] == sym and p["tf"] == tf and p.get("p1") is None]
        px = ev.get("exit_price")
        if not pos or not isinstance(px, (int, float)):
            self.ledger.append({"type": "orphan_exit", "book": b.name, "sym": sym, "tf": tf, "ts": t.isoformat()})
            self._dlog(b, "orphan_exit", sym, tf, t, reason="no_open_position" if not pos else "no_exit_price")
            return
        ix = self.bars.idx(_bar(t)) - 1
        self._close(b, pos[0], t, float(px), "real", ix if self.bars.valid_ie(sym, ix) else None)

    def process(self, events: list, now: datetime) -> None:
        syms = {e.get("sym") for e in events} | {p["sym"] for b in self.books.values() for p in b.positions}
        self.bars.refresh([s for s in syms if s], now)
        for ev in sorted(events, key=lambda e: _dt(e["ts"])):
            t = _dt(ev["ts"])
            for b in self.books.values():
                self._roll(b, t)
                if ev["event"] == "entry":
                    self._enter(b, ev)
                else:
                    self._exit_real(b, ev)
        for b in self.books.values():
            self._roll(b, now)

    # ----- views
    def trades(self, name):
        return list(self.books[name].closed)

    def marks(self, name):
        return sorted(self.books[name].day_start_eq.items())

    def series(self, name):
        m = self.marks(name)
        return [(a, (eb / ea - 1) * 100) for (a, ea), (_, eb) in zip(m, m[1:])]

    # ----- persistence
    def to_state(self) -> dict:
        out = {}
        for n, b in self.books.items():
            out[n] = {"cash": b.cash, "cur_day": b.cur_day.isoformat() if b.cur_day else None,
                      "day_start_eq": b.day_start_eq, "cooldown": b.cooldown, "day_count": dict(b.day_count),
                      "closed_n": len(b.closed),
                      "positions": [{**{k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in p.items() if k != "p1"},
                                     "p1": p["p1"].to_dict() if p.get("p1") else None} for p in b.positions]}
        return out

    def load(self, st: dict) -> None:
        for n, s in (st or {}).items():
            b = self.books.get(n)
            if b is None:
                continue
            b.cash, b.day_start_eq = float(s["cash"]), dict(s["day_start_eq"])
            b.cur_day = _dt(s["cur_day"]) if s.get("cur_day") else None
            b.cooldown, b.day_count = dict(s["cooldown"]), collections.Counter(s["day_count"])
            b.positions = []
            for p in s["positions"]:
                q = dict(p)
                q["entry_ts"] = _dt(q["entry_ts"])
                q["p1"] = P1Exit(**p["p1"]) if p.get("p1") else None
                b.positions.append(q)


# ---------------------------------------------------------------- step (live)

def step(now: datetime | None = None) -> dict:
    import exchange_client as X
    TC.bot_readonly()
    import config as BOT
    import exit_validator as EV
    now = now or datetime.now(timezone.utc)
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    st = load_state(STATE)
    first = st is None
    st = st or {"tail": {}, "books": None, "started_utc": now.isoformat()}
    tail = EventTail(TC.BOT_EVENTS, st["tail"])
    if first:                                             # start from NOW, never replay history
        tail.st["offset"] = TC.BOT_EVENTS.stat().st_size if TC.BOT_EVENTS.exists() else 0
    bars = LiveBars(X.ExchangeClient(key="", secret=""), atr_period=int(getattr(BOT, "ATR_PERIOD", 14)))
    eng = ShadowEngine(POLICIES, FM.model("taker", 1), floor_of=lambda m: EV.floor_of(m, BOT), bars=bars)
    if st.get("books"):
        eng.load(st["books"])
    for b in eng.books.values():                          # live: the day clock starts at the first step
        if b.cur_day is None:
            b.cur_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            b.day_start_eq.setdefault(b.cur_day.strftime("%Y-%m-%d"), b.cash)
    evs = tail.read_new()
    eng.process(evs, now)
    st["books"] = eng.to_state()
    st["last_step_utc"] = now.isoformat()
    if eng.ledger:
        with io.open(LEDGER, "a", encoding="utf-8") as fh:
            for r in eng.ledger:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    save_state(STATE, st)
    return {"events": len(evs), "ledger": len(eng.ledger),
            "open": {n: len(b.positions) for n, b in eng.books.items()},
            "equity_marks": {n: dict(sorted(b.day_start_eq.items())[-2:]) for n, b in eng.books.items()}}


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    A.set_process("shadow")
    if not getattr(TC, "SHADOW_PORTFOLIO_ENABLED", False):
        print("SHADOW_PORTFOLIO_ENABLED is False -- nothing done")
        return 0
    import faulthandler
    faulthandler.dump_traceback_later(120, exit=True)
    r = step()
    print(json.dumps(r, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
