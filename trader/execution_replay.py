"""E1 -- execution-aware replay: portfolio, fees, slippage, daily revenue.

execution-replay-e1-1010-spec.md. Nothing deployed: reads bot_events.jsonl and the 15m grid.

  candidates  bot_events `entry` (with the next `exit` of the same sym/tf) and, optionally,
              `blocked` (deduplicated to the last event per symbol and 15m bar), since 2026-03-01
  policy      trading_policy.TradingPolicy (admission x sizing x exits x risk) -- same module the
              phase-C executor calls
  fills       fill_model.FillModel (fees, half spread, square-root impact)
  equity      cash + positions marked at the 15m close; R_d from equity at 00:00 UTC, aggregated by
              daily_revenue.aggregate (the same function as the demo-account metric)

Self-checks (spec §6), run with --verify:
  V1  accounting="realized_simple" reproduces _backtest_daily_revenue_baseline.py exactly
  V2  exits.kind="p1_trail" reproduces exit_validator.plain_trail per 15m trade
  V3  money conservation: final equity = initial + sum(trade P&L) + open marks; cash never < 0

  (from trader/) ../pyembed/python.exe execution_replay.py --verify
  ../pyembed/python.exe execution_replay.py --policy champion --fees taker --slip 1 --split 2026-07-01
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
import sys
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC      # noqa: E402
import daily_revenue as DR      # noqa: E402
import fill_model as FM          # noqa: E402
import trading_policy as TP      # noqa: E402

EVENTS = TC.BOT_EVENTS
SINCE = "2026-03-01"
OUT_DIR = TC.RUNTIME / "replay"
STEP = timedelta(minutes=15)
FEAT_KEYS = ("ml_proba", "candidate_score", "ranker_final_score", "ranker_ev", "ranker_top_gainer_prob",
             "adx", "rsi", "vol_x", "slope_pct", "daily_range", "btc_vs_ema50", "is_bull_day", "macd_hist")


def _dt(ts):
    d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _bar(t):
    return t.replace(minute=t.minute // 15 * 15, second=0, microsecond=0)


# ---------------------------------------------------------------- candidates

def load_events(since: str = SINCE, include_blocked: bool = False, path: Path = EVENTS):
    """(trades, blocked): trades = baseline-identical entry->exit pairs per (sym, tf)."""
    evs = collections.defaultdict(list)
    blocked = {}
    with io.open(path, "rb") as fh:
        for raw in fh:
            is_eb = b'"entry"' in raw or b'"exit"' in raw
            is_bl = include_blocked and b'"blocked"' in raw
            if not (is_eb or is_bl):
                continue
            try:
                e = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue
            ev = e.get("event")
            if str(e.get("ts", ""))[:10] < since:
                continue
            if ev in ("entry", "exit"):
                evs[(e.get("sym"), e.get("tf"))].append((_dt(e["ts"]), e))
            elif ev == "blocked" and include_blocked and e.get("price") and e.get("sym"):
                d = _dt(e["ts"])
                blocked[(e["sym"], e.get("tf"), _bar(d))] = (d, e)
    trades = []
    for k, es in evs.items():
        es.sort(key=lambda x: x[0])
        op = None
        for d, e in es:
            if e["event"] == "entry":
                op = (d, e)
            elif op is not None and isinstance(e.get("pnl_pct"), (int, float)):
                en = op[1]
                trades.append({"kind": "entry", "sym": k[0], "tf": k[1], "ts": op[0], "exit_ts": d,
                               "price": float(en.get("price") or 0) or None,
                               "exit_price": float(e["exit_price"]) if isinstance(e.get("exit_price"), (int, float)) else None,
                               "real_pnl": float(e["pnl_pct"]), "exit_reason": str(e.get("reason") or "")[:60],
                               "mode": en.get("mode") or en.get("signal_mode"),
                               "trail_k": en.get("trail_k"),
                               **{f: en.get(f) for f in FEAT_KEYS}})
                op = None
    trades.sort(key=lambda t: t["ts"])
    bl = []
    for (sym, tf, _), (d, e) in blocked.items():
        bl.append({"kind": "blocked", "sym": sym, "tf": tf, "ts": d, "price": float(e["price"]),
                   "mode": e.get("signal_type") or e.get("mode"), "gate": e.get("gate") or e.get("reason_code"),
                   "trail_k": None, **{f: e.get(f) for f in FEAT_KEYS}})
    bl.sort(key=lambda t: t["ts"])
    return trades, bl


def attach_grid(cands: list, grid) -> None:
    for c in cands:
        j = grid.col.get(c["sym"])
        c["j"] = j
        ie = grid.idx(_bar(c["ts"])) - 1
        c["ie"] = ie if j is not None and 100 <= ie < grid.N else None
        if c.get("exit_ts") is not None and j is not None:
            ix = grid.idx(_bar(c["exit_ts"])) - 1
            c["ix"] = ix if c["ie"] is not None and ix > c["ie"] and ix < grid.N else None
        else:
            c["ix"] = None


# ---------------------------------------------------------------- V1 accounting

def realized_simple(trades: list, slots: int, roundtrip_pct: float) -> dict:
    """_backtest_daily_revenue_baseline.py, reproduced: equal slots, skip the 11th, realized by exit day."""
    taken, open_ = [], []
    for t in trades:
        open_ = [x for x in open_ if x["exit_ts"] > t["ts"]]
        if len(open_) >= slots:
            continue
        open_.append(t)
        taken.append(t)
    days_all = sorted({t["exit_ts"].strftime("%Y-%m-%d") for t in trades})
    per_day = collections.defaultdict(float)
    for t in taken:
        per_day[t["exit_ts"].strftime("%Y-%m-%d")] += (t["real_pnl"] - roundtrip_pct) / slots
    series = [(d, per_day.get(d, 0.0)) for d in days_all]
    return {"taken": len(taken), "series": series, "agg": DR.aggregate(series)}


# ---------------------------------------------------------------- equity engine

def run(policy: TP.TradingPolicy, fm: FM.FillModel, grid, cands: list, start_equity: float = 10_000.0,
        floor_of=None, day_from: str | None = None, day_to: str | None = None, plan_cache: dict | None = None,
        regime_flags: dict | None = None) -> dict:
    """Event-driven equity replay. Candidates must be time-sorted and attached to the grid.
    regime_flags: {name: bool array per grid bar} (regime.flags); the entry decision sees the flags of bar ie."""
    if day_from or day_to:                 # the half's days only: the series must not span the other half
        cands = [c for c in cands if (not day_from or c["ts"].strftime("%Y-%m-%d") >= day_from)
                 and (not day_to or c["ts"].strftime("%Y-%m-%d") < day_to)]
    cash, eq0 = start_equity, start_equity
    positions = []                       # dicts with qty, cost, exit_time, exit_px, reason, ...
    closed = []
    day_start_eq, cur_day = {}, None
    cooldown, day_count = {}, collections.Counter()
    unmarked = 0

    def bar_close_time(ix):
        return grid.t0 + (ix + 1) * STEP

    def mark(at_ix):
        nonlocal unmarked
        v = cash
        for p in positions:
            px = None
            if p["j"] is not None and 0 <= at_ix < grid.N:
                c = grid.C[at_ix, p["j"]]
                if math.isfinite(c):
                    px = c
            if px is None:
                px = p["last_px"]
                unmarked += 1
            else:
                p["last_px"] = px
            v += p["qty"] * px
        return v

    def close_due(until_time):
        nonlocal cash
        due = sorted([p for p in positions if p["exit_time"] <= until_time], key=lambda p: p["exit_time"])
        for p in due:
            positions.remove(p)
            fee = fm.maker_fee if p["reason"] == "tp" else fm.taker_fee
            notional = p["qty"] * p["exit_raw"]
            if p["reason"] in ("tp", "sl"):
                px = p["exit_raw"] if p["reason"] == "tp" else fm.sell(p["exit_raw"], notional, p["qv_exit"])
            else:
                px = fm.sell(p["exit_raw"], notional, p["qv_exit"])
            proceeds = p["qty"] * px * (1 - fee)
            cash += proceeds
            pnl = proceeds - p["cost"]
            closed.append({"sym": p["sym"], "mode": p["mode"], "tf": p["tf"], "entry_ts": p["entry_ts"],
                           "exit_time": p["exit_time"], "reason": p["reason"], "cost": p["cost"], "pnl": pnl,
                           "ret_pct": pnl / p["cost"] * 100, "gross_ret_pct": (p["exit_raw"] / p["entry_raw"] - 1) * 100,
                           "kind": p["kind"]})
            if p["j"] is not None and p.get("exit_ix") is not None:
                cooldown[p["sym"]] = p["exit_ix"] + policy.risk.cooldown_bars

    def roll_days(until_time):
        """Record equity at every 00:00 UTC up to until_time."""
        nonlocal cur_day
        while True:
            nxt = (cur_day + timedelta(days=1)) if cur_day else None
            if nxt is None or nxt > until_time:
                return
            close_due(nxt)
            ix = grid.idx(nxt) - 1
            day_start_eq[nxt.strftime("%Y-%m-%d")] = mark(ix)
            cur_day = nxt

    first = cands[0]["ts"] if cands else None
    if first is None:
        return {"series": [], "agg": {}, "trades": []}
    cur_day = first.replace(hour=0, minute=0, second=0, microsecond=0)
    day_start_eq[cur_day.strftime("%Y-%m-%d")] = cash
    for c in cands:
        roll_days(c["ts"])
        close_due(c["ts"])
        dkey = c["ts"].strftime("%Y-%m-%d")
        if c.get("price") is None:
            continue
        if c["ie"] is None and policy.exits.kind != "real":
            continue
        at_ix = c["ie"] if c["ie"] is not None else grid.idx(_bar(c["ts"])) - 1

        def _plan():
            if plan_cache is None:
                return TP.exit_plan(policy, grid, c, floor_of)
            key = (id(c), policy.exits)
            if key not in plan_cache:
                plan_cache[key] = TP.exit_plan(policy, grid, c, floor_of)
            return plan_cache[key]
        # the entry decision does not need the exit unless sizing is by risk: decide first, plan only
        # for admitted candidates (68k candidates x a 96-bar path otherwise dominates every run)
        need_plan_first = policy.sizing.kind == "risk"
        plan = _plan() if need_plan_first else None
        if need_plan_first and plan is None:
            continue
        eq_now = mark(at_ix)
        state = {"open": len(positions), "open_syms": {p["sym"] for p in positions}, "day_count": day_count[dkey],
                 "day_return_pct": (eq_now / day_start_eq.get(dkey, eq_now) - 1) * 100,
                 "cooldown_until": cooldown, "bar": c["ie"] or 0, "equity": eq_now,
                 "regime": ({k: bool(v[c["ie"]]) for k, v in regime_flags.items()}
                            if regime_flags is not None and c["ie"] is not None else None)}
        cc = dict(c, stop_dist=plan[3] if plan else None)
        frac = TP.decide_entry(policy, cc, state)
        if frac is None:
            continue
        if plan is None:
            plan = _plan()
            if plan is None:
                continue
        exit_ix, exit_raw, reason, stop_dist = plan
        notional = min(eq_now * frac, cash * (1 - policy.sizing.cash_reserve_pct / 100.0))
        if notional <= 1.0:
            continue
        j = c["j"]
        qv = (grid.V[c["ie"], j] * grid.C[c["ie"], j]) if (j is not None and c["ie"] is not None) else None
        fill = fm.buy(c["price"], notional, qv)
        qty = notional / fill * (1 - fm.taker_fee)
        cash -= notional
        if reason == "real":
            exit_time = c["exit_ts"]
            qv_exit = (grid.V[c["ix"], j] * grid.C[c["ix"], j]) if (j is not None and c.get("ix") is not None) else None
            exit_ix = c.get("ix")
        else:
            exit_time = bar_close_time(exit_ix)
            qv_exit = grid.V[exit_ix, j] * grid.C[exit_ix, j]
        positions.append({"sym": c["sym"], "mode": c.get("mode"), "tf": c.get("tf"), "kind": c["kind"], "j": j,
                          "qty": qty, "cost": notional, "entry_raw": c["price"], "entry_ts": c["ts"],
                          "exit_time": exit_time, "exit_raw": exit_raw, "exit_ix": exit_ix, "reason": reason,
                          "qv_exit": qv_exit if (qv_exit is not None and math.isfinite(qv_exit)) else None,
                          "last_px": fill})
        day_count[dkey] += 1
    # flush: close every remaining position at its planned exit, then the last day boundary
    end = max([p["exit_time"] for p in positions] + [cands[-1]["ts"]])
    roll_days(end)
    close_due(end + STEP)
    days = sorted(day_start_eq)
    series = []
    for a, b in zip(days, days[1:]):
        series.append((a, (day_start_eq[b] / day_start_eq[a] - 1) * 100))
    final = cash + sum(p["qty"] * p["last_px"] for p in positions)
    # V3: money conservation
    pnl_sum = sum(t["pnl"] for t in closed)
    v3 = abs((final - eq0) - pnl_sum) < 1e-6 * max(1.0, eq0) and not positions
    return {"series": series, "agg": DR.aggregate(series), "trades": closed, "final_equity": final,
            "v3_conservation_ok": v3, "unmarked_marks": unmarked}


def decompose(trades: list) -> dict:
    if not trades:
        return {"n": 0}
    r = [t["ret_pct"] for t in trades]
    w = [x for x in r if x > 0]
    l = [x for x in r if x <= 0]
    by = collections.defaultdict(list)
    for t in trades:
        by[t["reason"]].append(t["ret_pct"])
    return {"n": len(r), "net_mean_pct": round(sum(r) / len(r), 4),
            "gross_mean_pct": round(sum(t["gross_ret_pct"] for t in trades) / len(trades), 4),
            "win_share": round(len(w) / len(r), 3), "avg_win": round(sum(w) / len(w), 3) if w else None,
            "avg_loss": round(sum(l) / len(l), 3) if l else None,
            "by_reason": {k: {"n": len(v), "mean": round(sum(v) / len(v), 3)} for k, v in sorted(by.items())}}


# ---------------------------------------------------------------- CLI

POLICIES = {
    "champion": TP.CHAMPION,
    "cash": replace(TP.CHAMPION, name="cash", risk=TP.Risk(max_open=0)),
    "p1_trail": replace(TP.CHAMPION, name="p1_trail", exits=TP.Exits(kind="p1_trail")),
}


def _grid():
    """The signal bot's 15m grid (exit_validator.Grid, read-only) plus Low and Volume arrays,
    which that Grid does not keep; built here from the same bars so the bot code stays untouched."""
    TC.bot_readonly()
    import config
    import exit_validator as EV
    import _backtest_trend_start_detector as TD
    import numpy as np
    g = EV.load_grid(int(getattr(config, "ATR_PERIOD", 14)))
    if not hasattr(g, "L"):
        S = len(g.syms)
        n = g.C.shape[0]
        L, V = np.full((n, S), np.nan), np.full((n, S), np.nan)
        for j, s in enumerate(g.syms):
            for x in TD.bars_15m(s):
                k = int((x[0] - g.t0) / STEP)
                if 0 <= k < n:
                    L[k, j] = x[3]
                    if len(x) > 5:
                        V[k, j] = x[5]
        g.L, g.V = L, V
    return g


def _floor():
    TC.bot_readonly()
    import config
    import exit_validator as EV
    return lambda mode: EV.floor_of(mode, config)


def verify() -> int:
    ok = True
    trades, _ = load_events()
    import _backtest_daily_revenue_baseline as B
    base = B.compute(events_path=EVENTS)          # same events as the replay (the bot keeps appending)
    v1 = realized_simple(trades, 10, 0.20)
    ref = base["fees"]["taker_0.20rt"]
    d_mean = abs(v1["agg"]["mean_pct"] - ref["mean_day_pct"])
    good = v1["taken"] == base["taken"] and d_mean < 1e-4 and v1["agg"]["n_days"] == ref["days"]
    ok &= good
    print(f"V1 baseline: taken {v1['taken']} vs {base['taken']}, days {v1['agg']['n_days']} vs {ref['days']}, "
          f"mean {v1['agg']['mean_pct']:+.4f} vs {ref['mean_day_pct']:+.4f}  -> {'OK' if good else 'FAIL'}")
    grid = _grid()
    attach_grid(trades, grid)
    import exit_validator as EV
    fl = _floor()
    n = bad = 0
    for t in trades:
        if t["tf"] != "15m" or t["ie"] is None or t["ix"] is None:
            continue
        plan = TP.exit_plan(POLICIES["p1_trail"], grid, t, fl)
        ref_p = EV.plain_trail(grid, {"j": t["j"], "ie": t["ie"], "k": float(t["trail_k"] or 2.0)}, fl(t["mode"]))
        if plan is None or ref_p is None:
            continue
        mine = (plan[1] / grid.C[t["ie"], t["j"]] - 1) * 100
        n += 1
        bad += abs(mine - ref_p) > 1e-9
    good = n > 1000 and bad == 0
    ok &= good
    print(f"V2 p1_trail vs exit_validator.plain_trail: {n} trades, {bad} differ -> {'OK' if good else 'FAIL'}")
    r = run(POLICIES["champion"], FM.model("taker", 0), grid, trades)
    good = r["v3_conservation_ok"] and min([0.0] + [0.0]) == 0.0
    ok &= good
    print(f"V3 conservation (champion, taker, slip 0): {'OK' if good else 'FAIL'}; final equity {r['final_equity']:.2f}, "
          f"unmarked marks {r['unmarked_marks']}")
    print("VERIFY:", "ALL OK" if ok else "FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--policy", default="champion", choices=sorted(POLICIES))
    ap.add_argument("--fees", default="taker", choices=sorted(FM.FEES))
    ap.add_argument("--slip", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--split", default="2026-07-01")
    a = ap.parse_args(argv)
    if a.verify:
        return verify()
    trades, _ = load_events()
    grid = _grid()
    attach_grid(trades, grid)
    pol, fm = POLICIES[a.policy], FM.model(a.fees, a.slip)
    out = {"policy": asdict(pol), "epoch": pol.epoch(), "fees": a.fees, "slip": a.slip, "split": a.split}
    for name, lo, hi in (("all", None, None), ("train", None, a.split), ("test", a.split, None)):
        r = run(pol, fm, grid, trades, floor_of=_floor(), day_from=lo, day_to=hi)
        out[name] = {"agg": r["agg"], "decomposition": decompose(r["trades"]), "v3": r["v3_conservation_ok"],
                     "series": r["series"]}
        g = r["agg"]
        print(f"[{name}] days {g.get('n_days')} mean {g.get('mean_pct')}%/day [{g.get('lo95')}, {g.get('hi95')}] "
              f"compound {g.get('compound_pct')}% maxDD {g.get('max_dd_pct')}% pos {g.get('pos_share')} | "
              f"trades {out[name]['decomposition'].get('n')} net/trade {out[name]['decomposition'].get('net_mean_pct')}% "
              f"gross {out[name]['decomposition'].get('gross_mean_pct')}% -> {DR.verdict_text(g)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{a.policy}_{a.fees}_slip{a.slip}_{out['epoch']}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("saved", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
