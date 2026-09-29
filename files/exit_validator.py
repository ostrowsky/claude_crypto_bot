"""L3 exit validator -- judge exit / leader-mode hypotheses by what exits are FOR.

WHY (2026-09-29, agent-tasks-0929-spec.md, task 3)

The bot buys 57% of rockets before +10% but keeps ~8% of the move; the first
four incident reports (25 winner-days) show exits leaving the move behind
(HBAR 28.09: +36.6% day, exited on a WEAK signal). Every exit change of the last
month (X-7b, X-9c, the alignment trail) was validated by hand with
_backtest_leader_x8_x9.py; the L3 replay validator could not test a single exit
key, so the agent was never allowed to propose one. This module replays exit
keys on the bot's own 15m trades and hands L3 a verdict.

KEYS
  leader mode  LEADER_EXIT_RANK_MAX, _MIN_RET, _FLOOR_PCT, _LOST_RANK,
               _LOST_MIN_BARS, _MAX_HOLD_BARS -- every real 15m trade with an exit
               since 2026-03-01 is replayed under the CURRENT and the PROPOSED
               leader parameters (same engine as leader_exit.py and the X-7/X-9
               backtests): a trade that meets the switch rule before its real exit
               is held on the wide trail from the switch bar; otherwise its real
               exit stands. A trade that exited IN leader mode live (since
               2026-09-27) and does not switch under one of the two parameter
               sets has no observable old exit -- it is dropped from both arms
               and counted ("unobservable").
  trail floor  TRAIL_MIN_BUFFER_PCT_<MODE> -- that mode's 15m trades replayed
               from the entry bar on the calibrated live trail (P-1) with the
               current and the proposed floor; other modes are untouched.

THE CRITERION, FIXED BEFORE ANY HYPOTHESIS IS GRADED

  paired per-trade difference (proposed - current), bootstrap 95%:
    rocket   trades entered on a rocket-day (day high >= +10% from the UTC open,
             close keeps >= 60% of the rise; immutable label store)
    all      every replayed trade (the non-inferiority guard)
    months   rocket-day difference by month (>= 5 rocket trades)
  accept   rocket lower > 0 AND all lower >= -0.10 pp AND >= 60% of months >= 0
  reject   all lower < -0.10 pp OR rocket upper < 0 OR no trade changes
  needs_data  < 30 rocket-day trades, < 200 trades or < 30 trades that change
  otherwise   needs_review
Also reported: winner-day difference (immutable top-20) and, for leader keys,
the share of rocket-days held in leader mode (the operator's coverage measure).

Accept means "the evidence supports it", never "apply it".
"""
from __future__ import annotations

import collections
import io
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

EVENTS = HERE / "bot_events.jsonl"
SINCE_DEFAULT = "2026-03-01"      # first month of the 15m long store used by X-7/X-9
STEP = timedelta(minutes=15)
MIN_TRADES = 200
MIN_ROCKET = 30
MIN_CHANGED = 30                  # fewer changed trades cannot be told from noise
MONTH_MIN = 5
MONTH_AGREE = 0.6
NONINF_PP = -0.10
BOOT = 1000
LEADER_KEYS = {
    "LEADER_EXIT_RANK_MAX": "rank_max", "LEADER_EXIT_MIN_RET": "min_ret",
    "LEADER_EXIT_FLOOR_PCT": "floor", "LEADER_EXIT_LOST_RANK": "lost_rank",
    "LEADER_EXIT_LOST_MIN_BARS": "lost_min_bars", "LEADER_EXIT_MAX_HOLD_BARS": "max_hold_bars",
}
FLOOR_MODES = ("impulse_speed", "strong_trend", "impulse", "trend", "alignment", "retest", "breakout")
FLOOR_KEYS = {"TRAIL_MIN_BUFFER_PCT_" + m.upper(): m for m in FLOOR_MODES}
KEYS = tuple(LEADER_KEYS) + tuple(FLOOR_KEYS)
P1_MAX_BARS = 96                  # the P-1 calibration horizon of the plain trail


def validatable_keys() -> list:
    return sorted(KEYS)


def leader_params(cfg) -> dict:
    return {
        "rank_max": int(getattr(cfg, "LEADER_EXIT_RANK_MAX", 3)),
        "min_ret": float(getattr(cfg, "LEADER_EXIT_MIN_RET", 0.05)),
        "floor": float(getattr(cfg, "LEADER_EXIT_FLOOR_PCT", 0.08)),
        "lost_rank": int(getattr(cfg, "LEADER_EXIT_LOST_RANK", 10)),
        "lost_min_bars": int(getattr(cfg, "LEADER_EXIT_LOST_MIN_BARS", 4)),
        "max_hold_bars": int(getattr(cfg, "LEADER_EXIT_MAX_HOLD_BARS", 96 * 7)),
    }


def floor_of(mode, cfg) -> float:
    if not getattr(cfg, "TRAIL_MIN_BUFFER_PCT_ENABLED", False):
        return 0.0
    return float(getattr(cfg, "TRAIL_MIN_BUFFER_PCT_" + str(mode or "DEFAULT").upper(),
                         getattr(cfg, "TRAIL_MIN_BUFFER_PCT_DEFAULT", 0.0)))


# ---------------------------------------------------------------- the grid

class Grid:
    """Aligned 15m close/high/open/ATR and the day's return rank for the watchlist."""

    def __init__(self, bars: dict, atr_period: int, min_bars: int = 2000):
        import numpy as np
        import indicators as I
        bars = {s: b for s, b in bars.items() if len(b) >= min_bars}
        self.syms = sorted(bars)
        self.col = {s: j for j, s in enumerate(self.syms)}
        S = len(self.syms)
        if not S:
            raise ValueError("no symbol has enough 15m bars")
        self.t0 = min(b[0][0] for b in bars.values()).replace(hour=0, minute=0, second=0, microsecond=0)
        t1 = max(b[-1][0] for b in bars.values())
        self.last_bar = t1
        N = int((t1 - self.t0) / STEP) + 1
        self.days = N // 96
        self.N = self.days * 96
        O, H, L, C = (np.full((N, S), np.nan) for _ in range(4))
        for j, s in enumerate(self.syms):
            for x in bars[s]:
                k = int((x[0] - self.t0) / STEP)
                O[k, j], H[k, j], L[k, j], C[k, j] = x[1], x[2], x[3], x[4]
        fill = lambda a: np.where(np.isfinite(a), a, np.nanmean(a))  # noqa: E731
        self.ATR = np.column_stack([I._atr(fill(H[:, j]), fill(L[:, j]), fill(C[:, j]), atr_period) for j in range(S)])
        self.O, self.H, self.C = O, H, C
        n = self.N
        opens = np.repeat(O[:n:96][: self.days], 96, axis=0)
        ret = C[:n] / opens - 1
        rank = np.full((n, S), 999, dtype=int)
        for i in range(n):
            row = ret[i]
            ok = np.isfinite(row)
            if ok.any():
                order = np.argsort(-np.where(ok, row, -9))
                rank[i, order] = np.arange(1, S + 1)
                rank[i, ~ok] = 999
        self.RET, self.RANK = ret, rank
        self._lead = {}

    def idx(self, dt) -> int:
        return int((dt - self.t0) / STEP)

    def lead(self, rank_max: int, min_ret: float):
        key = (int(rank_max), round(float(min_ret), 6))
        if key not in self._lead:
            self._lead[key] = (self.RANK <= rank_max) & (self.RET >= min_ret)
        return self._lead[key]


_GRID = None


def load_grid(atr_period: int):
    global _GRID
    if _GRID is None:
        import _backtest_trend_start_detector as TD
        import _compute_early_capture as E
        _GRID = Grid({s: TD.bars_15m(s) for s in E.load_watchlist()}, atr_period)
    return _GRID


# ---------------------------------------------------------------- trades

def load_trades(grid: Grid, since: str, events_path=None) -> list:
    """Real 15m trades with an exit: entry bar index, real exit bar, real pnl."""
    ent, exs = collections.defaultdict(list), collections.defaultdict(list)
    with io.open(events_path or EVENTS, "rb") as fh:
        for raw in fh:
            if b'"entry"' not in raw and b'"exit"' not in raw:
                continue
            try:
                e = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue
            if e.get("event") not in ("entry", "exit") or e.get("tf") != "15m" \
                    or str(e.get("ts", "")) < since or e.get("sym") not in grid.col:
                continue
            d = datetime.fromisoformat(str(e["ts"]).replace("Z", "+00:00"))
            d = d if d.tzinfo else d.replace(tzinfo=timezone.utc)
            (ent if e["event"] == "entry" else exs)[e["sym"]].append((d, e))
    out = []
    for sym, es in ent.items():
        xs = sorted(exs.get(sym, []), key=lambda x: x[0])
        for d, e in sorted(es, key=lambda x: x[0]):
            x = next((x for x in xs if x[0] > d), None)
            if x is None or not isinstance(x[1].get("pnl_pct"), (int, float)) or not e.get("price"):
                continue
            bar = lambda t: t.replace(minute=t.minute // 15 * 15, second=0, microsecond=0)  # noqa: E731
            ie, ix = grid.idx(bar(d)) - 1, grid.idx(bar(x[0])) - 1
            if ie < 100 or ix <= ie or ix >= grid.N:
                continue
            k = e.get("trail_k")
            out.append({"sym": sym, "j": grid.col[sym], "ie": ie, "ix": ix, "ep": float(e["price"]),
                        "real": float(x[1]["pnl_pct"]), "k": float(k) if isinstance(k, (int, float)) else 2.0,
                        "mode": e.get("mode") or e.get("signal_mode"), "day": d.strftime("%Y-%m-%d"),
                        "leader_exit_live": str(x[1].get("reason") or "").startswith("режим лидера")})
    return out


def wide_from(grid: Grid, t: dict, s: int, P: dict) -> float:
    """Leader-mode exit from switch bar s (leader_exit.step)."""
    j, ep, k = t["j"], t["ep"], t["k"]
    C, A = grid.C, grid.ATR
    cs = C[s, j]
    stop = cs - max(k * A[s, j] if math.isfinite(A[s, j]) else 0.0, P["floor"] * cs)
    last = min(grid.N - 1, s + P["max_hold_bars"])
    for q in range(s + 1, last + 1):
        cq = C[q, j]
        if not math.isfinite(cq):
            continue
        if math.isfinite(A[q, j]):
            stop = max(stop, cq - max(k * A[q, j], P["floor"] * cq))
        if cq < stop or (q - s >= P["lost_min_bars"] and grid.RANK[q, j] > P["lost_rank"]):
            return (cq / ep - 1) * 100
    q = last
    while q > s and not math.isfinite(C[q, j]):
        q -= 1
    return (C[q, j] / ep - 1) * 100


def leader_arm(grid: Grid, trades: list, P: dict):
    """(pnl per trade or None if unobservable, switch bar or None)."""
    import numpy as np
    m = grid.lead(P["rank_max"], P["min_ret"])
    vals, sw = [], []
    for t in trades:
        hits = np.where(m[t["ie"]:t["ix"], t["j"]])[0]
        if len(hits):
            s = t["ie"] + int(hits[0])
            vals.append(wide_from(grid, t, s, P))
            sw.append(s)
        else:
            vals.append(None if t["leader_exit_live"] else t["real"])
            sw.append(None)
    return vals, sw


def plain_trail(grid: Grid, t: dict, floor: float, max_bars: int = P1_MAX_BARS):
    j, ie, k = t["j"], t["ie"], t["k"]
    C, A = grid.C, grid.ATR
    ep = C[ie, j]
    if not math.isfinite(ep):
        return None
    stop = ep - max(k * A[ie, j] if math.isfinite(A[ie, j]) else 0.0, floor * ep)
    last = min(grid.N - 1, ie + max_bars)
    for q in range(ie + 1, last + 1):
        cq = C[q, j]
        if not math.isfinite(cq):
            continue
        if math.isfinite(A[q, j]):
            stop = max(stop, cq - max(k * A[q, j], floor * cq))
        if cq < stop:
            return (cq / ep - 1) * 100
    return (C[last, j] / ep - 1) * 100 if math.isfinite(C[last, j]) else None


# ---------------------------------------------------------------- statistics

def paired(diffs: list, rnd: random.Random):
    if not diffs:
        return {"n": 0}
    bs = sorted(sum(rnd.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(BOOT))
    return {"n": len(diffs), "mean": round(sum(diffs) / len(diffs), 3),
            "lo95": round(bs[int(0.025 * BOOT)], 3), "hi95": round(bs[int(0.975 * BOOT) - 1], 3)}


def decide(all_d: dict, rocket_d: dict, agree: float, changed: int) -> str:
    if changed == 0:
        return "reject"
    if all_d["n"] < MIN_TRADES or rocket_d["n"] < MIN_ROCKET or changed < MIN_CHANGED:
        return "needs_data"
    if all_d["lo95"] < NONINF_PP or rocket_d["hi95"] < 0:
        return "reject"
    if rocket_d["lo95"] > 0 and agree >= MONTH_AGREE:
        return "accept"
    return "needs_review"


def day_labels():
    """(rocket-days, winner-days) from the immutable label store."""
    import immutable_labels as IL
    import label_store as LS
    import _compute_early_capture as E
    rockets = set()
    for r in LS.LabelStore().records():
        o, h, c = float(r.get("open") or 0), float(r.get("high") or 0), float(r.get("close") or 0)
        if o > 0 and h >= o * 1.10 and (c - o) >= 0.6 * (h - o):
            rockets.add((r["utc_day"], r["symbol"]))
    win, _ = IL.winners_by_day(top_n=20, watchlist=E.load_watchlist(), rank_before_filter=True)
    return rockets, set(win)


def covered_rocket_days(rockets: set, since: str, watchlist=None, full_days=None) -> set:
    """Rocket-days the bot could have held: watchlist coin, bot up all day."""
    import _compute_early_capture as E
    wl = watchlist if watchlist is not None else E.load_watchlist()
    if full_days is None:
        full_days, _, _ = E.load_uptime(datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc))
    return {k for k in rockets if k[0] >= since and k[1] in wl and k[0] in full_days}


def validate(hyp: dict, since: str = SINCE_DEFAULT, cfg_module=None, grid=None, trades=None,
             labels=None, events_path=None, rocket_days=None) -> dict:
    key = str(hyp.get("config_key") or "")
    diff = hyp.get("diff") or {}
    base = {"validator": "exit_validator", "config_key": key, "since": since,
            "criterion": "paired per-trade: rocket-days lower > 0, all trades lower >= -0.10 pp, months agree"}
    if cfg_module is None:
        import config as cfg_module
    if key not in KEYS or not hasattr(cfg_module, key):
        return {**base, "verdict": "reject", "reason": f"no exit replay for '{key}'"}
    try:
        new_v = float(diff.get("to"))
    except (TypeError, ValueError):
        return {**base, "verdict": "reject", "reason": "diff.to is not a number"}
    cur_v = float(getattr(cfg_module, key))
    if new_v == cur_v:
        return {**base, "verdict": "reject", "reason": "diff.to equals the live value"}
    grid = grid or load_grid(int(getattr(cfg_module, "ATR_PERIOD", 14)))
    trades = trades if trades is not None else load_trades(grid, since, events_path)
    rockets, winners = labels if labels is not None else day_labels()

    unobservable = 0
    coverage = None
    if key in LEADER_KEYS:
        cur_p = leader_params(cfg_module)
        new_p = dict(cur_p, **{LEADER_KEYS[key]: type(cur_p[LEADER_KEYS[key]])(new_v)})
        a_cur, sw_cur = leader_arm(grid, trades, cur_p)
        a_new, sw_new = leader_arm(grid, trades, new_p)
        rows = []
        for t, x, y in zip(trades, a_cur, a_new):
            if x is None or y is None:
                unobservable += 1
                continue
            rows.append((t, x, y))

        def held(sw):
            return {((grid.t0 + s * STEP).strftime("%Y-%m-%d"), t["sym"]) for t, s in zip(trades, sw) if s is not None}
        rd = rocket_days if rocket_days is not None else covered_rocket_days(rockets, since)
        coverage = {"rocket_days": len(rd), "held_in_leader_mode_now": len(rd & held(sw_cur)),
                    "held_in_leader_mode_after": len(rd & held(sw_new)),
                    "switched_now": sum(1 for s in sw_cur if s is not None),
                    "switched_after": sum(1 for s in sw_new if s is not None)}
        detail = {"params_now": cur_p, "params_after": new_p}
    else:
        mode = FLOOR_KEYS[key]
        rows = []
        for t in trades:
            if t["mode"] != mode:
                continue
            x, y = plain_trail(grid, t, cur_v), plain_trail(grid, t, new_v)
            if x is not None and y is not None:
                rows.append((t, x, y))
        detail = {"mode": mode, "engine": "P-1 calibrated live trail from the entry bar, %d bars" % P1_MAX_BARS}

    rnd = random.Random(29)
    d_all = [y - x for _, x, y in rows]
    d_rk = [y - x for t, x, y in rows if (t["day"], t["sym"]) in rockets]
    d_wn = [y - x for t, x, y in rows if (t["day"], t["sym"]) in winners]
    changed = sum(1 for v in d_all if abs(v) > 1e-9)
    all_s, rk_s, wn_s = paired(d_all, rnd), paired(d_rk, rnd), paired(d_wn, rnd)
    bym = collections.defaultdict(list)
    for t, x, y in rows:
        if (t["day"], t["sym"]) in rockets:
            bym[t["day"][:7]].append(y - x)
    months = {m: {"mean": round(sum(v) / len(v), 3), "n": len(v)} for m, v in sorted(bym.items())}
    scored = [m for m, x in months.items() if x["n"] >= MONTH_MIN]
    agree = (sum(1 for m in scored if months[m]["mean"] >= 0) / len(scored)) if scored else 0.0
    rk_now = [x for t, x, _ in rows if (t["day"], t["sym"]) in rockets]
    rk_new = [y for t, _, y in rows if (t["day"], t["sym"]) in rockets]
    report = {**base, "from": cur_v, "to": new_v, **detail,
              "bars_until": grid.last_bar.isoformat(timespec="minutes"),
              "trades": len(rows), "changed_trades": changed, "unobservable": unobservable,
              "all_trades_diff_pp": all_s, "rocket_day_diff_pp": rk_s, "winner_day_diff_pp": wn_s,
              "rocket_day_mean_now": round(sum(rk_now) / len(rk_now), 3) if rk_now else None,
              "rocket_day_mean_after": round(sum(rk_new) / len(rk_new), 3) if rk_new else None,
              "rocket_months": months, "month_agreement": round(agree, 3), "months_scored": len(scored),
              "leader_coverage": coverage}
    v = decide(all_s, rk_s, agree, changed)
    desc = (f"{key} {cur_v:g} -> {new_v:g}: {changed} of {len(rows)} trades change; rocket-days "
            f"{rk_s.get('mean', 0):+.3f} pp [{rk_s.get('lo95', 0):+.3f}, {rk_s.get('hi95', 0):+.3f}] "
            f"(n={rk_s['n']}), all {all_s.get('mean', 0):+.3f} pp [{all_s.get('lo95', 0):+.3f}, "
            f"{all_s.get('hi95', 0):+.3f}], winners {wn_s.get('mean', 0):+.3f} pp; months agree "
            f"{agree:.0%} of {len(scored)}" + (f"; unobservable {unobservable}" if unobservable else ""))
    if changed == 0:
        desc += " -- the change touches no trade"
    return {**report, "verdict": v, "reason": desc}


if __name__ == "__main__":
    import argparse
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--to", type=float, required=True)
    ap.add_argument("--since", default=SINCE_DEFAULT)
    a = ap.parse_args()
    print(json.dumps(validate({"config_key": a.key, "diff": {"to": a.to}}, a.since),
                     ensure_ascii=False, indent=1, default=str))
