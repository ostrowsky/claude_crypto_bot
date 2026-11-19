"""Rotation when the portfolio is full, judged by the goal.
Criteria fixed in docs/specs/features/rotation-goal-1119-spec.md before the run.

ML-gated rotation needs ml_proba >= 0.62, a level set on the ML scale before the
2026-09-07 peak-label switch; since then 0-2% of portfolio-blocked candidates reach
it and rotation has stopped without a decision. Would evicting the weakest leg for
a candidate that passed every other gate put the bot into more winner-days before
their +2.5% crossing, with trades no worse?
"""
from __future__ import annotations

import collections
import io
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

WEAK_BARS = 3
PROTECT_PCT = 0.5
EV_MAX = -0.40
TOP_SHARE = 0.25
CI = 0.975
BOOT = 2000
SINCE = "2026-03-20"
NEW_SCALE = datetime(2026, 9, 7, tzinfo=timezone.utc)
TF_LEN = {"15m": timedelta(minutes=15), "1h": timedelta(hours=1)}


def eligible(p, t, price, leader_ts) -> bool:
    """The live find_weakest_leg rules, at time t, with the price at t."""
    ev = p.get("ranker_ev")
    if not isinstance(ev, (int, float)) or ev >= EV_MAX:
        return False
    if leader_ts is not None and leader_ts <= t:
        return False
    if int((t - p["entry_dt"]) / TF_LEN[p["tf"]]) < WEAK_BARS:
        return False
    if not price or not p.get("entry_price"):
        return False
    return (price / p["entry_price"] - 1) * 100 < PROTECT_PCT


def weakest(cands):
    return min(cands, key=lambda p: p["ranker_ev"]) if cands else None


def top_share_mask(rows, share):
    """Per month: the top round(share * n) rows by ml_proba; an unknown proba never qualifies."""
    by = collections.defaultdict(list)
    for k, r in enumerate(rows):
        by[r["month"]].append(k)
    out = [False] * len(rows)
    for ks in by.values():
        n = int(round(share * len(ks)))
        known = sorted((k for k in ks if isinstance(rows[k]["ml_proba"], (int, float))),
                       key=lambda k: -rows[k]["ml_proba"])
        for k in known[:n]:
            out[k] = True
    return out


def gained(W, real_early, added):
    """Per winner-day: 1 if no real entry before the crossing but an added one is."""
    out = []
    for _day, sym, op, dd in W:
        if any(op <= t < dd for t in real_early.get(sym, ())):
            out.append(0)
        else:
            out.append(int(any(op <= t < dd for t in added.get(sym, ()))))
    return out


def verdict(n_win, n_added, d_pp, d_lo, t_lo) -> str:
    if n_win < 30 or n_added < 20:
        return "needs_data"
    if d_pp <= 0 or t_lo < -0.10:
        return "refuted"
    if d_pp >= 0.5 and d_lo > 0:
        return "supported"
    return "inconclusive"


def _dt(s):
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _maxdd(seq):
    peak = dd = acc = 0.0
    for v in seq:
        acc += v
        peak = max(peak, acc)
        dd = min(dd, acc - peak)
    return dd


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import config as cfg
    import goal_validator as GV
    import indicators as I
    import readouts as R
    import _backtest_trend_start_detector as TD

    # ---- population: candidates blocked at the portfolio limit
    rows, seen = [], set()
    with io.open(HERE / "critic_dataset.jsonl", "rb") as fh:
        for l in fh:
            if b'"portfolio"' not in l:
                continue
            try:
                r = json.loads(l)
            except Exception:
                continue
            d = r.get("decision") or {}
            if d.get("stage") != "portfolio" or d.get("reason_code") != "portfolio" or r.get("tf") not in TF_LEN:
                continue
            k = (r["sym"], r["tf"], r["ts_signal"])
            if k in seen:
                continue
            seen.add(k)
            bo = _dt(r["ts_signal"])
            rows.append({"sym": r["sym"], "tf": r["tf"], "bo": bo, "t": bo + TF_LEN[r["tf"]],
                         "month": r["ts_signal"][:7], "mode": r.get("signal_type"),
                         "ml_proba": d.get("ml_proba")})
    rows.sort(key=lambda x: x["t"])
    print("portfolio-blocked candidates (dedup): %d, %s .. %s" % (len(rows), rows[0]["t"].date(), rows[-1]["t"].date()))

    # ---- real positions
    evs = R.scan_events(SINCE, {"entry", "exit", "leader_exit_switch"})
    leader = collections.defaultdict(list)
    for e in evs:
        if e["event"] == "leader_exit_switch":
            leader[e.get("sym")].append(e["_dt"])
    trades = []
    for en, ex in R.pair_trades([e for e in evs if e["event"] in ("entry", "exit")]):
        tf = str(en.get("tf") or "")
        if tf not in TF_LEN:
            continue
        ls = [x for x in leader.get(en["sym"], ()) if en["_dt"] <= x <= ex["_dt"]]
        trades.append({"id": len(trades), "sym": en["sym"], "tf": tf, "entry_dt": en["_dt"], "exit_dt": ex["_dt"],
                       "entry_price": float(en.get("price") or 0) or None, "ranker_ev": en.get("ranker_ev"),
                       "trail_k": en.get("trail_k"), "mode": en.get("signal_mode") or en.get("mode"),
                       "leader_ts": min(ls) if ls else None})
    print("real closed trades since %s: %d (ranker_ev known %d, below %.2f: %d)" % (
        SINCE, len(trades), sum(isinstance(p["ranker_ev"], (int, float)) for p in trades), EV_MAX,
        sum(isinstance(p["ranker_ev"], (int, float)) and p["ranker_ev"] < EV_MAX for p in trades)))

    cache, c15 = {}, {}

    def series(sym, tf):
        if (sym, tf) not in cache:
            b = TD.load_bars(sym, tf) or []
            if b:
                import numpy as np
                h = np.array([x[2] for x in b], float)
                lo = np.array([x[3] for x in b], float)
                c = np.array([x[4] for x in b], float)
                t = [x[0] if x[0].tzinfo else x[0].replace(tzinfo=timezone.utc) for x in b]
                cache[(sym, tf)] = (t, c, I._atr(h, lo, c, cfg.ATR_PERIOD), {x: i for i, x in enumerate(t)})
            else:
                cache[(sym, tf)] = None
        return cache[(sym, tf)]

    def close_at(sym, t):
        """Close of the 15m bar that ends at t (or the last one before)."""
        s = series(sym, "15m")
        if not s:
            return None
        for back in range(0, 4):
            i = s[3].get(t - TF_LEN["15m"] * (1 + back))
            if i is not None:
                return float(s[1][i])
        return None

    ks = sorted(float(p["trail_k"]) for p in trades if isinstance(p["trail_k"], (int, float)))
    K = ks[len(ks) // 2] if ks else 2.0
    cooldown = int(getattr(cfg, "COOLDOWN_BARS", 19))

    # current arm: every real entry replayed on the live trail
    cur = {}
    for p in trades:
        s = series(p["sym"], p["tf"])
        i = s[3].get(GV.bar_open(p["entry_dt"], p["tf"]) - GV.STEP[p["tf"]]) if s else None
        if i is None:
            continue
        r = GV.live_trail(s[1], s[2], i, float(p["trail_k"]) if isinstance(p["trail_k"], (int, float)) else K,
                          GV.trail_floor(p["mode"], cfg), GV.MAX_BARS[p["tf"]])
        if r:
            cur[p["id"]] = {"day": p["entry_dt"].strftime("%Y-%m-%d"), "pnl": r[1], "c0": float(s[1][i]),
                            "sim_exit": s[0][r[0]] + TF_LEN[p["tf"]]}
    print("current trades on the live trail: %d, trail_k median %.2f" % (len(cur), K))

    W = GV.winner_days("2026-03-24")
    real_early = collections.defaultdict(list)
    for p in trades:
        real_early[p["sym"]].append(p["entry_dt"])
    print("winner-days (full uptime, crossing known): %d" % len(W))

    rnd = random.Random(1119)
    al = (1 - CI) / 2
    masks = {"ROT_ALL": [True] * len(rows), "ROT_TOP25": top_share_mask(rows, TOP_SHARE)}
    for arm, mask in masks.items():
        evicted, added, added_at = {}, [], collections.defaultdict(list)
        busy, stats = {}, collections.Counter()
        for r, ok in zip(rows, mask):
            if not ok:
                continue
            t = r["t"]
            stats["qualified"] += 1
            if busy.get(r["sym"]) and t < busy[r["sym"]]:
                stats["already_added"] += 1
                continue
            open_ = [p for p in trades if p["entry_dt"] <= t < p["exit_dt"] and p["id"] not in evicted]
            if any(p["sym"] == r["sym"] for p in open_):
                stats["in_position"] += 1
                continue
            el = [p for p in open_ if eligible(p, t, close_at(p["sym"], t), p["leader_ts"])]
            w = weakest(el)
            if w is None:
                stats["no_weak_leg"] += 1
                continue
            s = series(r["sym"], r["tf"])
            i = s[3].get(r["bo"]) if s else None
            res = GV.live_trail(s[1], s[2], i, K, GV.trail_floor(r["mode"], cfg), GV.MAX_BARS[r["tf"]]) \
                if i is not None else None
            if not res:
                stats["no_bars"] += 1
                continue
            px = close_at(w["sym"], t)
            if w["id"] in cur and px:
                if cur[w["id"]]["sim_exit"] > t:
                    evicted[w["id"]] = (px / cur[w["id"]]["c0"] - 1) * 100
                else:
                    evicted[w["id"]] = cur[w["id"]]["pnl"]          # the trail had already closed it
            else:
                evicted[w["id"]] = None
            exit_t = s[0][res[0]] + TF_LEN[r["tf"]]
            busy[r["sym"]] = exit_t + TF_LEN[r["tf"]] * cooldown
            added.append({"day": t.strftime("%Y-%m-%d"), "pnl": res[1], "t": t, "sym": r["sym"],
                          "slot_longer": exit_t > w["exit_dt"], "new_scale": t >= NEW_SCALE})
            added_at[r["sym"]].append(t)
            stats["evictions"] += 1
        # goal
        g = gained(W, real_early, added_at)
        n = len(W)
        gb = sorted(100 * sum(rnd.choices(g, k=n)) / n for _ in range(BOOT))
        d_pp, d_lo, d_hi = 100 * sum(g) / n, gb[int(al * BOOT)], gb[int((1 - al) * BOOT) - 1]
        # evicted early winner-day legs (capture given up)
        wd = {(day, sym): (op, dd) for day, sym, op, dd in W}
        ev_win = []
        for tid, pnl_t in evicted.items():
            p = trades[tid]
            k = (p["entry_dt"].strftime("%Y-%m-%d"), p["sym"])
            if k in wd and wd[k][0] <= p["entry_dt"] < wd[k][1] and tid in cur and pnl_t is not None:
                ev_win.append(pnl_t - cur[tid]["pnl"])
        # per trade
        comb_day, cur_day = collections.defaultdict(list), collections.defaultdict(list)
        for tid, x in cur.items():
            cur_day[x["day"]].append(x["pnl"])
            v = evicted.get(tid)
            comb_day[x["day"]].append(x["pnl"] if v is None else v)
        for a in added:
            comb_day[a["day"]].append(a["pnl"])
        days = sorted(set(cur_day) | set(comb_day))

        def diff(pick):
            sc = sum(sum(cur_day.get(d, [])) * w for d, w in pick.items())
            nc = sum(len(cur_day.get(d, [])) * w for d, w in pick.items())
            sm = sum(sum(comb_day.get(d, [])) * w for d, w in pick.items())
            nm = sum(len(comb_day.get(d, [])) * w for d, w in pick.items())
            return sm / nm - sc / nc if nc and nm else 0.0
        t_pt = diff({d: 1 for d in days})
        tb = sorted(diff(collections.Counter(rnd.choices(days, k=len(days)))) for _ in range(BOOT))
        t_lo, t_hi = tb[int(al * BOOT)], tb[int((1 - al) * BOOT) - 1]
        dd_c = _maxdd([sum(cur_day.get(d, [])) for d in days])
        dd_m = _maxdd([sum(comb_day.get(d, [])) for d in days])
        wc = min(sum(cur_day.get(d, [])) for d in days)
        wm = min(sum(comb_day.get(d, [])) for d in days)
        v = verdict(n, len(added), d_pp, d_lo, t_lo)
        new = [a for a in added if a["new_scale"]]
        print("\n=== %s ===" % arm)
        print("  qualified rows %d | evictions %d | no weak leg %d | already added %d | in position %d | no bars %d" % (
            stats["qualified"], stats["evictions"], stats["no_weak_leg"], stats["already_added"],
            stats["in_position"], stats["no_bars"]))
        print("  added trades %d (mean %+.3f%%), %d of them after %s; slot held longer than the evicted leg: %d" % (
            len(added), sum(a["pnl"] for a in added) / len(added) if added else 0.0, len(new), NEW_SCALE.date(),
            sum(a["slot_longer"] for a in added)))
        print("  goal: gained %d of %d winner-days -> %+.2f pp [%+.2f, %+.2f] (97.5%%)" % (sum(g), n, d_pp, d_lo, d_hi))
        print("  evicted early winner-day legs: %d, P&L given up mean %+.3f pp" % (
            len(ev_win), sum(ev_win) / len(ev_win) if ev_win else 0.0))
        print("  per trade combined-current %+.3f pp [%+.3f, %+.3f] (97.5%%)" % (t_pt, t_lo, t_hi))
        print("  tail (context): max drawdown %.1f -> %.1f, worst day %.1f -> %.1f" % (dd_c, dd_m, wc, wm))
        gm = collections.Counter(w[0][:7] for w, x in zip(W, g) if x)
        print("  gained days by month (context):", dict(sorted(gm.items())))
        print("  VERDICT %s: %s" % (arm, v))


if __name__ == "__main__":
    main()
