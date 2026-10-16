"""Can correlation_guard keep its tail protection and give back the early entries?
Criteria: docs/specs/features/corr-guard-compromise-1016-spec.md (fixed before the run).

  CAP3      admit a guard-blocked row when its logged cluster size is < 3
  PRECROSS  admit it when the coin is still < +2.5% above its UTC day open
  OFF       admit every row (reference)
"""
from __future__ import annotations

import collections
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

ARMS = ("CAP3", "PRECROSS", "OFF")
CI = 0.975
TAIL_TOL = 0.10
CROSS = 0.025
BOOT = 2000
_SIZE = re.compile(r"cluster size (\d+)/(\d+)")


def cluster_size(reason):
    m = _SIZE.search(str(reason or ""))
    return int(m.group(1)) if m else None


def admit(arm, row, day_open_px) -> bool:
    if arm == "OFF":
        return True
    if arm == "CAP3":
        return row["n"] is not None and row["n"] < 3
    if arm == "PRECROSS":
        return bool(day_open_px) and row["price"] < day_open_px * (1 + CROSS)
    raise ValueError(arm)


def day_open(bars, idx, dt):
    """Open of the first 15m bar of dt's UTC day, or None if the store lacks it."""
    i = idx.get(dt.replace(hour=0, minute=0, second=0, microsecond=0))
    return float(bars[i][1]) if i is not None else None


def tail(cur_by_day, add_by_day, p):
    """(max drawdown of the cumulative daily sum, worst day); added trades weighted by p."""
    import _backtest_corr_guard_goal as C
    days = sorted(set(cur_by_day) | set(add_by_day))
    daily = [sum(cur_by_day.get(d, [])) + p * sum(add_by_day.get(d, [])) for d in days]
    return float(C.maxdd(daily)), float(min(daily)) if daily else 0.0


def verdict(goal_pt, goal_lo, trade_lo, *, dd_cur, dd_new, worst_cur, worst_new) -> str:
    tail_bad = dd_new < dd_cur * (1 + TAIL_TOL) or worst_new < worst_cur * (1 + TAIL_TOL)
    if goal_pt <= 0 or trade_lo < -0.10 or tail_bad:
        return "refuted"
    if goal_pt >= 0.5 and goal_lo > 0:
        return "supported"
    return "inconclusive"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import config as cfg
    import goal_validator as GV
    import indicators as I
    import pipeline_replay_validator as RV
    import _backtest_corr_guard_goal as C
    import _backtest_trend_start_detector as TD

    spec = RV.ReplaySpec(frozenset({"correlation_guard"}), None,
                         lambda e, c: bool(e["_blocked_here"]) if c.get("_on") else False,
                         ("_on",), frozenset({"relax"}), "gate removed entirely")
    rows = [e for e in RV._iter_rows(spec, C.SINCE)]
    blocked = collections.defaultdict(dict)
    for e in rows:
        if e["_blocked_here"]:
            tf = str(e.get("tf") or "")
            blocked[e["sym"]].setdefault(GV.bar_open(e["_dt"], tf if tf in GV.STEP else "15m"), e)
    entries, windows = GV.load_bot_log(C.SINCE)
    W = GV.winner_days(C.SINCE)
    p = C.p_after_guard(True)
    print("enforcement period from %s: winner-days %d, guard-blocked bars %d, p = %.3f"
          % (C.SINCE, len(W), sum(len(v) for v in blocked.values()), p))

    def in_pos(sym, d):
        return any(a <= d <= b for a, b in windows.get(sym, ()))

    s15 = {}

    def opens(sym):
        if sym not in s15:
            b = TD.bars_15m(sym)
            s15[sym] = (b, {x[0]: i for i, x in enumerate(b)})
        return s15[sym]
    cache = {}

    def series(sym, tf):
        if (sym, tf) not in cache:
            b = TD.load_bars(sym, tf) or []
            if b:
                import numpy as np
                h = np.array([x[2] for x in b], float)
                lo = np.array([x[3] for x in b], float)
                c = np.array([x[4] for x in b], float)
                cache[(sym, tf)] = (b, c, I._atr(h, lo, c, cfg.ATR_PERIOD), {x[0]: i for i, x in enumerate(b)})
            else:
                cache[(sym, tf)] = None
        return cache[(sym, tf)]
    ks = sorted(float(x[3]) for v in entries.values() for x in v if isinstance(x[3], (int, float)))
    K = ks[len(ks) // 2] if ks else 2.0
    cooldown = int(getattr(cfg, "COOLDOWN_BARS", 19))

    # current trades
    cur, cur_day = [], collections.defaultdict(list)
    for sym, lst in entries.items():
        for d, tf, mode, k in lst:
            if tf not in GV.STEP:
                continue
            s = series(sym, tf)
            i = s[3].get(GV.bar_open(d, tf) - GV.STEP[tf]) if s else None
            r = GV.live_trail(s[1], s[2], i, float(k) if isinstance(k, (int, float)) else K,
                              GV.trail_floor(mode, cfg), GV.MAX_BARS[tf]) if i is not None else None
            if r:
                cur.append(r[1])
                cur_day[d.strftime("%Y-%m-%d")].append(r[1])
    dd_cur, worst_cur = tail(cur_day, {}, p)
    print("current: %d trades, mean %+.3f%%, max drawdown %.1f, worst day %.1f" % (
        len(cur), sum(cur) / len(cur), dd_cur, worst_cur))

    alpha = (1 - CI) / 2
    rnd = random.Random(1016)
    for arm in ARMS:
        adm = collections.defaultdict(dict)
        for sym, bars in blocked.items():
            b15, i15 = opens(sym)
            for bo, e in bars.items():
                row = {"n": cluster_size(e.get("reason")), "price": float(e["price"])}
                if admit(arm, row, day_open(b15, i15, e["_dt"])):
                    adm[sym][bo] = e
        # goal
        contrib, now = [], 0
        for day, sym, op, dd in W:
            if any(op <= x[0] < dd for x in entries.get(sym, ())):
                now += 1
                contrib.append(0.0)
                continue
            k = sum(1 for e in adm.get(sym, {}).values() if op <= e["_dt"] < dd and not in_pos(sym, e["_dt"]))
            contrib.append(1 - (1 - p) ** k if k else 0.0)
        n = len(W)
        gb = sorted(100 * sum(rnd.choices(contrib, k=n)) / n for _ in range(BOOT))
        gpt = 100 * sum(contrib) / n
        glo, ghi = gb[int(alpha * BOOT)], gb[int((1 - alpha) * BOOT) - 1]
        # added trades, taken as the bot would
        add, add_day = [], collections.defaultdict(list)
        for sym in sorted(adm):
            busy = None
            for bo in sorted(adm[sym]):
                e = adm[sym][bo]
                tf = str(e.get("tf") or "")
                if tf not in GV.STEP or (busy is not None and bo <= busy) or in_pos(sym, e["_dt"]):
                    continue
                s = series(sym, tf)
                i = s[3].get(bo - GV.STEP[tf]) if s else None
                mode = e.get("signal_mode") or e.get("mode")
                r = GV.live_trail(s[1], s[2], i, K, GV.trail_floor(mode, cfg), GV.MAX_BARS[tf]) if i is not None else None
                if not r:
                    continue
                add.append(r[1])
                add_day[bo.strftime("%Y-%m-%d")].append(r[1])
                busy = s[0][r[0]][0] + GV.STEP[tf] * cooldown
        tpt = GV.combined_minus_current(cur, add, p) if add else 0.0
        tb = sorted(GV.combined_minus_current(rnd.choices(cur, k=len(cur)), rnd.choices(add, k=len(add)), p)
                    for _ in range(BOOT)) if add else [0.0] * BOOT
        tlo, thi = tb[int(alpha * BOOT)], tb[int((1 - alpha) * BOOT) - 1]
        dd_new, worst_new = tail(cur_day, add_day, p)
        v = verdict(gpt, glo, tlo, dd_cur=dd_cur, dd_new=dd_new, worst_cur=worst_cur, worst_new=worst_new)
        print("\n%s: admitted bars %d, added trades %d (mean %+.3f%%)" % (
            arm, sum(len(x) for x in adm.values()), len(add), sum(add) / len(add) if add else 0.0))
        print("  goal %.2f%% -> %.2f%%  delta %+.2f pp [%+.2f, %+.2f] (97.5%%)" % (
            100 * now / n, 100 * now / n + gpt, gpt, glo, ghi))
        print("  per trade combined-current %+.3f pp [%+.3f, %+.3f]" % (tpt, tlo, thi))
        print("  tail: max drawdown %.1f vs %.1f (%+.0f%%), worst day %.1f vs %.1f (%+.0f%%)" % (
            dd_new, dd_cur, 100 * (dd_new / dd_cur - 1), worst_new, worst_cur, 100 * (worst_new / worst_cur - 1)))
        print("  VERDICT %s: %s" % (arm, v))


if __name__ == "__main__":
    main()
