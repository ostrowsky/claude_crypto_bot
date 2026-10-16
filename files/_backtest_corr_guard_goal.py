"""correlation_guard gate-off, re-judged with the downstream pass rate measured after the
guard (portfolio limits follow it) and with the guard's own purpose -- tail risk -- checked.

Criteria fixed before the run (2026-10-16):
  goal      same as goal_validator: winner-days entered before the +2.5% crossing,
            expected gain sum(1-(1-p)^k); p = entries / coin-hours that passed the guard
            (entry or a later block), from critic_dataset since the enforcement start.
            Two variants: clone_signal_guard counted after the guard, or not.
  tail      the 10% worst days by the CURRENT trades' summed P&L: if the added (guard-
            blocked) trades on those days are worse than the current trades on those days
            by more than 0.10 pp per trade, the guard is doing its risk job.
"""
import collections
import io
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as cfg  # noqa: E402
import goal_validator as GV  # noqa: E402
import indicators as I  # noqa: E402
import pipeline_replay_validator as RV  # noqa: E402
import _backtest_trend_start_detector as TD  # noqa: E402

SINCE = "2026-07-07"          # CORR_GUARD_SHADOW=False (enforced) since this day
TAIL_SHARE = 0.10
TAIL_PP = 0.10


def maxdd(seq):
    """Deepest fall of the cumulative sum below its running peak (<= 0)."""
    peak = dd = acc = 0.0
    for v in seq:
        acc += v
        peak = max(peak, acc)
        dd = min(dd, acc - peak)
    return dd


def p_after_guard(with_clone):
    later = {"portfolio", "late_impulse_rotation"} | ({"clone_signal_guard"} if with_clone else set())
    ent, blk = set(), set()
    with io.open(HERE / "critic_dataset.jsonl", "rb") as fh:
        for l in fh:
            try:
                r = json.loads(l)
            except Exception:
                continue
            if str(r.get("ts_signal", "")) < SINCE:
                continue
            d = r.get("decision") or {}
            k = (r["sym"], r["ts_signal"][:13])
            if d.get("stage") == "entry":
                ent.add(k)
            elif d.get("stage") == "portfolio" and d.get("reason_code") in later:
                blk.add(k)
    blk -= ent
    return len(ent) / max(1, len(ent) + len(blk))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    spec = RV.ReplaySpec(frozenset({"correlation_guard"}), None,
                         lambda e, c: bool(e["_blocked_here"]) if c.get("_on") else False,
                         ("_on",), frozenset({"relax"}), "gate removed entirely")
    rows = list(RV._iter_rows(spec, SINCE))
    relaxed = collections.defaultdict(dict)
    for e in rows:
        if e["_blocked_here"]:
            tf = str(e.get("tf") or "")
            relaxed[e["sym"]].setdefault(GV.bar_open(e["_dt"], tf if tf in GV.STEP else "15m"), e)
    entries, windows = GV.load_bot_log(SINCE)

    def in_pos(sym, d):
        return any(a <= d <= b for a, b in windows.get(sym, ()))
    W = GV.winner_days(SINCE)
    print("winner-days %d, guard-blocked candidate bars %d" % (len(W), sum(len(v) for v in relaxed.values())))
    for with_clone in (False, True):
        p = p_after_guard(with_clone)
        now, exp, reach = 0, 0.0, 0
        contrib = []
        for day, sym, op, dd in W:
            if any(op <= x[0] < dd for x in entries.get(sym, ())):
                now += 1
                contrib.append(0.0)
                continue
            k = sum(1 for e in relaxed.get(sym, {}).values() if op <= e["_dt"] < dd and not in_pos(sym, e["_dt"]))
            g = 1 - (1 - p) ** k if k else 0.0
            reach += 1 if k else 0
            exp += g
            contrib.append(g)
        rnd = random.Random(16)
        n = len(W)
        bs = sorted(100 * sum(rnd.choices(contrib, k=n)) / n for _ in range(2000))
        print("p after guard (clone guard counted after: %s) = %.3f -> goal %.2f%% -> %.2f%%  delta %+.2f pp [%+.2f, %+.2f], "
              "newly reachable days %d, expected %.2f"
              % (with_clone, p, 100 * now / n, 100 * (now + exp) / n, 100 * exp / n, bs[50], bs[1949], reach, exp))

    # ---- tail: per-day P&L of current vs added trades (live trail, as the validator)
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
    cur, add = collections.defaultdict(list), collections.defaultdict(list)
    for sym in sorted(set(entries) | set(relaxed)):
        for d, tf, mode, k in entries.get(sym, ()):
            if tf not in GV.STEP:
                continue
            s = series(sym, tf)
            i = s[3].get(GV.bar_open(d, tf) - GV.STEP[tf]) if s else None
            r = GV.live_trail(s[1], s[2], i, float(k) if isinstance(k, (int, float)) else K,
                              GV.trail_floor(mode, cfg), GV.MAX_BARS[tf]) if i is not None else None
            if r:
                cur[d.strftime("%Y-%m-%d")].append(r[1])
        busy = None
        for bo in sorted(relaxed.get(sym, {})):
            e = relaxed[sym][bo]
            tf = str(e.get("tf") or "")
            if tf not in GV.STEP or (busy is not None and bo <= busy) or in_pos(sym, e["_dt"]):
                continue
            s = series(sym, tf)
            i = s[3].get(bo - GV.STEP[tf]) if s else None
            mode = e.get("signal_mode") or e.get("mode")
            r = GV.live_trail(s[1], s[2], i, K, GV.trail_floor(mode, cfg), GV.MAX_BARS[tf]) if i is not None else None
            if not r:
                continue
            add[bo.strftime("%Y-%m-%d")].append(r[1])
            busy = s[0][r[0]][0] + GV.STEP[tf] * cooldown
    days = sorted(set(cur) | set(add))
    tot = {d: sum(cur.get(d, [])) for d in days}
    worst = sorted(days, key=lambda d: tot[d])[:max(1, len(days) // 10)]
    cw = [v for d in worst for v in cur.get(d, [])]
    aw = [v for d in worst for v in add.get(d, [])]
    print("\nTAIL: %d days, worst 10%% = %d days (by current summed P&L)" % (len(days), len(worst)))
    print("  on those days: current trades n=%d mean %+.3f%% | added trades n=%d mean %+.3f%% | diff %+.3f pp"
          % (len(cw), sum(cw) / max(1, len(cw)), len(aw), sum(aw) / max(1, len(aw)),
             sum(aw) / max(1, len(aw)) - sum(cw) / max(1, len(cw))))
    rest_c = [v for d in days if d not in worst for v in cur.get(d, [])]
    rest_a = [v for d in days if d not in worst for v in add.get(d, [])]
    print("  other days:    current mean %+.3f%% (n=%d) | added mean %+.3f%% (n=%d)"
          % (sum(rest_c) / max(1, len(rest_c)), len(rest_c), sum(rest_a) / max(1, len(rest_a)), len(rest_a)))
    for d in worst[:6]:
        print("   %s current sum %+.1f%% (n=%d) | added sum %+.1f%% (n=%d)" % (
            d, tot[d], len(cur.get(d, [])), sum(add.get(d, [])), len(add.get(d, []))))
    adds = sorted(days, key=lambda d: sum(add.get(d, [])))
    print("  most negative ADDED-trade days:", [(d, round(sum(add[d]), 1), len(add[d]), round(tot[d], 1)) for d in adds[:6] if d in add])
    comb = sorted(tot[d] + sum(add.get(d, [])) for d in days)
    cu = sorted(tot.values())
    print("  daily summed P&L, current: worst %.1f, 5th pct %.1f | combined: worst %.1f, 5th pct %.1f"
          % (cu[0], cu[len(cu) // 20], comb[0], comb[len(comb) // 20]))
    print("  max drawdown of the cumulative daily sum: current %.1f | combined %.1f | totals %.1f vs %.1f"
          % (maxdd([tot[d] for d in days]), maxdd([tot[d] + sum(add.get(d, [])) for d in days]),
             sum(tot.values()), sum(tot.values()) + sum(sum(v) for v in add.values())))
    worst_comb = sorted(days, key=lambda d: tot[d] + sum(add.get(d, [])))[:3]
    print("  worst combined days:", [(d, round(tot[d] + sum(add.get(d, [])), 1)) for d in worst_comb],
          "| worst current days:", [(d, round(tot[d], 1)) for d in worst[:3]])


if __name__ == "__main__":
    main()
