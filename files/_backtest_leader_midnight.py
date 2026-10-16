"""Leader-mode 'lost the lead' exit across the UTC day reset -- offline replay, nothing deployed.

WHY (2026-10-05, X-9c readout read-through)
The leader exit (leader_exit.py) closes a leader position on the FIRST bar the coin
is outside the day's top-10, ranked by return since the UTC open. At 00:00 UTC
every return restarts from ~0 and the rank of the first bars is close to random.
Live since 2026-09-29: 13 of 48 'lost top-10' exits fell in the 00 UTC hour (no
other hour above 5); the 15 that crossed a midnight after the switch were +1.59%
on average when closed. Whether the move continued after them is not observable
live -- hence this replay.

DESIGN, FIXED BEFORE THE RUN
  engine   exit_validator: same trades (every real 15m trade with an exit since
           2026-03-01), same switch rule (top-5 of the day at +3%), same 8% wide
           trail, same rocket-day / winner-day labels and statistics.
  current  live rule: out of the day's top-10 on any bar >= 4 bars after the switch.
  A        the lost-lead check is skipped during the first 8 bars (2 h) of a UTC day.
  B        the lost-lead check ranks by the trailing 24 h return (96 bars) instead of
           the return since the UTC open.
  verdict  exit_validator.decide: rocket lower95 > 0 AND all lower95 >= -0.10 pp AND
           >= 60% of months >= 0. Two variants are tested; read the intervals with that
           multiplicity in mind.
Run from files/:  ../pyembed/python.exe _backtest_leader_midnight.py
"""
from __future__ import annotations

import collections
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config                         # noqa: E402
import exit_validator as EV           # noqa: E402

SKIP_BARS_A = 8
WINDOW_B = 96
OUT = HERE.parent / ".runtime" / "leader_midnight_backtest.json"


def rank_trailing(grid, w):
    """Rank (1 = best) of the trailing-w-bar return across symbols at each bar."""
    C = grid.C[: grid.N]
    ret = np.full_like(C, np.nan)
    ret[w:] = C[w:] / C[:-w] - 1
    rank = np.full(C.shape, 999, dtype=int)
    for i in range(C.shape[0]):
        row = ret[i]
        ok = np.isfinite(row)
        if ok.any():
            order = np.argsort(-np.where(ok, row, -9))
            rank[i, order] = np.arange(1, C.shape[1] + 1)
            rank[i, ~ok] = 999
    return rank


def wide_from(grid, t, s, P, lost_rank_arr, skip_first=0):
    """exit_validator.wide_from with the rank array for the lost-lead check and an optional day-start skip.
    Returns (pnl %, exit bar, exit reason)."""
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
        if cq < stop:
            return (cq / ep - 1) * 100, q, "trail"
        if q - s >= P["lost_min_bars"] and (q % 96) >= skip_first and lost_rank_arr[q, j] > P["lost_rank"]:
            return (cq / ep - 1) * 100, q, "lost"
    q = last
    while q > s and not math.isfinite(C[q, j]):
        q -= 1
    return (C[q, j] / ep - 1) * 100, q, "max_hold"


def arm(grid, trades, P, lost_rank_arr, skip_first=0):
    m = grid.lead(P["rank_max"], P["min_ret"])
    vals, meta = [], []
    for t in trades:
        hits = np.where(m[t["ie"]:t["ix"], t["j"]])[0]
        if len(hits):
            s = t["ie"] + int(hits[0])
            pnl, qx, why = wide_from(grid, t, s, P, lost_rank_arr, skip_first)
            vals.append(pnl)
            meta.append((s, qx, why))
        else:
            vals.append(None if t["leader_exit_live"] else t["real"])
            meta.append(None)
    return vals, meta


def summarize(trades, cur, new, rockets, winners, rnd_seed=29):
    rows = [(t, x, y) for t, x, y in zip(trades, cur, new) if x is not None and y is not None]
    rnd = random.Random(rnd_seed)
    d_all = [y - x for _, x, y in rows]
    d_rk = [y - x for t, x, y in rows if (t["day"], t["sym"]) in rockets]
    d_wn = [y - x for t, x, y in rows if (t["day"], t["sym"]) in winners]
    changed = sum(1 for v in d_all if abs(v) > 1e-9)
    bym = collections.defaultdict(list)
    for t, x, y in rows:
        if (t["day"], t["sym"]) in rockets:
            bym[t["day"][:7]].append(y - x)
    months = {mo: {"mean": round(sum(v) / len(v), 3), "n": len(v)} for mo, v in sorted(bym.items())}
    scored = [mo for mo, v in months.items() if v["n"] >= EV.MONTH_MIN]
    agree = (sum(1 for mo in scored if months[mo]["mean"] >= 0) / len(scored)) if scored else 0.0
    all_s, rk_s, wn_s = EV.paired(d_all, rnd), EV.paired(d_rk, rnd), EV.paired(d_wn, rnd)
    rk_now = [x for t, x, _ in rows if (t["day"], t["sym"]) in rockets]
    rk_new = [y for t, _, y in rows if (t["day"], t["sym"]) in rockets]
    return {"trades": len(rows), "changed_trades": changed, "all_trades_diff_pp": all_s, "rocket_day_diff_pp": rk_s,
            "winner_day_diff_pp": wn_s, "rocket_day_mean_now": round(float(np.mean(rk_now)), 3) if rk_now else None,
            "rocket_day_mean_after": round(float(np.mean(rk_new)), 3) if rk_new else None,
            "rocket_months": months, "month_agreement": round(agree, 3), "months_scored": len(scored),
            "verdict": EV.decide(all_s, rk_s, agree, changed)}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    grid = EV.load_grid(int(getattr(config, "ATR_PERIOD", 14)))
    P = EV.leader_params(config)
    trades = EV.load_trades(grid, EV.SINCE_DEFAULT)
    rockets, winners = EV.day_labels()
    print(f"grid {len(grid.syms)} symbols to {grid.last_bar:%Y-%m-%d}; trades since {EV.SINCE_DEFAULT}: {len(trades)}; params {P}", flush=True)
    day_rank = grid.RANK
    r24 = rank_trailing(grid, WINDOW_B)
    cur, meta_c = arm(grid, trades, P, day_rank, 0)
    ref, _ = EV.leader_arm(grid, trades, P)                       # the replay must equal exit_validator's own arm
    mism = sum(1 for x, y in zip(cur, ref) if (x is None) != (y is None) or (x is not None and abs(x - y) > 1e-9))
    if mism:
        raise SystemExit(f"current arm differs from exit_validator.leader_arm on {mism} trades")
    a, meta_a = arm(grid, trades, P, day_rank, SKIP_BARS_A)
    b, meta_b = arm(grid, trades, P, r24, 0)
    # how the current rule exits: lost-lead exits by hour of the UTC day
    lost_hours = collections.Counter((m[1] % 96) // 4 for m in meta_c if m and m[2] == "lost")
    reasons = {name: dict(collections.Counter(m[2] for m in mm if m)) for name, mm in (("current", meta_c), ("A", meta_a), ("B", meta_b))}
    hold = {name: round(float(np.median([(m[1] - m[0]) / 4 for m in mm if m])), 2) for name, mm in (("current", meta_c), ("A", meta_a), ("B", meta_b))}
    report = {"since": EV.SINCE_DEFAULT, "params": P, "switched_trades": sum(1 for m in meta_c if m),
              "current_lost_exits_by_utc_hour": dict(sorted(lost_hours.items())), "exit_reasons": reasons, "median_hold_hours": hold,
              "A_skip_first_bars": SKIP_BARS_A, "B_window_bars": WINDOW_B,
              "A": summarize(trades, cur, a, rockets, winners), "B": summarize(trades, cur, b, rockets, winners)}
    print(f"switched trades {report['switched_trades']}; current lost-lead exits by UTC hour: {report['current_lost_exits_by_utc_hour']}")
    print(f"exit reasons: {reasons}\nmedian hold after switch (h): {hold}")
    for k in ("A", "B"):
        r = report[k]
        print(f"\n[{k}] verdict={r['verdict']} trades={r['trades']} changed={r['changed_trades']}\n"
              f"  rocket-days {r['rocket_day_diff_pp']}  (mean now {r['rocket_day_mean_now']} -> {r['rocket_day_mean_after']})\n"
              f"  all {r['all_trades_diff_pp']}\n  winners {r['winner_day_diff_pp']}\n"
              f"  months agree {r['month_agreement']:.0%} of {r['months_scored']}: {r['rocket_months']}", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
