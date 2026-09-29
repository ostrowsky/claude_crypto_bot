"""L3 goal validator -- judge a threshold change by what the bot is FOR.

WHY THIS EXISTS (2026-09-29)

`pipeline_replay_validator` grades a change by the forward 4h PEAK of the rows
that change side. That is a proxy (TH-11): on 2026-09-29 it rated the L2
hypothesis "TREND_1H_CHOP_ADX_MIN_BULL_DAY 22 -> 20" at 1.56x and parked it as
needs_review, while the project's own criterion -- applied by hand over the
maximum period -- found it adds no winner-day and fails per-trade
non-inferiority (chop-bull-adx-spec.md). Every P-series decision since
2026-09-25 was made with that criterion by hand; this module makes it the
automatic L3 judge, so the agent is graded by the goal and not by a proxy.

THE CRITERION, FIXED BEFORE ANY HYPOTHESIS IS GRADED

  population  rows of bot_events.jsonl that REACHED the gate (RV._iter_rows,
              deduplicated per coin-hour), replayed under the current and the
              proposed value with the same logged inputs (TH-06)
  goal        immutable later-EOD top-20 winner-days (watchlist, bot up all
              day, first +2.5% crossing known): share the bot entered BEFORE
              the crossing.
                relax    expected = now + sum over newly reachable days of
                         1-(1-p)^k, k = flipped candidates inside the birth
                         window while not in a position; p = the logged rate at
                         which candidates that passed this gate became entries
                tighten  days whose every early entry is removed are lost
  per trade   calibrated live_trail engine (P-1), both arms from the decision
              bar, trail_k = the entry's own (median of the arm for flipped
              rows), trail floor of the mode; 95% bootstrap interval of
              (combined mean - current mean), 1000 resamples
  months      mean of the changed trades per month vs the current arm

  relax    accept   goal >= +0.5 pp AND non-inferior (lower >= -0.10 pp at
                    weight p) AND the changed trades match or beat the current
                    arm in >= 50% of scored months
           reject   goal <= 0 OR the lower bound < -0.10 pp
  tighten  accept   goal loss < 0.5 pp AND per-trade lower bound > 0
           reject   goal loss >= 0.5 pp OR the per-trade upper bound < 0
  needs_data  < 30 winner-days or < 20 changed / current trades -- re-run next week
  otherwise   needs_review

Accept means "the evidence supports it", never "apply it" -- approval stays
with the operator. The legacy peak replay travels beside this verdict as
secondary evidence and is not collapsed into it (CLAUDE.md §0 rule 3).
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

import pipeline_replay_validator as RV  # noqa: E402

SINCE_DEFAULT = "2026-05-01"   # first month of the current gate set and of the P-1 calibration
GOAL_PP = 0.5
NONINF_PP = -0.10
MIN_DAYS = 30
MIN_TRADES = 20
MONTH_MIN = 5
MONTH_AGREE = 0.5
BOOT = 1000
STEP = {"15m": timedelta(minutes=15), "1h": timedelta(hours=1)}
MAX_BARS = {"15m": 96, "1h": 48}


def _dt(ts):
    try:
        d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def bar_open(d: datetime, tf: str) -> datetime:
    """Open time of the bar that was FORMING at d; the decision used the one before."""
    if tf == "1h":
        return d.replace(minute=0, second=0, microsecond=0)
    return d.replace(minute=d.minute // 15 * 15, second=0, microsecond=0)


def trail_floor(mode, cfg) -> float:
    if not getattr(cfg, "TRAIL_MIN_BUFFER_PCT_ENABLED", False):
        return 0.0
    name = "TRAIL_MIN_BUFFER_PCT_" + str(mode or "DEFAULT").upper()
    return float(getattr(cfg, name, getattr(cfg, "TRAIL_MIN_BUFFER_PCT_DEFAULT", 0.0)))


def live_trail(c, atr, ie, k, floor=0.0, max_bars=96):
    """P-1 calibrated live trail: close-anchored ratchet, exit at the first close
    below the stop. Returns (exit_index, pnl_pct) or None."""
    ep = c[ie]
    a0 = atr[ie] if math.isfinite(atr[ie]) else 0.0
    stop = ep - max(k * a0, floor * ep)
    last = min(len(c) - 1, ie + max_bars)
    if last <= ie:
        return None
    for j in range(ie + 1, last + 1):
        aj = atr[j] if math.isfinite(atr[j]) else 0.0
        if aj > 0:
            stop = max(stop, c[j] - max(k * aj, floor * c[j]))
        if c[j] < stop:
            return j, (c[j] / ep - 1) * 100
    return last, (c[last] / ep - 1) * 100


def load_bot_log(since: str, events_path=None):
    """entries: sym -> [(dt, tf, mode, trail_k)]; windows: sym -> [(start, end)] held."""
    entries, windows = collections.defaultdict(list), collections.defaultdict(list)
    with io.open(events_path or RV.EVENTS, "rb") as fh:
        for raw in fh:
            if b'"entry"' not in raw and b'"exit"' not in raw:
                continue
            try:
                e = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue
            ev = e.get("event")
            if ev not in ("entry", "exit") or str(e.get("ts", "")) < since:
                continue
            d = _dt(e.get("ts"))
            if d is None or not e.get("sym"):
                continue
            tf = str(e.get("tf") or "")
            if ev == "entry":
                entries[e["sym"]].append((d, tf, e.get("mode") or e.get("signal_mode"), e.get("trail_k")))
            elif isinstance(e.get("bars_held"), (int, float)):
                step = STEP.get(tf, STEP["15m"])
                windows[e["sym"]].append((d - step * (int(e["bars_held"]) + 1), d))
    return entries, windows


def winner_days(since: str, full_days=None):
    """[(day, sym, open_dt, deadline_dt)] -- immutable top-20 on the watchlist,
    bot up all day, crossing known."""
    import _compute_early_capture as E
    import immutable_labels as IL
    import label_store as LS
    wl = E.load_watchlist()
    if full_days is None:
        full_days, _, _ = E.load_uptime(_dt(since[:10] + "T00:00:00+00:00"))
    win, _ = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
    dl = LS.intraday_deadlines()
    out = []
    for k in sorted(win):
        if k[0] < since[:10] or k[0] not in full_days or k not in dl or dl[k][1] is None:
            continue
        out.append((k[0], k[1], dl[k][0], dl[k][1]))
    return out


def combined_minus_current(cur, changed, weight=1.0, remove=False):
    """Mean pnl change: add `changed` at `weight` (relax) or take it out (tighten)."""
    base = sum(cur) / len(cur)
    if remove:
        kept = len(cur) - len(changed)
        return (sum(cur) - sum(changed)) / kept - base if kept > 0 else 0.0
    return (sum(cur) + weight * sum(changed)) / (len(cur) + weight * len(changed)) - base


def bootstrap(cur, changed, weight, rnd, remove=False):
    """(point, lo95, hi95). For `remove`, `cur` holds only the KEPT trades and
    the two parts are resampled separately, so the removed set stays a subset."""
    if remove:
        full = cur + changed
        point = combined_minus_current(full, changed, remove=True)
        bs = []
        for _ in range(BOOT):
            k, r = rnd.choices(cur, k=len(cur)), rnd.choices(changed, k=len(changed))
            bs.append(combined_minus_current(k + r, r, remove=True))
    else:
        point = combined_minus_current(cur, changed, weight)
        bs = [combined_minus_current(rnd.choices(cur, k=len(cur)), rnd.choices(changed, k=len(changed)), weight)
              for _ in range(BOOT)]
    bs.sort()
    return point, bs[int(0.025 * BOOT)], bs[int(0.975 * BOOT) - 1]


def decide(direction: str, delta_pp: float, lo: float, hi: float, agree: float) -> str:
    if direction == "relax":
        if delta_pp <= 0 or lo < NONINF_PP:
            return "reject"
        if delta_pp >= GOAL_PP and agree >= MONTH_AGREE:
            return "accept"
        return "needs_review"
    if delta_pp <= -GOAL_PP or hi < 0:
        return "reject"
    if lo > 0:
        return "accept"
    return "needs_review"


def validate(hyp: dict, since: str = SINCE_DEFAULT, cfg_module=None, events_path=None,
             full_days=None, bars_loader=None, winners=None) -> dict:
    key = str(hyp.get("config_key") or "")
    diff = hyp.get("diff") or {}
    base = {"validator": "goal_validator", "config_key": key, "since": since,
            "criterion": "winner-days entered before the +2.5% crossing; per-trade non-inferiority -0.10 pp"}
    if cfg_module is None:
        import config as cfg_module
    spec = RV.REPLAY_SPECS.get(key)
    if spec is None or not hasattr(cfg_module, key):
        return {**base, "verdict": "reject",
                "reason": f"no goal replay for '{key}' (keys: {', '.join(RV.validatable_keys())})"}
    try:
        new_v = float(diff.get("to"))
    except (TypeError, ValueError):
        return {**base, "verdict": "reject", "reason": "diff.to is not a number"}
    cur = {k: getattr(cfg_module, k) for k in spec.inputs if hasattr(cfg_module, k)}
    cur_v = float(cur[key])
    if new_v == cur_v:
        return {**base, "verdict": "reject", "reason": "diff.to equals the live value"}
    new = dict(cur, **{key: new_v})
    window = max(since, spec.valid_since or since)

    old_events = RV.EVENTS
    if events_path is not None:
        RV.EVENTS = Path(events_path)
    try:
        rows = list(RV._iter_rows(spec, window))
    finally:
        RV.EVENTS = old_events

    relaxed, tightened, past = collections.defaultdict(dict), [], collections.defaultdict(set)
    for e in rows:
        b_cur, b_new = spec.blocks(e, cur), spec.blocks(e, new)
        hour = e["_dt"].replace(minute=0, second=0, microsecond=0)
        if not e["_blocked_here"]:
            past[(e["sym"], hour)].add("entry" if e.get("event") == "entry" else "b")
        if b_cur is None or b_new is None:
            continue
        tf = str(e.get("tf") or "")
        # only rows this gate BLOCKED on the day can be admitted by relaxing it,
        # only rows that PASSED it can be removed by tightening (see RV.validate)
        if b_cur and not b_new and e["_blocked_here"]:
            relaxed[e["sym"]].setdefault(bar_open(e["_dt"], tf if tf in STEP else "15m"), e)
        elif b_new and not b_cur and not e["_blocked_here"] and e.get("event") == "entry":
            tightened.append(e)
    n_relaxed = sum(len(v) for v in relaxed.values())
    direction = "relax" if n_relaxed >= len(tightened) else "tighten"
    if direction not in spec.directions:
        return {**base, "verdict": "reject",
                "reason": f"'{key}' can only be replayed as {sorted(spec.directions)}: {spec.note}"}
    p = sum(1 for v in past.values() if "entry" in v) / max(1, len(past))

    entries, windows = load_bot_log(window, events_path)

    def in_position(sym, d):
        return any(a <= d <= b for a, b in windows.get(sym, ()))

    W = winners if winners is not None else winner_days(window, full_days)
    now = reach = lost = 0
    expct = 0.0
    removed_ids = {(e["sym"], e["_dt"]) for e in tightened}
    for day, sym, op, dd in W:
        early = [x for x in entries.get(sym, ()) if op <= x[0] < dd]
        if early:
            now += 1
            if direction == "tighten" and all((sym, x[0]) in removed_ids for x in early):
                lost += 1
            continue
        if direction == "relax":
            f = [b for b, e in relaxed.get(sym, {}).items()
                 if op <= e["_dt"] < dd and not in_position(sym, e["_dt"])]
            if f:
                reach += 1
                expct += 1 - (1 - p) ** len(f)
    n = len(W)
    goal_now = now / n if n else 0.0
    goal_new = ((now + expct) if direction == "relax" else (now - lost)) / n if n else 0.0
    goal = {"winner_days": n, "entered_before_crossing_now": round(100 * goal_now, 2),
            "expected_after": round(100 * goal_new, 2),
            "delta_pp": round(100 * (goal_new - goal_now), 2)}
    if direction == "relax":
        goal.update(upper_after=round(100 * (now + reach) / n, 2) if n else 0.0,
                    newly_reachable_days=reach, downstream_pass_rate_p=round(p, 3))
    else:
        goal.update(days_lost=lost)

    # ---- per trade ----
    if bars_loader is None:
        import _backtest_trend_start_detector as TD
        bars_loader = TD.load_bars
    import indicators as I
    import numpy as np
    cooldown = int(getattr(cfg_module, "COOLDOWN_BARS", 8))
    ks = [float(x[3]) for v in entries.values() for x in v
          if (spec.tf is None or x[1] == spec.tf) and (not spec.control_mode or x[2] == spec.control_mode)
          and isinstance(x[3], (int, float))]
    K = sorted(ks)[len(ks) // 2] if ks else 2.0
    cur_tr, add_tr, rem_tr = [], [], []
    cache = {}

    def series(sym, tf):
        if (sym, tf) not in cache:
            b = bars_loader(sym, tf) or []
            if b:
                h = np.array([x[2] for x in b], dtype=float)
                lo_ = np.array([x[3] for x in b], dtype=float)
                c = np.array([x[4] for x in b], dtype=float)
                cache[(sym, tf)] = (b, c, I._atr(h, lo_, c, cfg_module.ATR_PERIOD),
                                    {x[0]: i for i, x in enumerate(b)})
            else:
                cache[(sym, tf)] = None
        return cache[(sym, tf)]

    for sym in sorted(set(entries) | set(relaxed)):
        for d, tf, mode, k in entries.get(sym, ()):
            if tf not in STEP or (spec.tf and tf != spec.tf) or (spec.control_mode and mode != spec.control_mode):
                continue
            s = series(sym, tf)
            i = s[3].get(bar_open(d, tf) - STEP[tf]) if s else None
            r = live_trail(s[1], s[2], i, float(k) if isinstance(k, (int, float)) else K,
                           trail_floor(mode, cfg_module), MAX_BARS[tf]) if i is not None else None
            if r:
                (rem_tr if (sym, d) in removed_ids else cur_tr).append((d.strftime("%Y-%m"), r[1]))
        busy = None
        for bo in sorted(relaxed.get(sym, {})):
            e = relaxed[sym][bo]
            tf = str(e.get("tf") or "")
            if tf not in STEP or (busy is not None and bo <= busy) or in_position(sym, e["_dt"]):
                continue
            s = series(sym, tf)
            i = s[3].get(bo - STEP[tf]) if s else None
            mode = e.get("signal_mode") or e.get("mode") or spec.control_mode or None
            r = live_trail(s[1], s[2], i, K, trail_floor(mode, cfg_module), MAX_BARS[tf]) if i is not None else None
            if not r:
                continue
            add_tr.append((bo.strftime("%Y-%m"), r[1]))
            busy = s[0][r[0]][0] + STEP[tf] * cooldown
    if direction == "relax":
        cur_all, changed = cur_tr, add_tr
    else:
        cur_all, changed = cur_tr + rem_tr, rem_tr      # cur_tr holds only the KEPT trades here
    cc = [x[1] for x in cur_all]
    ch = [x[1] for x in changed]
    trade = {"trail_k_median": round(K, 2), "current": {"n": len(cc)}, "changed": {"n": len(ch)}}
    if cc:
        trade["current"].update(mean=round(float(np.mean(cc)), 3), median=round(float(np.median(cc)), 3))
    if ch:
        trade["changed"].update(mean=round(float(np.mean(ch)), 3), median=round(float(np.median(ch)), 3))

    report = {**base, "direction": direction, "from": cur_v, "to": new_v, "goal": goal, "per_trade": trade}
    if n < MIN_DAYS or len(ch) < MIN_TRADES or len(cc) < MIN_TRADES:
        return {**report, "verdict": "needs_data",
                "reason": (f"{n} winner-days (need {MIN_DAYS}), {len(ch)} changed and {len(cc)} current "
                           f"trades (need {MIN_TRADES}); re-evaluated on the next run")}
    rnd = random.Random(29)
    if direction == "relax":
        pt, lo, hi = bootstrap(cc, ch, p, rnd)
    else:
        pt, lo, hi = bootstrap([x[1] for x in cur_tr], ch, 1.0, rnd, remove=True)
    trade["combined_minus_current_pp"] = {"point": round(pt, 3), "lo95": round(lo, 3), "hi95": round(hi, 3),
                                          "weight": round(p, 3) if direction == "relax" else 1.0}
    bym_c, bym_a = collections.defaultdict(list), collections.defaultdict(list)
    for m, v in cur_all:
        bym_c[m].append(v)
    for m, v in changed:
        bym_a[m].append(v)
    months = {m: {"changed": round(float(np.mean(v)), 3), "n": len(v),
                  "current": round(float(np.mean(bym_c[m])), 3) if bym_c.get(m) else None}
              for m, v in sorted(bym_a.items())}
    scored = [m for m, x in months.items() if x["n"] >= MONTH_MIN and x["current"] is not None]
    agree = (sum(1 for m in scored if months[m]["changed"] >= months[m]["current"]) / len(scored)) if scored else 0.0
    trade.update(by_month=months, month_agreement=round(agree, 3), months_scored=len(scored))
    d_pp = goal["delta_pp"]
    desc = (f"{direction} {key} {cur_v} -> {new_v}: goal {goal['entered_before_crossing_now']:.1f}% -> "
            f"{goal['expected_after']:.1f}% ({d_pp:+.2f} pp) on {n} winner-days; changed trades "
            f"{trade['changed'].get('mean', 0):+.2f}% (n={len(ch)}) vs current "
            f"{trade['current'].get('mean', 0):+.2f}% (n={len(cc)}); combined-current {pt:+.3f} pp "
            f"[{lo:+.3f}, {hi:+.3f}]; months agree {agree:.0%} of {len(scored)}")
    return {**report, "verdict": decide(direction, d_pp, lo, hi, agree), "reason": desc}


def combine(goal_res: dict | None, peak_res: dict) -> dict:
    """L3 final verdict: the goal validator decides; the 4h-peak replay is kept
    beside it as secondary evidence (never averaged in). If the goal validator
    could not run, the peak verdict stands and says so."""
    if not goal_res or goal_res.get("error"):
        out = dict(peak_res)
        out["goal_validator"] = goal_res or {"error": "not run"}
        out["reason"] = str(out.get("reason", "")) + " [goal validator unavailable: peak replay only]"
        return out
    out = dict(goal_res)
    out["peak_replay"] = {k: peak_res.get(k) for k in
                          ("verdict", "reason", "band", "control", "mean_ratio", "tail_ratio",
                           "month_consistency") if k in peak_res}
    return out


if __name__ == "__main__":
    import argparse
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--to", type=float, required=True)
    ap.add_argument("--since", default=SINCE_DEFAULT)
    a = ap.parse_args()
    print(json.dumps(validate({"config_key": a.key, "diff": {"to": a.to}}, a.since),
                     ensure_ascii=False, indent=1))
