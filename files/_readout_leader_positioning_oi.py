"""Pre-registered recheck LEADER-POSITIONING-OI: does a rising open interest (oi_1h)
tell (A) which day-leaders keep rising and (D) which rank <= 3 crossing entries pay?

WHY (2026-10-21, docs/reports/2026-10-18-leader-positioning-readout.md)
The first positioning test (60 days, 2026-08-21..10-19) found one feature that
pointed the same way in both time halves: oi_1h at the leader alert, AUC 0.617
(train) / 0.698 (test). With 9 test positives that is "too early", and a second
search over hundreds of rules on the same data would be another fishing trip
(TH-12). So the rule is fixed NOW, before its data exists:

  rule     oi_1h of the latest positioning snapshot (<= 3h old) >= the MEDIAN of
           oi_1h over the same population in the window already seen
           (2026-08-21..10-19). The two medians below are frozen constants.
  judged   only on days from 2026-10-20 (nobody has looked at them).

  A  population  leader alerts replayed with leader_alert.find_leaders (first
                 qualifying bar per coin per day, max 3 a day), with a snapshot.
     label       "continues": a 15m close after the alert, before the UTC day
                 ends, >= alert price + 5% AND the day closes above the alert price
     TOO_EARLY   < 30 positives in the population
     SUPPORTED   Wilson 95% lower bound of the rule's precision > base rate
     REFUTED     rule precision <= base rate (or the rule selects nothing)
     otherwise   INCONCLUSIVE
     reported    mean pnl of the rule's alerts on the k=3 / 8% / 7-day trail
                 (_backtest_leader_mode_v2.py) with 95% bootstrap -- information
  D  population  rank <= 3 crossings exactly as _backtest_crossing_entry.py builds
                 them (first 15m close >= +2.5% of the UTC open, one per coin per
                 day, bot up all day), with a snapshot; outcome sim_exit
     TOO_EARLY   < 60 rule entries
     SUPPORTED   precision (share on immutable top-20 winner-days) >= the bot's
                 own 15m entries on the same days AND mean pnl lower 95% >= bot
                 mean - 0.10 pp AND it reaches >= 10% of the late/none winner-days
     REFUTED     otherwise

Runs as a pipeline step before readouts.py (`--if-due`: nothing before DUE),
writes RESULT; readouts.readout_leader_positioning_oi only reads that file, so
the heavy grid never runs inside the readouts step's timeout. A verdict is a
recommendation: SUPPORTED means "propose a shadow", never "apply".
Spec: docs/specs/features/leader-positioning-oi-1021-spec.md
"""
from __future__ import annotations

import argparse
import bisect
import collections
import io
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

UTC = timezone.utc
SEEN_FROM, SEEN_TO = "2026-08-21", "2026-10-19"
NEW_FROM = "2026-10-20"
DUE = "2026-12-16"
# medians of oi_1h over the seen window (python _readout_leader_positioning_oi.py --seen, 2026-10-21)
OI1H_MEDIAN_A = 1.336080   # 129 leader alerts with a snapshot, 32 continuations
OI1H_MEDIAN_D = 0.111583   # 328 rank<=3 crossings with a snapshot, 56 full bot-up days
MIN_POS_A = 30
MIN_N_D = 60
MAX_AGE = timedelta(hours=3)
CONT_UP = 0.05
TRIG = 0.025
POSITIONING = HERE / "positioning_history.jsonl"
RESULT = ROOT / ".runtime" / "backtests" / "leader_positioning_oi.json"


# ---------------------------------------------------------------- pure parts

def passes(pos, threshold) -> bool:
    v = (pos or {}).get("oi_1h")
    return isinstance(v, (int, float)) and math.isfinite(v) and v >= threshold


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def verdict_a(pos_pop: int, n_pop: int, k_rule: int, n_rule: int) -> str:
    if pos_pop < MIN_POS_A:
        return "TOO_EARLY"
    base = pos_pop / n_pop
    if n_rule == 0 or k_rule / n_rule <= base:
        return "REFUTED"
    return "SUPPORTED" if wilson(k_rule, n_rule)[0] > base else "INCONCLUSIVE"


def verdict_d(n_rule, prec, bot_prec, pnl_lo, bot_mean, reach, n_late) -> str:
    if n_rule < MIN_N_D:
        return "TOO_EARLY"
    ok = prec >= bot_prec and pnl_lo >= bot_mean - 0.10 and reach >= 0.10 * n_late
    return "SUPPORTED" if ok else "REFUTED"


def overall(a: str, d: str) -> str:
    if "TOO_EARLY" in (a, d):
        return "TOO_EARLY"
    if "SUPPORTED" in (a, d):
        return "SUPPORTED"
    if "INCONCLUSIVE" in (a, d):
        return "INCONCLUSIVE"
    return "REFUTED"


def stage_of(events: list, op: datetime, t25: datetime) -> str:
    """held / early / late / none, as _research_late_entry.py: events with event + _dt."""
    prior = [e for e in events if e["_dt"] < op and e["event"] in ("entry", "exit")]
    if prior and sorted(prior, key=lambda e: e["_dt"])[-1]["event"] == "entry":
        return "held"
    ents = sorted((e for e in events if e["event"] == "entry" and op <= e["_dt"] < op + timedelta(days=1)),
                  key=lambda e: e["_dt"])
    if not ents:
        return "none"
    return "early" if ents[0]["_dt"] < t25 else "late"


def should_run(today: str) -> bool:
    return today >= DUE


def boot_ci(v, seed=7, B=1000):
    if not v:
        return (None, None)
    rnd = random.Random(seed)
    bs = sorted(sum(rnd.choices(v, k=len(v))) / len(v) for _ in range(B))
    return bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


# ---------------------------------------------------------------- data

def load_positioning(path: Path | None = None) -> dict:
    by = collections.defaultdict(list)
    with io.open(path or POSITIONING, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
                t = datetime.fromisoformat(r["ts"])
            except Exception:
                continue
            by[r["sym"]].append((t if t.tzinfo else t.replace(tzinfo=UTC), r))
    out = {}
    for s, v in by.items():
        v.sort(key=lambda x: x[0])
        out[s] = ([x[0] for x in v], [x[1] for x in v])
    return out


def snap_at(pos: dict, sym: str, t: datetime):
    if sym not in pos:
        return None
    ts, rows = pos[sym]
    k = bisect.bisect_right(ts, t) - 1
    if k < 0 or t - ts[k] > MAX_AGE:
        return None
    return rows[k]


def _days(a: str, b: str) -> list:
    d, out = datetime.strptime(a, "%Y-%m-%d"), []
    while d.strftime("%Y-%m-%d") <= b:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def replay_alerts(g, days: list) -> list:
    import numpy as np
    import leader_alert as LA
    rmax, min_ret, hold, max_day = LA.params()
    out = []
    for day in days:
        d = (datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC) - g.t0).days
        if d < 0 or (d + 1) * 96 > g.N:
            continue
        base = d * 96
        opens = {s: g.O[base, j] for s, j in g.col.items() if math.isfinite(g.O[base, j]) and g.O[base, j] > 0}
        done = []
        for b in range(96):
            if len(done) >= max_day:
                break
            dd = {}
            for s, op in opens.items():
                col = g.C[base:base + b + 1, g.col[s]]
                closes = {k: float(col[k]) for k in range(b + 1) if math.isfinite(col[k])}
                if closes:
                    dd[s] = (op, closes)
            if len(dd) < 20:
                continue
            for x in LA.find_leaders(dd, rmax, min_ret, hold):
                if x["sym"] in done or len(done) >= max_day:
                    continue
                done.append(x["sym"])
                j, i, ap = g.col[x["sym"]], base + b, x["price"]
                after = g.C[i + 1:base + 96, j]
                after = after[np.isfinite(after)]
                close = g.C[base + 95, j]
                cont = bool(len(after) and after.max() >= ap * (1 + CONT_UP) and math.isfinite(close) and close > ap)
                out.append({"day": day, "sym": x["sym"], "j": j, "i": i, "t": g.t0 + (i + 1) * timedelta(minutes=15),
                            "cont": cont})
    return out


def trail_v2(g, i, j, K=3.0, FLOOR=0.08, MAX_HOLD=96 * 7):
    """_backtest_leader_mode_v2.exit_of without the lead exit; pnl in %."""
    C, A = g.C, g.ATR
    ep = C[i, j]
    stop = ep - max(K * A[i, j], FLOOR * ep)
    last = min(g.N - 1, i + MAX_HOLD)
    for q in range(i + 1, last + 1):
        cq = C[q, j]
        if not math.isfinite(cq):
            continue
        if math.isfinite(A[q, j]):
            stop = max(stop, cq - max(K * A[q, j], FLOOR * cq))
        if cq < stop:
            return (cq / ep - 1) * 100
    q = last
    while q > i and not math.isfinite(C[q, j]):
        q -= 1
    return (C[q, j] / ep - 1) * 100


def crossings(g, days: set) -> list:
    import numpy as np
    out = []
    for d in range(g.days):
        day = (g.t0 + timedelta(days=d)).strftime("%Y-%m-%d")
        if day not in days:
            continue
        seg = g.RET[d * 96:(d + 1) * 96]
        for j, sym in enumerate(g.syms):
            hit = np.where(seg[:, j] >= TRIG)[0]
            if len(hit) == 0:
                continue
            i = d * 96 + int(hit[0])
            if i + 1 >= g.N or int(g.RANK[i, j]) > 3:
                continue
            out.append({"day": day, "sym": sym, "j": j, "i": i, "t": g.t0 + (i + 1) * timedelta(minutes=15)})
    return out


def late_like(days: set, winners: set) -> set:
    """(day, sym) winner-days entered late or never by the bot (all timeframes)."""
    import label_store as LS
    dl = LS.intraday_deadlines()
    keys = {k for k in winners if k[0] in days and k in dl and dl[k][1] is not None}
    syms = {s for _, s in keys}
    since = (datetime.strptime(min(days), "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d") if days else "9999"
    ev = collections.defaultdict(list)
    with io.open(HERE / "bot_events.jsonl", "rb") as fh:
        for raw in fh:
            if b'"entry"' not in raw and b'"exit"' not in raw:
                continue
            try:
                e = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue
            if e.get("event") not in ("entry", "exit") or e.get("sym") not in syms or str(e.get("ts", "")) < since:
                continue
            d = datetime.fromisoformat(str(e["ts"]).replace("Z", "+00:00"))
            e["_dt"] = d if d.tzinfo else d.replace(tzinfo=UTC)
            ev[e["sym"]].append(e)
    return {k for k in keys if stage_of(ev.get(k[1], []), dl[k][0], dl[k][1]) in ("late", "none")}


def compute(from_day: str, to_day: str, th_a=None, th_d=None) -> dict:
    import numpy as np
    import config as cfg
    import exit_validator as EV
    import immutable_labels as IL
    import _compute_early_capture as E
    import _backtest_crossing_entry as XE
    th_a = OI1H_MEDIAN_A if th_a is None else th_a
    th_d = OI1H_MEDIAN_D if th_d is None else th_d
    g = EV.load_grid(cfg.ATR_PERIOD)
    pos = load_positioning()
    wl = E.load_watchlist()
    win, _ = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
    winners = set(win)
    labelled = {d for d, _ in winners}
    to_day = min(to_day, max(labelled))
    days = _days(from_day, to_day)
    out = {"computed": datetime.now(UTC).strftime("%Y-%m-%d"), "window": [from_day, to_day],
           "oi1h_median_a": th_a, "oi1h_median_d": th_d}

    # A
    al = [a for a in replay_alerts(g, days) if a["day"] in labelled]
    for a in al:
        a["pos"] = snap_at(pos, a["sym"], a["t"])
    A = [a for a in al if a["pos"] is not None]
    pos_pop = sum(a["cont"] for a in A)
    rule = [a for a in A if th_a is not None and passes(a["pos"], th_a)]
    k_rule = sum(a["cont"] for a in rule)
    pnl = [trail_v2(g, a["i"], a["j"]) for a in rule]
    lo, hi = boot_ci(pnl)
    out["A"] = {"alerts": len(al), "with_positioning": len(A), "pos_pop": pos_pop,
                "base": round(pos_pop / len(A), 4) if A else None, "n_rule": len(rule), "k_rule": k_rule,
                "precision": round(k_rule / len(rule), 4) if rule else None,
                "wilson95": [round(x, 4) for x in wilson(k_rule, len(rule))],
                "rule_pnl_mean": round(float(np.mean(pnl)), 3) if pnl else None,
                "rule_pnl_ci95": [lo, hi], "oi1h_values": [a["pos"].get("oi_1h") for a in A],
                "verdict": verdict_a(pos_pop, len(A), k_rule, len(rule)) if A and th_a is not None else "TOO_EARLY"}

    # D
    full, _, _ = E.load_uptime(datetime.strptime(from_day, "%Y-%m-%d").replace(tzinfo=UTC))
    fdays = {d for d in days if d in full}
    P = EV.leader_params(cfg)
    trades = EV.load_trades(g, "2026-03-01")
    ks = sorted(t["k"] for t in trades if t["mode"] == "impulse_speed")
    KX = ks[len(ks) // 2] if ks else 2.0
    floor = EV.floor_of("impulse_speed", cfg)
    cr = crossings(g, fdays)
    for c in cr:
        c["pos"] = snap_at(pos, c["sym"], c["t"])
    D = [c for c in cr if c["pos"] is not None]
    sel = [c for c in D if th_d is not None and passes(c["pos"], th_d)]
    vals = [XE.sim_exit(g, c["j"], c["i"], KX, floor, P)[0] for c in sel]
    prec = (sum(1 for c in sel if (c["day"], c["sym"]) in winners) / len(sel)) if sel else 0.0
    dlo, dhi = boot_ci(vals)
    bot_days = [t for t in trades if t["day"] in fdays]
    bv, _ = EV.leader_arm(g, bot_days, P)
    bot = [(t, v) for t, v in zip(bot_days, bv) if v is not None]
    bot_prec = (sum(1 for t, _ in bot if (t["day"], t["sym"]) in winners) / len(bot)) if bot else 0.0
    bot_mean = float(np.mean([v for _, v in bot])) if bot else 0.0
    ll = late_like(fdays, winners)
    reach = len({(c["day"], c["sym"]) for c in sel} & ll)
    out["D"] = {"full_days": len(fdays), "crossings": len(cr), "with_positioning": len(D), "n_rule": len(sel),
                "precision": round(prec, 4), "pnl_mean": round(float(np.mean(vals)), 3) if vals else None,
                "pnl_ci95": [dlo, dhi], "bot": {"n": len(bot), "precision": round(bot_prec, 4), "mean": round(bot_mean, 3)},
                "late_like": len(ll), "reach": reach, "oi1h_values": [c["pos"].get("oi_1h") for c in D],
                "verdict": verdict_d(len(sel), prec, bot_prec, dlo if dlo is not None else -1e9, bot_mean, reach, len(ll))
                if th_d is not None else "TOO_EARLY"}
    out["verdict"] = overall(out["A"]["verdict"], out["D"]["verdict"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--if-due", action="store_true", help="do nothing before DUE")
    ap.add_argument("--seen", action="store_true", help="print the medians over the seen window (freezing step)")
    ap.add_argument("--today", default=None)
    a = ap.parse_args(argv)
    today = a.today or datetime.now(UTC).strftime("%Y-%m-%d")
    if a.seen:
        import numpy as np
        res = compute(SEEN_FROM, SEEN_TO, th_a=float("inf"), th_d=float("inf"))
        va = [v for v in res["A"]["oi1h_values"] if isinstance(v, (int, float))]
        vd = [v for v in res["D"]["oi1h_values"] if isinstance(v, (int, float))]
        print("seen window %s: A alerts with positioning %d (median oi_1h %.6f); D crossings with positioning %d"
              " (median oi_1h %.6f)" % (res["window"], len(va), float(np.median(va)), len(vd), float(np.median(vd))))
        return 0
    if a.if_due and not should_run(today):
        print("[leader_positioning_oi] collecting until %s, nothing to do" % DUE)
        return 0
    prev = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else {}
    if prev.get("verdict") not in (None, "TOO_EARLY"):
        print("[leader_positioning_oi] final: %s" % prev["verdict"])
        return 0
    yesterday = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    res = compute(NEW_FROM, yesterday)
    for k in ("A", "D"):
        res[k].pop("oi1h_values", None)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print("[leader_positioning_oi] %s  A=%s D=%s" % (res["verdict"], res["A"]["verdict"], res["D"]["verdict"]))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
