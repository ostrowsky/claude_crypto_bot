"""Pre-registered readouts of live changes, read out by code every day.

WHY (2026-09-29, agent-tasks-0929-spec.md)

Each live change of the last week was shipped with a promise: "read it out on
day X against the backtest". Those promises lived as free-text prompts of
one-off Claude tasks, which (a) recompute the definitions each time, so two
readouts of one change need not measure the same thing, (b) run once and cannot
say "too early, look again tomorrow", and (c) cannot raise an alarm before the
due date when something is visibly broken (a 1h position in leader mode).

Here each readout is an entry of readout_registry.json, fixed before its data
existed: window, minimum comparable days, expectation and the keep / rollback
criteria. The runner computes every readout daily:

  before due     status COLLECTING with days so far; anomaly checks already run
  at/after due   verdict KEEP / ROLLBACK_SUGGESTED / INCONCLUSIVE, or TOO_EARLY
                 while the minimum comparable days or trades are not there --
                 then it is re-read daily until a verdict, and frozen (final)
  delegated      research readouts stay with their scheduled Claude task; the
                 runner only reports whether the data suffices

A verdict is a recommendation. The runner never changes config (the rollback in
each entry is for the operator). An LLM may phrase a final verdict in Russian
for the morning report; it receives the computed numbers and cannot change the
verdict (READOUTS_LLM_SUMMARY_ENABLED).
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import math
import random
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

REGISTRY = HERE / "readout_registry.json"
EVENTS = HERE / "bot_events.jsonl"
OUT_DIR = ROOT / ".runtime" / "pipeline" / "readouts"
TS_RE = re.compile(rb'"ts":\s*"([^"]+)"')
BOOT = 1000


# ---------------------------------------------------------------- helpers

def load_registry(path: Path = REGISTRY) -> list:
    return json.loads(path.read_text(encoding="utf-8"))["readouts"]


def _dt(ts):
    d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def scan_events(since: str, kinds: set, path: Path = EVENTS) -> list:
    """Events of the given kinds at or after `since` (seeks, does not read the whole log)."""
    import incident_analyst as IA
    out = []
    kb = [f'"{k}"'.encode() for k in kinds]
    with io.open(path, "rb") as fh:
        fh.seek(IA.seek_offset(fh, since))
        for raw in fh:
            if not any(k in raw for k in kb):
                continue
            m = TS_RE.search(raw)
            if not m or m.group(1).decode("ascii", "replace") < since:
                continue
            try:
                e = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue
            if e.get("event") in kinds:
                e["_dt"] = _dt(e["ts"])
                out.append(e)
    return out


def comparable_days(window_from: str, until: str, full_days: set) -> list:
    """Days in [window_from, until) with the bot up all day AND immutable labels."""
    import immutable_labels as IL
    import _compute_early_capture as E
    win, _ = IL.winners_by_day(top_n=20, watchlist=E.load_watchlist(), rank_before_filter=True)
    labelled = {d for d, _ in win}
    d, end, out = datetime.strptime(window_from, "%Y-%m-%d"), datetime.strptime(until, "%Y-%m-%d"), []
    while d < end:
        k = d.strftime("%Y-%m-%d")
        if k in full_days and k in labelled:
            out.append(k)
        d += timedelta(days=1)
    return out


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def mean_ci(v: list, rnd: random.Random):
    if not v:
        return None
    bs = sorted(sum(rnd.choices(v, k=len(v))) / len(v) for _ in range(BOOT))
    return {"n": len(v), "mean": round(sum(v) / len(v), 3), "lo95": round(bs[int(0.025 * BOOT)], 3),
            "hi95": round(bs[int(0.975 * BOOT) - 1], 3)}


def pair_trades(events: list) -> list:
    """[(entry, exit)] -- each entry with the next exit of the same symbol and timeframe."""
    by = collections.defaultdict(list)
    for e in sorted(events, key=lambda e: e["_dt"]):
        by[(e.get("sym"), e.get("tf"))].append(e)
    out = []
    for evs in by.values():
        open_e = None
        for e in evs:
            if e["event"] == "entry":
                open_e = e
            elif e["event"] == "exit" and open_e is not None:
                out.append((open_e, e))
                open_e = None
    return out


def label_days():
    """{(day, sym): record} of the immutable label store."""
    import label_store as LS
    return {(r["utc_day"], r["symbol"]): r for r in LS.LabelStore().records()}


def is_rocket(rec) -> bool:
    o, h, c = float(rec.get("open") or 0), float(rec.get("high") or 0), float(rec.get("close") or 0)
    return o > 0 and h >= o * 1.10 and (c - o) >= 0.6 * (h - o)


# ---------------------------------------------------------------- readouts

def readout_e4(r: dict, ctx: dict) -> dict:
    days = ctx["days"]
    if not days:
        return {"days": 0}
    since = days[0]
    evs = scan_events(since, {"entry", "exit", "tq_zero_forecast_pass"})
    evs = [e for e in evs if e["ts"][:10] in set(days) or e["event"] == "exit"]
    passes = [e for e in evs if e["event"] == "tq_zero_forecast_pass"]
    trades = [(a, b) for a, b in pair_trades([e for e in evs if e["event"] in ("entry", "exit")])
              if a.get("tf") == "15m" and (a.get("mode") or a.get("signal_mode")) == "trend" and a["ts"][:10] in set(days)]
    rel = [float(b["pnl_pct"]) for a, b in trades if a.get("tq_zero_forecast_relaxed") and isinstance(b.get("pnl_pct"), (int, float))]
    oth = [float(b["pnl_pct"]) for a, b in trades if not a.get("tq_zero_forecast_relaxed") and isinstance(b.get("pnl_pct"), (int, float))]
    import goal_validator as GV
    W = [w for w in GV.winner_days(since, set(days)) if w[0] in set(days)]
    ent = collections.defaultdict(list)
    for e in evs:
        if e["event"] == "entry":
            ent[e["sym"]].append(e)
    early = relaxed_early = 0
    for day, sym, op, dd in W:
        ee = [e for e in ent.get(sym, ()) if op <= e["_dt"] < dd]
        if ee:
            early += 1
            relaxed_early += any(e.get("tq_zero_forecast_relaxed") for e in ee)
    out = {"days": len(days), "tq_zero_forecast_pass_events": len(passes),
           "relaxed_entries": sum(1 for a, _ in trades if a.get("tq_zero_forecast_relaxed")),
           "winner_days": len(W), "entered_before_crossing": early,
           "goal_pct": round(100 * early / len(W), 1) if W else None,
           "relaxed_early_on_winners": relaxed_early,
           "relaxed_trades": mean_ci(rel, random.Random(29)), "other_trades": mean_ci(oth, random.Random(30))}
    if rel and oth:
        pt, lo, hi = GV.bootstrap(oth, rel, 1.0, random.Random(31))
        out["combined_minus_other_pp"] = {"point": round(pt, 3), "lo95": round(lo, 3), "hi95": round(hi, 3)}
    out["_n_trades"] = len(rel)
    c = out.get("combined_minus_other_pp")
    out["_verdict"] = ("ROLLBACK_SUGGESTED" if c and c["hi95"] < -0.10 else
                       "KEEP" if c and c["lo95"] >= -0.10 and relaxed_early >= 1 else "INCONCLUSIVE")
    return out


def readout_leader_exit(r: dict, ctx: dict) -> dict:
    days = ctx["days"]
    since = r["window_from"]
    evs = scan_events(since, {"entry", "exit", "leader_exit_switch"})
    now = ctx["now"]
    switches = [e for e in evs if e["event"] == "leader_exit_switch"]
    trades = pair_trades([e for e in evs if e["event"] in ("entry", "exit")])
    t15 = [(a, b) for a, b in trades if a.get("tf") == "15m"]
    anomalies = [f"1h switch {e['sym']} {e['ts'][:16]}" for e in switches if e.get("tf") != "15m"]
    for a, b in trades:
        n = [s for s in switches if s["sym"] == a["sym"] and a["_dt"] <= s["_dt"] <= b["_dt"]]
        if len(n) > 1:
            anomalies.append(f"switched {len(n)}x {a['sym']} {a['ts'][:16]}")
    open_since = {}
    for e in sorted(evs, key=lambda e: e["_dt"]):
        if e["event"] == "leader_exit_switch":
            open_since.setdefault(e["sym"], e["_dt"])
        elif e["event"] == "exit":
            open_since.pop(e["sym"], None)
    for sym, t in open_since.items():
        if now - t > timedelta(days=8):
            anomalies.append(f"leader position {sym} open since {t.isoformat(timespec='minutes')}")
    sw_ids = {(s["sym"],) for s in switches}
    lab = label_days()
    sw_pnl, no_pnl, rocket_pnl = [], [], []
    exits = collections.Counter()
    for a, b in t15:
        if not isinstance(b.get("pnl_pct"), (int, float)) or a["ts"][:10] not in set(days):
            continue
        p = float(b["pnl_pct"])
        sw = any(s["sym"] == a["sym"] and a["_dt"] <= s["_dt"] <= b["_dt"] for s in switches)
        (sw_pnl if sw else no_pnl).append(p)
        rec = lab.get((a["ts"][:10], a["sym"]))
        if rec and is_rocket(rec):
            rocket_pnl.append(p)
        if str(b.get("reason", "")).startswith("режим лидера"):
            exits[str(b["reason"]).split(":")[0][:40]] += 1
    n15 = sum(1 for a, _ in t15 if a["ts"][:10] in set(days))
    rk = mean_ci(rocket_pnl, random.Random(29))
    out = {"days": len(days), "switches": len(switches), "trades_15m": n15,
           "switched_share": round(len(sw_pnl) / max(1, len(sw_pnl) + len(no_pnl)), 3),
           "switched_trades": mean_ci(sw_pnl, random.Random(30)), "other_trades": mean_ci(no_pnl, random.Random(31)),
           "rocket_day_trades": rk, "leader_exit_reasons": dict(exits), "anomalies": anomalies,
           "note": "the old exit of a switched trade is not observable live; compare with the backtest levels"}
    out["_n_trades"] = len(rocket_pnl)
    out["_anomaly"] = bool(anomalies)
    base = r["expected"]["old_exit_rocket_day_mean_pct"]
    out["_verdict"] = ("ROLLBACK_SUGGESTED" if anomalies or (rk and rk["hi95"] < base) else
                       "KEEP" if rk and rk["mean"] >= base else "INCONCLUSIVE")
    return out


def readout_leader_alert(r: dict, ctx: dict) -> dict:
    days = set(ctx["days"])
    alerts = [e for e in scan_events(r["window_from"], {"leader_alert"}) if e["ts"][:10] in days]
    import immutable_labels as IL
    import _compute_early_capture as E
    wl = E.load_watchlist()
    win, _ = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
    lab = label_days()
    per_day = collections.defaultdict(list)
    for (d, s), rec in lab.items():
        if d in days and s in wl:
            per_day[d].append((float(rec["eod_return_pct"]), s))
    top3 = {d: {s for _, s in sorted(v, reverse=True)[:3]} for d, v in per_day.items()}
    hit3 = sum(1 for e in alerts if e["sym"] in top3.get(e["ts"][:10], set()))
    hit20 = sum(1 for e in alerts if (e["ts"][:10], e["sym"]) in win)
    n = len(alerts)
    lo3, hi3 = wilson(hit3, n)
    out = {"days": len(days), "alerts": n, "alerts_per_day": round(n / len(days), 2) if days else None,
           "top3_watchlist_hit": {"k": hit3, "n": n, "rate": round(hit3 / n, 3) if n else None,
                                  "wilson95": [round(lo3, 3), round(hi3, 3)]},
           "global_top20_hit": {"k": hit20, "n": n, "rate": round(hit20 / n, 3) if n else None,
                                "wilson95": [round(x, 3) for x in wilson(hit20, n)]}}
    out["_n_trades"] = n
    exp = r["expected"]["top3_watchlist_hit"]
    out["_verdict"] = "ROLLBACK_SUGGESTED" if n and hi3 < exp else "KEEP" if n else "INCONCLUSIVE"
    return out


def readout_delegated(r: dict, ctx: dict) -> dict:
    out = {"days": len(ctx["days"]), "scheduled_task": r.get("scheduled_task")}
    if r["id"] == "LEADER-POSITIONING":
        p = HERE / "positioning_history.jsonl"
        seen = set()
        if p.exists():
            with io.open(p, "rb") as fh:
                for raw in fh:
                    m = TS_RE.search(raw)
                    if m:
                        seen.add(m.group(1)[:10])
        out["positioning_days"] = len(seen)
        out["days"] = len(seen)
    elif r["id"] == "P-2":
        hb = ROOT / ".runtime" / "poll_heartbeat"
        hb_days = {f.stem for f in hb.glob("*.jsonl")} if hb.exists() else set()
        out["heartbeat_days_bot_up"] = len([d for d in ctx["days"] if d in hb_days])
        out["days"] = out["heartbeat_days_bot_up"]
    return out


def _peak_label(sym, tf, dt, price, bars_ahead, thr_pct):
    """1/0: did the high reach +thr_pct within `bars_ahead` bars after the decision bar."""
    import goal_validator as GV
    import pipeline_replay_validator as RV
    if tf not in GV.STEP or not price:
        return None
    bars, idx = RV._bars(sym, tf)
    i = idx.get(GV.bar_open(dt, tf) - GV.STEP[tf])
    if i is None or i + bars_ahead >= len(bars):
        return None
    hi = max(b[2] for b in bars[i + 1: i + 1 + bars_ahead])
    return 1.0 if (hi / float(price) - 1) * 100 >= thr_pct else 0.0


def readout_ml_floor_shadow(r: dict, ctx: dict) -> dict:
    """Per-model ML floor in SHADOW vs the fixed floor, on the bot's own candidates.

    Rows: every candidate that reached the ML gate (blocked there, blocked later, or
    entered), deduplicated per coin-hour, with its logged ml_proba. The calibrated
    floor of a row is the one logged at the last model load before it.
    """
    import pipeline_replay_validator as RV
    import config as cfg
    evs = scan_events(r["window_from"], {"ml_floor_calibration", "blocked", "entry"})
    cals = sorted((e["_dt"], float(e["calibrated_floor"]), float(e.get("live_floor") or 0.0)) for e in evs
                  if e["event"] == "ml_floor_calibration" and isinstance(e.get("calibrated_floor"), (int, float)))
    if not cals:
        return {"days": 0, "note": "no model with a calibrated floor has been loaded yet", "_n_trades": 0}
    first = cals[0][0]
    days = set(d for d in ctx["days"] if d >= first.strftime("%Y-%m-%d"))
    horizon = int(getattr(cfg, "ML_PEAK_LABEL_HORIZON", 5))
    thr = float(getattr(cfg, "ML_PEAK_LABEL_THRESHOLD_PCT", 2.0))
    seen, rows = set(), []
    for e in evs:
        if e["event"] == "ml_floor_calibration" or e["_dt"] < first or e["ts"][:10] not in days:
            continue
        g = RV._gate_of(e)
        if g != "ENTRY" and RV.STAGE.get(g, 0) < 1:
            continue                                   # stopped before the ML gate
        p = e.get("ml_proba")
        if not isinstance(p, (int, float)) or not e.get("sym"):
            continue
        k = (e["sym"], e["_dt"].replace(minute=0, second=0, microsecond=0))
        if k in seen:
            continue
        seen.add(k)
        cal = [c for c in cals if c[0] <= e["_dt"]][-1]
        y = _peak_label(e["sym"], str(e.get("tf") or ""), e["_dt"], e.get("price"), horizon, thr)
        rows.append({"sym": e["sym"], "dt": e["_dt"], "p": float(p), "cal": cal[1], "fixed": cal[2], "y": y})

    def st(key):
        lab = [x for x in rows if x["y"] is not None]
        adm = [x for x in lab if x["p"] >= x[key]]
        pos = [x for x in lab if x["y"] > 0.5]
        tp = sum(1 for x in adm if x["y"] > 0.5)
        return {"admitted_share": round(len(adm) / len(lab), 3) if lab else None,
                "recall": round(tp / len(pos), 3) if pos else None,
                "precision": round(tp / len(adm), 3) if adm else None, "rows": len(lab)}
    import goal_validator as GV
    W = [w for w in GV.winner_days(min(days), set(days)) if w[0] in days] if days else []
    reach = {"fixed": 0, "cal": 0}
    has = 0
    for day, sym, op, dd in W:
        pre = [x for x in rows if x["sym"] == sym and op <= x["dt"] < dd]
        if not pre:
            continue
        has += 1
        for key in reach:
            reach[key] += any(x["p"] >= x[key] for x in pre)
    fixed, cal = st("fixed"), st("cal")
    out = {"days": len(days), "model_loads": len(cals), "calibrated_floors": sorted({round(c[1], 4) for c in cals}),
           "fixed_floor": fixed, "calibrated_floor": cal,
           "winner_days_with_candidate": has, "reached_fixed": reach["fixed"], "reached_calibrated": reach["cal"],
           "base_rate": round(sum(1 for x in rows if x["y"]) / max(1, sum(1 for x in rows if x["y"] is not None)), 3)}
    out["_n_trades"] = fixed["rows"]
    if reach["cal"] < reach["fixed"]:
        out["_verdict"] = "ROLLBACK_SUGGESTED"
    elif (cal["precision"] or 0) >= (fixed["precision"] or 0):
        out["_verdict"] = "KEEP"
    else:
        out["_verdict"] = "INCONCLUSIVE"
    return out


BLACKOUT_SHARE = 0.85      # 08-20 blackout = 0.886 by coin-hour; max of the other 97 days since June 0.791
BLACKOUT_MIN_ROWS = 50


def readout_ml_reload(r: dict, ctx: dict) -> dict:
    """ML model hot reload: were models swapped, and did a swap black out the gate?"""
    import pipeline_replay_validator as RV
    evs = scan_events(r["window_from"], {"ml_model_reload", "blocked", "entry"})
    reloads = [e for e in evs if e["event"] == "ml_model_reload"]
    per_day = collections.defaultdict(lambda: [set(), set()])      # day -> [reached ML gate, blocked by it]
    for e in evs:
        if e["event"] == "ml_model_reload":
            continue
        g = RV._gate_of(e)
        if g != "ENTRY" and RV.STAGE.get(g, 0) < 1:
            continue
        k = (e.get("sym"), e["_dt"].replace(minute=0, second=0, microsecond=0))
        d = per_day[e["ts"][:10]]
        d[0].add(k)
        if g in ("ml_zone", "ml_proba_zone"):
            d[1].add(k)
    days = {d: {"candidates": len(v[0]), "ml_blocked_share": round(len(v[1]) / len(v[0]), 3) if v[0] else None}
            for d, v in sorted(per_day.items())}
    anomalies = [f"{d}: ML gate blocked {x['ml_blocked_share']:.0%} of {x['candidates']} candidates"
                 for d, x in days.items()
                 if x["candidates"] >= BLACKOUT_MIN_ROWS and (x["ml_blocked_share"] or 0) >= BLACKOUT_SHARE]
    try:
        mt = (HERE / "ml_signal_model.json").stat().st_mtime
        last = max((e.get("file_mtime") or 0) for e in reloads) if reloads else 0
        age_h = (ctx["now"].timestamp() - mt) / 3600 if ctx.get("now") else 0
        if mt > last + 3600 and age_h > 24 and r["window_from"] <= datetime.fromtimestamp(mt, timezone.utc).strftime("%Y-%m-%d"):
            anomalies.append("model file changed %.0f h ago but no reload was logged" % age_h)
    except Exception:
        pass
    out = {"days": len(ctx["days"]), "reloads": len(reloads),
           "models": [f"{e.get('old_model')}->{e.get('new_model')} {e['ts'][:16]}" for e in reloads][-10:],
           "ml_block_share_by_day": days, "anomalies": anomalies}
    out["_n_trades"] = len(reloads)
    out["_anomaly"] = bool(anomalies)
    out["_verdict"] = "KEEP" if reloads and not anomalies else "INCONCLUSIVE"
    return out


def readout_mrq_off(r: dict, ctx: dict) -> dict:
    """mode_range_quality OFF: the entries it would have blocked, vs the rest."""
    import goal_validator as GV
    days = set(ctx["days"])
    evs = scan_events(r["window_from"], {"mode_range_shadow", "entry", "exit"})
    shadow = collections.defaultdict(list)
    for e in evs:
        if e["event"] == "mode_range_shadow":
            shadow[(e.get("sym"), e.get("tf"))].append(e["_dt"])
    trades = [(a, b) for a, b in pair_trades([e for e in evs if e["event"] in ("entry", "exit")])
              if a["ts"][:10] in days]

    def flagged(a):
        lag = timedelta(minutes=20 if a.get("tf") == "15m" else 90)
        return any(timedelta(0) <= a["_dt"] - t <= lag for t in shadow.get((a.get("sym"), a.get("tf")), ()))
    fl = [(a, b) for a, b in trades if flagged(a)]
    ot = [(a, b) for a, b in trades if not flagged(a)]
    f_p = [float(b["pnl_pct"]) for _, b in fl if isinstance(b.get("pnl_pct"), (int, float))]
    o_p = [float(b["pnl_pct"]) for _, b in ot if isinstance(b.get("pnl_pct"), (int, float))]
    W = {(d, s): dd for d, s, _, dd in GV.winner_days(min(days), days)} if days else {}
    early = {(a["ts"][:10], a["sym"]) for a, _ in fl if (a["ts"][:10], a["sym"]) in W and a["_dt"] < W[(a["ts"][:10], a["sym"])]}
    nd = max(1, len(days))
    out = {"days": len(days), "added_entries": len(fl), "added_per_day": round(len(fl) / nd, 2),
           "other_entries": len(ot), "added_trades": mean_ci(f_p, random.Random(29)),
           "other_trades": mean_ci(o_p, random.Random(30)), "early_winner_days_from_added": len(early)}
    if f_p and o_p:
        pt, lo, hi = GV.bootstrap(o_p, f_p, 1.0, random.Random(31))
        out["combined_minus_other_pp"] = {"point": round(pt, 3), "lo95": round(lo, 3), "hi95": round(hi, 3)}
    out["_n_trades"] = len(f_p)
    out["anomalies"] = ["%.1f added entries/day > 20" % (len(fl) / nd)] if len(fl) / nd > 20 else []
    out["_anomaly"] = bool(out["anomalies"])
    c = out.get("combined_minus_other_pp")
    out["_verdict"] = ("ROLLBACK_SUGGESTED" if c and c["hi95"] < -0.10 else
                       "KEEP" if c and c["lo95"] >= -0.10 else "INCONCLUSIVE")
    return out


PROMOTION_LOG = ROOT / ".runtime" / "ml_promotion_log.jsonl"
ADM_SHARE_TOL = 0.07       # |admitted share difference| the matched floor may drift on new rows
ADM_PREC_TOL = 0.03        # precision may not be shown worse by more than this
ADM_SHARE_BAND = (0.50, 0.85)  # incumbent's admitted share outside this = ANOMALY


def readout_ml_admission_match(r: dict, ctx: dict) -> dict:
    """Matched-admission floor in SHADOW: on rows that arrived after a candidate was
    built, does the candidate at its stored floor admit the incumbent's share, with
    no clearly worse precision or winner-day reach? Rows come from the nightly
    `transfer` blocks of ml_promotion_log.jsonl (ml-admission-match-1009-spec.md)."""
    recs = []
    try:
        for line in open(PROMOTION_LOG, encoding="utf-8"):
            if line.strip():
                recs.append(json.loads(line))
    except Exception:
        pass
    recs = [x for x in recs if str(x.get("ts", ""))[:10] >= r["window_from"]]
    errors = [x["ts"] for x in recs if x.get("matched_error")]
    seen, rows, nights = set(), [], 0
    for x in recs:
        tr = x.get("transfer") or {}
        new = 0
        for ts, sym, y, a_old, a_new in tr.get("rows") or []:
            if (ts, sym) in seen:
                continue
            seen.add((ts, sym))
            rows.append((ts[:10], sym, int(y), int(a_old), int(a_new)))
            new += 1
        nights += 1 if new else 0
    floors = [x.get("admission_floor") for x in recs if x.get("admission_floor")]
    # matching freezes the operating point: a share far outside what the fixed floor
    # admitted over the maximum period (0.56..0.75 by block) is locked in, not corrected
    # (walk-forward anchored on a one-week model: 10 vs 27 of 42 winner-days)
    off_band = sorted({"%s %s=%.3f" % (str(x.get("ts", ""))[:10], k, v)
                       for x in recs for k, v in ((x.get("admission_floor") or {}).get("incumbent_share") or {}).items()
                       if isinstance(v, (int, float)) and not (ADM_SHARE_BAND[0] <= v <= ADM_SHARE_BAND[1])})
    out = {"days": nights, "nights_logged": len(recs), "rows": len(rows), "share_off_band": off_band,
           "matched_decisions": collections.Counter((x.get("matched") or {}).get("decision") for x in recs),
           "fixed_decisions": collections.Counter(x.get("decision") for x in recs if not x.get("matched_live")),
           "last_admission_floor": floors[-1] if floors else None, "gate_errors": errors}
    out["_n_trades"] = len(rows)
    out["_anomaly"] = bool(errors or off_band)
    if not rows:
        out["note"] = "no rows newer than a candidate's holdout yet"
        return out
    try:
        import exit_validator as EV
        _, winners = EV.day_labels()
    except Exception:
        winners = set()
    rnd = random.Random(29)

    def stat(rs):
        n = len(rs)
        ao, an = sum(x[3] for x in rs), sum(x[4] for x in rs)
        tpo, tpn = sum(1 for x in rs if x[3] and x[2]), sum(1 for x in rs if x[4] and x[2])
        w = [x for x in rs if (x[0], x[1]) in winners]
        return {"share_diff": (an - ao) / n,
                "prec_diff": (tpn / an - tpo / ao) if an and ao else None,
                "win_diff": (sum(x[4] for x in w) - sum(x[3] for x in w)) / len(w) if w else None}

    point = stat(rows)
    bs = {k: [] for k in point}
    for _ in range(BOOT):
        st = stat(rnd.choices(rows, k=len(rows)))
        for k, v in st.items():
            if v is not None:
                bs[k].append(v)

    def ci(k):
        v = sorted(bs[k])
        if not v or point[k] is None:
            return None
        return {"mean": round(point[k], 4), "lo95": round(v[int(0.025 * len(v))], 4),
                "hi95": round(v[int(0.975 * len(v)) - 1], 4), "n": len(v)}
    sh, pr, wn = ci("share_diff"), ci("prec_diff"), ci("win_diff")
    n = len(rows)
    out.update(admitted_old=round(sum(x[3] for x in rows) / n, 3), admitted_new=round(sum(x[4] for x in rows) / n, 3),
               base_rate=round(sum(x[2] for x in rows) / n, 3),
               winner_rows=sum(1 for x in rows if (x[0], x[1]) in winners),
               share_diff=sh, precision_diff=pr, winner_rows_admitted_diff=wn)
    # winner-day rows are ~10% of rows: a guard (not clearly worse), not a target
    win_worse = bool(wn and wn["hi95"] < 0)
    if sh and pr and (sh["lo95"] > ADM_SHARE_TOL or sh["hi95"] < -ADM_SHARE_TOL or pr["hi95"] < -ADM_PREC_TOL
                      or win_worse):
        out["_verdict"] = "ROLLBACK_SUGGESTED"
    elif sh and pr and -ADM_SHARE_TOL <= sh["lo95"] and sh["hi95"] <= ADM_SHARE_TOL \
            and pr["lo95"] >= -ADM_PREC_TOL:
        out["_verdict"] = "KEEP"
    else:
        out["_verdict"] = "INCONCLUSIVE"
    return out


GS_NULL_MAX = 0.05          # share of shadow rows without a score
GS_ZERO_DAYS = 2            # comparable days with no shadow row at all
GS_MODEL_AGE_DAYS = 3.0     # nightly model older than this at the latest event
GS_GOAL_PP = 0.5
GS_NONINF_PP = -0.10


def goal_shadow_admit(rows: list) -> set:
    """Indices the shadow admits: per ISO week, the top round(s*n) by goal_score, where
    s is the share the live gate admitted that week; a null score is never admitted."""
    by = collections.defaultdict(list)
    for k, x in enumerate(rows):
        y, w, _ = x["dec"].isocalendar()
        by[(y, w)].append(k)
    out = set()
    for ks in by.values():
        n = int(round(sum(1 for k in ks if rows[k]["live_pass"]) / len(ks) * len(ks)))
        sc = sorted((k for k in ks if rows[k]["goal_score"] is not None), key=lambda k: -rows[k]["goal_score"])
        out.update(sc[:n])
    return out


def goal_shadow_days(W, entries, rows, shadow_pass, in_position, p):
    """Per winner-day contribution (gained - lost) and the count entered early now.
    entries: [(sym, dt, row index or None)]."""
    ent, by_sym = collections.defaultdict(list), collections.defaultdict(list)
    for e in entries:
        ent[e[0]].append(e)
    for j, x in enumerate(rows):
        by_sym[x["sym"]].append(j)
    d, now = [], 0
    for _day, sym, op, dd in W:
        early = [e for e in ent.get(sym, ()) if op <= e[1] < dd]
        if early:
            now += 1
            lost = all(e[2] is not None and rows[e[2]]["live_pass"] and e[2] not in shadow_pass for e in early)
            d.append(-1.0 if lost else 0.0)
            continue
        k = sum(1 for j in by_sym.get(sym, ()) if j in shadow_pass and not rows[j]["live_pass"]
                and op <= rows[j]["dec"] < dd and not in_position(sym, rows[j]["dec"]))
        d.append(1 - (1 - p) ** k if k else 0.0)
    return d, now


def goal_shadow_verdict(d_pp, d_lo, t_lo, t_hi) -> str:
    if d_pp >= GS_GOAL_PP and d_lo > 0 and t_lo >= GS_NONINF_PP:
        return "KEEP"
    if d_pp <= 0 or t_hi < GS_NONINF_PP:
        return "ROLLBACK_SUGGESTED"
    return "INCONCLUSIVE"


def goal_shadow_anomaly(rows, comparable, model_age_days) -> bool:
    if rows and sum(1 for x in rows if x["goal_score"] is None) / len(rows) > GS_NULL_MAX:
        return True
    have = {x["dec"].strftime("%Y-%m-%d") for x in rows}
    if sum(1 for d in comparable if d not in have) >= GS_ZERO_DAYS:
        return True
    return model_age_days is not None and model_age_days > GS_MODEL_AGE_DAYS


def readout_ml_goal_shadow(r: dict, ctx: dict) -> dict:
    """GOAL-label ML model in shadow vs the live gate (ml-goal-shadow-1012-spec.md):
    at the live gate's weekly admitted share, would it reach more winner-days before
    the crossing, with trades (as the bot takes them) no worse?"""
    import config as cfg
    import goal_validator as GV
    import indicators as I
    import _backtest_trend_start_detector as TD
    evs = scan_events(r["window_from"], {"ml_goal_shadow"})
    days = list(ctx["days"])
    seen, rows = set(), []
    for e in evs:
        tf = str(e.get("tf") or "")
        if tf not in GV.STEP or not e.get("sym") or not isinstance(e.get("bar_ts"), (int, float)):
            continue
        k = (e["sym"], tf, int(e["bar_ts"]))
        if k in seen:
            continue
        seen.add(k)
        bo = datetime.fromtimestamp(int(e["bar_ts"]) / 1000, timezone.utc)
        rows.append({"sym": e["sym"], "tf": tf, "bo": bo, "dec": bo + GV.STEP[tf], "mode": e.get("mode"),
                     "live_pass": bool(e.get("live_pass")),
                     "goal_score": float(e["goal_score"]) if isinstance(e.get("goal_score"), (int, float)) else None,
                     "trained_at": e.get("model_trained_at"), "ts": e["_dt"]})
    age = None
    if rows and rows[-1]["trained_at"]:
        try:
            age = (rows[-1]["ts"] - _dt(rows[-1]["trained_at"])).total_seconds() / 86400
        except Exception:
            age = None
    out = {"days": len(days), "rows": len(rows), "null_scores": sum(1 for x in rows if x["goal_score"] is None),
           "live_admitted_share": round(sum(x["live_pass"] for x in rows) / len(rows), 3) if rows else None,
           "model_age_days_at_last_row": round(age, 2) if age is not None else None}
    out["_anomaly"] = goal_shadow_anomaly(rows, [d for d in days if d > r["window_from"]], age)
    out["_n_trades"] = 0
    if not rows:
        out["note"] = "no shadow rows yet"
        return out
    shadow = goal_shadow_admit(rows)
    idx = {(x["sym"], x["tf"], x["bo"]): j for j, x in enumerate(rows)}
    entries, windows = GV.load_bot_log(r["window_from"])
    ent = []
    for sym, lst in entries.items():
        for d, tf, mode, k in lst:
            j = idx.get((sym, tf, GV.bar_open(d, tf) - GV.STEP[tf])) if tf in GV.STEP else None
            ent.append((sym, d, j, tf, mode, k))
    live_rows = [j for j, x in enumerate(rows) if x["live_pass"]]
    p = sum(1 for e in ent if e[2] is not None) / max(1, len(live_rows))

    def in_position(sym, d):
        return any(a <= d <= b for a, b in windows.get(sym, ()))
    dset = set(days)
    W = [w for w in GV.winner_days(r["window_from"], dset) if w[0] in dset]
    contrib, now = goal_shadow_days(W, [(e[0], e[1], e[2]) for e in ent], rows, shadow, in_position, p)
    rnd = random.Random(1012)
    n = len(W)
    if n:
        bs = sorted(100 * sum(rnd.choices(contrib, k=n)) / n for _ in range(BOOT))
        goal = {"winner_days": n, "entered_early_now": now, "delta_pp": round(100 * sum(contrib) / n, 2),
                "lo95": round(bs[int(0.025 * BOOT)], 2), "hi95": round(bs[int(0.975 * BOOT) - 1], 2),
                "days_lost": sum(1 for c in contrib if c < 0), "expected_gained": round(sum(c for c in contrib if c > 0), 2)}
    else:
        goal = {"winner_days": 0}
    out.update(goal=goal, shadow_admitted=len(shadow), downstream_p=round(p, 3))

    # ---- per trade, calibrated live trail
    cache = {}

    def series(sym, tf):
        if (sym, tf) not in cache:
            b = TD.load_bars(sym, tf) or []
            if b:
                import numpy as np
                h = np.array([x[2] for x in b], float)
                lo_ = np.array([x[3] for x in b], float)
                c = np.array([x[4] for x in b], float)
                t = [x[0] if x[0].tzinfo else x[0].replace(tzinfo=timezone.utc) for x in b]
                cache[(sym, tf)] = (t, c, I._atr(h, lo_, c, cfg.ATR_PERIOD), {x: i for i, x in enumerate(t)})
            else:
                cache[(sym, tf)] = None
        return cache[(sym, tf)]
    ks = sorted(float(e[5]) for e in ent if isinstance(e[5], (int, float)))
    K = ks[len(ks) // 2] if ks else 2.0
    cooldown = int(getattr(cfg, "COOLDOWN_BARS", 19))
    cur, rem, add = [], [], []
    for sym, d, j, tf, mode, k in ent:
        s = series(sym, tf) if tf in GV.STEP else None
        i = s[3].get(GV.bar_open(d, tf) - GV.STEP[tf]) if s else None
        res = GV.live_trail(s[1], s[2], i, float(k) if isinstance(k, (int, float)) else K,
                            GV.trail_floor(mode, cfg), GV.MAX_BARS[tf]) if i is not None else None
        if res:
            item = (d.strftime("%Y-%m-%d"), res[1])
            cur.append(item)
            if j is not None and rows[j]["live_pass"] and j not in shadow:
                rem.append(item)
    busy = {}
    for j in sorted((j for j in shadow if not rows[j]["live_pass"]), key=lambda j: rows[j]["dec"]):
        x = rows[j]
        if (busy.get(x["sym"]) and x["dec"] < busy[x["sym"]]) or in_position(x["sym"], x["dec"]):
            continue
        s = series(x["sym"], x["tf"])
        i = s[3].get(x["bo"]) if s else None
        res = GV.live_trail(s[1], s[2], i, K, GV.trail_floor(x["mode"], cfg), GV.MAX_BARS[x["tf"]]) \
            if i is not None else None
        if not res:
            continue
        add.append((x["dec"].strftime("%Y-%m-%d"), res[1]))
        busy[x["sym"]] = s[0][res[0]] + GV.STEP[x["tf"]] * (1 + cooldown)
    out["_n_trades"] = len(add)
    trade = {"current": len(cur), "removed": len(rem), "added": len(add), "trail_k_median": K}
    if cur:
        ud = sorted({t[0] for t in cur + add})

        def comb(pick):
            sc = sum(v * pick[dd] for dd, v in cur if dd in pick)
            nc = sum(pick[dd] for dd, _ in cur if dd in pick)
            sr = sum(v * pick[dd] for dd, v in rem if dd in pick)
            nr = sum(pick[dd] for dd, _ in rem if dd in pick)
            sa = sum(v * pick[dd] for dd, v in add if dd in pick)
            na = sum(pick[dd] for dd, _ in add if dd in pick)
            if nc == 0:
                return None
            den = nc - nr + p * na
            return ((sc - sr + p * sa) / den if den > 0 else 0.0) - sc / nc
        pt = comb({dd: 1 for dd in ud})
        bs = []
        for _ in range(BOOT):
            pick = collections.Counter(rnd.choices(ud, k=len(ud)))
            v = comb(pick)
            if v is not None:
                bs.append(v)
        bs.sort()
        trade.update(combined_minus_current_pp=round(pt, 3), lo95=round(bs[int(0.025 * len(bs))], 3),
                     hi95=round(bs[int(0.975 * len(bs)) - 1], 3),
                     current_mean=round(sum(v for _, v in cur) / len(cur), 3),
                     added_mean=round(sum(v for _, v in add) / len(add), 3) if add else None)
    out["per_trade"] = trade
    if n and "lo95" in trade:
        out["_verdict"] = goal_shadow_verdict(goal["delta_pp"], goal["lo95"], trade["lo95"], trade["hi95"])
    return out


CAP3_MATCH = timedelta(minutes=30)
CAP3_TAIL_TOL = 0.10
CAP3_NONINF_PP = -0.10


def cap3_relaxed_flags(trades, relaxed) -> list:
    """Per (entry, exit) pair: True if the same coin and timeframe logged corr_cap_relaxed
    within CAP3_MATCH before the entry (corr-cap3-live-1016-spec.md)."""
    by = collections.defaultdict(list)
    for e in relaxed:
        by[(e.get("sym"), e.get("tf"))].append(e["_dt"])
    out = []
    for en, _ex in trades:
        ts = by.get((en.get("sym"), en.get("tf")), ())
        out.append(any(en["_dt"] - CAP3_MATCH <= t <= en["_dt"] for t in ts))
    return out


def _maxdd(seq):
    peak = dd = acc = 0.0
    for v in seq:
        acc += v
        peak = max(peak, acc)
        dd = min(dd, acc - peak)
    return dd


def cap3_tail(daily: dict):
    """daily: {day: [(pnl, relaxed)]} -> (dd_with, dd_without, worst_with, worst_without)."""
    days = sorted(daily)
    w = [sum(v for v, _ in daily[d]) for d in days]
    wo = [sum(v for v, rl in daily[d] if not rl) for d in days]
    return float(_maxdd(w)), float(_maxdd(wo)), float(min(w) if w else 0.0), float(min(wo) if wo else 0.0)


def cap3_verdict(days_gained, t_lo, t_hi, *, dd_with, dd_without, worst_with, worst_without) -> str:
    tail_bad = dd_with < dd_without * (1 + CAP3_TAIL_TOL) or worst_with < worst_without * (1 + CAP3_TAIL_TOL)
    if t_hi < CAP3_NONINF_PP or tail_bad:
        return "ROLLBACK_SUGGESTED"
    if days_gained >= 1 and t_lo >= CAP3_NONINF_PP:
        return "KEEP"
    return "INCONCLUSIVE"


def readout_corr_cap3(r: dict, ctx: dict) -> dict:
    """correlation_guard cluster cap 2 -> 3 (corr-cap3-live-1016-spec.md)."""
    import goal_validator as GV
    evs = scan_events(r["window_from"], {"entry", "exit", "corr_cap_relaxed", "corr_prune"})
    days = set(ctx["days"])
    trades = [(en, ex) for en, ex in pair_trades([e for e in evs if e["event"] in ("entry", "exit")])
              if en["ts"][:10] in days and isinstance(ex.get("pnl_pct"), (int, float))]
    relaxed = [e for e in evs if e["event"] == "corr_cap_relaxed"]
    prunes = [e for e in evs if e["event"] == "corr_prune"]
    flags = cap3_relaxed_flags(trades, relaxed)
    pr_size = collections.Counter()
    for e in prunes:
        m = re.search(r"size (\d+)->", str(e.get("reason", "")))
        pr_size[int(m.group(1)) if m else -1] += 1
    out = {"days": len(days), "trades": len(trades), "relaxed_trades": sum(flags),
           "relaxed_events": len(relaxed), "prunes": len(prunes), "prunes_by_cluster_size": dict(pr_size)}
    out["_n_trades"] = sum(flags)
    if not trades:
        out["note"] = "no closed trades yet"
        return out
    pn = [float(ex["pnl_pct"]) for _en, ex in trades]
    rel = [v for v, f in zip(pn, flags) if f]
    keep = [v for v, f in zip(pn, flags) if not f]
    out.update(mean_all=round(sum(pn) / len(pn), 3), mean_relaxed=round(sum(rel) / len(rel), 3) if rel else None,
               mean_without=round(sum(keep) / len(keep), 3) if keep else None)
    # goal: winner-days whose early entries are all relaxed trades
    W = [w for w in GV.winner_days(min(days), days) if w[0] in days] if days else []
    gained = 0
    for _day, sym, op, dd in W:
        early = [f for (en, _ex), f in zip(trades, flags) if en["sym"] == sym and op <= en["_dt"] < dd]
        gained += bool(early) and all(early)
    out.update(winner_days=len(W), days_gained=gained)
    # per trade: all trades minus the same without the relaxed ones, trade bootstrap
    rnd = random.Random(1016)
    if rel and keep:
        pt = sum(pn) / len(pn) - sum(keep) / len(keep)
        bs = []
        for _ in range(BOOT):
            a, b = rnd.choices(keep, k=len(keep)), rnd.choices(rel, k=len(rel))
            bs.append((sum(a) + sum(b)) / (len(a) + len(b)) - sum(a) / len(a))
        bs.sort()
        t_lo, t_hi = bs[int(0.025 * BOOT)], bs[int(0.975 * BOOT) - 1]
        out["per_trade"] = {"combined_minus_without_pp": round(pt, 3), "lo95": round(t_lo, 3), "hi95": round(t_hi, 3)}
    else:
        t_lo = t_hi = 0.0
    daily = collections.defaultdict(list)
    for (en, _ex), v, f in zip(trades, pn, flags):
        daily[en["ts"][:10]].append((v, f))
    dd_w, dd_wo, wst_w, wst_wo = cap3_tail(daily)
    out["tail"] = {"max_drawdown_with": round(dd_w, 1), "max_drawdown_without": round(dd_wo, 1),
                   "worst_day_with": round(wst_w, 1), "worst_day_without": round(wst_wo, 1)}
    if rel and keep:
        out["_verdict"] = cap3_verdict(gained, t_lo, t_hi, dd_with=dd_w, dd_without=dd_wo,
                                       worst_with=wst_w, worst_without=wst_wo)
    return out


FNS = {"readout_corr_cap3": readout_corr_cap3,
       "readout_ml_goal_shadow": readout_ml_goal_shadow,
       "readout_e4": readout_e4, "readout_mrq_off": readout_mrq_off, "readout_leader_exit": readout_leader_exit,
       "readout_leader_alert": readout_leader_alert, "readout_ml_floor_shadow": readout_ml_floor_shadow,
       "readout_ml_reload": readout_ml_reload, "readout_ml_admission_match": readout_ml_admission_match}


# ---------------------------------------------------------------- runner

def status_of(r: dict, res: dict, today: str) -> str:
    """COLLECTING before due; TOO_EARLY while data is short; the verdict otherwise.
    An anomaly is reported at once, due or not."""
    if res.get("_anomaly"):
        return "ANOMALY"
    if r.get("kind") == "delegated":
        return "DELEGATED" if today >= r["due"] else "COLLECTING"
    enough = res.get("days", 0) >= r["min_days"] and res.get("_n_trades", 0) >= r.get("min_trades", 0)
    if today < r["due"]:
        return "COLLECTING"
    if not enough:
        return "TOO_EARLY"
    return res.get("_verdict", "INCONCLUSIVE")


def llm_summary(r: dict, res: dict, status: str) -> str | None:
    try:
        import config as cfg
        if not getattr(cfg, "READOUTS_LLM_SUMMARY_ENABLED", False):
            return None
        import pipeline_claude_client as CC
        if not CC.is_enabled():
            return None
        system = ("You write a 3-5 sentence Russian summary of a pre-registered readout of a live change "
                  "in a crypto-signal bot. The verdict is FIXED -- restate it, never change or soften it. "
                  "Quote only numbers present in the input. Say what the operator may do (the rollback) "
                  "only if the verdict is ROLLBACK_SUGGESTED or ANOMALY.")
        payload = {"readout": {k: r.get(k) for k in ("id", "title", "expected", "criteria", "rollback")},
                   "verdict": status, "numbers": {k: v for k, v in res.items() if not k.startswith("_")}}
        out = CC.call_claude_json(system, json.dumps(payload, ensure_ascii=False, default=str),
                                  schema_hint='{"summary_ru": str}', max_tokens=600,
                                  layer="readouts", purpose=f"readout_{r['id']}")
        s = (out or {}).get("summary_ru")
        if s and status not in s:
            s = f"[{status}] " + s          # the verdict must be visible even if the model omitted it
        return s
    except Exception as e:
        print(f"[readouts] summary skipped for {r.get('id')}: {e}")
        return None


def run(today: str | None = None, registry: Path = REGISTRY, out_dir: Path = OUT_DIR,
        use_llm: bool = True) -> list:
    import _compute_early_capture as E
    now = datetime.now(timezone.utc)
    today = today or now.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    first = min(r["window_from"] for r in load_registry(registry))
    full, _, _ = E.load_uptime(_dt(first + "T00:00:00Z"))
    results = []
    for r in load_registry(registry):
        p = out_dir / f"{r['id']}.json"
        prev = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        if prev.get("final"):
            results.append(prev)
            continue
        try:
            if r.get("kind") == "delegated":
                days = [d for d in comparable_days(r["window_from"], today, full)] if r["id"] != "LEADER-POSITIONING" else []
                res = readout_delegated(r, {"days": days, "now": now})
            else:
                days = comparable_days(r["window_from"], today, full)
                res = FNS[r["fn"]](r, {"days": days, "now": now})
            status = status_of(r, res, today)
        except Exception as e:
            res, status = {"error": repr(e)[:300]}, "ERROR"
        rec = {"id": r["id"], "title": r["title"], "date": today, "due": r["due"], "status": status,
               "min_days": r.get("min_days"), "result": {k: v for k, v in res.items() if not k.startswith("_")},
               "final": status in ("KEEP", "ROLLBACK_SUGGESTED", "INCONCLUSIVE", "DELEGATED")}
        if rec["final"] and use_llm and r.get("kind") != "delegated":
            rec["summary_ru"] = llm_summary(r, res, status)
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        with io.open(out_dir / "history.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({k: rec[k] for k in ("id", "date", "status")} | {"days": res.get("days")},
                                ensure_ascii=False) + "\n")
        results.append(rec)
    return results


STATUS_RU = {"COLLECTING": "идёт сбор", "TOO_EARLY": "рано судить", "KEEP": "оставить",
             "ROLLBACK_SUGGESTED": "предлагается откат", "INCONCLUSIVE": "не доказано",
             "ANOMALY": "АНОМАЛИЯ", "DELEGATED": "передано задаче Claude", "ERROR": "ошибка расчёта"}


def render_block(results: list, today: str | None = None) -> str | None:
    """One line per open readout; final verdicts appear on the day they are reached."""
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = []
    for x in results:
        if x.get("final") and x.get("date") != today:
            continue
        res = x.get("result") or {}
        days = res.get("days")
        tail = f"{days}/{x.get('min_days')} дн., срок {x['due']}" if days is not None else f"срок {x['due']}"
        line = f"• {x['id']}: {STATUS_RU.get(x['status'], x['status'])} ({tail})"
        if x["status"] == "ANOMALY":
            line += ": " + "; ".join(res.get("anomalies", [])[:3])
        lines.append(line)
        if x.get("summary_ru"):
            lines.append("  " + x["summary_ru"])
    return ("<b>Проверки изменений</b>\n" + "\n".join(lines)) if lines else None


def load_latest(out_dir: Path | None = None) -> list:
    out_dir = out_dir or OUT_DIR
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("*.json"))] if out_dir.exists() else []


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", help="YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args(argv)
    import config as cfg
    if not getattr(cfg, "READOUTS_ENABLED", False):
        print("[readouts] READOUTS_ENABLED is False -- skipped")
        return 0
    res = run(a.today, use_llm=not a.no_llm)
    for x in res:
        r = {k: v for k, v in (x.get("result") or {}).items() if k not in ("note",)}
        print(f"[readouts] {x['id']}: {x['status']} (due {x['due']}) {json.dumps(r, ensure_ascii=False, default=str)[:600]}")
    print(render_block(res, a.today) or "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
