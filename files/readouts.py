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


FNS = {"readout_e4": readout_e4, "readout_leader_exit": readout_leader_exit,
       "readout_leader_alert": readout_leader_alert}


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
