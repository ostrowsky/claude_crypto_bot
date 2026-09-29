"""Daily incident analyst: where did the bot lose each of yesterday's winners?

WHY (2026-09-29, agent-tasks-0929-spec.md)

Every behaviour change since 2026-09-25 started from the operator asking about
ONE coin -- "where is the FIL signal", "why did we exit QNT", "why is HBAR not
in the list" -- and a manual trace through bot_events.jsonl and the poll
heartbeat. The L2 agent saw none of those cases: it received aggregate red
flags only, so all it could propose was "move a threshold", and the three
proposals of 2026-09-29 were all rejected. This module makes the trace
automatic and hands the agent concrete cases with ids.

WHAT A CASE IS

  day, sym   one of the day's immutable later-EOD top-20 (global rank, then the
             watchlist -- the North Star denominator), from the label store
  profile    from Binance 15m klines of that UTC day: open, first 15m close at
             >= +2.5% (the "crossing" -- the goal's deadline) and >= +10%, day
             high, rocket flag (high >= +10% and close keeps >= 60% of the rise)
  stage      the FIRST place the bot lost it, deterministic, in this order:
               bot_down              bot active < 18 h that UTC day (TH-05: no data, not a miss)
               held_from_before      a position opened before the day was still open
               caught_early          first entry before the crossing  (the goal)
               caught_late           first entry after the crossing
               blocked               candidates existed, every one blocked (first gate)
               cooldown / not_polled / data_gap / no_rule_fired /
               rule_fired_no_candidate   from the poll heartbeat inside the birth window
               no_heartbeat          the heartbeat file for the day is missing
  exit       for entered cases: pnl, exit class, capture = pnl / day move,
             rise left after the exit (same UTC day), leader-mode switch
  missed_pct the opportunity cost used to rank cases: day move minus the
             realised pnl (the whole move when not entered)

The LLM (Analyst role, continuous-improvement-agent-spec.md §4.1) only GROUPS
cases by mechanism and must cite case ids; it may not propose remedies, and a
group citing an id that does not exist is dropped. Its text never changes a
stage -- the classification above is the record.

Output: .runtime/pipeline/incidents/<day>.json; `rollup()` feeds L2.
"""
from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

EVENTS = HERE / "bot_events.jsonl"
HEARTBEAT = ROOT / ".runtime" / "poll_heartbeat"
OUT_DIR = ROOT / ".runtime" / "pipeline" / "incidents"
MIN_ACTIVE_HOURS = 18
LOOKBACK_DAYS = 8          # to see a position opened before the day
BAR = timedelta(minutes=15)
TS_RE = re.compile(rb'"ts":\s*"([^"]+)"')

STAGE_ORDER = ["caught_early", "caught_late", "held_from_before", "blocked", "rule_fired_no_candidate",
               "no_rule_fired", "cooldown", "not_polled", "data_gap", "no_heartbeat", "bot_down"]
STAGE_RU = {
    "caught_early": "взяли до +2.5%", "caught_late": "взяли после +2.5%",
    "held_from_before": "держали с прошлых дней", "blocked": "заблокировал фильтр",
    "rule_fired_no_candidate": "правило сработало, кандидата нет", "no_rule_fired": "ни одно правило не сработало",
    "cooldown": "cooldown после выхода", "not_polled": "монету не опрашивали", "data_gap": "нет данных/короткая история",
    "no_heartbeat": "нет журнала опроса", "bot_down": "бот не работал",
}


# ---------------------------------------------------------------- inputs

def fetch_15m(sym: str, day: str, base: str = "https://api.binance.com") -> list:
    """[(open_dt, o, h, l, c)] -- the closed 15m bars of one UTC day."""
    d0 = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    q = urllib.parse.urlencode({"symbol": sym, "interval": "15m", "limit": 96,
                                "startTime": int(d0.timestamp() * 1000),
                                "endTime": int((d0 + timedelta(days=1)).timestamp() * 1000) - 1})
    with urllib.request.urlopen(f"{base}/api/v3/klines?{q}", timeout=20) as r:
        js = json.loads(r.read().decode("utf-8"))
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return [(datetime.fromtimestamp(int(b[0]) / 1000, timezone.utc), float(b[1]), float(b[2]),
             float(b[3]), float(b[4])) for b in js if int(b[6]) < now_ms]


def day_profile(bars: list, day: str) -> dict | None:
    """Open, crossings, high, rocket flag from the day's 15m bars."""
    d0 = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    rows = [b for b in bars if d0 <= b[0] < d0 + timedelta(days=1)]
    if not rows or rows[0][0] != d0:
        return None
    op = rows[0][1]
    hi = max(b[2] for b in rows)
    cl = rows[-1][4]

    def first_close(level):
        return next((b[0] + BAR for b in rows if b[4] >= op * level), None)

    t25, t10 = first_close(1.025), first_close(1.10)
    move = (hi / op - 1) * 100
    return {"open": op, "high": hi, "close": cl, "move_pct": round(move, 2),
            "close_pct": round((cl / op - 1) * 100, 2),
            "t25": t25.isoformat() if t25 else None, "t10": t10.isoformat() if t10 else None,
            "rocket": bool(hi >= op * 1.10 and (cl - op) >= 0.6 * (hi - op)),
            "bars": [(b[0].isoformat(), b[2]) for b in rows]}


def _ts_at(fh, pos: int):
    fh.seek(pos)
    if pos:
        fh.readline()
    for _ in range(50):
        line = fh.readline()
        if not line:
            return None
        m = TS_RE.search(line)
        if m:
            return m.group(1).decode("ascii", "replace")
    return None


def seek_offset(fh, since: str) -> int:
    """Byte offset of the first line at or after `since` (the log is append-ordered;
    a small backoff absorbs lines written slightly out of order)."""
    fh.seek(0, 2)
    lo, hi = 0, fh.tell()
    while hi - lo > 1 << 16:
        mid = (lo + hi) // 2
        ts = _ts_at(fh, mid)
        if ts is None or ts >= since:
            hi = mid
        else:
            lo = mid
    return max(0, lo - (1 << 20))


def load_events(since: str, until: str, syms: set, path: Path = EVENTS):
    """(events of `syms` in [since, until), active UTC hours per day)."""
    evs, hours = collections.defaultdict(list), collections.defaultdict(set)
    with io.open(path, "rb") as fh:
        fh.seek(seek_offset(fh, since))
        for raw in fh:
            m = TS_RE.search(raw)
            if not m:
                continue
            ts = m.group(1).decode("ascii", "replace")
            if ts < since:
                continue
            if ts >= until:
                break
            hours[ts[:10]].add(ts[11:13])
            if not any(s.encode() in raw for s in syms):
                continue
            try:
                e = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue
            if e.get("sym") not in syms:
                continue
            d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            e["_dt"] = d if d.tzinfo else d.replace(tzinfo=timezone.utc)
            evs[e["sym"]].append(e)
    return evs, {k: len(v) for k, v in hours.items()}


def load_heartbeat(day: str, syms: set, hb_dir: Path = HEARTBEAT):
    """(rows of `syms` whose bar is on `day`, whether any heartbeat file covers the day)."""
    d1 = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    files = [hb_dir / f"{x}.jsonl" for x in (day, d1)]
    have = any(f.exists() for f in files[:1])
    out = collections.defaultdict(list)
    for f in files:
        if not f.exists():
            continue
        for line in io.open(f, encoding="utf-8", errors="replace"):
            try:
                x = json.loads(line)
            except Exception:
                continue
            if x.get("sym") in syms and str(x.get("bar_utc", "")).startswith(day):
                out[x["sym"]].append(x)
    return out, have


# ---------------------------------------------------------------- classification

def gate_name(e: dict) -> str:
    st = str(e.get("signal_type") or "")
    r = str(e.get("reason") or "")
    if st == "trend_quality" or e.get("reason_code") == "trend_quality":
        sub = ("price_edge" if "price edge" in r else "daily_range" if "daily_range" in r else "RSI" if "RSI" in r
               else "forecast0" if "forecast 0.000" in r else "forecast/alt")
        return "trend_quality: " + sub
    if "портфель полон" in r:
        return "portfolio full"
    if st in ("", "buy", "None"):
        return str(e.get("reason_code") or re.sub(r"[\d.]+", "#", r)[:40])
    return st


def reason_class(t: str) -> str:
    t = re.sub(r"-?[\d.]+", "#", str(t))
    return ("daily_range late" if "daily_range" in t or "от дна дня" in t else "RSI out of zone" if "RSI" in t
            else "EMA structure" if "структура" in t or ("EMA" in t and "выше" in t) else "volume" if "vol" in t or "объём" in t
            else "slope/ADX" if "slope" in t or "наклон" in t or "ADX" in t else "MACD" if "MACD" in t
            else "momentum r1/r3" if "r1" in t or "r3" in t else "regime" if "режим" in t else t[:30])


def exit_class(r: str) -> str:
    r = str(r or "")
    if r.startswith("режим лидера"):
        return "leader mode"
    if "WEAK" in r:
        return "WEAK: " + ("RSI divergence" if "RSI" in r else "volume exhaustion" if "объём" in r
                           else "EMA fan" if "EMA-веер" in r else "other")
    return ("ATR trail" if "ATR" in r else "time max hold" if "время" in r else "EMA20 exit" if "EMA20" in r
            else "portfolio rotation" if "rotation" in r else "RSI overbought" if "перекуплен" in r
            else re.sub(r"[\d.]+", "#", r)[:30])


def classify_case(day: str, sym: str, prof: dict, evs: list, hb_rows: list, hb_have: bool,
                  active_hours: int) -> dict:
    d0 = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    d1 = d0 + timedelta(days=1)
    t25 = datetime.fromisoformat(prof["t25"]) if prof.get("t25") else None
    birth_end = t25 or d1
    case = {"case_id": f"{day}:{sym}", "day": day, "sym": sym, "move_pct": prof["move_pct"],
            "close_pct": prof["close_pct"], "rocket": prof["rocket"], "t25": prof.get("t25"), "t10": prof.get("t10")}
    evs = sorted(evs, key=lambda e: e["_dt"])
    day_ent = [e for e in evs if e.get("event") == "entry" and d0 <= e["_dt"] < d1]
    prior = [e for e in evs if e["_dt"] < d0 and e.get("event") in ("entry", "exit")]
    holding = bool(prior) and prior[-1].get("event") == "entry"
    blocks = [e for e in evs if e.get("event") == "blocked" and d0 <= e["_dt"] < d1]
    case["gates_birth"] = sorted({gate_name(e) for e in blocks if e["_dt"] < birth_end})
    case["gates_day"] = sorted({gate_name(e) for e in blocks})
    case["leader_switch"] = any(e.get("event") == "leader_exit_switch" and d0 <= e["_dt"] < d1 for e in evs)

    first = None
    if active_hours < MIN_ACTIVE_HOURS:
        stage = "bot_down"
    elif holding:
        stage = "held_from_before"
        first = prior[-1]
    elif day_ent:
        first = day_ent[0]
        stage = "caught_early" if t25 is None or first["_dt"] < t25 else "caught_late"
        if t25 is not None:
            case["lead_hours"] = round((t25 - first["_dt"]).total_seconds() / 3600, 2)
        px = first.get("price")
        if isinstance(px, (int, float)) and prof["open"]:
            case["entry_vs_open_pct"] = round((px / prof["open"] - 1) * 100, 2)
        case["entry_mode"] = f"{first.get('mode') or first.get('signal_mode')}/{first.get('tf')}"
    elif blocks:
        stage = "blocked"
        case["first_gate"] = gate_name(blocks[0])
    elif not hb_have:
        stage = "no_heartbeat"
    else:
        rows = [x for x in hb_rows if x.get("bar_utc", "") < birth_end.strftime("%Y-%m-%dT%H:%M")] or hb_rows
        st = collections.Counter(x.get("stage") for x in rows)
        case["heartbeat_stages"] = dict(st)
        case["heartbeat_tf"] = sorted({str(x.get("tf")) for x in rows})
        ev = [x for x in rows if x.get("stage") == "evaluated"]
        if not rows:
            stage = "not_polled"
        elif ev and any(x.get("fired") for x in ev):
            stage = "rule_fired_no_candidate"
            case["rules_fired"] = sorted({r for x in ev for r in (x.get("fired") or [])})
        elif ev:
            stage = "no_rule_fired"
            why = collections.Counter()
            for x in ev:
                for rule, r in (x.get("reasons") or {}).items():
                    if rule in ("breakout", "retest", "ema_cross"):
                        continue
                    why[f"{rule}: {reason_class(r)}"] += 1
            case["rule_reasons"] = [k for k, _ in why.most_common(4)]
        elif st.get("cooldown"):
            stage = "cooldown"
        elif st.get("position_open"):
            stage = "held_from_before"
        else:
            stage = "data_gap"
    case["stage"] = stage

    pnl = None
    if first is not None:
        ex = next((e for e in evs if e.get("event") == "exit" and e["_dt"] > first["_dt"]
                   and e.get("tf") == first.get("tf")), None)
        if ex is not None and isinstance(ex.get("pnl_pct"), (int, float)):
            pnl = float(ex["pnl_pct"])
            case["exit"] = {"pnl_pct": round(pnl, 2), "class": exit_class(ex.get("reason")),
                            "at": ex["_dt"].isoformat(timespec="minutes")}
            xp = ex.get("exit_price")
            after = [h for t, h in prof.get("bars", []) if datetime.fromisoformat(t) >= ex["_dt"]]
            if after and isinstance(xp, (int, float)) and xp > 0:
                case["exit"]["left_after_exit_pct"] = round((max(after) / xp - 1) * 100, 2)
            if prof["move_pct"] > 0:
                case["exit"]["capture"] = round(pnl / prof["move_pct"], 3)
        else:
            case["exit"] = {"open": True}
    case["missed_pct"] = round(prof["move_pct"] - max(pnl or 0.0, 0.0), 2) if stage != "bot_down" else 0.0
    return case


def summarise(cases: list) -> dict:
    live = [c for c in cases if c["stage"] != "bot_down"]
    st = collections.Counter(c["stage"] for c in cases)
    gates = collections.Counter(c.get("first_gate") for c in cases if c.get("first_gate"))
    why = collections.Counter(r for c in cases for r in c.get("rule_reasons", []))
    exits = collections.Counter(c["exit"]["class"] for c in cases if c.get("exit", {}).get("class"))
    early = sum(1 for c in live if c["stage"] == "caught_early")
    rockets = [c for c in live if c["rocket"]]
    return {
        "cases": len(cases), "cases_bot_up": len(live),
        "stages": {k: st[k] for k in STAGE_ORDER if st.get(k)},
        "goal_entered_before_crossing": {"n": early, "of": len(live),
                                         "share": round(early / len(live), 3) if live else None},
        "rockets": {"n": len(rockets), "caught_early": sum(1 for c in rockets if c["stage"] == "caught_early"),
                    "entered_any": sum(1 for c in rockets if c["stage"] in ("caught_early", "caught_late", "held_from_before"))},
        "first_gate": dict(gates.most_common()),
        "no_rule_reasons": dict(why.most_common(8)),
        "exit_classes": dict(exits.most_common()),
        "exited_left_5pp": sum(1 for c in cases if c.get("exit", {}).get("left_after_exit_pct", 0) >= 5),
    }


# ---------------------------------------------------------------- LLM grouping

ANALYST_SYSTEM = """You are the Analyst of a crypto-signal bot's improvement loop.
You receive CASES: coins that were among the day's top-20 gainers, each with the
deterministic stage where the bot lost it (or caught it), gates, rule reasons and
exit data. Your ONLY job: group the cases by the mechanism that lost them, so a
later step can write hypotheses. Rules:
 1. Cite case_ids for every group. Never invent a case_id.
 2. Do NOT propose remedies, parameter changes or fixes. Describe mechanisms only.
 3. Do NOT state numbers that are not in the input.
 4. At most 5 groups. Cases that fit no pattern may be left out.
 5. The stage field is authoritative; do not reclassify it.
"""
ANALYST_SCHEMA = '{"groups": [{"title": str, "mechanism": str, "case_ids": [str]}], "note": str}'
_REMEDY = re.compile(r"\b(should|recommend|increase|decrease|lower|raise|relax|tighten|change|set)\b", re.I)


def sanitize_groups(res, case_ids: set) -> list:
    """Keep only groups citing real case ids; drop remedy-like wording (Analyst
    may not propose fixes -- continuous-improvement-agent-spec.md §4.1)."""
    out = []
    for g in (res or {}).get("groups") or []:
        if not isinstance(g, dict):
            continue
        ids = [i for i in (g.get("case_ids") or []) if i in case_ids]
        if not ids:
            continue
        mech = str(g.get("mechanism") or "")[:400]
        out.append({"title": str(g.get("title") or "")[:80], "mechanism": mech, "case_ids": ids,
                    "remedy_wording": bool(_REMEDY.search(mech))})
    return out[:5]


def llm_groups(cases: list) -> list:
    try:
        import config as cfg
        if not getattr(cfg, "INCIDENT_ANALYST_LLM_ENABLED", False):
            return []
        import pipeline_claude_client as CC
        if not CC.is_enabled():
            return []
        slim = [{k: c.get(k) for k in ("case_id", "stage", "move_pct", "rocket", "first_gate", "gates_birth",
                                          "rule_reasons", "rules_fired", "heartbeat_stages", "entry_mode",
                                          "lead_hours", "entry_vs_open_pct", "exit", "missed_pct") if c.get(k) is not None}
                for c in cases]
        res = CC.call_claude_json(ANALYST_SYSTEM, json.dumps({"cases": slim}, ensure_ascii=False),
                                  schema_hint=ANALYST_SCHEMA, max_tokens=1500, layer="L1b",
                                  purpose="incident_grouping")
        return sanitize_groups(res, {c["case_id"] for c in cases})
    except Exception as e:
        print(f"[incidents] LLM grouping skipped: {e}")
        return []


# ---------------------------------------------------------------- run

def winners_of(day: str) -> list:
    """[(sym, global_rank, eod_return)] of the day's immutable top-20 on the watchlist."""
    import _compute_early_capture as E
    import immutable_labels as IL
    wl = E.load_watchlist()
    win, eod = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
    ranked = sorted(((s, r) for (d, s), r in eod.items() if d == day), key=lambda x: -float(x[1]))
    rank = {s: i + 1 for i, (s, _) in enumerate(ranked)}
    return sorted(((s, rank.get(s), eod[(d, s)]) for d, s in win if d == day), key=lambda x: x[1] or 99)


def analyse_day(day: str, *, use_llm: bool = True, fetch=fetch_15m) -> dict:
    w = winners_of(day)
    report = {"day": day, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "definition": "immutable later-EOD top-20 (global rank) on the watchlist"}
    if not w:
        return {**report, "status": "labels_missing", "cases": [], "summary": {}}
    syms = {s for s, _, _ in w}
    d0 = datetime.strptime(day, "%Y-%m-%d")
    evs, hours = load_events((d0 - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d"),
                             (d0 + timedelta(days=1)).strftime("%Y-%m-%d"), syms)
    hb, hb_have = load_heartbeat(day, syms)
    cases = []
    for sym, grank, eod in w:
        try:
            prof = day_profile(fetch(sym, day), day)
        except Exception as e:
            prof = None
            print(f"[incidents] {sym}: klines failed: {e}")
        if prof is None:
            cases.append({"case_id": f"{day}:{sym}", "day": day, "sym": sym, "stage": "data_gap",
                          "note": "no 15m klines for the day", "global_rank": grank, "eod_return_pct": eod,
                          "move_pct": 0.0, "rocket": False, "missed_pct": 0.0})
            continue
        c = classify_case(day, sym, prof, evs.get(sym, []), hb.get(sym, []), hb_have, hours.get(day, 0))
        c.update(global_rank=grank, eod_return_pct=round(float(eod), 2))
        cases.append(c)
    cases.sort(key=lambda c: -c.get("missed_pct", 0))
    report.update(status="ok", bot_active_hours=hours.get(day, 0), heartbeat_available=hb_have,
                  cases=cases, summary=summarise(cases))
    report["llm_groups"] = llm_groups(cases) if use_llm and cases else []
    return report


def write_report(rep: dict, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{rep['day']}.json"
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def load_reports(days: int, until: str | None = None, out_dir: Path = OUT_DIR) -> list:
    end = datetime.strptime(until, "%Y-%m-%d") if until else datetime.now(timezone.utc).replace(tzinfo=None)
    out = []
    for i in range(1, days + 1):
        p = out_dir / f"{(end - timedelta(days=i)).strftime('%Y-%m-%d')}.json"
        if p.exists():
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
    return out


def rollup(days: int = 14, until: str | None = None, out_dir: Path = OUT_DIR, top: int = 12) -> dict:
    """The L2 input: stage counts over the last `days` reports + the costliest cases."""
    reps = [r for r in load_reports(days, until, out_dir) if r.get("status") == "ok"]
    cases = [c for r in reps for c in r.get("cases", [])]
    if not cases:
        return {"days": 0, "cases": 0}
    s = summarise(cases)
    worst = sorted((c for c in cases if c["stage"] != "bot_down"), key=lambda c: -c.get("missed_pct", 0))[:top]
    keep = ("case_id", "stage", "move_pct", "rocket", "first_gate", "rule_reasons", "rules_fired", "entry_mode",
            "lead_hours", "exit", "missed_pct")
    return {"days": len(reps), "first_day": min(r["day"] for r in reps), "last_day": max(r["day"] for r in reps),
            **s, "costliest_cases": [{k: c[k] for k in keep if k in c} for c in worst],
            "llm_groups": [g for r in reps for g in r.get("llm_groups", [])][-8:]}


def render_block(rep: dict) -> str | None:
    """Short Telegram-HTML block for the morning report."""
    if not rep or rep.get("status") != "ok" or not rep.get("cases"):
        return None
    s = rep["summary"]
    g = s["goal_entered_before_crossing"]
    lines = [f"<b>Лидеры {rep['day']}</b> (top-20 биржи в watchlist: {s['cases']})"]
    if g["of"]:
        lines.append(f"Взяли до +2.5%: {g['n']} из {g['of']}; ракет {s['rockets']['n']}, "
                     f"из них взяли рано {s['rockets']['caught_early']}")
    else:
        lines.append("Бот не работал весь день -- данных нет, это не промахи")
    st = ", ".join(f"{STAGE_RU.get(k, k)} {v}" for k, v in s["stages"].items() if k != "caught_early")
    if st:
        lines.append("Потери: " + st)
    for c in rep["cases"][:3]:
        if c["stage"] in ("bot_down", "caught_early") and not c.get("exit"):
            continue
        extra = c.get("first_gate") or (", ".join(c.get("rule_reasons", [])[:1])) or \
            (c.get("exit", {}).get("class") if c.get("exit") else "") or ""
        lines.append(f"• {c['sym']} +{c['move_pct']:.1f}%: {STAGE_RU.get(c['stage'], c['stage'])}"
                     + (f" ({extra})" if extra else ""))
    return "\n".join(lines)


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", help="UTC day, default yesterday")
    ap.add_argument("--backfill", type=int, default=0, help="also analyse the N days before --day")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--print", dest="do_print", action="store_true")
    a = ap.parse_args(argv)
    import config as cfg
    if not getattr(cfg, "INCIDENT_ANALYST_ENABLED", False):
        print("[incidents] INCIDENT_ANALYST_ENABLED is False -- skipped")
        return 0
    day = a.day or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    days = [(datetime.strptime(day, "%Y-%m-%d") - timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(a.backfill, -1, -1)]
    for d in days:
        rep = analyse_day(d, use_llm=not a.no_llm and d == day)
        p = write_report(rep)
        s = rep.get("summary") or {}
        print(f"[incidents] {d}: {rep['status']}, cases {s.get('cases', 0)}, stages {s.get('stages')}, "
              f"goal {s.get('goal_entered_before_crossing')} -> {p}")
        if a.do_print:
            print(render_block(rep) or "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
