"""Which limits suppressed the bot on rocket days? Every rocket since the bot's log starts.

Rocket = watchlist coin >= +10% from the UTC open that holds it (close >= 60% of
the rise) -- rocket-segment-spec.md. Window of interest: from the UTC open to
the first 15m close at >= +10% (the part of the move a signal must precede).
Days: bot up all day (_compute_early_capture.load_uptime), 2026-03-01 .. yesterday.

For each rocket-day, from bot_events.jsonl (every timeframe) and the 15m rule
replay (.runtime/backtests/lateness_rows.jsonl; ema_cross dropped -- it is off
live since 2026-04-19):
  stage      entered before +2.5% / entered between +2.5% and +10% / already
             holding / candidates all blocked / a live rule fired on the stored
             bars but no candidate / only with daily_range caps lifted / no rule
  gates      every gate that blocked the coin inside the window (a day counts
             once per gate), and the gate of the first block
For the days the poll heartbeat covers (2026-09-26 on) also the reasons each
entry rule gave for not firing on the rocket's own bars.
"""
import collections
import glob
import io
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import _backtest_trend_start_detector as TD  # noqa: E402
import _compute_early_capture as E  # noqa: E402

START = "2026-03-01"
R = [json.loads(l) for l in io.open(FILES.parent / ".runtime/backtests/rocket_events.jsonl", encoding="utf-8")]
full, _, _ = E.load_uptime(datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=timezone.utc))
rockets = {(r["day"], r["sym"]): r for r in R if r["trig"] == 0.025 and r["rocket"] and r["day"] >= START and r["day"] in full}
print("rocket-days with the bot up all day since %s: %d" % (START, len(rockets)))

# time of the first 15m close >= +10%
win = {}
for (day, sym) in rockets:
    b = TD.bars_15m(sym)
    d0 = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    rows = [x for x in b if d0 <= x[0] < d0 + timedelta(days=1)]
    if not rows or rows[0][0] != d0:
        continue
    op = rows[0][1]
    t10 = next((x[0] + timedelta(minutes=15) for x in rows if x[4] >= op * 1.10), None)
    t25 = datetime.fromisoformat(rockets[(day, sym)]["t"] + "+00:00")
    if t10:
        win[(day, sym)] = (d0, t25, t10)
print("with a +10%% close: %d; median hours open->+2.5%%: %.1f, +2.5%%->+10%%: %.1f" % (
    len(win), sorted((v[1] - v[0]).total_seconds() / 3600 for v in win.values())[len(win) // 2],
    sorted((v[2] - v[1]).total_seconds() / 3600 for v in win.values())[len(win) // 2]))

fires = collections.defaultdict(list)
for l in io.open(FILES.parent / ".runtime/backtests/lateness_rows.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if (r["day"], r["sym"]) in win and r["rule"] != "ema_cross":
        fires[(r["day"], r["sym"])].append((datetime.fromisoformat(r["ts"]) + timedelta(minutes=15), r["band"], r["rule"]))

want = {s for _, s in win}
evs = collections.defaultdict(list)
with io.open(FILES / "bot_events.jsonl", "rb") as fh:
    for raw in fh:
        if not any(x in raw for x in (b'"entry"', b'"exit"', b'"blocked"', b'"cooldown_start"')):
            continue
        try:
            e = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            continue
        if e.get("sym") not in want or str(e.get("ts", "")) < "2026-02-27":
            continue
        d = datetime.fromisoformat(str(e["ts"]).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        e["_dt"] = d
        evs[e["sym"]].append(e)


def gate(e):
    st = str(e.get("signal_type") or "")
    r = str(e.get("reason") or "")
    if st == "trend_quality" or e.get("reason_code") == "trend_quality":
        sub = ("price_edge" if "price edge" in r else "daily_range" if "daily_range" in r else "RSI" if "RSI" in r
               else "forecast0" if "forecast 0.000" in r else "forecast/alt")
        return "trend_quality: " + sub
    if "портфель полон" in r:
        return "portfolio full"
    if st in ("", "buy", "None"):
        return (e.get("reason_code") or re.sub(r"[\d.]+", "#", r)[:40])
    return st


stage = collections.Counter()
gate_any = collections.Counter()
gate_first = collections.Counter()
gate_early = collections.Counter()
by_stage_days = collections.defaultdict(list)
for key, (d0, t25, t10) in win.items():
    day, sym = key
    ev = [e for e in evs.get(sym, ()) if d0 <= e["_dt"] < t10]
    ent = [e for e in ev if e["event"] == "entry"]
    blk = [e for e in ev if e["event"] == "blocked"]
    prior = [e for e in evs.get(sym, ()) if e["_dt"] < d0 and e["event"] in ("entry", "exit")]
    holding = bool(prior) and prior[-1]["event"] == "entry"
    for g in {gate(e) for e in blk}:
        gate_any[g] += 1
    for g in {gate(e) for e in blk if e["_dt"] < t25}:
        gate_early[g] += 1
    if ent and min(e["_dt"] for e in ent) < t25:
        s = "1 entered before +2.5%"
    elif ent:
        s = "2 entered between +2.5% and +10%"
    elif holding:
        s = "3 already holding from before the day"
    elif blk:
        s = "4 candidates, all blocked"
        gate_first[gate(min(blk, key=lambda e: e["_dt"]))] += 1
    else:
        f = [x for x in fires.get(key, ()) if d0 <= x[0] < t10]
        cd = [e for e in evs.get(sym, ()) if e["event"] == "cooldown_start" and d0 - timedelta(hours=5) <= e["_dt"] < t10]
        if any(x[1] == "FIRE" for x in f):
            s = "5 a live rule fired, no candidate" + (" (cooldown)" if cd else "")
        elif f:
            s = "6 only with daily_range caps lifted"
        else:
            s = "7 no entry rule fired"
    stage[s] += 1
    by_stage_days[s].append(key)
n = len(win)
print("\n=== what the bot did before the +10%% close, %d rocket-days ===" % n)
for k in sorted(stage):
    print("  %-46s %4d  %5.1f%%" % (k, stage[k], 100 * stage[k] / n))
print("\n=== gates that blocked the rocket inside the window (days, a gate counted once per day) ===")
print("  %-34s %6s %8s %12s" % ("gate", "days", "% of all", "before +2.5%"))
for g, c in gate_any.most_common(20):
    print("  %-34s %6d %7.1f%% %12d" % (g, c, 100 * c / n, gate_early.get(g, 0)))
print("\n  first block on days where every candidate was blocked:", dict(gate_first.most_common(10)))

# heartbeat detail (rule layer) for the covered days
hb = collections.defaultdict(list)
for fp in sorted(glob.glob(str(FILES.parent / ".runtime/poll_heartbeat/*.jsonl"))):
    for l in io.open(fp, encoding="utf-8"):
        x = json.loads(l)
        if x.get("stage") == "evaluated" and x.get("tf") == "15m":
            hb[(x["bar_utc"][:10], x["sym"])].append(x)
cov = [k for k in win if k in hb]
print("\n=== rule layer (poll heartbeat), rocket-days it covers: %d ===" % len(cov))
why = collections.Counter()
for k in cov:
    d0, t25, t10 = win[k]
    rows = [x for x in hb[k] if x["bar_utc"] < t10.strftime("%Y-%m-%dT%H:%M")]
    seen = set()
    for x in rows:
        for rule, r in (x.get("reasons") or {}).items():
            if rule in ("breakout", "retest", "ema_cross"):
                continue
            t = re.sub(r"-?[\d.]+", "#", r)
            t = ("daily_range late" if "daily_range" in t or "от дна дня" in t else "RSI out of zone" if "RSI" in t
                 else "EMA structure" if "структура" in t or "EMA" in t and "выше" in t else "volume" if "vol" in t or "объём" in t
                 else "slope/ADX" if "slope" in t or "наклон" in t or "ADX" in t else "MACD" if "MACD" in t
                 else "momentum r1/r3" if "r1" in t or "r3" in t else "regime" if "режим" in t else t[:30])
            seen.add((rule, t))
    for s_ in seen:
        why[s_] += 1
for (rule, t), c in why.most_common(18):
    print("  %-10s %-24s %3d of %d days" % (rule, t, c, len(cov)))
for k in cov:
    print("  covered:", k, "stage:", next(s for s, ks in by_stage_days.items() if k in ks))

# ---------------- exits on the rockets the bot did buy ----------------
print("\n=== rockets the bot bought before +10%: how the position ended ===")
dayhi = {}
cls = collections.Counter()
rows = []
for key in by_stage_days["1 entered before +2.5%"] + by_stage_days["2 entered between +2.5% and +10%"]:
    day, sym = key
    d0, t25, t10 = win[key]
    ent = sorted((e for e in evs.get(sym, ()) if e["event"] == "entry" and d0 <= e["_dt"] < t10), key=lambda e: e["_dt"])[0]
    ex = next((e for e in sorted(evs.get(sym, ()), key=lambda e: e["_dt"]) if e["event"] == "exit" and e["_dt"] > ent["_dt"]), None)
    b = TD.bars_15m(sym)
    day_rows = [x for x in b if d0 <= x[0] < d0 + timedelta(days=1)]
    op, hi = day_rows[0][1], max(x[2] for x in day_rows)
    move = hi / op - 1
    if ex is None or not isinstance(ex.get("pnl_pct"), (int, float)):
        continue
    after = [x[2] for x in day_rows if x[0] >= ex["_dt"]]
    left = (max(after) / float(ex["exit_price"]) - 1) * 100 if after and ex.get("exit_price") else 0.0
    r = str(ex.get("reason") or "")
    c = ("WEAK: " + ("RSI divergence" if "RSI" in r else "volume exhaustion" if "объём" in r else "EMA fan" if "EMA-веер" in r
                     else "recheck" if "recheck" in r else "other")) if "WEAK" in r else \
        ("ATR trail" if "ATR" in r else "time max hold" if "время" in r else "EMA20 exit" if "EMA20" in r
         else "portfolio rotation" if "rotation" in r else "RSI overbought" if "перекуплен" in r else re.sub(r"[\d.]+", "#", r)[:30])
    cls[c] += 1
    rows.append((c, ex["pnl_pct"], left, 100 * move, (float(ex["pnl_pct"]) / (100 * move)) if move > 0 else 0))
n2 = len(rows)
import numpy as _np  # noqa: E402
print("  bought rockets with a closed position: %d; median pnl %+.1f%%, median day move %.1f%%, median capture %.0f%%, median rise LEFT after the exit (same day) %+.1f%%" % (
    n2, _np.median([x[1] for x in rows]), _np.median([x[3] for x in rows]), 100 * _np.median([x[4] for x in rows]), _np.median([x[2] for x in rows])))
print("  %-26s %5s %9s %11s" % ("exit reason", "n", "pnl med", "left med"))
for c, k in cls.most_common(10):
    v = [x for x in rows if x[0] == c]
    print("  %-26s %5d %+8.1f%% %+10.1f%%" % (c, k, _np.median([x[1] for x in v]), _np.median([x[2] for x in v])))
