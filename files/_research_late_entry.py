"""Anatomy of the late entry -- the dominant loss (incident reports 09-25..29).

Population: immutable later-EOD top-20 winner-days on the watchlist, bot up all
day, first +2.5% crossing known (label store), 2026-03-01 .. last labelled day.
For each winner-day, from the 15m long store and bot_events.jsonl:

  timing   hours from the UTC open to the crossing; share of the day's move
           (move low -> day high) already done at the UTC open (the move began
           the day before -- no UTC-anchored entry can be "early" for it)
  bot      first entry of the day: before the crossing (early), after (late),
           none; a position held from before the day is its own class
  late     hours after the crossing; share of the move done at the entry price
           (the operator's definition of early: relative to the MOVE)
  before   what existed in the birth window [UTC open, crossing) that a change
           could have acted on:
             blocked   the bot logged a candidate and a gate blocked it (gate)
             fired     a live 15m entry rule fired on the stored bars (strategy
                       replay, ema_cross excluded) but no candidate was logged
             nothing   no rule fired -- a detection problem, not a gate problem
"""
import bisect
import collections
import io
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import _backtest_trend_start_detector as TD  # noqa: E402
import _compute_early_capture as E  # noqa: E402
import immutable_labels as IL  # noqa: E402
import label_store as LS  # noqa: E402
import incident_analyst as IA  # noqa: E402

START = "2026-03-01"
UTC = timezone.utc


def dt(s):
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=UTC)


wl = E.load_watchlist()
full, _, _ = E.load_uptime(dt(START + "T00:00:00Z"))
win, _ = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
dl = LS.intraday_deadlines()
W = sorted(k for k in win if k[0] >= START and k[0] in full and k in dl and dl[k][1] is not None)
syms = {s for _, s in W}
print("winner-days %s .. %s, bot up all day, crossing known: %d (%d coins)" % (START, W[-1][0], len(W), len(syms)))

# bot events for these coins
ev = collections.defaultdict(list)
with io.open(FILES / "bot_events.jsonl", "rb") as fh:
    for raw in fh:
        if not (b'"entry"' in raw or b'"exit"' in raw or b'"blocked"' in raw):
            continue
        try:
            e = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            continue
        if e.get("sym") not in syms or str(e.get("ts", "")) < "2026-02-25":
            continue
        e["_dt"] = dt(e["ts"])
        ev[e["sym"]].append(e)
for v in ev.values():
    v.sort(key=lambda e: e["_dt"])

# live-rule fires on stored 15m bars (strategy replay), ema_cross is not live
fires = collections.defaultdict(list)
for line in io.open(FILES.parent / ".runtime/backtests/lateness_rows.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["sym"] in syms and r["rule"] != "ema_cross" and r["band"] == "FIRE" and r["day"] >= START:
        fires[(r["day"], r["sym"])].append((dt(r["ts"]) + timedelta(minutes=15), r["rule"]))
replay_last = max(d for d, _ in fires) if fires else None

rows = []
for day, sym in W:
    op, t25 = dl[(day, sym)]
    b = TD.bars_15m(sym)
    ts = [x[0] for x in b]
    i0 = bisect.bisect_left(ts, op)
    i1 = bisect.bisect_left(ts, op + timedelta(days=1))
    if i0 >= len(b) or b[i0][0] != op or i1 - i0 < 90:
        continue
    day_bars = b[i0:i1]
    ih = max(range(len(day_bars)), key=lambda k: day_bars[k][2])
    t_high, hi = day_bars[ih][0], day_bars[ih][2]
    j0 = bisect.bisect_left(ts, op - timedelta(days=1))
    low = min(x[3] for x in b[j0:i0 + ih + 1])
    open_px = day_bars[0][1]
    span = hi - low
    if span <= 0:
        continue
    r = {"day": day, "sym": sym, "cross_h": (t25 - op).total_seconds() / 3600,
         "pre_open_done": (open_px - low) / span, "move_pct": (hi / low - 1) * 100,
         "day_move_pct": (hi / open_px - 1) * 100}
    evs = ev.get(sym, [])
    prior = [e for e in evs if e["_dt"] < op and e["event"] in ("entry", "exit")]
    held = bool(prior) and prior[-1]["event"] == "entry"
    ents = [e for e in evs if e["event"] == "entry" and op <= e["_dt"] < op + timedelta(days=1)]
    if held:
        r["stage"] = "held"
    elif ents and ents[0]["_dt"] < t25:
        r["stage"] = "early"
    elif ents:
        r["stage"] = "late"
    else:
        r["stage"] = "none"
    if ents:
        e0 = ents[0]
        px = float(e0.get("price") or 0)
        r["entry_after_cross_h"] = (e0["_dt"] - t25).total_seconds() / 3600
        r["entry_done"] = (px - low) / span if px else None
        r["entry_mode"] = "%s/%s" % (e0.get("mode") or e0.get("signal_mode"), e0.get("tf"))
    blk = [e for e in evs if e["event"] == "blocked" and op <= e["_dt"] < t25]
    fr = [f for f in fires.get((day, sym), []) if op <= f[0] < t25]
    r["blocked_gates"] = sorted({IA.gate_name(e) for e in blk})
    r["first_gate"] = IA.gate_name(blk[0]) if blk else None
    r["fired_rules"] = sorted({f[1] for f in fr})
    r["before"] = "blocked" if blk else ("fired" if fr else ("nothing" if replay_last and day <= replay_last else "no_replay"))
    rows.append(r)

n = len(rows)
st = collections.Counter(r["stage"] for r in rows)
print("\n=== 1. WHERE THE WINNER-DAYS END UP (n=%d) ===" % n)
for k in ("early", "late", "none", "held"):
    print("  %-6s %4d  %5.1f%%" % (k, st[k], 100 * st[k] / n))

ch = sorted(r["cross_h"] for r in rows)
pod = sorted(r["pre_open_done"] for r in rows)
print("\n=== 2. TIMING OF THE MOVE ITSELF ===")
print("  hours UTC open -> +2.5%% crossing: median %.1f, p25 %.1f, p75 %.1f" % (
    np.median(ch), np.percentile(ch, 25), np.percentile(ch, 75)))
for h in (1, 2, 4):
    print("  crossing within %d h of the open: %.1f%%" % (h, 100 * sum(1 for x in ch if x <= h) / n))
print("  share of the move (low since D-1 00:00 -> day high) already done at the UTC open: median %.0f%%;"
      " >= 30%% done on %.1f%% of winner-days" % (100 * np.median(pod), 100 * sum(1 for x in pod if x >= 0.3) / n))

late = [r for r in rows if r["stage"] == "late"]
print("\n=== 3. THE LATE ENTRIES (n=%d) ===" % len(late))
if late:
    a = sorted(r["entry_after_cross_h"] for r in late)
    d = sorted(r["entry_done"] for r in late if r.get("entry_done") is not None)
    print("  entered after the crossing by: median %.1f h (p25 %.1f, p75 %.1f)" % (np.median(a), np.percentile(a, 25), np.percentile(a, 75)))
    print("  share of the move already done at the entry price: median %.0f%% (p25 %.0f%%, p75 %.0f%%)" % (
        100 * np.median(d), 100 * np.percentile(d, 25), 100 * np.percentile(d, 75)))
    print("  entry mode: %s" % dict(collections.Counter(r["entry_mode"] for r in late).most_common(6)))
    early = [r for r in rows if r["stage"] == "early" and r.get("entry_done") is not None]
    if early:
        print("  (early entries for comparison: move done at entry median %.0f%%, n=%d)" % (
            100 * np.median([r["entry_done"] for r in early]), len(early)))

print("\n=== 4. WHAT EXISTED BEFORE THE CROSSING -- late and never-entered winner-days ===")
for grp in ("late", "none"):
    g = [r for r in rows if r["stage"] == grp]
    if not g:
        continue
    c = collections.Counter(r["before"] for r in g)
    print("  %s (n=%d): %s" % (grp, len(g), ", ".join("%s %d (%.0f%%)" % (k, v, 100 * v / len(g)) for k, v in c.most_common())))
    fg = collections.Counter(r["first_gate"] for r in g if r["first_gate"])
    ag = collections.Counter(x for r in g for x in r["blocked_gates"])
    print("     first gate: %s" % dict(fg.most_common(8)))
    print("     any gate in the window (days): %s" % dict(ag.most_common(8)))
    fr = collections.Counter(x for r in g if r["before"] == "fired" for x in r["fired_rules"])
    print("     rules that fired without a candidate: %s" % dict(fr.most_common(6)))

print("\n=== 5. LATE x TIMING: is the lateness structural? ===")
for grp in ("early", "late", "none"):
    g = [r for r in rows if r["stage"] == grp]
    if g:
        print("  %-5s crossing median %.1f h after open; move done at open median %.0f%%; day move median %.1f%%" % (
            grp, np.median([r["cross_h"] for r in g]), 100 * np.median([r["pre_open_done"] for r in g]),
            np.median([r["day_move_pct"] for r in g])))
print("\nreplay covers up to %s; winner-days after it are 'no_replay'" % replay_last)
json.dump(rows, io.open(FILES.parent / ".runtime/backtests/late_entry_rows.json", "w", encoding="utf-8"), default=str)
