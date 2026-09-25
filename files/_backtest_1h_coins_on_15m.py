"""E-1: poll coins the scan assigned to 1h ALSO on 15m -- judged by the goal.

WHY
strategy._run_analysis gives every coin ONE timeframe (the "best" report), and
_poll_coin polls it on that timeframe only. The audit (2026-09-25) found that
on winner-days whose first entry was on 1h the entry came +6.0 h after the
coin's +2.5% crossing (4% before it), against +2.3 h (20%) on 15m.

POPULATION
Immutable top-20 winner-days (watchlist INTERSECT global top-20, later-EOD
klines), bot up all day, intraday crossing time known. The coin's timeframe
that day is read from its own logged events (blocked / entry / exit /
ranker_shadow / cooldown / forward / shadow rows): "1h" = only 1h rows that day,
"15m" = only 15m, "mixed", "none" = nothing logged (timeframe unknown).

WHAT E-1 WOULD ADD
The 15m live-rule replay (.runtime/backtests/lateness_rows.jsonl, band FIRE =
the rules with the caps as coded; first bar per coin-hour) on "1h" days. A
replayed fire is a candidate, not an entry: the live gates were never run on
it, so the goal is given as upper bound (every fire enters), expected (each
fire passes with p = 15m candidate-bar -> entry rate from the log) and one try.

GUARD (per trade)
Added 15m trades vs the bot's current 15m entries, both on the calibrated
engine (live_trail, P-1), non-overlapping with real positions and each other,
cooldown COOLDOWN_BARS. Non-inferiority: lower 95% bound of
(combined mean - current mean) >= -0.10 pp.
"""
import collections
import io
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config as cfg  # noqa: E402
import immutable_labels as IL  # noqa: E402
import indicators as I  # noqa: E402
import label_store as LS  # noqa: E402
import _backtest_trend_start_detector as TD  # noqa: E402
import _compute_early_capture as E  # noqa: E402

BAR = timedelta(minutes=15)
START = "2026-03-01"
# classify a winner-day by the timeframe of its events inside the birth window
# (open .. crossing) rather than over the whole day; the day-level view is kept
# for comparison with --day-tf
BIRTH_WINDOW_TF = "--day-tf" not in sys.argv
COOLDOWN = int(getattr(cfg, "COOLDOWN_BARS", 8))
TF_EVENTS = {"blocked", "entry", "exit", "ranker_shadow", "cooldown_start", "forward",
             "surge_shadow_win", "peak_risk_shadow", "cooldown_realert"}
RULE_MODE = {"trend": "trend", "impulse_speed": "impulse_speed", "impulse": "impulse",
             "alignment": "alignment"}
FLOORS = {"impulse_speed": "TRAIL_MIN_BUFFER_PCT_IMPULSE_SPEED", "strong_trend": "TRAIL_MIN_BUFFER_PCT_STRONG_TREND",
          "impulse": "TRAIL_MIN_BUFFER_PCT_IMPULSE", "trend": "TRAIL_MIN_BUFFER_PCT_TREND",
          "alignment": "TRAIL_MIN_BUFFER_PCT_ALIGNMENT", "retest": "TRAIL_MIN_BUFFER_PCT_RETEST",
          "breakout": "TRAIL_MIN_BUFFER_PCT_BREAKOUT"}


def floor_pct(mode):
    if not getattr(cfg, "TRAIL_MIN_BUFFER_PCT_ENABLED", False):
        return 0.0
    return float(getattr(cfg, FLOORS.get(mode, "TRAIL_MIN_BUFFER_PCT_DEFAULT"), 0.0))


def live_trail(c, atr, ie, k, fl, max_bars=96):
    """P-1 calibrated live trail: close-anchored ratchet, exit at the first close below the stop."""
    ep = c[ie]
    a0 = atr[ie] if np.isfinite(atr[ie]) else 0.0
    stop = ep - max(k * a0, fl * ep)
    last = min(len(c) - 1, ie + max_bars)
    if last <= ie:
        return None
    for j in range(ie + 1, last + 1):
        aj = atr[j] if np.isfinite(atr[j]) else 0.0
        if aj > 0:
            stop = max(stop, c[j] - max(k * aj, fl * c[j]))
        if c[j] < stop:
            return j, (c[j] / ep - 1) * 100
    return last, (c[last] / ep - 1) * 100


def bar_of(d):
    return d.replace(minute=d.minute // 15 * 15, second=0, microsecond=0)


# ---------------- events ----------------
day_tf = collections.defaultdict(set)
tf_marks = collections.defaultdict(list)   # sym -> [(dt, tf)]
entries = collections.defaultdict(list)       # sym -> [(dt, tf, mode, trail_k)]
windows = collections.defaultdict(list)       # sym -> [(start, end)] real positions
cand15 = collections.defaultdict(set)         # (sym, bar) -> {"entry"/"blocked"} 15m
with io.open(FILES / "bot_events.jsonl", "rb") as fh:
    for raw in fh:
        try:
            e = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            continue
        ev = e.get("event")
        if ev not in TF_EVENTS:
            continue
        ts = str(e.get("ts", ""))
        if ts < START:
            continue
        d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        sym, tf = e.get("sym"), e.get("tf")
        if tf in ("15m", "1h"):
            day_tf[(d.strftime("%Y-%m-%d"), sym)].add(tf)
            tf_marks[sym].append((d, tf))
        if ev == "entry":
            entries[sym].append((d, tf, e.get("mode") or e.get("signal_mode"), e.get("trail_k")))
        if ev == "exit" and isinstance(e.get("bars_held"), (int, float)):
            step = BAR if tf == "15m" else timedelta(hours=1)
            windows[sym].append((d - step * (int(e["bars_held"]) + 1), d))
        if tf == "15m" and ev in ("entry", "blocked"):
            cand15[(sym, bar_of(d))].add(ev)

p = sum(1 for v in cand15.values() if "entry" in v) / max(1, len(cand15))
print("period %s .. %s" % (START, E.NOW.strftime("%Y-%m-%d")))
print("p = 15m candidate coin-bars -> entry: %d / %d = %.1f%%" % (
    sum(1 for v in cand15.values() if "entry" in v), len(cand15), 100 * p))


def in_position(sym, d):
    return any(a <= d <= b for a, b in windows.get(sym, ()))


def tf_class(key, window=None):
    """Timeframe the coin was polled on: over the whole day, or -- with window=(a, b) --
    only from its events inside [a, b] (the birth window is what E-1 is about)."""
    if window is None:
        s = day_tf.get(key, set())
    else:
        s = {tf for d, tf in tf_marks.get(key[1], ()) if window[0] <= d <= window[1]}
    return "none" if not s else ("mixed" if len(s) == 2 else next(iter(s)))


fires = collections.defaultdict(list)          # (day, sym) -> [(signal_dt, rule)]
fires_by_sym = collections.defaultdict(list)
for l in io.open(FILES.parent / ".runtime/backtests/lateness_rows.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["band"] != "FIRE" or r["day"] < START:
        continue
    if r["rule"] == "ema_cross":
        continue   # EMA_CROSS_ENABLED = False since 2026-04-19: not a live rule (checked last -> cross-only bar)
    t = datetime.fromisoformat(r["ts"])
    fires[(r["day"], r["sym"])].append((t + BAR, r["rule"]))
    fires_by_sym[r["sym"]].append((t, r["rule"], r["day"]))

# ---------------- 1. goal ----------------
wl = E.load_watchlist()
full, _, _ = E.load_uptime(datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=timezone.utc))
win, _ = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
dl = LS.intraday_deadlines()
W = sorted(k for k in win if START <= k[0] and k[0] in full and k in dl and dl[k][1] is not None)
print("\n=== 1. GOAL on %d winner-days (bot up all day, crossing known) ===" % len(W))
tab = collections.defaultdict(lambda: [0, 0, 0, 0.0, 0.0])   # n, early now, reachable, expected, one-try
lead_new = []
silent_with_fire = 0
by_month = collections.defaultdict(lambda: [0, 0, 0.0, 0])
for key in W:
    day, sym = key
    op, dd = dl[key]
    cls = tf_class(key, (op, dd)) if BIRTH_WINDOW_TF else tf_class(key)
    early = any(op <= x[0] < dd for x in entries.get(sym, ()))
    t = tab[cls]
    t[0] += 1
    t[1] += early
    m = by_month[day[:7]]
    m[0] += 1
    m[1] += early
    if early:
        t[3] += 1
        t[4] += 1
        m[2] += 1
        continue
    if cls == "none" and any(op <= x[0] < dd for x in fires.get(key, ())):
        silent_with_fire += 1          # timeframe unknown: no event in the window
    if cls != "1h":
        continue
    f = sorted(x for x in fires.get(key, ()) if op <= x[0] < dd and not in_position(sym, x[0]))
    if f:
        t[2] += 1
        t[3] += 1 - (1 - p) ** len(f)
        t[4] += p
        m[2] += 1 - (1 - p) ** len(f)
        m[3] += 1
        lead_new.append((dd - f[0][0]).total_seconds() / 3600)
n = len(W)
now = sum(t[1] for t in tab.values())
print("  coin timeframe (%s)   days   entered-before-crossing now" % ("birth window" if BIRTH_WINDOW_TF else "whole day"))
for cls in ("15m", "1h", "mixed", "none"):
    t = tab.get(cls)
    if t and t[0]:
        print("    %-6s                 %4d   %5.1f%% (%d)" % (cls, t[0], 100 * t[1] / t[0], t[1]))
t1 = tab["1h"]
exp_total = now + (t1[3] - t1[1])
one_total = now + (t1[4] - t1[1])
print("  winner-days with NO logged event in the birth window but a 15m fire there: %d -- their timeframe is unknown;"
      " the poll heartbeat (2026-09-25) records it" % silent_with_fire)
print("  1h winner-days with a 15m live-rule fire in the birth window (not entered now): %d of %d" % (t1[2], t1[0] - t1[1]))
print("  ALL winner-days entered before the crossing: now %5.1f%%  -> expected %5.1f%%  one-try %5.1f%%  upper %5.1f%%" % (
    100 * now / n, 100 * exp_total / n, 100 * one_total / n, 100 * (now + t1[2]) / n))
if lead_new:
    lead_new.sort()
    print("  newly reachable: first 15m fire %.1f h before the crossing (median, n=%d)" % (lead_new[len(lead_new) // 2], len(lead_new)))
print("  by month: days / now / expected / reachable 1h days")
for mo, (a, b, c_, d_) in sorted(by_month.items()):
    print("    %s  %4d   %5.1f%%   %5.1f%%   +%d" % (mo, a, 100 * b / a, 100 * c_ / a, d_))

# ---------------- 2. per trade ----------------
arm_cur, arm_add = [], []
one_h_days = {k for k in day_tf if tf_class(k) == "1h"}
for sym in sorted(set(fires_by_sym) | set(entries)):
    b = TD.bars_15m(sym)
    if not b:
        continue
    h = np.array([x[2] for x in b]); l = np.array([x[3] for x in b]); c = np.array([x[4] for x in b])
    atr = I._atr(h, l, c, cfg.ATR_PERIOD)
    idx = {x[0]: i for i, x in enumerate(b)}
    for d, tf, mode, k in entries.get(sym, ()):
        if tf != "15m":
            continue
        i = idx.get(bar_of(d) - BAR)
        if i is None:
            continue
        r = live_trail(c, atr, i, float(k) if isinstance(k, (int, float)) else 2.0, floor_pct(mode))
        if r:
            arm_cur.append((d.strftime("%Y-%m"), r[1]))
    busy = None
    for t, rule, day in sorted(fires_by_sym.get(sym, ())):
        if (day, sym) not in one_h_days:
            continue
        if busy is not None and t <= busy:
            continue
        if in_position(sym, t + BAR):
            continue
        i = idx.get(t)
        if i is None:
            continue
        mode = RULE_MODE.get(rule, "trend")
        r = live_trail(c, atr, i, 2.0, floor_pct(mode))
        if not r:
            continue
        arm_add.append((day[:7], r[1], rule))
        busy = b[r[0]][0] + BAR * COOLDOWN

rnd = random.Random(5)


def diff_ci(cur, add, w):
    def comb(a, bb):
        return (sum(a) + w * sum(bb)) / (len(a) + w * len(bb)) - sum(a) / len(a)
    pt = comb(cur, add)
    bs = sorted(comb([rnd.choice(cur) for _ in cur], [rnd.choice(add) for _ in add]) for _ in range(1000))
    return pt, bs[25], bs[-26]


cur = [x[1] for x in arm_cur]
add = [x[1] for x in arm_add]
print("\n=== 2. PER TRADE, calibrated engine for both arms ===")
print("  current 15m entries (all modes): n=%d  mean %+.3f%%  median %+.3f%%" % (len(cur), np.mean(cur), np.median(cur)))
print("  added 15m fires on 1h days:       n=%d  mean %+.3f%%  median %+.3f%%" % (len(add), np.mean(add), np.median(add)))
for rule in sorted({x[2] for x in arm_add}):
    v = [x[1] for x in arm_add if x[2] == rule]
    print("     %-14s n=%4d  mean %+.3f%%" % (rule, len(v), np.mean(v)))
for w, name in ((p, "expected (x p)"), (1.0, "upper bound (all)")):
    pt, lo, hi = diff_ci(cur, add, w)
    print("  combined - current, %-18s %+.3f pp  95%% CI [%+.3f, %+.3f]  -> %s" % (
        name, pt, lo, hi, "NON-INFERIOR" if lo >= -0.10 else "FAILS the -0.10 pp bound"))
print("  by month: current mean (n) / added mean (n)")
for mo in sorted({x[0] for x in arm_cur} | {x[0] for x in arm_add}):
    a = [x[1] for x in arm_cur if x[0] == mo]
    bb = [x[1] for x in arm_add if x[0] == mo]
    print("    %s  %+.3f (%4d)   %+.3f (%4d)" % (mo, np.mean(a) if a else float("nan"), len(a), np.mean(bb) if bb else float("nan"), len(bb)))
