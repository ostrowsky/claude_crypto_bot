"""E-5 restated: should EMA_CROSS be switched back on? Judged by the goal.

WHY
The audit (2026-09-25) counted ema_cross among the live entry rules and
proposed lowering its volume threshold. It is not live: EMA_CROSS_ENABLED =
False since 2026-04-19 ("7 trades in 7 days, win 14%, sum -9.2%") -- a verdict
on one week. The heartbeat confirms it ("disabled" on every evaluated poll).
The 15m rule replay (.runtime/backtests/lateness_rows.jsonl) checks ema_cross
LAST, so rows with rule == "ema_cross" are bars where ONLY that rule fired:
exactly what switching it on would add.

LIVE PATH OF SUCH A CANDIDATE
_poll_coin gives a cross-only candidate preview_mode "alignment" (the else
branch), so it meets the alignment gates: mode_range_quality needs
daily_range >= ALIGNMENT_15M_RANGE_MIN (4%) on 15m, while ema_cross itself
needs daily_range <= CROSS_RANGE_MAX (6%). Rows are filtered to that band; the
unfiltered count is printed beside it. Trail = alignment floor.

CRITERION (as P-3 / E-1)
goal     immutable top-20 winner-days entered before the +2.5% crossing:
         now vs expected (each added fire passes the later gates with p, the
         logged 15m candidate -> entry rate) vs upper bound
guard    per-trade non-inferiority on the calibrated engine: lower 95% bound of
         (combined mean - current mean) >= -0.10 pp; plus 4h-peak precision vs
         the base of every live-rule fire
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
COOLDOWN = int(getattr(cfg, "COOLDOWN_BARS", 8))
RMIN = float(getattr(cfg, "ALIGNMENT_15M_RANGE_MIN", 4.0))
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


entries = collections.defaultdict(list)
windows = collections.defaultdict(list)
cand15 = collections.defaultdict(set)
with io.open(FILES / "bot_events.jsonl", "rb") as fh:
    for raw in fh:
        if not (b'"entry"' in raw or b'"exit"' in raw or b'"blocked"' in raw):
            continue
        try:
            e = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            continue
        ts = str(e.get("ts", ""))
        if ts < START:
            continue
        d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        ev, sym, tf = e.get("event"), e.get("sym"), e.get("tf")
        if ev == "entry":
            entries[sym].append((d, tf, e.get("mode") or e.get("signal_mode"), e.get("trail_k")))
        elif ev == "exit" and isinstance(e.get("bars_held"), (int, float)):
            step = BAR if tf == "15m" else timedelta(hours=1)
            windows[sym].append((d - step * (int(e["bars_held"]) + 1), d))
        if tf == "15m" and ev in ("entry", "blocked"):
            cand15[(sym, bar_of(d))].add(ev)
p = sum(1 for v in cand15.values() if "entry" in v) / max(1, len(cand15))


def in_position(sym, d):
    return any(a <= d <= b for a, b in windows.get(sym, ()))


cross = collections.defaultdict(list)       # (day, sym) -> [signal time]
cross_rows = collections.defaultdict(list)  # sym -> [(bar time, day, dr, peak_4h)]
n_all, n_band, live_pk = 0, 0, []
for l in io.open(FILES.parent / ".runtime/backtests/lateness_rows.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["band"] != "FIRE" or r["day"] < START:
        continue
    if r["rule"] != "ema_cross":
        live_pk.append(r["peak_4h"])
        continue
    n_all += 1
    if not (RMIN <= float(r["dr"]) <= float(getattr(cfg, "CROSS_RANGE_MAX", 6.0))):
        continue
    n_band += 1
    t = datetime.fromisoformat(r["ts"])
    cross[(r["day"], r["sym"])].append(t + BAR)
    cross_rows[r["sym"]].append((t, r["day"], r["dr"], r["peak_4h"]))
print("period %s .. %s | cross-only fires (first bar per coin-hour): %d, in the %.0f..%.0f%% range band: %d" % (
    START, E.NOW.strftime("%Y-%m-%d"), n_all, RMIN, getattr(cfg, "CROSS_RANGE_MAX", 6.0), n_band))
print("p = 15m candidate coin-bars -> entry: %.1f%%" % (100 * p))
pk_c = [x[3] for v in cross_rows.values() for x in v if x[3] is not None]
live_pk = [x for x in live_pk if x is not None]   # last bars of the store have no 4h future
print("4h peak >= 3%%: cross-only fires %.1f%% (n=%d) vs every live-rule fire %.1f%% (n=%d)" % (
    100 * np.mean([x >= 3 for x in pk_c]), len(pk_c), 100 * np.mean([x >= 3 for x in live_pk]), len(live_pk)))

wl = E.load_watchlist()
full, _, _ = E.load_uptime(datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=timezone.utc))
win, _ = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
dl = LS.intraday_deadlines()
W = sorted(k for k in win if START <= k[0] and k[0] in full and k in dl and dl[k][1] is not None)
now = reach = 0
expct = 0.0
lead = []
by_m = collections.defaultdict(lambda: [0, 0, 0.0, 0])
for key in W:
    day, sym = key
    op, dd = dl[key]
    m = by_m[day[:7]]
    m[0] += 1
    if any(op <= x[0] < dd for x in entries.get(sym, ())):
        now += 1
        m[1] += 1
        continue
    f = sorted(t for t in cross.get(key, ()) if op <= t < dd and not in_position(sym, t))
    if f:
        reach += 1
        m[3] += 1
        expct += 1 - (1 - p) ** len(f)
        m[2] += 1 - (1 - p) ** len(f)
        lead.append((dd - f[0]).total_seconds() / 3600)
n = len(W)
print("\n=== GOAL on %d winner-days (bot up all day, crossing known) ===" % n)
print("  entered before the crossing: now %.1f%%  -> expected %.1f%%  upper %.1f%%  (+%d reachable days)" % (
    100 * now / n, 100 * (now + expct) / n, 100 * (now + reach) / n, reach))
if lead:
    lead.sort()
    print("  reachable days: first cross fire %.1f h before the crossing (median)" % lead[len(lead) // 2])
print("  by month: days / now / expected / reachable")
for mo, (a, b, c_, d_) in sorted(by_m.items()):
    print("    %s  %4d  %5.1f%%  %5.1f%%  +%d" % (mo, a, 100 * b / a, 100 * (b + c_) / a, d_))

arm_cur, arm_add = [], []
for sym in sorted(set(cross_rows) | set(entries)):
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
    for t, day, dr, pk in sorted(cross_rows.get(sym, ())):
        if busy is not None and t <= busy:
            continue
        if in_position(sym, t + BAR):
            continue
        i = idx.get(t)
        if i is None:
            continue
        r = live_trail(c, atr, i, 2.0, floor_pct("alignment"))
        if not r:
            continue
        arm_add.append((day[:7], r[1]))
        busy = b[r[0]][0] + BAR * COOLDOWN
rnd = random.Random(9)


def diff_ci(cur, add, w):
    def comb(a, bb):
        return (sum(a) + w * sum(bb)) / (len(a) + w * len(bb)) - sum(a) / len(a)
    pt = comb(cur, add)
    bs = sorted(comb([rnd.choice(cur) for _ in cur], [rnd.choice(add) for _ in add]) for _ in range(1000))
    return pt, bs[25], bs[-26]


cur = [x[1] for x in arm_cur]
add = [x[1] for x in arm_add]
print("\n=== PER TRADE, calibrated engine ===")
print("  current 15m entries: n=%d  mean %+.3f%%  median %+.3f%%" % (len(cur), np.mean(cur), np.median(cur)))
print("  added cross-only:    n=%d  mean %+.3f%%  median %+.3f%%  (alignment trail floor %.2f)" % (
    len(add), np.mean(add), np.median(add), floor_pct("alignment")))
for w, name in ((p, "expected (x p)"), (1.0, "upper bound (all)")):
    pt, lo, hi = diff_ci(cur, add, w)
    print("  combined - current, %-18s %+.3f pp  95%% CI [%+.3f, %+.3f]  -> %s" % (
        name, pt, lo, hi, "NON-INFERIOR" if lo >= -0.10 else "FAILS the -0.10 pp bound"))
print("  by month: current mean (n) / added mean (n)")
for mo in sorted({x[0] for x in arm_cur} | {x[0] for x in arm_add}):
    a = [x[1] for x in arm_cur if x[0] == mo]
    bb = [x[1] for x in arm_add if x[0] == mo]
    print("    %s  %+.3f (%4d)   %+.3f (%4d)" % (mo, np.mean(a) if a else float("nan"), len(a), np.mean(bb) if bb else float("nan"), len(bb)))
