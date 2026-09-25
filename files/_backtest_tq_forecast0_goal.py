"""P0 hypothesis P-3 / E-4: judge the trend_quality "forecast 0.000" block by the GOAL.

The forecast is the last check of trend_quality (monitor._trend_entry_quality_guard_reason):
a block reading "weak 15m trend (forecast 0.000 ...)" passed price edge,
daily_range and RSI and failed only because the coin had no rule signals yet
today -- no data, scored as a bad forecast. On 2026-09-18 this was refuted
per-trade (4h peak 0.94x of what passes). The audit's criterion is different:

  goal     share of immutable top-20 winner-days the bot entered BEFORE the
           coin's first +2.5% crossing from the UTC open (and how much earlier)
  guard    per-trade non-inferiority: lower 95% bound of
           (mean pnl with the added entries - mean pnl without) >= -0.10 pp

Relaxation under test: forecast exactly 0.000 counts as "no data" -> the
forecast requirement is skipped. A relaxed block is not an entry yet: the gates
after trend_quality were never evaluated for it. So three readings are given:
upper bound (every relaxed candidate enters), expected (each passes with the
downstream pass rate p measured on 15m candidates the bot logged after
trend_quality), one-try (only the first relaxed candidate of the day, with p).

Per-trade pnl of BOTH arms is computed with the same calibrated engine
(`live_trail`, P-1: mean bias +0.007% on real ATR-trail exits), from the entry
bar, trail_k = median of real 15m trend entries, so the comparison is matched.
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
START = "2026-05-01"          # first month with "forecast 0.000" trend_quality blocks
# gates evaluated AFTER trend_quality in monitor._poll_coin, by their logged signal_type
POST_TQ = {"mode_range_quality", "ranker_hard_veto", "clone_guard", "open_cluster_cap",
           "correlation_guard", "late_impulse_rotation", "bandit_skip"}
FLOOR = float(getattr(cfg, "TRAIL_MIN_BUFFER_PCT_TREND", 0.0)) if getattr(cfg, "TRAIL_MIN_BUFFER_PCT_ENABLED", False) else 0.0
COOLDOWN = int(getattr(cfg, "COOLDOWN_BARS", 8))


def bar_of(d):
    return d.replace(minute=d.minute // 15 * 15, second=0, microsecond=0)


def live_trail(c, atr, ie, k, max_bars=96):
    """P-1 calibrated live trail, trend floor; exit at the first close below the stop."""
    ep = c[ie]
    a0 = atr[ie] if np.isfinite(atr[ie]) else 0.0
    stop = ep - max(k * a0, FLOOR * ep)
    last = min(len(c) - 1, ie + max_bars)
    if last <= ie:
        return None
    for j in range(ie + 1, last + 1):
        aj = atr[j] if np.isfinite(atr[j]) else 0.0
        if aj > 0:
            stop = max(stop, c[j] - max(k * aj, FLOOR * c[j]))
        if c[j] < stop:
            return j, (c[j] / ep - 1) * 100
    return last, (c[last] / ep - 1) * 100


# ---------------- events ----------------
blocks = collections.defaultdict(set)      # sym -> {bar} forecast-0 TQ blocks, 15m
entries = collections.defaultdict(list)    # sym -> [(ts, tf, mode, trail_k)]
windows = collections.defaultdict(list)    # sym -> [(start, end)] real positions (any tf)
after_tq = collections.defaultdict(set)    # (sym, bar) -> outcome after trend_quality, 15m trend
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
        sym, ev, tf = e.get("sym"), e.get("event"), e.get("tf")
        if ev == "blocked" and tf == "15m":
            rc = e.get("reason_code")
            if rc == "trend_quality":
                if "forecast 0.000" in str(e.get("reason")):
                    blocks[sym].add(bar_of(d))
            elif str(e.get("signal_type")) in POST_TQ:
                after_tq[(sym, bar_of(d))].add("blocked:" + str(e.get("signal_type")))
        elif ev == "entry":
            entries[sym].append((d, tf, e.get("mode") or e.get("signal_mode"), e.get("trail_k")))
            if tf == "15m":
                after_tq[(sym, bar_of(d))].add("entry")
        elif ev == "exit" and isinstance(e.get("bars_held"), (int, float)):
            step = BAR if tf == "15m" else timedelta(hours=1)
            windows[sym].append((d - step * (int(e["bars_held"]) + 1), d))

n_after = len(after_tq)
n_ent = sum(1 for v in after_tq.values() if "entry" in v)
p = n_ent / max(1, n_after)
late = {k: v for k, v in after_tq.items() if k[1] >= datetime(2026, 9, 7, 20, tzinfo=timezone.utc)}
p_late = sum(1 for v in late.values() if "entry" in v) / max(1, len(late))
post = collections.Counter(x for v in after_tq.values() if "entry" not in v for x in v)
print("post-trend_quality blocks by gate (15m coin-bars, no entry on the bar):", dict(post.most_common()))
print("p since the bandit gate went off (2026-09-07): %.1f%% of %d coin-bars" % (100 * p_late, len(late)))
ks = sorted(float(x[3]) for v in entries.values() for x in v if x[1] == "15m" and x[2] == "trend" and isinstance(x[3], (int, float)))
K = ks[len(ks) // 2] if ks else 2.0
print("period %s .. %s | forecast-0 TQ blocks: %d coin-bars on %d coins" % (
    START, E.NOW.strftime("%Y-%m-%d"), sum(len(v) for v in blocks.values()), len(blocks)))
print("downstream pass rate p (15m candidate-bars that reached the gates after trend_quality -> entry): %d / %d = %.1f%%"
      "  [all modes: blocked events do not record the mode]" % (n_ent, n_after, 100 * p))
print("trail_k for both arms = median of real 15m trend entries: %.2f (n=%d), trend floor %.3f" % (K, len(ks), FLOOR))


def in_position(sym, d):
    return any(a <= d <= b for a, b in windows.get(sym, ()))


# ---------------- 1. goal: winner-days entered before the crossing ----------------
wl = E.load_watchlist()
full, _, _ = E.load_uptime(datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=timezone.utc))
win, _ = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
dl = LS.intraday_deadlines()
W = sorted(k for k in win if START <= k[0] and k[0] in full and k in dl and dl[k][1] is not None)
cur_hit, up_hit, exp_hit, one_hit = 0, 0, 0.0, 0.0
gain_h, first_rel_h = [], []
by_month = collections.defaultdict(lambda: [0, 0, 0.0, 0])
for day, sym in W:
    op, dd = dl[(day, sym)]
    early = [x[0] for x in entries.get(sym, ()) if op <= x[0] < dd]
    rel = sorted(b for b in blocks.get(sym, ()) if op <= b + BAR < dd and not in_position(sym, b + BAR))
    cur = bool(early)
    m = by_month[day[:7]]
    m[0] += 1
    cur_hit += cur
    m[1] += cur
    if cur:
        up_hit += 1
        exp_hit += 1
        one_hit += 1
        m[2] += 1
        if rel and rel[0] + BAR < min(early):
            gain_h.append((min(early) - rel[0] - BAR).total_seconds() / 3600)
        continue
    if rel:
        up_hit += 1
        m[3] += 1
        exp_hit += 1 - (1 - p) ** len(rel)
        m[2] += 1 - (1 - p) ** len(rel)
        one_hit += p
        first_rel_h.append((dd - rel[0] - BAR).total_seconds() / 3600)
n = len(W)
print("\n=== 1. GOAL on %d winner-days (immutable top-20, bot up all day, crossing time known) ===" % n)
print("  entered before the +2.5%% crossing, now:             %5.1f%%  (%d)" % (100 * cur_hit / n, cur_hit))
print("  + forecast-0 relaxation, upper bound (all enter):   %5.1f%%  (+%d days)" % (100 * up_hit / n, up_hit - cur_hit))
print("  + relaxation, expected (each relaxed bar passes p): %5.1f%%" % (100 * exp_hit / n))
print("  + relaxation, one try per day (p):                  %5.1f%%" % (100 * one_hit / n))
if first_rel_h:
    first_rel_h.sort()
    print("  newly reachable days: first relaxed candidate %.1f h before the crossing (median, n=%d)" % (first_rel_h[len(first_rel_h) // 2], len(first_rel_h)))
if gain_h:
    gain_h.sort()
    print("  already-entered days where a relaxed candidate came earlier: n=%d, median %.1f h earlier" % (len(gain_h), gain_h[len(gain_h) // 2]))
print("  by month: days / now / expected / upper-bound additions")
for mo, (a, b, c_, d_) in sorted(by_month.items()):
    print("    %s  %4d   %5.1f%%   %5.1f%%   +%d" % (mo, a, 100 * b / a, 100 * c_ / a, d_))

# ---------------- 2. per-trade non-inferiority ----------------
arm_cur, arm_add = [], []
rel_windows = collections.defaultdict(list)
displaced = []
for sym in sorted(set(blocks) | set(entries)):
    b = TD.bars_15m(sym)
    if not b:
        continue
    h = np.array([x[2] for x in b]); l = np.array([x[3] for x in b]); c = np.array([x[4] for x in b])
    atr = I._atr(h, l, c, cfg.ATR_PERIOD)
    idx = {x[0]: i for i, x in enumerate(b)}
    for d, tf, mode, _k in entries.get(sym, ()):
        if tf != "15m" or mode != "trend":
            continue
        i = idx.get(bar_of(d) - BAR)          # decision taken on the last closed bar
        if i is None:
            continue
        r = live_trail(c, atr, i, K)
        if r:
            arm_cur.append((d.strftime("%Y-%m"), r[1]))
    busy_until = None
    for bk in sorted(blocks.get(sym, ())):
        if busy_until is not None and bk <= busy_until:
            continue
        if in_position(sym, bk + BAR):
            continue
        i = idx.get(bk - BAR)
        if i is None:
            continue
        r = live_trail(c, atr, i, K)
        if not r:
            continue
        arm_add.append((bk.strftime("%Y-%m"), r[1]))
        busy_until = b[r[0]][0] + BAR * COOLDOWN
        rel_windows[sym].append((bk + BAR, busy_until))
    for d, tf, mode, _k in entries.get(sym, ()):
        if tf == "15m" and mode == "trend" and any(a < d <= z for a, z in rel_windows[sym]):
            displaced.append(d)
rnd = random.Random(11)


def diff_ci(cur, add, w):
    """combined mean (added entries weighted by w) minus current mean, with a bootstrap 95% CI."""
    def comb(a, b):
        return (sum(a) + w * sum(b)) / (len(a) + w * len(b)) - sum(a) / len(a)
    pt = comb(cur, add)
    bs = sorted(comb([rnd.choice(cur) for _ in cur], [rnd.choice(add) for _ in add]) for _ in range(1000))
    return pt, bs[25], bs[-26]


cur = [x[1] for x in arm_cur]
add = [x[1] for x in arm_add]
print("\n=== 2. PER-TRADE, same calibrated engine for both arms ===")
print("  current 15m trend entries: n=%d  mean %+.3f%%  median %+.3f%%" % (len(cur), np.mean(cur), np.median(cur)))
print("  relaxed forecast-0 entries: n=%d  mean %+.3f%%  median %+.3f%%  (non-overlapping, cooldown %d bars)" % (
    len(add), np.mean(add), np.median(add), COOLDOWN))
for w, name in ((p, "expected (x p)"), (1.0, "upper bound (all)")):
    pt, lo, hi = diff_ci(cur, add, w)
    print("  combined - current, %-18s %+.3f pp  95%% CI [%+.3f, %+.3f]  -> %s" % (
        name, pt, lo, hi, "NON-INFERIOR" if lo >= -0.10 else "FAILS the -0.10 pp bound"))
print("  displacement: %d of %d current 15m trend entries fall inside a relaxed position or its cooldown"
      " (upper bound -- at weight p only ~%.0f would)" % (len(displaced), len(cur), p * len(displaced)))
print("  by month: current mean / relaxed mean (n)")
for mo in sorted({x[0] for x in arm_cur} | {x[0] for x in arm_add}):
    a = [x[1] for x in arm_cur if x[0] == mo]
    bb = [x[1] for x in arm_add if x[0] == mo]
    print("    %s  %+.3f (%4d)   %+.3f (%4d)" % (mo, np.mean(a) if a else float("nan"), len(a), np.mean(bb) if bb else float("nan"), len(bb)))
