"""What would switching the ML gate off do to the goal? Maximum period of the log.

The ML signal model blocks candidates in two places: the general zone gate
(ml_proba_zone: ml_proba < ML_GENERAL_HARD_BLOCK_MIN, live 0.15) and the
non-bull trend filter (ml_filter: ML_TREND_NONBULL_MIN_PROBA 0.35). Both sit
early in _poll_coin (after impulse_guard, before entry_score / trend_quality /
every later gate), so a candidate they blocked never met the later gates.

Counterfactual "ML off": every ML-blocked candidate (dedup per coin, timeframe,
bar) continues down the pipeline and enters with probability p = the logged
rate at which candidates that got PAST the ML gate became entries, per
timeframe. Criterion as P-3 (p0-validation-0925-spec.md):
  goal    immutable top-20 winner-days (bot up all day, +2.5% crossing time
          known) entered before the crossing: now vs expected vs upper bound
  guard   per trade, calibrated live trail (P-1 engine) for BOTH arms,
          non-inferiority: lower 95% bound of (combined - current) >= -0.10 pp
Split by the model change (peak-label model live 2026-09-07 19:31) and by
month: the gate's threshold and the model's scale changed over time, but
switching it off is judged on whatever it actually blocked then.
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

START = "2026-04-06"                     # first logged ml_proba_zone block
MODEL_CHANGE = "2026-09-07T19:31"
ML_GATES = {"ml_proba_zone", "ml_filter"}
PRE_ML = {"impulse_guard"}               # gates before the ML gate (not "past" it)
STEP = {"15m": timedelta(minutes=15), "1h": timedelta(hours=1)}
COOLDOWN = int(getattr(cfg, "COOLDOWN_BARS", 8))
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


def bar_of(d, tf):
    if tf == "1h":
        return d.replace(minute=0, second=0, microsecond=0)
    return d.replace(minute=d.minute // 15 * 15, second=0, microsecond=0)


ml_blocks = collections.defaultdict(dict)      # (sym, tf) -> {bar: ml_proba}
past_ml = collections.defaultdict(set)         # (tf, sym, bar) -> outcomes of candidates that got past ML
entries = collections.defaultdict(list)        # sym -> [(dt, tf, mode, trail_k)]
windows = collections.defaultdict(list)
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
        if tf not in STEP:
            continue
        if ev == "entry":
            entries[sym].append((d, tf, e.get("mode") or e.get("signal_mode"), e.get("trail_k")))
            past_ml[(tf, sym, bar_of(d, tf))].add("entry")
        elif ev == "exit" and isinstance(e.get("bars_held"), (int, float)):
            windows[sym].append((d - STEP[tf] * (int(e["bars_held"]) + 1), d))
        elif ev == "blocked":
            st = str(e.get("signal_type"))
            if st in ML_GATES:
                ml_blocks[(sym, tf)].setdefault(bar_of(d, tf), e.get("ml_proba"))
            elif st not in PRE_ML:
                past_ml[(tf, sym, bar_of(d, tf))].add("blocked")

p = {}
for tf in STEP:
    rows = [v for k, v in past_ml.items() if k[0] == tf]
    p[tf] = sum(1 for v in rows if "entry" in v) / max(1, len(rows))
n_ml = {tf: sum(len(v) for (s, t), v in ml_blocks.items() if t == tf) for tf in STEP}
print("period %s .. %s" % (START, E.NOW.strftime("%Y-%m-%d")))
print("ML-blocked candidate-bars: 15m %d, 1h %d" % (n_ml["15m"], n_ml["1h"]))
print("pass rate past the ML gate -> entry: 15m %.1f%%, 1h %.1f%%" % (100 * p["15m"], 100 * p["1h"]))


def in_position(sym, d):
    return any(a <= d <= b for a, b in windows.get(sym, ()))


# ---------------- goal ----------------
wl = E.load_watchlist()
full, _, _ = E.load_uptime(datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=timezone.utc))
win, _ = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
dl = LS.intraday_deadlines()
W = sorted(k for k in win if START <= k[0] and k[0] in full and k in dl and dl[k][1] is not None)


def goal(days_filter, tfs=("15m", "1h")):
    Wd = [k for k in W if days_filter(k[0])]
    now = reach = 0
    expct = 0.0
    lead = []
    for day, sym in Wd:
        op, dd = dl[(day, sym)]
        if any(op <= x[0] < dd for x in entries.get(sym, ())):
            now += 1
            continue
        tries = []
        for tf in tfs:
            for b in ml_blocks.get((sym, tf), {}):
                t = b + STEP[tf]
                if op <= t < dd and not in_position(sym, t):
                    tries.append((t, tf))
        if tries:
            reach += 1
            miss = 1.0
            for t, tf in tries:
                miss *= (1 - p[tf])
            expct += 1 - miss
            lead.append((dd - min(tries)[0]).total_seconds() / 3600)
    return len(Wd), now, expct, reach, lead


print("\n=== GOAL: immutable top-20 winner-days entered before the +2.5% crossing ===")
for name, f in (("all", lambda d: True), ("old model (< 09-07)", lambda d: d < MODEL_CHANGE[:10]),
                ("peak-label model (>= 09-08)", lambda d: d >= "2026-09-08")):
    n, now, ex, rc, lead = goal(f)
    if not n:
        continue
    print("  %-28s days %3d | now %5.1f%% -> ML off expected %5.1f%% (upper %5.1f%%, +%d reachable)%s" % (
        name, n, 100 * now / n, 100 * (now + ex) / n, 100 * (now + rc) / n, rc,
        ("  first ML-blocked try %.1f h before the crossing (median)" % sorted(lead)[len(lead) // 2]) if lead else ""))
for tfs in (("15m",), ("1h",)):
    for name, f in (("all", lambda d: True), ("peak-label model (>= 09-08)", lambda d: d >= "2026-09-08")):
        n, now, ex, rc, _ = goal(f, tfs)
        print("  ML off on %-3s only, %-28s now %5.1f%% -> expected %5.1f%% (upper %5.1f%%, +%d)" % (
            tfs[0], name, 100 * now / n, 100 * (now + ex) / n, 100 * (now + rc) / n, rc))
print("  by month:")
for m in sorted({k[0][:7] for k in W}):
    n, now, ex, rc, _ = goal(lambda d, m=m: d.startswith(m))
    print("    %s  days %3d  now %5.1f%%  ML off %5.1f%%  +%d" % (m, n, 100 * now / n, 100 * (now + ex) / n, rc))

# ---------------- per trade ----------------
rnd = random.Random(21)


def diff_ci(cur, add, w):
    def comb(a, bb):
        return (sum(a) + w * sum(bb)) / (len(a) + w * len(bb)) - sum(a) / len(a)
    pt = comb(cur, add)
    bs = sorted(comb(rnd.choices(cur, k=len(cur)), rnd.choices(add, k=len(add))) for _ in range(1000))
    return pt, bs[25], bs[-26]


print("\n=== PER TRADE, calibrated trail for both arms (the 1h engine is the same rule, not separately calibrated) ===")
for tf in ("15m", "1h"):
    cur, add, addp = [], [], []
    for sym in sorted({s for (s, t) in ml_blocks if t == tf} | set(entries)):
        b = TD.load_bars(sym, tf)
        if not b:
            continue
        h = np.array([x[2] for x in b]); l = np.array([x[3] for x in b]); c = np.array([x[4] for x in b])
        atr = I._atr(h, l, c, cfg.ATR_PERIOD)
        idx = {x[0]: i for i, x in enumerate(b)}
        mb = 96 if tf == "15m" else 24
        for d, t, mode, k in entries.get(sym, ()):
            if t != tf:
                continue
            i = idx.get(bar_of(d, tf) - STEP[tf])
            r = live_trail(c, atr, i, float(k) if isinstance(k, (int, float)) else 2.0, floor_pct(mode), mb) if i is not None else None
            if r:
                cur.append((d.strftime("%Y-%m-%d"), r[1]))
        busy = None
        for bk, mlp in sorted(ml_blocks.get((sym, tf), {}).items()):
            if (busy is not None and bk <= busy) or in_position(sym, bk + STEP[tf]):
                continue
            i = idx.get(bk - STEP[tf]) if tf == "15m" else idx.get(bk)
            if i is None:
                continue
            r = live_trail(c, atr, i, 2.0, 0.0, mb)
            if not r:
                continue
            add.append((bk.strftime("%Y-%m-%d"), r[1]))
            addp.append((mlp, r[1]))
            busy = b[r[0]][0] + STEP[tf] * COOLDOWN
    for name, f in (("all", lambda d: True), ("peak-label model (>= 09-08)", lambda d: d >= "2026-09-08")):
        cc = [x[1] for x in cur if f(x[0])]
        aa = [x[1] for x in add if f(x[0])]
        if len(cc) < 20 or len(aa) < 20:
            print("  %s %-28s too few (current %d, added %d)" % (tf, name, len(cc), len(aa)))
            continue
        print("  %s %-28s current n=%5d mean %+.3f%% | ML-blocked n=%5d mean %+.3f%% median %+.3f%%" % (
            tf, name, len(cc), np.mean(cc), len(aa), np.mean(aa), np.median(aa)))
        for w, wn in ((p[tf], "expected (x p)"), (1.0, "upper (all)")):
            pt, lo, hi = diff_ci(cc, aa, w)
            print("      combined - current, %-15s %+.3f pp [%+.3f, %+.3f] -> %s" % (
                wn, pt, lo, hi, "NON-INFERIOR" if lo >= -0.10 else "FAILS -0.10 pp"))
    if addp:
        q = [x for x in addp if isinstance(x[0], (int, float))]
        if q:
            qs = np.quantile([x[0] for x in q], [0.25, 0.5, 0.75])
            print("      ML-blocked trades by ml_proba quartile: " + "  ".join(
                "q%d mean %+.2f%%" % (k + 1, np.mean([x[1] for x in q if (k == 0 or x[0] > qs[k - 1]) and (k == 3 or x[0] <= qs[k])]))
                for k in range(4)))
