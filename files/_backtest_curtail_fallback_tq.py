"""E-6: impulse_speed candidates downgraded to trend by the curtail fallback meet trend_quality.

WHY
When impulse_speed is regime-curtailed (impulse_speed_curtail: trailing-14d mean
realized pnl < 0), _poll_coin reclassifies an impulse_speed candidate as
"trend" (IMPULSE_SPEED_CURTAIL_FALLBACK_TO_TREND). trend_quality judges only
15m trend candidates -- so on curtailed days the fast movers it was never
tuned for (price edge 3.2% / 4.0% bull, daily_range 10/14%, RSI 72/76, the
forecast) now meet it. The audit saw QNT blocked on price edge 4.6-6.8%.

HOW
1. Curtail history: the same computation as impulse_speed_curtail.compute_and_write,
   as of each day's 00:30 UTC run, from critic_dataset.jsonl (impulse_speed
   takes and their trade_exit_pnl). Fallback-to-trend is live since 2026-06-12.
2. Fallback candidates = 15m replay rows (.runtime/backtests/lateness_rows.jsonl,
   band FIRE) whose rule is "impulse_speed" -- the surge fired and the trend
   rule did not, so live preview_mode was impulse_speed -> trend -- on
   curtailed days.
3. Joined per coin-hour with trend_quality blocks in bot_events.jsonl: those
   are the blocks E-6 would lift. By the failing check.
4. Criterion as P-3: goal (winner-days entered before the +2.5% crossing) and
   per-trade non-inferiority on the calibrated engine, trend trail (the
   fallback enters as trend); later gates passed with p measured on the log.
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
START = "2026-06-12"          # fallback-to-trend live
WIN = int(getattr(cfg, "IMPULSE_SPEED_CURTAIL_WINDOW_DAYS", 14))
THR = float(getattr(cfg, "IMPULSE_SPEED_CURTAIL_PNL_THRESHOLD", 0.0))
MIN_N = int(getattr(cfg, "IMPULSE_SPEED_CURTAIL_MIN_TRADES", 8))
COOLDOWN = int(getattr(cfg, "COOLDOWN_BARS", 8))
POST_TQ = {"mode_range_quality", "ranker_hard_veto", "clone_guard", "open_cluster_cap",
           "correlation_guard", "late_impulse_rotation", "bandit_skip"}


def live_trail(c, atr, ie, k, fl=0.0, max_bars=96):
    """P-1 calibrated live trail, trend floor."""
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


# ---------------- 1. curtail history ----------------
takes = []
with io.open(FILES / "critic_dataset.jsonl", "rb") as fh:
    for raw in fh:
        if b"impulse_speed" not in raw:
            continue
        try:
            e = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            continue
        if e.get("signal_type") != "impulse_speed" or str((e.get("decision") or {}).get("action")) != "take":
            continue
        pnl = (e.get("labels") or {}).get("trade_exit_pnl")
        if isinstance(pnl, (int, float)):
            takes.append((str(e.get("ts_signal", ""))[:10], float(pnl)))
curt = {}
d = datetime.strptime(START, "%Y-%m-%d")
while d <= E.NOW.replace(tzinfo=None):
    day = d.strftime("%Y-%m-%d")
    lo = (d - timedelta(days=WIN)).strftime("%Y-%m-%d")
    v = [p for t, p in takes if lo <= t < day]
    curt[day] = bool(len(v) >= MIN_N and np.mean(v) < THR)
    d += timedelta(days=1)
print("curtail history %s .. %s: curtailed on %d of %d days" % (START, max(curt), sum(curt.values()), len(curt)))
by_m = collections.Counter(k[:7] for k, v in curt.items() if v)
print("  curtailed days by month:", dict(sorted(by_m.items())))

# ---------------- 2. fallback candidates in the replay ----------------
fb = collections.defaultdict(list)          # (sym, hour) -> row
for l in io.open(FILES.parent / ".runtime/backtests/lateness_rows.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["band"] != "FIRE" or r["day"] < START or r["rule"] != "impulse_speed" or not curt.get(r["day"]):
        continue
    t = datetime.fromisoformat(r["ts"])
    fb[(r["sym"], t.strftime("%Y-%m-%d %H"))].append(r)
print("fallback candidates (impulse_speed-only fires on curtailed days, first bar per coin-hour): %d" % len(fb))

# ---------------- 3. join with trend_quality blocks ----------------
tq = {}
entries = collections.defaultdict(list)
windows = collections.defaultdict(list)
after_tq = collections.defaultdict(set)
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
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        ev, sym, tf = e.get("event"), e.get("sym"), e.get("tf")
        if ev == "entry":
            entries[sym].append((dt, tf))
            if tf == "15m":
                after_tq[(sym, bar_of(dt))].add("entry")
        elif ev == "exit" and isinstance(e.get("bars_held"), (int, float)):
            step = BAR if tf == "15m" else timedelta(hours=1)
            windows[sym].append((dt - step * (int(e["bars_held"]) + 1), dt))
        elif ev == "blocked" and tf == "15m":
            if e.get("reason_code") == "trend_quality":
                k = (sym, dt.strftime("%Y-%m-%d %H"))
                if k in fb and k not in tq:
                    r = str(e.get("reason"))
                    why = ("price_edge" if "price edge" in r else "daily_range" if "daily_range" in r
                           else "RSI" if "RSI" in r
                           # forecast 0.000 is lifted by E-4 since 2026-09-25 -- kept apart
                           else "forecast0 (E-4)" if "forecast 0.000" in r else "forecast>0/alt")
                    tq[k] = (bar_of(dt), why)
            elif str(e.get("signal_type")) in POST_TQ:
                after_tq[(sym, bar_of(dt))].add("b")
p = sum(1 for v in after_tq.values() if "entry" in v) / max(1, len(after_tq))
print("fallback candidates blocked by trend_quality in the same coin-hour: %d of %d (%.0f%%)" % (
    len(tq), len(fb), 100 * len(tq) / max(1, len(fb))))
print("  by failing check:", dict(collections.Counter(v[1] for v in tq.values()).most_common()))
print("  downstream pass rate after trend_quality p = %.1f%%" % (100 * p))


def in_position(sym, dt):
    return any(a <= dt <= b for a, b in windows.get(sym, ()))


# ---------------- 4a. goal ----------------
wl = E.load_watchlist()
full, _, _ = E.load_uptime(datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=timezone.utc))
win, _ = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
dl = LS.intraday_deadlines()
W = sorted(k for k in win if START <= k[0] and k[0] in full and k in dl and dl[k][1] is not None)
now = reach = 0
expct = 0.0
by_sym_tq = collections.defaultdict(list)
for (sym, _h), (bk, why) in tq.items():
    by_sym_tq[sym].append((bk, why))
for day, sym in W:
    op, dd = dl[(day, sym)]
    if any(op <= x[0] < dd for x in entries.get(sym, ())):
        now += 1
        continue
    f = [bk for bk, _ in by_sym_tq.get(sym, ()) if op <= bk + BAR < dd and not in_position(sym, bk + BAR)]
    if f:
        reach += 1
        expct += 1 - (1 - p) ** len(f)
n = len(W)
print("\n=== GOAL on %d winner-days since %s ===" % (n, START))
print("  entered before the crossing: now %.1f%% -> expected %.1f%%, upper %.1f%% (+%d reachable days)" % (
    100 * now / n, 100 * (now + expct) / n, 100 * (now + reach) / n, reach))

# ---------------- 4b. per trade ----------------
arm_cur, arm_add = [], []
for sym in sorted(set(by_sym_tq) | set(entries)):
    b = TD.bars_15m(sym)
    if not b:
        continue
    h = np.array([x[2] for x in b]); l = np.array([x[3] for x in b]); c = np.array([x[4] for x in b])
    atr = I._atr(h, l, c, cfg.ATR_PERIOD)
    idx = {x[0]: i for i, x in enumerate(b)}
    for dt, tf in entries.get(sym, ()):
        if tf != "15m":
            continue
        i = idx.get(bar_of(dt) - BAR)
        r = live_trail(c, atr, i, 2.0) if i is not None else None
        if r:
            arm_cur.append((dt.strftime("%Y-%m"), r[1]))
    busy = None
    for bk, why in sorted(by_sym_tq.get(sym, ())):
        if (busy is not None and bk <= busy) or in_position(sym, bk + BAR):
            continue
        i = idx.get(bk - BAR)
        r = live_trail(c, atr, i, 2.0) if i is not None else None
        if not r:
            continue
        arm_add.append((bk.strftime("%Y-%m"), r[1], why))
        busy = b[r[0]][0] + BAR * COOLDOWN
rnd = random.Random(13)


def diff_ci(cur, add, w):
    def comb(a, bb):
        return (sum(a) + w * sum(bb)) / (len(a) + w * len(bb)) - sum(a) / len(a)
    pt = comb(cur, add)
    bs = sorted(comb([rnd.choice(cur) for _ in cur], [rnd.choice(add) for _ in add]) for _ in range(1000))
    return pt, bs[25], bs[-26]


cur = [x[1] for x in arm_cur]
add = [x[1] for x in arm_add]
print("\n=== PER TRADE, calibrated engine (trend trail, k=2.0) ===")
print("  current 15m entries: n=%d mean %+.3f%%" % (len(cur), np.mean(cur)))
if add:
    print("  lifted fallback blocks: n=%d mean %+.3f%% median %+.3f%%" % (len(add), np.mean(add), np.median(add)))
    for why in sorted({x[2] for x in arm_add}):
        v = [x[1] for x in arm_add if x[2] == why]
        print("     %-12s n=%4d mean %+.3f%%" % (why, len(v), np.mean(v)))
    if len(add) >= 20:
        for w, name in ((p, "expected (x p)"), (1.0, "upper bound (all)")):
            pt, lo, hi = diff_ci(cur, add, w)
            print("  combined - current, %-18s %+.3f pp  95%% CI [%+.3f, %+.3f]  -> %s" % (
                name, pt, lo, hi, "NON-INFERIOR" if lo >= -0.10 else "FAILS the -0.10 pp bound"))
    for mo in sorted({x[0] for x in arm_add}):
        v = [x[1] for x in arm_add if x[0] == mo]
        print("    %s  n=%4d  mean %+.3f%%" % (mo, len(v), np.mean(v)))
