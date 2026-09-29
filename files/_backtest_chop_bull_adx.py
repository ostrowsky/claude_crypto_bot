"""L2 hypothesis 2026-09-29: TREND_1H_CHOP_ADX_MIN_BULL_DAY 22 -> 20, on the maximum period.

L3 marked it needs_review: 44 rows, one month, 4h peak 3.10% vs 1.99% (1.56x).
This judges it by the project's criterion (as P-3, p0-validation-0925-spec.md):

  who flips  1h trend candidates blocked by the chop filter on a bull day with
             20 <= ADX < 22 while slope >= TREND_1H_CHOP_SLOPE_MIN_BULL_DAY and
             vol_x >= TREND_1H_CHOP_VOL_MIN_BULL_DAY -- i.e. ADX is the only
             failing check under the proposed floor
  p          the logged rate at which 1h candidates that got past the chop filter
             (later gates or entry) became entries
  goal       immutable top-20 winner-days entered before the +2.5% crossing
  guard      per-trade non-inferiority on the calibrated trail (1h bars; same
             rule as P-1, not separately calibrated for 1h) vs current 1h trend
             entries, lower 95% bound >= -0.10 pp; plus month by month
Rows are deduplicated per coin-hour; added trades never overlap a real position.
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

START = "2026-05-01"
H = timedelta(hours=1)
FROM, TO = float(cfg.TREND_1H_CHOP_ADX_MIN_BULL_DAY), 20.0
SLOPE_B = float(cfg.TREND_1H_CHOP_SLOPE_MIN_BULL_DAY)
VOL_B = float(cfg.TREND_1H_CHOP_VOL_MIN_BULL_DAY)
COOLDOWN = int(getattr(cfg, "COOLDOWN_BARS", 8))
POST_CHOP = {"mode_range_quality", "ranker_hard_veto", "clone_guard", "open_cluster_cap",
             "correlation_guard", "late_impulse_rotation", "bandit_skip"}


def live_trail(c, atr, ie, k, max_bars=48):
    ep = c[ie]
    a0 = atr[ie] if np.isfinite(atr[ie]) else 0.0
    stop = ep - k * a0
    last = min(len(c) - 1, ie + max_bars)
    if last <= ie:
        return None
    for j in range(ie + 1, last + 1):
        aj = atr[j] if np.isfinite(atr[j]) else 0.0
        if aj > 0:
            stop = max(stop, c[j] - k * aj)
        if c[j] < stop:
            return j, (c[j] / ep - 1) * 100
    return last, (c[last] / ep - 1) * 100


def hour_of(d):
    return d.replace(minute=0, second=0, microsecond=0)


flip = collections.defaultdict(dict)       # sym -> {hour: row}
chop_all = 0
past = collections.defaultdict(set)
entries = collections.defaultdict(list)
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
        if ev == "entry":
            entries[sym].append((d, tf, e.get("mode") or e.get("signal_mode"), e.get("trail_k")))
            if tf == "1h":
                past[(sym, hour_of(d))].add("entry")
        elif ev == "exit" and isinstance(e.get("bars_held"), (int, float)):
            step = timedelta(minutes=15) if tf == "15m" else H
            windows[sym].append((d - step * (int(e["bars_held"]) + 1), d))
        elif ev == "blocked" and tf == "1h":
            st, rc = str(e.get("signal_type")), str(e.get("reason_code"))
            if rc in ("trend_chop", "trend_1h_chop") or st in ("trend_chop", "trend_1h_chop"):
                chop_all += 1
                adx, slope, vol = e.get("adx"), e.get("slope_pct"), e.get("vol_x")
                if not e.get("is_bull_day") or not all(isinstance(x, (int, float)) for x in (adx, slope, vol)):
                    continue
                if TO <= adx < FROM and slope >= SLOPE_B and vol >= VOL_B:
                    flip[sym].setdefault(hour_of(d), e)
            elif st in POST_CHOP:
                past[(sym, hour_of(d))].add("b")
p = sum(1 for v in past.values() if "entry" in v) / max(1, len(past))
n_flip = sum(len(v) for v in flip.values())
print("period %s .. %s | chop blocks on 1h: %d | rows the change flips (bull day, ADX %.0f..%.0f, slope>=%.1f, vol>=%.1f): %d coin-hours on %d coins" % (
    START, E.NOW.strftime("%Y-%m-%d"), chop_all, TO, FROM, SLOPE_B, VOL_B, n_flip, len(flip)))
print("p (1h candidates past the chop filter -> entry): %.1f%% of %d coin-hours" % (100 * p, len(past)))


def in_position(sym, d):
    return any(a <= d <= b for a, b in windows.get(sym, ()))


wl = E.load_watchlist()
full, _, _ = E.load_uptime(datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=timezone.utc))
win, _ = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
dl = LS.intraday_deadlines()
W = sorted(k for k in win if START <= k[0] and k[0] in full and k in dl and dl[k][1] is not None)
now = reach = 0
expct = 0.0
for day, sym in W:
    op, dd = dl[(day, sym)]
    if any(op <= x[0] < dd for x in entries.get(sym, ())):
        now += 1
        continue
    f = [h for h in flip.get(sym, {}) if op <= h + H < dd and not in_position(sym, h + H)]
    if f:
        reach += 1
        expct += 1 - (1 - p) ** len(f)
n = len(W)
print("\n=== GOAL on %d winner-days (bot up all day, crossing known) ===" % n)
print("  entered before the crossing: now %.1f%% -> expected %.1f%% (upper %.1f%%, +%d reachable days)" % (
    100 * now / n, 100 * (now + expct) / n, 100 * (now + reach) / n, reach))

ks = [float(x[3]) for v in entries.values() for x in v if x[1] == "1h" and x[2] == "trend" and isinstance(x[3], (int, float))]
K = sorted(ks)[len(ks) // 2] if ks else 2.0
cur, add = [], []
for sym in sorted(set(flip) | set(entries)):
    b = TD.load_bars(sym, "1h")
    if not b:
        continue
    h = np.array([x[2] for x in b]); lo = np.array([x[3] for x in b]); c = np.array([x[4] for x in b])
    atr = I._atr(h, lo, c, cfg.ATR_PERIOD)
    idx = {x[0]: i for i, x in enumerate(b)}
    for d, tf, mode, k in entries.get(sym, ()):
        if tf != "1h" or mode != "trend":
            continue
        i = idx.get(hour_of(d) - H)
        r = live_trail(c, atr, i, float(k) if isinstance(k, (int, float)) else K) if i is not None else None
        if r:
            cur.append((d.strftime("%Y-%m"), r[1]))
    busy = None
    for hr in sorted(flip.get(sym, {})):
        if (busy is not None and hr <= busy) or in_position(sym, hr + H):
            continue
        i = idx.get(hr - H) if idx.get(hr - H) is not None else None
        r = live_trail(c, atr, i, K) if i is not None else None
        if not r:
            continue
        add.append((hr.strftime("%Y-%m"), r[1]))
        busy = b[r[0]][0] + H * COOLDOWN
rnd = random.Random(29)
cc = [x[1] for x in cur]
aa = [x[1] for x in add]
print("\n=== PER TRADE, calibrated trail on 1h (k=%.2f) ===" % K)
print("  current 1h trend entries: n=%d mean %+.3f%% median %+.3f%%" % (len(cc), np.mean(cc), np.median(cc)))
if len(aa) >= 5:
    print("  flipped candidates:       n=%d mean %+.3f%% median %+.3f%%" % (len(aa), np.mean(aa), np.median(aa)))
    for w, name in ((p, "expected (x p)"), (1.0, "upper (all)")):
        def comb(a, bb):
            return (sum(a) + w * sum(bb)) / (len(a) + w * len(bb)) - sum(a) / len(a)
        bs = sorted(comb(rnd.choices(cc, k=len(cc)), rnd.choices(aa, k=len(aa))) for _ in range(1000))
        print("  combined - current, %-15s %+.3f pp [%+.3f, %+.3f] -> %s" % (
            name, comb(cc, aa), bs[25], bs[-26], "NON-INFERIOR" if bs[25] >= -0.10 else "FAILS -0.10 pp"))
    bym = collections.defaultdict(list)
    for m, v in add:
        bym[m].append(v)
    print("  flipped by month: " + "  ".join("%s %+.2f%%(n=%d)" % (m, np.mean(v), len(v)) for m, v in sorted(bym.items())))
else:
    print("  flipped candidates: n=%d -- too few to judge" % len(aa))
