"""Winner-days where no entry rule fired between the UTC open and the +2.5%
crossing: which condition of each rule was binding, bar by bar?

Features are computed on a slice (400 bars of history + the window) of the
repaired long 15m store; the five live rules are evaluated on every closed bar
in the window, and each rule's FIRST failing reason is categorised.
"""
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
import _compute_early_capture as E  # noqa: E402
import immutable_labels as IL  # noqa: E402
import label_store as LS  # noqa: E402
import _backtest_trend_start_detector as TD  # noqa: E402
import indicators as I  # noqa: E402
import strategy as S  # noqa: E402
import config  # noqa: E402

config._effective_range_max = config.DAILY_RANGE_MAX
config._bull_day_active = False

cut = E.NOW - timedelta(days=60)
wl = E.load_watchlist()
full, _, _ = E.load_uptime(cut)
win, _ = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
dl = LS.intraday_deadlines()
W = sorted(k for k in win if k[0] >= cut.strftime("%Y-%m-%d") and k[0] in full and k in dl)

fires = collections.defaultdict(list)
for l in io.open(FILES.parent / ".runtime/backtests/lateness_rows.jsonl", encoding="utf-8"):
    r = json.loads(l)
    fires[(r["day"], r["sym"])].append(datetime.fromisoformat(r["ts"]) + timedelta(minutes=15))

CATS = (("режим", "regime: new BUY forbidden (bear)"), ("bear", "regime: new BUY forbidden (bear)"),
        ("не выше EMA", "price/EMA20/EMA50 structure"), ("структура", "price/EMA20/EMA50 structure"),
        ("close", "price/EMA20/EMA50 structure"), ("EMA sep", "EMA20~EMA50 (flat)"),
        ("ADX", "ADX low / not rising"), ("наклон", "EMA20 slope too low"), ("slope", "EMA20 slope too low"),
        ("объём", "volume ratio too low"), ("vol", "volume ratio too low"),
        ("RSI", "RSI outside zone"), ("MACD", "MACD histogram"), ("r1", "1-bar impulse too small"),
        ("r3", "3-bar impulse too small"), ("daily_range", "daily_range cap"), ("от дна", "daily_range cap"),
        ("пробоя", "no EMA20 cross with volume"), ("G1", "alignment quality gate"), ("G4", "alignment quality gate"),
        ("price_edge", "price edge vs EMA20"), ("edge", "price edge vs EMA20"))


def cat(reason):
    for k, name in CATS:
        if k in reason:
            return name
    return re.sub(r"[\d.,+\-%×]+", "#", reason)[:40]


RULES = (("trend", lambda f, i, c: S.check_entry_conditions(f, i, c, tf="15m")),
         ("impulse_speed", lambda f, i, c: S.check_trend_surge_conditions(f, i)),
         ("impulse", lambda f, i, c: S.check_impulse_conditions(f, i)),
         ("alignment", lambda f, i, c: S.check_alignment_conditions(f, i, tf="15m")),
         ("ema_cross", lambda f, i, c: S.check_ema_cross_conditions(f, i)))

per_rule = {n: collections.Counter() for n, _ in RULES}
per_day_binding = collections.Counter()
n_days = n_bars = 0
cache = {}
for day, sym in W:
    o, T = dl[(day, sym)]
    if any(o <= t <= T for t in fires.get((day, sym), [])):
        continue
    if sym not in cache:
        cache[sym] = TD.bars_15m(sym)
    bars = cache[sym]
    idx = [i for i, b in enumerate(bars) if o <= b[0] + timedelta(minutes=15) <= T]
    if not idx:
        continue
    lo = max(0, idx[0] - 400)
    sl = bars[lo:idx[-1] + 1]
    a = lambda k: np.array([b[k] for b in sl])
    feat = I.compute_features(a(1), a(2), a(3), a(4), a(5))
    c = a(4)
    n_days += 1
    day_cats = collections.Counter()
    for i in [j - lo for j in idx]:
        n_bars += 1
        for name, fn in RULES:
            ok, reason = fn(feat, i, c)
            k = "PASS" if ok else cat(reason)
            per_rule[name][k] += 1
            if name in ("trend", "alignment"):
                day_cats[k] += 1
    if day_cats:
        per_day_binding[day_cats.most_common(1)[0][0]] += 1

print("winner-days with no rule firing in the birth window: %d  (%d closed 15m bars)" % (n_days, n_bars))
for name, cnt in per_rule.items():
    tot = sum(cnt.values())
    print("\n  %-14s" % name + "  ".join("%s %.0f%%" % (k, 100.0 * v / tot) for k, v in cnt.most_common(5)))
print("\nmost frequent binding condition per day (trend + alignment bars):")
for k, v in per_day_binding.most_common(8):
    print("  %-40s %3d days" % (k, v))
