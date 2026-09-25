"""P0 follow-up: is FAST_LOSS_EMA_EXIT (first close below EMA20 in a losing trade) premature?

The calibrated exit engine (_p0_exit_calibration.py) says the plain live trail
beats this exit by +0.84%/trade. But "plain trail" is not what happens if the
rule is switched off: the position then meets the remaining exit rules, and the
next one ("2 closes below EMA20") usually fires one bar later. So two
counterfactuals are graded against the real exit:

  A  plain live trail only                            (upper bound)
  B  live trail + every non-WEAK strategy.check_exit_conditions reason from the
     next bar on (2 closes < EMA20, price < EMA20 + weakness, EMA20 turning
     down, ADX fading, RSI overbought)                (the rule OFF, rest ON)

B is the one a flag flip would ship. Split by month (stability), by mode and by
whether the exit day was an immutable top-20 winner-day for that coin.
"""
import collections
import io
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config as cfg  # noqa: E402
import immutable_labels as IL  # noqa: E402
import indicators as I  # noqa: E402
import strategy as ST  # noqa: E402
import _backtest_trend_start_detector as TD  # noqa: E402

FLOORS = {"impulse_speed": "TRAIL_MIN_BUFFER_PCT_IMPULSE_SPEED", "strong_trend": "TRAIL_MIN_BUFFER_PCT_STRONG_TREND",
          "impulse": "TRAIL_MIN_BUFFER_PCT_IMPULSE", "trend": "TRAIL_MIN_BUFFER_PCT_TREND",
          "alignment": "TRAIL_MIN_BUFFER_PCT_ALIGNMENT", "retest": "TRAIL_MIN_BUFFER_PCT_RETEST",
          "breakout": "TRAIL_MIN_BUFFER_PCT_BREAKOUT"}
KEY = "первое закрытие ниже EMA"
MAX_BARS = 96


def buffer(price, k, atr, mode):
    f = float(getattr(cfg, FLOORS.get(mode, "TRAIL_MIN_BUFFER_PCT_DEFAULT"), 0.0)) if getattr(cfg, "TRAIL_MIN_BUFFER_PCT_ENABLED", False) else 0.0
    return max(k * atr if atr > 0 else 0.0, f * price)


def run(c, atr, feat, i_entry, i_exit, entry_price, k, mode, with_rules):
    a0 = atr[i_entry] if np.isfinite(atr[i_entry]) else 0.0
    stop = entry_price - buffer(entry_price, k, a0, mode)
    last = min(len(c) - 1, i_exit + MAX_BARS)
    if last <= i_exit:
        return None
    for j in range(i_entry + 1, last + 1):
        aj = atr[j] if np.isfinite(atr[j]) else 0.0
        if aj > 0:
            stop = max(stop, c[j] - buffer(c[j], k, aj, mode))
        if j <= i_exit:
            continue
        if c[j] < stop:
            return j, (c[j] / entry_price - 1) * 100, "trail"
        if with_rules:
            r = ST.check_exit_conditions(feat, j, c, mode=mode, bars_elapsed=j - i_entry, tf="15m")
            if r and not r.startswith("⚠️ WEAK:"):
                return j, (c[j] / entry_price - 1) * 100, r.split(" (")[0][:24]
    return last, (c[last] / entry_price - 1) * 100, "cap96"


wl = json.load(io.open(FILES / "watchlist.json", encoding="utf-8"))
wl = wl if isinstance(wl, list) else wl.get("symbols", wl)
win, _ = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
win = set(win)

raw_exits = collections.defaultdict(list)
with io.open(FILES / "bot_events.jsonl", "rb") as fh:
    for raw in fh:
        if KEY.encode("utf-8") not in raw:
            continue
        try:
            e = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            continue
        if e.get("event") == "exit" and e.get("tf") == "15m" and KEY in str(e.get("reason")):
            raw_exits[e.get("sym")].append(e)

rows = []
for sym, evs in raw_exits.items():
    b = TD.bars_15m(sym)
    if not b:
        continue
    o = np.array([x[1] for x in b]); h = np.array([x[2] for x in b]); l = np.array([x[3] for x in b])
    c = np.array([x[4] for x in b]); v = np.array([x[5] for x in b])
    feat = I.compute_features(o, h, l, c, v)
    atr = I._atr(h, l, c, cfg.ATR_PERIOD)
    idx = {x[0]: i for i, x in enumerate(b)}
    for e in evs:
        if not all(isinstance(e.get(k), (int, float)) for k in ("entry_price", "pnl_pct", "bars_held")):
            continue
        d = datetime.fromisoformat(str(e["ts"]).replace("Z", "+00:00"))
        i = idx.get(d.replace(minute=d.minute // 15 * 15, second=0, microsecond=0))
        if i is None:
            continue
        ix = i - 1
        ie = ix - int(e["bars_held"])
        if ie < 60:
            continue
        k = float(e.get("trail_k") or 2.0)
        mode = e.get("mode") or "?"
        a = run(c, atr, feat, ie, ix, float(e["entry_price"]), k, mode, False)
        bb = run(c, atr, feat, ie, ix, float(e["entry_price"]), k, mode, True)
        if a is None or bb is None:
            continue
        rows.append({"pnl": float(e["pnl_pct"]), "A": a[1], "B": bb[1], "Bwhy": bb[2], "Bbars": bb[0] - ix,
                     "m": d.strftime("%Y-%m"), "mode": mode, "win": (d.strftime("%Y-%m-%d"), sym) in win})

rnd = random.Random(7)


def ci(d):
    bs = sorted(sum(rnd.choice(d) for _ in d) / len(d) for _ in range(2000))
    return bs[49], bs[-50]


def line(name, v):
    if len(v) < 15:
        if v:
            print("  %-26s n=%4d  (too few)" % (name, len(v)))
        return
    out = []
    for key in ("A", "B"):
        d = [x[key] - x["pnl"] for x in v]
        lo, hi = ci(d)
        out.append("%s-actual %+5.2f%% [%+.2f,%+.2f]%s" % (key, sum(d) / len(d), lo, hi, "*" if lo > 0 or hi < 0 else " "))
    print("  %-26s n=%4d  actual %+5.2f%%   %s" % (name, len(v), sum(x["pnl"] for x in v) / len(v), "   ".join(out)))


print("FAST_LOSS_EMA_EXIT exits on 15m: %d graded (%s .. %s)" % (
    len(rows), min(x["m"] for x in rows), max(x["m"] for x in rows)))
line("ALL", rows)
print("\n by month:")
for m in sorted({x["m"] for x in rows}):
    line(m, [x for x in rows if x["m"] == m])
print("\n by mode:")
for m in sorted({x["mode"] for x in rows}):
    line(m, [x for x in rows if x["mode"] == m])
print("\n by winner-day (immutable top-20 that day):")
line("winner-day", [x for x in rows if x["win"]])
line("other day", [x for x in rows if not x["win"]])
print("\n B: which rule would have closed it instead, and how many bars later")
cnt = collections.Counter(x["Bwhy"] for x in rows)
for k2, n in cnt.most_common(8):
    sub = [x for x in rows if x["Bwhy"] == k2]
    bars = sorted(x["Bbars"] for x in sub)
    print("  %-26s n=%4d  median +%d bars  B-actual %+5.2f%%" % (k2, n, bars[len(bars) // 2], sum(x["B"] - x["pnl"] for x in sub) / n))
d = sorted(x["B"] - x["pnl"] for x in rows)
print("\n B-actual distribution: p5 %+.2f  p25 %+.2f  median %+.2f  p75 %+.2f  p95 %+.2f   B better in %.0f%%" % (
    d[len(d) // 20], d[len(d) // 4], d[len(d) // 2], d[3 * len(d) // 4], d[19 * len(d) // 20], 100 * sum(x > 0 for x in d) / len(d)))
worst = sorted(x["B"] for x in rows)
print(" B tail: worst 5%% of counterfactual trades average %+.2f%% (actual on the same rows %+.2f%%)" % (
    sum(worst[: max(1, len(worst) // 20)]) / max(1, len(worst) // 20),
    sum(sorted(rows, key=lambda x: x["B"])[i]["pnl"] for i in range(max(1, len(worst) // 20))) / max(1, len(worst) // 20)))
