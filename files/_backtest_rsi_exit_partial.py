"""RSI-overbought exits: full trail and PARTIAL exit instead of the RSI rule?

Same method, same code as _backtest_weak_exit_above_breakeven.py (stop tested
against the low BEFORE the peak absorbs the bar's high, peak seeded from entry).
The population is exits whose reason is "RSI перекуплен (x)" -- the hard rule
_h5_should_suppress never suppresses (`"rsi" in r`). QNT 2026-09-24: entered
73.32, left at 77.00 (+5.02%, RSI 87.1), the coin went on to 91.40 that day and
104.68 the next.

Also reports the forward PEAK after the exit (an upper bound, not an outcome --
nobody sells the high), and both against a control: RSI-overbought exits that
happened while the trade was small (< 2%), where the rule is not "taking a big
win" but exiting anything.

VERDICT 2026-09-25: NOT SUPPORTED. Neither a full switch to the trail nor a
partial exit is justified. Do not re-test without new evidence.

103 RSI-overbought exits with pnl >= 2%, 2026-03-03 .. 2026-09-25:

    f (kept on trail)   mean     median   p10     worst    mean delta   95% CI
    0.00 (today)        8.00%    5.85%    2.88%   2.12%       --
    0.25                8.15%    6.63%    2.66%   0.88%    +0.153%   [-0.19, +0.56]
    0.50                8.31%    6.94%    2.41%  -0.66%    +0.307%   [-0.38, +1.09]
    1.00 (full trail)   8.61%    6.73%    1.48%  -3.72%    +0.614%   [-0.76, +2.22]

The delta is exactly f x (trail - actual): partial exit changes the size of the
bet, never its sign. The trail beats the RSI exit on 35% of trades and loses on
65%; a few very large wins (best +53%) carry the mean, and months split 3 of 7
positive at f=0.5 (June -0.98%, September +1.73%). This bot is an alert system;
a partial exit would be a new message type, and the evidence does not pay for it.

"""
import collections
import random
import re
import sys
from pathlib import Path

FILES = Path("D:/Projects/claude_crypto_bot/files")
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import _backtest_weak_exit_above_breakeven as W  # noqa: E402
import config as cfg  # noqa: E402

MAX_BARS = 96
rows = W.load_exits("2026-01-01")
rsi = [r for r in rows if "RSI перекуплен" in str(r.get("reason") or "")]
print("exits %d ; RSI-overbought exits %d (%.1f%%)  window %s .. %s" % (
    len(rows), len(rsi), 100.0 * len(rsi) / len(rows), rows[0]["_dt"].date(), rows[-1]["_dt"].date()))
by_mode = collections.Counter(str(r.get("mode")) for r in rsi)
print("by mode:", dict(by_mode.most_common()))


def build(group):
    out, miss = [], 0
    for e in group:
        tf = str(e.get("tf") or "1h")
        bars, idx = W.index_for(e["sym"], tf)
        i = idx.get(W.bar_key(e["_dt"], tf)) if bars else None
        if i is None:
            miss += 1
            continue
        cf = W.replay(bars, i, i - int(e["bars_held"]), float(e["entry_price"]),
                      float(e.get("trail_k") or 2.0), e.get("mode"), cfg, MAX_BARS)
        # forward peak after the exit (upper bound)
        fut = bars[i + 1: i + 1 + MAX_BARS]
        fp = ((max(b[2] for b in fut) / float(e["entry_price"]) - 1.0) * 100.0) if len(fut) >= 8 else None
        if cf is None:
            miss += 1
            continue
        out.append((float(e["pnl_pct"]), cf, fp, e))
    return out, miss


big = [r for r in rsi if r["pnl_pct"] >= 2.0]
small = [r for r in rsi if r["pnl_pct"] < 2.0]
rb, mb = build(big)
rs, ms = build(small)
print("resolved: %d of %d (pnl >= 2%%), %d of %d (pnl < 2%%)" % (len(rb), len(big), len(rs), len(small)))


def med(v):
    v = sorted(v)
    return v[len(v) // 2]


def show(name, res):
    if not res:
        print("%-26s n=0" % name)
        return
    a = [x[0] for x in res]
    c = [x[1] for x in res]
    d = [x[1] - x[0] for x in res]
    fp = [x[2] for x in res if x[2] is not None]
    better = sum(1 for x in d if x > 0)
    print("%-26s n=%3d  actual med %6.2f%%  trail med %6.2f%%  delta med %+6.2f%%  trail better %3.0f%%  "
          "delta avg %+6.2f%%  | fwd-peak med %6.2f%%" % (
              name, len(res), med(a), med(c), med(d), 100.0 * better / len(res), sum(d) / len(d),
              med(fp) if fp else float("nan")))



import random as _r
def pct(v, q):
    v = sorted(v); return v[min(len(v) - 1, max(0, int(q * (len(v) - 1))))]

print()
print("PARTIAL EXIT on RSI-overbought exits with pnl >= 2%%  (n=%d, max period)" % len(rb))
print("f = share kept on the ATR trail; the rest is closed at the RSI signal as today.")
print("%-10s %9s %9s %9s %9s %8s %22s %12s" % ("f", "mean", "median", "p10", "worst", "better", "mean delta vs today", "95% CI"))
print("-" * 104)
rnd = _r.Random(11)
for f in (0.0, 0.25, 0.5, 0.75, 1.0):
    p = [(1 - f) * a + f * c for a, c, _, _ in rb]
    d = [x - a for x, (a, _, _, _) in zip(p, rb)]
    bs = sorted(sum(rnd.choice(d) for _ in d) / len(d) for _ in range(4000))
    print("%-10s %8.2f%% %8.2f%% %8.2f%% %8.2f%% %7.0f%% %+21.3f%% [%+.2f, %+.2f]" % (
        "%.2f%s" % (f, "  (today)" if f == 0 else ("  (full trail)" if f == 1 else "")),
        sum(p) / len(p), med(p), pct(p, 0.10), min(p),
        100.0 * sum(1 for x in d if x > 1e-9) / len(d), sum(d) / len(d), bs[100], bs[3899]))
print("-" * 104)
print("delta is exactly f x (trail - actual), so the SIGN of the verdict cannot change with f;")
print("only the size of the bet and the downside do.")

print()
print("by month, f=0.5 mean delta vs today:")
bm = collections.defaultdict(list)
for a, c, _, e in rb:
    bm[e["_dt"].strftime("%Y-%m")].append(0.5 * (c - a))
for m in sorted(bm):
    v = bm[m]; print("  %s  n=%3d  %+6.2f%%" % (m, len(v), sum(v) / len(v)))
pos = sum(1 for m in bm if sum(bm[m]) > 0)
print("months positive: %d of %d" % (pos, len(bm)))
