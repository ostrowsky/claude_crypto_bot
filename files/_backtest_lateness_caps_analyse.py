"""Part 1 of the read-out for _backtest_lateness_caps.py: outcome by band, the
same-daily_range view, by month, and the immutable winner-day coverage.
Part 2 (downside and trailed outcome) is _backtest_lateness_caps_downside.py.
Verdict and numbers: see _backtest_lateness_caps.py."""
import collections
import io
import json
import random
import sys
from pathlib import Path

FILES = Path("D:/Projects/claude_crypto_bot/files")
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import immutable_labels as IL  # noqa: E402

ROWS = FILES.parent / ".runtime" / "backtests" / "lateness_rows.jsonl"   # written by _backtest_lateness_caps.py
rows = [json.loads(l) for l in io.open(ROWS, encoding="utf-8")]
days_all = sorted({r["day"] for r in rows})
print("rows %d, symbols %d, days %s .. %s" % (len(rows), len({r["sym"] for r in rows}), days_all[0], days_all[-1]))


def stat(v4, vd):
    v4 = sorted(x for x in v4 if x is not None)
    n = len(v4)
    if not n:
        return "n=0"
    vd = sorted(vd)
    return ("n=%6d  4h: median %5.2f%% mean %5.2f%% >=3%% %4.1f%% >=5%% %4.1f%% | rest of day: median %5.2f%% "
            ">=5%% %4.1f%% >=10%% %4.1f%%" % (
                n, v4[n // 2], sum(v4) / n, 100 * sum(x >= 3 for x in v4) / n, 100 * sum(x >= 5 for x in v4) / n,
                vd[len(vd) // 2], 100 * sum(x >= 5 for x in vd) / len(vd), 100 * sum(x >= 10 for x in vd) / len(vd)))


by = collections.defaultdict(list)
for r in rows:
    by[r["band"]].append(r)
print("\n=== 1. OUTCOME BY BAND (first bar per symbol-hour) ===")
for b in ("FIRE", "BULLBAND", "LATE"):
    v = by[b]
    print("%-9s %s" % (b, stat([r["peak_4h"] for r in v], [r["peak_day"] for r in v])))
print("   LATE rule mix:", dict(collections.Counter(r["rule"] for r in by["LATE"]).most_common()))

print("\n=== 2. SAME daily_range BUCKET: is it lateness, or the rows? (mean 4h peak, n) ===")
buckets = [(0, 5), (5, 7), (7, 10), (10, 15), (15, 25), (25, 1e9)]
print("%-10s %22s %22s %22s" % ("dr bucket", "FIRE", "BULLBAND", "LATE"))
for lo, hi in buckets:
    cells = []
    for b in ("FIRE", "BULLBAND", "LATE"):
        v = [r["peak_4h"] for r in by[b] if r["dr"] is not None and lo <= r["dr"] < hi and r["peak_4h"] is not None]
        cells.append(("%5.2f%%  [%6d]" % (sum(v) / len(v), len(v))) if len(v) >= 30 else "          --")
    print("%-10s %22s %22s %22s" % ("%g-%s" % (lo, "" if hi > 1e8 else "%g" % hi), *cells))

print("\n=== 3. BY MONTH: LATE vs FIRE mean 4h peak ===")
bm = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    if r["peak_4h"] is not None:
        bm[r["day"][:7]][r["band"]].append(r["peak_4h"])
wins = tot = 0
for m in sorted(bm):
    f, l = bm[m]["FIRE"], bm[m]["LATE"]
    if len(f) >= 30 and len(l) >= 30:
        tot += 1
        wins += (sum(l) / len(l) >= sum(f) / len(f))
        print("  %s  FIRE %5.2f%% [%5d]   LATE %5.2f%% [%5d]   %s" % (
            m, sum(f) / len(f), len(f), sum(l) / len(l), len(l), "LATE>=FIRE" if sum(l) / len(l) >= sum(f) / len(f) else ""))
print("  months where LATE >= FIRE: %d of %d" % (wins, tot))

print("\n=== 4. THE GOAL: immutable top-20 winner-days (watchlist INTERSECT global top-20) ===")
wl = json.load(io.open(FILES / "watchlist.json", encoding="utf-8"))
wl = wl if isinstance(wl, list) else wl.get("symbols", wl)
win, eod = IL.winners_by_day(top_n=20, watchlist=set(wl), rank_before_filter=True)
lo_day, hi_day = days_all[0], days_all[-1]
win = {k for k in win if lo_day <= k[0] <= hi_day}
syms_replayed = {r["sym"] for r in rows}
win = {k for k in win if k[1] in syms_replayed}
first = {}
for r in sorted(rows, key=lambda r: r["ts"]):
    k = (r["day"], r["sym"], r["band"])
    first.setdefault(k, r)
n = len(win)
f_only = sum(1 for d, s in win if (d, s, "FIRE") in first)
fb = sum(1 for d, s in win if (d, s, "FIRE") in first or (d, s, "BULLBAND") in first)
fbl = sum(1 for d, s in win if any((d, s, b) in first for b in ("FIRE", "BULLBAND", "LATE")))
late_only = [(d, s) for d, s in win if (d, s, "LATE") in first
             and (d, s, "FIRE") not in first and (d, s, "BULLBAND") not in first]
print("winner-days in the replayed window: %d" % n)
print("  reached by a rule under today's caps (FIRE)      : %4d  (%.1f%%)" % (f_only, 100.0 * f_only / n))
print("  ... plus the bull-day band                        : %4d  (%.1f%%)" % (fb, 100.0 * fb / n))
print("  ... plus LATE (caps lifted)                       : %4d  (%.1f%%)" % (fbl, 100.0 * fbl / n))
print("  winner-days reached ONLY by LATE                  : %4d" % len(late_only))
if late_only:
    rem = sorted(first[(d, s, "LATE")]["peak_day"] for d, s in late_only)
    done = sorted(first[(d, s, "LATE")]["dr"] or 0 for d, s in late_only)
    print("    at that first LATE bar: move already done (dr from day low) median %.1f%%, move LEFT in the day"
          " median %.1f%%, >=5%% left on %.0f%%, >=10%% left on %.0f%%" % (
              done[len(done) // 2], rem[len(rem) // 2], 100 * sum(x >= 5 for x in rem) / len(rem),
              100 * sum(x >= 10 for x in rem) / len(rem)))
# is LATE richer in winners than FIRE? base rate = share of all coin-days that are winners
coin_days = {(r["day"], r["sym"]) for r in rows}
base = len(win & coin_days) / max(1, len(coin_days))
for b in ("FIRE", "BULLBAND", "LATE"):
    cd = {(r["day"], r["sym"]) for r in by[b]}
    p = len(cd & win) / max(1, len(cd))
    print("  %-9s coin-days %6d, share that are winner-days %5.1f%%   lift vs base %.2fx" % (b, len(cd), 100 * p, p / base if base else 0))
print("  base: %.1f%% of coin-days with any rule activity are winner-days" % (100 * base))

rnd = random.Random(5)
f4 = [r["peak_4h"] for r in by["FIRE"] if r["peak_4h"] is not None]
l4 = [r["peak_4h"] for r in by["LATE"] if r["peak_4h"] is not None]
if f4 and l4:
    bs = sorted(sum(rnd.choice(l4) for _ in range(2000)) / 2000 - sum(rnd.choice(f4) for _ in range(2000)) / 2000
                for _ in range(1000))
    print("\nLATE minus FIRE mean 4h peak, bootstrap 95%%: [%+.2f, %+.2f]%%" % (bs[25], bs[975]))
