"""How many catchable uptrends are there, how long, and how much is left?

The operator's target, stated in their own words: catch the START of an uptrend
and leave just before it ENDS -- and a coin may collapse by EOD while still
having had the day's largest run. That is a DIFFERENT target from everything
this project measures. `EarlyCapture@top20`, the label store's
`eod_return_pct`, the top-gainer tiers and the early-ranking shadow all score
where a coin CLOSES the day. An uptrend is a segment inside the day.

Even the store's `max_move_pct` is only a partial match: it is
`(day_high / day_open - 1)`, anchored at the open rather than at the trend's
start, so a coin that drops 10% and then runs 20% off the low is scored on the
smaller number.

This measures the population itself, before any prediction is attempted:

  * how many uptrends of a given size exist per day across the watchlist,
  * how long they last,
  * how much of the move is still ahead once the first N% has printed --
    which is the ceiling on what "catch it early" can ever be worth,
  * how often the coin that had the day's largest uptrend also finished in the
    day's EOD top-20 -- i.e. how far the current target sits from this one.

No model, no split, no threshold to tune. A base rate cannot be argued with
later if it is established first.

    pyembed\\python.exe files\\_diag_uptrend_population.py

Spec: docs/specs/features/uptrend-target-spec.md
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from datetime import timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import _backtest_continuation_signal as CS   # kline cache, merged and fresh
from zigzag_labeler import detect_uptrends


def shape_of(sym: str, start, end) -> tuple:
    """(max internal drawdown %, R^2 of a linear fit) over the trend window.

    The operator's premise is that a LINEAR uptrend -- one that climbs without
    dropping -- is easier to detect and to ride. That is a claim about the
    shape of the segment, so it needs measuring directly rather than being
    assumed from the detector's parameters: `max_drawdown_pct` bounds the
    give-back that ENDS a trend, it does not describe what happened inside it.
    """
    b = [r for r in CS.bars(sym) if start <= r[0] <= end]
    if len(b) < 3:
        return (None, None)
    closes = [r[4] for r in b]
    peak = closes[0]
    dd = 0.0
    for c in closes:
        peak = max(peak, c)
        if peak:
            dd = max(dd, (peak - c) / peak * 100.0)
    n = len(closes)
    xm = (n - 1) / 2.0
    ym = sum(closes) / n
    sxy = sum((i - xm) * (v - ym) for i, v in enumerate(closes))
    sxx = sum((i - xm) ** 2 for i in range(n)) or 1.0
    slope = sxy / sxx
    ss_tot = sum((v - ym) ** 2 for v in closes)
    ss_res = sum((v - (ym + slope * (i - xm))) ** 2 for i, v in enumerate(closes))
    r2 = (1 - ss_res / ss_tot) if ss_tot else None
    return (dd, r2)


def watchlist() -> list:
    return json.loads((HERE / "watchlist.json").read_text(encoding="utf-8"))


def trends_for(sym: str, swing: float, drawdown: float, min_bars: int) -> list:
    b = CS.bars(sym)
    if len(b) < 100:
        return []
    # zigzag_labeler expects objects/tuples with ts/high/low/close; the cache
    # rows are (ts, open, high, low, close, volume).
    rows = [{"ts": r[0], "open": r[1], "high": r[2], "low": r[3],
             "close": r[4], "volume": r[5]} for r in b]
    try:
        return detect_uptrends(rows, symbol=sym, swing_pct=swing,
                               max_drawdown_pct=drawdown,
                               min_duration_bars=min_bars)
    except Exception as exc:
        print("  [%s: %s]" % (sym, exc))
        return []


def attr(t, *names):
    for n in names:
        if isinstance(t, dict) and n in t:
            return t[n]
        if hasattr(t, n):
            return getattr(t, n)
    return None


def pct(v, q):
    v = sorted(x for x in v if x is not None)
    return v[int(q * (len(v) - 1))] if v else float("nan")


def sweep(args) -> None:
    """How much does demanding a LINEAR trend cost in count and size?

    Tightening the give-back threshold makes each detected segment more
    monotonic -- and there is no free lunch: the same tightening cuts a long
    run into several short ones and discards the rest. Printing count, size and
    shape together is the only way to see the trade instead of picking the
    setting that flatters whichever column is being watched.
    """
    wl = watchlist()
    print("=" * 92)
    print("LINEARITY SWEEP -- what demanding a clean uptrend costs")
    print("1h bars, swing >= %.1f%%, >= %d bars" % (args.swing, args.min_bars))
    print("=" * 92)
    print("%9s%9s%9s%11s%10s%10s%10s%12s" % (
        "give-back", "trends", "per day", "med gain%", "med hrs",
        "med dd%", "med R2", "left after+3"))
    print("-" * 92)
    for dd_thr in (1.0, 1.5, 2.0, 2.5, 4.0):
        trends = []
        per_day = defaultdict(int)
        for sym in wl:
            for t in trends_for(sym, args.swing, dd_thr, args.min_bars):
                start = attr(t, "start_ts", "start", "low_ts")
                end = attr(t, "end_ts", "end", "high_ts")
                gain = attr(t, "gain_pct", "gain")
                if start is None or gain is None:
                    continue
                d, r2 = (shape_of(sym, start, end)
                         if end is not None else (None, None))
                hrs = ((end - start).total_seconds() / 3600.0
                       if end is not None else None)
                trends.append({"gain": float(gain), "hrs": hrs, "dd": d, "r2": r2})
                per_day[start.strftime("%Y-%m-%d")] += 1
        if not trends:
            print("%9.1f%9s" % (dd_thr, "none"))
            continue
        left = [((1 + t["gain"] / 100) / 1.03 - 1) * 100
                for t in trends if t["gain"] > 3.0]
        print("%9.1f%9d%9.1f%11.2f%10.1f%10.2f%10.3f%12.2f" % (
            dd_thr, len(trends),
            len(trends) / max(len(per_day), 1),
            pct([t["gain"] for t in trends], 0.5),
            pct([t["hrs"] for t in trends], 0.5),
            pct([t["dd"] for t in trends], 0.5),
            pct([t["r2"] for t in trends], 0.5),
            pct(left, 0.5) if left else float("nan")))
    print()
    print("'med dd%' is the deepest drop INSIDE the trend, 'med R2' how close it")
    print("sits to a straight line. 'left after+3' is the ceiling on entering")
    print("once 3%% has already printed: no detector can capture more than that.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swing", type=float, default=5.0,
                    help="minimum uptrend size; 1h guidance in the labeler is 5-7")
    ap.add_argument("--drawdown", type=float, default=2.5)
    ap.add_argument("--min-bars", type=int, default=4)
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the give-back threshold: cleaner trends are "
                         "rarer and shorter, and that cost must be visible")
    args = ap.parse_args()

    if args.sweep:
        return sweep(args)

    print("=" * 78)
    print("UPTREND POPULATION -- the target the operator actually described")
    print("1h bars, swing >= %.1f%%, ends on %.1f%% give-back, >= %d bars"
          % (args.swing, args.drawdown, args.min_bars))
    print("=" * 78)

    wl = watchlist()
    all_trends = []
    per_day = defaultdict(int)
    covered_syms = 0
    for sym in wl:
        ts = trends_for(sym, args.swing, args.drawdown, args.min_bars)
        if CS.bars(sym):
            covered_syms += 1
        for t in ts:
            start = attr(t, "start_ts", "start", "low_ts")
            end = attr(t, "end_ts", "end", "high_ts")
            gain = attr(t, "gain_pct", "gain")
            if start is None or gain is None:
                continue
            day = start.strftime("%Y-%m-%d") if hasattr(start, "strftime") else str(start)[:10]
            dur = None
            if end is not None and hasattr(end, "__sub__") and hasattr(start, "__sub__"):
                try:
                    dur = (end - start).total_seconds() / 3600.0
                except Exception:
                    dur = None
            dd = r2 = None
            if end is not None and hasattr(start, "strftime"):
                dd, r2 = shape_of(sym, start, end)
            all_trends.append({"sym": sym, "day": day, "gain": float(gain),
                               "dur_h": dur, "dd": dd, "r2": r2})
            per_day[day] += 1

    if not all_trends:
        print("no uptrends detected -- check the kline cache")
        return

    days = sorted(per_day)
    gains = sorted(t["gain"] for t in all_trends)
    durs = sorted(t["dur_h"] for t in all_trends if t["dur_h"] is not None)

    print()
    print("symbols with klines      %d of %d" % (covered_syms, len(wl)))
    print("days covered             %d  (%s .. %s)" % (len(days), days[0], days[-1]))
    print("uptrends found           %d" % len(all_trends))
    print("uptrends per day         median %.1f   mean %.1f   max %d"
          % (st.median(list(per_day.values())),
             sum(per_day.values()) / len(per_day), max(per_day.values())))
    print()
    print("gain%%    p25 %.1f   median %.1f   p75 %.1f   p90 %.1f   max %.1f"
          % (gains[len(gains) // 4], gains[len(gains) // 2],
             gains[3 * len(gains) // 4], gains[int(0.9 * len(gains))], gains[-1]))
    if durs:
        print("hours    p25 %.1f   median %.1f   p75 %.1f   p90 %.1f   max %.1f"
              % (durs[len(durs) // 4], durs[len(durs) // 2],
                 durs[3 * len(durs) // 4], durs[int(0.9 * len(durs))], durs[-1]))

    # The ceiling on earliness. If the first N% of the move must print before
    # anything can be detected, this is what remains to be captured -- an upper
    # bound no detector can beat, and the number that decides whether the whole
    # approach is worth pursuing.
    print()
    print("%-12s%10s%12s%12s" % ("detected at", "trends", "median left",
                                 "share >= 5% left"))
    print("-" * 78)
    for trigger in (0.0, 1.0, 2.0, 3.0, 5.0):
        rem = []
        for t in all_trends:
            g = t["gain"]
            if g <= trigger:
                continue
            # what is left, in % of price, after `trigger`% has already printed
            left = ((1 + g / 100) / (1 + trigger / 100) - 1) * 100
            rem.append(left)
        if not rem:
            continue
        rem.sort()
        share = sum(1 for x in rem if x >= 5.0) / len(rem)
        print("%-12s%10d%12.2f%%%11.1f%%" % (
            "+%.0f%% in" % trigger, len(rem), rem[len(rem) // 2], share * 100))

    print()
    print("READING THIS")
    print("  'median left' is the ceiling: a perfect detector firing after the")
    print("  first N%% has printed cannot capture more than this. If it collapses")
    print("  as the trigger rises, the move is mostly over by the time anything")
    print("  is detectable, and no model fixes that -- it is a property of the")
    print("  price series, not of the predictor.")


if __name__ == "__main__":
    main()
