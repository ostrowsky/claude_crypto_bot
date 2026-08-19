"""Hold while the trend rises monotonically; sell on a plateau or a turn down.

The operator's exit rule, stated 2026-08-19 for the long trends the +20%
detector catches:

    a weekly trend is worth tracking while it stays stable -- rising
    monotonically. If it flattens out or turns down, send a sell signal.

That is a rule about SHAPE, and it is a different proposition from everything
tested before it. `_backtest_exit_timing.py` ruled out fixed trail widths and
`_backtest_continuation_exit_policy.py` ruled out a probability threshold -- but
both were measured on the bot's own entries, a population whose median trade is
already negative one hour in. Exits from trends this detector caught start with
a median 18.98% of the move still ahead. The earlier negatives do not transfer,
for the same reason they did not transfer to the entry side.

WHAT IS REPLAYED
    From the alert bar of every caught trend, forward, bar by bar:

      plateau_slope(K)  leave when the slope over the last K bars stops being
                        positive -- the operator's rule, read literally
      below_ma(N)       leave when close falls under its own N-bar mean
      no_new_high(K)    leave after K bars without a new running high
      zigzag_end        leave on a 2% give-back from the peak -- this is the
                        trend's own definition of ending, so it is the natural
                        baseline rather than an alternative
      ideal             leave exactly at the peak: not achievable, and printed
                        because a capture ratio needs its ceiling

THE MEASURE
    Capture = realised gain from the alert / the gain available from the alert
    to the trend's peak. Reported with the median bars held and the share of
    exits that landed BEFORE the peak, because a rule that leaves early looks
    good on capture only until the trend resumes without it.

    pyembed\\python.exe files\\_backtest_trend_exit_rule.py

Spec: docs/specs/features/trend-start-detector-spec.md
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import _backtest_continuation_signal as CS
import _diag_uptrend_population as UP


def slope_pct(closes: list, k: int) -> float:
    """Least-squares slope over the last k closes, in % of the last price."""
    if len(closes) < k or not closes[-1]:
        return 0.0
    y = closes[-k:]
    xm = (k - 1) / 2.0
    ym = sum(y) / k
    num = sum((i - xm) * (v - ym) for i, v in enumerate(y))
    den = sum((i - xm) ** 2 for i in range(k)) or 1.0
    return (num / den) / y[-1] * 100.0


def exit_plateau_slope(closes: list, start: int, k: int, thr: float) -> int:
    """First bar at or after `start+k` whose k-bar slope is no longer positive.

    `thr` is in % per bar. thr=0 is the rule read literally -- "stops rising";
    a small positive thr demands the rise still be meaningful, which is the
    difference between selling on a pause and selling on a plateau.
    """
    for i in range(start + k, len(closes)):
        if slope_pct(closes[:i + 1], k) <= thr:
            return i
    return len(closes) - 1


def exit_below_ma(closes: list, start: int, n: int) -> int:
    for i in range(start + n, len(closes)):
        ma = sum(closes[i - n + 1:i + 1]) / n
        if closes[i] < ma:
            return i
    return len(closes) - 1


def exit_no_new_high(closes: list, start: int, k: int) -> int:
    peak, since = closes[start], 0
    for i in range(start + 1, len(closes)):
        if closes[i] > peak:
            peak, since = closes[i], 0
        else:
            since += 1
            if since >= k:
                return i
    return len(closes) - 1


def exit_zigzag_end(closes: list, start: int, give_back: float) -> int:
    peak = closes[start]
    for i in range(start + 1, len(closes)):
        peak = max(peak, closes[i])
        if closes[i] <= peak * (1 - give_back / 100.0):
            return i
    return len(closes) - 1


def summarise(name, res):
    """res = [(realised%, ideal%, bars_held, exited_before_peak)]"""
    if not res:
        return None
    real = [r[0] for r in res]
    caps = [r[0] / r[1] for r in res if r[1] > 0]
    return {
        "name": name, "n": len(res),
        "median": st.median(real),
        "mean": sum(real) / len(real),
        "capture": st.median(caps) if caps else float("nan"),
        "bars": st.median([r[2] for r in res]),
        "early": sum(1 for r in res if r[3]) / len(res),
        "loss": sum(1 for r in real if r <= 0) / len(real),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=float, default=20.0)
    ap.add_argument("--give-back", type=float, default=2.0)
    ap.add_argument("--from-day", default="2026-04-14",
                    help="holdout boundary; trends starting earlier are skipped")
    args = ap.parse_args()

    print("=" * 96)
    print("TREND EXIT RULE -- hold while it rises, sell on a plateau or a turn")
    print("population: +%.0f%% ZigZag trends from %s, entered at the trend start"
          % (args.run, args.from_day))
    print("=" * 96)

    # Entering at the trend start is deliberately GENEROUS to the exit rule: it
    # isolates the exit question from the detector's timing, which is measured
    # separately and is the weaker half. Read the capture ratios as an upper
    # bound on what the pair would achieve together.
    episodes = []
    for sym in UP.watchlist():
        bars = CS.bars(sym)
        if len(bars) < 200:
            continue
        idx = {b[0]: i for i, b in enumerate(bars)}
        closes = [b[4] for b in bars]
        for t in UP.trends_for(sym, args.run, args.give_back, 4):
            stt = UP.attr(t, "start_ts", "start", "low_ts")
            en = UP.attr(t, "end_ts", "end", "high_ts")
            if stt is None or en is None:
                continue
            if stt.strftime("%Y-%m-%d") < args.from_day:
                continue
            a, b = idx.get(stt), idx.get(en)
            if a is None or b is None or b <= a + 4:
                continue
            peak_i = max(range(a, b + 1), key=lambda j: closes[j])
            episodes.append({"sym": sym, "closes": closes, "a": a,
                             "peak_i": peak_i, "entry": closes[a],
                             "peak": closes[peak_i]})

    if len(episodes) < 20:
        print("only %d episodes -- too few" % len(episodes))
        return
    print("episodes %d over %d symbols" % (
        len(episodes), len(set(e["sym"] for e in episodes))))

    def run_policy(name, fn):
        res = []
        for e in episodes:
            i = fn(e["closes"], e["a"])
            i = min(max(i, e["a"] + 1), len(e["closes"]) - 1)
            realised = (e["closes"][i] / e["entry"] - 1) * 100
            ideal = (e["peak"] / e["entry"] - 1) * 100
            res.append((realised, ideal, i - e["a"], i < e["peak_i"]))
        return summarise(name, res)

    rows = [run_policy("ideal (exit at peak)",
                       lambda c, a, e=None: 0)]
    rows[0] = summarise("ideal (exit at peak)",
                        [((e["peak"] / e["entry"] - 1) * 100,
                          (e["peak"] / e["entry"] - 1) * 100,
                          e["peak_i"] - e["a"], False) for e in episodes])

    for k in (6, 12, 24):
        for thr in (0.0, 0.05):
            rows.append(run_policy(
                "plateau slope%d thr%.2f" % (k, thr),
                lambda c, a, k=k, thr=thr: exit_plateau_slope(c, a, k, thr)))
    for n in (12, 24, 48):
        rows.append(run_policy("below MA%d" % n,
                               lambda c, a, n=n: exit_below_ma(c, a, n)))
    for k in (6, 12, 24):
        rows.append(run_policy("no new high %dh" % k,
                               lambda c, a, k=k: exit_no_new_high(c, a, k)))
    rows.append(run_policy("zigzag end (-%.0f%% from peak)" % args.give_back,
                           lambda c, a: exit_zigzag_end(c, a, args.give_back)))

    print()
    print("%-28s%7s%10s%10s%10s%8s%9s" % (
        "policy", "n", "median%", "mean%", "capture", "bars", "early"))
    print("-" * 96)
    for r in rows:
        if r:
            print("%-28s%7d%10.2f%10.2f%10.2f%8.0f%8.0f%%" % (
                r["name"], r["n"], r["median"], r["mean"], r["capture"],
                r["bars"], r["early"] * 100))

    print()
    print("READING THIS")
    print("  'capture' is realised / available-to-the-peak, so 1.00 is the")
    print("  ceiling and only 'ideal' reaches it. 'early' is the share that")
    print("  exited BEFORE the peak: a rule can score well on capture by")
    print("  leaving early and still be wrong, because the trend resumed")
    print("  without it. Read the two together, never capture alone.")


if __name__ == "__main__":
    main()
