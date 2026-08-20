"""Does gpt_crypto_bot's 4h_leader_watch rule earn its 40% hit rate?

Origin: measured on that bot's own log, the mode showed median +3.15% forward
move within 24h and 40% of signals exceeding +5% -- roughly double every other
mode it runs. On **n = 10**. This script re-derives the rule from its source and
replays it over the maximum available window so the number can be believed or
discarded.

THE RULE, reproduced from gpt_crypto_bot/files/market_signal_agent.py
(_four_h_leader_watch_reason) and its config, evaluated on 1h bars:

    4h context score >= 7.0        (a weighted 4h trend-quality score, below)
    today_change_pct >= 4.0        (close vs the day's opening bar)
    daily_range <= 35.0            (close vs the lowest low of 96 bars)
    ADX >= 30, slope(5) >= 0.35
    50 <= RSI <= 78
    vol_x >= 0.35
    STRENGTH GATE: today_change_pct >= 10.0 AND vol_x >= 3.0
    MACD hist > 0
    price > EMA20 > EMA50
    price_edge = price/EMA20-1 <= 8%
    and (a fresh close above the previous one OR price_edge <= 3.5%)

The strength gate is the interesting part and the reason this is NOT an
early-start detector: it requires the coin to be **already up 10% on the day on
triple volume**. By construction it enters a confirmed leader, so a high hit
rate is what one would expect -- the question this script answers is whether the
rate is higher than simply firing on any bar that is already up 10% on 3x volume.

That is the comparison that matters, so three populations are measured:
  FULL      the complete rule
  NO_4H     the rule without the 4h context requirement
  STRENGTH  the strength gate ALONE (up 10% today, vol_x >= 3)
plus a random-bar baseline matched in count.

No lookahead: the 4h score uses the last CLOSED 4h bar strictly before the
current 1h bar, mirroring i = len(c) - 2 in the original.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _backtest_trend_start_detector as TD     # noqa: E402
import _diag_uptrend_population as UP           # noqa: E402

EMA_FAST, EMA_SLOW, RSI_P, ADX_P, SLOPE_LB, VOL_LB, DR_BARS = 20, 50, 14, 14, 5, 20, 96


def _ema(v, n):
    k = 2.0 / (n + 1.0)
    out = [v[0]]
    for x in v[1:]:
        out.append(x * k + out[-1] * (1 - k))
    return out


def _rsi(c, n=RSI_P):
    out = [50.0]
    ag = al = 0.0
    for i in range(1, len(c)):
        d = c[i] - c[i - 1]
        g, l = max(d, 0.0), max(-d, 0.0)
        if i <= n:
            ag += g / n
            al += l / n
        else:
            ag = (ag * (n - 1) + g) / n
            al = (al * (n - 1) + l) / n
        out.append(100.0 if al == 0 else 100 - 100 / (1 + ag / al))
    return out


def _adx(bars, n=ADX_P):
    """Wilder ADX. Needed because the rule's ADX >= 30 is one of its teeth."""
    if len(bars) < n + 2:
        return [0.0] * len(bars)
    tr, pdm, ndm = [0.0], [0.0], [0.0]
    for i in range(1, len(bars)):
        h, l, pc = bars[i][2], bars[i][3], bars[i - 1][4]
        ph, pl = bars[i - 1][2], bars[i - 1][3]
        tr.append(max(h - l, abs(h - pc), abs(l - pc)))
        up, dn = h - ph, pl - l
        pdm.append(up if (up > dn and up > 0) else 0.0)
        ndm.append(dn if (dn > up and dn > 0) else 0.0)
    atr = sum(tr[1:n + 1]) / n
    ap = sum(pdm[1:n + 1]) / n
    an = sum(ndm[1:n + 1]) / n
    out = [0.0] * (n + 1)
    dxs = []
    for i in range(n + 1, len(bars)):
        atr = (atr * (n - 1) + tr[i]) / n
        ap = (ap * (n - 1) + pdm[i]) / n
        an = (an * (n - 1) + ndm[i]) / n
        pdi = 100 * ap / atr if atr else 0.0
        ndi = 100 * an / atr if atr else 0.0
        dx = 100 * abs(pdi - ndi) / (pdi + ndi) if (pdi + ndi) else 0.0
        dxs.append(dx)
        out.append(sum(dxs[-n:]) / min(len(dxs), n))
    return out


def to_4h(bars):
    """Aggregate 1h into 4h on the 0/4/8/12/16/20 UTC grid the exchange uses."""
    buckets = {}
    for b in bars:
        k = b[0].replace(hour=(b[0].hour // 4) * 4, minute=0, second=0, microsecond=0)
        g = buckets.get(k)
        if g is None:
            buckets[k] = [k, b[1], b[2], b[3], b[4], b[5]]
        else:
            g[2] = max(g[2], b[2])
            g[3] = min(g[3], b[3])
            g[4] = b[4]
            g[5] += b[5]
    return [tuple(buckets[k]) for k in sorted(buckets)]


def four_h_scores(b4):
    """Reproduces monitor.py::_four_h_context_score, one value per 4h bar."""
    if len(b4) < 60:
        return {}
    c = [x[4] for x in b4]
    o = [x[1] for x in b4]
    v = [x[5] for x in b4]
    ef, es = _ema(c, EMA_FAST), _ema(c, EMA_SLOW)
    rsi = _rsi(c)
    macd = [a - b for a, b in zip(_ema(c, 12), _ema(c, 26))]
    hist = [m - s for m, s in zip(macd, _ema(macd, 9))]
    out = {}
    for i in range(len(b4)):
        if i < 60:
            continue
        vs = sum(v[max(0, i - VOL_LB + 1):i + 1]) / min(i + 1, VOL_LB)
        vol_x = v[i] / vs if vs > 0 else 0.0
        slope = (((ef[i] / ef[i - SLOPE_LB]) - 1) * 100.0
                 if i >= SLOPE_LB and ef[i - SLOPE_LB] > 0 else 0.0)
        greens = sum(1 for j in range(max(0, i - 2), i + 1) if c[j] > o[j])
        s = 0.0
        s += 2.0 if c[i] > ef[i] > 0 else -1.5
        s += 1.2 if c[i] > es[i] > 0 else -0.8
        s += 1.3 if ef[i] > es[i] > 0 else -0.8
        s += max(-3.0, min(3.0, slope * 1.6))
        s += 0.8 * greens
        s += 1.0 if hist[i] > 0 else -1.0
        if 45.0 <= rsi[i] <= 68.0:
            s += 0.8
        elif rsi[i] < 42.0 or rsi[i] > 75.0:
            s -= 0.8
        s += 0.5 if vol_x >= 0.8 else -0.5
        out[b4[i][0]] = max(-6.0, min(8.0, s))
    return out


def fires(sym, variant):
    """Bars where the rule (or a stated subset of it) triggers."""
    bars = TD.load_bars(sym, "1h")
    if len(bars) < 400:
        return []
    b4 = to_4h(bars)
    sc4 = four_h_scores(b4)
    if not sc4 and variant == "full":
        return []
    keys = sorted(sc4)
    c = [x[4] for x in bars]
    v = [x[5] for x in bars]
    ef, es = _ema(c, EMA_FAST), _ema(c, EMA_SLOW)
    rsi = _rsi(c)
    adx = _adx(bars)
    macd = [a - b for a, b in zip(_ema(c, 12), _ema(c, 26))]
    hist = [m - s for m, s in zip(macd, _ema(macd, 9))]
    day_open = {}
    for b in bars:
        day_open.setdefault(b[0].strftime("%Y-%m-%d"), b[1])
    out = []
    ki = 0
    for i in range(max(DR_BARS, 300), len(bars)):
        ts = bars[i][0]
        # last CLOSED 4h bar strictly before now (mirrors i = len(c) - 2)
        while ki + 1 < len(keys) and keys[ki + 1] < ts:
            ki += 1
        s4 = sc4.get(keys[ki], -99.0) if keys and keys[ki] < ts else -99.0
        vs = sum(v[i - VOL_LB + 1:i + 1]) / VOL_LB
        vol_x = v[i] / vs if vs > 0 else 0.0
        base = day_open.get(ts.strftime("%Y-%m-%d"), c[i])
        today = (c[i] / base - 1) * 100.0 if base > 0 else 0.0
        low96 = min(x[3] for x in bars[i - DR_BARS + 1:i + 1])
        drange = (c[i] / low96 - 1) * 100.0 if low96 > 0 else 0.0
        slope = (((ef[i] / ef[i - SLOPE_LB]) - 1) * 100.0
                 if ef[i - SLOPE_LB] > 0 else 0.0)

        strength = today >= 10.0 and vol_x >= 3.0
        if variant == "strength":
            if strength:
                out.append((ts, c[i]))
            continue
        if variant != "no4h" and s4 < 7.0:
            continue
        if today < 4.0 or drange > 35.0:
            continue
        if adx[i] < 30.0 or slope < 0.35:
            continue
        if not (50.0 <= rsi[i] <= 78.0) or vol_x < 0.35:
            continue
        if not strength:
            continue
        if hist[i] <= 0.0:
            continue
        if not (c[i] > ef[i] > es[i] > 0):
            continue
        edge = (c[i] / ef[i] - 1) * 100.0
        if edge > 8.0:
            continue
        if not (c[i] > c[i - 1] or edge <= 3.5):
            continue
        out.append((ts, c[i]))
    return out


def forward(sym, hits, hours=24):
    bars = TD.load_bars(sym, "1h")
    idx = {b[0]: i for i, b in enumerate(bars)}
    out = []
    for ts, px in hits:
        i = idx.get(ts)
        if i is None:
            continue
        fut = bars[i + 1:i + 1 + hours]
        if not fut:
            continue
        out.append((max(x[2] for x in fut) / px - 1) * 100.0)
    return out


def q(v, p):
    v = sorted(v)
    return v[int(p * (len(v) - 1))] if v else float("nan")


def report(name, per_sym, syms, hours):
    allf, n = [], 0
    intos, aheads = [], []
    for s in syms:
        hits = per_sym.get(s) or []
        n += len(hits)
        allf.extend(forward(s, hits, hours))
        if not hits:
            continue
        bars = TD.load_bars(s, "1h")
        for t in TD.trends_tf(s, 20.0, 2.0, "1h"):
            st = UP.attr(t, "start_ts", "start", "low_ts")
            en = UP.attr(t, "end_ts", "end", "high_ts")
            g = UP.attr(t, "gain_pct", "gain")
            if not (st and en and g):
                continue
            hit = next(((ts, px) for ts, px in hits if st <= ts <= en), None)
            if hit:
                peak = max((x[2] for x in bars if st <= x[0] <= en), default=hit[1])
                a = (peak / hit[1] - 1) * 100
                aheads.append(a)
                intos.append((1 - a / g) * 100)
    print("%-14s%8d%11.2f%%%11.2f%%%8.0f%%%8.0f%%%9s%9s" % (
        name, n, q(allf, .5), q(allf, .75),
        100 * sum(1 for x in allf if x > 5) / len(allf) if allf else 0,
        100 * sum(1 for x in allf if x > 10) / len(allf) if allf else 0,
        "%.0f%%" % q(intos, .5) if intos else "-",
        "%.1f%%" % q(aheads, .5) if aheads else "-"))
    return allf, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--symbols", type=int, default=0)
    args = ap.parse_args()

    syms = [s for s in UP.watchlist() if len(TD.load_bars(s, "1h")) >= 400]
    if args.symbols:
        syms = syms[:args.symbols]

    print("=" * 92)
    print("4h_leader_watch, reproduced from gpt_crypto_bot and replayed on the MAX window")
    print("population: every 1h bar of %d watchlist symbols" % len(syms))
    print("=" * 92)
    print()
    print("%-14s%8s%12s%12s%8s%8s%9s%9s" % (
        "variant", "fires", "fwd med", "fwd p75", ">5%", ">10%", "into", "ahead"))
    print("-" * 92)

    results = {}
    for variant, label in (("full", "FULL RULE"), ("no4h", "NO 4h GATE"),
                           ("strength", "STRENGTH ONLY")):
        per = {s: fires(s, variant) for s in syms}
        results[variant] = report(label, per, syms, args.hours)

    # Baseline: random bars, matched in count. Without it a hit rate on an
    # already-moving population measures the population, not the rule.
    n_full = results["full"][1]
    rng = random.Random(7)
    rnd = []
    for _ in range(max(n_full, 300)):
        s = rng.choice(syms)
        bars = TD.load_bars(s, "1h")
        if len(bars) < 400:
            continue
        i = rng.randrange(300, len(bars) - args.hours - 1)
        fut = bars[i + 1:i + 1 + args.hours]
        if fut:
            rnd.append((max(x[2] for x in fut) / bars[i][4] - 1) * 100)
    print("%-14s%8d%11.2f%%%11.2f%%%8.0f%%%8.0f%%%9s%9s" % (
        "RANDOM BAR", len(rnd), q(rnd, .5), q(rnd, .75),
        100 * sum(1 for x in rnd if x > 5) / len(rnd) if rnd else 0,
        100 * sum(1 for x in rnd if x > 10) / len(rnd) if rnd else 0, "-", "-"))

    print()
    print("READ THIS")
    print("  The strength gate requires the coin to be ALREADY up 10% today on 3x")
    print("  volume, so a high hit rate is expected and proves nothing on its own.")
    print("  What the rule must beat is the STRENGTH ONLY row -- the same population")
    print("  without the 4h context, the ADX/RSI band and the EMA stack.")
    print("  'into' is how far into a +20% move the fire landed; high = late.")


if __name__ == "__main__":
    main()
