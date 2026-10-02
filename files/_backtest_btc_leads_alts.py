"""Operator hypothesis (2026-10-03): SOL and ETH start rising right after BTC, with a
move several times larger; watching BTC therefore predicts -- "guaranteed" -- the
early start of their trend (and of other coins'), and its end.

Maximum period: the 15m long store (~459 days) for the whole watchlist.
Every claim is split into a testable part; criteria fixed before the run; each
statistic is reported for the full period and for its first and second half.

A  amplitude "several times larger": beta of the alt's return on BTC's (OLS) at
   15m, 1h and 1d.                          claim holds if beta >= 2
B  "right after": does BTC's return in bar t predict the alt's in t+1..t+4 beyond
   the alt's own bar t?  corr(btc_t, alt_t+L); OLS alt_t+1 ~ btc_t + alt_t.
                                            claim holds if the lead coefficient is
                                            > 0 in both halves AND economically
                                            non-trivial (alt move per +1% BTC >= 0.1%)
C  trend START: event = a 1h bar with BTC >= +theta (0.5%, 1.0%) after 6 quiet hours
   (BTC 6h return <= +0.3%). Causal: measured from the event's close. Alt move
   INSIDE the event hour (already done) vs AFTER it (1h / 4h / 12h), against the
   unconditional mean of the same horizon.  "guaranteed": P(alt 4h fwd > 0) >= 80%
   in both halves
D  trend END: after BTC +2% over 12h, a 1h bar with BTC <= -0.5%. Alt forward
   1h / 4h / 12h vs unconditional.          "guaranteed": P(alt 4h fwd < 0) >= 80%
E  tradable: buy the alt at the start event's close, sell 4h / 12h later, 0.1%
   round-trip fee; mean with bootstrap 95%.
F  which coin: on days with a BTC start event, is the share of a coin being an
   immutable top-20 winner higher than on other days, and do SOL/ETH ever win?
"""
import collections
import math
import random
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config as cfg  # noqa: E402
import exit_validator as EV  # noqa: E402
import immutable_labels as IL  # noqa: E402
import _compute_early_capture as E  # noqa: E402

FOCUS = ("SOLUSDT", "ETHUSDT")
FEE = 0.10
THETAS = (0.005, 0.010)
QUIET_6H = 0.003


def lr(x):
    return np.diff(np.log(x))


def beta(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    return float(np.cov(a, b)[0, 1] / np.var(b)) if len(b) > 100 else float("nan")


def agg(C, k):
    n = (len(C) // k) * k
    return C[:n][k - 1::k]


def ci(v, rnd):
    if len(v) < 5:
        return float("nan"), float("nan")
    bs = sorted(np.mean(rnd.choices(v, k=len(v))) for _ in range(1000))
    return bs[25], bs[-26]


def main():
    g = EV.load_grid(cfg.ATR_PERIOD)
    jb = g.col["BTCUSDT"]
    C = g.C[: g.N]
    halves = {"full": (0, g.N), "1st half": (0, g.N // 2), "2nd half": (g.N // 2, g.N)}
    alts = [s for s in g.syms if s != "BTCUSDT"]
    print("grid: %d coins, %s .. %s (%d days)" % (len(g.syms), g.t0.date(), g.last_bar.date(), g.days))

    # ---------------- A amplitude
    print("\n=== A. AMPLITUDE: beta to BTC (claim: >= 2) ===")
    for k, name in ((1, "15m"), (4, "1h"), (96, "1d")):
        cb = agg(C[:, jb], k)
        rb = lr(cb)
        bet = {s: beta(lr(agg(C[:, g.col[s]], k)), rb) for s in alts}
        allb = np.array([v for v in bet.values() if np.isfinite(v)])
        print("  %-4s SOL %.2f  ETH %.2f | watchlist median %.2f, share >= 2: %.0f%%" % (
            name, bet["SOLUSDT"], bet["ETHUSDT"], np.median(allb), 100 * np.mean(allb >= 2)))

    # ---------------- B lead at 15m and 1h
    print("\n=== B. 'RIGHT AFTER': does BTC lead? (alt move per +1%% BTC, next bar, beyond the alt's own bar) ===")
    for k, name in ((1, "15m"), (4, "1h")):
        for hname, (a0, a1) in halves.items():
            cb = agg(C[a0:a1, jb], k)
            rb = lr(cb)
            row = []
            coefs = []
            for s in alts:
                ra = lr(agg(C[a0:a1, g.col[s]], k))
                y, x1, x2 = ra[1:], rb[:-1], ra[:-1]
                ok = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
                if ok.sum() < 200:
                    continue
                X = np.column_stack([np.ones(ok.sum()), x1[ok], x2[ok]])
                coef = np.linalg.lstsq(X, y[ok], rcond=None)[0]
                coefs.append(coef[1])
                if s in FOCUS:
                    c0 = np.corrcoef(rb[ok[:]] if False else rb[:-1][ok], ra[:-1][ok])[0, 1]
                    c1 = np.corrcoef(x1[ok], y[ok])[0, 1]
                    row.append("%s same-bar corr %.2f, next-bar corr %+.3f, lead %+.3f%%" % (
                        s[:3], c0, c1, coef[1]))
            print("  %-4s %-8s %s | watchlist median lead %+.3f%% per +1%% BTC, >0 in %.0f%%" % (
                name, hname, "; ".join(row), np.median(coefs), 100 * np.mean(np.array(coefs) > 0)))

    # ---------------- C/D events on 1h
    C1 = agg(C, 4)
    R1 = np.log(C1[1:] / C1[:-1]) * 100            # % per 1h bar, index t = bar t+1's return
    n1 = R1.shape[0]
    rb1 = R1[:, jb]
    cb1 = C1[:, jb]

    def fwd(j, t, h):                                # % from close of bar t to close of t+h
        if t + h >= C1.shape[0]:
            return float("nan")
        a, b = C1[t, j], C1[t + h, j]
        return (b / a - 1) * 100 if np.isfinite(a) and np.isfinite(b) and a > 0 else float("nan")

    def base(j, h, t0, t1):
        v = [fwd(j, t, h) for t in range(t0, t1, 3)]
        v = [x for x in v if math.isfinite(x)]
        return float(np.mean(v)), float(np.mean(np.array(v) > 0)), float(np.mean(np.array(v) < 0))

    start_events = {}
    for th in THETAS:
        ev = []
        last = -99
        for t in range(7, C1.shape[0] - 13):
            r = (cb1[t] / cb1[t - 1] - 1) if cb1[t - 1] > 0 else 0
            prior = (cb1[t - 1] / cb1[t - 7] - 1) if cb1[t - 7] > 0 else 0
            if r >= th and prior <= QUIET_6H and t - last > 6:
                ev.append(t)
                last = t
        start_events[th] = ev
    end_events = []
    last = -99
    for t in range(13, C1.shape[0] - 13):
        run = (cb1[t - 1] / cb1[t - 13] - 1) if cb1[t - 13] > 0 else 0
        r = (cb1[t] / cb1[t - 1] - 1) if cb1[t - 1] > 0 else 0
        if run >= 0.02 and r <= -0.005 and t - last > 12:
            end_events.append(t)
            last = t
    H = C1.shape[0]
    hb = {"full": (0, H), "1st half": (0, H // 2), "2nd half": (H // 2, H)}
    rnd = random.Random(3)

    def report(name, events, want_up):
        print("\n=== %s: %d events ===" % (name, len(events)))
        for s in FOCUS + ("watchlist mean",):
            for hname, (t0, t1) in hb.items():
                ev = [t for t in events if t0 <= t < t1]
                if not ev:
                    continue
                if s == "watchlist mean":
                    js = [g.col[x] for x in alts]
                else:
                    js = [g.col[s]]
                inside, f1, f4, f12, btc4 = [], [], [], [], []
                for t in ev:
                    for j in js:
                        a, b = C1[t - 1, j], C1[t, j]
                        if np.isfinite(a) and np.isfinite(b) and a > 0:
                            inside.append((b / a - 1) * 100)
                        f1.append(fwd(j, t, 1)); f4.append(fwd(j, t, 4)); f12.append(fwd(j, t, 12))
                    btc4.append(fwd(jb, t, 4))
                f1 = [x for x in f1 if math.isfinite(x)]
                f4 = [x for x in f4 if math.isfinite(x)]
                f12 = [x for x in f12 if math.isfinite(x)]
                b4m, b4up, b4dn = (base(js[0], 4, t0, t1) if len(js) == 1 else (float("nan"),) * 3)
                p_dir = np.mean(np.array(f4) > 0) if want_up else np.mean(np.array(f4) < 0)
                print("  %-15s %-8s n=%3d | inside the BTC hour %+.2f%% | after: 1h %+.2f%%  4h %+.2f%%  12h %+.2f%% | "
                      "P(4h %s) %.0f%%%s | BTC 4h after %+.2f%%" % (
                          s, hname, len(ev), np.nanmean(inside), np.mean(f1), np.mean(f4), np.mean(f12),
                          "up" if want_up else "down", 100 * p_dir,
                          (" (base %.0f%%, base 4h %+.2f%%)" % (100 * (b4up if want_up else b4dn), b4m)) if len(js) == 1 else "",
                          np.nanmean(btc4)))

    for th in THETAS:
        report("C. TREND START: BTC 1h >= +%.1f%% after 6 quiet hours" % (100 * th), start_events[th], True)
    report("D. TREND END: BTC +2%% in 12h, then a 1h bar <= -0.5%%", end_events, False)

    # ---------------- E tradable
    print("\n=== E. TRADE: buy at the BTC start event's close, fee %.2f%% round trip ===" % FEE)
    for th in THETAS:
        for s in FOCUS:
            for h in (4, 12):
                for hname, (t0, t1) in hb.items():
                    v = [fwd(g.col[s], t, h) - FEE for t in start_events[th] if t0 <= t < t1]
                    v = [x for x in v if math.isfinite(x)]
                    lo, hi = ci(v, rnd)
                    print("  BTC >= +%.1f%%  %s  hold %2dh  %-8s n=%3d mean %+.2f%% [%+.2f, %+.2f] win %.0f%%" % (
                        100 * th, s[:3], h, hname, len(v), np.mean(v) if v else float("nan"), lo, hi,
                        100 * np.mean(np.array(v) > 0) if v else 0))

    # ---------------- F which coin
    print("\n=== F. WHICH COIN: immutable top-20 winners on BTC start-event days ===")
    win, _ = IL.winners_by_day(top_n=20, watchlist=E.load_watchlist(), rank_before_filter=True)
    win = set(win)
    days_all = {(g.t0 + timedelta(hours=t)).strftime("%Y-%m-%d") for t in range(H)}
    for th in THETAS:
        ev_days = {(g.t0 + timedelta(hours=t)).strftime("%Y-%m-%d") for t in start_events[th]}
        labelled = {d for d, _ in win}
        ed = [d for d in ev_days if d in labelled]
        od = [d for d in days_all if d in labelled and d not in ev_days]
        rate = lambda ds: np.mean([sum(1 for s in alts if (d, s) in win) / len(alts) for d in ds]) if ds else float("nan")  # noqa: E731
        print("  BTC >= +%.1f%%: share of watchlist coins that are top-20 winners: event days %.3f (n=%d) vs other days %.3f (n=%d)" % (
            100 * th, rate(ed), len(ed), rate(od), len(od)))
    for s in FOCUS:
        print("  %s immutable top-20 winner-days in the label store: %d" % (s, sum(1 for d, x in win if x == s)))


if __name__ == "__main__":
    main()
