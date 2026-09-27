"""Leader mode v2: the three follow-ups pre-registered in leader-mode-spec.md.

The v1 finding: the FIRST leader of the day is a real leader only 38-50% of the
time, and a wide stop pays 3-4% on each false one. Three changes, fixed BEFORE
this run (12 variants, no selection -- every variant is printed on train and on
test; the baseline is v1's R<=2, Y>=7.5%, N=1, k3/8%):

  L-1 persistence   enter only after the coin has held rank <= 3 (and >= +7.5%)
                    for P consecutive hours (P = 1, 2)
  L-2 pullback      a leader (rank <= 3, >= +7.5%) is armed; enter on the first
                    green 15m close after it has pulled back >= D from its day
                    high (D = 3%, 5%) while still rank <= 5
  L-3 leadership exit  exit when the coin falls out of the day's top-10 (after a
                    1h minimum hold), the k3/8% trail stays as a safety net
  L-1 + L-3         both

Everything else as v1: 15m grid of every watchlist coin, ranks by return since
the UTC open from bars <= t, entry at the signal bar's close, P-1 trail engine,
7-day cap, at most N entries per UTC day. Split: before / from 2026-03-01.
Spec: docs/specs/features/leader-mode-spec.md
"""
import collections
import json
import random
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config as cfg  # noqa: E402
import immutable_labels as IL  # noqa: E402
import indicators as I  # noqa: E402
import _backtest_trend_start_detector as TD  # noqa: E402
import _compute_early_capture as E  # noqa: E402

SPLIT = "2026-03-01"
STEP = timedelta(minutes=15)
MAX_HOLD = 96 * 7
Y = 0.075
K, FLOOR = 3.0, 0.08

wl = E.load_watchlist()
raw = {s: TD.bars_15m(s) for s in wl}
raw = {s: b for s, b in raw.items() if len(b) > 2000}
syms = sorted(raw)
t0 = min(b[0][0] for b in raw.values()).replace(hour=0, minute=0)
t1 = max(b[-1][0] for b in raw.values())
N = int((t1 - t0) / STEP) + 1
S = len(syms)
O, H, L, C = (np.full((N, S), np.nan) for _ in range(4))
for j, s in enumerate(syms):
    for (t, o, h, l, c, v) in raw[s]:
        k = int((t - t0) / STEP)
        O[k, j], H[k, j], L[k, j], C[k, j] = o, h, l, c
ATR = np.column_stack([I._atr(np.nan_to_num(H[:, j], nan=np.nanmean(H[:, j])),
                              np.nan_to_num(L[:, j], nan=np.nanmean(L[:, j])),
                              np.nan_to_num(C[:, j], nan=np.nanmean(C[:, j])), cfg.ATR_PERIOD) for j in range(S)])
days = N // 96
N2 = days * 96
day_open = np.repeat(O[::96][:days], 96, axis=0)
RET = C[:N2] / day_open - 1
RANK = np.full((N2, S), 999, dtype=int)
for i in range(N2):
    row = RET[i]
    ok = np.isfinite(row)
    if ok.any():
        order = np.argsort(-np.where(ok, row, -9))
        RANK[i, order] = np.arange(1, S + 1)
        RANK[i, ~ok] = 999
DAYHI = np.full((N2, S), np.nan)
for d in range(days):
    DAYHI[d * 96:(d + 1) * 96] = np.fmax.accumulate(H[d * 96:(d + 1) * 96], axis=0)
DAY = [(t0 + d * 96 * STEP).strftime("%Y-%m-%d") for d in range(days)]
eod_top3 = set()
for d in range(days):
    last = C[d * 96 + 95] / O[d * 96] - 1
    for j in np.argsort(-np.nan_to_num(last, nan=-9))[:3]:
        eod_top3.add((DAY[d], syms[j]))
win, _ = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
win = set(win)
labelled = {k[0] for k in win}


def exit_of(i, j, lead_exit):
    ep = C[i, j]
    stop = ep - max(K * ATR[i, j], FLOOR * ep)
    last = min(N2 - 1, i + MAX_HOLD)
    for q in range(i + 1, last + 1):
        cq = C[q, j]
        if not np.isfinite(cq):
            continue
        if np.isfinite(ATR[q, j]):
            stop = max(stop, cq - max(K * ATR[q, j], FLOOR * cq))
        if cq < stop:
            return q, "trail"
        if lead_exit and q - i >= 4 and RANK[q, j] > 10:
            return q, "lost lead"
    q = last
    while q > i and not np.isfinite(C[q, j]):
        q -= 1
    return q, "cap"


def run(entry, NMAX, lead_exit, P=0, D=0.0):
    trades = []
    busy = np.full(S, -1)
    for d in range(1, days - 1):
        ds = d * 96
        taken = 0
        streak = np.zeros(S, dtype=int)
        armed = np.zeros(S, dtype=bool)
        pulled = np.zeros(S, dtype=bool)
        for i in range(ds, ds + 96):
            if taken >= NMAX:
                break
            lead = (RANK[i] <= 3) & (RET[i] >= Y)
            streak = np.where(lead, streak + 1, 0)
            cands = []
            if entry == "first":
                cands = [j for j in np.where((RANK[i] <= 2) & (RET[i] >= Y))[0]]
            elif entry == "persist":
                cands = [j for j in np.where(streak >= 4 * P)[0]]
            elif entry == "pullback":
                armed |= lead
                pulled |= armed & (C[i] <= DAYHI[i] * (1 - D))
                green = C[i] > C[i - 1]
                cands = [j for j in np.where(pulled & green & (RANK[i] <= 5))[0]]
            for j in sorted(cands, key=lambda j: RANK[i, j]):
                if taken >= NMAX or busy[j] >= i or not np.isfinite(ATR[i, j]):
                    continue
                x, why = exit_of(i, j, lead_exit)
                if x <= i:
                    continue
                pnl = C[x, j] / C[i, j] - 1
                peak = np.nanmax(H[i + 1:min(N2, x + 97), j])
                move = peak / O[ds, j] - 1
                trades.append({"day": DAY[d], "sym": syms[j], "pnl": pnl, "bars": x - i, "why": why,
                               "capture": pnl / move if move > 0 else np.nan,
                               "top3": (DAY[d], syms[j]) in eod_top3,
                               "top20": ((DAY[d], syms[j]) in win) if DAY[d] in labelled else None})
                busy[j] = x
                taken += 1
                if entry == "pullback":
                    armed[j] = pulled[j] = False
    return trades


rnd = random.Random(3)


def summ(tr, nd):
    if len(tr) < 10:
        return "n=%d (too few)" % len(tr)
    p = np.array([t["pnl"] for t in tr]) * 100
    bs = sorted(np.mean(rnd.choices(p, k=len(p))) for _ in range(1000))
    cap = np.array([t["capture"] for t in tr if np.isfinite(t["capture"])]) * 100
    t20 = [t["top20"] for t in tr if t["top20"] is not None]
    return "n/day %.2f  mean %+5.2f%% [%+5.2f,%+5.2f]  med %+5.2f%%  win %2.0f%%  top3 %2.0f%%  top20 %2.0f%%  capture med %+4.0f%%  hold %4.1fh" % (
        len(tr) / nd, p.mean(), bs[25], bs[-26], np.median(p), (p > 0).mean() * 100, np.mean([t["top3"] for t in tr]) * 100,
        np.mean(t20) * 100 if t20 else float("nan"), np.median(cap), np.median([t["bars"] for t in tr]) / 4)


ntr = sum(1 for d in DAY if d < SPLIT)
nte = sum(1 for d in DAY if d >= SPLIT)
VARIANTS = [("v1 baseline (rank<=2 first)", "first", 1, False, {}),
            ("L-1 persist 1h", "persist", 1, False, {"P": 1}), ("L-1 persist 1h, N=3", "persist", 3, False, {"P": 1}),
            ("L-1 persist 2h", "persist", 1, False, {"P": 2}), ("L-1 persist 2h, N=3", "persist", 3, False, {"P": 2}),
            ("L-2 pullback 3%", "pullback", 1, False, {"D": 0.03}), ("L-2 pullback 3%, N=3", "pullback", 3, False, {"D": 0.03}),
            ("L-2 pullback 5%", "pullback", 1, False, {"D": 0.05}), ("L-2 pullback 5%, N=3", "pullback", 3, False, {"D": 0.05}),
            ("L-3 lead exit", "first", 1, True, {}), ("L-3 lead exit, N=3", "first", 3, True, {}),
            ("L-1 2h + L-3", "persist", 1, True, {"P": 2}), ("L-1 2h + L-3, N=3", "persist", 3, True, {"P": 2})]
out = {}
for name, entry, NMAX, lx, kw in VARIANTS:
    tr = run(entry, NMAX, lx, **kw)
    a = [t for t in tr if t["day"] < SPLIT]
    b = [t for t in tr if t["day"] >= SPLIT]
    out[name] = tr
    print("%-30s TRAIN %s" % (name, summ(a, ntr)))
    print("%-30s TEST  %s" % ("", summ(b, nte)), flush=True)
    if lx:
        print("%-30s exits: %s" % ("", dict(collections.Counter(t["why"] for t in b))))
for name in out:
    ex = []
    for s_, d_ in (("QNTUSDT", "2026-09-24"), ("STRKUSDT", "2026-09-18"), ("QNTUSDT", "2026-09-25")):
        x = [t for t in out[name] if t["sym"] == s_ and t["day"] == d_]
        ex.append("%s %s %s" % (s_[:4], d_[5:], ("%+.1f%%" % (100 * x[0]["pnl"])) if x else "-"))
    print("  examples %-30s %s" % (name, " | ".join(ex)))
json.dump({k: v for k, v in out.items()}, open(FILES.parent / ".runtime/backtests/leader_mode_v2_trades.json", "w", encoding="utf-8"), default=str)
