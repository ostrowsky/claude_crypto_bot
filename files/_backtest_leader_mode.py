"""Leader mode: at most 1-3 entries a day into the CURRENT day-leaders, held on a wide trail.

WHY (operator, 2026-09-27)
"New goal: catch one, at most three leaders and hold them as long as needed,
collecting their rise. No more junk signals." QNT 09-26: the bot entered at
107.79, left at 121.20 (+12.4%) on a WEAK exit, and the coin went to 194.95.

WHAT IS TESTED
A rocket cannot be told apart at +2.5% (rocket-segment-spec.md: 13% ceiling),
but a coin that ALREADY leads the day is visible in real time: its rank by
return since the UTC open across the watchlist. The strategy:

  entry  at a 15m close, a coin with rank <= R by return since the UTC open and
         that return >= Y; at most N entries per UTC day; not already held
  exit   only a close-anchored trail (the P-1 engine: stop = max(stop,
         close - max(k*ATR, floor*close)), exit at the first close below it),
         no WEAK exits; hard cap MAX_HOLD bars

Parameters are chosen on days before 2026-03-01 and judged on days from
2026-03-01 (TH-03). No look-ahead: ranks and ATR use bars <= t; the trade
enters at the close of the signal bar.

METRICS
  per trade   pnl, win rate
  per day     summed pnl of the day's entries (the portfolio view)
  capture     realised pnl / the move available: from the day open to the
              highest high reached while the position was open (+ the 24h after
              the exit, to see what was left)
  leader hit  the entered coin finished its entry day in the watchlist top-3 /
              the immutable global top-20
"""
import collections
import io
import itertools
import json
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
day_open = np.repeat(O[::96][:days], 96, axis=0)
RET = C[:days * 96] / day_open - 1
DAY = [(t0 + d * 96 * STEP).strftime("%Y-%m-%d") for d in range(days)]
eod_rank = {}
for d in range(days):
    last = C[d * 96 + 95] / O[d * 96] - 1
    order = np.argsort(-np.nan_to_num(last, nan=-9))
    for r, j in enumerate(order[:3]):
        eod_rank[(DAY[d], syms[j])] = r + 1
win, _ = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
win = set(win)
labelled_days = {k[0] for k in win}
print("grid %s .. %s, %d days x %d coins" % (DAY[0], DAY[-1], days, S))


def run(R, Y, NMAX, k, floor):
    trades = []
    busy_until = np.full(S, -1)
    for d in range(1, days - 1):
        ds = d * 96
        taken = 0
        for i in range(ds, ds + 96):
            if taken >= NMAX:
                break
            row = RET[i]
            if not np.isfinite(row).any():
                continue
            order = np.argsort(-np.nan_to_num(row, nan=-9))
            for r in range(R):
                j = order[r]
                if taken >= NMAX:
                    break
                if row[j] < Y or busy_until[j] >= i or not np.isfinite(ATR[i, j]):
                    continue
                ep = C[i, j]
                stop = ep - max(k * ATR[i, j], floor * ep)
                x = None
                last = min(N - 1, i + MAX_HOLD)
                for q in range(i + 1, last + 1):
                    cq = C[q, j]
                    if not np.isfinite(cq):
                        continue
                    if np.isfinite(ATR[q, j]):
                        stop = max(stop, cq - max(k * ATR[q, j], floor * cq))
                    if cq < stop:
                        x = q
                        break
                if x is None:
                    x = last
                    while x > i and not np.isfinite(C[x, j]):
                        x -= 1
                if x <= i:
                    continue
                pnl = C[x, j] / ep - 1
                peak = np.nanmax(H[i + 1:min(N, x + 97), j])
                move = peak / O[ds, j] - 1
                trades.append({"day": DAY[d], "sym": syms[j], "rank": r + 1, "ret_at_entry": row[j], "pnl": pnl,
                               "bars": x - i, "capture": pnl / move if move > 0 else np.nan,
                               "late": row[j] / move if move > 0 else np.nan,
                               "eod_top3": (DAY[d], syms[j]) in eod_rank,
                               "top20": (DAY[d], syms[j]) in win if DAY[d] in labelled_days else None})
                busy_until[j] = x
                taken += 1
    return trades


def summary(tr, ndays):
    if not tr:
        return None
    p = np.array([t["pnl"] for t in tr]) * 100
    byday = collections.defaultdict(float)
    for t in tr:
        byday[t["day"]] += t["pnl"] * 100
    cap = np.array([t["capture"] for t in tr if np.isfinite(t["capture"])])
    t20 = [t["top20"] for t in tr if t["top20"] is not None]
    return {"n": len(tr), "per_day": len(tr) / ndays, "mean": p.mean(), "median": np.median(p), "win": (p > 0).mean() * 100,
            "sum": p.sum(), "day_mean": np.mean(list(byday.values())), "capture_med": np.median(cap) * 100 if len(cap) else np.nan,
            "capture_mean": np.mean(cap) * 100 if len(cap) else np.nan, "top3": np.mean([t["eod_top3"] for t in tr]) * 100,
            "top20": np.mean(t20) * 100 if t20 else np.nan, "hold_h": np.median([t["bars"] for t in tr]) / 4,
            "big": np.mean(p >= 20) * 100}


ntr = sum(1 for d in DAY if d < SPLIT)
nte = sum(1 for d in DAY if d >= SPLIT)
res = []
GRID = list(itertools.product((1, 2, 3), (0.05, 0.075, 0.10, 0.15), (1, 3), ((3.0, 0.08), (3.0, 0.12), (4.0, 0.15), (2.0, 0.05))))
print("grid of %d variants ..." % len(GRID), flush=True)
for R, Y, NMAX, (k, fl) in GRID:
    tr = run(R, Y, NMAX, k, fl)
    a = summary([t for t in tr if t["day"] < SPLIT], ntr)
    b = summary([t for t in tr if t["day"] >= SPLIT], nte)
    res.append(((R, Y, NMAX, k, fl), a, b, tr))
res.sort(key=lambda x: -(x[1]["mean"] if x[1] else -99))
print("\nTOP 10 variants chosen on TRAIN (mean pnl per trade), judged on TEST")
hdr = "  rank<=R  ret>=Y  N/day  trail          | train: n/day mean%  win%  | TEST: n/day  mean%  med%  win%  day-sum%  capture med/mean  EOD top3  top20  >=+20%  hold h"
print(hdr)
for (R, Y, NMAX, k, fl), a, b, _ in res[:10]:
    print("  %d        %4.1f%%  %d      k%.0f floor %2.0f%% | %.2f %+6.2f %4.0f | %.2f %+6.2f %+6.2f %4.0f %+7.2f  %5.1f / %5.1f  %4.0f%%  %4.0f%%  %4.1f%%  %5.1f" % (
        R, 100 * Y, NMAX, k, 100 * fl, a["per_day"], a["mean"], a["win"], b["per_day"], b["mean"], b["median"], b["win"],
        b["day_mean"], b["capture_med"], b["capture_mean"], b["top3"], b["top20"], b["big"], b["hold_h"]))
best = res[0]
json.dump([{k2: (float(v) if isinstance(v, (np.floating, float)) else v) for k2, v in t.items()} for t in best[3]],
          io.open(FILES.parent / ".runtime/backtests/leader_mode_best_trades.json", "w", encoding="utf-8"), default=str)
print("\nbest-on-train variant, TEST by month (n, mean pnl, day-sum):")
bym = collections.defaultdict(list)
for t in best[3]:
    if t["day"] >= SPLIT:
        bym[t["day"][:7]].append(t["pnl"] * 100)
for m, v in sorted(bym.items()):
    print("   %s  n=%3d  mean %+6.2f%%  median %+6.2f%%  win %3.0f%%" % (m, len(v), np.mean(v), np.median(v), 100 * np.mean([x > 0 for x in v])))
for s_, d_ in (("QNTUSDT", "2026-09-24"), ("STRKUSDT", "2026-09-18"), ("FILUSDT", "2026-09-25")):
    x = [t for t in best[3] if t["sym"] == s_ and t["day"] == d_]
    print("   example %s %s: %s" % (s_, d_, [(round(t["ret_at_entry"] * 100, 1), round(t["pnl"] * 100, 1), round(t["bars"] / 4, 1)) for t in x] or "not entered"))
