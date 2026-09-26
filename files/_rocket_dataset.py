"""Rocket segment: one row per coin-day at the coin's first 15m CLOSE >= +2.5% from the UTC open.

WHY
The bot's real target is the "rocket": a watchlist coin that runs >= +10% from
the UTC open within the day and HOLDS it into the close (FIL 2026-09-26 +16.8% /
close +15.8%, QNT 09-24 +29.3% / +27.5%, STRK 09-18 +56.2% / +52.6%). Each gave
hours between its first +2.5% and +10%. The question this dataset answers:
at the first moment a coin is visibly moving (+2.5% on a closed 15m bar -- the
same threshold as the North Star's early deadline), what, observable THEN,
separates the days that become rockets from the days that fizzle?

NO LOOK-AHEAD
Features use bars <= i (the crossing bar, known at its close) and earlier days
only. Labels use bars > i. Cross-sectional features (breadth, rank, BTC) use
every coin's bar i on the same 15m grid.

LABELS
  day_max      max high of the UTC day / open - 1
  day_close    last close of the day / open - 1
  rocket       day_max >= 10% AND day_close >= 0.6 * day_max   (runs and holds)
  big          day_max >= 10%
  left_day     max high after i within the day / close_i - 1   (upside left)
  left_24h     max high over the next 96 bars / close_i - 1
  dd_before    lowest low after i until that day high / close_i - 1 (pain before the move)
  eod          last close of the day / close_i - 1              (hold to EOD)

Output: .runtime/backtests/rocket_events.jsonl
"""
import io
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import immutable_labels as IL  # noqa: E402
import _backtest_trend_start_detector as TD  # noqa: E402
import _compute_early_capture as E  # noqa: E402

OUT = FILES.parent / ".runtime" / "backtests" / "rocket_events.jsonl"
CROSS = 0.025
TRIGGERS = (0.025, 0.05, 0.075)   # alert at the first 15m close >= open * (1 + trigger)
STEP = timedelta(minutes=15)


def ema(x, n):
    out = np.full_like(x, np.nan)
    a = 2.0 / (n + 1)
    m = np.nan
    for k, v in enumerate(x):
        if not np.isfinite(v):
            out[k] = m
            continue
        m = v if not np.isfinite(m) else a * v + (1 - a) * m
        out[k] = m
    return out


def rsi(c, n=14):
    d = np.diff(c, prepend=np.nan)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    au, ad = ema(up, 2 * n - 1), ema(dn, 2 * n - 1)   # Wilder smoothing
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 - 100 / (1 + au / ad)


wl = E.load_watchlist()
raw = {s: TD.bars_15m(s) for s in wl}
raw = {s: b for s, b in raw.items() if len(b) > 2000}
t0 = min(b[0][0] for b in raw.values())
t1 = max(b[-1][0] for b in raw.values())
t0 = t0.replace(hour=0, minute=0)                       # grid starts at a UTC midnight
N = int((t1 - t0) / STEP) + 1
syms = sorted(raw)
S = len(syms)
O, H, L, C, V = (np.full((N, S), np.nan) for _ in range(5))
for j, s in enumerate(syms):
    for (t, o, h, l, c, v) in raw[s]:
        k = int((t - t0) / STEP)
        O[k, j], H[k, j], L[k, j], C[k, j], V[k, j] = o, h, l, c, v
print("grid %s .. %s: %d bars x %d coins" % (t0, t1, N, S))
days = N // 96
bix = syms.index("BTCUSDT") if "BTCUSDT" in syms else None

# per-bar return since the UTC open, for every coin (cross-section)
day_open = np.full((N, S), np.nan)
for d in range(days):
    day_open[d * 96:(d + 1) * 96] = O[d * 96]
ret_open = C / day_open - 1
# 1h log returns and the equal-weight basket, for decoupling
C1h = C[3::4]
lr1h = np.diff(np.log(C1h), axis=0)
basket = np.nanmean(lr1h, axis=1)

win, _ = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
win = set(win)

rows = 0
with io.open(OUT, "w", encoding="utf-8") as fo:
    for j, s in enumerate(syms):
        o, h, l, c, v = O[:, j], H[:, j], L[:, j], C[:, j], V[:, j]
        e20, e50 = ema(c, 20), ema(c, 50)
        r14 = rsi(c)
        c1h = c[3::4]
        e20h, e50h = ema(c1h, 20), ema(c1h, 50)
        lr = np.diff(np.log(c), prepend=np.nan)
        qv = v * c
        daymax_hist = []
        for d in range(8, days):
            ds = d * 96
            op = o[ds]
            if not np.isfinite(op) or op <= 0 or not np.isfinite(c[ds + 95]):
                daymax_hist.append(np.nan)
                continue
            dmax = np.nanmax(h[ds:ds + 96]) / op - 1
            dclose = c[ds + 95] / op - 1
            prior_big = [x for x in daymax_hist[-30:] if np.isfinite(x)]
            daymax_hist.append(dmax)
            first = np.where(c[ds:ds + 96] >= op * (1 + CROSS))[0]
            if len(first) == 0:
                continue
            i0 = ds + int(first[0])
            for trig in TRIGGERS:
              hit = np.where(c[ds:ds + 96] >= op * (1 + trig))[0]
              if len(hit) == 0:
                continue
              i = ds + int(hit[0])
              if i >= ds + 95 or i < 2880:
                continue
              if True:
                ci = c[i]
                # ---- features: bars <= i, earlier days ----
                n_since = i - ds + 1
                vol_prev = np.nanmean(v[i - 96:i])
                cum = np.nansum(v[ds:i + 1])
                prev_cum = [np.nansum(v[ds - k * 96:ds - k * 96 + n_since]) for k in range(1, 8)]
                rvol = cum / np.nanmean(prev_cum) if np.nanmean(prev_cum) > 0 else np.nan
                sd24 = np.nanstd(lr[ds - 96:ds])
                sd7d = np.nanstd(lr[ds - 672:ds])
                hi7 = np.nanmax(h[i - 672:i])
                hi30 = np.nanmax(h[i - 2880:i])
                lo7 = np.nanmin(l[i - 672:i])
                k1h = (i - 3) // 4                               # last CLOSED 1h bar at bar i close
                if (i - 3) % 4 != 0:
                    k1h = (i - 3) // 4 if i >= 3 else 0
                wnd = lr1h[max(0, k1h - 168):k1h, j]
                bk = basket[max(0, k1h - 168):k1h]
                m = np.isfinite(wnd) & np.isfinite(bk)
                corr = float(np.corrcoef(wnd[m], bk[m])[0, 1]) if m.sum() > 50 else np.nan
                cs = ret_open[i]
                cs_f = cs[np.isfinite(cs)]
                rank = int(1 + np.sum(cs_f > cs[j])) if np.isfinite(cs[j]) else None
                qv7 = np.nansum(qv[ds - 672:ds]) / 7
                feat = {
                    "hrs": n_since / 4.0,
                    "ret_t": ci / op - 1,
                    "bar_ret": ci / o[i] - 1 if o[i] > 0 else np.nan,
                    "vol_x": v[i] / vol_prev if vol_prev > 0 else np.nan,
                    "rvol": rvol,
                    "up_share": float(np.mean(c[ds:i + 1] > o[ds:i + 1])),
                    "dd_open": np.nanmin(l[ds:i + 1]) / op - 1,
                    "prev1d": c[ds - 1] / o[ds - 96] - 1,
                    "prev3d": c[ds - 1] / c[ds - 289] - 1,
                    "prev7d": c[ds - 1] / c[ds - 673] - 1,
                    "dist7h": ci / hi7 - 1,
                    "dist30h": ci / hi30 - 1,
                    "range7": hi7 / lo7 - 1,
                    "compress": sd24 / sd7d if sd7d > 0 else np.nan,
                    "atr_pct": float(np.nanmean(h[i - 13:i + 1] - l[i - 13:i + 1]) / ci),
                    "rsi": r14[i],
                    "above_e20": float(ci > e20[i]),
                    "stack15": float(ci > e20[i] > e50[i]),
                    "e20_slope": e20[i] / e20[i - 4] - 1,
                    "stack1h": float(c1h[k1h] > e20h[k1h] > e50h[k1h]) if k1h > 50 else np.nan,
                    "above1h_e50": float(ci > e50h[k1h]) if k1h > 50 else np.nan,
                    "corr7d": corr,
                    "btc_ret": ret_open[i, bix] if bix is not None else np.nan,
                    "breadth": float(np.mean(cs_f >= CROSS)),
                    "med_ret": float(np.median(cs_f)),
                    "excess": ci / op - 1 - float(np.median(cs_f)),
                    "rank": rank,
                    "n_big30": int(sum(x >= 0.10 for x in prior_big)),
                    "prev_dmax": prior_big[-1] if prior_big else np.nan,
                    "log_qv": math.log10(qv7) if qv7 > 0 else np.nan,
                # path since the first +2.5% close (0 at the 2.5% trigger itself)
                "trig": trig,
                "hrs_since_first": (i - i0) / 4.0,
                "pull_hi": ci / np.nanmax(h[ds:i + 1]) - 1,
                "above_e20_since": float(np.mean(c[i0:i + 1] > e20[i0:i + 1])),
                "min_since_first": np.nanmin(l[i0:i + 1]) / c[i0] - 1,
                }
                # ---- labels: bars > i ----
                after = slice(i + 1, ds + 96)
                hi_after = np.nanmax(h[after])
                k_hi = i + 1 + int(np.nanargmax(h[after]))
                lab = {
                    "day_max": dmax, "day_close": dclose,
                    "rocket": int(dmax >= 0.10 and dclose >= 0.6 * dmax),
                    "big": int(dmax >= 0.10),
                    "left_day": hi_after / ci - 1,
                    "left_24h": np.nanmax(h[i + 1:i + 97]) / ci - 1 if i + 97 <= N else np.nan,
                    "dd_before": np.nanmin(l[i + 1:k_hi + 1]) / ci - 1,
                    "eod": c[ds + 95] / ci - 1,
                    "top20": int((t0 + ds * STEP).strftime("%Y-%m-%d") and ((t0 + ds * STEP).strftime("%Y-%m-%d"), s) in win),
                }
                rec = {"sym": s, "day": (t0 + ds * STEP).strftime("%Y-%m-%d"),
                       "t": (t0 + (i + 1) * STEP).strftime("%Y-%m-%dT%H:%M"), **feat, **lab}
                fo.write(json.dumps({k: (None if isinstance(x, float) and not np.isfinite(x) else
                                         (round(float(x), 5) if isinstance(x, (float, np.floating)) else x))
                                     for k, x in rec.items()}) + "\n")
                rows += 1
        print("  %-12s done" % s, flush=True) if j % 20 == 0 else None
print("rows:", rows, "->", OUT)
