"""Do the "too late" daily_range caps cut the day's biggest moves?

WHY THIS EXISTS

On 2026-09-24 QNTUSDT rose +29% (70.69 -> 91.40) and, after one +5% trade, no
entry rule fired for 33 consecutive 15m bars while the price went 78.96 ->
89.48. A replay with the bot's own functions showed why: alignment was stopped
by `daily_range > 9%` on all 33 bars, and trend by volume/RSI and, later, by
`daily_range > 10%`. Candidates that no rule produces are never written to
bot_events.jsonl, so the gate-level backtests of this repo cannot see them. The
only way to measure these caps is to replay the strategy on raw klines.

METHOD

Every watchlist coin, every closed 15m bar in the long kline store, the five
live entry rules (trend/BUY, impulse_speed surge, impulse, alignment,
ema_cross -- retest and breakout are disabled in config), each evaluated three
times with the same features:

    CAP7     caps as coded, non-bull day (_effective_range_max = DAILY_RANGE_MAX)
    CAP10    bull-day caps (_effective_range_max = BULL_DAY_RANGE_MAX)
    NOCAP    every daily_range cap lifted (DAILY_RANGE_MAX, _effective_range_max,
             ALIGNMENT_RANGE_MAX, BREAKOUT_RANGE_MAX, EARLY_*_RANGE_MAX -> 1e9)

A bar is in band
    FIRE      a rule passes under CAP7                   -> the bot's own flow
    BULLBAND  passes only once the bull-day cap applies
    LATE      passes only with the caps lifted            -> what the caps remove

Outcome, measured from that bar's close (strictly later bars):
    peak_4h    highest high of the next 16 bars, %
    peak_day   highest high until the end of that UTC day, %  (the day's move left)
Rows are deduplicated to the FIRST bar per (symbol, UTC hour, band).

The goal metric view: on immutable top-20 winner-days (watchlist INTERSECT
global top-20, later-EOD klines), does the capped strategy fire at all that
day, and how many winner-days does only the LATE band reach?

SCOPE (stated, not hidden)

This is the rule layer only. Hour blocks, cooldowns, the ml / trend_quality /
chop gates, rotation and MAX_OPEN still apply live, so "a rule fired" is not "an
alert was sent". LATE rows are, by construction, later in the day's move than
FIRE rows, so the comparison is exactly the trade-off the cap was built to make:
less move left vs. a move the bot otherwise never touches.

VERDICT 2026-09-25: REFUTED as a lever. The caps stay. Do not re-test without new evidence.

Maximum period 2025-06-29 .. 2026-09-25, 101 watchlist coins, 43.6k 15m bars
each, 126 456 rows (first bar per symbol-hour-band):

                     n      4h peak  4h trough  4h close  trailed mean  trailed median
    FIRE        120 542      1.67%    -1.51%     -0.02%      +0.01%        -0.34%
    BULLBAND      3 135      3.18%    -2.66%     -0.15%      +0.04%        -0.67%
    LATE          2 744      4.83%    -3.92%     -0.11%      -0.01%        -1.07%

The LATE band's bigger peak is volatility, not quality: its trough grows just as
much, and inside the same daily_range bucket it matches or trails FIRE
(15-25%: trailed +0.27% vs +0.29%; 25%+: -0.62% vs -0.33%; 4h peak 5.00% vs
5.02%, 7.58% vs 8.76%). "Trailed" = ATR trail from the bar's close, trail_k 2.0
with the mode's floor, 96-bar cap (_backtest_weak_exit_above_breakeven.replay).

The goal view: of 829 immutable top-20 winner-days in the window, a rule already
fires under today's caps on 819 (98.8%); lifting every cap adds 4 (99.3%), and
the single winner-day reached ONLY by LATE had 1.2% of its move left. The rule
layer is not where winner-days are lost -- it fires on 38 212 coin-days at lift
1.00. LATE rows do concentrate on winners (21.1% vs a 2.1% base, 9.87x), which
is the old finding again: lateness says WHICH coin, and entering late does not
pay on a trailed outcome.

QNT 2026-09-24, which raised the question, has NO LATE row that day: lifting the
caps would not have produced a signal in its silent 13:45-21:45 UTC window
(volume, RSI and 1-bar impulse stop the rules there). A FIRE row at 07:30 UTC
(alignment) is a data artefact: vol_x 1.01 in the long store vs 0.94 on live
spot klines, against the non-bull floor of 1.00 -- is_bull was False all morning
(bot_events bull_day, eff_range_max 7.0), so live correctly stayed silent.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

FILES = Path("D:/Projects/claude_crypto_bot/files")
OUT = FILES.parent / ".runtime" / "backtests" / "lateness_rows.jsonl"

CAP_KEYS = ("DAILY_RANGE_MAX", "_effective_range_max", "ALIGNMENT_RANGE_MAX",
            "BREAKOUT_RANGE_MAX", "EARLY_1H_CONTINUATION_ENTRY_RANGE_MAX",
            "EARLY_15M_CONTINUATION_ENTRY_RANGE_MAX")


def _worker(sym: str) -> dict:
    sys.path.insert(0, str(FILES))
    import numpy as np
    import config
    import indicators as I
    import strategy as S
    import _backtest_trend_start_detector as TD
    t0 = time.time()
    bars = TD.bars_15m(sym)
    if len(bars) < 500:
        return {"sym": sym, "rows": [], "n_bars": len(bars), "secs": 0.0}
    o = np.array([b[1] for b in bars]); h = np.array([b[2] for b in bars])
    l = np.array([b[3] for b in bars]); c = np.array([b[4] for b in bars])
    v = np.array([b[5] for b in bars])
    feat = I.compute_features(o, h, l, c, v)
    days = [b[0].strftime("%Y-%m-%d") for b in bars]

    saved = {k: getattr(config, k, None) for k in CAP_KEYS}

    def setcaps(mode):
        for k in CAP_KEYS:
            if saved[k] is not None or k == "_effective_range_max":
                setattr(config, k, saved[k])
        if mode == "CAP7":
            config._effective_range_max = config.DAILY_RANGE_MAX
        elif mode == "CAP10":
            config._effective_range_max = float(getattr(config, "BULL_DAY_RANGE_MAX", 10.0))
        else:
            for k in CAP_KEYS:
                setattr(config, k, 1e9)

    def fires(i):
        if S.check_entry_conditions(feat, i, c, tf="15m")[0]:
            return "trend"
        if S.check_trend_surge_conditions(feat, i)[0]:
            return "impulse_speed"
        if S.check_impulse_conditions(feat, i)[0]:
            return "impulse"
        if S.check_alignment_conditions(feat, i, tf="15m")[0]:
            return "alignment"
        if S.check_ema_cross_conditions(feat, i)[0]:
            return "ema_cross"
        return None

    n = len(bars)
    res = {}
    for mode in ("CAP7", "CAP10", "NOCAP"):
        setcaps(mode)
        r = [None] * n
        for i in range(250, n - 1):
            r[i] = fires(i)
        res[mode] = r
    setcaps("CAP7")
    for k in CAP_KEYS:
        if saved[k] is not None:
            setattr(config, k, saved[k])

    # end-of-day index per bar, for the day's remaining move
    end_of_day = [0] * n
    j = n - 1
    for i in range(n - 1, -1, -1):
        if i == n - 1 or days[i] != days[i + 1]:
            j = i
        end_of_day[i] = j

    rows, seen = [], set()
    dr = feat["daily_range_pct"]
    for i in range(250, n - 1):
        a, b, cc = res["CAP7"][i], res["CAP10"][i], res["NOCAP"][i]
        if a:
            band, rule = "FIRE", a
        elif b:
            band, rule = "BULLBAND", b
        elif cc:
            band, rule = "LATE", cc
        else:
            continue
        hour = bars[i][0].strftime("%Y-%m-%d %H")
        key = (hour, band)
        if key in seen:
            continue
        seen.add(key)
        px = c[i]
        fut4 = h[i + 1: i + 17]
        futd = h[i + 1: end_of_day[i] + 1]
        rows.append({
            "sym": sym, "ts": bars[i][0].isoformat(), "day": days[i], "band": band,
            "rule": rule, "dr": round(float(dr[i]), 2) if np.isfinite(dr[i]) else None,
            "peak_4h": round((float(fut4.max()) / px - 1) * 100, 3) if len(fut4) == 16 else None,
            "peak_day": round((float(futd.max()) / px - 1) * 100, 3) if len(futd) else 0.0,
        })
    return {"sym": sym, "rows": rows, "n_bars": n, "secs": time.time() - t0}


def run(workers: int):
    wl = json.load(io.open(FILES / "watchlist.json", encoding="utf-8"))
    wl = wl if isinstance(wl, list) else wl.get("symbols", wl)
    t0 = time.time()
    n_rows = 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as fo, ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, s): s for s in wl}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as exc:
                print("  FAIL %s: %s" % (futs[f], exc), flush=True)
                continue
            for row in r["rows"]:
                fo.write(json.dumps(row) + "\n")
            n_rows += len(r["rows"])
            if k % 10 == 0 or k == len(wl):
                print("  %3d/%d  %-10s bars %6d  rows %5d  %4.0fs  (elapsed %.0fs)" % (
                    k, len(wl), r["sym"], r["n_bars"], len(r["rows"]), r["secs"], time.time() - t0), flush=True)
    print("done: %d rows -> %s  (%.0fs)" % (n_rows, OUT, time.time() - t0), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    run(ap.parse_args().workers)
