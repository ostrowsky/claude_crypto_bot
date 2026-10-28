"""N7 -- upper-bound check of intraday / weekly seasonality on BTC and ETH (seasonality-n7-1028-spec.md).

Train bars only (open time < 2026-01-01). For each of 74 windows: mean gross return of holding exactly that window,
bootstrap CI by day, against one round trip of costs (BNB 0.31%, taker 0.36%). Writes runtime/experiments/N7_BOUND.json.

  (from trader/) ../pyembed/python.exe -u _bound_seasonality_n7.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import history_store as HS        # noqa: E402
import trader_config as TC        # noqa: E402

TRAIN_FROM_MS = int(datetime(2023, 3, 1, tzinfo=timezone.utc).timestamp() * 1000)
TEST_FROM_MS = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
HOUR_MS, DAY_MS = 3_600_000, 86_400_000
COST_BNB, COST_TAKER = 2 * (0.00075 + 0.0008), 2 * (0.001 + 0.0008)
BOOT = 2000


def hourly(sym):
    a = HS.load(HS.path_of(sym))
    t = np.asarray(a["t"], dtype=np.int64)
    keep = (t >= TRAIN_FROM_MS) & (t < TEST_FROM_MS)
    t, o, c = t[keep], np.asarray(a["o"], float)[keep], np.asarray(a["c"], float)[keep]
    h = t // HOUR_MS
    starts = np.flatnonzero(np.r_[True, h[1:] != h[:-1]])
    ends = np.r_[starts[1:], len(t)]
    full = (ends - starts) == 4
    return h[starts][full], o[starts][full], c[ends - 1][full]


def daily(sym):
    a = HS.load(HS.path_of(sym))
    t = np.asarray(a["t"], dtype=np.int64)
    keep = (t >= TRAIN_FROM_MS) & (t < TEST_FROM_MS)
    t, o, c = t[keep], np.asarray(a["o"], float)[keep], np.asarray(a["c"], float)[keep]
    d = t // DAY_MS
    starts = np.flatnonzero(np.r_[True, d[1:] != d[:-1]])
    ends = np.r_[starts[1:], len(t)]
    full = (ends - starts) == 96
    return d[starts][full], o[starts][full], c[ends - 1][full]


def ci(x, seed=29):
    rnd = np.random.default_rng(seed)
    x = np.asarray(x, float)
    bs = np.sort(rnd.choice(x, size=(BOOT, len(x)), replace=True).mean(axis=1))
    return float(x.mean()), float(bs[int(0.025 * BOOT)]), float(bs[int(0.975 * BOOT)])


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        h, o, c = hourly(sym)
        r = c / o - 1
        hod = h % 24
        for k in range(24):
            m, lo, hi = ci(r[hod == k])
            rows.append({"sym": sym, "window": f"H{k:02d}", "n": int((hod == k).sum()), "mean": m, "lo95": lo, "hi95": hi})
        for k, hh in enumerate((23, 0, 7, 8, 15, 16)):          # hour before / after each funding time 00/08/16
            m, lo, hi = ci(r[hod == hh])
            rows.append({"sym": sym, "window": f"S{'before' if k % 2 == 0 else 'after'}_{(hh + (1 if k % 2 == 0 else 0)) % 24:02d}",
                         "n": int((hod == hh).sum()), "mean": m, "lo95": lo, "hi95": hi})
        d, o, c = daily(sym)
        r = c / o - 1
        dow = (d + 3) % 7                                       # 1970-01-01 was a Thursday -> 0 = Monday
        for k, name in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")):
            m, lo, hi = ci(r[dow == k])
            rows.append({"sym": sym, "window": f"D{name}", "n": int((dow == k).sum()), "mean": m, "lo95": lo, "hi95": hi})
    best = max(rows, key=lambda x: x["mean"])
    above = [x for x in rows if x["mean"] > COST_BNB]
    verdict = "rejected_by_bound" if not above else "bound_not_met_needs_full_test"
    for x in sorted(rows, key=lambda x: -x["mean"])[:10]:
        print("%-8s %-10s n %4d mean %+.4f%% [%+.4f, %+.4f]" % (x["sym"], x["window"], x["n"], x["mean"] * 100,
                                                              x["lo95"] * 100, x["hi95"] * 100))
    print("windows %d | best %s %s %+.4f%% | round trip BNB %.2f%% taker %.2f%% | above BNB cost %d | %s"
          % (len(rows), best["sym"], best["window"], best["mean"] * 100, COST_BNB * 100, COST_TAKER * 100, len(above), verdict))
    out = {"spec": "seasonality-n7-1028-spec.md", "window": ["2023-03-01", "2026-01-01"], "cost_bnb": COST_BNB,
           "cost_taker": COST_TAKER, "rows": rows, "best": best, "n_above_bnb_cost": len(above), "verdict": verdict,
           "utc": datetime.now(timezone.utc).isoformat()}
    p = TC.RUNTIME / "experiments" / "N7_BOUND.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
