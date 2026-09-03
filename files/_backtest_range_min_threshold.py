"""Would lowering the daily-range floor from 4% to 2% help or hurt?

Question came from TRXUSDT on 2026-09-03: a chart that looks like a clean
staircase but moved 1.38% in 24h, blocked 18 times by the range floor. The
operator asked what a 2% floor would admit.

The guard's own docstring records where 4.0 came from: "trend/15m range>=4.0 ->
+17.1pp precision (29.9% -> 47.0%)". That was a precision study. This one asks a
different question with the operator's stated goal in mind -- coins that give the
largest growth -- so the measure here is the size of the forward move, not the
hit rate.

METHOD. Every candidate the range guard rejected carries its `daily_range` in the
event log. Bucket those by range, replay the forward path, and read three things
per bucket:

    what the band did next          median and p75 peak over the next `--hours`
    how many candidates it adds     the cost side: alerts per day
    how it compares to the pool     every candidate in the same hours, so a
                                    rising market cannot be mistaken for skill

The band that decides is **2.0-4.0%**: exactly what a 4% -> 2% change would let
through. If it behaves like the pool, the change buys noise; if it behaves like
the >=4% candidates the bot already takes, the floor is set too high.

Dedup is by symbol-hour: a guard that re-fires every poll would otherwise weight
its own opinion by how often the loop happened to run.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _backtest_trend_start_detector as TD  # noqa: E402

EVENTS = HERE / "bot_events.jsonl"
BANDS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 6.0), (6.0, 1e9))


def load(since: str, tail_mb: int = 400):
    sz = os.path.getsize(EVENTS)
    off = max(0, sz - tail_mb * 1_000_000)
    out = []
    with io.open(EVENTS, "rb") as fh:
        fh.seek(off)
        if off:
            fh.readline()
        for raw in fh:
            if b'"blocked"' not in raw and b'"entry"' not in raw:
                continue
            try:
                e = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue
            ev = e.get("event")
            if ev not in ("blocked", "entry"):
                continue
            ts = str(e.get("ts") or "")
            if ts < since:
                continue
            sym, px = e.get("sym"), e.get("price")
            dr = e.get("daily_range")
            if not sym or not isinstance(px, (int, float)) or px <= 0:
                continue
            try:
                d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            rc = str(e.get("reason_code") or "")
            out.append((sym, d.replace(minute=0, second=0, microsecond=0),
                        float(px), ev, rc,
                        float(dr) if isinstance(dr, (int, float)) else None,
                        str(e.get("tf") or ""), str(e.get("mode") or "")))
    return out


def peak(sym, when, px, hours):
    bars = TD.load_bars(sym, "1h")
    if len(bars) < 50:
        return None
    fut = [b for b in bars if b[0] > when][:hours]
    if len(fut) < max(2, hours // 2):
        return None
    return (max(b[2] for b in fut) / px - 1.0) * 100.0


def q(v, p):
    v = sorted(v)
    return v[int(p * (len(v) - 1))] if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-06-01")
    ap.add_argument("--hours", type=int, default=8)
    args = ap.parse_args()

    rows = load(args.since)
    print("events since %s: %d" % (args.since, len(rows)))

    seen = {}
    for sym, hour, px, ev, rc, dr, tf, mode in rows:
        seen.setdefault((sym, hour, ev, rc), (sym, hour, px, dr, tf, mode))
    print("deduplicated by symbol-hour: %d" % len(seen))

    pool, entries = [], []
    by_band = collections.defaultdict(list)
    days = set()
    for (sym, hour, ev, rc), (s2, h2, px, dr, tf, mode) in seen.items():
        f = peak(sym, hour, px, args.hours)
        if f is None:
            continue
        pool.append(f)
        days.add(hour.date())
        if ev == "entry":
            entries.append(f)
            continue
        if rc != "mode_range_quality" or dr is None:
            continue
        for lo, hi in BANDS:
            if lo <= dr < hi:
                by_band[(lo, hi)].append(f)
                break

    if not pool:
        print("nothing resolvable")
        return
    nd = max(1, len(days))

    print()
    print("=" * 92)
    print("WHAT THE RANGE GUARD REJECTS, BY daily_range BAND")
    print("peak over the next %dh, %d days of data" % (args.hours, nd))
    print("=" * 92)
    print("%-16s%8s%8s%11s%11s%9s%9s" % (
        "daily_range", "n", "/day", "median", "p75", ">3%", ">5%"))
    print("-" * 92)
    print("%-16s%8d%8.1f%10.2f%%%10.2f%%%8.0f%%%8.0f%%" % (
        "POOL (all cand)", len(pool), len(pool) / nd, q(pool, .5), q(pool, .75),
        100.0 * sum(1 for x in pool if x > 3) / len(pool),
        100.0 * sum(1 for x in pool if x > 5) / len(pool)))
    if entries:
        print("%-16s%8d%8.1f%10.2f%%%10.2f%%%8.0f%%%8.0f%%" % (
            "ENTRIES TAKEN", len(entries), len(entries) / nd,
            q(entries, .5), q(entries, .75),
            100.0 * sum(1 for x in entries if x > 3) / len(entries),
            100.0 * sum(1 for x in entries if x > 5) / len(entries)))
    print("-" * 92)
    for lo, hi in BANDS:
        v = by_band.get((lo, hi)) or []
        if len(v) < 20:
            continue
        label = "%.0f-%.0f%%" % (lo, hi) if hi < 1e9 else ">=%.0f%%" % lo
        print("%-16s%8d%8.1f%10.2f%%%10.2f%%%8.0f%%%8.0f%%" % (
            label, len(v), len(v) / nd, q(v, .5), q(v, .75),
            100.0 * sum(1 for x in v if x > 3) / len(v),
            100.0 * sum(1 for x in v if x > 5) / len(v)))

    # The band a 4% -> 2% change would admit, read as one number.
    add = (by_band.get((2.0, 3.0)) or []) + (by_band.get((3.0, 4.0)) or [])
    print("-" * 92)
    if len(add) >= 20:
        print("%-16s%8d%8.1f%10.2f%%%10.2f%%%8.0f%%%8.0f%%   <- what 4%% -> 2%% admits" % (
            "2-4% COMBINED", len(add), len(add) / nd, q(add, .5), q(add, .75),
            100.0 * sum(1 for x in add if x > 3) / len(add),
            100.0 * sum(1 for x in add if x > 5) / len(add)))
    else:
        print("2-4%% band too thin to judge: n=%d" % len(add))

    print()
    print("READ THIS")
    print("  The band decides against TWO rows, not zero. Against POOL: does it")
    print("  beat a coin-blind policy in the same hours. Against ENTRIES TAKEN:")
    print("  is it as good as what the bot already buys. A band that matches the")
    print("  pool and trails the entries is extra alerts of below-average quality,")
    print("  which under the goal of catching the biggest movers is a cost.")
    print("  '/day' is the volume the change would add -- the other half of the")
    print("  trade, since every admitted candidate competes for MAX_OPEN slots.")


if __name__ == "__main__":
    main()
