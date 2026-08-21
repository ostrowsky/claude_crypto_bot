"""Which gate is over-blocking now? The section-7 test, on the max period.

Context. The ml_zone gate was fixed on 2026-08-20 (segment routing off, floor
0.10) and stopped blocking: since the restart it appears in ZERO of 3909
rejections. Two gates now account for 88% of them:

    bandit skip          2720   70%
    trend_quality         688   18%
    trend_chop            383   10%
    mode_range_quality    118    3%

and the portfolio is still empty on a day when ENA ran +43%, XTZ +30%, CELO +28%.
ENA, XRP and ORDI were each rejected by trend_quality alone; CRV, BCH and XTZ by
the bandit alone.

The test. CLAUDE.md section 7 states the condition plainly: a filter that blocks
the eventual winners is broken. So for every gate, take the candidates it
rejected and measure what those coins actually did next. A gate whose rejects go
nowhere is doing its job; a gate whose rejects run is costing the operator the
moves the bot exists to catch.

Baseline. "Rejects rose 40% of the time" means nothing alone — in a rising market
everything rises. Every gate is therefore read against the SAME-HOUR behaviour of
the whole candidate pool, so the number reported is lift over what a coin-blind
policy would have got at that moment.

Horizon. Forward move is the peak over the next `--hours` from the blocked price,
not the close: the target is the day's largest MOVE, and a run that is given back
was still a run the bot should have caught.
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


def load_events(since: str, tail_mb: int = 400):
    """Blocked + entry events with a usable price, newest tail only."""
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
            if not sym or not isinstance(px, (int, float)) or px <= 0:
                continue
            try:
                d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            rc = str(e.get("reason_code") or "")
            if ev == "blocked" and (not rc or rc == "None"):
                rc = str(e.get("reason") or "?").split(":")[0].strip()[:24]
            out.append((sym, d.replace(minute=0, second=0, microsecond=0),
                        float(px), ev if ev == "entry" else rc))
    return out


def forward_peak(sym, when, px, hours):
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
    ap.add_argument("--min-n", type=int, default=100)
    args = ap.parse_args()

    ev = load_events(args.since)
    print("events since %s with a price: %d" % (args.since, len(ev)))

    # One forward measurement per (symbol, hour) so a gate that re-fires every
    # poll cannot weight itself up: 98 rejections of CRV in one day is one
    # opinion about CRV, not 98.
    seen = {}
    for sym, hour, px, tag in ev:
        seen.setdefault((sym, hour, tag), (sym, hour, px))
    print("deduplicated to one row per symbol-hour-gate: %d" % len(seen))

    by = collections.defaultdict(list)
    pool_by_hour = collections.defaultdict(list)
    for (sym, hour, tag), (s2, h2, px) in seen.items():
        f = forward_peak(sym, hour, px, args.hours)
        if f is None:
            continue
        by[tag].append(f)
        pool_by_hour[hour].append(f)

    # Coin-blind baseline: what the average candidate did in the same hours.
    pool = [x for v in pool_by_hour.values() for x in v]
    if not pool:
        print("no resolvable rows")
        return
    base_med = q(pool, .5)
    base3 = 100.0 * sum(1 for x in pool if x > 3) / len(pool)
    base5 = 100.0 * sum(1 for x in pool if x > 5) / len(pool)

    print()
    print("=" * 86)
    print("WHAT HAPPENED NEXT, per gate — peak over the following %dh" % args.hours)
    print("=" * 86)
    print("%-24s%8s%11s%11s%10s%10s" % (
        "gate / outcome", "n", "median", "p75", ">3%", ">5%"))
    print("-" * 86)
    print("%-24s%8d%10.2f%%%10.2f%%%9.0f%%%9.0f%%" % (
        "ALL CANDIDATES (base)", len(pool), base_med, q(pool, .75), base3, base5))
    print("-" * 86)
    for tag, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(v) < args.min_n:
            continue
        s3 = 100.0 * sum(1 for x in v if x > 3) / len(v)
        s5 = 100.0 * sum(1 for x in v if x > 5) / len(v)
        flag = ""
        if tag != "entry" and s3 > base3 * 1.15:
            flag = "  <-- rejects beat the pool"
        print("%-24s%8d%10.2f%%%10.2f%%%9.0f%%%9.0f%%%s" % (
            tag[:23], len(v), q(v, .5), q(v, .75), s3, s5, flag))

    print()
    print("READ THIS")
    print("  The pool row is the comparison, not zero. A gate whose rejects match")
    print("  the pool is filtering nothing; a gate whose rejects BEAT the pool is")
    print("  removing the better half of the candidates, which is the section-7")
    print("  failure. 'entry' is what the bot actually took -- if a gate's rejects")
    print("  outrun it, the gate is on the wrong side of its own pipeline.")


if __name__ == "__main__":
    main()
