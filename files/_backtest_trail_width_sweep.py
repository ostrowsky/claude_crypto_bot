"""How wide should the trailing stop be, per mode? Max-period sweep.

Why now. FILUSDT on 2026-09-01: entered 08:32 at 0.6939 — eight hours BEFORE the
run — stopped out at 12:48 at 0.7034 for +1.37%, and the coin then went to 0.7848,
+13.10% from the entry. The bot captured a tenth of the move it had correctly
identified early. The entry was right; the exit was not.

What this is NOT. `_backtest_premature_trail.py` already tested widening the trail
CONDITIONALLY on a learned proba and found it does not generalise (in-sample gain
+0.378 evaporating to ~0 out-of-sample). This asks the simpler question that one
did not: is the FIXED width per mode set correctly at all?

The measurement. For every real entry, replay the forward path bar by bar under a
trailing stop of `k` percent below the running peak, and report both numbers that
matter and disagree:

    realized   mean % per trade, the cost side -- a wider stop bleeds more on
               genuine reversals, and at entry the bot cannot tell which is which
    capture    realized move / available move, the goal side -- how much of what
               was actually there did the rule take

A rule that raises capture while sinking realized is buying the winners with the
losers, so both are printed side by side and neither is summarised away.

Baselines printed with every mode: HOLD (no stop, exit at the horizon) is the
ceiling on capture and the floor on discipline; the coin's own available move is
the denominator.

Honest limits, up front. This replays the bot's OWN entries, so it measures the
exit rule on the population the gates admit and says nothing about entries never
taken (TH-06). And a mode with few entries is reported but flagged -- a width
that wins on 20 trades has not been shown to win.
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
WIDTHS = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)


def load_entries(since: str, tail_mb: int = 400):
    sz = os.path.getsize(EVENTS)
    off = max(0, sz - tail_mb * 1_000_000)
    out = []
    with io.open(EVENTS, "rb") as fh:
        fh.seek(off)
        if off:
            fh.readline()
        for raw in fh:
            if b'"entry"' not in raw:
                continue
            try:
                e = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue
            if e.get("event") != "entry":
                continue
            ts = str(e.get("ts") or "")
            if ts < since:
                continue
            sym = e.get("sym")
            px = e.get("price")
            mode = e.get("mode") or e.get("signal_mode") or "?"
            if not sym or not isinstance(px, (int, float)) or px <= 0:
                continue
            try:
                d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            out.append((sym, d, float(px), str(mode), str(e.get("tf") or "1h")))
    return out


def replay(bars, entry_px, width_pct, max_bars):
    """Trailing stop `width_pct` below the running peak.

    The stop is checked against the bar LOW before the peak absorbs that bar's
    high, so a single bar cannot both set a new peak and be forgiven the
    drawdown it printed on the way -- the same ordering the ZigZag labeller uses.
    Returns (realized_pct, available_pct).
    """
    peak = entry_px
    avail = 0.0
    for i, b in enumerate(bars[:max_bars]):
        hi, lo, cl = b[2], b[3], b[4]
        stop = peak * (1.0 - width_pct / 100.0)
        if lo <= stop:
            avail = max(avail, (max(peak, hi) / entry_px - 1.0) * 100.0)
            return (stop / entry_px - 1.0) * 100.0, avail
        peak = max(peak, hi)
        avail = max(avail, (peak / entry_px - 1.0) * 100.0)
    if not bars[:max_bars]:
        return None, None
    return (bars[:max_bars][-1][4] / entry_px - 1.0) * 100.0, avail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-03-01")
    ap.add_argument("--max-bars", type=int, default=48,
                    help="horizon in 1h bars; the live max_hold is mode-dependent")
    ap.add_argument("--min-n", type=int, default=25)
    args = ap.parse_args()

    ents = load_entries(args.since)
    print("entries since %s: %d" % (args.since, len(ents)))

    by_mode = collections.defaultdict(list)
    for sym, when, px, mode, tf in ents:
        bars = TD.load_bars(sym, "1h")
        if len(bars) < 50:
            continue
        fut = [b for b in bars if b[0] > when]
        if len(fut) < 6:
            continue
        by_mode[mode].append((fut, px))
        by_mode["__ALL__"].append((fut, px))

    print("replayable: %d  over %d modes"
          % (len(by_mode.get("__ALL__") or []), len(by_mode) - 1))

    order = sorted((m for m in by_mode if m != "__ALL__"),
                   key=lambda m: -len(by_mode[m]))
    for mode in ["__ALL__"] + order:
        rows = by_mode[mode]
        if len(rows) < args.min_n:
            continue
        print()
        print("=" * 78)
        label = "ALL MODES" if mode == "__ALL__" else mode
        thin = "" if len(rows) >= 100 else "   [THIN — %d trades, not shown to win]" % len(rows)
        print("%s   n=%d%s" % (label, len(rows), thin))
        print("=" * 78)
        print("%-12s%12s%12s%12s%10s" % (
            "trail", "realized", "median", "available", "capture"))
        print("-" * 78)
        best = None
        for w in WIDTHS:
            real, avail = [], []
            for fut, px in rows:
                r, a = replay(fut, px, w, args.max_bars)
                if r is None:
                    continue
                real.append(r)
                avail.append(a)
            if not real:
                continue
            mr = sum(real) / len(real)
            ma = sum(avail) / len(avail)
            cap = mr / ma if ma > 0 else float("nan")
            srt = sorted(real)
            med = srt[len(srt) // 2]
            mark = ""
            if best is None or mr > best[1]:
                best = (w, mr)
            print("%-12s%11.3f%%%11.3f%%%11.3f%%%9.2f%s" % (
                "%.1f%%" % w, mr, med, ma, cap, mark))
        # HOLD ceiling: no stop at all over the same horizon
        real = []
        avail = []
        for fut, px in rows:
            seg = fut[:args.max_bars]
            if not seg:
                continue
            real.append((seg[-1][4] / px - 1.0) * 100.0)
            avail.append((max(b[2] for b in seg) / px - 1.0) * 100.0)
        if real:
            mr = sum(real) / len(real)
            ma = sum(avail) / len(avail)
            print("%-12s%11.3f%%%11.3f%%%11.3f%%%9.2f" % (
                "HOLD (none)", mr, sorted(real)[len(real) // 2], ma,
                mr / ma if ma > 0 else float("nan")))
        if best:
            print("-" * 78)
            print("best realized at %.1f%% trail" % best[0])

    print()
    print("READ THIS")
    print("  'available' is the mean peak the trade actually offered; 'capture' is")
    print("  realized/available. A width that lifts capture while sinking realized")
    print("  is paying for winners out of the losers -- both columns decide, not one.")
    print("  HOLD is the discipline floor, not a proposal: no stop at all.")
    print("  This replays the bot's OWN entries, so it says nothing about the")
    print("  trades the gates never admitted (TH-06).")


if __name__ == "__main__":
    main()
