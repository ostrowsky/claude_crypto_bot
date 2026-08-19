"""Median forward return by bar index, across every entry the bot made.

Asked because the exit-policy replay found that leaving almost immediately beats
the bot's actual exits AND every trailing width AND holding to 48 bars. If that
is right, the position is losing from the first hour, and no exit rule can fix
a population whose drift is negative from the start -- the loss would be at
entry, not at exit.

No features, no model, no split: this is arithmetic on the price path, reported
on every trade rather than on winner-days.
"""
from __future__ import annotations
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import _backtest_continuation_signal as CS

MAX_BARS = 48


def main() -> None:
    rows = {k: [] for k in range(MAX_BARS)}
    n_trades = 0
    for sym, edt in CS.entries():
        b = CS.bars(sym)
        if not b or not (b[0][0] <= edt <= b[-1][0]):
            continue
        idx = next((i for i, bar in enumerate(b) if bar[0] >= edt), None)
        if idx is None:
            continue
        entry = b[idx][4]
        if not entry:
            continue
        n_trades += 1
        for k in range(idx, min(idx + MAX_BARS, len(b))):
            rows[k - idx].append((b[k][4] / entry - 1) * 100)

    print("forward return from entry, ALL %d trades" % n_trades)
    print("%6s%8s%10s%10s%9s" % ("bar", "n", "median", "mean", "win%"))
    for k in range(MAX_BARS):
        v = rows[k]
        if len(v) < 50:
            continue
        if k not in (0, 1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 47):
            continue
        print("%6d%8d%10.3f%10.3f%9.1f" % (
            k, len(v), st.median(v), sum(v) / len(v),
            100.0 * sum(1 for x in v if x > 0) / len(v)))

    first = rows[1]
    print()
    print("At bar 1 the median trade is already %+.3f%% (n=%d)."
          % (st.median(first), len(first)))
    print("A median that starts negative and stays negative is an ENTRY")
    print("problem: an exit rule chooses WHEN to realise the path, it cannot")
    print("change the path.")


if __name__ == "__main__":
    main()
