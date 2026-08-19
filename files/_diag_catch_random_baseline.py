"""What share of trends does a RANDOM alerter catch at the same budget?

Needed before "caught 95.7% of trends" can be called a result. A long trend is
easy to hit by accident: firing on 2% of bars at random gives roughly a 64%
chance of landing inside a 50-bar trend and 87% inside a 100-bar one, purely
from its length. Without this baseline the catch rate measures how long the
trends are, not whether the model found them.

    pyembed\python.exe files\_diag_catch_random_baseline.py
"""
from __future__ import annotations
import random, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import _backtest_continuation_signal as CS
import _diag_uptrend_population as UP

CUT_DAY = "2026-04-14"       # the holdout boundary used by the detector
RATE = 0.02                  # the same alert budget
DRAWS = 40


def main() -> None:
    wl = UP.watchlist()
    print("random-alert baseline, budget %.1f%% of bars, holdout from %s"
          % (RATE * 100, CUT_DAY))
    print("%8s%10s%12s%14s%22s" % (
        "run%", "trends", "med bars", "model caught", "random caught 95% band"))
    print("-" * 70)
    model_caught = {5.0: 3.8, 10.0: 28.1, 20.0: 95.7}
    for run in (5.0, 10.0, 20.0):
        spans, lens = [], []
        for sym in wl:
            bars = CS.bars(sym)
            if not bars:
                continue
            idx = {b[0]: i for i, b in enumerate(bars)}
            n_test = sum(1 for b in bars if b[0].strftime("%Y-%m-%d") >= CUT_DAY)
            for t in UP.trends_for(sym, run, 2.0, 4):
                st = UP.attr(t, "start_ts", "start", "low_ts")
                en = UP.attr(t, "end_ts", "end", "high_ts")
                if st is None or en is None:
                    continue
                if st.strftime("%Y-%m-%d") < CUT_DAY:
                    continue
                a, b = idx.get(st), idx.get(en)
                if a is None or b is None or b <= a:
                    continue
                spans.append((sym, b - a + 1, n_test))
                lens.append(b - a + 1)
        if not spans:
            print("%8.0f%10s" % (run, "none"))
            continue
        rng = random.Random(3)
        shares = []
        for _ in range(DRAWS):
            hit = 0
            for _sym, length, _n in spans:
                # P(no alert in `length` bars) under independent per-bar firing
                if any(rng.random() < RATE for _ in range(length)):
                    hit += 1
            shares.append(100.0 * hit / len(spans))
        shares.sort()
        lo = shares[int(0.025 * len(shares))]
        hi = shares[int(0.975 * len(shares))]
        lens.sort()
        print("%8.0f%10d%12d%13.1f%%%13.1f%% - %.1f%%" % (
            run, len(spans), lens[len(lens) // 2],
            model_caught[run], lo, hi))
    print()
    print("If the model's catch rate sits inside the random band, the trends")
    print("were caught by their own length and not by the model.")


if __name__ == "__main__":
    main()
