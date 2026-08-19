"""What would the trend-start detector have said about the last day?

Asked after the live bot sat out a market-wide rally on 2026-08-19: every gate
rejected it -- ml_proba 0.04-0.19 against a 0.22 floor, the bandit preferring
SKIP, trend_quality on RSI > 76, and an explicit "late continuation" guard. The
question is whether the detector built today would have spoken instead.

HOW THIS STAYS HONEST
    The model is trained only on bars STRICTLY BEFORE the scoring window, so it
    has never seen the rally. The alert threshold is a budget over the WHOLE
    universe in that window -- the top 0.5% of all watchlist bars -- not a
    per-coin cutoff, because a threshold picked per coin after the fact is not
    a threshold, it is a story.

    The score is P(a +20% run starts here before a 2% give-back). The moves in
    question are 12-15%, so a coin can be correctly scored low and still look
    like it was missed: that is the target being what it is, not the model
    failing, and it is reported rather than smoothed over.

    pyembed\\python.exe files\\_diag_score_recent.py

Spec: docs/specs/features/trend-start-detector-spec.md
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import _backtest_trend_start_detector as TD
import _backtest_continuation_signal as CS
import _diag_uptrend_population as UP

WATCHED = ["RENDERUSDT", "TIAUSDT", "INJUSDT", "STRKUSDT", "SEIUSDT",
           "C98USDT", "BNBUSDT", "XRPUSDT", "BTCUSDT", "LINKUSDT", "SOLUSDT"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-before", default="2026-08-18")
    ap.add_argument("--score-from", default="2026-08-19")
    ap.add_argument("--run", type=float, default=20.0)
    ap.add_argument("--give-back", type=float, default=2.0)
    ap.add_argument("--budget", type=float, default=0.005)
    args = ap.parse_args()

    print("=" * 88)
    print("WHAT THE DETECTOR WOULD HAVE SAID -- trained before %s, scored from %s"
          % (args.train_before, args.score_from))
    print("score = P(a +%.0f%% run starts here before a %.0f%% give-back)"
          % (args.run, args.give_back))
    print("=" * 88)

    symbols = UP.watchlist()
    train, score = [], []
    for sym in symbols:
        bars = CS.bars(sym)
        if len(bars) < TD.WARMUP + 60:
            continue
        feats = TD.feature_table(bars)
        for i in range(TD.WARMUP, len(bars)):
            day = bars[i][0].strftime("%Y-%m-%d")
            row = dict(feats[i])
            row["_sym"] = sym
            row["_ts"] = bars[i][0]
            row["_close"] = bars[i][4]
            if day < args.train_before:
                y = TD.will_run(bars, i, args.run, args.give_back, 0)
                if y is None:
                    continue
                row["_y"] = y
                train.append(row)
            elif day >= args.score_from:
                score.append(row)

    print("train rows %d (base %.4f) | scored rows %d over %d symbols"
          % (len(train), sum(r["_y"] for r in train) / max(len(train), 1),
             len(score), len(set(r["_sym"] for r in score))))
    if len(train) < 10000 or not score:
        print("not enough data")
        return

    probs, _ = TD.fit(train, score)
    if probs is None:
        print("degenerate labels")
        return

    order = sorted(range(len(probs)), key=lambda i: -probs[i])
    k = max(1, int(len(order) * args.budget))
    thr = probs[order[k - 1]]
    print("universe alert threshold at the %.1f%% budget: p >= %.4f  (%d alerts)"
          % (args.budget * 100, thr, k))

    best = {}
    for r, p in zip(score, probs):
        s = r["_sym"]
        if s not in best or p > best[s][0]:
            best[s] = (p, r["_ts"], r["_close"])

    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
    fired = [s for s, (p, _, _) in ranked if p >= thr]

    print()
    print("TOP 15 OF THE WHOLE WATCHLIST in the scoring window")
    print("%-13s%9s%18s%12s%9s" % ("symbol", "best p", "at", "price", "alert"))
    print("-" * 88)
    for s, (p, ts, px) in ranked[:15]:
        print("%-13s%9.4f%18s%12.5g%9s" % (
            s, p, ts.strftime("%m-%d %H:%M"), px, "YES" if p >= thr else ""))

    print()
    print("THE COINS YOU SENT")
    print("%-13s%9s%18s%12s%12s%9s" % (
        "symbol", "best p", "at", "price then", "price now", "alert"))
    print("-" * 88)
    for s in WATCHED:
        if s not in best:
            print("%-13s%9s  (no klines / not scored)" % (s, "-"))
            continue
        p, ts, px = best[s]
        bars = CS.bars(s)
        now = bars[-1][4] if bars else float("nan")
        print("%-13s%9.4f%18s%12.5g%12.5g%9s" % (
            s, p, ts.strftime("%m-%d %H:%M"), px, now,
            "YES" if p >= thr else "no"))

    print()
    print("fired on %d of %d watchlist symbols; of the 11 you sent: %s"
          % (len(fired), len(best),
             ", ".join(s for s in WATCHED if s in fired) or "none"))
    print()
    print("READ THIS")
    print("  A low score is not automatically a miss. The target is a +%.0f%% run;"
          % args.run)
    print("  moves of 12-15%% can be scored low and be scored CORRECTLY. What the")
    print("  comparison shows is which coins the detector ranked highest while")
    print("  the live gates were rejecting all of them.")


if __name__ == "__main__":
    main()
