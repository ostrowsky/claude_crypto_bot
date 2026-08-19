"""Does the continuation ranking survive being turned into an exit policy?

`_backtest_continuation_signal.py` found the information exists: on the label
"+10% before -3% within 24h", a shape-only model reaches z=4.53 and 1.60x lift
on a 9.7% base. A ranking is not a policy, and this is the step that decides
whether it becomes one -- the same replay that killed every fixed trail width.

WHAT IS REPLAYED
    Hold from entry; at each hourly bar score P(continues) from the model and
    leave the moment it drops below a threshold. Terminal condition is a time
    stop at MAX_BARS, because a policy that can decline to ever exit is not a
    policy.

THE THREE THINGS THAT MAKE THIS HONEST

1.  **The model never sees the trades it is replayed on.** Trained on the
    earlier days, replayed only on the later ones. Anything else is asking the
    model to grade its own homework.

2.  **The comparison is against the actual exits OF THE SAME TRADES.** Not
    against all 4 174 -- the test window is a different market from the train
    window, and comparing a policy on late trades against actual exits on all
    trades would be the incomparable-windows error (TH-04) dressed up as a
    result.

3.  **A random control matched on holding time.** A threshold policy changes
    how long positions are held, and holding longer is itself a change with a
    P&L. The control exits at random bars drawn from the SAME holding-time
    distribution the policy produced, so what is left is the contribution of the
    RANKING rather than of the duration.

Every number is reported on every trade in the window, winners and losers, and
the population is stated with each. The predecessor of this file reported a
triple-capture result measured on winner-days alone.

    pyembed\\python.exe files\\_backtest_continuation_exit_policy.py

Spec: docs/specs/features/continuation-signal-spec.md
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import _backtest_continuation_signal as CS  # noqa: E402

# The shape-only set. Volatility is excluded on evidence, not on taste: alone it
# reproduces the AUC (z 3.20) and ranks nothing usable (lift 1.11x), while the
# set below reaches z 4.53 / 1.60x and beats the full model.
SHAPE_FEATS = ["ret_1", "ret_3", "ret_6", "ret_12", "rsi", "dist_ema20",
               "dist_ema50", "slope_6", "slope_12", "dd_from_run_max",
               "consec_up", "bars_since_entry", "pnl_since_entry", "vol_ratio"]

LABEL = (10.0, 3.0, 24)     # +10 / -3 within 24h -- the one label that passed
MAX_BARS = CS.MAX_BARS


# --------------------------------------------------------------- the trades ---

def actual_exits() -> dict:
    """(sym, entry_hour) -> realized pnl_pct from the event log.

    Paired by walking the log and matching each exit to the open entry for that
    symbol, which is how the bot itself tracks a position: there is at most one
    open position per symbol.
    """
    opens: dict = {}
    out: dict = {}
    with io.open(HERE / "bot_events.jsonl", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"event"' not in line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            ev, sym, ts = e.get("event"), e.get("sym"), e.get("ts")
            if ev not in ("entry", "exit") or not sym or not ts:
                continue
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except ValueError:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if ev == "entry":
                opens[sym] = dt.replace(minute=0, second=0, microsecond=0)
            else:
                key = opens.pop(sym, None)
                if key is not None and e.get("pnl_pct") is not None:
                    try:
                        out[(sym, key)] = float(e["pnl_pct"])
                    except (TypeError, ValueError):
                        pass
    return out


def build_paths(verbose: bool = True) -> list:
    """One entry per trade: every bar with its features, in order.

    Unlike the ranking experiment this keeps bars whose LABEL is undecided --
    a policy has to act on every bar it holds through, and dropping the
    undecided ones would silently let it skip the hard moments.
    """
    up_pct, dn_pct, horizon = LABEL
    paths = []
    ents = CS.entries()
    skipped: dict = defaultdict(int)
    for sym, edt in ents:
        b = CS.bars(sym)
        if not b or not (b[0][0] <= edt <= b[-1][0]):
            skipped["no_klines_or_outside_cache"] += 1
            continue
        idx = next((i for i, bar in enumerate(b) if bar[0] >= edt), None)
        if idx is None or idx < CS.WARMUP:
            skipped["no_warmup"] += 1
            continue
        entry_price = b[idx][4]
        if not entry_price:
            skipped["no_entry_price"] += 1
            continue
        run_max = entry_price
        steps = []
        for k in range(idx, min(idx + MAX_BARS, len(b))):
            run_max = max(run_max, b[k][2])
            f = CS.features(b[max(0, k - CS.WARMUP):k + 1], entry_price,
                            k - idx, run_max)
            future = b[k + 1:k + 1 + horizon]
            y = (CS.label(future, b[k][4], up_pct, dn_pct)
                 if len(future) >= horizon else None)
            steps.append({"feat": f, "y": y, "close": b[k][4],
                          "day": b[k][0].strftime("%Y-%m-%d"),
                          "pnl": (b[k][4] / entry_price - 1) * 100})
        if len(steps) < 2:
            skipped["too_short"] += 1
            continue
        paths.append({"sym": sym, "entry": edt, "entry_price": entry_price,
                      "day": b[idx][0].strftime("%Y-%m-%d"), "steps": steps})
    if verbose:
        print("entries %d -> replayable paths %d" % (len(ents), len(paths)))
        for k, v in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print("  skipped %-28s%d" % (k, v))
    return paths


# ---------------------------------------------------------------- the model ---

def train_on_early_days(paths: list, frac: float = 0.7):
    """Fit on the earlier days only, and return the cut so the replay obeys it."""
    from catboost import CatBoostClassifier
    days = sorted(set(p["day"] for p in paths))
    cut = days[int(len(days) * frac)]
    X, y = [], []
    for p in paths:
        if p["day"] >= cut:
            continue
        for s in p["steps"]:
            if s["y"] is None:
                continue
            # Embargo on the BAR, not just on the trade. A trade entered before
            # the cut runs on for up to 48 bars, so its later bars sit on the
            # same days as the test trades and share their market conditions.
            # Filtering only by entry day would let those through and quietly
            # make the holdout adjacent rather than separate.
            if s["day"] >= cut:
                continue
            X.append([s["feat"][f] for f in SHAPE_FEATS])
            y.append(s["y"])
    if len(set(y)) < 2:
        raise SystemExit("degenerate training labels")
    m = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05,
                           verbose=0, random_seed=0, allow_writing_files=False)
    m.fit(X, y)
    return m, cut, len(y)


def score_paths(model, paths: list) -> None:
    """Attach P(continues) to every step, in one batched call per path set."""
    flat, index = [], []
    for pi, p in enumerate(paths):
        for si, s in enumerate(p["steps"]):
            flat.append([s["feat"][f] for f in SHAPE_FEATS])
            index.append((pi, si))
    if not flat:
        return
    probs = model.predict_proba(flat)[:, 1]
    for (pi, si), pr in zip(index, probs):
        paths[pi]["steps"][si]["p"] = float(pr)


# --------------------------------------------------------------- the replay ---

def replay_threshold(path: dict, thr: float) -> tuple:
    """(pnl_pct, bars_held). Leave at the first bar scoring below `thr`.

    Bar 0 is never an exit: the entry decision was made by a different system,
    and letting this policy reverse it in the same hour would measure an entry
    filter rather than an exit rule.
    """
    steps = path["steps"]
    for i, s in enumerate(steps):
        if i == 0:
            continue
        if s.get("p", 1.0) < thr:
            return steps[i]["pnl"], i
    return steps[-1]["pnl"], len(steps) - 1


def replay_fixed_trail(path: dict, width_pct: float) -> tuple:
    """A trailing stop `width_pct` below the running max close since entry."""
    steps = path["steps"]
    peak = steps[0]["close"]
    for i, s in enumerate(steps):
        if i == 0:
            peak = max(peak, s["close"])
            continue
        if s["close"] <= peak * (1 - width_pct / 100):
            return s["pnl"], i
        peak = max(peak, s["close"])
    return steps[-1]["pnl"], len(steps) - 1


def replay_random(path: dict, hold: int, rng) -> tuple:
    """Exit at a given bar, ignoring every feature. The duration control."""
    steps = path["steps"]
    i = min(max(1, hold), len(steps) - 1)
    return steps[i]["pnl"], i


def summarise(name: str, results: list, actual: list) -> dict:
    """results/actual are aligned lists of pnl_pct over the SAME trades."""
    n = len(results)
    if not n:
        return {}
    beats = sum(1 for r, a in zip(results, actual) if r > a) / n
    return {"name": name, "n": n,
            "median": st.median(results), "mean": sum(results) / n,
            "win": sum(1 for r in results if r > 0) / n,
            "beats": beats,
            "total": sum(results)}


def row(d: dict) -> str:
    hold = d.get("median_hold")
    return ("%-26s%6d%10.3f%10.3f%9.1f%%%10.1f%%%12.1f%9s" % (
        d["name"], d["n"], d["median"], d["mean"], d["win"] * 100,
        d["beats"] * 100, d["total"],
        ("%.0f" % hold) if hold is not None else "-"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200,
                    help="bootstrap draws for the median-difference CI")
    args = ap.parse_args()

    print("=" * 101)
    print("CONTINUATION EXIT POLICY -- replayed on every trade, model blind to them")
    print("label +%.0f / -%.0f in %dh; shape-only features (volatility excluded on evidence)"
          % LABEL)
    print("=" * 101)

    paths = build_paths()
    model, cut, n_train = train_on_early_days(paths)
    test = [p for p in paths if p["day"] >= cut]
    print("time cut %s: %d training rows from earlier days, %d trades replayed"
          % (cut, n_train, len(test)))
    if len(test) < 100:
        print("too few test trades to conclude anything")
        return

    score_paths(model, test)
    real = actual_exits()

    # Only trades whose actual exit is known can be compared to one, and the
    # count is printed rather than quietly shrinking the denominator.
    keyed = [(p, real.get((p["sym"], p["entry"]))) for p in test]
    have = [(p, a) for p, a in keyed if a is not None]
    print("actual exit known for %d of %d replayed trades" % (len(have), len(test)))
    if len(have) < 100:
        print("too few matched trades")
        return

    actual = [a for _, a in have]
    trades = [p for p, _ in have]
    base = summarise("ACTUAL exits (bot today)", actual, actual)

    print()
    print("%-26s%6s%10s%10s%10s%11s%12s%9s" % (
        "policy", "n", "median", "mean", "win", "beats", "sum pnl", "hold"))
    print("-" * 101)
    print(row(base))

    rows = []
    for thr in (0.03, 0.05, 0.08, 0.12, 0.20):
        res = [replay_threshold(p, thr)[0] for p in trades]
        holds = [replay_threshold(p, thr)[1] for p in trades]
        d = summarise("continuation p<%.2f" % thr, res, actual)
        d["holds"] = holds
        d["median_hold"] = st.median(holds)
        rows.append(d)
        print(row(d))

    print("-" * 101)
    for w in (1.5, 3.0, 8.0):
        pairs = [replay_fixed_trail(p, w) for p in trades]
        d = summarise("fixed trail %.1f%%" % w, [x[0] for x in pairs], actual)
        d["median_hold"] = st.median([x[1] for x in pairs])
        print(row(d))

    res = [p["steps"][-1]["pnl"] for p in trades]
    print(row(summarise("hold to %d bars" % MAX_BARS, res, actual)))

    # The duration control. A threshold policy changes holding time, and holding
    # longer has a P&L of its own; matching the duration distribution isolates
    # what the RANKING contributed.
    print("-" * 101)
    best = max(rows, key=lambda d: d["median"]) if rows else None
    if best:
        rng = random.Random(11)
        pool = best["holds"]
        ctrl_medians = []
        for _ in range(50):
            res = [replay_random(p, rng.choice(pool), rng)[0] for p in trades]
            ctrl_medians.append(st.median(res))
        ctrl_medians.sort()
        lo = ctrl_medians[int(0.025 * len(ctrl_medians))]
        hi = ctrl_medians[int(0.975 * len(ctrl_medians))]
        print("duration control for %s" % best["name"])
        print("  random exits drawn from the SAME holding-time distribution:")
        print("  median pnl across 50 draws: [%.3f, %.3f]  vs policy %.3f"
              % (lo, hi, best["median"]))

    print()
    print("VERDICT")
    if not rows:
        print("  no policy produced results")
        return
    better = [d for d in rows if d["median"] > base["median"]
              and d["beats"] > 0.50]
    if not better:
        print("  NEGATIVE. No threshold beats the bot's actual exits on both")
        print("  median P&L and the head-to-head rate. The ranking is real (it")
        print("  cleared a multi-seed null by 4.5 sd) and still does not convert")
        print("  into a better exit -- ordering bars by P(continue) is not the")
        print("  same as knowing WHEN to leave, and the replay is where that")
        print("  difference shows up. Same outcome as every fixed trail width.")
    else:
        print("  %d threshold(s) beat actual exits on median AND head-to-head:"
              % len(better))
        for d in better:
            print("    %s: median %+.3f vs %+.3f, beats %.0f%%"
                  % (d["name"], d["median"], base["median"], d["beats"] * 100))
        if best and best["median"] <= hi:
            print("  BUT the duration control reaches the same median, so what")
            print("  improved is HOW LONG positions are held, not the ranking.")
            print("  That is a one-parameter change, not a model.")
        else:
            print("  The duration control does NOT reach it, so the ranking is")
            print("  contributing beyond holding time. Next: shadow, not deploy.")


if __name__ == "__main__":
    main()
