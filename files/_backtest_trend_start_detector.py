"""Can the START of a linear uptrend be detected, and which ones would we catch?

The operator's target, in their words: catch trends like INJ 18-19 Aug (+11.98%
over 37h, internal drawdown 1.11%) and TIA 19 Aug (+7.00% over 9h) at their
beginning, and leave just before they end. Top-20 means the day's largest MOVE,
not the day's close.

THE LABEL
    "A trend starts at the swing low" cannot be a label: that bar is only
    identifiable in hindsight, and a detector firing exactly on it is not a
    thing that can exist. So the label is stated forward from every bar:

        from here, does price gain RUN_PCT before giving back GIVE_BACK_PCT
        from its running peak, within HORIZON bars?

    That is the same definition of an uptrend the ZigZag labeler uses
    (swing_pct / max_drawdown_pct), evaluated at every bar instead of only at
    the extremes -- so a model trained on it is being asked exactly the
    question the operator asks, at exactly the moments a live bot could act.

THE FEATURES
    Chosen from what the operator's own two charts show, so the model is given
    the evidence a human reads off them rather than a generic dump: a long
    quiet base, price crossing MA25 then MA99, MACD histogram turning up from
    below zero, RSI climbing out of the 40s, and volume expanding.

THE POPULATION
    Every bar of every watchlist symbol with hourly klines -- NOT the bot's own
    entries. Everything measured against the bot's entries so far reflects the
    gates upstream of them; this target is upstream of the bot entirely.

WHAT IS SHOWN AT THE END
    Not just an AUC. The actual ZigZag trends in the holdout, which of them an
    alert would have caught, how far into each move the first alert fired, and
    how many alerts hit nothing -- because "which trends would we have caught"
    is the question, and a ranking metric does not answer it.

    pyembed\\python.exe files\\_backtest_trend_start_detector.py

Spec: docs/specs/features/trend-start-detector-spec.md
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from datetime import timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import _backtest_continuation_signal as CS
import _diag_uptrend_population as UP

RUN_PCT = 5.0            # the move that makes a trend worth catching
GIVE_BACK_PCT = 2.0      # give-back from the running peak that ends it
HORIZON = 48             # bars the run must complete within
WARMUP = 120             # bars needed before features are meaningful (MA99)


# ------------------------------------------------------------------- label ---

def will_run(bars: list, i: int, run_pct: float, give_back_pct: float,
             horizon: int):
    """1 if a qualifying uptrend begins at bar `i`. `None` if it cannot be told.

    Walks forward from the close of bar `i`: track the running peak; the
    attempt dies the moment price falls `give_back_pct` below that peak, and
    succeeds if it first reaches `+run_pct` above the entry close.

    `horizon <= 0` runs to resolution with no time limit, which is the exact
    ZigZag definition and the right setting once multi-week trends count as
    targets too: a 48h horizon trains the model on a one-day question while the
    catch report grades it against trends of any length, and that mismatch is
    the model's problem, not the target's.

    Returning `None` when the horizon runs out before either happens matters:
    those bars are genuinely undecided, and scoring them 0 would teach the
    model that "nothing yet" looks like "no trend", which is the more
    flattering error for anything that learns to predict stagnation.
    """
    entry = bars[i][4]
    if not entry:
        return None
    target = entry * (1 + run_pct / 100.0)
    peak = entry
    last = len(bars) if horizon <= 0 else min(i + 1 + horizon, len(bars))
    for k in range(i + 1, last):
        hi, lo = bars[k][2], bars[k][3]
        # The give-back is checked against the peak BEFORE this bar's high, and
        # the target against this bar's high. Order matters: checking the new
        # peak first would let a single bar both set a peak and be forgiven its
        # own drawdown, which quietly inflates the positive rate.
        if lo <= peak * (1 - give_back_pct / 100.0):
            return 0
        if hi >= target:
            return 1
        peak = max(peak, hi)
    return None


# ---------------------------------------------------------------- features ---

def _ema_series(values: list, n: int) -> list:
    k = 2.0 / (n + 1.0)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def _sma_series(values: list, n: int) -> list:
    out, run = [], 0.0
    for i, v in enumerate(values):
        run += v
        if i >= n:
            run -= values[i - n]
        out.append(run / min(i + 1, n))
    return out


def _rsi_series(closes: list, n: int = 14) -> list:
    out = [50.0]
    ag = al = 0.0
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        g, l = max(d, 0.0), max(-d, 0.0)
        if i <= n:
            ag = (ag * (i - 1) + g) / i
            al = (al * (i - 1) + l) / i
        else:
            ag = (ag * (n - 1) + g) / n
            al = (al * (n - 1) + l) / n
        out.append(100.0 if al <= 0 else 100.0 - 100.0 / (1.0 + ag / al))
    return out


def _atr_pct_series(bars: list, n: int = 14) -> list:
    out, prev, run = [], bars[0][4], 0.0
    trs = []
    for i, b in enumerate(bars):
        tr = max(b[2] - b[3], abs(b[2] - prev), abs(b[3] - prev))
        trs.append(tr)
        run += tr
        if i >= n:
            run -= trs[i - n]
        prev = b[4]
        out.append(run / min(i + 1, n) / b[4] * 100.0 if b[4] else 0.0)
    return out


FEATS = ["close_vs_ma25", "close_vs_ma99", "ma25_vs_ma99",
         "macd_hist", "macd_hist_d1", "bars_since_macd_cross",
         "rsi", "rsi_d6",
         "vol_ratio", "vol_ratio_3",
         "base_range_24", "base_range_48", "bars_since_high_50",
         "bars_in_base", "base_tightness", "vol_now_vs_base", "dist_base_high",
         "atr_pct", "ret_3", "ret_6", "ret_12", "dist_high_50"]


def feature_table(bars: list) -> list:
    """One dict per bar, using only information available at that bar."""
    closes = [b[4] for b in bars]
    vols = [b[5] for b in bars]
    ma25, ma99 = _sma_series(closes, 25), _sma_series(closes, 99)
    ema12, ema26 = _ema_series(closes, 12), _ema_series(closes, 26)
    macd = [a - b for a, b in zip(ema12, ema26)]
    signal = _ema_series(macd, 9)
    hist = [m - s for m, s in zip(macd, signal)]
    rsi = _rsi_series(closes)
    atr = _atr_pct_series(bars)
    vol20 = _sma_series(vols, 20)

    since_cross = 999
    out = []
    for i in range(len(bars)):
        if i > 0 and hist[i] > 0 >= hist[i - 1]:
            since_cross = 0
        else:
            since_cross = min(since_cross + 1, 999)
        lo24 = min(closes[max(0, i - 23):i + 1])
        hi24 = max(closes[max(0, i - 23):i + 1])
        lo48 = min(closes[max(0, i - 47):i + 1])
        hi48 = max(closes[max(0, i - 47):i + 1])
        w50 = closes[max(0, i - 49):i + 1]
        hi50 = max(w50)
        since_high = len(w50) - 1 - max(range(len(w50)), key=lambda j: w50[j])
        c = closes[i]
        # How long the coin has been standing still. Both charts the operator
        # showed have two to three days of flat range before the move, and that
        # duration was the one thing the feature set could not see: base_range
        # measures how TIGHT a fixed window was, never how LONG the quiet
        # lasted.
        bib = 0
        for j in range(i, max(-1, i - 200), -1):
            if abs(closes[j] / c - 1) > 0.03:
                break
            bib += 1
        base = closes[max(0, i - bib):i + 1] or [c]
        bhigh = max(base)
        bvol = vols[max(0, i - bib):i + 1] or [vols[i]]
        med_bvol = sorted(bvol)[len(bvol) // 2] or 1.0
        out.append({
            "bars_in_base": float(min(bib, 200)),
            "base_tightness": ((hi48 / lo48 - 1) * 100 / atr[i]) if atr[i] else 0.0,
            "vol_now_vs_base": vols[i] / med_bvol,
            "dist_base_high": (c / bhigh - 1) * 100 if bhigh else 0.0,
            "close_vs_ma25": (c / ma25[i] - 1) * 100 if ma25[i] else 0.0,
            "close_vs_ma99": (c / ma99[i] - 1) * 100 if ma99[i] else 0.0,
            "ma25_vs_ma99": (ma25[i] / ma99[i] - 1) * 100 if ma99[i] else 0.0,
            "macd_hist": hist[i] / c * 100 if c else 0.0,
            "macd_hist_d1": (hist[i] - hist[i - 1]) / c * 100 if i and c else 0.0,
            "bars_since_macd_cross": float(min(since_cross, 100)),
            "rsi": rsi[i],
            "rsi_d6": rsi[i] - rsi[max(0, i - 6)],
            "vol_ratio": vols[i] / vol20[i] if vol20[i] else 1.0,
            "vol_ratio_3": (sum(vols[max(0, i - 2):i + 1]) / 3) / vol20[i]
                            if vol20[i] else 1.0,
            # The quiet base the operator's charts both show before the move.
            "base_range_24": (hi24 / lo24 - 1) * 100 if lo24 else 0.0,
            "base_range_48": (hi48 / lo48 - 1) * 100 if lo48 else 0.0,
            "bars_since_high_50": float(since_high),
            "atr_pct": atr[i],
            "ret_3": (c / closes[i - 3] - 1) * 100 if i >= 3 else 0.0,
            "ret_6": (c / closes[i - 6] - 1) * 100 if i >= 6 else 0.0,
            "ret_12": (c / closes[i - 12] - 1) * 100 if i >= 12 else 0.0,
            "dist_high_50": (c / hi50 - 1) * 100 if hi50 else 0.0,
        })
    return out


# ------------------------------------------------------------------ build ---

def start_bars(sym: str, run_pct: float, give_back: float, window: int) -> dict:
    """bar-index -> 1 for the first `window` bars of each qualifying uptrend.

    The forward label ("from here, +X% before a give-back") is satisfied just as
    well by a bar in the MIDDLE of a move as by one at its start -- which is why
    the detector trained on it fires a median 40% into the move. It is not
    missing the start; it was never asked for it.

    This label asks for it directly: positive only inside the opening window of
    a trend, and every later bar of that same trend is a NEGATIVE. That is the
    part that does the work -- without it the model is free to keep scoring the
    middle and lose nothing.
    """
    bars = CS.bars(sym)
    idx = {b[0]: i for i, b in enumerate(bars)}
    out = {}
    for t in UP.trends_for(sym, run_pct, give_back, 4):
        st = UP.attr(t, "start_ts", "start", "low_ts")
        en = UP.attr(t, "end_ts", "end", "high_ts")
        if st is None or en is None:
            continue
        a, b = idx.get(st), idx.get(en)
        if a is None or b is None or b <= a:
            continue
        for i in range(a, b + 1):
            out[i] = 1 if i <= a + window else 0
    return out


def build(symbols: list, args) -> list:
    rows = []
    for si, sym in enumerate(symbols):
        bars = CS.bars(sym)
        if len(bars) < WARMUP + HORIZON + 50:
            continue
        feats = feature_table(bars)
        tail = HORIZON if args.horizon <= 0 else args.horizon
        starts = (start_bars(sym, args.run, args.give_back, args.start_window)
                  if args.label == "start" else None)
        for i in range(WARMUP, len(bars) - tail):
            if starts is not None:
                y = starts.get(i, 0)
            else:
                y = will_run(bars, i, args.run, args.give_back, args.horizon)
            if y is None:
                continue
            r = dict(feats[i])
            r["_y"] = y
            r["_sym"] = sym
            r["_i"] = i
            r["_ts"] = bars[i][0]
            r["_day"] = bars[i][0].strftime("%Y-%m-%d")
            r["_close"] = bars[i][4]
            rows.append(r)
        if (si + 1) % 25 == 0:
            print("  built %d/%d symbols, %d rows" % (si + 1, len(symbols), len(rows)))
    return rows


def fit(train, test, seed=0, shuffle=False):
    from catboost import CatBoostClassifier
    y = [r["_y"] for r in train]
    if shuffle:
        y = y[:]
        random.Random(seed).shuffle(y)
    if len(set(y)) < 2:
        return None, None
    m = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05,
                           verbose=0, random_seed=seed, allow_writing_files=False)
    m.fit([[r[f] for f in FEATS] for r in train], y)
    return list(m.predict_proba([[r[f] for f in FEATS] for r in test])[:, 1]), m


# ------------------------------------------------- which trends were caught ---

def catch_report(test, probs, thr, args, top_n_examples=25):
    """Fire an alert wherever p >= thr, then ask which real trends it caught.

    A ranking metric cannot answer "which trends would we have caught", so this
    matches alerts against the ZigZag trends themselves and reports how far into
    each move the first alert landed -- an alert at 80% of the way up is a
    catch by any hit-rate metric and worthless to act on.
    """
    alerts = defaultdict(list)
    for r, p in zip(test, probs):
        if p >= thr:
            alerts[r["_sym"]].append((r["_ts"], r["_close"], p))
    for s in alerts:
        alerts[s].sort()

    first_day = min(r["_day"] for r in test)
    caught, missed, rows = 0, 0, []
    for sym in sorted(set(r["_sym"] for r in test)):
        bars = CS.bars(sym)
        for t in UP.trends_for(sym, args.run, args.give_back, 4):
            st = UP.attr(t, "start_ts", "start", "low_ts")
            en = UP.attr(t, "end_ts", "end", "high_ts")
            gain = UP.attr(t, "gain_pct", "gain")
            if st is None or en is None or gain is None:
                continue
            if st.strftime("%Y-%m-%d") < first_day:
                continue
            hit = next(((ts, px, p) for ts, px, p in alerts.get(sym, [])
                        if st <= ts <= en), None)
            if hit is None:
                missed += 1
                continue
            caught += 1
            ts, px, p = hit
            peak = max((b[2] for b in bars if st <= b[0] <= en), default=px)
            ahead = (peak / px - 1) * 100 if px else 0.0
            into = (1 - ahead / gain) * 100 if gain else 0.0
            rows.append({"sym": sym, "start": st, "end": en, "gain": gain,
                         "alert": ts, "ahead": ahead, "into": into, "p": p})

    n_alerts = sum(len(v) for v in alerts.values())
    inside = len(rows)
    return {"caught": caught, "missed": missed, "rows": rows,
            "n_alerts": n_alerts,
            "false_rate": 1 - (inside / n_alerts) if n_alerts else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=float, default=RUN_PCT)
    ap.add_argument("--give-back", type=float, default=GIVE_BACK_PCT)
    ap.add_argument("--horizon", type=int, default=0,
                    help="bars the run must complete within; 0 = to resolution")
    ap.add_argument("--max-rsi", type=float, default=None,
                    help="forbid alerts on bars whose RSI is already above this. "
                         "Measures the PRICE of earliness: the budget can only be "
                         "spent before the move is visible in momentum.")
    ap.add_argument("--label", choices=("forward", "start"), default="forward",
                    help="'forward' = there is a move ahead; 'start' = this bar "
                         "is in the opening hours of one")
    ap.add_argument("--start-window", type=int, default=6,
                    help="hours after the trend low that still count as its start")
    ap.add_argument("--stability", action="store_true",
                    help="repeat the whole evaluation at several time cuts")
    ap.add_argument("--cuts", default="0.55,0.65,0.70,0.80",
                    help="train fractions to repeat the whole evaluation at; "
                         "one split is one observation")
    ap.add_argument("--alert-rate", type=float, default=0.02,
                    help="share of bars that fire, i.e. the alert budget")
    args = ap.parse_args()

    print("=" * 96)
    print("TREND-START DETECTOR -- can the beginning of a linear uptrend be seen?")
    if args.label == "start":
        print("label: this bar is within %dh of the start of a +%.1f%% trend "
              "(its later bars are NEGATIVES)" % (args.start_window, args.run))
    else:
        print("label: from this bar, +%.1f%% before giving back %.1f%% from the peak%s"
              % (args.run, args.give_back,
                 " (no time limit)" if args.horizon <= 0 else
                 ", within %dh" % args.horizon))
    print("population: every bar of every watchlist symbol -- NOT the bot's entries")
    print("=" * 96)

    symbols = UP.watchlist()
    rows = build(symbols, args)
    if len(rows) < 20000:
        print("only %d rows" % len(rows))
        return

    base_all = sum(r["_y"] for r in rows) / len(rows)
    days = sorted(set(r["_day"] for r in rows))

    if args.stability:
        # One split is one observation. A result that only exists at 70/30 is a
        # property of that boundary, not of the market.
        print()
        print("STABILITY ACROSS TIME CUTS -- run target +%.0f%%" % args.run)
        print("%-12s%8s%9s%9s%9s%8s%10s%10s" % (
            "cut", "train", "test", "base", "AUC", "z", "trends", "caught%"))
        print("-" * 78)
        for frac in [float(x) for x in args.cuts.split(",")]:
            c = days[int(len(days) * frac)]
            tr = [r for r in rows if r["_day"] < c]
            te = [r for r in rows if r["_day"] >= c]
            if not tr or not te:
                continue
            pp, _ = fit(tr, te)
            if pp is None:
                continue
            yy = [r["_y"] for r in te]
            aa = CS.auc(yy, pp)
            nn = []
            for sd in range(3):
                qq, _ = fit(tr, te, seed=200 + sd, shuffle=True)
                if qq:
                    nn.append(CS.auc(yy, qq))
            nmean = sum(nn) / len(nn) if nn else float("nan")
            nsdev = ((sum((v - nmean) ** 2 for v in nn) / max(len(nn) - 1, 1)) ** 0.5
                     if len(nn) > 1 else float("nan"))
            oo = sorted(range(len(pp)), key=lambda i: -pp[i])
            kk = max(1, int(len(oo) * args.alert_rate))
            rr = catch_report(te, pp, pp[oo[kk - 1]], args)
            tt = rr["caught"] + rr["missed"]
            print("%-12s%8d%9d%9.4f%9.4f%8.2f%10d%9.1f%%" % (
                c, len(tr), len(te), sum(yy) / len(yy), aa,
                (aa - nmean) / nsdev if nsdev else float("nan"),
                tt, 100.0 * rr["caught"] / tt if tt else 0))
        return
    cut = days[int(len(days) * 0.7)]
    train = [r for r in rows if r["_day"] < cut]
    test = [r for r in rows if r["_day"] >= cut]
    print()
    print("rows %d  (%d symbols, %d days)  base rate %.4f"
          % (len(rows), len(set(r["_sym"] for r in rows)), len(days), base_all))
    print("time cut %s: train %d / test %d" % (cut, len(train), len(test)))

    probs, model = fit(train, test)
    if probs is None:
        print("degenerate labels")
        return
    yte = [r["_y"] for r in test]
    base = sum(yte) / len(yte)
    a = CS.auc(yte, probs)

    nulls = []
    for s in range(5):
        ps, _ = fit(train, test, seed=100 + s, shuffle=True)
        if ps:
            nulls.append(CS.auc(yte, ps))
    nm = sum(nulls) / len(nulls) if nulls else float("nan")
    nsd = (sum((v - nm) ** 2 for v in nulls) / max(len(nulls) - 1, 1)) ** 0.5 \
        if len(nulls) > 1 else float("nan")

    print()
    print("test base rate      %.4f" % base)
    print("AUC                 %.4f" % a)
    print("shuffled-label null %.4f +- %.4f   -> z = %.2f"
          % (nm, nsd, (a - nm) / nsd if nsd else float("nan")))

    print()
    print("%-10s%12s%10s%10s" % ("top slice", "precision", "base", "lift"))
    for frac in (0.005, 0.01, 0.02, 0.05, 0.10):
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        k = max(1, int(len(order) * frac))
        prec = sum(test[i]["_y"] for i in order[:k]) / k
        print("%-10s%12.4f%10.4f%10.2fx" % (
            "%.1f%%" % (frac * 100), prec, base, prec / base if base else 0))

    if args.max_rsi is not None:
        # Suppress rather than re-rank: an alert on a bar at RSI 80 is a
        # confirmation whatever its score, and letting it keep the budget slot
        # would hide the cost this flag exists to measure.
        probs = [0.0 if r["rsi"] >= args.max_rsi else p
                 for r, p in zip(test, probs)]
        alive = sum(1 for p in probs if p > 0)
        print()
        print("EARLINESS CONSTRAINT: alerts forbidden at RSI >= %.0f "
              "-- %d of %d test bars remain eligible (%.1f%%)"
              % (args.max_rsi, alive, len(probs), 100.0 * alive / len(probs)))
    order = sorted(range(len(probs)), key=lambda i: -probs[i])
    k = max(1, int(len(order) * args.alert_rate))
    thr = probs[order[k - 1]]
    print()
    print("=" * 96)
    print("WHICH TRENDS WOULD HAVE BEEN CAUGHT  (alert budget %.1f%% of bars, "
          "threshold p >= %.3f)" % (args.alert_rate * 100, thr))
    print("=" * 96)
    rep = catch_report(test, probs, thr, args)
    tot = rep["caught"] + rep["missed"]

    # Check 2 -- does a tighter budget keep the catches? If 0.5% of bars catch
    # nearly as many trends, the false-alert load falls fourfold for free.
    print()
    print("%-10s%9s%10s%12s%14s" % (
        "budget", "alerts", "caught", "caught %", "still ahead"))
    print("-" * 60)
    for rate in (0.005, 0.01, 0.02, 0.05):
        kk = max(1, int(len(order) * rate))
        t2 = probs[order[kk - 1]]
        r2 = catch_report(test, probs, t2, args)
        tt = r2["caught"] + r2["missed"]
        ah = sorted(x["ahead"] for x in r2["rows"]) or [float("nan")]
        print("%-10s%9d%10d%11.1f%%%13.2f%%" % (
            "%.1f%%" % (rate * 100), r2["n_alerts"], r2["caught"],
            100.0 * r2["caught"] / tt if tt else 0, ah[len(ah) // 2]))

    # Check 3 -- how long do the caught trends run? An unbounded +20% target can
    # last weeks, and "exit before it ends" is a different problem at that scale.
    durs = sorted((x["end"] - x["start"]).total_seconds() / 3600.0
                  for x in rep["rows"])
    if durs:
        print()
        print("duration of CAUGHT trends (hours): p25 %.0f  median %.0f  "
              "p75 %.0f  p90 %.0f  max %.0f"
              % (durs[len(durs) // 4], durs[len(durs) // 2],
                 durs[3 * len(durs) // 4], durs[int(0.9 * len(durs))], durs[-1]))
        wk = sum(1 for d in durs if d > 168) / len(durs)
        print("  share running longer than a week: %.0f%%" % (wk * 100))
    print("trends in holdout        %d" % tot)
    print("caught (alert inside)    %d  (%.1f%%)"
          % (rep["caught"], 100.0 * rep["caught"] / tot if tot else 0))
    print("missed                   %d" % rep["missed"])
    print("alerts fired             %d, of which outside any trend %.1f%%"
          % (rep["n_alerts"], 100.0 * rep["false_rate"]))

    rows_r = sorted(rep["rows"], key=lambda r: -r["ahead"])
    if rows_r:
        aheads = sorted(r["ahead"] for r in rows_r)
        intos = sorted(r["into"] for r in rows_r)
        print()
        print("still ahead at the first alert: median %.2f%%   p25 %.2f%%   p75 %.2f%%"
              % (aheads[len(aheads) // 2], aheads[len(aheads) // 4],
                 aheads[3 * len(aheads) // 4]))
        print("how far INTO the move the alert fired: median %.0f%%"
              % intos[len(intos) // 2])
        print()
        print("%-12s%-18s%9s%9s%9s%7s" % (
            "symbol", "alert", "trend%", "ahead%", "into%", "p"))
        print("-" * 96)
        for r in rows_r[:25]:
            print("%-12s%-18s%9.2f%9.2f%9.0f%7.3f" % (
                r["sym"], r["alert"].strftime("%m-%d %H:%M"),
                r["gain"], r["ahead"], r["into"], r["p"]))

    print()
    print("READ THIS")
    print("  'ahead%' is what remained AFTER the alert -- the only part that")
    print("  could be traded. 'into%' is how far up the move had already gone;")
    print("  a high into% with a hit is a confirmation, not a catch.")


if __name__ == "__main__":
    main()
