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
import io
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


HISTORY = Path(__file__).resolve().parent.parent / "history"
_BARS15: dict = {}


def bars_15m(sym: str) -> list:
    """419 days of 15m bars, the same window the 1h experiments used.

    Kept separate from CS.bars rather than folded into it: the 1h results are
    already committed against that loader, and silently changing what it returns
    would make every earlier number irreproducible.
    """
    if sym in _BARS15:
        return _BARS15[sym]
    out = []
    fp = HISTORY / ("%s_15m_419d.csv" % sym)
    if fp.exists():
        import csv as _csv
        from datetime import datetime as _dt
        with io.open(fp, encoding="utf-8") as fh:
            for r in _csv.DictReader(fh):
                try:
                    out.append((_dt.fromisoformat(r["ts"]), float(r["open"]),
                                float(r["high"]), float(r["low"]),
                                float(r["close"]), float(r.get("volume") or 0)))
                except (KeyError, ValueError, TypeError):
                    continue
    _BARS15[sym] = out
    return out


def load_bars(sym: str, tf: str) -> list:
    return bars_15m(sym) if tf == "15m" else CS.bars(sym)


def trends_tf(sym: str, run_pct: float, give_back: float, tf: str) -> list:
    """The trends to be caught -- ALWAYS defined on the grid `tf` names.

    Measured across the watchlist, the ZigZag population is not merely
    resampled by a finer grid, it is replaced: at a 2% give-back the 15m grid
    finds 99 trends where 1h finds 342, because a give-back is now detected on
    intra-hour lows the hourly bar hides, so runs are cut before reaching +20%.
    The survivors are the unusually smooth ones (median 10.8h vs 7.0h). Widening
    the give-back to 3% restores the COUNT (351) and still not the population --
    those trends run a median 18.8h.

    No parameter makes the two identical, so the experiment does not try. The
    TARGET is held fixed on the 1h grid (see --trend-grid) and only the
    detector's input resolution varies, which is the question actually being
    asked (TH-04).

    min_duration is scaled with the timeframe: `min_bars=4` means "at least four
    hours" on 1h, and leaving it at 4 bars on 15m would admit one-hour trends.
    """
    b = load_bars(sym, tf)
    if len(b) < 100:
        return []
    rows = [{"ts": r[0], "open": r[1], "high": r[2], "low": r[3],
             "close": r[4], "volume": r[5]} for r in b]
    try:
        from zigzag_labeler import detect_uptrends
        return detect_uptrends(rows, symbol=sym, swing_pct=run_pct,
                               max_drawdown_pct=give_back,
                               min_duration_bars=4 * (4 if tf == "15m" else 1))
    except Exception:
        return []


def feature_table(bars: list, sc: int = 1) -> list:
    """One dict per bar, using only information available at that bar.

    `sc` multiplies every lookback. All windows here are counted in BARS, so on
    15m bars an unscaled MA99 spans one day where the 1h version spanned four.
    sc=4 restores the physical time span, isolating the resolution change from
    a change in what the features look at.
    """
    closes = [b[4] for b in bars]
    vols = [b[5] for b in bars]
    ma25, ma99 = _sma_series(closes, 25*sc), _sma_series(closes, 99*sc)
    ema12, ema26 = _ema_series(closes, 12*sc), _ema_series(closes, 26*sc)
    macd = [a - b for a, b in zip(ema12, ema26)]
    signal = _ema_series(macd, 9*sc)
    hist = [m - s for m, s in zip(macd, signal)]
    rsi = _rsi_series(closes, 14*sc)
    atr = _atr_pct_series(bars, 14*sc)
    vol20 = _sma_series(vols, 20*sc)

    since_cross = 999
    out = []
    for i in range(len(bars)):
        if i > 0 and hist[i] > 0 >= hist[i - 1]:
            since_cross = 0
        else:
            since_cross = min(since_cross + 1, 999)
        lo24 = min(closes[max(0, i - (24*sc-1)):i + 1])
        hi24 = max(closes[max(0, i - (24*sc-1)):i + 1])
        lo48 = min(closes[max(0, i - (48*sc-1)):i + 1])
        hi48 = max(closes[max(0, i - (48*sc-1)):i + 1])
        w50 = closes[max(0, i - (50*sc-1)):i + 1]
        hi50 = max(w50)
        since_high = len(w50) - 1 - max(range(len(w50)), key=lambda j: w50[j])
        c = closes[i]
        # How long the coin has been standing still. Both charts the operator
        # showed have two to three days of flat range before the move, and that
        # duration was the one thing the feature set could not see: base_range
        # measures how TIGHT a fixed window was, never how LONG the quiet
        # lasted.
        bib = 0
        for j in range(i, max(-1, i - 200*sc), -1):
            if abs(closes[j] / c - 1) > 0.03:
                break
            bib += 1
        base = closes[max(0, i - bib):i + 1] or [c]
        bhigh = max(base)
        bvol = vols[max(0, i - bib):i + 1] or [vols[i]]
        med_bvol = sorted(bvol)[len(bvol) // 2] or 1.0
        out.append({
            "bars_in_base": float(min(bib, 200*sc)),
            "base_tightness": ((hi48 / lo48 - 1) * 100 / atr[i]) if atr[i] else 0.0,
            "vol_now_vs_base": vols[i] / med_bvol,
            "dist_base_high": (c / bhigh - 1) * 100 if bhigh else 0.0,
            "close_vs_ma25": (c / ma25[i] - 1) * 100 if ma25[i] else 0.0,
            "close_vs_ma99": (c / ma99[i] - 1) * 100 if ma99[i] else 0.0,
            "ma25_vs_ma99": (ma25[i] / ma99[i] - 1) * 100 if ma99[i] else 0.0,
            "macd_hist": hist[i] / c * 100 if c else 0.0,
            "macd_hist_d1": (hist[i] - hist[i - 1]) / c * 100 if i and c else 0.0,
            "bars_since_macd_cross": float(min(since_cross, 100*sc)),
            "rsi": rsi[i],
            "rsi_d6": rsi[i] - rsi[max(0, i - 6*sc)],
            "vol_ratio": vols[i] / vol20[i] if vol20[i] else 1.0,
            "vol_ratio_3": (sum(vols[max(0, i - (3*sc-1)):i + 1]) / (3*sc)) / vol20[i]
                            if vol20[i] else 1.0,
            # The quiet base the operator's charts both show before the move.
            "base_range_24": (hi24 / lo24 - 1) * 100 if lo24 else 0.0,
            "base_range_48": (hi48 / lo48 - 1) * 100 if lo48 else 0.0,
            "bars_since_high_50": float(since_high),
            "atr_pct": atr[i],
            "ret_3": (c / closes[i - 3*sc] - 1) * 100 if i >= 3*sc else 0.0,
            "ret_6": (c / closes[i - 6*sc] - 1) * 100 if i >= 6*sc else 0.0,
            "ret_12": (c / closes[i - 12*sc] - 1) * 100 if i >= 12*sc else 0.0,
            "dist_high_50": (c / hi50 - 1) * 100 if hi50 else 0.0,
        })
    return out


# ------------------------------------------------------------------ build ---

def window_scale(args) -> int:
    """1 unless asked otherwise. On 15m, sc=4 gives the features the same
    physical horizon the 1h run had; sc=1 lets them react four times faster."""
    return max(1, int(args.window_scale or 1))


def start_bars(sym: str, run_pct: float, give_back: float, window: int,
               tf: str = "1h", grid: str = "1h") -> dict:
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
    bars = load_bars(sym, tf)
    idx = {b[0]: i for i, b in enumerate(bars)}
    out = {}
    for t in trends_tf(sym, run_pct, give_back, grid):
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


def build(symbols: list, args):
    """Feature matrix + metadata, as numpy arrays rather than a list of dicts.

    A dict per row costs ~1.7 KB. On 1h that is 1.6 GB and merely large; on 15m
    the same experiment is 3.97 M rows and 6.8 GB, against 2.3 GB of free RAM --
    it does not run at all. float32 columns bring the same data to ~350 MB, so
    the resolution question can actually be asked.
    """
    import numpy as np

    tf = args.tf
    sc = window_scale(args)
    warm = WARMUP * sc
    Xs, ys, syms_i, iis, tss, days, closes = [], [], [], [], [], [], []
    names = []
    for si, sym in enumerate(symbols):
        bars = load_bars(sym, tf)
        if len(bars) < warm + HORIZON * sc + 50:
            continue
        feats = feature_table(bars, sc)
        tail = HORIZON if args.horizon <= 0 else args.horizon
        grid = args.tf if args.trend_grid == "same" else args.trend_grid
        starts = (start_bars(sym, args.run, args.give_back, args.start_window,
                             args.tf, grid)
                  if args.label == "start" else None)
        keep_x, keep_y, keep_i = [], [], []
        for i in range(warm, len(bars) - tail):
            if starts is not None:
                y = starts.get(i, 0)
            else:
                y = will_run(bars, i, args.run, args.give_back, args.horizon)
            if y is None:
                continue
            f = feats[i]
            keep_x.append([f[k] for k in FEATS])
            keep_y.append(y)
            keep_i.append(i)
        if not keep_x:
            continue
        sidx = len(names)
        names.append(sym)
        # float64, not float32: the 6.8 GB came from ~1.7 KB of dict overhead
        # per row, not from the width of the numbers. Narrowing to float32 also
        # shifted the 1h AUC 0.6944 -> 0.6725, which would have put a storage
        # artefact inside a resolution comparison. float64 still costs 700 MB
        # for the whole 15m dataset.
        Xs.append(np.asarray(keep_x, dtype=np.float64))
        # int64, not int8: the memory win is in the 22 float32 feature columns,
        # and an int8 label makes `len(y) - sum(y)` overflow inside auc().
        ys.append(np.asarray(keep_y, dtype=np.int64))
        syms_i.append(np.full(len(keep_x), sidx, dtype=np.int32))
        iis.append(np.asarray(keep_i, dtype=np.int32))
        tss.append(np.asarray([bars[i][0].timestamp() for i in keep_i],
                              dtype=np.int64))
        days.append(np.asarray([int(bars[i][0].strftime("%Y%m%d")) for i in keep_i],
                               dtype=np.int32))
        closes.append(np.asarray([bars[i][4] for i in keep_i], dtype=np.float64))
        if len(names) % 25 == 0:
            print("  built %d/%d symbols, %d rows"
                  % (len(names), len(symbols), sum(len(a) for a in ys)))
    if not Xs:
        return None, None
    meta = {"y": np.concatenate(ys), "sym": np.concatenate(syms_i),
            "i": np.concatenate(iis), "ts": np.concatenate(tss),
            "day": np.concatenate(days), "close": np.concatenate(closes),
            "names": names}
    return np.concatenate(Xs), meta


def fit(Xtr, ytr, Xte, seed=0, shuffle=False):
    from catboost import CatBoostClassifier
    import numpy as np
    y = np.asarray(ytr)
    if shuffle:
        y = y.copy()
        np.random.RandomState(seed).shuffle(y)
    if len(np.unique(y)) < 2:
        return None, None
    m = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05,
                           verbose=0, random_seed=seed, allow_writing_files=False)
    m.fit(Xtr, y)
    return m.predict_proba(Xte)[:, 1], m


# ------------------------------------------------- which trends were caught ---

def catch_report(meta, probs, thr, args, top_n_examples=25):
    """Fire an alert wherever p >= thr, then ask which real trends it caught.

    A ranking metric cannot answer "which trends would we have caught", so this
    matches alerts against the ZigZag trends themselves and reports how far into
    each move the first alert landed -- an alert at 80% of the way up is a
    catch by any hit-rate metric and worthless to act on.
    """
    from datetime import datetime as _dt
    import numpy as np

    tf = args.tf
    names = meta["names"]
    hot = np.nonzero(probs >= thr)[0]
    alerts = defaultdict(list)
    for j in hot:
        alerts[names[meta["sym"][j]]].append(
            (_dt.fromtimestamp(int(meta["ts"][j]), tz=timezone.utc),
             float(meta["close"][j]), float(probs[j])))
    for k in alerts:
        alerts[k].sort()

    first_day = int(meta["day"].min())
    present = sorted(set(int(v) for v in np.unique(meta["sym"])))
    caught, missed, rows = 0, 0, []
    for sidx in present:
        sym = names[sidx]
        bars = load_bars(sym, tf)
        grid = args.tf if args.trend_grid == "same" else args.trend_grid
        for t in trends_tf(sym, args.run, args.give_back, grid):
            st = UP.attr(t, "start_ts", "start", "low_ts")
            en = UP.attr(t, "end_ts", "end", "high_ts")
            gain = UP.attr(t, "gain_pct", "gain")
            if st is None or en is None or gain is None:
                continue
            if int(st.strftime("%Y%m%d")) < first_day:
                continue
            hit = next(((ts, px, p) for ts, px, p in alerts.get(sym, [])
                        if st <= ts <= en), None)
            if hit is None:
                missed += 1
                continue
            caught += 1
            ts, px, p = hit
            peak = max((bb[2] for bb in bars if st <= bb[0] <= en), default=px)
            ahead = (peak / px - 1) * 100 if px else 0.0
            into = (1 - ahead / gain) * 100 if gain else 0.0
            rows.append({"sym": sym, "start": st, "end": en, "gain": gain,
                         "alert": ts, "ahead": ahead, "into": into, "p": p})

    n_alerts = int(len(hot))
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
    ap.add_argument("--trend-grid", choices=("1h", "15m", "same"), default="1h",
                    help="grid the TARGET trends are defined on, independent of "
                         "--tf. Default 1h holds the population fixed at the 342 "
                         "trends the committed results were scored against, so a "
                         "15m run changes only the detector's resolution.")
    ap.add_argument("--match-1h-universe", action="store_true",
                    help="restrict to symbols the 1h experiments could use. "
                         "The 15m backfill covers 101 symbols against 99 on 1h "
                         "(BAKE, MKR), and letting two extra symbols in would "
                         "put a population change inside a resolution comparison.")
    ap.add_argument("--tf", choices=("1h", "15m"), default="1h",
                    help="bar grid for features, labels AND trend detection")
    ap.add_argument("--window-scale", type=int, default=None,
                    help="multiply every feature lookback. Default 1 for 1h and "
                         "for 15m-native; pass 4 with --tf 15m to give the "
                         "features the same physical horizon as the 1h run.")
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

    import numpy as np

    symbols = UP.watchlist()
    if args.match_1h_universe:
        keep = [s for s in symbols if len(load_bars(s, "1h")) >= WARMUP + 50]
        print("universe restricted to the %d symbols the 1h runs used "
              "(dropped %d)" % (len(keep), len(symbols) - len(keep)))
        symbols = keep
    X, meta = build(symbols, args)
    if X is None or len(X) < 20000:
        print("only %d rows" % (0 if X is None else len(X)))
        return

    yall = meta["y"]
    base_all = float(yall.mean())
    days = sorted(set(int(v) for v in np.unique(meta["day"])))

    def sub(mask):
        """A view of the dataset restricted to `mask`, metadata kept aligned."""
        m = {k: (v if k == "names" else v[mask]) for k, v in meta.items()}
        return X[mask], m

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
            Xtr, mtr = sub(meta["day"] < c)
            Xte, mte = sub(meta["day"] >= c)
            if not len(Xtr) or not len(Xte):
                continue
            pp, _ = fit(Xtr, mtr["y"], Xte)
            if pp is None:
                continue
            yy = mte["y"]
            aa = CS.auc(yy, pp)
            nn = []
            for sd in range(3):
                qq, _ = fit(Xtr, mtr["y"], Xte, seed=200 + sd, shuffle=True)
                if qq is not None:
                    nn.append(CS.auc(yy, qq))
            nmean = sum(nn) / len(nn) if nn else float("nan")
            nsdev = ((sum((v - nmean) ** 2 for v in nn) / max(len(nn) - 1, 1)) ** 0.5
                     if len(nn) > 1 else float("nan"))
            oo = sorted(range(len(pp)), key=lambda i: -pp[i])
            kk = max(1, int(len(oo) * args.alert_rate))
            rr = catch_report(mte, pp, pp[oo[kk - 1]], args)
            tt = rr["caught"] + rr["missed"]
            print("%-12s%8d%9d%9.4f%9.4f%8.2f%10d%9.1f%%" % (
                c, len(Xtr), len(Xte), float(yy.mean()), aa,
                (aa - nmean) / nsdev if nsdev else float("nan"),
                tt, 100.0 * rr["caught"] / tt if tt else 0))
        return
    cut = days[int(len(days) * 0.7)]
    Xtr, mtr = sub(meta["day"] < cut)
    Xte, test = sub(meta["day"] >= cut)
    print()
    print("rows %d  (%d symbols, %d days)  base rate %.4f"
          % (len(X), len(meta["names"]), len(days), base_all))
    print("time cut %s: train %d / test %d" % (cut, len(Xtr), len(Xte)))

    probs, model = fit(Xtr, mtr["y"], Xte)
    if probs is None:
        print("degenerate labels")
        return
    yte = test["y"]
    base = float(yte.mean())
    a = CS.auc(yte, probs)

    nulls = []
    for s in range(5):
        ps, _ = fit(Xtr, mtr["y"], Xte, seed=100 + s, shuffle=True)
        if ps is not None:
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
        order = np.argsort(-probs)
        k = max(1, int(len(order) * frac))
        prec = float(yte[order[:k]].mean())
        print("%-10s%12.4f%10.4f%10.2fx" % (
            "%.1f%%" % (frac * 100), prec, base, prec / base if base else 0))

    if args.max_rsi is not None:
        # Suppress rather than re-rank: an alert on a bar at RSI 80 is a
        # confirmation whatever its score, and letting it keep the budget slot
        # would hide the cost this flag exists to measure.
        rsi_col = Xte[:, FEATS.index("rsi")]
        probs = np.where(rsi_col >= args.max_rsi, 0.0, probs)
        alive = int((probs > 0).sum())
        print()
        print("EARLINESS CONSTRAINT: alerts forbidden at RSI >= %.0f "
              "-- %d of %d test bars remain eligible (%.1f%%)"
              % (args.max_rsi, alive, len(probs), 100.0 * alive / len(probs)))
    order = np.argsort(-probs)
    k = max(1, int(len(order) * args.alert_rate))
    thr = float(probs[order[k - 1]])
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
