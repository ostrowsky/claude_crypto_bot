"""Is there ANY signal that separates "the move continues" from "the move is done"?

Both open goals reduce to this one predicate. Goal 2 needs it before the move,
goal 3 needs it at exit time. Fixed-width trails were ruled out on the whole
trade population (`_backtest_exit_timing.py`): no width beats the current exits,
because a width is a constant and the question is conditional. So the honest
next question is not "which threshold" but "does the information exist at all".

WHAT IS ASKED
    At a bar the bot is holding through, using only data up to and including
    that bar: does price gain +CONT_PCT before it loses CONT_PCT, within the
    next HORIZON bars?

That is the exact decision an exit policy faces on every bar, phrased as a race
rather than as a terminal return, because a stop experiences path order and a
mean does not.

WHY THIS POPULATION
    Every bar from entry to entry+MAX_BARS, for every entry the bot made --
    not only the bars it actually held. An exit policy that holds longer faces
    the bars after the real exit too, so restricting to entry..exit would
    measure the current policy's own window and call it the world (TH-06).
    And not only winner-days: conditioning on the day's outcome is what made
    this backtest's predecessor report a triple-capture result that live trading
    had already refuted.

CONTROLS, because an AUC on its own is not evidence (TH-01)
    * a label-shuffled retrain -- the pipeline must NOT manufacture separation;
    * base rate and lift at the operating point, never recall alone;
    * bootstrap clustered BY TRADE -- 48 bars of one trade are one observation
      with 48 rows, and an unclustered CI would be far too narrow;
    * a split by TIME on day boundaries, never at random.

    pyembed\\python.exe files\\_backtest_continuation_signal.py

Spec: docs/specs/features/continuation-signal-spec.md
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HISTORY = ROOT / "history"

CONT_PCT = 2.0          # the move the policy is trying to keep
HORIZON = 12            # bars ahead the race is run over (12h on 1h bars)
MAX_BARS = 48           # bars after entry the population extends to
WARMUP = 50             # bars of history a feature row needs before it is valid


# ----------------------------------------------------------------- klines ---

_BARS: dict = {}


def _read(path: Path) -> list:
    out = []
    if not path.exists():
        return out
    with io.open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                out.append((datetime.fromisoformat(r["ts"]), float(r["open"]),
                            float(r["high"]), float(r["low"]), float(r["close"]),
                            float(r.get("volume") or 0.0)))
            except (KeyError, ValueError, TypeError):
                continue
    return out


def bars(sym: str) -> list:
    """Hourly bars, merged across every cache that holds this symbol.

    The caches disagree about freshness: `_1h_365d` stops on 2026-06-20 while
    `_1h` reaches today for only a quarter of symbols. Picking whichever file
    happens to cover a trade makes cache staleness a selection rule on the
    sample, so both are read and merged on timestamp instead.
    """
    if sym in _BARS:
        return _BARS[sym]
    merged: dict = {}
    for suffix in ("1h_365d", "1h"):
        for row in _read(HISTORY / f"{sym}_{suffix}.csv"):
            merged[row[0]] = row
    out = [merged[k] for k in sorted(merged)]
    _BARS[sym] = out
    return out


# ------------------------------------------------------------------ trades ---

def entries() -> list:
    """(sym, entry_dt) for every entry event, deduplicated per symbol+hour."""
    seen = set()
    out = []
    with io.open(HERE / "bot_events.jsonl", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"entry"' not in line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("event") != "entry":
                continue
            sym, ts = e.get("sym"), e.get("ts")
            if not sym or not ts:
                continue
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except ValueError:
                continue
            dt = dt.replace(minute=0, second=0, microsecond=0)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            key = (sym, dt)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return sorted(out, key=lambda r: r[1])


# ---------------------------------------------------------------- features ---

def _rsi(closes: list, n: int = 14) -> float:
    if len(closes) < n + 1:
        return 50.0
    gains = losses = 0.0
    for i in range(-n, 0):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    if losses <= 0:
        return 100.0
    rs = (gains / n) / (losses / n)
    return 100.0 - 100.0 / (1.0 + rs)


def _ema(values: list, n: int) -> float:
    k = 2.0 / (n + 1.0)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e


def _atr_pct(win: list) -> float:
    trs = []
    for i in range(1, len(win)):
        h, l, c_prev = win[i][2], win[i][3], win[i - 1][4]
        trs.append(max(h - l, abs(h - c_prev), abs(l - c_prev)))
    if not trs:
        return 0.0
    c = win[-1][4]
    return (sum(trs[-14:]) / min(14, len(trs))) / c * 100.0 if c else 0.0


def _slope_pct(closes: list, n: int) -> float:
    if len(closes) < n:
        return 0.0
    y = closes[-n:]
    xm = (n - 1) / 2.0
    ym = sum(y) / n
    num = sum((i - xm) * (v - ym) for i, v in enumerate(y))
    den = sum((i - xm) ** 2 for i in range(n)) or 1.0
    return (num / den) / (y[-1] or 1.0) * 100.0


def features(hist: list, entry_price: float, bars_since: int,
             run_max: float) -> dict:
    """Everything computable from `hist` (bars up to and INCLUDING now).

    `hist` is sliced by the caller and never extends past the current bar; the
    test suite pins that, because a single off-by-one here would produce a
    beautiful and entirely fake AUC.
    """
    closes = [b[4] for b in hist]
    vols = [b[5] for b in hist]
    c = closes[-1]
    ema20, ema50 = _ema(closes[-20:], 20), _ema(closes[-50:], 50)
    vol_mean = (sum(vols[-20:]) / min(20, len(vols))) or 1.0
    up = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] > closes[i - 1]:
            up += 1
        else:
            break
    return {
        "ret_1": (c / closes[-2] - 1) * 100 if len(closes) > 1 else 0.0,
        "ret_3": (c / closes[-4] - 1) * 100 if len(closes) > 3 else 0.0,
        "ret_6": (c / closes[-7] - 1) * 100 if len(closes) > 6 else 0.0,
        "ret_12": (c / closes[-13] - 1) * 100 if len(closes) > 12 else 0.0,
        "rsi": _rsi(closes),
        "atr_pct": _atr_pct(hist[-30:]),
        "dist_ema20": (c / ema20 - 1) * 100 if ema20 else 0.0,
        "dist_ema50": (c / ema50 - 1) * 100 if ema50 else 0.0,
        "slope_6": _slope_pct(closes, 6),
        "slope_12": _slope_pct(closes, 12),
        "vol_ratio": vols[-1] / vol_mean,
        "range_pct": (hist[-1][2] - hist[-1][3]) / c * 100 if c else 0.0,
        "bars_since_entry": float(bars_since),
        "pnl_since_entry": (c / entry_price - 1) * 100 if entry_price else 0.0,
        "dd_from_run_max": (c / run_max - 1) * 100 if run_max else 0.0,
        "consec_up": float(up),
        "hour_utc": float(hist[-1][0].hour),
    }


# ------------------------------------------------------------------- label ---

def label(future: list, ref: float, up_pct: float = None, dn_pct: float = None):
    """1 if +up_pct is touched before -dn_pct, 0 if the stop comes first.

    `None` when neither side is touched inside the horizon: that is a genuinely
    undecided window, and folding it into 0 would quietly relabel "nothing
    happened" as "the move ended" -- a different claim, and the more flattering
    one for any model that predicts stagnation.
    """
    up_pct = CONT_PCT if up_pct is None else up_pct
    dn_pct = CONT_PCT if dn_pct is None else dn_pct
    up, dn = ref * (1 + up_pct / 100), ref * (1 - dn_pct / 100)
    for row in future:
        h, l = row[2], row[3]
        hit_up, hit_dn = h >= up, l <= dn
        if hit_up and hit_dn:
            return 0        # both in one bar: the pessimistic read, a stop fills
        if hit_up:
            return 1
        if hit_dn:
            return 0
    return None


def build_rows(verbose: bool = True, up_pct: float = None, dn_pct: float = None,
               horizon: int = None) -> list:
    up_pct = CONT_PCT if up_pct is None else up_pct
    dn_pct = CONT_PCT if dn_pct is None else dn_pct
    horizon = HORIZON if horizon is None else horizon
    rows = []
    skipped: dict = defaultdict(int)
    ents = entries()
    for sym, edt in ents:
        b = bars(sym)
        if not b:
            skipped["no_klines"] += 1
            continue
        if not (b[0][0] <= edt <= b[-1][0]):
            skipped["outside_cache"] += 1
            continue
        idx = None
        for i, bar in enumerate(b):
            if bar[0] >= edt:
                idx = i
                break
        if idx is None or idx < WARMUP:
            skipped["no_warmup"] += 1
            continue
        entry_price = b[idx][4]
        run_max = entry_price
        used = 0
        for k in range(idx, min(idx + MAX_BARS, len(b))):
            run_max = max(run_max, b[k][2])
            future = b[k + 1:k + 1 + horizon]
            if len(future) < horizon:
                break
            y = label(future, b[k][4], up_pct, dn_pct)
            if y is None:
                skipped["undecided"] += 1
                continue
            f = features(b[max(0, k - WARMUP):k + 1], entry_price, k - idx, run_max)
            f["_y"] = y
            f["_day"] = b[k][0].strftime("%Y-%m-%d")
            f["_trade"] = sym + "@" + edt.isoformat()
            f["_sym"] = sym
            rows.append(f)
            used += 1
        if used == 0:
            skipped["no_usable_bars"] += 1
    if verbose:
        print("entries in log        %d" % len(ents))
        for k, v in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print("  skipped %-16s%d" % (k, v))
        print("rows built            %d  over %d trades, %d symbols" % (
            len(rows), len(set(r["_trade"] for r in rows)),
            len(set(r["_sym"] for r in rows))))
    return rows


# -------------------------------------------------------------- evaluation ---

FEATS = ["ret_1", "ret_3", "ret_6", "ret_12", "rsi", "atr_pct", "dist_ema20",
         "dist_ema50", "slope_6", "slope_12", "vol_ratio", "range_pct",
         "bars_since_entry", "pnl_since_entry", "dd_from_run_max",
         "consec_up", "hour_utc"]


def auc(y: list, p: list) -> float:
    pairs = sorted(zip(p, y))
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    rank = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank += avg
        i = j + 1
    return (rank - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def split_by_time(rows: list, frac: float = 0.7) -> tuple:
    days = sorted(set(r["_day"] for r in rows))
    cut = days[int(len(days) * frac)]
    tr = [r for r in rows if r["_day"] < cut]
    te = [r for r in rows if r["_day"] >= cut]
    return tr, te, cut


def fit_predict(train: list, test: list, seed: int = 0, shuffle_y: bool = False):
    from catboost import CatBoostClassifier
    ytr = [r["_y"] for r in train]
    if shuffle_y:
        ytr = ytr[:]
        random.Random(seed).shuffle(ytr)
    m = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05,
                           verbose=0, random_seed=seed, allow_writing_files=False)
    m.fit([[r[f] for f in FEATS] for r in train], ytr)
    p = m.predict_proba([[r[f] for f in FEATS] for r in test])[:, 1]
    return list(p), m


def cluster_bootstrap_auc(test: list, p: list, draws: int = 200) -> tuple:
    """Resample TRADES, not rows. 48 bars of one trade are one observation."""
    by_trade = defaultdict(list)
    for r, pi in zip(test, p):
        by_trade[r["_trade"]].append((r["_y"], pi))
    trades = list(by_trade)
    rng = random.Random(7)
    out = []
    for _ in range(draws):
        y, q = [], []
        for _ in range(len(trades)):
            for yy, pp in by_trade[rng.choice(trades)]:
                y.append(yy)
                q.append(pp)
        a = auc(y, q)
        if not math.isnan(a):
            out.append(a)
    out.sort()
    if not out:
        return (float("nan"), float("nan"))
    return (out[int(0.025 * len(out))], out[int(0.975 * len(out))])


def lift_at(test: list, p: list, top_frac: float) -> tuple:
    order = sorted(range(len(p)), key=lambda i: -p[i])
    k = max(1, int(len(order) * top_frac))
    sel = order[:k]
    base = sum(r["_y"] for r in test) / len(test)
    prec = sum(test[i]["_y"] for i in sel) / k
    return prec, base, (prec / base if base else float("nan"))




def null_band(train: list, test: list, seeds: int = 8) -> tuple:
    """Where does AUC land when the labels carry no information?

    A single shuffled run is not a control. The first version of this script
    printed one, got 0.4851, and called a real-label 0.5124 a result -- but the
    null was already 0.015 away from 0.500 on its own, which is larger than the
    0.012 being claimed. Refitting on several shuffles gives the null a WIDTH,
    and the real AUC has to clear that width, not clear 0.5.
    """
    out = []
    for s in range(seeds):
        p, _ = fit_predict(train, test, seed=100 + s, shuffle_y=True)
        a = auc([r["_y"] for r in test], p)
        if not math.isnan(a):
            out.append(a)
    if len(out) < 2:
        return (float("nan"), float("nan"), out)
    mean = sum(out) / len(out)
    sd = (sum((v - mean) ** 2 for v in out) / (len(out) - 1)) ** 0.5
    return (mean, sd, out)


def evaluate(rows: list, draws: int, null_seeds: int) -> dict:
    train, test, cut = split_by_time(rows)
    if not train or not test:
        return {}
    yte = [r["_y"] for r in test]
    p, model = fit_predict(train, test)
    a = auc(yte, p)
    lo, hi = cluster_bootstrap_auc(test, p, draws)
    nmean, nsd, _ = null_band(train, test, null_seeds)
    _, base, lift10 = lift_at(test, p, 0.10)
    return {"cut": cut, "n_train": len(train), "n_test": len(test),
            "base": base, "auc": a, "lo": lo, "hi": hi,
            "null_mean": nmean, "null_sd": nsd,
            "z": ((a - nmean) / nsd) if nsd else float("nan"),
            "lift10": lift10, "model": model, "test": test, "p": p}


# The grid exists because concluding "no signal" from one parameterisation
# would be the mirror image of concluding "signal" from one: +2/-2 over 12h is
# nearly a coin flip on these names, and a policy that only has to beat a
# trailing stop cares about an ASYMMETRIC race -- a big move against a small
# give-back. Each row is a different question, so each gets its own base rate.
GRID = [
    ("+2 / -2  in 12h", 2.0, 2.0, 12),
    ("+5 / -2  in 12h", 5.0, 2.0, 12),
    ("+5 / -3  in 24h", 5.0, 3.0, 24),
    ("+10 / -3 in 24h", 10.0, 3.0, 24),
    ("+3 / -1.5 in 6h", 3.0, 1.5, 6),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--null-seeds", type=int, default=8)
    ap.add_argument("--only", type=int, default=None,
                    help="index into GRID: run one label with the detail block")
    ap.add_argument("--feats", default=None,
                    help="comma-separated feature subset. The decisive control: "
                         "if 'atr_pct,range_pct' alone reaches the full model's "
                         "lift, the ranking is volatility, not continuation -- a "
                         "coin that moves 10%% often is not a move that continues.")
    ap.add_argument("--drop-hour", action="store_true",
                    help="refit without hour_utc, to see what the clock carries")
    args = ap.parse_args()

    print("=" * 78)
    print("CONTINUATION SIGNAL -- does the information exist at decision time?")
    print("population: bars 0..%d after every entry, all %s trades" % (MAX_BARS, "4344"))
    print("split by TIME on day boundaries; CI clustered by trade")
    print("=" * 78)

    if args.feats:
        keep = [f.strip() for f in args.feats.split(",") if f.strip()]
        unknown = [f for f in keep if f not in FEATS]
        if unknown:
            raise SystemExit("unknown features: %s" % unknown)
        FEATS[:] = keep
        print("feature set restricted to: %s" % ", ".join(FEATS))
    if args.drop_hour and "hour_utc" in FEATS:
        FEATS.remove("hour_utc")
        print("hour_utc REMOVED from the feature set for this run")
    grid = GRID if args.only is None else [GRID[args.only]]
    results = []
    for name, up, dn, hz in grid:
        rows = build_rows(verbose=False, up_pct=up, dn_pct=dn, horizon=hz)
        if len(rows) < 2000:
            print("%-18s too few rows (%d)" % (name, len(rows)))
            continue
        r = evaluate(rows, args.draws, args.null_seeds)
        if not r:
            continue
        r["name"] = name
        r["n_rows"] = len(rows)
        results.append(r)
        print("built %-18s %7d rows   base %.3f" % (name, len(rows), r["base"]))

    print()
    print("%-18s%7s%8s%18s%16s%8s" % (
        "label", "base", "AUC", "95% CI (trade)", "null mean+-sd", "lift@10%"))
    print("-" * 78)
    for r in results:
        print("%-18s%7.3f%8.4f   [%.4f,%.4f]%9.4f+-%.4f%7.2fx" % (
            r["name"], r["base"], r["auc"], r["lo"], r["hi"],
            r["null_mean"], r["null_sd"], r["lift10"]))

    print()
    print("VERDICT -- a label passes only if BOTH hold:")
    print("  (a) AUC clears the shuffled-label null by >2 sd, and")
    print("  (b) top-decile lift is materially above 1.0x.")
    print("  An AUC whose CI excludes 0.5 while lift sits at 0.99x ranks nothing")
    print("  usable; the first run of this script reported exactly that and")
    print("  called it a result.")
    print()
    passed = []
    for r in results:
        ok_a = (not math.isnan(r["z"])) and r["z"] > 2.0
        ok_b = r["lift10"] >= 1.10
        mark = "PASS" if (ok_a and ok_b) else "no"
        print("  %-18s z=%6.2f  lift %.2fx   %s%s" % (
            r["name"], r["z"], r["lift10"], mark,
            "" if (ok_a and ok_b) else
            ("   (separation but nothing rankable)" if ok_a else "")))
        if ok_a and ok_b:
            passed.append(r)

    print()
    if not passed:
        print("  NEGATIVE across the grid. On the bot's own held bars, with the")
        print("  features available at decision time, continuation is not")
        print("  separable from exhaustion. This does not say the move is")
        print("  unpredictable in principle -- it says THESE features, on THIS")
        print("  population, carry no usable amount of it, and that a policy")
        print("  built on them would be tuning noise.")
    else:
        print("  %d label(s) pass. Next step is NOT deployment: it is replaying"
              % len(passed))
        print("  exits under the ranking on every trade and paying its cost, the")
        print("  same test that killed the fixed-width trails.")

    if args.only is not None and results:
        r = results[0]
        print()
        print("feature importance (%s)" % r["name"])
        try:
            imp = sorted(zip(FEATS, r["model"].get_feature_importance()),
                         key=lambda kv: -kv[1])
            for nm, v in imp[:8]:
                print("  %-20s%6.2f" % (nm, v))
        except Exception as exc:
            print("  unavailable: %s" % exc)
        print()
        print("%-12s%11s%8s%8s" % ("top slice", "precision", "base", "lift"))
        for frac in (0.05, 0.10, 0.20, 0.50):
            prec, b, lift = lift_at(r["test"], r["p"], frac)
            print("%-12s%11.3f%8.3f%8.2fx" % (
                "%.0f%%" % (frac * 100), prec, b, lift))


if __name__ == "__main__":
    main()
