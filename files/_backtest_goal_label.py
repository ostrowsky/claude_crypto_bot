"""Would the ML gate admit winners earlier if it were trained on the goal's move?

Trigger (2026-10-11): ICPUSDT climbed +13% slowly over ten hours while the gate
scored it 0.02-0.06. The live label `peak5_2.0` asks "+2% within the next five
bars" -- a volatility question -- and ICP's 13:00 candidate really did make only
+1.34% in the next 75 minutes. A slow day-long climb is invisible to that label.

Four alternative labels are trained on the SAME rows, features and estimator as
the live one, walk-forward by month, and graded on the GOAL: winner-days with an
admitted candidate before the +2.5% crossing, at matched admission, with a
per-trade non-inferiority check on the calibrated live trail.

Criteria are fixed in docs/specs/features/goal-aligned-ml-label-1012-spec.md.
"""
from __future__ import annotations

import collections
import io
import json
import math
import random
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

ARMS = ("C_peak5", "D24_5", "D24_10", "DVOL_2", "GOAL")
CONTROL = "C_peak5"
A_PRIMARY = 0.60
A_SECONDARY = 0.40
EMBARGO = timedelta(days=1)
CI_LEVEL = 0.9875                     # Bonferroni: 0.95 over 4 arms
GOAL_PP = 0.5
NONINF_PP = -0.10
BOOT = 2000
TEST_FROM = "2026-05"
PEAK5_BARS = {"15m": 5, "1h": 20}     # five bars of the row's own tf, in 15m bars
DAY_BARS = 96
STEP15 = timedelta(minutes=15)
TF_LEN = {"15m": timedelta(minutes=15), "1h": timedelta(hours=1)}
VOL_DAYS = 14
VOL_MIN_DAYS = 7


# ---------------------------------------------------------------- helpers

def decision_index(idx: dict, ts: datetime, tf: str):
    """15m bar whose close is the decision: the bar itself on 15m, the hour's last on 1h."""
    return idx.get(ts + (TF_LEN[tf] - STEP15))


def decision_time(ts: datetime, tf: str) -> datetime:
    return ts + TF_LEN[tf]


def peak_after(high, close, i: int, n: int):
    """Highest high over the n bars strictly after i, % of close[i]; None if truncated."""
    if i + n >= len(close) or close[i] <= 0:
        return None
    return (float(np.max(high[i + 1:i + 1 + n])) / close[i] - 1.0) * 100.0


def prior_range_median(day_range: dict, day: str):
    """Median daily range of the VOL_DAYS calendar days before `day` (never the day itself)."""
    d0 = datetime.strptime(day, "%Y-%m-%d")
    v = [day_range[k] for k in ((d0 - timedelta(days=j)).strftime("%Y-%m-%d") for j in range(1, VOL_DAYS + 1))
         if k in day_range]
    return statistics.median(v) if len(v) >= VOL_MIN_DAYS else None


def labels_for(*, peak5, peak24, vol_unit, early_winner) -> dict:
    return {
        "C_peak5": int(peak5 >= 2.0),
        "D24_5": int(peak24 >= 5.0),
        "D24_10": int(peak24 >= 10.0),
        "DVOL_2": int(vol_unit is not None and peak24 >= 2.0 * vol_unit),
        "GOAL": int(bool(early_winner)),
    }


def folds(times, months):
    """[(month, train_idx, test_idx)]: train strictly before month start minus the embargo."""
    t = np.array([x.timestamp() for x in times])
    ym = np.array([x.strftime("%Y-%m") for x in times])
    out = []
    for m in months:
        start = datetime.strptime(m + "-01", "%Y-%m-%d").replace(tzinfo=timezone.utc)
        out.append((m, np.where(t < (start - EMBARGO).timestamp())[0], np.where(ym == m)[0]))
    return out


def admit_top(scores, share):
    """Boolean mask of the top `share` by score (the same admission volume for every arm)."""
    s = np.asarray(scores, dtype=float)
    k = int(round(share * len(s)))
    m = np.zeros(len(s), dtype=bool)
    if k > 0:
        m[np.argsort(-s, kind="stable")[:k]] = True
    return m


def reachable(rows, W):
    """For each winner-day: 1 if an admitted (sym, decision_dt) falls in [open, crossing)."""
    by = collections.defaultdict(list)
    for sym, d in rows:
        by[sym].append(d)
    return [int(any(op <= d < dd for d in by.get(sym, ()))) for _day, sym, op, dd in W]


def verdict(goal_pt, goal_lo, trade_lo, month_share) -> str:
    if goal_pt <= 0 or trade_lo < NONINF_PP:
        return "refuted"
    if goal_pt >= GOAL_PP and goal_lo > 0 and month_share >= 0.5:
        return "supported"
    return "inconclusive"


def _q(v, p):
    return float(np.percentile(v, 100 * p)) if len(v) else float("nan")


# ---------------------------------------------------------------- data

def _dt(s):
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


class Coin:
    """15m long store of one coin + the 1h series built from it (for the 1h live trail)."""

    def __init__(self, sym, atr_period):
        import _backtest_trend_start_detector as TD
        import indicators as I
        b = TD.bars_15m(sym)
        self.t = [x[0] if x[0].tzinfo else x[0].replace(tzinfo=timezone.utc) for x in b]
        self.h = np.array([x[2] for x in b], dtype=float)
        self.l = np.array([x[3] for x in b], dtype=float)
        self.c = np.array([x[4] for x in b], dtype=float)
        self.idx = {d: i for i, d in enumerate(self.t)}
        self.atr = I._atr(self.h, self.l, self.c, atr_period) if len(b) > atr_period + 1 else np.full(len(b), np.nan)
        hours = collections.OrderedDict()
        for i, d in enumerate(self.t):
            hours.setdefault(d.replace(minute=0), []).append(i)
        self.h1_t = [k for k, v in hours.items() if len(v) == 4]
        self.h1_idx = {d: i for i, d in enumerate(self.h1_t)}
        hv = [hours[k] for k in self.h1_t]
        self.h1_h = np.array([self.h[v].max() for v in hv]) if hv else np.zeros(0)
        self.h1_l = np.array([self.l[v].min() for v in hv]) if hv else np.zeros(0)
        self.h1_c = np.array([self.c[v[-1]] for v in hv]) if hv else np.zeros(0)
        self.h1_atr = (I._atr(self.h1_h, self.h1_l, self.h1_c, atr_period)
                       if len(hv) > atr_period + 1 else np.full(len(hv), np.nan))
        days = collections.defaultdict(list)
        for i, d in enumerate(self.t):
            days[d.strftime("%Y-%m-%d")].append(i)
        self.day_range = {k: (self.h[v].max() / self.l[v].min() - 1) * 100
                          for k, v in days.items() if len(v) == DAY_BARS and self.l[v].min() > 0}


def load_rows(path, names, M, keep=None):
    """Deduplicated candidates as (meta, feature_vector); `keep(row)` filters raw rows."""
    seen, meta, X = set(), [], []
    with io.open(path, "rb") as fh:
        for raw in fh:
            try:
                r = json.loads(raw)
            except Exception:
                continue
            if keep is not None and not keep(r):
                continue
            tf = str(r.get("tf") or "")
            if tf not in TF_LEN or not r.get("sym") or not r.get("ts_signal"):
                continue
            key = (r["sym"], tf, r["ts_signal"])
            if key in seen:
                continue
            seen.add(key)
            try:
                ts = _dt(r["ts_signal"])
            except ValueError:
                continue
            d = r.get("decision") or {}
            fmap = M.build_feature_dict(r)
            X.append([M._safe_float(fmap.get(n)) for n in names])
            meta.append({"sym": r["sym"], "tf": tf, "ts": ts, "mode": r.get("signal_type"),
                         "dec": decision_time(ts, tf), "stage": d.get("stage"),
                         "ml_proba": d.get("ml_proba")})
    return meta, np.asarray(X, dtype=float)


# ---------------------------------------------------------------- main

def prepare():
    """Critic candidates with labels, live-trail trades and the winner-day tables.

    Shared with _backtest_ml_label_followups.py so both scripts grade the same rows.
    """
    import config as cfg
    import goal_validator as GV
    import immutable_labels as IL
    import label_store as LS
    import ml_signal_model as M
    import _compute_early_capture as E
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    names = M.safe_feature_names()
    meta, X = load_rows(HERE / "critic_dataset.jsonl", names, M)
    print("critic candidates, deduplicated per (sym, tf, bar): %d" % len(meta))

    # live admission share, context only
    rec = [m for m in meta if m["ml_proba"] is not None and m["ts"] >= datetime(2026, 9, 7, tzinfo=timezone.utc)]
    if rec:
        print("live gate since 2026-09-07 (context): %d rows carry ml_proba, %.1f%% passed the ml stage"
              % (len(rec), 100 * sum(1 for m in rec if m["stage"] != "ml_proba_zone") / len(rec)))

    wl = E.load_watchlist()
    win, _ = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
    dl = LS.intraday_deadlines()

    entries, _ = GV.load_bot_log("2026-05-01")
    ks = sorted(float(x[3]) for v in entries.values() for x in v if isinstance(x[3], (int, float)))
    K = ks[len(ks) // 2] if ks else 2.0
    print("trail_k = median of %d logged entries since 2026-05-01: %.2f" % (len(ks), K))

    coins = {}
    keep, lab, pnl = [], [], []
    miss = collections.Counter()
    for j, m in enumerate(meta):
        sym, tf = m["sym"], m["tf"]
        if sym not in coins:
            coins[sym] = Coin(sym, cfg.ATR_PERIOD)
        co = coins[sym]
        i = decision_index(co.idx, m["ts"], tf)
        if i is None:
            miss["no_bar"] += 1
            continue
        keep.append(j)
        day = m["dec"].strftime("%Y-%m-%d")
        k = (day, sym)
        early = k in win and k in dl and dl[k][1] is not None and dl[k][0] <= m["dec"] < dl[k][1]
        p5 = peak_after(co.h, co.c, i, PEAK5_BARS[tf])
        p24 = peak_after(co.h, co.c, i, DAY_BARS)
        if p5 is None or p24 is None:
            lab.append(None)
        else:
            lab.append(labels_for(peak5=p5, peak24=p24, vol_unit=prior_range_median(co.day_range, day),
                                  early_winner=early))
        m["peak24"] = p24
        m["early"] = early
        fl = GV.trail_floor(m["mode"], cfg)
        m["exit"] = None
        if tf == "15m":
            r = GV.live_trail(co.c, co.atr, i, K, fl, 96) if i + 96 < len(co.c) else None
            if r:
                m["exit"] = co.t[r[0]] + TF_LEN[tf]
        else:
            ih = co.h1_idx.get(m["ts"])
            r = (GV.live_trail(co.h1_c, co.h1_atr, ih, K, fl, 48)
                 if ih is not None and ih + 48 < len(co.h1_c) else None)
            if r:
                m["exit"] = co.h1_t[r[0]] + TF_LEN[tf]
        pnl.append(r[1] if r else float("nan"))
    meta = [meta[j] for j in keep]
    X = X[keep]
    pnl = np.asarray(pnl, dtype=float)
    print("resolvable decision bars: %d (dropped %s); with complete labels: %d; with a complete trade: %d"
          % (len(meta), dict(miss), sum(1 for x in lab if x), int(np.isfinite(pnl).sum())))
    print("period: %s .. %s" % (min(m["ts"] for m in meta).date(), max(m["ts"] for m in meta).date()))
    import types
    return types.SimpleNamespace(meta=meta, X=X, lab=lab, pnl=pnl, win=win, dl=dl, K=K, coins=coins,
                                 names=names, M=M, GV=GV, cfg=cfg)


def main():
    P = prepare()
    meta, X, lab, pnl, win, dl, M, GV = P.meta, P.X, P.lab, P.pnl, P.win, P.dl, P.M, P.GV

    labelled = [x for x in lab if x]
    print("\nlabel base rates (all rows with complete labels, n=%d):" % len(labelled))
    for a in ARMS:
        print("  %-8s %.1f%%" % (a, 100 * np.mean([x[a] for x in labelled])))

    times = [m["dec"] for m in meta]
    months = sorted({t.strftime("%Y-%m") for t in times if t.strftime("%Y-%m") >= TEST_FROM})
    scores = {a: np.full(len(meta), np.nan) for a in ARMS}
    fold_info = []
    for mo, tr, te in folds(times, months):
        tr = np.array([i for i in tr if lab[i] is not None])
        if len(tr) < 2000 or len(te) == 0:
            fold_info.append((mo, len(tr), len(te), "skipped"))
            continue
        sc = M.StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        for a in ARMS:
            y = np.array([lab[i][a] for i in tr], dtype=float)
            if len(np.unique(y)) < 2:
                continue
            mdl = M.CatBoostModel(Xtr.shape[1]).fit(Xtr, y)
            scores[a][te] = mdl.predict_proba(Xte)
        fold_info.append((mo, len(tr), len(te), "ok"))
    print("\nwalk-forward folds (month, train rows, test rows):")
    for f in fold_info:
        print("  %s  train %6d  test %6d  %s" % f)

    test = np.array([i for i in range(len(meta)) if all(np.isfinite(scores[a][i]) for a in ARMS)])
    tmonths = sorted({times[i].strftime("%Y-%m") for i in test})
    print("out-of-sample rows: %d, months %s" % (len(test), ", ".join(tmonths)))

    W = [w for w in GV.winner_days(TEST_FROM + "-01") if w[0][:7] in tmonths]
    print("winner-days in the test months (bot up all day, crossing known): %d" % len(W))
    early_rows = np.array([meta[i]["early"] for i in test])
    print("rows that are early winner rows: %.2f%%" % (100 * early_rows.mean()))

    rnd = random.Random(12)
    alpha = (1 - CI_LEVEL) / 2
    test_month = np.array([times[i].strftime("%Y-%m") for i in test])
    test_day = np.array([times[i].strftime("%Y-%m-%d") for i in test])
    udays = sorted(set(test_day))
    day_pos = {d: k for k, d in enumerate(udays)}

    def admitted(a, share):
        mask = np.zeros(len(test), dtype=bool)
        for mo in tmonths:
            sel = np.where(test_month == mo)[0]
            mask[sel] = admit_top(scores[a][test[sel]], share)
        return mask

    results = {}
    for share, tag in ((A_PRIMARY, "PRIMARY"), (A_SECONDARY, "secondary")):
        print("\n" + "=" * 100)
        print("%s operating point: every arm admits the top %.0f%% of each test month's candidates" % (tag, 100 * share))
        print("=" * 100)
        masks = {a: admitted(a, share) for a in ARMS}
        reach = {}
        for a in ARMS:
            rows = [(meta[test[k]]["sym"], times[test[k]]) for k in np.where(masks[a])[0]]
            reach[a] = np.array(reachable(rows, W), dtype=float)
        # per-day pnl aggregates for the cluster bootstrap
        agg = {}
        for a in ARMS:
            s = np.zeros(len(udays)); n = np.zeros(len(udays))
            for k in np.where(masks[a])[0]:
                v = pnl[test[k]]
                if math.isfinite(v):
                    s[day_pos[test_day[k]]] += v
                    n[day_pos[test_day[k]]] += 1
            agg[a] = (s, n)
        print("%-8s %9s %9s %18s %10s %10s %24s %8s %13s" % (
            "arm", "goal %", "delta pp", "98.75% CI", "trade %", "delta pp", "98.75% CI", "months+", "early-row prec"))
        nW = len(W)
        cr = reach[CONTROL]
        cs, cn = agg[CONTROL]
        for a in ARMS:
            g = 100 * reach[a].mean() if nW else float("nan")
            s, n = agg[a]
            tm = s.sum() / max(1, n.sum())
            prec = 100 * early_rows[masks[a]].mean()
            if a == CONTROL:
                print("%-8s %8.1f%% %9s %18s %9.3f%% %10s %24s %8s %12.2f%%" % (a, g, "-", "-", tm, "-", "-", "-", prec))
                continue
            d = reach[a] - cr
            gb = sorted(100 * float(np.mean(d[[rnd.randrange(nW) for _ in range(nW)]])) for _ in range(BOOT))
            tb = []
            for _ in range(BOOT):
                pick = [rnd.randrange(len(udays)) for _ in range(len(udays))]
                tb.append(s[pick].sum() / max(1, n[pick].sum()) - cs[pick].sum() / max(1, cn[pick].sum()))
            tb.sort()
            glo, ghi = gb[int(alpha * BOOT)], gb[int((1 - alpha) * BOOT) - 1]
            tlo, thi = tb[int(alpha * BOOT)], tb[int((1 - alpha) * BOOT) - 1]
            mshare = []
            for mo in tmonths:
                sel = [k for k, w in enumerate(W) if w[0][:7] == mo]
                if sel:
                    mshare.append(float(np.mean(d[sel])) >= 0)
            ms = float(np.mean(mshare)) if mshare else 0.0
            gpt = 100 * float(d.mean())
            tpt = tm - cs.sum() / max(1, cn.sum())
            v = verdict(gpt, glo, tlo, ms)
            if tag == "PRIMARY":
                results[a] = v
            print("%-8s %8.1f%% %+9.2f [%+6.2f, %+6.2f] %9.3f%% %+10.3f [%+8.3f, %+8.3f] %7.0f%% %12.2f%%   %s" % (
                a, g, gpt, glo, ghi, tm, tpt, tlo, thi, 100 * ms, prec, v if tag == "PRIMARY" else ""))
        print("base: %.2f%% of test rows are early winner rows; %d winner-days" % (100 * early_rows.mean(), nW))
        print("by month, reachable early days (arm count / control count / winner-days):")
        for mo in tmonths:
            sel = [k for k, w in enumerate(W) if w[0][:7] == mo]
            print("  %s  " % mo + "  ".join("%s %d" % (a, int(reach[a][sel].sum())) for a in ARMS) + "  / %d" % len(sel))

    # ------------------------------------------------------------ context
    print("\n" + "=" * 100)
    print("CONTEXT (never changes the verdict)")
    print("=" * 100)
    pk = np.array([meta[i]["peak24"] if meta[i]["peak24"] is not None else np.nan for i in test])
    print("top decile per month, mean forward 24h peak (no model: %.2f%%):" % np.nanmean(pk))
    for a in ARMS:
        top = admitted(a, 0.10)
        print("  %-8s %.2f%%   AUC vs early-winner row %.3f" % (
            a, np.nanmean(pk[top]), M.roc_auc_score_np(early_rows.astype(float), scores[a][test]) or float("nan")))
    print("\nICPUSDT 2026-10-11, first candidate each arm would admit at the primary point:")
    masks = {a: admitted(a, A_PRIMARY) for a in ARMS}
    icp = [k for k in range(len(test)) if meta[test[k]]["sym"] == "ICPUSDT"
           and times[test[k]].strftime("%Y-%m-%d") == "2026-10-11"]
    print("  ICP candidates that day: %d (%s)" % (len(icp), ", ".join(
        "%s %s" % (times[test[k]].strftime("%H:%M"), meta[test[k]]["tf"]) for k in icp[:12])))
    for a in ARMS:
        first = [k for k in icp if masks[a][k]]
        print("  %-8s %s" % (a, ("%s (%s), score %.3f" % (times[test[first[0]]].strftime("%H:%M"), meta[test[first[0]]]["tf"],
                                                         scores[a][test[first[0]]])) if first else "none admitted"))
    k = ("2026-10-11", "ICPUSDT")
    print("  ICP 10-11 winner-day: %s, crossing %s" % (k in win, dl[k][1].strftime("%H:%M") if k in dl and dl[k][1] else "n/a"))

    print("\nVERDICT at the primary point: " + ", ".join("%s %s" % (a, v) for a, v in results.items()))


if __name__ == "__main__":
    main()
